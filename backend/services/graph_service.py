"""Graph extraction and layout service."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from backend.schemas.graph import GraphData, GraphEdge, GraphNode, NodeInput
from backend.utils.op_categories import get_op_category, get_op_color

# Path to ELK layout script
ELK_SCRIPT = Path(__file__).parent.parent / "utils" / "elk_layout.js"

# Node sizing constants (must match frontend svgRenderer.ts)
NODE_HEIGHT = 32
NODE_MIN_WIDTH = 100
NODE_PADDING = 20
CHAR_WIDTH = 7  # approximate average char width for 11px sans-serif


def _compute_node_size(op_type: str) -> tuple[float, float]:
    """Compute node width/height from op type label, matching frontend text measurement."""
    text_width = len(op_type) * CHAR_WIDTH
    width = max(NODE_MIN_WIDTH, text_width + NODE_PADDING * 2)
    return (width, NODE_HEIGHT)


def load_model(model_path: str, ov_core: Any) -> Any:
    """Load an OpenVINO model from XML path."""
    return ov_core.read_model(model_path)


def extract_graph(model: Any) -> GraphData:
    """Extract graph structure from an OpenVINO model."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen_nodes: set[str] = set()

    for op in model.get_ordered_ops():
        node_id = op.get_friendly_name()
        op_type = op.get_type_name()

        if op_type == "Constant":
            continue

        if node_id in seen_nodes:
            continue

        # Skip nodes whose ALL inputs come from filtered nodes (Constants
        # or other weight-prep ops).  Parameters have 0 inputs and are kept.
        input_count = op.get_input_size()
        if input_count > 0:
            has_visible_input = False
            for i in range(input_count):
                try:
                    src = op.input(i).get_source_output().get_node()
                    if src.get_friendly_name() in seen_nodes:
                        has_visible_input = True
                        break
                except Exception:
                    pass
            if not has_visible_input:
                continue

        seen_nodes.add(node_id)

        # Get output shape if available
        shape = None
        element_type = None
        try:
            if op.get_output_size() > 0:
                pshape = op.output(0).get_partial_shape()
                if pshape.is_static:
                    shape = list(pshape.get_shape())
                else:
                    shape = [str(d) for d in pshape]
                element_type = str(op.output(0).get_element_type())
        except Exception:
            pass

        # Get op attributes
        attributes = {}
        try:
            for attr_name in op.get_attributes():
                val = op.get_attributes()[attr_name]
                attributes[attr_name] = str(val) if not isinstance(val, (int, float, bool, str)) else val
        except Exception:
            pass

        category = get_op_category(op_type)
        color = get_op_color(op_type)
        w, h = _compute_node_size(op_type)

        # Collect input info and build edges
        node_inputs: list[NodeInput] = []
        for i in range(op.get_input_size()):
            try:
                source_output = op.input(i).get_source_output()
                source_node = source_output.get_node()
                source_id = source_node.get_friendly_name()
                source_port = source_output.get_index()

                # Get source shape/type info
                src_shape = None
                src_etype = None
                try:
                    if source_node.get_output_size() > 0:
                        ps = source_node.output(0).get_partial_shape()
                        if ps.is_static:
                            src_shape = list(ps.get_shape())
                        else:
                            src_shape = [str(d) for d in ps]
                        src_etype = str(source_node.output(0).get_element_type())
                except Exception:
                    pass

                if source_id in seen_nodes:
                    # Visible edge
                    edges.append(GraphEdge(
                        source=source_id,
                        target=node_id,
                        source_port=source_port,
                        target_port=i,
                    ))
                    node_inputs.append(NodeInput(
                        name=source_id, port=i,
                        shape=src_shape, element_type=src_etype,
                        is_const=False,
                    ))
                else:
                    # Filtered (constant/weight-prep) input
                    node_inputs.append(NodeInput(
                        name=source_id, port=i,
                        shape=src_shape, element_type=src_etype,
                        is_const=True,
                    ))
            except Exception:
                pass

        nodes.append(GraphNode(
            id=node_id,
            name=node_id,
            type=op_type,
            shape=shape,
            element_type=element_type,
            category=category,
            color=color,
            attributes=attributes,
            inputs=node_inputs,
            width=w,
            height=h,
        ))

    return GraphData(nodes=nodes, edges=edges)


async def compute_layout(graph_data: GraphData) -> dict:
    """Compute layout using ELK via Node.js subprocess.

    Returns dict with 'nodes' mapping node_id to {x, y} and
    'edges' mapping edge index to {waypoints: [{x,y}]}.
    Falls back to topological layer assignment if ELK fails.
    """
    import asyncio

    elk_input = {
        "nodes": [{"id": n.id, "width": n.width or 100, "height": n.height or NODE_HEIGHT} for n in graph_data.nodes],
        "edges": [{"source": e.source, "target": e.target} for e in graph_data.edges],
    }

    try:
        proc = await asyncio.create_subprocess_exec(
            "node", str(ELK_SCRIPT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(json.dumps(elk_input).encode())

        if proc.returncode == 0:
            return json.loads(stdout.decode())
        else:
            print(f"ELK layout failed: {stderr.decode()}", file=sys.stderr)
    except Exception as e:
        print(f"ELK layout error: {e}", file=sys.stderr)

    # Fallback: topological layer assignment
    return _fallback_layout(graph_data)


def _fallback_layout(graph_data: GraphData) -> dict[str, dict[str, float]]:
    """Simple layered layout fallback using topological ordering."""
    # Build adjacency
    children: dict[str, list[str]] = {}
    parents: dict[str, list[str]] = {}
    for e in graph_data.edges:
        children.setdefault(e.source, []).append(e.target)
        parents.setdefault(e.target, []).append(e.source)

    # Compute layers via BFS from sources
    node_ids = {n.id for n in graph_data.nodes}
    sources = [n.id for n in graph_data.nodes if n.id not in parents or not parents.get(n.id)]
    if not sources:
        sources = [graph_data.nodes[0].id] if graph_data.nodes else []

    layer: dict[str, int] = {}
    queue = list(sources)
    for s in queue:
        layer[s] = 0

    visited = set(queue)
    while queue:
        current = queue.pop(0)
        for child in children.get(current, []):
            new_layer = layer[current] + 1
            if child not in layer or new_layer > layer[child]:
                layer[child] = new_layer
            if child not in visited:
                visited.add(child)
                queue.append(child)

    # Assign remaining nodes
    for n in graph_data.nodes:
        if n.id not in layer:
            layer[n.id] = 0

    # Position nodes
    layer_nodes: dict[int, list[str]] = {}
    for nid, l in layer.items():
        layer_nodes.setdefault(l, []).append(nid)

    positions = {}
    y_spacing = 100
    x_spacing = 220
    for l, nids in sorted(layer_nodes.items()):
        for i, nid in enumerate(nids):
            positions[nid] = {"x": i * x_spacing, "y": l * y_spacing}

    return {"nodes": positions, "edges": {}}


def apply_layout(graph_data: GraphData, layout_result: dict) -> GraphData:
    """Apply positions and edge waypoints from layout result."""
    positions = layout_result.get("nodes", layout_result)
    edge_waypoints = layout_result.get("edges", {})

    for node in graph_data.nodes:
        if node.id in positions:
            node.x = positions[node.id]["x"]
            node.y = positions[node.id]["y"]

    # Apply waypoints to edges
    for i, edge in enumerate(graph_data.edges):
        edge_key = f"e{i}"
        if edge_key in edge_waypoints:
            edge.waypoints = edge_waypoints[edge_key].get("waypoints")

    return graph_data


def search_nodes(graph_data: GraphData, query: str) -> list[GraphNode]:
    """Search nodes by name or type (case-insensitive substring match)."""
    q = query.lower()
    return [n for n in graph_data.nodes if q in n.name.lower() or q in n.type.lower()]
