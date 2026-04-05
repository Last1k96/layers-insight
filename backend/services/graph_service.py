"""Graph extraction and layout service."""
from __future__ import annotations

import json
import re
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


def _normalize_element_type(et: Any) -> str:
    """Extract inner type name from OV's str(element_type) like \"<Type: 'float32'>\"."""
    s = str(et)
    m = re.search(r"'([^']+)'", s)
    return m.group(1) if m else s


def _find_root_constant(op: Any) -> str | None:
    """Walk backwards through a weight-prep chain to find the root Constant node name."""
    visited = set()
    current = op
    while current is not None:
        name = current.get_friendly_name()
        if name in visited:
            return None
        visited.add(name)
        if current.get_type_name() == "Constant":
            return name
        # Follow first input upstream
        if current.get_input_size() == 0:
            return None
        try:
            current = current.input(0).get_source_output().get_node()
        except Exception:
            return None
    return None


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
                    shape = [d.get_length() if d.is_static else "?" for d in pshape]
                element_type = _normalize_element_type(op.output(0).get_element_type())
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
                            src_shape = [d.get_length() if d.is_static else "?" for d in ps]
                        src_etype = _normalize_element_type(source_node.output(0).get_element_type())
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
                    # Filtered (constant/weight-prep) input — trace to root Constant
                    root_name = _find_root_constant(source_node)
                    node_inputs.append(NodeInput(
                        name=source_id, port=i,
                        shape=src_shape, element_type=src_etype,
                        is_const=True,
                        const_node_name=root_name,
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
            "node", "--stack-size=65536", str(ELK_SCRIPT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(json.dumps(elk_input).encode())

        if proc.returncode == 0:
            return json.loads(stdout.decode())
        else:
            raise RuntimeError(f"ELK layout failed (rc={proc.returncode}): {stderr.decode()}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"ELK layout error: {e}") from e


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
