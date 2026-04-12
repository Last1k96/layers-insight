"""Graph extraction and layout service."""
from __future__ import annotations

import asyncio
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from backend.schemas.graph import GraphData, GraphEdge, GraphNode, NodeInput
from backend.utils.block_detection import detect_blocks
from backend.utils.block_layout import compute_block_layout as _compute_block_layout_sync
from backend.utils.dag_layout import compute_dag_layout
from backend.utils.op_categories import get_op_category, get_op_color

ELK_SCRIPT = Path(__file__).parent.parent / "utils" / "elk_layout.js"

# Node sizing constants (must match frontend svgRenderer.ts)
NODE_HEIGHT = 32
NODE_MIN_WIDTH = 100
NODE_PADDING = 20
CHAR_WIDTH = 7  # approximate average char width for 11px sans-serif
PORT_MIN_SPACING = 12  # minimum pixels between adjacent port positions


def _compute_node_size(op_type: str, in_degree: int = 0, out_degree: int = 0) -> tuple[float, float]:
    """Compute node width/height from op type label and edge degree.

    Widens nodes that have many incoming or outgoing edges so that
    port-spread edges have enough horizontal room.
    """
    text_width = len(op_type) * CHAR_WIDTH
    width = max(NODE_MIN_WIDTH, text_width + NODE_PADDING * 2)
    port_degree = max(in_degree, out_degree)
    if port_degree > 1:
        degree_width = PORT_MIN_SPACING * (port_degree + 1)
        width = max(width, degree_width)
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

    # Post-process: widen nodes based on distinct port count (not total edges).
    # A node with 12 edges all from port 0 has 1 output port — no widening.
    # A node with 3 distinct input ports needs space for port spreading.
    src_port_count: dict[str, int] = defaultdict(int)
    tgt_port_count: dict[str, int] = defaultdict(int)
    for e in edges:
        sp = e.source_port + 1
        if sp > src_port_count[e.source]:
            src_port_count[e.source] = sp
        tp = e.target_port + 1
        if tp > tgt_port_count[e.target]:
            tgt_port_count[e.target] = tp
    for n in nodes:
        new_w, _ = _compute_node_size(
            n.type, tgt_port_count.get(n.id, 0), src_port_count.get(n.id, 0),
        )
        if new_w > n.width:
            n.width = new_w

    return GraphData(nodes=nodes, edges=edges)


async def compute_layout(graph_data: GraphData) -> dict:
    """Compute layered DAG layout in pure Python.

    Returns dict with 'nodes' mapping node_id to {x, y} and
    'edges' mapping edge index to {waypoints: [{x,y}]}.
    """
    layout_nodes = [
        {
            "id": n.id,
            "width": n.width or 100,
            "height": n.height or NODE_HEIGHT,
            # Total input count from the original model (including
            # constant inputs that get filtered from the displayed
            # graph). Lets the layout place the visible edge at its
            # original port slot rather than centering when other
            # ports are constants.
            "total_inputs": len(n.inputs or []),
        }
        for n in graph_data.nodes
    ]
    layout_edges = [
        {"source": e.source, "target": e.target, "source_port": e.source_port, "target_port": e.target_port}
        for e in graph_data.edges
    ]
    return await asyncio.to_thread(compute_dag_layout, layout_nodes, layout_edges)


def should_use_block_layout(graph_data: GraphData) -> bool:
    """Detect whether a graph should use block-aware layout.

    Returns True when the graph has many nodes AND a block-structured
    naming pattern covering a significant fraction of the graph.
    """
    if len(graph_data.nodes) < 1500:
        return False
    nodes_dicts = [{"id": n.id, "name": n.name} for n in graph_data.nodes]
    edges_dicts = [{"source": e.source, "target": e.target} for e in graph_data.edges]
    bs = detect_blocks(nodes_dicts, edges_dicts, max_absorption_rounds=0)
    # Only count regex-matched nodes (no absorption) for the quick check
    if len(bs.blocks) < 3:
        return False
    coverage = sum(len(b.node_ids) for b in bs.blocks.values()) / len(graph_data.nodes)
    return coverage > 0.3


async def compute_block_aware_layout(graph_data: GraphData) -> dict:
    """Compute two-level block-aware layout for large transformer models.

    Returns dict with same format as compute_layout.
    """
    layout_nodes = [
        {
            "id": n.id,
            "name": n.name,
            "type": n.type,
            "width": n.width or 100,
            "height": n.height or NODE_HEIGHT,
            "total_inputs": len(n.inputs or []),
            "inputs": [inp.model_dump() for inp in n.inputs] if n.inputs else [],
        }
        for n in graph_data.nodes
    ]
    layout_edges = [
        {"source": e.source, "target": e.target,
         "source_port": e.source_port, "target_port": e.target_port}
        for e in graph_data.edges
    ]
    return await asyncio.to_thread(_compute_block_layout_sync, layout_nodes, layout_edges)


def _find_node_binary() -> str:
    """Locate the node executable, preferring the project-local install.

    start.sh installs Node.js into ``.node/`` at the project root and
    prepends it to PATH, but the server may be launched without sourcing
    that script (e.g. directly via the venv). Look for the local install
    first, then fall back to PATH.
    """
    project_root = Path(__file__).parent.parent.parent
    local = project_root / ".node" / "bin" / "node"
    if local.exists():
        return str(local)
    found = shutil.which("node")
    if found:
        return found
    raise RuntimeError(
        "node executable not found. Run start.sh once to install the project-local "
        "Node.js, or ensure 'node' is on PATH."
    )


async def compute_elk_layout(graph_data: GraphData) -> dict:
    """Compute layout via the elkjs reference engine in a Node.js subprocess.

    Slower than ``compute_layout`` for large graphs but useful as a
    reference for visual comparison. Requires the project-local Node.js
    install (see start.sh) and the ``elkjs`` package in node_modules.
    """
    elk_input = {
        "nodes": [
            {"id": n.id, "width": n.width or 100, "height": n.height or NODE_HEIGHT}
            for n in graph_data.nodes
        ],
        "edges": [
            {"source": e.source, "target": e.target}
            for e in graph_data.edges
        ],
    }

    node_bin = _find_node_binary()
    project_root = ELK_SCRIPT.parent.parent.parent  # cwd for node_modules resolution

    proc = await asyncio.create_subprocess_exec(
        node_bin, "--stack-size=65536", str(ELK_SCRIPT),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(project_root),
    )
    stdout, stderr = await proc.communicate(json.dumps(elk_input).encode())

    if proc.returncode != 0:
        raise RuntimeError(
            f"ELK layout failed (rc={proc.returncode}): {stderr.decode().strip()}"
        )
    return json.loads(stdout.decode())


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
