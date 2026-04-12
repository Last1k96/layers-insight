"""Two-level Sugiyama layout for block-structured graphs.

Detects transformer blocks, lays out a coarse graph of super-nodes,
then lays out each block internally and positions its nodes within
the coarse bounding box. Bypass routing splits long-span edges to
left/right channels to minimize crossings.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from backend.utils.block_detection import (
    BlockStructure,
    detect_blocks,
    extract_block_subgraph,
    get_block_signature,
)
from backend.utils.dag_layout import compute_dag_layout

BLOCK_PADDING = 20
BLOCK_GAP = 40


@dataclass
class _InternalLayout:
    positions: dict[str, dict]
    edge_waypoints: dict[tuple[str, str], list[dict]]
    width: float = 0
    height: float = 0


def _compute_internal_layout(
    sub_nodes: list[dict],
    sub_edges: list[dict],
    node_map: dict[str, dict],
) -> _InternalLayout:
    if not sub_nodes:
        return _InternalLayout(positions={}, edge_waypoints={})

    layout_nodes = [
        {"id": n["id"], "width": n.get("width", 100), "height": n.get("height", 32),
         "total_inputs": len(n.get("inputs", []) or [])}
        for n in sub_nodes
    ]
    layout_edges = [
        {"source": e["source"], "target": e["target"],
         "source_port": e.get("source_port", 0), "target_port": e.get("target_port", 0)}
        for e in sub_edges
    ]

    raw = compute_dag_layout(layout_nodes, layout_edges)
    positions = raw["nodes"]
    if not positions:
        return _InternalLayout(positions={}, edge_waypoints={})

    min_x = min(p["x"] for p in positions.values())
    min_y = min(p["y"] for p in positions.values())
    for nid in positions:
        positions[nid]["x"] -= min_x
        positions[nid]["y"] -= min_y

    edge_wps: dict[tuple[str, str], list[dict]] = {}
    for i, e in enumerate(sub_edges):
        ek = f"e{i}"
        if ek in raw["edges"]:
            wps = raw["edges"][ek].get("waypoints", [])
            edge_wps[(e["source"], e["target"])] = [
                {"x": wp["x"] - min_x, "y": wp["y"] - min_y} for wp in wps
            ]

    max_x = max(positions[nid]["x"] + node_map[nid].get("width", 100) for nid in positions)
    max_y = max(positions[nid]["y"] + node_map[nid].get("height", 32) for nid in positions)

    return _InternalLayout(
        positions=positions, edge_waypoints=edge_wps,
        width=max_x + BLOCK_PADDING * 2, height=max_y + BLOCK_PADDING * 2,
    )


def _clone_internal_layout(
    rep_layout: _InternalLayout,
    rep_block, target_block,
    nodes, edges, block_of, node_map,
) -> _InternalLayout:
    rep_sub_nodes, rep_sub_edges = extract_block_subgraph(rep_block, nodes, edges, block_of)
    tgt_sub_nodes, tgt_sub_edges = extract_block_subgraph(target_block, nodes, edges, block_of)

    if len(rep_sub_nodes) != len(tgt_sub_nodes):
        return _compute_internal_layout(tgt_sub_nodes, tgt_sub_edges, node_map)

    def _sort_key(n, sub_edges):
        nid = n["id"]
        in_deg = sum(1 for e in sub_edges if e["target"] == nid)
        out_deg = sum(1 for e in sub_edges if e["source"] == nid)
        return (n.get("type", ""), in_deg, out_deg)

    rep_sorted = sorted(rep_sub_nodes, key=lambda n: _sort_key(n, rep_sub_edges))
    tgt_sorted = sorted(tgt_sub_nodes, key=lambda n: _sort_key(n, tgt_sub_edges))
    id_map = {rep_sorted[i]["id"]: tgt_sorted[i]["id"] for i in range(len(rep_sorted))}

    cloned_positions = {}
    for rep_id, pos in rep_layout.positions.items():
        tgt_id = id_map.get(rep_id)
        if tgt_id:
            cloned_positions[tgt_id] = {"x": pos["x"], "y": pos["y"]}

    cloned_edges: dict[tuple[str, str], list[dict]] = {}
    for (rep_src, rep_tgt), wps in rep_layout.edge_waypoints.items():
        tgt_src = id_map.get(rep_src)
        tgt_tgt = id_map.get(rep_tgt)
        if tgt_src and tgt_tgt:
            cloned_edges[(tgt_src, tgt_tgt)] = [{"x": wp["x"], "y": wp["y"]} for wp in wps]

    return _InternalLayout(
        positions=cloned_positions, edge_waypoints=cloned_edges,
        width=rep_layout.width, height=rep_layout.height,
    )


def compute_block_layout(
    nodes: list[dict],
    edges: list[dict],
) -> dict:
    """Compute a two-level layout for a block-structured graph.

    Returns the same format as compute_dag_layout:
    {nodes: {id: {x, y}}, edges: {eN: {waypoints: [{x, y}]}}}
    """
    t0 = time.perf_counter()

    bs = detect_blocks(nodes, edges)
    if len(bs.blocks) < 2:
        return compute_dag_layout(
            [{"id": n["id"], "width": n.get("width", 100), "height": n.get("height", 32)} for n in nodes],
            [{"source": e["source"], "target": e["target"]} for e in edges],
        )

    node_map = {n["id"]: n for n in nodes}

    # --- Phase 1: Per-block internal layouts (clone identical blocks) ---
    sig_to_blocks: dict[tuple, list[int]] = defaultdict(list)
    for bi, block in bs.blocks.items():
        sig = get_block_signature(block, nodes, edges, bs.block_of)
        sig_to_blocks[sig].append(bi)

    internal_layouts: dict[int, _InternalLayout] = {}
    for sig, block_indices in sig_to_blocks.items():
        rep_bi = block_indices[0]
        rep_block = bs.blocks[rep_bi]
        sub_nodes, sub_edges = extract_block_subgraph(rep_block, nodes, edges, bs.block_of)
        rep_layout = _compute_internal_layout(sub_nodes, sub_edges, node_map)
        internal_layouts[rep_bi] = rep_layout
        for bi in block_indices[1:]:
            internal_layouts[bi] = _clone_internal_layout(
                rep_layout, rep_block, bs.blocks[bi],
                nodes, edges, bs.block_of, node_map,
            )

    # --- Phase 2: Vertical block stack ---
    block_order = sorted(bs.blocks.keys())
    n_blocks = len(block_order)
    max_block_w = max(
        (internal_layouts[bi].width for bi in block_order if bi in internal_layouts),
        default=200,
    )

    infra_y = 0
    infra_x = max_block_w / 2
    coarse_layout: dict = {"nodes": {}, "edges": {}}
    for j, nid in enumerate(bs.infra_node_ids):
        n = node_map[nid]
        nw = n.get("width", 100)
        coarse_layout["nodes"][nid] = {
            "x": infra_x - nw / 2 + j * 120,
            "y": infra_y,
        }

    infra_h = 32 + BLOCK_GAP if bs.infra_node_ids else 0
    cur_y = infra_h
    for bi in block_order:
        il = internal_layouts.get(bi)
        bw = il.width if il else 200
        bh = il.height if il else 80
        bx = (max_block_w - bw) / 2
        coarse_layout["nodes"][f"__block_{bi}"] = {"x": bx, "y": cur_y}
        cur_y += bh + BLOCK_GAP

    total_grid_w = max_block_w
    mid_block = block_order[len(block_order) // 2] if block_order else 0

    # --- Phase 3: Global positioning ---
    final_positions: dict[str, dict] = {}
    final_edges: dict[str, dict] = {}

    for bi, block in bs.blocks.items():
        coarse_pos = coarse_layout["nodes"].get(f"__block_{bi}")
        if not coarse_pos:
            continue
        bx = coarse_pos["x"] + BLOCK_PADDING
        by = coarse_pos["y"] + BLOCK_PADDING
        il = internal_layouts.get(bi)
        if not il:
            continue
        for nid, pos in il.positions.items():
            final_positions[nid] = {"x": bx + pos["x"], "y": by + pos["y"]}

    for nid in bs.infra_node_ids:
        if nid in coarse_layout["nodes"]:
            final_positions[nid] = coarse_layout["nodes"][nid]

    # --- Phase 4: Edge routing ---
    for i, e in enumerate(edges):
        ek = f"e{i}"
        src_block = bs.block_of.get(e["source"])
        tgt_block = bs.block_of.get(e["target"])

        # Intra-block: offset internal waypoints
        if src_block is not None and tgt_block is not None and src_block == tgt_block:
            il = internal_layouts.get(src_block)
            coarse_pos = coarse_layout["nodes"].get(f"__block_{src_block}")
            if il and coarse_pos:
                bx = coarse_pos["x"] + BLOCK_PADDING
                by = coarse_pos["y"] + BLOCK_PADDING
                int_wps = il.edge_waypoints.get((e["source"], e["target"]))
                if int_wps:
                    final_edges[ek] = {
                        "waypoints": [{"x": wp["x"] + bx, "y": wp["y"] + by} for wp in int_wps]
                    }
                    continue

        # Inter-block or infra: route to avoid cutting through blocks
        src_pos = final_positions.get(e["source"])
        tgt_pos = final_positions.get(e["target"])
        if src_pos and tgt_pos:
            src_n = node_map[e["source"]]
            tgt_n = node_map[e["target"]]
            sw = src_n.get("width", 100)
            sh = src_n.get("height", 32)
            tw = tgt_n.get("width", 100)

            sx = src_pos["x"] + sw / 2
            sy = src_pos["y"] + sh
            ex = tgt_pos["x"] + tw / 2
            ey = tgt_pos["y"]

            src_b = bs.block_of.get(e["source"])
            tgt_b = bs.block_of.get(e["target"])

            is_adjacent = (
                src_b is not None and tgt_b is not None
                and abs(tgt_b - src_b) == 1
            )
            needs_bypass = False
            if src_b is not None and tgt_b is not None and abs(tgt_b - src_b) > 1:
                needs_bypass = True
            elif src_b is None and tgt_b is not None:
                for obi in block_order:
                    if obi == tgt_b:
                        continue
                    opos = coarse_layout["nodes"].get(f"__block_{obi}")
                    if opos:
                        oil = internal_layouts.get(obi)
                        oh = oil.height if oil else 80
                        if opos["y"] > sy and opos["y"] + oh < ey:
                            needs_bypass = True
                            break

            if needs_bypass:
                target_idx = tgt_b if tgt_b is not None else 0
                span = abs((tgt_b or 0) - (src_b or 0))
                if target_idx <= mid_block:
                    route_x = -BLOCK_PADDING - span * 6
                else:
                    route_x = total_grid_w + BLOCK_PADDING + span * 6
                wps = [
                    {"x": sx, "y": sy},
                    {"x": route_x, "y": sy},
                    {"x": route_x, "y": ey},
                    {"x": ex, "y": ey},
                ]
            elif is_adjacent:
                src_block_pos = coarse_layout["nodes"].get(f"__block_{src_b}")
                src_il = internal_layouts.get(src_b)
                if src_block_pos and src_il:
                    block_bottom_y = src_block_pos["y"] + src_il.height
                    mid_y = block_bottom_y + BLOCK_GAP / 2
                    if abs(sx - ex) < 8:
                        wps = [{"x": sx, "y": sy}, {"x": ex, "y": ey}]
                    else:
                        wps = [
                            {"x": sx, "y": sy},
                            {"x": sx, "y": mid_y},
                            {"x": ex, "y": mid_y},
                            {"x": ex, "y": ey},
                        ]
                else:
                    wps = [{"x": sx, "y": sy}, {"x": ex, "y": ey}]
            else:
                wps = [{"x": sx, "y": sy}]
                if abs(sx - ex) > 8:
                    mid_y = (sy + ey) / 2
                    wps.append({"x": sx, "y": mid_y})
                    wps.append({"x": ex, "y": mid_y})
                wps.append({"x": ex, "y": ey})
            final_edges[ek] = {"waypoints": wps}

    t1 = time.perf_counter()

    return {
        "nodes": final_positions,
        "edges": final_edges,
        "_meta": {
            "layout_time_s": round(t1 - t0, 3),
            "block_count": len(bs.blocks),
            "infra_count": len(bs.infra_node_ids),
            "unique_signatures": len(sig_to_blocks),
        },
    }
