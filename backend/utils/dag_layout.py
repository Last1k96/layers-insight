"""Pure-Python layered DAG layout (Sugiyama + Brandes-Köpf).

Produces Netron-like appearance: straight edges for sequential chains,
compact layout with minimized crossings. Uses dummy nodes for long-edge
routing and 4-variant Brandes-Köpf for near-optimal coordinate assignment.
"""
from __future__ import annotations

from collections import defaultdict, deque

from backend.utils.dag_regions import (
    Region,
    build_node_region_map,
    find_sese_regions,
)

# Spacing constants (match Netron/ELK dagre config)
NODE_SPACING = 20   # horizontal gap between nodes in same layer
LAYER_SPACING = 20  # vertical gap between layers
DUMMY_SPACING = 2   # horizontal gap between dummy nodes (tight bundling)
ARROW_LENGTH = 8    # must match frontend edgesPipeline.ts
MAX_WAYPOINTS = 200


def _simplify_waypoints(wps: list[dict], max_points: int = MAX_WAYPOINTS) -> list[dict]:
    """Reduce waypoint count for long edges via Ramer-Douglas-Peucker.

    Edges spanning hundreds of layers produce a waypoint per dummy node. The
    frontend tessellates each span into B-spline segments, so 2000+ waypoints
    create millions of GPU vertices.
    """
    if len(wps) <= max_points:
        return wps

    # Iterative RDP with adaptive tolerance (increase until under budget)
    tolerance = 1.0
    result = wps
    for _ in range(20):
        result = _rdp(wps, tolerance)
        if len(result) <= max_points:
            return result
        tolerance *= 2.0

    # Fallback: uniform subsample
    step = (len(wps) - 1) / (max_points - 1)
    return [wps[min(int(j * step), len(wps) - 1)] for j in range(max_points)]


def _rdp(points: list[dict], epsilon: float) -> list[dict]:
    """Iterative Ramer-Douglas-Peucker polyline simplification."""
    n = len(points)
    if n <= 2:
        return points
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        ax, ay = points[lo]["x"], points[lo]["y"]
        bx, by = points[hi]["x"], points[hi]["y"]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        max_dist = 0.0
        max_idx = lo
        for i in range(lo + 1, hi):
            px, py = points[i]["x"] - ax, points[i]["y"] - ay
            if seg_len_sq > 0:
                cross = abs(px * dy - py * dx)
                d = cross * cross / seg_len_sq
            else:
                d = px * px + py * py
            if d > max_dist:
                max_dist = d
                max_idx = i
        if max_dist > epsilon * epsilon:
            keep[max_idx] = True
            stack.append((lo, max_idx))
            stack.append((max_idx, hi))
    return [p for p, k in zip(points, keep) if k]


def compute_dag_layout(
    nodes: list[dict],
    edges: list[dict],
) -> dict:
    """Compute layered DAG layout in pure Python.

    Args:
        nodes: [{id, width, height}, ...]
        edges: [{source, target, source_port?}, ...]

    Returns:
        {nodes: {id: {x, y}}, edges: {"e{i}": {waypoints: [{x, y}]}}}
    """
    if not nodes:
        return {"nodes": {}, "edges": {}}

    node_map = {n["id"]: n for n in nodes}
    node_ids = [n["id"] for n in nodes]

    # Build adjacency on original graph
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    for e in edges:
        adj[e["source"]].append(e["target"])
        in_degree.setdefault(e["target"], 0)
        in_degree[e["target"]] += 1

    # Phase 1: Layer assignment (longest path from sources)
    layer_of = _assign_layers(node_ids, adj, in_degree)
    num_layers = max(layer_of.values(), default=0) + 1

    # Initial ordering via topological sort
    topo = _topo_sort(node_ids, adj, in_degree)
    layers: list[list[str]] = [[] for _ in range(num_layers)]
    for nid in topo:
        layers[layer_of[nid]].append(nid)

    # Phase 2: Insert dummy nodes for long edges (spans > 1 layer)
    ext_adj: dict[str, list[str]] = defaultdict(list)
    ext_rev: dict[str, list[str]] = defaultdict(list)
    ext_nmap = dict(node_map)  # extended: includes dummies
    edge_chains: list[list[str]] = []  # per-edge chain of node IDs
    _dummy_counter = 0

    for e in edges:
        src, tgt = e["source"], e["target"]
        sL, tL = layer_of[src], layer_of[tgt]
        chain = [src]
        prev = src
        for L in range(sL + 1, tL):
            did = f"__d{_dummy_counter}"
            _dummy_counter += 1
            ext_nmap[did] = {"id": did, "width": 0, "height": 0}
            layer_of[did] = L
            layers[L].append(did)
            ext_adj[prev].append(did)
            ext_rev[did].append(prev)
            chain.append(did)
            prev = did
        ext_adj[prev].append(tgt)
        ext_rev[tgt].append(prev)
        chain.append(tgt)
        edge_chains.append(chain)

    # Phase 3: Crossing minimization (barycenter, 4 sweeps, with dummies)
    pos: dict[str, int] = {}
    for L in range(num_layers):
        for i, nid in enumerate(layers[L]):
            pos[nid] = i

    for _ in range(4):
        for L in range(1, num_layers):
            _sort_by_barycenter(layers[L], ext_rev, pos)
            for i, nid in enumerate(layers[L]):
                pos[nid] = i
        for L in range(num_layers - 2, -1, -1):
            _sort_by_barycenter(layers[L], ext_adj, pos)
            for i, nid in enumerate(layers[L]):
                pos[nid] = i

    # Phase 4: Brandes-Köpf coordinate assignment (4 variants, median)
    x_of = _brandes_kopf(layers, num_layers, ext_nmap, ext_adj, ext_rev, pos)

    # Phase 4a': Pull single-input "auxiliary" nodes back toward their
    # predecessor's column when the median-of-4 placement drifted
    # because of a long-edge skip child. The Slice_1368-style case:
    # one short input from Shape_1364, one long output to Concat_1370.
    # Brandes-Köpf balances between the two, leaving the node 1000+
    # px from its only visible input. The snap only fires when the
    # target column is empty in the node's layer.
    _snap_aux_nodes_to_input_column(
        x_of, node_ids, edges, layer_of, layers, ext_nmap,
    )

    # Phase 4a: Detect Single-Entry Single-Exit regions so the
    # straightening pass can restrict its obstacle field per edge to
    # nodes that are structurally relevant. A bypass that lives inside
    # a residual block won't see the rest of the network as obstacles,
    # which keeps it from being pushed into a far corridor.
    region_root = find_sese_regions(node_ids, edges)
    node_region = build_node_region_map(region_root)

    # Phase 4b: Straighten long edges and route alongside subgraph boundaries
    _straighten_long_edges(
        x_of, edge_chains, layer_of, ext_nmap, layers, num_layers, edges,
        node_region=node_region,
    )

    # Phase 5: Y-coordinates
    # The gap below each layer grows with the maximum single-node fan-
    # out (so a node with many outputs has room for its edges to spread
    # cleanly) and the local fan-in of the next layer (so converging
    # edges don't crowd a single target).
    #
    # Source-side fan-out counts EVERY outgoing edge — even long ones
    # need to depart from the source's bottom and share the post-layer
    # space, so a node with 4 outputs (3 short + 1 long) deserves the
    # same gap as one with 4 short outputs.
    #
    # Destination-side fan-in stays local-only; long edges arrive
    # through corridors and don't crowd the destination layer.
    node_total_out: dict[str, int] = defaultdict(int)
    node_local_in: dict[str, int] = defaultdict(int)
    for chain in edge_chains:
        src_L, tgt_L = layer_of[chain[0]], layer_of[chain[-1]]
        node_total_out[chain[0]] += 1
        if tgt_L - src_L <= 2:
            node_local_in[chain[-1]] += 1

    layer_max_out: list[int] = [0] * num_layers
    layer_max_in: list[int] = [0] * num_layers
    for nid in node_ids:
        L = layer_of[nid]
        if node_total_out.get(nid, 0) > layer_max_out[L]:
            layer_max_out[L] = node_total_out[nid]
        if node_local_in.get(nid, 0) > layer_max_in[L]:
            layer_max_in[L] = node_local_in[nid]

    # Long skip edges (span > 2) arriving at each layer
    layer_long_in: list[int] = [0] * num_layers
    for chain in edge_chains:
        src_L, tgt_L = layer_of[chain[0]], layer_of[chain[-1]]
        if tgt_L - src_L > 2:
            layer_long_in[tgt_L] += 1

    DEGREE_THRESHOLD = 2
    EXTRA_PER_DEGREE = 32   # px per local connection above threshold
    EXTRA_PER_LONG = 10     # px per long skip edge arriving
    typical_h = max((node_map[nid]["height"] for nid in node_ids), default=40)

    layer_y: list[float] = [0.0] * num_layers
    layer_mid: list[float] = [0.0] * num_layers
    y = 0.0
    for L in range(num_layers):
        layer_y[L] = y
        max_h = max((ext_nmap[nid]["height"] for nid in layers[L]), default=0)
        layer_mid[L] = y + max_h / 2
        # Degree-based: max single-node fan-out/fan-in at gap boundary
        fan = layer_max_out[L]
        if L + 1 < num_layers:
            fan = max(fan, layer_max_in[L + 1])
        deg_extra = max(0, fan - DEGREE_THRESHOLD) * EXTRA_PER_DEGREE
        # Long-edge bonus: skip edges arriving at L+1 (convergence)
        long_extra = (layer_long_in[L + 1] if L + 1 < num_layers else 0) * EXTRA_PER_LONG
        # Cap at 4× the typical node height so extreme fan-outs (e.g.
        # Shape → 12 reshape ops) get plenty of room without going
        # completely unbounded.
        extra = min(4 * typical_h, deg_extra + long_extra)
        y += max_h + LAYER_SPACING + extra

    # Phase 6: Build result (only real nodes in positions)
    positions = {nid: {"x": x_of[nid], "y": layer_y[layer_of[nid]]} for nid in node_ids}

    # Edge waypoints (using dummy positions for intermediate points)
    # Source port counts (max source_port + 1 per node)
    port_counts: dict[str, int] = defaultdict(int)
    for e in edges:
        p = e.get("source_port", 0) + 1
        if p > port_counts[e["source"]]:
            port_counts[e["source"]] = p

    # Target port counts (max target_port + 1 per node).
    # Prefer the original input count from the model when available
    # (passed as `total_inputs` on each node) so edges arrive at their
    # correct port slot even when sibling inputs are constants that
    # were filtered from the visible graph. Falls back to scanning
    # visible edges otherwise.
    target_port_counts: dict[str, int] = defaultdict(int)
    for nid, n in node_map.items():
        if n.get("total_inputs"):
            target_port_counts[nid] = n["total_inputs"]
    for e in edges:
        p = e.get("target_port", 0) + 1
        if p > target_port_counts[e["target"]]:
            target_port_counts[e["target"]] = p

    # Fan-out groups: edges sharing same (source, source_port),
    # sorted by actual route direction so exit order matches (no crossing).
    # For long edges, use the first dummy (corridor) position.
    # For short edges, use the target center.
    fan_groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, e in enumerate(edges):
        fan_groups[(e["source"], e.get("source_port", 0))].append(i)
    for group in fan_groups.values():
        if len(group) > 1:
            def _route_x(ei, _chains=edge_chains, _xof=x_of, _edges=edges, _nmap=ext_nmap):
                chain = _chains[ei]
                if len(chain) > 2:
                    return _xof[chain[1]]  # first dummy = corridor position
                tid = _edges[ei]["target"]
                return _xof.get(tid, 0) + _nmap.get(tid, {"width": 0})["width"] / 2
            group.sort(key=_route_x)

    ewp: dict[str, dict] = {}
    for i, (e, chain) in enumerate(zip(edges, edge_chains)):
        sid, tid = e["source"], e["target"]
        sn, tn = ext_nmap[sid], ext_nmap[tid]

        # Start point (bottom of source, port-spread + fan-out)
        nports = port_counts.get(sid, 1)
        sport = e.get("source_port", 0)
        group_key = (sid, sport)
        fan_list = fan_groups[group_key]
        fan_count = len(fan_list)
        fan_idx = fan_list.index(i)

        if nports > 1:
            # Multiple distinct output ports: divide node width among ports
            ps = sn["width"] / (nports + 1)
            port_center = x_of[sid] + ps * (sport + 1)
            if fan_count > 1:
                # Sub-spread within this port's slice
                half_slice = ps * 0.4
                fan_step = (2 * half_slice) / (fan_count + 1)
                sx = port_center - half_slice + fan_step * (fan_idx + 1)
            else:
                sx = port_center
        elif fan_count > 1:
            # Single port but multiple consumers: spread across node width
            fs = sn["width"] / (fan_count + 1)
            sx = x_of[sid] + fs * (fan_idx + 1)
        else:
            sx = x_of[sid] + sn["width"] / 2
        sy = layer_y[layer_of[sid]] + sn["height"]

        # End point (top of target, target-port-spread)
        ntports = target_port_counts.get(tid, 1)
        if ntports > 1:
            tpt = e.get("target_port", 0)
            tps = tn["width"] / (ntports + 1)
            ex = x_of[tid] + tps * (tpt + 1)
        else:
            ex = x_of[tid] + tn["width"] / 2
        ey = layer_y[layer_of[tid]]

        wps: list[dict[str, float]] = [{"x": sx, "y": sy}]
        # Intermediate waypoints from dummy node positions
        dummies = chain[1:-1]
        for did in dummies:
            dL = layer_of[did]
            wps.append({"x": x_of[did], "y": layer_mid[dL]})
        if dummies:
            # Corridor/nudge: vertical guides when last dummy diverges
            if abs(x_of[dummies[-1]] - ex) > ARROW_LENGTH:
                prev_y = layer_mid[layer_of[dummies[-1]]]
                gap = ey - prev_y
                step = ARROW_LENGTH
                n_guides = max(1, min(3, int((gap * 0.4) / step)))
                for g in range(n_guides, 0, -1):
                    wps.append({"x": ex, "y": ey - step * g})
        elif abs(sx - ex) > ARROW_LENGTH:
            # Span-1 edge with offset: convert to 4-point step routing
            # (vertical exit, smooth transition, vertical entry) so the
            # cubic B-spline settles at target x before the arrow.
            #
            # Skip the step pattern when at least one sibling in this
            # fan-out group is a long edge (has dummies). Mixing step-
            # patterned span-1 edges with diagonal long edges in the
            # same group causes the rendered curves to cross because
            # the step's horizontal jump happens at a y where the
            # diagonal sibling is still moving.
            sibling_has_dummies = False
            if fan_count > 1:
                for j in fan_list:
                    if j == i:
                        continue
                    if len(edge_chains[j]) > 2:
                        sibling_has_dummies = True
                        break
            if not sibling_has_dummies:
                gap = ey - sy
                margin = gap * 0.35
                wps.append({"x": sx, "y": sy + margin})
                wps.append({"x": ex, "y": ey - margin})
                # Remove the bare start — rebuild with step pattern
                wps[0] = {"x": sx, "y": sy}
        wps.append({"x": ex, "y": ey})

        ewp[f"e{i}"] = {"waypoints": _simplify_waypoints(wps)}

    return {"nodes": positions, "edges": ewp}


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------

def _assign_layers(node_ids, adj, in_degree):
    """Assign layers via longest-path from sources (Kahn's BFS)."""
    layer_of: dict[str, int] = {}
    remaining = dict(in_degree)
    queue = deque()
    for nid in node_ids:
        if remaining.get(nid, 0) == 0:
            queue.append(nid)
            layer_of[nid] = 0
    while queue:
        nid = queue.popleft()
        for child in adj.get(nid, []):
            layer_of[child] = max(layer_of.get(child, 0), layer_of[nid] + 1)
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)
    for nid in node_ids:
        layer_of.setdefault(nid, 0)
    return layer_of


def _topo_sort(node_ids, adj, in_degree):
    """Kahn's topological sort."""
    remaining = dict(in_degree)
    queue = deque(nid for nid in node_ids if remaining.get(nid, 0) == 0)
    order: list[str] = []
    while queue:
        nid = queue.popleft()
        order.append(nid)
        for child in adj.get(nid, []):
            remaining[child] -= 1
            if remaining[child] == 0:
                queue.append(child)
    seen = set(order)
    for nid in node_ids:
        if nid not in seen:
            order.append(nid)
    return order


# ---------------------------------------------------------------------------
# Phase 3: crossing minimization
# ---------------------------------------------------------------------------

def _sort_by_barycenter(layer, neighbor_adj, pos):
    """Sort layer in-place by barycenter of neighbors."""
    bc: dict[str, float] = {}
    for nid in layer:
        nbs = neighbor_adj.get(nid, [])
        bc[nid] = (
            sum(pos.get(nb, 0) for nb in nbs) / len(nbs)
            if nbs else pos.get(nid, 0)
        )
    layer.sort(key=lambda nid: bc[nid])


# ---------------------------------------------------------------------------
# Phase 4b: long-edge straightening & subgraph-boundary routing
# ---------------------------------------------------------------------------

EDGE_SPACING = 8  # px between parallel long-edge corridors


def _nudge_clear(ideal_x: float, intervals: list[tuple[float, float]]) -> float:
    """Find the nearest x to *ideal_x* that clears all node intervals.

    Each interval is (left, right) of a real node. The returned x is
    NOT inside any padded interval ``[left-NODE_SPACING, right+NODE_SPACING]``.

    Builds the union of padded intervals, finds the gaps between them,
    and returns the closest gap-edge or unbounded extent to *ideal_x*.
    Handles tightly-packed obstacles (where adjacent padded intervals
    overlap) by skipping past the merged obstacle entirely.
    """
    if not intervals:
        return ideal_x

    padded = sorted(
        (l - NODE_SPACING, r + NODE_SPACING) for l, r in intervals
    )
    # Merge overlapping padded intervals
    merged: list[tuple[float, float]] = []
    for l, r in padded:
        if merged and l <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r))
        else:
            merged.append((l, r))

    # Check if ideal is inside any merged blocked range
    blocking: tuple[float, float] | None = None
    for l, r in merged:
        if l < ideal_x < r:
            blocking = (l, r)
            break
    if blocking is None:
        return ideal_x

    # Pick the closer side of the blocking interval
    dl = ideal_x - blocking[0]
    dr = blocking[1] - ideal_x
    return blocking[0] if dl <= dr else blocking[1]


def _snap_aux_nodes_to_input_column(
    x_of: dict[str, float],
    node_ids: list[str],
    edges: list[dict],
    layer_of: dict,
    layers: list[list[str]],
    ext_nmap: dict,
) -> None:
    """Pull single-input nodes back to their input's column when safe.

    Targets the Slice_1368 pattern: a node with one real input from
    the immediately preceding layer plus one or more long-edge outputs
    that pulled it horizontally far from its input. Snaps the node to
    align with the input only if the target slot is empty in the
    node's layer (no overlap with siblings).
    """
    # Build adjacency: node -> list of (real) predecessors
    pred_of: dict[str, list[str]] = defaultdict(list)
    succ_of: dict[str, list[str]] = defaultdict(list)
    node_set = set(node_ids)
    for e in edges:
        s, t = e["source"], e["target"]
        if s in node_set and t in node_set:
            pred_of[t].append(s)
            succ_of[s].append(t)

    SNAP_THRESHOLD = 300  # only snap when misalignment exceeds this
    SEARCH_WINDOW = 4 * 100  # 400 px around the ideal column
    MIN_IMPROVEMENT = 200    # snap must reduce misalignment by at least this
    for nid in node_ids:
        preds = pred_of.get(nid, [])
        succs = succ_of.get(nid, [])
        if not preds or not succs:
            continue
        # Allow 1 or 2 real inputs, all from the immediately preceding
        # layer. Two inputs covers tail subgraphs that take a value and
        # an axis from the same parent block.
        if len(preds) > 2:
            continue
        if any(layer_of[p] != layer_of[nid] - 1 for p in preds):
            continue
        # Only fire when every successor is a long edge (target is at
        # least 2 layers below). A node with short successors should
        # stay where Brandes-Köpf placed it for the local routing.
        if any(layer_of[c] - layer_of[nid] < 2 for c in succs):
            continue

        n = ext_nmap[nid]
        node_w = n["width"]
        cur_cx = x_of[nid] + node_w / 2
        # Target is the centroid of the (1 or 2) parents
        target_cx = sum(
            x_of[p] + ext_nmap[p]["width"] / 2 for p in preds
        ) / len(preds)
        cur_misalign = abs(cur_cx - target_cx)
        if cur_misalign < SNAP_THRESHOLD:
            continue

        L = layer_of[nid]
        # Build sibling intervals for collision detection
        siblings: list[tuple[float, float]] = []
        for other in layers[L]:
            if other == nid:
                continue
            o = ext_nmap.get(other, {})
            ow = o.get("width", 0)
            if ow == 0:
                continue
            ox = x_of.get(other)
            if ox is None:
                continue
            siblings.append((ox - NODE_SPACING, ox + ow + NODE_SPACING))
        siblings.sort()

        ideal_x = target_cx - node_w / 2

        def is_clear(x: float) -> bool:
            left = x
            right = x + node_w
            for ol, or_ in siblings:
                if right < ol or left > or_:
                    continue
                return False
            return True

        # Generate candidate positions near the ideal: ideal itself,
        # plus the just-right-of and just-left-of every sibling.
        candidates: list[float] = [ideal_x]
        for ol, or_ in siblings:
            candidates.append(or_ + 0.01)
            candidates.append(ol - node_w - 0.01)
        # Restrict to candidates that are within SEARCH_WINDOW of the
        # ideal AND would move the node by at least MIN_IMPROVEMENT
        # closer to ideal.
        candidates = [
            c for c in candidates
            if abs(c - ideal_x) <= SEARCH_WINDOW
            and abs(c + node_w / 2 - target_cx) <= cur_misalign - MIN_IMPROVEMENT
        ]
        if not candidates:
            continue

        candidates.sort(key=lambda c: abs(c - ideal_x))
        for c in candidates:
            if is_clear(c):
                x_of[nid] = c
                break


def _find_nearest_clear_corridor_base(
    center: float,
    budget: float,
    max_extent: float,
    dummies: list[str],
    layer_intervals: list[list[tuple[float, float]]],
    layer_of: dict,
    side: int = 0,
) -> float | None:
    """Find the nearest x to `center` (within ±budget) that allows a
    corridor of half-width `max_extent` to clear every dummy layer's
    obstacles. ``side`` constrains the search direction:
        +1 → only positions ≥ center (right of source)
        −1 → only positions ≤ center (left of source)
         0 → either side
    Returns None if no clean position exists in the search range.
    """
    if side > 0:
        lo = center
        hi = center + budget
    elif side < 0:
        lo = center - budget
        hi = center
    else:
        lo = center - budget
        hi = center + budget
    blocks: list[tuple[float, float]] = []
    for did in dummies:
        for left, right in layer_intervals[layer_of[did]]:
            l_padded = left - NODE_SPACING - max_extent
            r_padded = right + NODE_SPACING + max_extent
            if l_padded > hi or r_padded < lo:
                continue
            blocks.append((max(l_padded, lo), min(r_padded, hi)))
    blocks.sort()

    merged: list[tuple[float, float]] = []
    for l, r in blocks:
        if merged and l <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r))
        else:
            merged.append((l, r))

    # Build clear ranges between blocked intervals
    candidates: list[tuple[float, float]] = []
    cur = lo
    for l, r in merged:
        if cur < l:
            candidates.append((cur, l))
        cur = max(cur, r)
    if cur < hi:
        candidates.append((cur, hi))

    if not candidates:
        return None

    best_x: float | None = None
    best_dist = float("inf")
    for l, r in candidates:
        x = max(l, min(center, r))  # closest point in [l, r] to center
        d = abs(x - center)
        if d < best_dist:
            best_dist = d
            best_x = x
    return best_x


def _region_lca(r1: "Region", r2: "Region") -> "Region | None":
    """Lowest common ancestor of two regions in the region tree."""
    if r1 is None or r2 is None:
        return None
    if r1 is r2:
        return r1
    ancestors1: set[int] = set()
    cur: "Region | None" = r1
    while cur is not None:
        ancestors1.add(id(cur))
        cur = cur.parent
    cur = r2
    while cur is not None:
        if id(cur) in ancestors1:
            return cur
        cur = cur.parent
    return None


def _snap_nudges_to_column(
    nudges: list[float],
    dummies: list[str],
    layer_of: dict,
    layer_intervals: list[list[tuple[float, float]]],
    anchor: float | None = None,
) -> list[float]:
    """Collapse a per-dummy nudge list to a single x when geometrically safe.

    Builds the union of padded blocked intervals across every dummy
    layer, then picks the closest gap edge (or unbounded extent) to
    ``anchor`` (defaults to the nudge median). Gap-aware so a column
    can be found even when no per-layer nudge value happens to clear
    every other layer.
    """
    if len(nudges) < 2:
        return nudges
    if max(nudges) == min(nudges):
        return nudges

    median = anchor if anchor is not None else sorted(nudges)[len(nudges) // 2]

    # Collect every padded blocked interval across all dummy layers
    blocks: list[tuple[float, float]] = []
    for did in dummies:
        for left, right in layer_intervals[layer_of[did]]:
            blocks.append((left - NODE_SPACING, right + NODE_SPACING))
    if not blocks:
        return [median] * len(nudges)
    blocks.sort()

    # Merge overlapping intervals
    merged: list[tuple[float, float]] = []
    for l, r in blocks:
        if merged and l <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r))
        else:
            merged.append((l, r))

    # If median is in a clear gap, use it directly
    in_blocked: tuple[float, float] | None = None
    for l, r in merged:
        if l < median < r:
            in_blocked = (l, r)
            break
    if in_blocked is None:
        return [median] * len(nudges)

    # Pick the nearer side of the blocking interval
    dl = median - in_blocked[0]
    dr = in_blocked[1] - median
    candidate = in_blocked[0] if dl <= dr else in_blocked[1]
    return [candidate] * len(nudges)


def _straighten_long_edges(
    x_of, edge_chains, layer_of, ext_nmap, layers, num_layers, edges,
    node_region: dict[str, "Region"] | None = None,
):
    """Post-process long-edge dummy positions for visual quality.

    Strategy per chain:
      1. Compute the ideal straight-line path (linear interpolation src→tgt).
      2. If the straight line is clear of real nodes → use it (fan-out edges).
      3. Otherwise → route alongside the subgraph boundary (skip-connections).

    Parallel edges from the same source get consistent spacing.
    Uses actual target-port arrival x (not node center) for smooth interpolation.

    When ``node_region`` is provided, the obstacle field is filtered per
    source-group so an edge inside a residual block doesn't see nodes
    from unrelated parts of the network as obstacles.
    """
    # Build real-node intervals per layer for overlap checking. Each
    # interval also remembers its node id so we can filter the field
    # per source-group later.
    layer_intervals: list[list[tuple[float, float]]] = [[] for _ in range(num_layers)]
    layer_intervals_with_id: list[list[tuple[float, float, str]]] = (
        [[] for _ in range(num_layers)]
    )
    corridor_obstacles: list[list[tuple[float, float]]] = [[] for _ in range(num_layers)]
    for L in range(num_layers):
        rows: list[tuple[float, float, str]] = []
        for nid in layers[L]:
            w = ext_nmap[nid]["width"]
            if w > 0:
                rows.append((x_of[nid], x_of[nid] + w, nid))
        rows.sort()
        layer_intervals_with_id[L] = rows
        layer_intervals[L] = [(l, r) for (l, r, _) in rows]

    # Cache of per-scope filtered intervals so each unique scope only
    # pays the filter cost once.
    _scope_cache: dict[int, list[list[tuple[float, float]]]] = {}

    def _intervals_for_scope(scope: "Region | None") -> list[list[tuple[float, float]]]:
        if scope is None:
            return layer_intervals
        key = id(scope)
        cached = _scope_cache.get(key)
        if cached is not None:
            return cached
        scope_nodes = scope.nodes
        filtered = [
            [(l, r) for (l, r, nid) in layer_intervals_with_id[L] if nid in scope_nodes]
            for L in range(num_layers)
        ]
        _scope_cache[key] = filtered
        return filtered

    def _scope_for(src_n: str, tgt_n: str) -> "Region | None":
        if node_region is None:
            return None
        rs = node_region.get(src_n)
        rt = node_region.get(tgt_n)
        if rs is None or rt is None:
            return None
        return _region_lca(rs, rt)

    # Pre-compute target port counts for accurate arrival positions.
    # Use the model's original input count when available so edges
    # arrive at their actual port slot even when sibling inputs are
    # constants that were filtered from the visible graph.
    tgt_port_cnt: dict[str, int] = defaultdict(int)
    for nid, n in ext_nmap.items():
        if isinstance(n, dict) and n.get("total_inputs"):
            tgt_port_cnt[nid] = n["total_inputs"]
    for e in edges:
        p = e.get("target_port", 0) + 1
        if p > tgt_port_cnt[e["target"]]:
            tgt_port_cnt[e["target"]] = p

    # Group edge chains by source for parallel spacing
    source_groups: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    for i, chain in enumerate(edge_chains):
        if len(chain) > 2:
            source_groups[chain[0]].append((i, chain))

    # Classify source groups: "major" (high-fan-out, e.g. Where_2 with
    # 12 corridor edges) vs "minor" (1-3 skip connections).  Only major
    # groups participate in global corridor avoidance; minor groups stay
    # local and don't pollute corridor_obstacles.
    MAJOR_THRESHOLD = 4  # corridor edges needed to be "major"
    major_side_taken: str | None = None  # 'left' or 'right'

    # Process major groups first (most edges first, then left-to-right)
    # so they get first pick of sides.  Minor groups after.
    sorted_sources = sorted(
        source_groups.items(),
        key=lambda sc: (-len(sc[1]), x_of[sc[0]] + ext_nmap[sc[0]]["width"] / 2),
    )

    for src, chains in sorted_sources:
        # Sort by target x for deterministic parallel ordering
        chains.sort(key=lambda ic: x_of[ic[1][-1]])
        n_parallel = len(chains)
        src_cx = x_of[src] + ext_nmap[src]["width"] / 2
        is_major = n_parallel >= MAJOR_THRESHOLD

        # --- Pass 1: compute per-edge routing (straight vs corridor) ---
        edge_routes: list[dict] = []  # per-chain routing info
        for rank, (idx, chain) in enumerate(chains):
            dummies = chain[1:-1]
            if not dummies:
                edge_routes.append({"mode": "skip"})
                continue

            src_n, tgt_n = chain[0], chain[-1]
            src_L, tgt_L = layer_of[src_n], layer_of[tgt_n]
            span = tgt_L - src_L
            if span == 0:
                edge_routes.append({"mode": "skip"})
                continue

            e = edges[idx]
            ntports = tgt_port_cnt.get(tgt_n, 1)
            if ntports > 1:
                tpt = e.get("target_port", 0)
                tgt_cx = x_of[tgt_n] + ext_nmap[tgt_n]["width"] / (ntports + 1) * (tpt + 1)
            else:
                tgt_cx = x_of[tgt_n] + ext_nmap[tgt_n]["width"] / 2

            # Region-scoped obstacle field: an edge inside a residual
            # block (or any nested SESE region) only "sees" obstacles
            # in its smallest enclosing region. Outside that scope the
            # edge_intervals fall back to the global layer_intervals.
            edge_scope = _scope_for(src_n, tgt_n)
            edge_intervals = _intervals_for_scope(edge_scope)

            # Try step-routing strategies and pick the first that clears
            # all intermediate layers.  For major fan-out groups (≥4
            # parallel long edges), only try the standard step — we want
            # those to fall through to corridor mode for bundled routing.
            par_off = (rank - (n_parallel - 1) / 2) * EDGE_SPACING if n_parallel > 1 else 0
            strategies = [
                lambda _t: (src_cx + par_off) if _t <= 0.5 else tgt_cx,
            ]
            if n_parallel < MAJOR_THRESHOLD:
                strategies += [
                    lambda _t: tgt_cx,
                    lambda _t: src_cx + par_off,
                ]
            chosen_strat = None
            for strat in strategies:
                ok = True
                for did in dummies:
                    dL = layer_of[did]
                    t = (dL - src_L) / span
                    check_x = strat(t)
                    for left, right in edge_intervals[dL]:
                        if left - NODE_SPACING <= check_x <= right + NODE_SPACING:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    chosen_strat = strat
                    break

            if chosen_strat is not None:
                edge_routes.append({
                    "mode": "straight", "chain": chain, "dummies": dummies,
                    "src_L": src_L, "tgt_L": tgt_L, "src_cx": src_cx,
                    "tgt_cx": tgt_cx, "span": span,
                    "strat": chosen_strat,
                })
            else:
                # Compute the best nudge. For minor groups (typical
                # skip-connections), accept it whenever the worst-case
                # per-dummy displacement stays small — that lets a
                # column-aligned skip hug the backbone even when its
                # span exceeds 5 layers. Major fan-out groups always
                # bundle into a shared corridor (the only way to keep
                # 4+ parallel edges visually distinguishable), so for
                # them we keep the original span<=5 nudge fallback.
                NUDGE_MAX_DISP = 8 * NODE_SPACING  # 160 px per dummy
                best_nudge: list[float] | None = None
                best_max_disp = float("inf")
                for strat in strategies:
                    xs: list[float] = []
                    max_disp = 0.0
                    for did in dummies:
                        dL = layer_of[did]
                        t = (dL - src_L) / span
                        baseline = strat(t)
                        raw = baseline - par_off  # remove par_off for baseline nudge
                        nudged = _nudge_clear(raw, edge_intervals[dL])
                        final = _nudge_clear(nudged + par_off, edge_intervals[dL])
                        xs.append(final)
                        max_disp = max(max_disp, abs(final - baseline))
                    if max_disp < best_max_disp:
                        best_max_disp = max_disp
                        best_nudge = xs

                if is_major:
                    accept_nudge = best_nudge is not None and span <= 10
                else:
                    accept_nudge = best_nudge is not None and best_max_disp <= NUDGE_MAX_DISP

                if accept_nudge:
                    # Collapse the per-dummy zigzag to a single column when
                    # safe, so the rendered spline traces a straight vertical
                    # segment instead of weaving between obstacles.
                    best_nudge = _snap_nudges_to_column(
                        best_nudge, dummies, layer_of, edge_intervals,
                    )
                    pass

                if accept_nudge:
                    edge_routes.append({
                        "mode": "nudge", "chain": chain, "dummies": dummies,
                        "nudge_xs": best_nudge,
                    })
                else:
                    edge_routes.append({
                        "mode": "corridor", "chain": chain, "dummies": dummies,
                        "tgt_cx": tgt_cx, "tgt_L": tgt_L, "span": span,
                    })

        # Count corridor edges to decide routing strategy
        corridor_routes = [r for r in edge_routes if r.get("mode") == "corridor"]
        if not corridor_routes:
            # Apply nudge and straight-line edges only
            for rank, (idx, chain) in enumerate(chains):
                route = edge_routes[rank]
                if route["mode"] == "nudge":
                    for did, nx in zip(route["dummies"], route["nudge_xs"]):
                        x_of[did] = nx
                elif route["mode"] == "straight":
                    strat = route["strat"]
                    for did in route["dummies"]:
                        dL = layer_of[did]
                        t = (dL - route["src_L"]) / route["span"]
                        x_of[did] = strat(t)
            continue

        # --- Compute corridor boundary (real nodes only for side decision) ---
        # Scan all intermediate layers to find the real-node extent.
        all_corridor_dummies = []
        for r in corridor_routes:
            all_corridor_dummies.extend(r["dummies"])

        max_boundary = -float("inf")
        min_boundary = float("inf")
        if is_major:
            # Major groups: scan ALL real nodes in intermediate layers
            for did in all_corridor_dummies:
                dL = layer_of[did]
                for left, right in layer_intervals[dL]:
                    if right > max_boundary:
                        max_boundary = right
                    if left < min_boundary:
                        min_boundary = left
        else:
            # Minor groups: bounded scan around src/tgt
            src_x = x_of[src]
            src_r = src_x + ext_nmap[src]["width"]
            all_tgt_x = [r["tgt_cx"] for r in corridor_routes]
            max_span = max(r["span"] for r in corridor_routes)
            margin = min(NODE_SPACING * 5, NODE_SPACING + max_span * NODE_SPACING // 2)
            range_lo = min(src_x, min(all_tgt_x)) - margin
            range_hi = max(src_r, max(all_tgt_x)) + margin
            for did in all_corridor_dummies:
                dL = layer_of[did]
                for left, right in layer_intervals[dL]:
                    if right >= range_lo and left <= range_hi:
                        if right > max_boundary:
                            max_boundary = right
                        if left < min_boundary:
                            min_boundary = left

        left_corr = min_boundary - NODE_SPACING if min_boundary < float("inf") else src_cx
        right_corr = max_boundary + NODE_SPACING if max_boundary > -float("inf") else src_cx

        # --- Side decision ---
        if is_major:
            # Major groups: go TOWARD the median target so edges never
            # swing away from their destinations.  Multiple major groups
            # on the same side are handled by conflict resolution
            # (stacking), not by forcing to opposite sides.
            median_tgt = sorted(r["tgt_cx"] for r in corridor_routes)[len(corridor_routes) // 2]
            group_go_right = median_tgt > src_cx

            for r in corridor_routes:
                r["go_right"] = group_go_right
                r["base"] = right_corr if group_go_right else left_corr
        else:
            # Minor groups: per-edge decision based on local boundary
            for r in corridor_routes:
                tgt_cx = r["tgt_cx"]
                if src_cx > max_boundary or src_cx < min_boundary:
                    dl = abs(src_cx - left_corr) + abs(tgt_cx - left_corr)
                    dr = abs(src_cx - right_corr) + abs(tgt_cx - right_corr)
                    r["go_right"] = dr <= dl
                else:
                    r["go_right"] = tgt_cx >= src_cx
                r["base"] = right_corr if r["go_right"] else left_corr

        # --- Pass 2 & 3: compute shared base per side, assign by target order ---
        left_indices = [i for i, r in enumerate(edge_routes) if r.get("mode") == "corridor" and not r["go_right"]]
        right_indices = [i for i, r in enumerate(edge_routes) if r.get("mode") == "corridor" and r["go_right"]]

        for go_right, side_indices in ((False, left_indices), (True, right_indices)):
            if not side_indices:
                continue
            side_routes = [edge_routes[i] for i in side_indices]
            side_count = len(side_routes)
            max_extent = (side_count - 1) / 2 * EDGE_SPACING if side_count > 1 else 0

            # Initial base: median of per-edge bases
            bases = sorted(r["base"] for r in side_routes)
            base = bases[len(bases) // 2]

            # Collect ALL dummies from all edges in this side group,
            # plus a few layers near the source so the corridor entry
            # curve doesn't cross nearby nodes.
            all_dummies = []
            for r in side_routes:
                all_dummies.extend(r["dummies"])

            src_L = layer_of[src]
            # Only check the source layer and one below for B-spline
            # entry curve clearance (not above or two below).
            entry_layers = [L for L in range(src_L, min(num_layers, src_L + 2))
                            if layer_intervals[L]]

            # Resolve: major groups avoid real nodes AND other major
            # corridors.  Minor groups only avoid real nodes.
            src_right = x_of[src] + ext_nmap[src]["width"]
            src_left = x_of[src]
            for _ in range(20):
                clear = True
                lo_x = base - max_extent
                hi_x = base + max_extent
                for did in all_dummies:
                    dL = layer_of[did]
                    obstacles = layer_intervals[dL] + corridor_obstacles[dL] if is_major else layer_intervals[dL]
                    for left, right in obstacles:
                        if left - NODE_SPACING < hi_x and lo_x < right + NODE_SPACING:
                            base = ((right + NODE_SPACING + max_extent)
                                    if go_right
                                    else (left - NODE_SPACING - max_extent))
                            clear = False
                # B-spline entry curve check: only for nodes between
                # the source and the corridor (not on the opposite side).
                for eL in entry_layers:
                    for left, right in layer_intervals[eL]:
                        if go_right:
                            # Skip nodes entirely to the left of source
                            if right + NODE_SPACING < src_right:
                                continue
                            min_corr = (6 * (right + NODE_SPACING) - src_right) / 5
                            needed = min_corr + max_extent
                            if base < needed:
                                base = needed
                                clear = False
                        else:
                            # Skip nodes entirely to the right of source
                            if left - NODE_SPACING > src_left:
                                continue
                            max_corr = (6 * (left - NODE_SPACING) - src_left) / 5
                            needed = max_corr - max_extent
                            if base > needed:
                                base = needed
                                clear = False
                if clear:
                    break

            # Smart re-anchoring: replace the walked-outward base with
            # the closest overlap-free position to src_cx within a
            # ±budget window. The conflict-resolution loop above
            # settles on the FIRST clean position in one direction;
            # smart re-anchor checks for any closer clean position on
            # either side. If no clean position exists in the window,
            # fall back to the walked-far base (lesser evil — far but
            # no overlap). Major groups get a wider budget because
            # their parallel-edge spread is large.
            CORRIDOR_DRIFT_BUDGET = (
                20 * NODE_SPACING if is_major else 25 * NODE_SPACING
            )
            closer = _find_nearest_clear_corridor_base(
                src_cx, CORRIDOR_DRIFT_BUDGET, max_extent,
                all_dummies, layer_intervals, layer_of,
            )
            if closer is not None and abs(closer - src_cx) < abs(base - src_cx):
                base = closer

            # Assign par_off by peel-off order
            order = sorted(range(side_count), key=lambda k: side_routes[k]["tgt_L"])
            if not go_right:
                order = list(reversed(order))
            for rank_in_side, k in enumerate(order):
                par_off = (rank_in_side - (side_count - 1) / 2) * EDGE_SPACING if side_count > 1 else 0
                pos_x = base + par_off
                dummies_k = side_routes[k]["dummies"]
                for did in dummies_k:
                    x_of[did] = pos_x

            # Only major groups register corridor obstacles — minor
            # groups are local and shouldn't affect global routing.
            if is_major:
                for did in all_dummies:
                    dL = layer_of[did]
                    corridor_obstacles[dL].append((x_of[did], x_of[did]))

        # Apply nudge and straight-line edges
        for rank, (idx, chain) in enumerate(chains):
            route = edge_routes[rank]
            if route["mode"] == "nudge":
                for did, nx in zip(route["dummies"], route["nudge_xs"]):
                    x_of[did] = nx
            elif route["mode"] == "straight":
                strat = route["strat"]
                for did in route["dummies"]:
                    dL = layer_of[did]
                    t = (dL - route["src_L"]) / route["span"]
                    x_of[did] = strat(t)


# ---------------------------------------------------------------------------
# Phase 4: Brandes-Köpf coordinate assignment
# ---------------------------------------------------------------------------

def _brandes_kopf(layers, num_layers, nmap, ext_adj, ext_rev, pos):
    """4-variant Brandes-Köpf with median combination.

    Runs vertical alignment + horizontal compaction for each of:
      up-left, up-right, down-left, down-right
    Then takes the median of the 4 x-values per node.
    """
    all_nodes = [nid for L in range(num_layers) for nid in layers[L]]

    results: list[dict[str, float]] = []
    for v_dir in ("up", "down"):
        for h_dir in ("left", "right"):
            if h_dir == "right":
                work = [list(reversed(layers[L])) for L in range(num_layers)]
                wpos = {nid: i for L in range(num_layers) for i, nid in enumerate(work[L])}
            else:
                work = layers
                wpos = dict(pos)

            nadj = ext_rev if v_dir == "up" else ext_adj
            root, aln = _vert_align(work, num_layers, nadj, wpos, v_dir)
            x = _horiz_compact(
                work, num_layers, root, aln, nmap, all_nodes,
                reverse=(h_dir == "right"),
            )

            if h_dir == "right":
                for nid in x:
                    x[nid] = -x[nid]

            results.append(x)

    # Align each result so min x = 0
    for x in results:
        if x:
            mn = min(x.values())
            for nid in x:
                x[nid] -= mn

    # Median of 4 values per node
    x_of: dict[str, float] = {}
    for nid in all_nodes:
        vals = sorted(r.get(nid, 0) for r in results)
        x_of[nid] = (vals[1] + vals[2]) / 2  # median of 4

    # Global centering: align each layer to the widest bounding box
    max_right = 0.0
    for L in range(num_layers):
        for nid in layers[L]:
            r = x_of[nid] + nmap[nid]["width"]
            if r > max_right:
                max_right = r
    for L in range(num_layers):
        if not layers[L]:
            continue
        mn = min(x_of[nid] for nid in layers[L])
        mx = max(x_of[nid] + nmap[nid]["width"] for nid in layers[L])
        off = (max_right - (mx - mn)) / 2 - mn
        for nid in layers[L]:
            x_of[nid] += off

    return x_of


def _vert_align(layers, num_layers, nadj, pos, v_dir):
    """Vertical alignment for one direction (up or down).

    Aligns each node with its median neighbor in the adjacent layer,
    building blocks of vertically aligned nodes.
    """
    root: dict[str, str] = {}
    aln: dict[str, str] = {}
    for L in range(num_layers):
        for nid in layers[L]:
            root[nid] = nid
            aln[nid] = nid

    layer_range = range(1, num_layers) if v_dir == "up" else range(num_layers - 2, -1, -1)

    for L in layer_range:
        r = -1
        for v in layers[L]:
            nbs = nadj.get(v, [])
            if not nbs:
                continue
            nbrs = sorted(nbs, key=lambda u: pos.get(u, 0))
            d = len(nbrs)
            lo = (d - 1) // 2
            hi = d // 2
            for m in range(lo, hi + 1):
                if aln[v] != v:
                    break
                u = nbrs[m]
                p = pos.get(u, 0)
                if p > r:
                    aln[u] = v
                    root[v] = root[u]
                    aln[v] = root[v]
                    r = p

    return root, aln


def _horiz_compact(layers, num_layers, root, aln, nmap, all_nodes, *, reverse=False):
    """Horizontal compaction: assign x-coordinates to blocks.

    Uses topological ordering of block roots to avoid recursion depth issues.
    When reverse=True, adjusts separation for right-to-left packing.
    """
    # Left neighbor in each layer
    pred: dict[str, str | None] = {}
    for L in range(num_layers):
        for i, nid in enumerate(layers[L]):
            pred[nid] = layers[L][i - 1] if i > 0 else None

    sink: dict[str, str] = {v: v for v in all_nodes}
    shift: dict[str, float] = {v: float("inf") for v in all_nodes}
    x: dict[str, float] = {}

    # Topological sort of block roots by left-neighbor dependencies
    roots = [v for v in all_nodes if root[v] == v]
    deps: dict[str, set[str]] = defaultdict(set)
    for v in roots:
        w = v
        while True:
            p = pred.get(w)
            if p is not None:
                u = root[p]
                if u != v:
                    deps[v].add(u)
            w = aln[w]
            if w == v:
                break

    rdeps: dict[str, set[str]] = defaultdict(set)
    ideg: dict[str, int] = {}
    for v in roots:
        ideg[v] = len(deps[v])
        for u in deps[v]:
            rdeps[u].add(v)

    queue = deque(v for v in roots if ideg.get(v, 0) == 0)
    order: list[str] = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for w in rdeps[v]:
            ideg[w] -= 1
            if ideg[w] == 0:
                queue.append(w)
    # Safety: append any remaining roots (shouldn't happen in a DAG)
    placed = set(order)
    for v in roots:
        if v not in placed:
            order.append(v)

    # Place blocks in topological order
    for v in order:
        x[v] = 0
        w = v
        while True:
            p = pred.get(w)
            if p is not None:
                u = root[p]
                # Separation: use width[w] for reversed (right-to-left) packing
                # to get correct gaps after negation; width[pred] for normal
                # Use tight spacing when both nodes are dummies (invisible routing points)
                w_dummy = nmap[w]["width"] == 0 and nmap[w]["height"] == 0
                p_dummy = nmap[p]["width"] == 0 and nmap[p]["height"] == 0
                gap = DUMMY_SPACING if (w_dummy and p_dummy) else NODE_SPACING
                sep = nmap[w]["width"] + gap if reverse else nmap[p]["width"] + gap
                if sink[v] == v:
                    sink[v] = sink[u]
                if sink[v] != sink[u]:
                    shift[sink[u]] = min(shift[sink[u]], x[v] - x[u] - sep)
                else:
                    x[v] = max(x[v], x[u] + sep)
            w = aln[w]
            if w == v:
                break

    # Resolve absolute coordinates
    result: dict[str, float] = {}
    for v in all_nodes:
        rv = root[v]
        result[v] = x.get(rv, 0)
        s = shift.get(sink.get(rv, rv), float("inf"))
        if s < float("inf"):
            result[v] += s

    return result
