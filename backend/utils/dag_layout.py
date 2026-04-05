"""Pure-Python layered DAG layout (Sugiyama + Brandes-Köpf).

Produces Netron-like appearance: straight edges for sequential chains,
compact layout with minimized crossings. Uses dummy nodes for long-edge
routing and 4-variant Brandes-Köpf for near-optimal coordinate assignment.
"""
from __future__ import annotations

from collections import defaultdict, deque

# Spacing constants (match Netron/ELK dagre config)
NODE_SPACING = 20   # horizontal gap between nodes in same layer
LAYER_SPACING = 20  # vertical gap between layers
DUMMY_SPACING = 2   # horizontal gap between dummy nodes (tight bundling)


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

    # Phase 4b: Straighten long edges and route alongside subgraph boundaries
    _straighten_long_edges(x_of, edge_chains, layer_of, ext_nmap, layers, num_layers, edges)

    # Phase 5: Y-coordinates
    layer_y: list[float] = [0.0] * num_layers
    layer_mid: list[float] = [0.0] * num_layers
    y = 0.0
    for L in range(num_layers):
        layer_y[L] = y
        max_h = max((ext_nmap[nid]["height"] for nid in layers[L]), default=0)
        layer_mid[L] = y + max_h / 2
        y += max_h + LAYER_SPACING

    # Phase 6: Build result (only real nodes in positions)
    positions = {nid: {"x": x_of[nid], "y": layer_y[layer_of[nid]]} for nid in node_ids}

    # Edge waypoints (using dummy positions for intermediate points)
    # Source port counts (max source_port + 1 per node)
    port_counts: dict[str, int] = defaultdict(int)
    for e in edges:
        p = e.get("source_port", 0) + 1
        if p > port_counts[e["source"]]:
            port_counts[e["source"]] = p

    # Target port counts (max target_port + 1 per node)
    target_port_counts: dict[str, int] = defaultdict(int)
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
        for did in chain[1:-1]:
            dL = layer_of[did]
            wps.append({"x": x_of[did], "y": layer_mid[dL]})
        wps.append({"x": ex, "y": ey})

        ewp[f"e{i}"] = {"waypoints": wps}

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


def _straighten_long_edges(x_of, edge_chains, layer_of, ext_nmap, layers, num_layers, edges):
    """Post-process long-edge dummy positions for visual quality.

    Strategy per chain:
      1. Compute the ideal straight-line path (linear interpolation src→tgt).
      2. If the straight line is clear of real nodes → use it (fan-out edges).
      3. Otherwise → route alongside the subgraph boundary (skip-connections).

    Parallel edges from the same source get consistent spacing.
    Uses actual target-port arrival x (not node center) for smooth interpolation.
    """
    # Build real-node intervals per layer for overlap checking
    layer_intervals: list[list[tuple[float, float]]] = [[] for _ in range(num_layers)]
    for L in range(num_layers):
        for nid in layers[L]:
            w = ext_nmap[nid]["width"]
            if w > 0:
                layer_intervals[L].append((x_of[nid], x_of[nid] + w))
        layer_intervals[L].sort()

    # Pre-compute target port counts for accurate arrival positions
    tgt_port_cnt: dict[str, int] = defaultdict(int)
    for e in edges:
        p = e.get("target_port", 0) + 1
        if p > tgt_port_cnt[e["target"]]:
            tgt_port_cnt[e["target"]] = p

    # Group edge chains by source for parallel spacing
    source_groups: dict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    for i, chain in enumerate(edge_chains):
        if len(chain) > 2:
            source_groups[chain[0]].append((i, chain))

    for src, chains in source_groups.items():
        # Sort by target x for deterministic parallel ordering
        chains.sort(key=lambda ic: x_of[ic[1][-1]])
        n_parallel = len(chains)
        src_cx = x_of[src] + ext_nmap[src]["width"] / 2

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

            # Check whether the ideal straight line is clear
            straight_ok = True
            for did in dummies:
                dL = layer_of[did]
                t = (dL - src_L) / span
                ix = src_cx + t * (tgt_cx - src_cx)
                for left, right in layer_intervals[dL]:
                    if left - NODE_SPACING <= ix <= right + NODE_SPACING:
                        straight_ok = False
                        break
                if not straight_ok:
                    break

            # Reject steep diagonals
            if straight_ok and abs(src_cx - tgt_cx) > ext_nmap[src_n]["width"]:
                straight_ok = False

            if straight_ok:
                edge_routes.append({
                    "mode": "straight", "chain": chain, "dummies": dummies,
                    "src_L": src_L, "tgt_L": tgt_L, "src_cx": src_cx,
                    "tgt_cx": tgt_cx, "span": span,
                })
            else:
                # Compute corridor base for this edge
                src_x = x_of[src_n]
                tgt_x = x_of[tgt_n]
                src_r = src_x + ext_nmap[src_n]["width"]
                tgt_r = tgt_x + ext_nmap[tgt_n]["width"]
                margin = NODE_SPACING * 5
                range_lo = min(src_x, tgt_x) - margin
                range_hi = max(src_r, tgt_r) + margin

                max_boundary = -float("inf")
                min_boundary = float("inf")
                for did in dummies:
                    dL = layer_of[did]
                    for left, right in layer_intervals[dL]:
                        if right >= range_lo and left <= range_hi:
                            if right > max_boundary:
                                max_boundary = right
                            if left < min_boundary:
                                min_boundary = left

                left_corr = min_boundary - NODE_SPACING if min_boundary < float("inf") else src_cx
                right_corr = max_boundary + NODE_SPACING if max_boundary > -float("inf") else src_cx
                dist_left = abs(src_cx - left_corr) + abs(tgt_cx - left_corr)
                dist_right = abs(src_cx - right_corr) + abs(tgt_cx - right_corr)
                go_right = dist_right <= dist_left
                base = right_corr if go_right else left_corr

                edge_routes.append({
                    "mode": "corridor", "chain": chain, "dummies": dummies,
                    "base": base, "go_right": go_right, "tgt_cx": tgt_cx,
                })

        # --- Pass 2 & 3: compute shared base per side, assign by target order ---
        # Group corridor edges by side, compute ONE shared base that clears
        # ALL edges in the side, then assign par_off by target x order.
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

            # Collect ALL dummies from all edges in this side group
            all_dummies = []
            for r in side_routes:
                all_dummies.extend(r["dummies"])

            # Resolve: ensure base ± max_extent clears all nodes at all layers
            for _ in range(20):
                clear = True
                lo_x = base - max_extent
                hi_x = base + max_extent
                for did in all_dummies:
                    dL = layer_of[did]
                    for left, right in layer_intervals[dL]:
                        if left - NODE_SPACING < hi_x and lo_x < right + NODE_SPACING:
                            base = ((right + NODE_SPACING + max_extent)
                                    if go_right
                                    else (left - NODE_SPACING - max_extent))
                            clear = False
                if clear:
                    break

            # Assign par_off by target x order (leftmost target → leftmost slot)
            order = sorted(range(side_count), key=lambda k: side_routes[k]["tgt_cx"])
            for rank_in_side, k in enumerate(order):
                par_off = (rank_in_side - (side_count - 1) / 2) * EDGE_SPACING if side_count > 1 else 0
                for did in side_routes[k]["dummies"]:
                    x_of[did] = base + par_off

        # Apply straight-line edges
        for rank, (idx, chain) in enumerate(chains):
            route = edge_routes[rank]
            if route["mode"] == "straight":
                par_off = (rank - (n_parallel - 1) / 2) * EDGE_SPACING if n_parallel > 1 else 0
                for did in route["dummies"]:
                    dL = layer_of[did]
                    t = (dL - route["src_L"]) / route["span"]
                    x_of[did] = route["src_cx"] + t * (route["tgt_cx"] - route["src_cx"]) + par_off


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
