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
    port_counts: dict[str, int] = defaultdict(int)
    for e in edges:
        p = e.get("source_port", 0) + 1
        if p > port_counts[e["source"]]:
            port_counts[e["source"]] = p

    ewp: dict[str, dict] = {}
    for i, (e, chain) in enumerate(zip(edges, edge_chains)):
        sid, tid = e["source"], e["target"]
        sn, tn = ext_nmap[sid], ext_nmap[tid]

        # Start point (bottom of source, port-spread)
        nports = port_counts.get(sid, 1)
        if nports > 1:
            pt = e.get("source_port", 0)
            ps = sn["width"] / (nports + 1)
            sx = x_of[sid] + ps * (pt + 1)
        else:
            sx = x_of[sid] + sn["width"] / 2
        sy = layer_y[layer_of[sid]] + sn["height"]

        # End point (top-center of target)
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
                sep = nmap[w]["width"] + NODE_SPACING if reverse else nmap[p]["width"] + NODE_SPACING
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
