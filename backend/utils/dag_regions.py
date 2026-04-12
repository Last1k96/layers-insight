"""Single-Entry Single-Exit (SESE) region detection for DAGs.

Detects structurally cohesive subgraphs (residual blocks, inception
modules, transformer layers) by finding (entry, exit) pairs where
every path entering the region goes through `entry` and every path
leaving the region goes through `exit`. Uses dominator analysis.

Public API:

    region = find_sese_regions(node_ids, edges)
    # `region` is the root Region containing every node, with
    # `.children` for each minimal SESE subgraph nested inside.

    # Look up the smallest enclosing region for any node:
    node_to_region = build_node_region_map(region)
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Region:
    entry: str
    exit: str
    nodes: set[str] = field(default_factory=set)
    children: list["Region"] = field(default_factory=list)
    parent: Optional["Region"] = None

    def __repr__(self) -> str:
        return (
            f"Region({self.entry}..{self.exit}, "
            f"{len(self.nodes)} nodes, {len(self.children)} children)"
        )


def find_sese_regions(node_ids: list[str], edges: list[dict]) -> Region:
    """Find SESE regions in a DAG.

    Returns the root Region containing every node, with nested children
    for each minimal SESE subgraph found. Linear chains contain no
    nested regions (they're already simple).
    """
    if not node_ids:
        return Region(entry="", exit="", nodes=set())

    succ, pred = _build_adjacency(node_ids, edges)
    topo = _topo_sort(node_ids, succ, pred)
    if not topo:
        # Cycle (shouldn't happen for OV DAGs) — bail out with a flat root.
        return Region(entry=node_ids[0], exit=node_ids[-1], nodes=set(node_ids))

    topo_rank = {n: i for i, n in enumerate(topo)}

    # Add a virtual source so dominator analysis has a single entry.
    sources = [n for n in node_ids if not pred[n]]
    if not sources:
        sources = [topo[0]]

    idom = _dominators(topo, sources, pred)

    # Each merge node (in-degree > 1) is a candidate SESE region exit;
    # its corresponding entry is the lowest common dominator of its
    # predecessors. (Fork points without merges aren't SESE.)
    candidates: list[tuple[str, str, set[str]]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for v in topo:
        if len(pred[v]) <= 1:
            continue
        fork = _dom_tree_lca(list(pred[v]), idom)
        if fork is None or fork == v or fork not in topo_rank:
            continue
        pair = (fork, v)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        contained = _collect_between(fork, v, succ, pred, topo_rank)
        if not contained or len(contained) < 3:
            continue  # trivially small (just entry+exit) — not interesting
        # Verify SESE: every node in contained must have all its preds
        # also in contained (or be the entry), and all its succs in
        # contained (or be the exit).
        if _verify_sese(fork, v, contained, succ, pred):
            candidates.append((fork, v, contained))

    # Build the region tree: smallest first, then nest into the next
    # enclosing candidate.
    regions: list[Region] = [
        Region(entry=f, exit=v, nodes=set(c)) for f, v, c in candidates
    ]
    root = Region(entry=topo[0], exit=topo[-1], nodes=set(node_ids))
    regions.append(root)

    # Sort by size (smallest first) so that when we look for a parent
    # we find the smallest enclosing region.
    regions.sort(key=lambda r: len(r.nodes))
    for i, r in enumerate(regions):
        for j in range(i + 1, len(regions)):
            cand = regions[j]
            if r.nodes < cand.nodes:  # strict subset
                r.parent = cand
                cand.children.append(r)
                break

    return root


def build_node_region_map(root: Region) -> dict[str, Region]:
    """Map each node id to its smallest enclosing region."""
    mapping: dict[str, Region] = {}
    # DFS the region tree from root, deepest first wins.
    stack = [root]
    visit_order: list[Region] = []
    while stack:
        r = stack.pop()
        visit_order.append(r)
        stack.extend(r.children)
    # Visit deepest regions last so they overwrite ancestors.
    for r in sorted(visit_order, key=lambda x: -_depth(x)):
        for n in r.nodes:
            if n not in mapping:
                mapping[n] = r
    return mapping


def _depth(r: Region) -> int:
    d = 0
    cur: Optional[Region] = r
    while cur is not None:
        d += 1
        cur = cur.parent
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_adjacency(
    node_ids: list[str], edges: list[dict],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    succ: dict[str, set[str]] = {n: set() for n in node_ids}
    pred: dict[str, set[str]] = {n: set() for n in node_ids}
    node_set = set(node_ids)
    for e in edges:
        s = e.get("source")
        t = e.get("target")
        if s in node_set and t in node_set:
            succ[s].add(t)
            pred[t].add(s)
    return succ, pred


def _topo_sort(
    node_ids: list[str],
    succ: dict[str, set[str]],
    pred: dict[str, set[str]],
) -> list[str]:
    indeg = {n: len(pred[n]) for n in node_ids}
    queue: deque[str] = deque(n for n in node_ids if indeg[n] == 0)
    order: list[str] = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for c in succ[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    if len(order) != len(node_ids):
        return []  # cycle
    return order


def _dominators(
    topo: list[str],
    sources: list[str],
    pred: dict[str, set[str]],
) -> dict[str, str]:
    """Cooper-Harvey-Kennedy iterative dominators.

    Multiple sources are unified via a virtual root. Returns idom[v]
    for every node except the roots, which dominate themselves.
    """
    rank = {n: i for i, n in enumerate(topo)}
    VROOT = "__sese_vroot__"
    idom: dict[str, str] = {VROOT: VROOT}
    for s in sources:
        idom[s] = VROOT
    # Treat VROOT as having rank -1 so that intersect can compare it.
    rank[VROOT] = -1

    def intersect(b1: str, b2: str) -> str:
        # Walk both pointers up the dominator tree until they meet.
        # Both must already be in idom; rank gives "depth" via topo
        # order (smaller rank = closer to root in topo, deeper in dom).
        finger1, finger2 = b1, b2
        while finger1 != finger2:
            while rank.get(finger1, 10**9) > rank.get(finger2, 10**9):
                finger1 = idom.get(finger1, finger1)
                if finger1 == VROOT and finger2 == VROOT:
                    return VROOT
            while rank.get(finger2, 10**9) > rank.get(finger1, 10**9):
                finger2 = idom.get(finger2, finger2)
                if finger1 == VROOT and finger2 == VROOT:
                    return VROOT
        return finger1

    changed = True
    while changed:
        changed = False
        for v in topo:
            if v in sources:
                continue
            preds = [p for p in pred[v] if p in idom]
            if not preds:
                # Unreachable from any source — skip.
                continue
            new_idom = preds[0]
            for p in preds[1:]:
                if p in idom:
                    new_idom = intersect(p, new_idom)
            if idom.get(v) != new_idom:
                idom[v] = new_idom
                changed = True

    # Drop VROOT entries — convert to "node dominates itself" for sources.
    result: dict[str, str] = {}
    for v in topo:
        d = idom.get(v, v)
        if d == VROOT:
            result[v] = v
        else:
            result[v] = d
    return result


def _dom_tree_lca(nodes: list[str], idom: dict[str, str]) -> Optional[str]:
    """Lowest common ancestor of `nodes` in the dominator tree."""
    if not nodes:
        return None
    if len(nodes) == 1:
        return idom.get(nodes[0])

    def ancestors(n: str) -> list[str]:
        chain = [n]
        cur = idom.get(n)
        seen = {n}
        while cur is not None and cur not in seen:
            chain.append(cur)
            seen.add(cur)
            nxt = idom.get(cur)
            if nxt == cur:
                break
            cur = nxt
        return chain

    common = set(ancestors(nodes[0]))
    for n in nodes[1:]:
        common &= set(ancestors(n))
        if not common:
            return None
    # Pick the element of `common` with the deepest dominator-tree
    # position — equivalently, the one closest to any of the input
    # nodes. Use the first node's ancestor chain order.
    chain = ancestors(nodes[0])
    for c in chain:
        if c in common:
            return c
    return None


def _collect_between(
    entry: str,
    exit_node: str,
    succ: dict[str, set[str]],
    pred: dict[str, set[str]],
    topo_rank: dict[str, int],
) -> set[str]:
    """Collect all nodes reachable from entry that can also reach exit_node.

    Limited to nodes with topo rank between entry and exit_node so the
    BFS stays bounded. Uses the precomputed `pred` map for the reverse
    pass instead of scanning every candidate node.
    """
    if entry == exit_node:
        return {entry}
    lo = topo_rank.get(entry, 0)
    hi = topo_rank.get(exit_node, 10**9)
    if lo > hi:
        return set()

    # Forward BFS from entry, restricted to nodes with topo rank <= hi.
    forward: set[str] = {entry}
    queue: deque[str] = deque([entry])
    while queue:
        n = queue.popleft()
        for c in succ[n]:
            if c in forward:
                continue
            if topo_rank.get(c, 10**9) > hi:
                continue
            forward.add(c)
            queue.append(c)

    if exit_node not in forward:
        return set()

    # Reverse BFS from exit_node, restricted to the forward set.
    reachable: set[str] = {exit_node}
    queue = deque([exit_node])
    while queue:
        n = queue.popleft()
        for p in pred.get(n, ()):
            if p in forward and p not in reachable:
                reachable.add(p)
                queue.append(p)
    return reachable


def _verify_sese(
    entry: str,
    exit_node: str,
    contained: set[str],
    succ: dict[str, set[str]],
    pred: dict[str, set[str]],
) -> bool:
    """Verify that `contained` is a SESE region (entry, exit_node).

    The region is SESE iff:
      - Every in-edge of any node in `contained` (except entry) comes
        from inside `contained`.
      - Every out-edge of any node in `contained` (except exit_node)
        goes inside `contained`.
    """
    for v in contained:
        if v != entry:
            for p in pred[v]:
                if p not in contained:
                    return False
        if v != exit_node:
            for s in succ[v]:
                if s not in contained:
                    return False
    return True
