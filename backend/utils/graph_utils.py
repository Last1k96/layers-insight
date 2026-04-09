"""Graph traversal utilities operating on GraphData (no OpenVINO dependency)."""
from __future__ import annotations

from collections import defaultdict

from backend.schemas.graph import GraphData, GraphEdge


def build_reverse_adj(edges: list[GraphEdge]) -> dict[str, set[str]]:
    """Build reverse adjacency: target -> set of source nodes."""
    rev: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        rev[e.target].add(e.source)
    return dict(rev)


def get_ancestors_in_set(
    node_id: str,
    candidates: set[str],
    reverse_adj: dict[str, set[str]],
) -> set[str]:
    """Return node_id and all its ancestors that are in the candidate set.

    Traversal only follows edges through nodes in `candidates`,
    so removed candidates act as barriers.
    """
    result: set[str] = set()
    stack = [node_id]
    while stack:
        n = stack.pop()
        if n not in candidates or n in result:
            continue
        result.add(n)
        for parent in reverse_adj.get(n, ()):
            stack.append(parent)
    return result


def find_output_nodes(graph_data: GraphData) -> list[tuple[str, str]]:
    """Return [(result_node_id, predecessor_id), ...] for each Result node.

    The predecessor is the computation node feeding into the Result.
    """
    # Build edge lookup: target -> list of sources
    preds: dict[str, list[str]] = defaultdict(list)
    for e in graph_data.edges:
        preds[e.target].append(e.source)

    outputs: list[tuple[str, str]] = []
    for node in graph_data.nodes:
        if node.type == "Result":
            pred_list = preds.get(node.id, [])
            if pred_list:
                outputs.append((node.id, pred_list[0]))
    return outputs


def find_best_bisect_point(
    candidates: set[str],
    reverse_adj: dict[str, set[str]],
    topo_order: list[str],
) -> str:
    """Pick the candidate whose ancestor-count is closest to len(candidates)/2.

    Falls back to topological-order median for large candidate sets (>500)
    to avoid O(n²) ancestor counting.
    """
    if len(candidates) == 1:
        return next(iter(candidates))

    ordered = [n for n in topo_order if n in candidates]

    if len(candidates) > 500:
        return ordered[len(ordered) // 2]

    target = len(candidates) / 2.0
    best_node = ordered[len(ordered) // 2]  # fallback
    best_score = float("inf")

    for node in ordered:
        count = len(get_ancestors_in_set(node, candidates, reverse_adj))
        score = abs(count - target)
        if score < best_score or (score == best_score and node != best_node):
            best_score = score
            best_node = node

    return best_node
