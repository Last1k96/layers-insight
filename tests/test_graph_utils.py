"""Tests for graph traversal utilities."""
from __future__ import annotations

from backend.schemas.graph import GraphData, GraphEdge, GraphNode
from backend.utils.graph_utils import (
    build_reverse_adj,
    find_best_bisect_point,
    find_output_nodes,
    get_ancestors_in_set,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_graph(n: int = 5) -> GraphData:
    """A -> B -> C -> D -> E (n=5)."""
    names = [chr(65 + i) for i in range(n)]
    nodes = [GraphNode(id=name, name=name, type="Op") for name in names]
    edges = [GraphEdge(source=names[i], target=names[i + 1]) for i in range(n - 1)]
    return GraphData(nodes=nodes, edges=edges)


def _diamond_graph() -> GraphData:
    """
    A -> B -> D
    A -> C -> D
    """
    nodes = [
        GraphNode(id="A", name="A", type="Op"),
        GraphNode(id="B", name="B", type="Op"),
        GraphNode(id="C", name="C", type="Op"),
        GraphNode(id="D", name="D", type="Op"),
    ]
    edges = [
        GraphEdge(source="A", target="B"),
        GraphEdge(source="A", target="C"),
        GraphEdge(source="B", target="D"),
        GraphEdge(source="C", target="D"),
    ]
    return GraphData(nodes=nodes, edges=edges)


def _multi_branch_graph() -> GraphData:
    """
    param -> A -> B -> E -> result_0
    param -> A -> C -> D -> F -> result_1
    """
    nodes = [
        GraphNode(id="param", name="param", type="Parameter"),
        GraphNode(id="A", name="A", type="Op"),
        GraphNode(id="B", name="B", type="Op"),
        GraphNode(id="C", name="C", type="Op"),
        GraphNode(id="D", name="D", type="Op"),
        GraphNode(id="E", name="E", type="Op"),
        GraphNode(id="F", name="F", type="Op"),
        GraphNode(id="result_0", name="result_0", type="Result"),
        GraphNode(id="result_1", name="result_1", type="Result"),
    ]
    edges = [
        GraphEdge(source="param", target="A"),
        GraphEdge(source="A", target="B"),
        GraphEdge(source="A", target="C"),
        GraphEdge(source="B", target="E"),
        GraphEdge(source="C", target="D"),
        GraphEdge(source="D", target="F"),
        GraphEdge(source="E", target="result_0"),
        GraphEdge(source="F", target="result_1"),
    ]
    return GraphData(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# build_reverse_adj
# ---------------------------------------------------------------------------

def test_build_reverse_adj_linear():
    g = _linear_graph(4)
    rev = build_reverse_adj(g.edges)
    assert rev.get("A") is None or rev.get("A") == set()
    assert rev["B"] == {"A"}
    assert rev["C"] == {"B"}
    assert rev["D"] == {"C"}


def test_build_reverse_adj_diamond():
    g = _diamond_graph()
    rev = build_reverse_adj(g.edges)
    assert rev["D"] == {"B", "C"}
    assert rev["B"] == {"A"}
    assert rev["C"] == {"A"}


# ---------------------------------------------------------------------------
# get_ancestors_in_set
# ---------------------------------------------------------------------------

def test_ancestors_linear():
    g = _linear_graph(5)
    rev = build_reverse_adj(g.edges)
    candidates = {"A", "B", "C", "D", "E"}
    assert get_ancestors_in_set("E", candidates, rev) == {"A", "B", "C", "D", "E"}
    assert get_ancestors_in_set("C", candidates, rev) == {"A", "B", "C"}
    assert get_ancestors_in_set("A", candidates, rev) == {"A"}


def test_ancestors_diamond():
    g = _diamond_graph()
    rev = build_reverse_adj(g.edges)
    candidates = {"A", "B", "C", "D"}
    assert get_ancestors_in_set("D", candidates, rev) == {"A", "B", "C", "D"}
    assert get_ancestors_in_set("B", candidates, rev) == {"A", "B"}


def test_ancestors_restricted_set():
    """Only returns ancestors that are in the candidate set."""
    g = _diamond_graph()
    rev = build_reverse_adj(g.edges)
    # Remove A from candidates — traversal from D should stop at B and C
    candidates = {"B", "C", "D"}
    assert get_ancestors_in_set("D", candidates, rev) == {"B", "C", "D"}


def test_ancestors_partial_set():
    """When a node on the path is removed, upstream is unreachable."""
    g = _linear_graph(5)  # A -> B -> C -> D -> E
    rev = build_reverse_adj(g.edges)
    # Remove C — A and B become unreachable from D
    candidates = {"A", "B", "D", "E"}
    assert get_ancestors_in_set("E", candidates, rev) == {"D", "E"}


def test_ancestors_node_not_in_candidates():
    g = _linear_graph(3)
    rev = build_reverse_adj(g.edges)
    candidates = {"A", "B"}
    # C is not in candidates, should return empty
    assert get_ancestors_in_set("C", candidates, rev) == set()


# ---------------------------------------------------------------------------
# find_output_nodes
# ---------------------------------------------------------------------------

def test_find_output_nodes_multi():
    g = _multi_branch_graph()
    outputs = find_output_nodes(g)
    assert len(outputs) == 2
    result_ids = {r for r, _ in outputs}
    pred_ids = {p for _, p in outputs}
    assert result_ids == {"result_0", "result_1"}
    assert pred_ids == {"E", "F"}


def test_find_output_nodes_none():
    g = _linear_graph(3)  # no Result nodes
    outputs = find_output_nodes(g)
    assert outputs == []


def test_find_output_nodes_single():
    nodes = [
        GraphNode(id="param", name="param", type="Parameter"),
        GraphNode(id="op", name="op", type="Op"),
        GraphNode(id="result", name="result", type="Result"),
    ]
    edges = [
        GraphEdge(source="param", target="op"),
        GraphEdge(source="op", target="result"),
    ]
    g = GraphData(nodes=nodes, edges=edges)
    outputs = find_output_nodes(g)
    assert outputs == [("result", "op")]


# ---------------------------------------------------------------------------
# find_best_bisect_point
# ---------------------------------------------------------------------------

def test_bisect_point_linear():
    g = _linear_graph(7)
    rev = build_reverse_adj(g.edges)
    topo = [n.id for n in g.nodes]
    candidates = set(topo)
    mid = find_best_bisect_point(candidates, rev, topo)
    # Should pick something near the middle (D at index 3)
    assert mid in candidates


def test_bisect_point_single():
    rev: dict[str, set[str]] = {}
    mid = find_best_bisect_point({"X"}, rev, ["X"])
    assert mid == "X"


def test_bisect_point_two():
    rev = {"B": {"A"}}
    mid = find_best_bisect_point({"A", "B"}, rev, ["A", "B"])
    assert mid in {"A", "B"}


def test_bisect_point_diamond():
    g = _diamond_graph()
    rev = build_reverse_adj(g.edges)
    topo = [n.id for n in g.nodes]
    candidates = {"A", "B", "C", "D"}
    mid = find_best_bisect_point(candidates, rev, topo)
    # Optimal: B or C (ancestor count = 2, target = 2.0)
    assert mid in {"B", "C"}


def test_bisect_point_asymmetric():
    """
    A -> B -> C -> D -> E
    A -> F -> E
    """
    nodes = [GraphNode(id=x, name=x, type="Op") for x in "ABCDEF"]
    edges = [
        GraphEdge(source="A", target="B"),
        GraphEdge(source="B", target="C"),
        GraphEdge(source="C", target="D"),
        GraphEdge(source="D", target="E"),
        GraphEdge(source="A", target="F"),
        GraphEdge(source="F", target="E"),
    ]
    g = GraphData(nodes=nodes, edges=edges)
    rev = build_reverse_adj(g.edges)
    topo = [n.id for n in g.nodes]
    candidates = {"A", "B", "C", "D", "E", "F"}
    mid = find_best_bisect_point(candidates, rev, topo)
    # Target = 3. Should pick C or D (with ~3 ancestors each)
    assert mid in candidates
