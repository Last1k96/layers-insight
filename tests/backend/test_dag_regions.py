"""Tests for SESE region detection."""
from __future__ import annotations

from backend.utils.dag_regions import (
    Region,
    build_node_region_map,
    find_sese_regions,
)


def _edge(s: str, t: str) -> dict:
    return {"source": s, "target": t}


def _all_descendant_regions(root: Region) -> list[Region]:
    out: list[Region] = []
    stack = [root]
    while stack:
        r = stack.pop()
        out.append(r)
        stack.extend(r.children)
    return out


class TestEmptyAndTrivial:
    def test_empty_graph(self):
        root = find_sese_regions([], [])
        assert root.nodes == set()
        assert root.children == []

    def test_single_node(self):
        root = find_sese_regions(["a"], [])
        assert root.nodes == {"a"}
        assert root.children == []

    def test_linear_chain(self):
        nodes = ["a", "b", "c", "d"]
        edges = [_edge("a", "b"), _edge("b", "c"), _edge("c", "d")]
        root = find_sese_regions(nodes, edges)
        # No merges → no nested regions, only the root.
        assert root.nodes == set(nodes)
        assert root.children == []


class TestResidualBlock:
    def test_residual_block_detected(self):
        """a -> b -> c -> d with skip a -> d.

        The (a, d) pair is the only merge → root region with no
        children (since the contained set is just {a,b,c,d} which is
        the whole graph).
        """
        nodes = ["a", "b", "c", "d"]
        edges = [
            _edge("a", "b"), _edge("b", "c"), _edge("c", "d"),
            _edge("a", "d"),
        ]
        root = find_sese_regions(nodes, edges)
        assert root.nodes == set(nodes)

    def test_residual_block_inside_chain(self):
        """src -> a -> b -> c -> d -> sink with skip a -> d.

        The block (a, d) should be detected as a child of the root.
        """
        nodes = ["src", "a", "b", "c", "d", "sink"]
        edges = [
            _edge("src", "a"),
            _edge("a", "b"), _edge("b", "c"), _edge("c", "d"),
            _edge("a", "d"),
            _edge("d", "sink"),
        ]
        root = find_sese_regions(nodes, edges)
        all_regions = _all_descendant_regions(root)
        # We expect the root and one block region.
        block_regions = [r for r in all_regions if r is not root]
        assert len(block_regions) >= 1
        block = block_regions[0]
        assert block.entry == "a"
        assert block.exit == "d"
        assert {"a", "b", "c", "d"} <= block.nodes
        assert "src" not in block.nodes
        assert "sink" not in block.nodes


class TestNestedBlocks:
    def test_two_stacked_residual_blocks(self):
        """src -> a -> ... block1 ... -> mid -> ... block2 ... -> sink.

        Each block has a skip from its entry to its exit.
        """
        nodes = ["src", "a", "b", "c", "mid", "d", "e", "f", "sink"]
        edges = [
            _edge("src", "a"),
            # Block 1: a -> b -> c -> mid, skip a -> mid
            _edge("a", "b"), _edge("b", "c"), _edge("c", "mid"),
            _edge("a", "mid"),
            # Block 2: mid -> d -> e -> f, skip mid -> f
            _edge("mid", "d"), _edge("d", "e"), _edge("e", "f"),
            _edge("mid", "f"),
            _edge("f", "sink"),
        ]
        root = find_sese_regions(nodes, edges)
        all_regions = _all_descendant_regions(root)
        block_regions = [r for r in all_regions if r is not root]
        # Two blocks expected.
        block_pairs = {(r.entry, r.exit) for r in block_regions}
        assert ("a", "mid") in block_pairs
        assert ("mid", "f") in block_pairs


class TestNodeRegionMap:
    def test_map_returns_smallest_enclosing_region(self):
        nodes = ["src", "a", "b", "c", "d", "sink"]
        edges = [
            _edge("src", "a"),
            _edge("a", "b"), _edge("b", "c"), _edge("c", "d"),
            _edge("a", "d"),
            _edge("d", "sink"),
        ]
        root = find_sese_regions(nodes, edges)
        mapping = build_node_region_map(root)
        # b and c should be inside the (a, d) block.
        block_b = mapping["b"]
        block_c = mapping["c"]
        assert block_b is block_c
        assert block_b is not root
        assert block_b.entry == "a"
        assert block_b.exit == "d"
        # src and sink should be in the root region (no nested block
        # contains them).
        assert mapping["src"] is root
        assert mapping["sink"] is root


class TestVerifySeseExclusion:
    def test_overlapping_skips_are_not_falsely_nested(self):
        """Two skips that share a target shouldn't both be valid SESE
        candidates if one of them leaks an in-edge.
        """
        nodes = ["a", "b", "c", "d", "e"]
        edges = [
            _edge("a", "b"),
            _edge("b", "c"),
            _edge("c", "d"),
            _edge("a", "d"),  # skip
            _edge("b", "d"),  # additional skip
            _edge("d", "e"),
        ]
        root = find_sese_regions(nodes, edges)
        # The graph is still SESE-detectable: (a, d) is the merge.
        all_regions = _all_descendant_regions(root)
        # No region should claim {a, b, c} since b also has an
        # external out-edge (b->d).
        for r in all_regions:
            if r.entry == "a" and r.exit == "c":
                raise AssertionError("invalid SESE region detected")
