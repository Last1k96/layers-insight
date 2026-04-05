"""Tests for the pure-Python DAG layout algorithm."""
from __future__ import annotations

import time

import pytest

from backend.utils.dag_layout import compute_dag_layout


def _node(nid: str, width: float = 120, height: float = 32) -> dict:
    return {"id": nid, "width": width, "height": height}


def _edge(src: str, tgt: str, source_port: int = 0) -> dict:
    return {"source": src, "target": tgt, "source_port": source_port}


class TestEmptyAndSingle:
    def test_empty_graph(self):
        result = compute_dag_layout([], [])
        assert result == {"nodes": {}, "edges": {}}

    def test_single_node(self):
        result = compute_dag_layout([_node("a")], [])
        assert "a" in result["nodes"]
        assert result["nodes"]["a"]["x"] >= 0
        assert result["nodes"]["a"]["y"] == 0


class TestLinearChain:
    def test_chain_layers(self):
        """Nodes in a chain should be placed in successive layers."""
        nodes = [_node(f"n{i}") for i in range(5)]
        edges = [_edge(f"n{i}", f"n{i+1}") for i in range(4)]
        result = compute_dag_layout(nodes, edges)

        positions = result["nodes"]
        # Each node should be at a greater Y than the previous
        for i in range(4):
            assert positions[f"n{i}"]["y"] < positions[f"n{i+1}"]["y"]

    def test_chain_x_alignment(self):
        """Nodes in a chain should be roughly aligned on X."""
        nodes = [_node("a"), _node("b"), _node("c")]
        edges = [_edge("a", "b"), _edge("b", "c")]
        result = compute_dag_layout(nodes, edges)

        positions = result["nodes"]
        # All centers should be close (within half a node width)
        centers = [positions[n]["x"] + 60 for n in ["a", "b", "c"]]  # 120/2
        assert max(centers) - min(centers) < 120

    def test_chain_edge_waypoints(self):
        """Each edge should have at least start and end waypoints."""
        nodes = [_node("a"), _node("b")]
        edges = [_edge("a", "b")]
        result = compute_dag_layout(nodes, edges)

        assert "e0" in result["edges"]
        wps = result["edges"]["e0"]["waypoints"]
        assert len(wps) >= 2
        # Start Y should be at bottom of node a (y + height)
        assert wps[0]["y"] == 32  # layer 0 y=0 + height=32
        # End Y should be at top of node b
        assert wps[-1]["y"] == 52  # layer 1 y = 32 + 20 = 52


class TestDiamondDAG:
    def test_diamond_layers(self):
        """A->B, A->C, B->D, C->D: B and C share a layer."""
        nodes = [_node("a"), _node("b"), _node("c"), _node("d")]
        edges = [_edge("a", "b"), _edge("a", "c"), _edge("b", "d"), _edge("c", "d")]
        result = compute_dag_layout(nodes, edges)

        positions = result["nodes"]
        # A at top, D at bottom
        assert positions["a"]["y"] < positions["b"]["y"]
        assert positions["b"]["y"] == positions["c"]["y"]  # same layer
        assert positions["b"]["y"] < positions["d"]["y"]

    def test_diamond_no_overlap(self):
        """B and C should not overlap horizontally."""
        nodes = [_node("a"), _node("b"), _node("c"), _node("d")]
        edges = [_edge("a", "b"), _edge("a", "c"), _edge("b", "d"), _edge("c", "d")]
        result = compute_dag_layout(nodes, edges)

        pos = result["nodes"]
        b_right = pos["b"]["x"] + 120
        c_left = pos["c"]["x"]
        if pos["b"]["x"] < pos["c"]["x"]:
            assert b_right + 20 <= c_left + 0.01  # 20px spacing
        else:
            c_right = pos["c"]["x"] + 120
            assert c_right + 20 <= pos["b"]["x"] + 0.01


class TestLongEdges:
    def test_long_edge_intermediate_waypoints(self):
        """A->B->C and A->C: the A->C edge should have intermediate waypoints."""
        nodes = [_node("a"), _node("b"), _node("c")]
        edges = [_edge("a", "b"), _edge("b", "c"), _edge("a", "c")]
        result = compute_dag_layout(nodes, edges)

        # A->C is edge index 2 (e2), spans 2 layers
        wps = result["edges"]["e2"]["waypoints"]
        assert len(wps) >= 3  # start + intermediate + end


class TestMultiOutputPorts:
    def test_port_spread(self):
        """A node with 2 output ports should spread start X of its edges."""
        nodes = [_node("src", width=200), _node("t0"), _node("t1")]
        edges = [_edge("src", "t0", source_port=0), _edge("src", "t1", source_port=1)]
        result = compute_dag_layout(nodes, edges)

        wp0 = result["edges"]["e0"]["waypoints"][0]
        wp1 = result["edges"]["e1"]["waypoints"][0]
        # Different start X for different ports
        assert wp0["x"] != wp1["x"]
        # Same start Y (both from bottom of src)
        assert wp0["y"] == wp1["y"]


class TestWideLayer:
    def test_no_overlap(self):
        """10 nodes in same layer should not overlap."""
        nodes = [_node("root")] + [_node(f"n{i}", width=100) for i in range(10)]
        edges = [_edge("root", f"n{i}") for i in range(10)]
        result = compute_dag_layout(nodes, edges)

        pos = result["nodes"]
        # All n0-n9 should be at the same Y (layer 1)
        ys = [pos[f"n{i}"]["y"] for i in range(10)]
        assert len(set(ys)) == 1

        # Check no horizontal overlap
        layer_nodes = sorted(
            [(pos[f"n{i}"]["x"], f"n{i}") for i in range(10)],
            key=lambda t: t[0],
        )
        for j in range(len(layer_nodes) - 1):
            x1, _ = layer_nodes[j]
            x2, _ = layer_nodes[j + 1]
            assert x2 >= x1 + 100 + 20 - 0.01  # width + spacing


class TestOutputFormat:
    def test_format_matches_apply_layout(self):
        """Output format must match what apply_layout() expects."""
        nodes = [_node("a"), _node("b")]
        edges = [_edge("a", "b")]
        result = compute_dag_layout(nodes, edges)

        assert "nodes" in result
        assert "edges" in result

        # Node positions
        assert "a" in result["nodes"]
        assert "x" in result["nodes"]["a"]
        assert "y" in result["nodes"]["a"]

        # Edge waypoints keyed as "e{index}"
        assert "e0" in result["edges"]
        assert "waypoints" in result["edges"]["e0"]
        wps = result["edges"]["e0"]["waypoints"]
        assert isinstance(wps, list)
        assert "x" in wps[0]
        assert "y" in wps[0]


class TestStraightEdges:
    def test_chain_straight_edges(self):
        """Brandes-Köpf should produce straight (vertically aligned) edges for chains."""
        nodes = [_node(f"n{i}") for i in range(5)]
        edges = [_edge(f"n{i}", f"n{i+1}") for i in range(4)]
        result = compute_dag_layout(nodes, edges)

        pos = result["nodes"]
        # All nodes in a simple chain should have the same X center
        centers = [pos[f"n{i}"]["x"] + 60 for i in range(5)]  # 120/2 = 60
        # With Brandes-Köpf, they should be exactly aligned
        for c in centers[1:]:
            assert abs(c - centers[0]) < 1.0, f"Chain nodes not vertically aligned: {centers}"

    def test_branch_parent_centered(self):
        """A node with two children should be centered between them."""
        nodes = [_node("p"), _node("a"), _node("b")]
        edges = [_edge("p", "a"), _edge("p", "b")]
        result = compute_dag_layout(nodes, edges)

        pos = result["nodes"]
        p_center = pos["p"]["x"] + 60
        a_center = pos["a"]["x"] + 60
        b_center = pos["b"]["x"] + 60
        children_mid = (a_center + b_center) / 2
        # Parent should be roughly centered over children
        assert abs(p_center - children_mid) < 30


class TestLongEdgeStraightening:
    def test_parallel_long_edges_straight_and_spaced(self):
        """Parallel long edges from same source should be straight and spaced apart."""
        # src -> mid -> t0, src -> mid -> t1, src -> t0 (long), src -> t1 (long)
        nodes = [_node("src"), _node("mid"), _node("t0"), _node("t1")]
        edges = [
            _edge("src", "mid"),
            _edge("mid", "t0"),
            _edge("mid", "t1"),
            _edge("src", "t0"),  # long edge, span=2
            _edge("src", "t1"),  # long edge, span=2
        ]
        result = compute_dag_layout(nodes, edges)
        # e3 and e4 are long edges with intermediate waypoints
        wp3 = result["edges"]["e3"]["waypoints"]
        wp4 = result["edges"]["e4"]["waypoints"]
        assert len(wp3) >= 3
        assert len(wp4) >= 3
        # Intermediate dummies should be distinguishable (not collapsed to same x)
        mid3 = wp3[1]["x"]
        mid4 = wp4[1]["x"]
        assert mid3 != mid4

    def test_single_long_edge_has_waypoints(self):
        """A single long edge should have intermediate waypoints."""
        nodes = [_node("a"), _node("b"), _node("c")]
        edges = [_edge("a", "b"), _edge("b", "c"), _edge("a", "c")]
        result = compute_dag_layout(nodes, edges)
        wps = result["edges"]["e2"]["waypoints"]
        assert len(wps) >= 3

    def test_bypass_edge_hugs_boundary(self):
        """A long edge bypassing a subgraph should route alongside, not through it."""
        # Create a subgraph: a -> b -> c -> d, with a -> d bypassing b and c
        nodes = [_node("a"), _node("b"), _node("c"), _node("d")]
        edges = [_edge("a", "b"), _edge("b", "c"), _edge("c", "d"), _edge("a", "d")]
        result = compute_dag_layout(nodes, edges)
        # e3 (a->d) is the bypass edge spanning 3 layers
        wp = result["edges"]["e3"]["waypoints"]
        assert len(wp) >= 4  # start + 2 intermediate + end

        # The intermediate waypoints should be outside the real nodes' x-range
        # (i.e., not overlapping b or c)
        pos = result["nodes"]
        for midwp in wp[1:-1]:
            for nid in ["b", "c"]:
                n_left = pos[nid]["x"]
                n_right = pos[nid]["x"] + 120
                assert midwp["x"] <= n_left - 10 or midwp["x"] >= n_right + 10, \
                    f"waypoint x={midwp['x']:.0f} overlaps node {nid} [{n_left:.0f}, {n_right:.0f}]"


class TestTargetPortSpreading:
    def test_target_ports_spread(self):
        """Edges to different target ports should arrive at different X."""
        nodes = [_node("a"), _node("b"), _node("c", width=200)]
        edges = [
            {"source": "a", "target": "c", "source_port": 0, "target_port": 0},
            {"source": "b", "target": "c", "source_port": 0, "target_port": 1},
        ]
        result = compute_dag_layout(nodes, edges)
        wp0_end = result["edges"]["e0"]["waypoints"][-1]["x"]
        wp1_end = result["edges"]["e1"]["waypoints"][-1]["x"]
        assert wp0_end != wp1_end

    def test_single_target_port_centered(self):
        """Single incoming edge should arrive at center."""
        nodes = [_node("a"), _node("b", width=200)]
        edges = [_edge("a", "b")]
        result = compute_dag_layout(nodes, edges)
        wps = result["edges"]["e0"]["waypoints"]
        # End x should be at center of b
        b_x = result["nodes"]["b"]["x"]
        assert abs(wps[-1]["x"] - (b_x + 100)) < 1.0  # 200/2 = 100


class TestFanOutSpreading:
    def test_fan_out_same_port(self):
        """Multiple edges from same source+port should start at different X."""
        nodes = [_node("src", width=200), _node("t0"), _node("t1"), _node("t2")]
        edges = [_edge("src", "t0"), _edge("src", "t1"), _edge("src", "t2")]
        result = compute_dag_layout(nodes, edges)
        starts = [result["edges"][f"e{i}"]["waypoints"][0]["x"] for i in range(3)]
        # All three should be at different X positions
        assert len(set(starts)) == 3
        # They should be within the source node's width
        src_x = result["nodes"]["src"]["x"]
        for sx in starts:
            assert src_x < sx < src_x + 200


class TestPerformance:
    def test_500_nodes_under_1s(self):
        """Layout of 500 nodes should complete in under 1 second."""
        # Build a synthetic DAG: 50 layers x 10 nodes, with edges between layers
        nodes = []
        edges = []
        for layer in range(50):
            for i in range(10):
                nid = f"L{layer}_n{i}"
                nodes.append(_node(nid, width=120, height=32))
                if layer > 0:
                    # Connect to 2 random nodes in previous layer
                    edges.append(_edge(f"L{layer-1}_n{i}", nid))
                    edges.append(_edge(f"L{layer-1}_n{(i+3) % 10}", nid))

        t0 = time.perf_counter()
        result = compute_dag_layout(nodes, edges)
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0, f"Layout took {elapsed:.3f}s, expected <1s"
        assert len(result["nodes"]) == 500
        assert len(result["edges"]) == 980  # 49 layers * 10 nodes * 2 edges
