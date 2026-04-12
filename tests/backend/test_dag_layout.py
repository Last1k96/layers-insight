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

        # The intermediate waypoints should not overlap b or c (2D bounding box)
        pos = result["nodes"]
        for midwp in wp[1:-1]:
            for nid in ["b", "c"]:
                n_left = pos[nid]["x"]
                n_right = n_left + 120
                n_top = pos[nid]["y"]
                n_bottom = n_top + 32
                x_in = n_left - 10 < midwp["x"] < n_right + 10
                y_in = n_top - 10 < midwp["y"] < n_bottom + 10
                assert not (x_in and y_in), \
                    f"waypoint ({midwp['x']:.0f},{midwp['y']:.0f}) overlaps node {nid} [{n_left:.0f},{n_right:.0f}]x[{n_top:.0f},{n_bottom:.0f}]"

    def test_skip_inside_small_sese_region_stays_local(self):
        """A skip edge whose enclosing SESE region is small should not
        be pushed into a far corridor.

        Regression for the BatchNormalization_885 → Resize_914 case in
        human-pose-estimation-0007: a small "resize-tail" subgraph
        contains both endpoints of the skip and a handful of dynamic-
        shape helper ops. Pre-fix, the obstacle field was global and
        the corridor branch walked the route 3000+ px sideways. Post-
        fix, the obstacle filter restricts to the 6-node region and the
        skip stays within tens of pixels of its endpoints.
        """
        # A small SESE block: src -> skip target with a side chain that
        # also reads from src and feeds the target.
        nodes = [
            _node("upstream"),
            _node("src"),
            _node("h1"),
            _node("h2"),
            _node("tgt"),
            _node("downstream"),
            # Decoy: a wide column of unrelated nodes far to the side
            # at the same y values as the skip's intermediate layers.
            _node("d0", width=200),
            _node("d1", width=200),
            _node("d2", width=200),
            _node("d3", width=200),
            _node("d4", width=200),
            _node("d5", width=200),
        ]
        edges = [
            _edge("upstream", "src"),
            _edge("src", "h1"),
            _edge("h1", "h2"),
            _edge("h2", "tgt"),
            _edge("src", "tgt"),  # the skip we care about
            _edge("tgt", "downstream"),
            # Independent decoy chain
            _edge("d0", "d1"),
            _edge("d1", "d2"),
            _edge("d2", "d3"),
            _edge("d3", "d4"),
            _edge("d4", "d5"),
        ]
        result = compute_dag_layout(nodes, edges)
        # Find the src -> tgt edge (the skip)
        skip_idx = None
        for i, e in enumerate(edges):
            if e["source"] == "src" and e["target"] == "tgt":
                skip_idx = i
                break
        assert skip_idx is not None
        skip_wps = result["edges"][f"e{skip_idx}"]["waypoints"]
        src_cx = result["nodes"]["src"]["x"] + 60
        tgt_cx = result["nodes"]["tgt"]["x"] + 60
        for wp in skip_wps:
            min_dev = min(abs(wp["x"] - src_cx), abs(wp["x"] - tgt_cx))
            assert min_dev < 200, (
                f"skip waypoint at x={wp['x']:.0f} is more than 200 px "
                f"from BOTH src={src_cx:.0f} and tgt={tgt_cx:.0f}"
            )

    def test_long_skip_has_few_local_direction_changes(self):
        """A long skip edge should not weave back and forth.

        The renderer's B-spline traces every waypoint, so a per-layer
        zigzag of nudged dummies becomes a visibly wavy line. The
        layout collapses zigzags to a single column when feasible and
        otherwise demotes to a corridor route. Either way, the
        resulting waypoint sequence should have at most one local
        change of horizontal direction.
        """
        # Backbone n0..n9 plus a parallel chain whose nodes shift across
        # multiple columns at different layers, forcing a per-layer
        # nudge that would zigzag without smoothing.
        nodes = [_node(f"n{i}") for i in range(10)]
        edges = [_edge(f"n{i}", f"n{i+1}") for i in range(9)]
        for i in range(10):
            nodes.append(_node(f"r{i}", width=120 + (i % 3) * 30))
        edges += [_edge(f"r{i}", f"r{i+1}") for i in range(9)]
        edges.append(_edge("n1", "n9"))  # span-8 skip

        result = compute_dag_layout(nodes, edges)
        wps = result["edges"][f"e{len(edges) - 1}"]["waypoints"]

        # Count direction changes in horizontal motion (ignore tiny dx).
        eps = 0.5
        last_dir = 0
        changes = 0
        for i in range(len(wps) - 1):
            dx = wps[i + 1]["x"] - wps[i]["x"]
            if abs(dx) < eps:
                continue
            d = 1 if dx > 0 else -1
            if last_dir != 0 and d != last_dir:
                changes += 1
            last_dir = d
        assert changes <= 1, (
            f"long skip wove with {changes} direction changes; "
            f"waypoints: {[(round(w['x']), round(w['y'])) for w in wps]}"
        )

    def test_aligned_long_skip_hugs_backbone_not_far_corridor(self):
        """A column-aligned span-6 skip should hug its backbone via nudge,
        not detour to a distant corridor.

        Regression for human-pose-estimation-0007's Relu_265 → Add_280:
        each intermediate layer has a backbone neighbour that minimally
        overlaps the source x, plus an unrelated parallel chain. Pre-fix,
        the span-6 cutoff forced corridor mode and the edge swung 300+ px
        out to clear the parallel chain before snapping back. Post-fix,
        per-dummy nudge keeps it within ~50 px of the source column.
        """
        # Backbone n0 → ... → n7 with a span-6 skip from n1 to n7.
        nodes = [_node(f"n{i}") for i in range(8)]
        edges = [_edge(f"n{i}", f"n{i+1}") for i in range(7)]
        # Independent parallel chain that lands in the same intermediate
        # layers and nudges the backbone-aligned column off-centre.
        nodes += [_node(f"r{i}") for i in range(8)]
        edges += [_edge(f"r{i}", f"r{i+1}") for i in range(7)]
        # The skip edge under test (last edge added).
        edges.append(_edge("n1", "n7"))

        result = compute_dag_layout(nodes, edges)
        skip_wps = result["edges"][f"e{len(edges) - 1}"]["waypoints"]
        src_cx = result["nodes"]["n1"]["x"] + 60  # 120/2
        max_dev = max(abs(wp["x"] - src_cx) for wp in skip_wps)
        assert max_dev < 100, (
            f"span-6 skip detoured {max_dev:.0f} px from src_cx={src_cx:.0f} — "
            f"corridor over-detour bug. Waypoints: "
            f"{[(round(w['x']), round(w['y'])) for w in skip_wps]}"
        )


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


class TestTransformerFanOut:
    """Tests for transformer-like graph patterns.

    Reproduces CamemBERT layout issues:
    - Where_2 (Select) fans out to 12 SDPA nodes
    - Shape (ShapeOf) fans out to 12 Reshape nodes
    - Many single-edge skip connections (residual LayerNorm→Add)
    - Dozens of other single-edge source groups
    """

    @staticmethod
    def _build_transformer_graph():
        """Build a simplified CamemBERT-like graph.

        Returns (nodes, edges, key_names) where key_names is a dict
        with 'where', 'shape', 'sdpa_*', 'reshape_*', 'ln_*', 'add_*'.
        """
        nodes = []
        edges = []
        key = {}

        # Two early fan-out sources (Where_2 = Select mask, Shape = ShapeOf)
        nodes.append(_node("where", width=100))
        nodes.append(_node("shape", width=100))
        edges.append(_edge("where", "shape"))

        sdpa_names = []
        reshape_names = []
        ln_names = []
        add_names = []
        all_intermediate = []

        prev_main = "shape"
        for t in range(12):
            # Each transformer layer: 5 ops in the main chain,
            # plus a residual (skip) connection from ln → add.
            ln = f"ln_{t}"
            nodes.append(_node(ln))
            edges.append(_edge(prev_main, ln))
            ln_names.append(ln)

            prev = ln
            for step in range(4):
                nid = f"L{t}_op{step}"
                nodes.append(_node(nid))
                edges.append(_edge(prev, nid))
                prev = nid
                all_intermediate.append(nid)

            # SDPA consumes Q/K/V (from chain) + mask (from where)
            sdpa = f"sdpa_{t}"
            nodes.append(_node(sdpa, width=200))
            edges.append(_edge(prev, sdpa))
            edges.append(_edge("where", sdpa))
            sdpa_names.append(sdpa)

            # Reshape consumes SDPA output + shape (from shape)
            reshape = f"reshape_{t}"
            nodes.append(_node(reshape))
            edges.append(_edge(sdpa, reshape))
            edges.append(_edge("shape", reshape))
            reshape_names.append(reshape)

            # Residual add: ln → (skip over 5 ops + sdpa + reshape) → add
            add = f"add_{t}"
            nodes.append(_node(add))
            edges.append(_edge(reshape, add))
            edges.append(_edge(ln, add))  # residual skip
            add_names.append(add)

            prev_main = add

        key.update(sdpa=sdpa_names, reshape=reshape_names,
                   ln=ln_names, add=add_names, intermediate=all_intermediate)
        return nodes, edges, key

    def test_high_fanout_corridors_avoid_nodes(self):
        """Corridor waypoints from fan-out sources must not overlap real nodes
        in the same vertical region (same layer range)."""
        nodes, edges, key = self._build_transformer_graph()
        result = compute_dag_layout(nodes, edges)
        pos = result["nodes"]
        node_h = 32  # NODE_HEIGHT

        all_real = key["intermediate"] + key["ln"] + key["add"] + key["sdpa"] + key["reshape"]
        for source in ("where", "shape"):
            for eidx, e in enumerate(edges):
                if e["source"] != source:
                    continue
                wps = result["edges"][f"e{eidx}"]["waypoints"]
                tgt = e["target"]
                for wp in wps[1:-1]:
                    for nid in all_real:
                        if nid == tgt or nid == source:
                            continue
                        ny = pos[nid]["y"]
                        nw = [n for n in nodes if n["id"] == nid][0]["width"]
                        n_left = pos[nid]["x"]
                        n_right = n_left + nw
                        # 2D bounding-box overlap check
                        x_in = n_left - 5 < wp["x"] < n_right + 5
                        y_in = ny - 5 < wp["y"] < ny + node_h + 5
                        assert not (x_in and y_in), \
                            f"{source} corridor wp ({wp['x']:.0f},{wp['y']:.0f}) overlaps {nid} [{n_left:.0f},{n_right:.0f}]x[{ny:.0f},{ny+node_h:.0f}]"

    def test_competing_fanout_corridors_go_toward_targets(self):
        """Corridors should route toward their targets, not away."""
        nodes, edges, key = self._build_transformer_graph()
        result = compute_dag_layout(nodes, edges)
        pos = result["nodes"]

        def _corridor_and_targets(source_name):
            corr_xs = []
            tgt_xs = []
            for eidx, e in enumerate(edges):
                if e["source"] != source_name:
                    continue
                wps = result["edges"][f"e{eidx}"]["waypoints"]
                corr_xs.extend(wp["x"] for wp in wps[1:-1])
                tgt_xs.append(wps[-1]["x"])
            corr_med = sorted(corr_xs)[len(corr_xs) // 2] if corr_xs else None
            tgt_med = sorted(tgt_xs)[len(tgt_xs) // 2] if tgt_xs else None
            return corr_med, tgt_med

        for source in ("where", "shape"):
            src_cx = pos[source]["x"] + [n for n in nodes if n["id"] == source][0]["width"] / 2
            corr_med, tgt_med = _corridor_and_targets(source)
            assert corr_med is not None and tgt_med is not None
            # Corridor should be between source and targets (same side as targets)
            if tgt_med < src_cx:
                assert corr_med < src_cx, \
                    f"{source}: targets at {tgt_med:.0f} (left of src {src_cx:.0f}) but corridor at {corr_med:.0f} (right)"
            else:
                assert corr_med > src_cx, \
                    f"{source}: targets at {tgt_med:.0f} (right of src {src_cx:.0f}) but corridor at {corr_med:.0f} (left)"

    def test_residual_skip_not_pushed_to_fanout_corridors(self):
        """Residual connections (ln→add) must stay close to the graph, not
        get pushed to the same position as the large fan-out corridors."""
        nodes, edges, key = self._build_transformer_graph()
        result = compute_dag_layout(nodes, edges)
        pos = result["nodes"]

        # Find fan-out corridor positions
        fanout_xs = []
        for source in ("where", "shape"):
            for eidx, e in enumerate(edges):
                if e["source"] != source:
                    continue
                wps = result["edges"][f"e{eidx}"]["waypoints"]
                fanout_xs.extend(wp["x"] for wp in wps[1:-1])

        if not fanout_xs:
            return
        fanout_left = min(fanout_xs)
        fanout_right = max(fanout_xs)

        # Check each residual skip edge (ln → add)
        for eidx, e in enumerate(edges):
            if not (e["source"].startswith("ln_") and e["target"].startswith("add_")):
                continue
            wps = result["edges"][f"e{eidx}"]["waypoints"]
            skip_xs = [wp["x"] for wp in wps[1:-1]]
            if not skip_xs:
                continue

            src_x = pos[e["source"]]["x"]
            tgt_x = pos[e["target"]]["x"]
            local_center = (src_x + tgt_x) / 2 + 60  # approx node center

            for wx in skip_xs:
                # Skip corridor must be closer to its own nodes than to fan-out corridors
                dist_to_local = abs(wx - local_center)
                dist_to_fanout = min(abs(wx - fanout_left), abs(wx - fanout_right))
                assert dist_to_fanout > dist_to_local * 0.5, \
                    f"residual {e['source']}→{e['target']} wp x={wx:.0f} is at fanout corridor " \
                    f"(dist_local={dist_to_local:.0f}, dist_fanout={dist_to_fanout:.0f})"


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
