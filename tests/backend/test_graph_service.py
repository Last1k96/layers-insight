"""Tests for graph service."""
import pytest
from unittest.mock import MagicMock, PropertyMock

from backend.services.graph_service import extract_graph, search_nodes, _fallback_layout
from backend.schemas.graph import GraphData, GraphNode, GraphEdge


def _make_mock_op(name, op_type, outputs=None, inputs=None, output_shape=None):
    """Create a mock OpenVINO operation."""
    op = MagicMock()
    op.get_friendly_name.return_value = name
    op.get_type_name.return_value = op_type

    # Output
    out_size = 1 if outputs is None else len(outputs)
    op.get_output_size.return_value = out_size

    if output_shape:
        pshape = MagicMock()
        pshape.is_static = True
        pshape.get_shape.return_value = output_shape
        output_mock = MagicMock()
        output_mock.get_partial_shape.return_value = pshape
        output_mock.get_element_type.return_value = "f32"
        op.output.return_value = output_mock
    else:
        pshape = MagicMock()
        pshape.is_static = True
        pshape.get_shape.return_value = [1]
        output_mock = MagicMock()
        output_mock.get_partial_shape.return_value = pshape
        output_mock.get_element_type.return_value = "f32"
        op.output.return_value = output_mock

    # Input
    in_size = 0 if inputs is None else len(inputs)
    op.get_input_size.return_value = in_size

    if inputs:
        def get_input(idx):
            inp = MagicMock()
            source_out = MagicMock()
            source_node = MagicMock()
            source_node.get_friendly_name.return_value = inputs[idx]
            source_out.get_node.return_value = source_node
            source_out.get_index.return_value = 0
            inp.get_source_output.return_value = source_out
            return inp
        op.input = get_input

    op.get_attributes.return_value = {}

    return op


def _make_mock_model(ops):
    """Create a mock OV model."""
    model = MagicMock()
    model.get_ordered_ops.return_value = ops
    return model


class TestExtractGraph:
    def test_basic_extraction(self):
        ops = [
            _make_mock_op("input", "Parameter", output_shape=[1, 3, 224, 224]),
            _make_mock_op("conv1", "Convolution", inputs=["input"], output_shape=[1, 64, 112, 112]),
            _make_mock_op("relu1", "Relu", inputs=["conv1"], output_shape=[1, 64, 112, 112]),
            _make_mock_op("output", "Result", inputs=["relu1"]),
        ]
        model = _make_mock_model(ops)

        graph = extract_graph(model)

        assert len(graph.nodes) == 4
        assert len(graph.edges) == 3

    def test_node_categories(self):
        ops = [
            _make_mock_op("conv", "Convolution"),
            _make_mock_op("relu", "Relu"),
            _make_mock_op("add", "Add"),
            _make_mock_op("reshape", "Reshape"),
        ]
        model = _make_mock_model(ops)

        graph = extract_graph(model)
        categories = {n.name: n.category for n in graph.nodes}

        assert categories["conv"] == "Convolution"
        assert categories["relu"] == "Activation"
        assert categories["add"] == "Elementwise"
        assert categories["reshape"] == "DataMovement"

    def test_node_colors(self):
        ops = [
            _make_mock_op("conv", "Convolution"),
            _make_mock_op("unknown_op", "MyCustomOp"),
        ]
        model = _make_mock_model(ops)

        graph = extract_graph(model)
        colors = {n.name: n.color for n in graph.nodes}

        assert colors["conv"] == "#4A90D9"
        assert colors["unknown_op"] == "#78909C"  # Other/default

    def test_constants_filtered(self):
        """Constant nodes should be excluded from the graph."""
        ops = [
            _make_mock_op("input", "Parameter", output_shape=[1, 3, 224, 224]),
            _make_mock_op("weights", "Constant", output_shape=[64, 3, 3, 3]),
            _make_mock_op("conv1", "Convolution", inputs=["input", "weights"], output_shape=[1, 64, 112, 112]),
            _make_mock_op("output", "Result", inputs=["conv1"]),
        ]
        model = _make_mock_model(ops)
        graph = extract_graph(model)

        node_names = {n.name for n in graph.nodes}
        assert "weights" not in node_names
        assert "input" in node_names
        assert "conv1" in node_names
        # Only input->conv1 and conv1->output (weights->conv1 filtered)
        assert len(graph.edges) == 2

    def test_deduplication(self):
        op = _make_mock_op("same_name", "Relu")
        ops = [op, op]  # Same op appears twice
        model = _make_mock_model(ops)

        graph = extract_graph(model)
        assert len(graph.nodes) == 1


class TestSearchNodes:
    def test_search_by_name(self):
        nodes = [
            GraphNode(id="conv1", name="conv1", type="Convolution"),
            GraphNode(id="relu1", name="relu1", type="Relu"),
            GraphNode(id="conv2", name="conv2", type="Convolution"),
        ]
        graph = GraphData(nodes=nodes, edges=[])

        results = search_nodes(graph, "conv")
        assert len(results) == 2

    def test_search_by_type(self):
        nodes = [
            GraphNode(id="r1", name="relu1", type="Relu"),
            GraphNode(id="r2", name="my_relu", type="Relu"),
            GraphNode(id="c1", name="conv1", type="Convolution"),
        ]
        graph = GraphData(nodes=nodes, edges=[])

        results = search_nodes(graph, "relu")
        assert len(results) == 2

    def test_case_insensitive(self):
        nodes = [
            GraphNode(id="c1", name="Conv1", type="Convolution"),
        ]
        graph = GraphData(nodes=nodes, edges=[])

        results = search_nodes(graph, "CONV")
        assert len(results) == 1


class TestFallbackLayout:
    def test_linear_graph(self):
        nodes = [
            GraphNode(id="a", name="a", type="T"),
            GraphNode(id="b", name="b", type="T"),
            GraphNode(id="c", name="c", type="T"),
        ]
        edges = [
            GraphEdge(source="a", target="b"),
            GraphEdge(source="b", target="c"),
        ]
        graph = GraphData(nodes=nodes, edges=edges)

        result = _fallback_layout(graph)
        positions = result["nodes"]

        assert positions["a"]["y"] < positions["b"]["y"]
        assert positions["b"]["y"] < positions["c"]["y"]

    def test_empty_graph(self):
        graph = GraphData(nodes=[], edges=[])
        result = _fallback_layout(graph)
        assert result == {"nodes": {}, "edges": {}}
