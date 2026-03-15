"""Tests for graph service."""
import pytest
from unittest.mock import MagicMock, PropertyMock

from backend.services.graph_service import extract_graph, search_nodes
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
    """Create a mock OV model.

    Wires up input mocks so that source_node references point back to the
    actual op mock objects, allowing shape/type lookups to work correctly.
    """
    op_by_name = {op.get_friendly_name(): op for op in ops}

    # Rewire input source nodes to point to actual op mocks
    for op in ops:
        if op.get_input_size() == 0:
            continue
        original_input = op.input
        def make_input_fn(orig_fn, lookup):
            def get_input(idx):
                inp = orig_fn(idx)
                src_name = inp.get_source_output().get_node().get_friendly_name()
                if src_name in lookup:
                    inp.get_source_output().get_node.return_value = lookup[src_name]
                return inp
            return get_input
        op.input = make_input_fn(original_input, op_by_name)

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

        assert colors["conv"] == "#335588"  # Netron "layer" color
        assert colors["unknown_op"] == "#333333"  # Other/default

    def test_constants_filtered(self):
        """Constant nodes should be excluded from the graph but tracked as inputs."""
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

        # conv1 should have 2 inputs: one visible, one const
        conv1 = next(n for n in graph.nodes if n.name == "conv1")
        assert len(conv1.inputs) == 2
        visible = [i for i in conv1.inputs if not i.is_const]
        const = [i for i in conv1.inputs if i.is_const]
        assert len(visible) == 1
        assert visible[0].name == "input"
        assert len(const) == 1
        assert const[0].name == "weights"
        assert const[0].shape == [64, 3, 3, 3]

    def test_constant_convert_chain_filtered(self):
        """Constant -> Convert chains (weight prep) should also be filtered."""
        ops = [
            _make_mock_op("input", "Parameter", output_shape=[1, 3, 224, 224]),
            _make_mock_op("w_const", "Constant", output_shape=[64, 3, 3, 3]),
            _make_mock_op("w_convert", "Convert", inputs=["w_const"], output_shape=[64, 3, 3, 3]),
            _make_mock_op("scale_const", "Constant", output_shape=[64]),
            _make_mock_op("w_multiply", "Multiply", inputs=["w_convert", "scale_const"], output_shape=[64, 3, 3, 3]),
            _make_mock_op("conv1", "Convolution", inputs=["input", "w_multiply"], output_shape=[1, 64, 112, 112]),
            _make_mock_op("output", "Result", inputs=["conv1"]),
        ]
        model = _make_mock_model(ops)
        graph = extract_graph(model)

        node_names = {n.name for n in graph.nodes}
        assert "w_const" not in node_names
        assert "w_convert" not in node_names
        assert "w_multiply" not in node_names
        assert "input" in node_names
        assert "conv1" in node_names
        assert len(graph.edges) == 2  # input->conv1, conv1->output

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


