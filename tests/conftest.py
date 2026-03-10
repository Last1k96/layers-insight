"""Shared test fixtures."""
import pytest


@pytest.fixture
def sample_graph_data():
    """Minimal graph data for testing."""
    return {
        "nodes": [
            {"id": "param_0", "name": "input", "type": "Parameter", "shape": [1, 3, 224, 224]},
            {"id": "conv_0", "name": "conv1", "type": "Convolution", "shape": [1, 64, 112, 112]},
            {"id": "relu_0", "name": "relu1", "type": "Relu", "shape": [1, 64, 112, 112]},
            {"id": "result_0", "name": "output", "type": "Result", "shape": [1, 64, 112, 112]},
        ],
        "edges": [
            {"source": "param_0", "target": "conv_0", "source_port": 0, "target_port": 0},
            {"source": "conv_0", "target": "relu_0", "source_port": 0, "target_port": 0},
            {"source": "relu_0", "target": "result_0", "source_port": 0, "target_port": 0},
        ],
    }
