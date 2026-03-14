"""Tests for model cut service."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from backend.services.model_cut_service import ModelCutService


@pytest.fixture
def mock_ov_core():
    return MagicMock()


@pytest.fixture
def model_cut_service(mock_ov_core):
    return ModelCutService(mock_ov_core)


def _make_mock_op(name, type_name="Unknown", inputs=None, outputs=None,
                  target_inputs=None, is_result=False):
    """Build a mock OV operation with graph traversal support."""
    op = MagicMock()
    op.get_friendly_name.return_value = name
    op.get_type_name.return_value = type_name

    # Default single output
    mock_output = MagicMock()
    mock_output.get_node.return_value = op
    mock_output.get_target_inputs.return_value = target_inputs or []
    op.output.return_value = mock_output
    op.outputs.return_value = [mock_output]

    # Inputs
    if inputs:
        op.get_input_size.return_value = len(inputs)
        mock_inputs = []
        for src_output in inputs:
            mi = MagicMock()
            mi.get_source_output.return_value = src_output
            mock_inputs.append(mi)
        op.input.side_effect = lambda i: mock_inputs[i]
    else:
        op.get_input_size.return_value = 0

    return op


class TestModelCutService:
    def test_find_op_existing(self, model_cut_service):
        op = MagicMock()
        op.get_friendly_name.return_value = "conv1"
        model = MagicMock()
        model.get_ordered_ops.return_value = [op]

        found = model_cut_service._find_op(model, "conv1")
        assert found is op

    def test_find_op_not_found(self, model_cut_service):
        model = MagicMock()
        model.get_ordered_ops.return_value = []

        found = model_cut_service._find_op(model, "nonexistent")
        assert found is None

    def test_make_output_node_not_found(self, model_cut_service):
        model = MagicMock()
        model.get_ordered_ops.return_value = []

        with pytest.raises(ValueError, match="not found"):
            model_cut_service.make_output_node(model, "nonexistent")

    def test_make_input_node_not_found(self, model_cut_service, tmp_path):
        """ValueError when the target node doesn't exist."""
        mock_model = MagicMock()
        mock_model.get_ordered_ops.return_value = []
        model_cut_service.core.read_model.return_value = mock_model

        npy = tmp_path / "dummy.npy"
        np.save(str(npy), np.zeros((1, 3)))

        with pytest.raises(ValueError, match="not found"):
            model_cut_service.make_input_node(
                "/fake/model.xml", "nonexistent", str(npy)
            )

    def test_make_input_node_reads_fresh_model(self, model_cut_service, tmp_path):
        """Verify core.read_model is called with model_path (fresh copy)."""
        mock_model = MagicMock()
        mock_model.get_ordered_ops.return_value = []
        model_cut_service.core.read_model.return_value = mock_model

        npy = tmp_path / "dummy.npy"
        np.save(str(npy), np.zeros((1, 3)))

        with pytest.raises(ValueError):
            model_cut_service.make_input_node(
                "/some/model.xml", "node1", str(npy)
            )

        model_cut_service.core.read_model.assert_called_once_with("/some/model.xml")
