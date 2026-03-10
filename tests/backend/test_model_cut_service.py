"""Tests for model cut service."""
import pytest
from unittest.mock import MagicMock

from backend.services.model_cut_service import ModelCutService


@pytest.fixture
def mock_ov_core():
    return MagicMock()


@pytest.fixture
def model_cut_service(mock_ov_core):
    return ModelCutService(mock_ov_core)


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
