"""Tests for inference service."""
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from backend.services.inference_service import InferenceService
from backend.schemas.inference import InferenceTask, TaskStatus, AccuracyMetrics


@pytest.fixture
def mock_ov_core():
    core = MagicMock()
    return core


@pytest.fixture
def inference_service(mock_ov_core):
    return InferenceService(mock_ov_core)


class TestComputeMetrics:
    def test_identical_tensors(self, inference_service):
        a = np.array([1.0, 2.0, 3.0])
        metrics = inference_service._compute_metrics(a, a.copy())

        assert metrics.mse == 0.0
        assert metrics.max_abs_diff == 0.0
        assert abs(metrics.cosine_similarity - 1.0) < 1e-6

    def test_different_tensors(self, inference_service):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        metrics = inference_service._compute_metrics(a, b)

        assert metrics.mse > 0
        assert metrics.max_abs_diff == 1.0
        assert abs(metrics.cosine_similarity) < 1e-6

    def test_zero_tensors(self, inference_service):
        a = np.zeros(10)
        b = np.zeros(10)
        metrics = inference_service._compute_metrics(a, b)

        assert metrics.mse == 0.0
        assert metrics.cosine_similarity == 1.0


class TestFindOp:
    def test_find_existing(self, inference_service):
        op = MagicMock()
        op.get_friendly_name.return_value = "conv1"
        model = MagicMock()
        model.get_ordered_ops.return_value = [op]

        found = inference_service._find_op(model, "conv1")
        assert found is op

    def test_find_nonexistent(self, inference_service):
        model = MagicMock()
        model.get_ordered_ops.return_value = []

        found = inference_service._find_op(model, "nonexistent")
        assert found is None


class TestCutAndInfer:
    def test_node_not_found(self, inference_service):
        model = MagicMock()
        model.get_ordered_ops.return_value = []

        result = inference_service.cut_and_infer(
            model, "nonexistent", "CPU", "CPU"
        )

        assert isinstance(result, InferenceTask)
        assert result.status == TaskStatus.FAILED
        assert "not found" in result.error_detail
