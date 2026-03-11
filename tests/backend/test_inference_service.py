"""Tests for inference service."""
import json
import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from backend.services.inference_service import InferenceService
from backend.schemas.inference import InferenceTask, TaskStatus


@pytest.fixture
def mock_ov_core():
    core = MagicMock()
    return core


@pytest.fixture
def inference_service(mock_ov_core):
    return InferenceService(mock_ov_core)


class TestComputeMetrics:
    """Test the metrics computation in the worker module."""

    def test_identical_tensors(self):
        from backend.utils.inference_worker import _compute_metrics

        a = np.array([1.0, 2.0, 3.0])
        metrics = _compute_metrics(a, a.copy())

        assert metrics["mse"] == 0.0
        assert metrics["max_abs_diff"] == 0.0
        assert abs(metrics["cosine_similarity"] - 1.0) < 1e-6

    def test_different_tensors(self):
        from backend.utils.inference_worker import _compute_metrics

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        metrics = _compute_metrics(a, b)

        assert metrics["mse"] > 0
        assert metrics["max_abs_diff"] == 1.0
        assert abs(metrics["cosine_similarity"]) < 1e-6

    def test_zero_tensors(self):
        from backend.utils.inference_worker import _compute_metrics

        a = np.zeros(10)
        b = np.zeros(10)
        metrics = _compute_metrics(a, b)

        assert metrics["mse"] == 0.0
        assert metrics["cosine_similarity"] == 1.0


class TestCutAndInfer:
    def test_subprocess_timeout(self, inference_service):
        """Test that timeouts are handled gracefully."""
        model = MagicMock()

        with patch("backend.services.inference_service.subprocess.run") as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=300)

            result = inference_service.cut_and_infer(
                model, "node1", "CPU", "CPU", model_path="/tmp/test.xml"
            )

            assert isinstance(result, InferenceTask)
            assert result.status == TaskStatus.FAILED
            assert "timed out" in result.error_detail

    def test_subprocess_crash(self, inference_service):
        """Test that segfaults are reported as errors, not crashes."""
        model = MagicMock()

        with patch("backend.services.inference_service.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=-11,  # SIGSEGV
                stdout="",
                stderr="segfault in libopenvino",
            )

            result = inference_service.cut_and_infer(
                model, "node1", "CPU", "CPU", model_path="/tmp/test.xml"
            )

            assert isinstance(result, InferenceTask)
            assert result.status == TaskStatus.FAILED
            assert "SIGSEGV" in result.error_detail

    def test_subprocess_success(self, inference_service, tmp_path):
        """Test successful subprocess result parsing."""
        model = MagicMock()

        main_npy = tmp_path / "main_output.npy"
        ref_npy = tmp_path / "ref_output.npy"
        np.save(str(main_npy), np.array([1.0, 2.0]))
        np.save(str(ref_npy), np.array([1.0, 2.0]))

        worker_result = {
            "main_result": {
                "device": "CPU",
                "output_shapes": [[2]],
                "dtype": "float32",
                "min_val": 1.0,
                "max_val": 2.0,
                "mean_val": 1.5,
                "std_val": 0.5,
            },
            "ref_result": {
                "device": "CPU",
                "output_shapes": [[2]],
                "dtype": "float32",
                "min_val": 1.0,
                "max_val": 2.0,
                "mean_val": 1.5,
                "std_val": 0.5,
            },
            "metrics": {
                "mse": 0.0,
                "max_abs_diff": 0.0,
                "cosine_similarity": 1.0,
            },
        }

        with patch("backend.services.inference_service.subprocess.run") as mock_run, \
             patch("backend.services.inference_service.tempfile.TemporaryDirectory") as mock_tmpdir:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(worker_result),
                stderr="",
            )
            mock_tmpdir.return_value.__enter__ = lambda s: str(tmp_path)
            mock_tmpdir.return_value.__exit__ = lambda s, *a: None

            result = inference_service.cut_and_infer(
                model, "node1", "CPU", "CPU", model_path="/tmp/test.xml"
            )

            # Success returns (task, main_output, ref_output)
            assert isinstance(result, tuple)
            task, main_out, ref_out = result
            assert task.status == TaskStatus.SUCCESS
            assert task.metrics.mse == 0.0
            assert main_out is not None
