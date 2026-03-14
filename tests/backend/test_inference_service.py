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


def _make_mock_popen(returncode, stdout="", stderr_lines=None):
    """Create a mock Popen object with the given behavior."""
    mock_proc = MagicMock()
    mock_proc.stdin = MagicMock()
    mock_proc.returncode = returncode
    mock_proc.poll.return_value = returncode

    # stdout.read() returns the stdout string
    mock_proc.stdout.read.return_value = stdout

    # stderr.readline() yields lines then empty string
    lines = list(stderr_lines or [])
    line_iter = iter(lines + [""])
    mock_proc.stderr.readline.side_effect = lambda: next(line_iter)

    mock_proc.wait.return_value = returncode
    return mock_proc


class TestCutAndInfer:
    def test_subprocess_timeout(self, inference_service):
        """Test that process killed by timer returns timeout error."""
        model = MagicMock()

        mock_proc = _make_mock_popen(returncode=-9)

        with patch("backend.services.inference_service.subprocess.Popen", return_value=mock_proc), \
             patch("backend.services.inference_service.tempfile.mkdtemp", return_value="/tmp/test_infer"):
            result = inference_service.cut_and_infer(
                model, "node1", "CPU", "CPU", model_path="/tmp/test.xml"
            )

            assert isinstance(result, InferenceTask)
            assert result.status == TaskStatus.FAILED
            assert "timed out" in result.error_detail

    def test_subprocess_crash(self, inference_service):
        """Test that segfaults are reported as errors, not crashes."""
        model = MagicMock()

        mock_proc = _make_mock_popen(
            returncode=-11,
            stderr_lines=["segfault in libopenvino\n"],
        )

        with patch("backend.services.inference_service.subprocess.Popen", return_value=mock_proc), \
             patch("backend.services.inference_service.tempfile.mkdtemp", return_value="/tmp/test_infer"):
            result = inference_service.cut_and_infer(
                model, "node1", "CPU", "CPU", model_path="/tmp/test.xml"
            )

            assert isinstance(result, InferenceTask)
            assert result.status == TaskStatus.FAILED
            assert "SIGSEGV" in result.error_detail

    def test_subprocess_success(self, inference_service, tmp_path):
        """Test successful subprocess result parsing."""
        model = MagicMock()

        # Create artifacts in tmp_path (simulating worker output)
        np.save(str(tmp_path / "main_output.npy"), np.array([1.0, 2.0]))
        np.save(str(tmp_path / "ref_output.npy"), np.array([1.0, 2.0]))

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

        mock_proc = _make_mock_popen(
            returncode=0,
            stdout=json.dumps(worker_result),
        )

        with patch("backend.services.inference_service.subprocess.Popen", return_value=mock_proc), \
             patch("backend.services.inference_service.tempfile.mkdtemp", return_value=str(tmp_path)):
            result = inference_service.cut_and_infer(
                model, "node1", "CPU", "CPU", model_path="/tmp/test.xml"
            )

            # Success returns (task, artifacts_dir)
            assert isinstance(result, tuple)
            task, artifacts_dir = result
            assert task.status == TaskStatus.SUCCESS
            assert task.metrics.mse == 0.0
            assert artifacts_dir == str(tmp_path)

    def test_custom_model_path_passed_to_worker(self, inference_service):
        """Custom model_path (e.g. sub-session) is forwarded to worker config."""
        model = MagicMock()
        captured_cfg = {}

        mock_proc = _make_mock_popen(returncode=-9)

        original_stdin_write = mock_proc.stdin.write
        def capturing_write(data):
            captured_cfg.update(json.loads(data))
            return original_stdin_write(data)
        mock_proc.stdin.write = capturing_write

        with patch("backend.services.inference_service.subprocess.Popen", return_value=mock_proc), \
             patch("backend.services.inference_service.tempfile.mkdtemp", return_value="/tmp/test_infer"):
            inference_service.cut_and_infer(
                model, "node1", "CPU", "CPU",
                model_path="/sub_sessions/abc/cut_model.xml",
            )

        assert captured_cfg["model_path"] == "/sub_sessions/abc/cut_model.xml"
        assert "skip_cut" not in captured_cfg
