"""API integration tests for tensor endpoints."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest


class TestGetTensor:
    @pytest.mark.asyncio
    async def test_get_tensor(self, async_client, test_session, test_app, tmp_path):
        """Tensor download returns fp16 binary with correct headers."""
        svc = test_app.state.session_service

        # Create tensor artifact
        artifacts_dir = tmp_path / "tensor_artifacts"
        artifacts_dir.mkdir()
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        np.save(str(artifacts_dir / "main_output.npy"), data)

        svc.save_task_result(
            test_session.id, "t1",
            {"status": "success", "node_name": "conv1"},
            artifacts_dir=str(artifacts_dir),
        )

        resp = await async_client.get(
            f"/api/tensors/{test_session.id}/t1/main_output"
        )
        assert resp.status_code == 200
        assert resp.headers["x-tensor-shape"] == "2,2"
        assert resp.headers["x-tensor-dtype"] == "float32"
        assert resp.headers["content-type"] == "application/octet-stream"

        # Verify binary content is fp16
        result = np.frombuffer(resp.content, dtype=np.float16).reshape(2, 2)
        np.testing.assert_allclose(result.astype(np.float32), data, rtol=1e-3)

    @pytest.mark.asyncio
    async def test_get_tensor_not_found(self, async_client, test_session):
        resp = await async_client.get(
            f"/api/tensors/{test_session.id}/nonexistent/main_output"
        )
        assert resp.status_code == 404


class TestGetTensorMeta:
    @pytest.mark.asyncio
    async def test_get_tensor_meta(self, async_client, test_session, test_app, tmp_path):
        svc = test_app.state.session_service

        artifacts_dir = tmp_path / "meta_artifacts"
        artifacts_dir.mkdir()
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.save(str(artifacts_dir / "main_output.npy"), data)

        svc.save_task_result(
            test_session.id, "t1",
            {"status": "success", "node_name": "relu1"},
            artifacts_dir=str(artifacts_dir),
        )

        resp = await async_client.get(
            f"/api/tensors/{test_session.id}/t1/main_output/meta"
        )
        assert resp.status_code == 200
        meta = resp.json()
        assert meta["shape"] == [4]
        assert meta["dtype"] == "float32"
        assert meta["min"] == 1.0
        assert meta["max"] == 4.0
        assert meta["mean"] == 2.5

    @pytest.mark.asyncio
    async def test_get_tensor_meta_not_found(self, async_client, test_session):
        resp = await async_client.get(
            f"/api/tensors/{test_session.id}/nonexistent/main_output/meta"
        )
        assert resp.status_code == 404


class TestExportReproducer:
    def _create_task_artifacts(self, svc, session_id, tmp_path):
        """Create a successful task with full artifacts (model + inputs + outputs)."""
        artifacts_dir = tmp_path / "export_artifacts"
        artifacts_dir.mkdir()

        # Simulate cut model files
        (artifacts_dir / "cut_model.xml").write_text("<model/>")
        (artifacts_dir / "cut_model.bin").write_bytes(b"\x00" * 32)

        # Simulate input tensor
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        np.save(str(artifacts_dir / "input_image.npy"), input_data)

        # Simulate output tensors
        main_out = np.random.randn(1, 64, 112, 112).astype(np.float32)
        ref_out = np.random.randn(1, 64, 112, 112).astype(np.float32)
        np.save(str(artifacts_dir / "main_output.npy"), main_out)
        np.save(str(artifacts_dir / "ref_output.npy"), ref_out)

        task_data = {
            "status": "success",
            "node_name": "Conv2D_42",
            "node_type": "Convolution",
            "metrics": {"mse": 0.0042, "cosine_similarity": 0.9987, "max_abs_diff": 0.31},
        }

        svc.save_task_result(session_id, "t_export", task_data, artifacts_dir=str(artifacts_dir))

        return input_data, main_out, ref_out

    @pytest.mark.asyncio
    async def test_export_reproducer(self, async_client, test_session, test_app, tmp_path):
        """Export endpoint returns a valid ZIP with expected contents."""
        svc = test_app.state.session_service
        input_data, main_out, ref_out = self._create_task_artifacts(
            svc, test_session.id, tmp_path
        )

        resp = await async_client.get(
            f"/api/tensors/{test_session.id}/t_export/export"
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert "reproducer_Conv2D_42.zip" in resp.headers["content-disposition"]

        # Parse the ZIP
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = zf.namelist()

        # Check all expected files exist
        assert "reproducer/cut_model.xml" in names
        assert "reproducer/cut_model.bin" in names
        assert "reproducer/info.json" in names
        assert "reproducer/convert_npy.py" not in names

        # Check output bins
        assert "reproducer/main_output.bin" in names
        assert "reproducer/ref_output.bin" in names

        # Check at least one input bin
        input_bins = [n for n in names if n.startswith("reproducer/input_") and n.endswith(".bin")]
        assert len(input_bins) >= 1

        # Validate info.json
        info = json.loads(zf.read("reproducer/info.json"))
        assert info["node_name"] == "Conv2D_42"
        assert info["node_type"] == "Convolution"
        assert info["main_device"] == "CPU"
        assert info["ref_device"] == "CPU"
        assert info["metrics"]["mse"] == 0.0042
        assert len(info["inputs"]) >= 1
        assert len(info["outputs"]) == 2

        # Verify binary data is correct (raw, no numpy header)
        main_bin = zf.read("reproducer/main_output.bin")
        main_reconstructed = np.frombuffer(main_bin, dtype=np.float32).reshape(main_out.shape)
        np.testing.assert_array_equal(main_reconstructed, main_out)

        ref_bin = zf.read("reproducer/ref_output.bin")
        ref_reconstructed = np.frombuffer(ref_bin, dtype=np.float32).reshape(ref_out.shape)
        np.testing.assert_array_equal(ref_reconstructed, ref_out)

    @pytest.mark.asyncio
    async def test_export_task_not_found(self, async_client, test_session):
        """Export returns 404 for nonexistent task."""
        resp = await async_client.get(
            f"/api/tensors/{test_session.id}/nonexistent/export"
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_export_failed_task(self, async_client, test_session, test_app):
        """Export returns 400 for a failed task."""
        svc = test_app.state.session_service
        svc.save_task_result(
            test_session.id, "t_failed",
            {"status": "failed", "node_name": "bad_node", "error_detail": "crash"},
        )
        resp = await async_client.get(
            f"/api/tensors/{test_session.id}/t_failed/export"
        )
        assert resp.status_code == 400
