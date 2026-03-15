"""API integration tests for tensor endpoints."""
from __future__ import annotations

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
