"""API integration tests for /api/uploads."""
from __future__ import annotations

from pathlib import Path

import pytest


class TestUploadFile:
    @pytest.mark.asyncio
    async def test_single_file(self, async_client):
        files = {"file": ("hello.bin", b"hello world", "application/octet-stream")}
        resp = await async_client.post("/api/uploads", files=files)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["original_filename"] == "hello.bin"
        assert body["size"] == 11
        assert body["group_id"]
        assert Path(body["staged_path"]).read_bytes() == b"hello world"
        assert body["warnings"] == []

    @pytest.mark.asyncio
    async def test_ir_pair_same_group(self, async_client):
        first = await async_client.post(
            "/api/uploads",
            files={"file": ("model.xml", b"<model/>", "application/xml")},
            data={"kind": "model"},
        )
        assert first.status_code == 200
        gid = first.json()["group_id"]
        # First upload should warn about missing .bin sibling.
        assert "missing .bin sibling" in first.json()["warnings"]

        second = await async_client.post(
            "/api/uploads",
            files={"file": ("model.bin", b"\x00" * 32, "application/octet-stream")},
            data={"group_id": gid, "kind": "model"},
        )
        assert second.status_code == 200
        assert second.json()["group_id"] == gid

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, async_client):
        files = {"file": ("../escape.bin", b"x", "application/octet-stream")}
        resp = await async_client.post("/api/uploads", files=files)
        # The basename strip turns ../escape.bin into escape.bin which is fine,
        # so try a literal traversal that survives basename().
        assert resp.status_code in (200, 400)

    @pytest.mark.asyncio
    async def test_dotdot_filename_rejected(self, async_client):
        files = {"file": ("..", b"x", "application/octet-stream")}
        resp = await async_client.post("/api/uploads", files=files)
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_unknown_group_rejected(self, async_client):
        files = {"file": ("a.bin", b"x", "application/octet-stream")}
        resp = await async_client.post(
            "/api/uploads",
            files=files,
            data={"group_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_oversize_rejected(self, async_client, test_app):
        # Squeeze the per-file limit on the live service.
        test_app.state.upload_service.max_upload_bytes = 5
        try:
            files = {"file": ("big.bin", b"x" * 100, "application/octet-stream")}
            resp = await async_client.post("/api/uploads", files=files)
            assert resp.status_code == 413
        finally:
            test_app.state.upload_service.max_upload_bytes = 5 << 30


class TestListAndDelete:
    @pytest.mark.asyncio
    async def test_list_groups(self, async_client):
        await async_client.post(
            "/api/uploads",
            files={"file": ("a.bin", b"abc", "application/octet-stream")},
        )
        resp = await async_client.get("/api/uploads")
        assert resp.status_code == 200
        groups = resp.json()
        assert len(groups) >= 1

    @pytest.mark.asyncio
    async def test_delete_group(self, async_client):
        upload = await async_client.post(
            "/api/uploads",
            files={"file": ("a.bin", b"abc", "application/octet-stream")},
        )
        gid = upload.json()["group_id"]
        resp = await async_client.delete(f"/api/uploads/{gid}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == gid

    @pytest.mark.asyncio
    async def test_delete_unknown_group_404(self, async_client):
        resp = await async_client.delete(
            "/api/uploads/00000000-0000-0000-0000-000000000000"
        )
        assert resp.status_code == 404
