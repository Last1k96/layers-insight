"""API integration tests for session endpoints."""
from __future__ import annotations

import pytest


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_session_valid(self, async_client, sample_model_files):
        resp = await async_client.post("/api/sessions", json={
            "model_path": str(sample_model_files),
            "main_device": "CPU",
            "ref_device": "CPU",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["model_name"] == "test_model"
        assert data["main_device"] == "CPU"

    @pytest.mark.asyncio
    async def test_create_session_with_plugin_config(self, async_client, sample_model_files):
        resp = await async_client.post("/api/sessions", json={
            "model_path": str(sample_model_files),
            "main_device": "CPU",
            "ref_device": "CPU",
            "plugin_config": {"NUM_STREAMS": "2", "INFERENCE_NUM_THREADS": "4"},
        })
        assert resp.status_code == 200
        data = resp.json()
        session_id = data["id"]

        # Verify plugin_config is persisted in session detail
        detail = await async_client.get(f"/api/sessions/{session_id}")
        assert detail.status_code == 200
        detail_data = detail.json()
        assert detail_data["config"]["plugin_config"] == {"NUM_STREAMS": "2", "INFERENCE_NUM_THREADS": "4"}

    @pytest.mark.asyncio
    async def test_create_session_without_plugin_config(self, async_client, sample_model_files):
        resp = await async_client.post("/api/sessions", json={
            "model_path": str(sample_model_files),
            "main_device": "CPU",
            "ref_device": "CPU",
        })
        assert resp.status_code == 200
        session_id = resp.json()["id"]

        detail = await async_client.get(f"/api/sessions/{session_id}")
        assert detail.json()["config"]["plugin_config"] == {}

    @pytest.mark.asyncio
    async def test_create_session_invalid_path(self, async_client):
        with pytest.raises(Exception):
            await async_client.post("/api/sessions", json={
                "model_path": "/nonexistent/model.xml",
                "main_device": "CPU",
                "ref_device": "CPU",
            })


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_empty(self, async_client):
        resp = await async_client.get("/api/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_populated(self, async_client, test_session):
        resp = await async_client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == test_session.id


class TestGetSession:
    @pytest.mark.asyncio
    async def test_get_found(self, async_client, test_session):
        resp = await async_client.get(f"/api/sessions/{test_session.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == test_session.id
        assert "config" in data
        assert "info" in data

    @pytest.mark.asyncio
    async def test_get_not_found(self, async_client):
        resp = await async_client.get("/api/sessions/nonexistent")
        assert resp.status_code == 404


class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_delete_found(self, async_client, test_session):
        resp = await async_client.delete(f"/api/sessions/{test_session.id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # Verify gone
        resp = await async_client.get(f"/api/sessions/{test_session.id}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_not_found(self, async_client):
        resp = await async_client.delete("/api/sessions/nonexistent")
        assert resp.status_code == 404


class TestSubSessions:
    @pytest.mark.asyncio
    async def test_list_sub_sessions(self, async_client, test_session, test_app):
        svc = test_app.state.session_service
        svc.create_sub_session(
            session_id=test_session.id,
            cut_type="output",
            cut_node="conv1",
            grayed_nodes=["relu1"],
        )
        resp = await async_client.get(f"/api/sessions/{test_session.id}/sub-sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["cut_node"] == "conv1"

    @pytest.mark.asyncio
    async def test_delete_sub_session(self, async_client, test_session, test_app):
        svc = test_app.state.session_service
        sub = svc.create_sub_session(
            session_id=test_session.id,
            cut_type="output",
            cut_node="conv1",
            grayed_nodes=["relu1"],
        )
        resp = await async_client.delete(
            f"/api/sessions/{test_session.id}/sub-sessions/{sub.id}"
        )
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_delete_sub_session_not_found(self, async_client, test_session):
        resp = await async_client.delete(
            f"/api/sessions/{test_session.id}/sub-sessions/nonexistent"
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_sub_session_cascade(self, async_client, test_session, test_app):
        """Deleting a parent sub-session also deletes children."""
        svc = test_app.state.session_service
        parent = svc.create_sub_session(
            session_id=test_session.id,
            cut_type="output",
            cut_node="conv1",
            grayed_nodes=["relu1"],
        )
        child = svc.create_sub_session(
            session_id=test_session.id,
            cut_type="input",
            cut_node="relu1",
            grayed_nodes=["param_0"],
            parent_sub_session_id=parent.id,
        )

        resp = await async_client.delete(
            f"/api/sessions/{test_session.id}/sub-sessions/{parent.id}"
        )
        assert resp.status_code == 200

        # Both should be gone
        resp = await async_client.get(f"/api/sessions/{test_session.id}/sub-sessions")
        assert resp.json() == []
