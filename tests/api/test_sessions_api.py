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


class TestCloneSession:
    @pytest.mark.asyncio
    async def test_clone_basic(self, async_client, test_session, test_app):
        """Clone a session with no overrides."""
        resp = await async_client.post(
            f"/api/sessions/{test_session.id}/clone",
            json={},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session" in data
        assert "inferred_nodes" in data
        new_id = data["session"]["id"]
        assert new_id != test_session.id

    @pytest.mark.asyncio
    async def test_clone_with_device_override(self, async_client, test_session, test_app):
        """Clone with device override applies to new session."""
        resp = await async_client.post(
            f"/api/sessions/{test_session.id}/clone",
            json={"main_device": "GPU", "ref_device": "GPU"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session"]["main_device"] == "GPU"
        assert data["session"]["ref_device"] == "GPU"

    @pytest.mark.asyncio
    async def test_clone_stores_source_id(self, async_client, test_session, test_app):
        """Cloned session metadata references the source session."""
        resp = await async_client.post(
            f"/api/sessions/{test_session.id}/clone",
            json={},
        )
        data = resp.json()
        svc = test_app.state.session_service
        source_id = svc.get_source_session_id(data["session"]["id"])
        assert source_id == test_session.id

    @pytest.mark.asyncio
    async def test_clone_returns_inferred_nodes(self, async_client, test_session, test_app):
        """Clone returns source session's inferred nodes sorted by worst accuracy."""
        svc = test_app.state.session_service
        # Add some tasks with metrics
        svc.save_task_result(test_session.id, "t1", {
            "status": "success", "node_name": "conv1", "node_type": "Convolution",
            "metrics": {"cosine_similarity": 0.99, "mse": 0.001, "max_abs_diff": 0.01},
        })
        svc.save_task_result(test_session.id, "t2", {
            "status": "success", "node_name": "relu1", "node_type": "Relu",
            "metrics": {"cosine_similarity": 0.5, "mse": 0.1, "max_abs_diff": 0.5},
        })

        resp = await async_client.post(
            f"/api/sessions/{test_session.id}/clone", json={},
        )
        data = resp.json()
        nodes = data["inferred_nodes"]
        assert len(nodes) == 2
        # Worst accuracy (lowest cosine) first
        assert nodes[0]["node_name"] == "relu1"
        assert nodes[1]["node_name"] == "conv1"

    @pytest.mark.asyncio
    async def test_clone_not_found(self, async_client):
        """Clone of nonexistent session returns 404."""
        resp = await async_client.post(
            "/api/sessions/nonexistent/clone", json={},
        )
        assert resp.status_code == 404


class TestCloneEnqueue:
    @pytest.mark.asyncio
    async def test_clone_enqueue(self, async_client, test_session, test_app):
        """Enqueue nodes from source session into target session."""
        svc = test_app.state.session_service
        svc.save_task_result(test_session.id, "t1", {
            "status": "success", "node_name": "conv1", "node_type": "Convolution",
            "node_id": "conv_0",
            "metrics": {"cosine_similarity": 0.99, "mse": 0.001, "max_abs_diff": 0.01},
        })

        # Clone first to get a target session
        clone_resp = await async_client.post(
            f"/api/sessions/{test_session.id}/clone", json={},
        )
        target_id = clone_resp.json()["session"]["id"]

        # Start queue worker so enqueue works
        queue_svc = test_app.state.queue_service
        await queue_svc.start_worker()
        try:
            resp = await async_client.post(
                f"/api/sessions/{test_session.id}/clone-enqueue",
                json={"target_session_id": target_id, "node_names": ["conv1"]},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["enqueued"] == 1
            assert data["batch_id"]
        finally:
            await queue_svc.stop_worker()

    @pytest.mark.asyncio
    async def test_clone_enqueue_source_not_found(self, async_client):
        resp = await async_client.post(
            "/api/sessions/nonexistent/clone-enqueue",
            json={"target_session_id": "x", "node_names": ["a"]},
        )
        assert resp.status_code == 404


class TestCompare:
    @pytest.mark.asyncio
    async def test_compare_sessions(self, async_client, test_app, sample_model_files):
        """Compare two sessions with overlapping nodes."""
        svc = test_app.state.session_service
        from backend.schemas.session import SessionConfig
        config = SessionConfig(model_path=str(sample_model_files), main_device="CPU", ref_device="CPU")

        s1 = svc.create_session(config)
        s2 = svc.create_session(config)

        svc.save_task_result(s1.id, "t1", {
            "status": "success", "node_name": "conv1", "node_type": "Convolution",
            "metrics": {"cosine_similarity": 0.90, "mse": 0.01, "max_abs_diff": 0.1},
        })
        svc.save_task_result(s1.id, "t2", {
            "status": "success", "node_name": "relu1", "node_type": "Relu",
            "metrics": {"cosine_similarity": 0.95, "mse": 0.005, "max_abs_diff": 0.05},
        })

        svc.save_task_result(s2.id, "t3", {
            "status": "success", "node_name": "conv1", "node_type": "Convolution",
            "metrics": {"cosine_similarity": 0.95, "mse": 0.005, "max_abs_diff": 0.05},
        })
        svc.save_task_result(s2.id, "t4", {
            "status": "success", "node_name": "bn1", "node_type": "BatchNorm",
            "metrics": {"cosine_similarity": 0.99, "mse": 0.001, "max_abs_diff": 0.01},
        })

        resp = await async_client.get(
            f"/api/sessions/compare?session_a={s1.id}&session_b={s2.id}",
        )
        assert resp.status_code == 200
        data = resp.json()

        summary = data["summary"]
        assert summary["total_compared"] == 1  # conv1
        assert summary["improved"] == 1  # conv1 cosine went up
        assert summary["only_in_a"] == 1  # relu1
        assert summary["only_in_b"] == 1  # bn1

        # Check that conv1's delta is positive (improved)
        conv1_node = next(n for n in data["nodes"] if n["node_name"] == "conv1")
        assert conv1_node["delta_cosine"] > 0

    @pytest.mark.asyncio
    async def test_compare_session_not_found(self, async_client, test_session):
        resp = await async_client.get(
            f"/api/sessions/compare?session_a={test_session.id}&session_b=nonexistent",
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_compare_empty_sessions(self, async_client, test_app, sample_model_files):
        """Compare two sessions with no tasks."""
        svc = test_app.state.session_service
        from backend.schemas.session import SessionConfig
        config = SessionConfig(model_path=str(sample_model_files), main_device="CPU", ref_device="CPU")

        s1 = svc.create_session(config)
        s2 = svc.create_session(config)

        resp = await async_client.get(
            f"/api/sessions/compare?session_a={s1.id}&session_b={s2.id}",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["total_compared"] == 0
        assert data["nodes"] == []
