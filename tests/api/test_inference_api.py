"""API integration tests for inference endpoints."""
from __future__ import annotations

import pytest


class TestEnqueue:
    @pytest.mark.asyncio
    async def test_enqueue_task(self, async_client, test_session):
        resp = await async_client.post("/api/inference/enqueue", json={
            "session_id": test_session.id,
            "node_id": "conv_0",
            "node_name": "conv1",
            "node_type": "Convolution",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "waiting"
        assert data["node_name"] == "conv1"
        assert data["session_id"] == test_session.id

    @pytest.mark.asyncio
    async def test_enqueue_grayed_node(self, async_client, test_session, test_app):
        """Enqueuing a grayed-out node should return 400."""
        svc = test_app.state.session_service
        sub = svc.create_sub_session(
            session_id=test_session.id,
            cut_type="output",
            cut_node="conv1",
            grayed_nodes=["relu1"],
        )

        resp = await async_client.post("/api/inference/enqueue", json={
            "session_id": test_session.id,
            "node_id": "relu_0",
            "node_name": "relu1",
            "node_type": "Relu",
            "sub_session_id": sub.id,
        })
        assert resp.status_code == 400


class TestEnqueueBatch:
    @pytest.mark.asyncio
    async def test_enqueue_batch(self, async_client, test_session):
        resp = await async_client.post("/api/inference/enqueue-batch", json={
            "session_id": test_session.id,
            "nodes": [
                {"node_id": "conv_0", "node_name": "conv1", "node_type": "Convolution"},
                {"node_id": "relu_0", "node_name": "relu1", "node_type": "Relu"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        # All should share the same batch_id
        assert data[0]["batch_id"] is not None
        assert data[0]["batch_id"] == data[1]["batch_id"]


class TestGetTask:
    @pytest.mark.asyncio
    async def test_get_task_found(self, async_client, test_session):
        # First enqueue
        enq = await async_client.post("/api/inference/enqueue", json={
            "session_id": test_session.id,
            "node_id": "conv_0",
            "node_name": "conv1",
            "node_type": "Convolution",
        })
        task_id = enq.json()["task_id"]

        resp = await async_client.get(f"/api/inference/{task_id}")
        assert resp.status_code == 200
        assert resp.json()["task_id"] == task_id

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, async_client):
        resp = await async_client.get("/api/inference/nonexistent")
        assert resp.status_code == 404


class TestDeleteTask:
    @pytest.mark.asyncio
    async def test_delete_found(self, async_client, test_session, test_app):
        enq = await async_client.post("/api/inference/enqueue", json={
            "session_id": test_session.id,
            "node_id": "conv_0",
            "node_name": "conv1",
            "node_type": "Convolution",
        })
        task_id = enq.json()["task_id"]

        resp = await async_client.delete(f"/api/inference/{task_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

        # Should be gone
        resp = await async_client.get(f"/api/inference/{task_id}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_not_found(self, async_client):
        resp = await async_client.delete("/api/inference/nonexistent")
        assert resp.status_code == 404


class TestRerunTask:
    @pytest.mark.asyncio
    async def test_rerun_found(self, async_client, test_session):
        enq = await async_client.post("/api/inference/enqueue", json={
            "session_id": test_session.id,
            "node_id": "conv_0",
            "node_name": "conv1",
            "node_type": "Convolution",
        })
        task_id = enq.json()["task_id"]

        resp = await async_client.post(f"/api/inference/{task_id}/rerun")
        assert resp.status_code == 200
        new_data = resp.json()
        assert new_data["task_id"] != task_id
        assert new_data["node_name"] == "conv1"

    @pytest.mark.asyncio
    async def test_rerun_not_found(self, async_client):
        resp = await async_client.post("/api/inference/nonexistent/rerun")
        assert resp.status_code == 404


class TestReorderTasks:
    @pytest.mark.asyncio
    async def test_reorder(self, async_client, test_session):
        # Enqueue two tasks
        r1 = await async_client.post("/api/inference/enqueue", json={
            "session_id": test_session.id,
            "node_id": "conv_0", "node_name": "conv1", "node_type": "Convolution",
        })
        r2 = await async_client.post("/api/inference/enqueue", json={
            "session_id": test_session.id,
            "node_id": "relu_0", "node_name": "relu1", "node_type": "Relu",
        })

        t1 = r1.json()["task_id"]
        t2 = r2.json()["task_id"]

        # Reverse order
        resp = await async_client.put("/api/inference/reorder", json={
            "task_ids": [t2, t1],
        })
        assert resp.status_code == 200
        assert resp.json()["reordered"] is True
