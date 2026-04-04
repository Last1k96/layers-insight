"""Tests for bisection search service and router."""
from __future__ import annotations

import asyncio
import math

import pytest

from backend.schemas.bisect import (
    BisectMetric,
    BisectProgress,
    BisectRequest,
    BisectSearchFor,
    BisectStatus,
)
from backend.schemas.graph import GraphData, GraphEdge, GraphNode
from backend.schemas.inference import AccuracyMetrics, InferenceTask, TaskStatus
from backend.services.bisect_service import BisectService, _nodes_between


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(n: int = 10) -> GraphData:
    """Create a linear chain: param -> op_0 -> op_1 -> ... -> op_{n-1} -> result."""
    nodes = [GraphNode(id="param_0", name="param_0", type="Parameter")]
    edges = []
    prev = "param_0"
    for i in range(n):
        nid = f"op_{i}"
        nodes.append(GraphNode(id=nid, name=nid, type="Convolution"))
        edges.append(GraphEdge(source=prev, target=nid))
        prev = nid
    nodes.append(GraphNode(id="result_0", name="result_0", type="Result"))
    edges.append(GraphEdge(source=prev, target="result_0"))
    return GraphData(nodes=nodes, edges=edges)


def _make_task(task_id: str, node_name: str, status: TaskStatus,
               cos: float = 1.0, error: str | None = None) -> InferenceTask:
    t = InferenceTask(
        task_id=task_id,
        session_id="s1",
        node_id=node_name,
        node_name=node_name,
        node_type="Convolution",
        status=status,
        batch_id="bisect",
    )
    if status == TaskStatus.SUCCESS:
        t.metrics = AccuracyMetrics(cosine_similarity=cos, mse=0.0, max_abs_diff=0.0)
    if error:
        t.error_detail = error
    return t


# ---------------------------------------------------------------------------
# _nodes_between tests
# ---------------------------------------------------------------------------

def test_nodes_between_full_range():
    g = _make_graph(5)
    nodes = _nodes_between(g, None, None)
    # Should exclude Parameter and Result
    assert nodes == [f"op_{i}" for i in range(5)]


def test_nodes_between_partial_range():
    g = _make_graph(5)
    nodes = _nodes_between(g, "op_1", "op_3")
    assert nodes == ["op_1", "op_2", "op_3"]


# ---------------------------------------------------------------------------
# BisectService unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bisect_finds_accuracy_drop():
    """Simulate a model where op_5 is the first node with cos < 0.999."""
    svc = BisectService()
    graph = _make_graph(10)
    req = BisectRequest(
        session_id="s1",
        metric=BisectMetric.COSINE_SIMILARITY,
        threshold=0.999,
        search_for=BisectSearchFor.ACCURACY_DROP,
    )

    created_tasks: list[InferenceTask] = []
    broadcast_msgs: list[dict] = []

    class FakeQueue:
        def create_task(self, session_id, node_id, node_name, node_type):
            return InferenceTask(
                task_id=f"t_{node_name}",
                session_id=session_id,
                node_id=node_id,
                node_name=node_name,
                node_type=node_type,
                status=TaskStatus.WAITING,
            )

        async def enqueue(self, task):
            created_tasks.append(task)
            return task

    async def fake_broadcast(session_id, msg):
        broadcast_msgs.append(msg)

    queue = FakeQueue()

    # Start bisection
    job = await svc.start(
        request=req,
        graph_data=graph,
        queue_service=queue,
        session_service=None,
        broadcast=fake_broadcast,
    )
    assert job.status == BisectStatus.RUNNING

    # Simulate inference results: op_0..op_4 pass, op_5+ fail
    while svc.is_running:
        await asyncio.sleep(0.01)
        if created_tasks:
            task = created_tasks[-1]
            node_idx = int(task.node_name.split("_")[1])
            if node_idx < 5:
                result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS, cos=1.0)
            else:
                result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS, cos=0.5)
            svc.on_task_complete(result)
            await asyncio.sleep(0.01)

    assert svc.job.status == BisectStatus.DONE
    assert svc.job.found_node == "op_5"


@pytest.mark.asyncio
async def test_bisect_finds_compilation_failure():
    """Simulate a model where op_3 is the first node that fails compilation."""
    svc = BisectService()
    graph = _make_graph(8)
    req = BisectRequest(
        session_id="s1",
        search_for=BisectSearchFor.COMPILATION_FAILURE,
        threshold=0.0,
    )

    created_tasks: list[InferenceTask] = []

    class FakeQueue:
        def create_task(self, session_id, node_id, node_name, node_type):
            return InferenceTask(
                task_id=f"t_{node_name}",
                session_id=session_id,
                node_id=node_id,
                node_name=node_name,
                node_type=node_type,
                status=TaskStatus.WAITING,
            )

        async def enqueue(self, task):
            created_tasks.append(task)
            return task

    async def fake_broadcast(session_id, msg):
        pass

    queue = FakeQueue()
    job = await svc.start(
        request=req,
        graph_data=graph,
        queue_service=queue,
        session_service=None,
        broadcast=fake_broadcast,
    )

    while svc.is_running:
        await asyncio.sleep(0.01)
        if created_tasks:
            task = created_tasks[-1]
            node_idx = int(task.node_name.split("_")[1])
            if node_idx < 3:
                result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS)
            else:
                result = _make_task(task.task_id, task.node_name, TaskStatus.FAILED,
                                    error="Compilation failed")
            svc.on_task_complete(result)
            await asyncio.sleep(0.01)

    assert svc.job.status == BisectStatus.DONE
    assert svc.job.found_node == "op_3"


@pytest.mark.asyncio
async def test_bisect_stop():
    """Starting then stopping a bisection sets status to STOPPED."""
    svc = BisectService()
    graph = _make_graph(10)
    req = BisectRequest(session_id="s1", threshold=0.999)

    class FakeQueue:
        def create_task(self, **kw):
            return InferenceTask(
                task_id="t1", session_id="s1", node_id="op_0",
                node_name="op_0", node_type="Conv", status=TaskStatus.WAITING,
            )

        async def enqueue(self, task):
            return task

        def get_all_tasks(self, session_id=None):
            return []

        async def cancel(self, task_id):
            return True

    async def noop_broadcast(sid, msg):
        pass

    await svc.start(
        request=req, graph_data=graph,
        queue_service=FakeQueue(), session_service=None,
        broadcast=noop_broadcast,
    )

    job = await svc.stop()
    assert job.status == BisectStatus.STOPPED


@pytest.mark.asyncio
async def test_bisect_pause_resume():
    """Pausing preserves state, resuming continues."""
    svc = BisectService()
    graph = _make_graph(10)
    req = BisectRequest(session_id="s1", threshold=0.999)

    created_tasks: list[InferenceTask] = []

    class FakeQueue:
        def create_task(self, session_id, node_id, node_name, node_type):
            return InferenceTask(
                task_id=f"t_{node_name}",
                session_id=session_id,
                node_id=node_id,
                node_name=node_name,
                node_type=node_type,
                status=TaskStatus.WAITING,
            )

        async def enqueue(self, task):
            created_tasks.append(task)
            return task

        def get_all_tasks(self, session_id=None):
            return []

        async def cancel(self, task_id):
            return True

    async def noop_broadcast(sid, msg):
        pass

    await svc.start(
        request=req, graph_data=graph,
        queue_service=FakeQueue(), session_service=None,
        broadcast=noop_broadcast,
    )

    # Let it enqueue the first task
    await asyncio.sleep(0.05)
    assert len(created_tasks) >= 1

    # Pause
    job = await svc.pause()
    assert job.status == BisectStatus.PAUSED
    assert svc._lo >= 0  # State preserved

    # Resume
    job = await svc.resume()
    assert job.status == BisectStatus.RUNNING

    # Stop to clean up
    await svc.stop()


@pytest.mark.asyncio
async def test_bisect_pause_resume_with_completed_task():
    """Regression: task completing while bisect is paused must not hang on resume.

    Reproduces the race where on_task_complete fires after pause sets status to
    PAUSED, so the result must be picked up on resume without waiting forever.
    """
    svc = BisectService()
    graph = _make_graph(10)
    req = BisectRequest(session_id="s1", threshold=0.999)

    created_tasks: list[InferenceTask] = []

    class FakeQueue:
        def create_task(self, session_id, node_id, node_name, node_type):
            return InferenceTask(
                task_id=f"t_{node_name}",
                session_id=session_id,
                node_id=node_id,
                node_name=node_name,
                node_type=node_type,
                status=TaskStatus.WAITING,
            )

        async def enqueue(self, task):
            created_tasks.append(task)
            return task

        def get_all_tasks(self, session_id=None):
            return []

        async def cancel(self, task_id):
            return True

    async def noop_broadcast(sid, msg):
        pass

    await svc.start(
        request=req, graph_data=graph,
        queue_service=FakeQueue(), session_service=None,
        broadcast=noop_broadcast,
    )

    # Let it enqueue the first task
    await asyncio.sleep(0.05)
    assert len(created_tasks) >= 1
    assert svc._pending_task_id is not None

    # Pause the bisect (loop exits with _pending_task_id still set)
    await svc.pause()
    assert svc.is_paused

    # Simulate the race: task completes AFTER pause (on_task_complete fires
    # while bisect is paused — the is_active fix allows this)
    task = created_tasks[-1]
    result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS, cos=1.0)
    svc.on_task_complete(result)

    # Resume — the loop must pick up the stored result without hanging
    await svc.resume()

    # Give the loop time to process (with timeout to detect hangs)
    async def wait_for_next_step():
        for _ in range(100):
            await asyncio.sleep(0.01)
            # Loop should have consumed the result and moved on
            if svc._pending_task_id is None:
                return True
            # Or if it enqueued a new task for the next step
            if len(created_tasks) > 1:
                return True
        return False

    ok = await asyncio.wait_for(wait_for_next_step(), timeout=3.0)
    assert ok, "Bisect loop hung after resume — _pending_task_id was never cleared"

    # Clean up
    await svc.stop()


@pytest.mark.asyncio
async def test_bisect_too_few_nodes():
    """Bisection with fewer than 2 nodes should raise ValueError."""
    svc = BisectService()
    # Graph with only 1 non-param/result node
    graph = _make_graph(1)
    req = BisectRequest(session_id="s1", threshold=0.999)

    with pytest.raises(ValueError, match="at least 2 nodes"):
        await svc.start(
            request=req, graph_data=graph,
            queue_service=None, session_service=None,
            broadcast=None,
        )


# ---------------------------------------------------------------------------
# Router tests (via httpx)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bisect_status_endpoint(async_client, test_app):
    """GET /api/inference/bisect/status returns idle when nothing is running."""
    resp = await async_client.get("/api/inference/bisect/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "idle"


@pytest.mark.asyncio
async def test_bisect_start_no_graph(async_client, test_app, test_session):
    """POST /api/inference/bisect returns 404 when graph is not loaded."""
    resp = await async_client.post("/api/inference/bisect", json={
        "session_id": test_session.id,
        "threshold": 0.999,
    })
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_bisect_stop_when_idle(async_client, test_app):
    """POST /api/inference/bisect/stop returns 404 when nothing is running."""
    resp = await async_client.post("/api/inference/bisect/stop")
    assert resp.status_code == 404
