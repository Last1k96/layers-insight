"""Tests for bisection search service and router."""
from __future__ import annotations

import asyncio
import math

import pytest

from backend.schemas.bisect import (
    BisectMetric,
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
               cos: float = 1.0, error: str | None = None,
               batch_id: str = "bisect") -> InferenceTask:
    t = InferenceTask(
        task_id=task_id,
        session_id="s1",
        node_id=node_name,
        node_name=node_name,
        node_type="Convolution",
        status=status,
        batch_id=batch_id,
    )
    if status == TaskStatus.SUCCESS:
        t.metrics = AccuracyMetrics(cosine_similarity=cos, mse=0.0, max_abs_diff=0.0)
    if error:
        t.error_detail = error
    return t


class FakeQueue:
    def __init__(self):
        self.created_tasks: list[InferenceTask] = []

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
        self.created_tasks.append(task)
        return task

    def get_all_tasks(self, session_id=None):
        return []

    async def cancel(self, task_id):
        return True


async def noop_broadcast(sid, msg):
    pass


# ---------------------------------------------------------------------------
# _nodes_between tests
# ---------------------------------------------------------------------------

def test_nodes_between_full_range():
    g = _make_graph(5)
    nodes = _nodes_between(g, None, None)
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

    queue = FakeQueue()
    job = await svc.start(
        request=req, graph_data=graph,
        queue_service=queue, session_service=None,
        broadcast=noop_broadcast,
    )
    assert job.status == BisectStatus.RUNNING

    while any(s.job.status == BisectStatus.RUNNING for s in svc._jobs.values()):
        await asyncio.sleep(0.01)
        if queue.created_tasks:
            task = queue.created_tasks[-1]
            node_idx = int(task.node_name.split("_")[1])
            if node_idx < 5:
                result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                                    cos=1.0, batch_id=task.batch_id)
            else:
                result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                                    cos=0.5, batch_id=task.batch_id)
            svc.on_task_complete(result)
            await asyncio.sleep(0.01)

    final_job = svc.get_job(job.job_id)
    assert final_job.status == BisectStatus.DONE
    assert final_job.found_node == "op_5"


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

    queue = FakeQueue()
    job = await svc.start(
        request=req, graph_data=graph,
        queue_service=queue, session_service=None,
        broadcast=noop_broadcast,
    )

    while any(s.job.status == BisectStatus.RUNNING for s in svc._jobs.values()):
        await asyncio.sleep(0.01)
        if queue.created_tasks:
            task = queue.created_tasks[-1]
            node_idx = int(task.node_name.split("_")[1])
            if node_idx < 3:
                result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                                    batch_id=task.batch_id)
            else:
                result = _make_task(task.task_id, task.node_name, TaskStatus.FAILED,
                                    error="Compilation failed", batch_id=task.batch_id)
            svc.on_task_complete(result)
            await asyncio.sleep(0.01)

    final_job = svc.get_job(job.job_id)
    assert final_job.status == BisectStatus.DONE
    assert final_job.found_node == "op_3"


@pytest.mark.asyncio
async def test_bisect_stop():
    """Starting then stopping a bisection sets status to STOPPED."""
    svc = BisectService()
    graph = _make_graph(10)
    req = BisectRequest(session_id="s1", threshold=0.999)

    job = await svc.start(
        request=req, graph_data=graph,
        queue_service=FakeQueue(), session_service=None,
        broadcast=noop_broadcast,
    )

    stopped = await svc.stop(job.job_id)
    assert stopped.status == BisectStatus.STOPPED
    assert svc.get_job(job.job_id) is None  # removed from _jobs


@pytest.mark.asyncio
async def test_bisect_pause_resume():
    """Pausing preserves state, resuming continues."""
    svc = BisectService()
    graph = _make_graph(10)
    req = BisectRequest(session_id="s1", threshold=0.999)

    queue = FakeQueue()
    job = await svc.start(
        request=req, graph_data=graph,
        queue_service=queue, session_service=None,
        broadcast=noop_broadcast,
    )

    # Let it enqueue the first task
    await asyncio.sleep(0.05)
    assert len(queue.created_tasks) >= 1

    # Pause all
    await svc.pause_all()
    state = svc._jobs[job.job_id]
    assert state.job.status == BisectStatus.PAUSED

    # Resume all
    await svc.resume_all()
    assert state.job.status == BisectStatus.RUNNING

    # Stop to clean up
    await svc.stop(job.job_id)


@pytest.mark.asyncio
async def test_bisect_pause_resume_with_completed_task():
    """Regression: task completing while bisect is paused must not hang on resume."""
    svc = BisectService()
    graph = _make_graph(10)
    req = BisectRequest(session_id="s1", threshold=0.999)

    queue = FakeQueue()
    job = await svc.start(
        request=req, graph_data=graph,
        queue_service=queue, session_service=None,
        broadcast=noop_broadcast,
    )

    await asyncio.sleep(0.05)
    assert len(queue.created_tasks) >= 1
    state = svc._jobs[job.job_id]
    assert state.pending_task_id is not None

    await svc.pause_all()
    assert state.job.status == BisectStatus.PAUSED

    # Simulate the race: task completes AFTER pause
    task = queue.created_tasks[-1]
    result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                        cos=1.0, batch_id=task.batch_id)
    svc.on_task_complete(result)

    await svc.resume_all()

    async def wait_for_next_step():
        for _ in range(100):
            await asyncio.sleep(0.01)
            if state.pending_task_id is None:
                return True
            if len(queue.created_tasks) > 1:
                return True
        return False

    ok = await asyncio.wait_for(wait_for_next_step(), timeout=3.0)
    assert ok, "Bisect loop hung after resume"

    await svc.stop(job.job_id)


@pytest.mark.asyncio
async def test_bisect_multiple_concurrent_jobs():
    """Multiple bisect jobs can run concurrently."""
    svc = BisectService()
    graph = _make_graph(10)

    queue = FakeQueue()
    job1 = await svc.start(
        request=BisectRequest(session_id="s1", threshold=0.999),
        graph_data=graph,
        queue_service=queue, session_service=None,
        broadcast=noop_broadcast,
    )
    job2 = await svc.start(
        request=BisectRequest(session_id="s1", threshold=0.999),
        graph_data=graph,
        queue_service=queue, session_service=None,
        broadcast=noop_broadcast,
    )

    assert job1.job_id != job2.job_id
    assert len(svc._jobs) == 2
    assert svc.has_active_jobs
    assert svc.has_running_jobs

    # Each job's tasks have distinct batch_ids
    await asyncio.sleep(0.05)
    batch_ids = {t.batch_id for t in queue.created_tasks}
    assert f"bisect:{job1.job_id}" in batch_ids
    assert f"bisect:{job2.job_id}" in batch_ids

    # Stopping one doesn't affect the other
    await svc.stop(job1.job_id)
    assert svc.get_job(job1.job_id) is None
    assert svc.get_job(job2.job_id) is not None
    assert svc.get_job(job2.job_id).status == BisectStatus.RUNNING

    await svc.stop(job2.job_id)
    assert len(svc._jobs) == 0


@pytest.mark.asyncio
async def test_bisect_on_task_complete_routes_correctly():
    """on_task_complete routes to the correct job based on batch_id."""
    svc = BisectService()
    graph = _make_graph(10)
    queue = FakeQueue()

    job1 = await svc.start(
        request=BisectRequest(session_id="s1", threshold=0.999),
        graph_data=graph,
        queue_service=queue, session_service=None,
        broadcast=noop_broadcast,
    )

    await asyncio.sleep(0.05)
    state1 = svc._jobs[job1.job_id]

    # Task for job1 should route to job1's state
    task = queue.created_tasks[-1]
    result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                        cos=1.0, batch_id=f"bisect:{job1.job_id}")
    svc.on_task_complete(result)

    assert state1.step_result is result
    assert state1.step_done.is_set()

    # Task with wrong batch_id should be ignored
    state1.step_done.clear()
    state1.step_result = None
    bogus = _make_task("bogus", "bogus", TaskStatus.SUCCESS, batch_id="bisect:nonexistent")
    svc.on_task_complete(bogus)
    assert state1.step_result is None

    await svc.stop(job1.job_id)


@pytest.mark.asyncio
async def test_bisect_too_few_nodes():
    """Bisection with fewer than 2 nodes should raise ValueError."""
    svc = BisectService()
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
    """GET /api/inference/bisect/status returns empty jobs when nothing is running."""
    resp = await async_client.get("/api/inference/bisect/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["jobs"] == []


@pytest.mark.asyncio
async def test_bisect_start_no_graph(async_client, test_app, test_session):
    """POST /api/inference/bisect returns 404 when graph is not loaded."""
    resp = await async_client.post("/api/inference/bisect", json={
        "session_id": test_session.id,
        "threshold": 0.999,
    })
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_bisect_stop_when_not_found(async_client, test_app):
    """POST /api/inference/bisect/{job_id}/stop returns 404 when job doesn't exist."""
    resp = await async_client.post("/api/inference/bisect/nonexistent/stop")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Sub-session bisect tests
# ---------------------------------------------------------------------------

class FakeSessionService:
    """Minimal session service that returns configurable grayed_nodes."""
    def __init__(self, grayed_nodes: list[str] | None = None):
        self._grayed = grayed_nodes or []

    def get_sub_session_meta(self, session_id: str, sub_session_id: str):
        return {"grayed_nodes": self._grayed}

    def find_task_for_node(self, session_id, node_name, sub_session_id=None):
        return None

    def _read_metadata(self, session_id):
        return {"tasks": {}}

    def save_bisect_job(self, session_id, job_id, data):
        pass

    def load_bisect_jobs(self, session_id):
        return {}

    def clear_bisect_job(self, session_id, job_id):
        pass


@pytest.mark.asyncio
async def test_bisect_filters_grayed_nodes_for_sub_session():
    """Bisect in a sub-session should only search non-grayed nodes."""
    svc = BisectService()
    graph = _make_graph(10)  # op_0 .. op_9

    # Gray out the first 5 ops — sub-model only has op_5..op_9
    grayed = [f"op_{i}" for i in range(5)]
    session_svc = FakeSessionService(grayed_nodes=grayed)

    req = BisectRequest(
        session_id="s1",
        threshold=0.999,
        sub_session_id="cut1",
    )

    queue = FakeQueue()
    job = await svc.start(
        request=req, graph_data=graph,
        queue_service=queue, session_service=session_svc,
        broadcast=noop_broadcast,
    )

    # 5 non-grayed nodes → ceil(log2(5)) = 3 total steps
    assert job.total_steps == math.ceil(math.log2(5))
    assert job.sub_session_id == "cut1"

    # Verify bisect only probes non-grayed nodes
    probed_nodes = set()
    while any(s.job.status == BisectStatus.RUNNING for s in svc._jobs.values()):
        await asyncio.sleep(0.01)
        if queue.created_tasks:
            task = queue.created_tasks[-1]
            probed_nodes.add(task.node_name)
            assert task.sub_session_id == "cut1"
            # All pass — bisect should converge to the last node
            result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                                cos=1.0, batch_id=task.batch_id)
            svc.on_task_complete(result)
            await asyncio.sleep(0.01)

    # No grayed node should have been probed
    for node in grayed:
        assert node not in probed_nodes


@pytest.mark.asyncio
async def test_bisect_sub_session_all_grayed_raises():
    """Bisect should raise when all nodes are grayed out in a sub-session."""
    svc = BisectService()
    graph = _make_graph(5)  # op_0 .. op_4

    grayed = [f"op_{i}" for i in range(5)]
    session_svc = FakeSessionService(grayed_nodes=grayed)

    req = BisectRequest(
        session_id="s1",
        threshold=0.999,
        sub_session_id="cut1",
    )

    with pytest.raises(ValueError, match="sub-session"):
        await svc.start(
            request=req, graph_data=graph,
            queue_service=FakeQueue(), session_service=session_svc,
            broadcast=noop_broadcast,
        )
