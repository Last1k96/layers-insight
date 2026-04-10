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
from backend.services.bisect_service import BisectService


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


def _make_diamond_graph() -> GraphData:
    """
    param -> A -> B -> D -> result_0
                  C ↗
    param -> A -> C
    """
    nodes = [
        GraphNode(id="param_0", name="param_0", type="Parameter"),
        GraphNode(id="A", name="A", type="Convolution"),
        GraphNode(id="B", name="B", type="Relu"),
        GraphNode(id="C", name="C", type="Relu"),
        GraphNode(id="D", name="D", type="Add"),
        GraphNode(id="result_0", name="result_0", type="Result"),
    ]
    edges = [
        GraphEdge(source="param_0", target="A"),
        GraphEdge(source="A", target="B"),
        GraphEdge(source="A", target="C"),
        GraphEdge(source="B", target="D"),
        GraphEdge(source="C", target="D"),
        GraphEdge(source="D", target="result_0"),
    ]
    return GraphData(nodes=nodes, edges=edges)


def _make_multi_output_graph() -> GraphData:
    """
    param -> A -> B -> result_0
    param -> A -> C -> result_1
    """
    nodes = [
        GraphNode(id="param_0", name="param_0", type="Parameter"),
        GraphNode(id="A", name="A", type="Convolution"),
        GraphNode(id="B", name="B", type="Relu"),
        GraphNode(id="C", name="C", type="Sigmoid"),
        GraphNode(id="result_0", name="result_0", type="Result"),
        GraphNode(id="result_1", name="result_1", type="Result"),
    ]
    edges = [
        GraphEdge(source="param_0", target="A"),
        GraphEdge(source="A", target="B"),
        GraphEdge(source="A", target="C"),
        GraphEdge(source="B", target="result_0"),
        GraphEdge(source="C", target="result_1"),
    ]
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
        self.completed_tasks: list[InferenceTask] = []

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

    async def add_completed_task(self, task):
        self.completed_tasks.append(task)
        return task

    def get_all_tasks(self, session_id=None):
        return []

    async def cancel(self, task_id):
        return True


class FakeSession:
    """Mock session service that returns pre-configured cached results."""
    def __init__(self, cached_nodes: dict[str, dict] | None = None):
        self._cached = cached_nodes or {}

    def find_task_for_node(self, session_id, node_name, sub_session_id=None):
        data = self._cached.get(node_name)
        return data.get("task_id") if data else None

    def _read_metadata(self, session_id):
        tasks = {}
        for node_name, data in self._cached.items():
            tasks[data["task_id"]] = data
        return {"tasks": tasks}

    def save_task_result(self, session_id, task_id, task_data, artifacts_dir=None, sub_session_id=None):
        pass

    def save_bisect_job(self, session_id, job_id, data):
        pass


async def noop_broadcast(sid, msg):
    pass


# ---------------------------------------------------------------------------
# Accuracy-drop & compilation-failure (linear graph)
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

    assert job.status == BisectStatus.DONE
    assert job.found_node == "op_5"


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

    assert job.status == BisectStatus.DONE
    assert job.found_node == "op_3"


# ---------------------------------------------------------------------------
# Stop, pause, resume
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Multiple concurrent jobs
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Graph-aware: diamond graph
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bisect_diamond_graph():
    """Graph-aware bisect on a diamond graph correctly identifies the faulty branch."""
    svc = BisectService()
    graph = _make_diamond_graph()
    # param -> A -> B -> D -> result_0
    #               C ↗

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

    # Simulate: C is the faulty node. A, B pass. C and D fail.
    while any(s.job.status == BisectStatus.RUNNING for s in svc._jobs.values()):
        await asyncio.sleep(0.01)
        if queue.created_tasks:
            task = queue.created_tasks[-1]
            if task.node_name in ("A", "B"):
                result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                                    cos=1.0, batch_id=task.batch_id)
            else:  # C, D
                result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                                    cos=0.5, batch_id=task.batch_id)
            svc.on_task_complete(result)
            await asyncio.sleep(0.01)

    assert job.status == BisectStatus.DONE
    assert job.found_node == "C"


# ---------------------------------------------------------------------------
# Per-output bisect
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bisect_per_output_early_termination():
    """Per-output bisect finishes in 1 inference when the output is correct."""
    svc = BisectService()
    graph = _make_multi_output_graph()
    # param -> A -> B -> result_0
    # param -> A -> C -> result_1

    req = BisectRequest(
        session_id="s1",
        metric=BisectMetric.COSINE_SIMILARITY,
        threshold=0.999,
        output_node="result_0",  # check output 0
    )

    queue = FakeQueue()
    job = await svc.start(
        request=req, graph_data=graph,
        queue_service=queue, session_service=None,
        broadcast=noop_broadcast,
    )

    # The first inference should be on B (predecessor of result_0)
    # If it passes, the job should be done
    while any(s.job.status == BisectStatus.RUNNING for s in svc._jobs.values()):
        await asyncio.sleep(0.01)
        if queue.created_tasks:
            task = queue.created_tasks[-1]
            assert task.node_name == "B"
            result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                                cos=1.0, batch_id=task.batch_id)
            svc.on_task_complete(result)
            await asyncio.sleep(0.01)

    assert job.status == BisectStatus.DONE
    assert job.found_node is None  # no problem found
    assert job.step == 1  # only 1 inference


@pytest.mark.asyncio
async def test_bisect_per_output_finds_fault():
    """Per-output bisect narrows down to the faulty node when output fails."""
    svc = BisectService()
    graph = _make_multi_output_graph()
    # param -> A -> B -> result_0
    # param -> A -> C -> result_1

    req = BisectRequest(
        session_id="s1",
        metric=BisectMetric.COSINE_SIMILARITY,
        threshold=0.999,
        output_node="result_1",  # check output 1
    )

    queue = FakeQueue()
    job = await svc.start(
        request=req, graph_data=graph,
        queue_service=queue, session_service=None,
        broadcast=noop_broadcast,
    )

    # C fails (predecessor of result_1). A passes. So C is the faulty node.
    while any(s.job.status == BisectStatus.RUNNING for s in svc._jobs.values()):
        await asyncio.sleep(0.01)
        if queue.created_tasks:
            task = queue.created_tasks[-1]
            if task.node_name == "A":
                result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                                    cos=1.0, batch_id=task.batch_id)
            else:
                result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                                    cos=0.5, batch_id=task.batch_id)
            svc.on_task_complete(result)
            await asyncio.sleep(0.01)

    assert job.status == BisectStatus.DONE
    assert job.found_node == "C"


# ---------------------------------------------------------------------------
# start_all_outputs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bisect_all_outputs():
    """start_all_outputs creates one job per output."""
    svc = BisectService()
    graph = _make_multi_output_graph()

    req = BisectRequest(
        session_id="s1",
        metric=BisectMetric.COSINE_SIMILARITY,
        threshold=0.999,
    )

    queue = FakeQueue()
    jobs = await svc.start_all_outputs(
        request=req, graph_data=graph,
        queue_service=queue, session_service=None,
        broadcast=noop_broadcast,
    )

    assert len(jobs) == 2
    output_nodes = {j.output_node for j in jobs}
    assert output_nodes == {"result_0", "result_1"}
    assert all(j.status == BisectStatus.RUNNING for j in jobs)

    # Clean up
    for j in jobs:
        await svc.stop(j.job_id)


@pytest.mark.asyncio
async def test_bisect_all_outputs_mixed_results():
    """Per-output bisect: one output passes (1 inference), the other finds a fault."""
    svc = BisectService()
    graph = _make_multi_output_graph()
    # param -> A -> B -> result_0  (B will pass)
    # param -> A -> C -> result_1  (C will fail, A will pass → C is faulty)

    req = BisectRequest(
        session_id="s1",
        metric=BisectMetric.COSINE_SIMILARITY,
        threshold=0.999,
    )

    queue = FakeQueue()
    jobs = await svc.start_all_outputs(
        request=req, graph_data=graph,
        queue_service=queue, session_service=None,
        broadcast=noop_broadcast,
    )

    # Drive both jobs to completion — process ALL unprocessed tasks each iteration
    processed: set[int] = set()
    while any(s.job.status == BisectStatus.RUNNING for s in svc._jobs.values()):
        await asyncio.sleep(0.01)
        for idx, task in enumerate(queue.created_tasks):
            if idx in processed:
                continue
            processed.add(idx)
            if task.node_name in ("A", "B"):
                result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                                    cos=1.0, batch_id=task.batch_id)
            else:  # C
                result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                                    cos=0.5, batch_id=task.batch_id)
            svc.on_task_complete(result)
        await asyncio.sleep(0.01)

    # Find the jobs by output_node (use the original references — jobs are removed from _jobs on completion)
    job_map = {j.output_node: j for j in jobs}

    # result_0's predecessor B passes → output OK
    assert job_map["result_0"].status == BisectStatus.DONE
    assert job_map["result_0"].found_node is None

    # result_1's predecessor C fails, A passes → C is faulty
    assert job_map["result_1"].status == BisectStatus.DONE
    assert job_map["result_1"].found_node == "C"


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


@pytest.mark.asyncio
async def test_bisect_auto_no_graph(async_client, test_app, test_session):
    """POST /api/inference/bisect/auto returns 404 when graph is not loaded."""
    resp = await async_client.post("/api/inference/bisect/auto", json={
        "session_id": test_session.id,
        "threshold": 0.999,
    })
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Cached/reused node tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bisect_with_all_cached_nodes():
    """Bisect converges when ALL nodes are already cached (reused path)."""
    svc = BisectService()
    graph = _make_graph(10)  # param -> op_0..op_9 -> result

    # Pre-cache all nodes: op_0..op_4 pass (cos=1.0), op_5..op_9 fail (cos=0.5)
    cached = {}
    for i in range(10):
        name = f"op_{i}"
        cos = 1.0 if i < 5 else 0.5
        cached[name] = {
            "task_id": f"orig_{name}",
            "session_id": "s1",
            "node_id": name,
            "node_name": name,
            "node_type": "Convolution",
            "status": "success",
            "metrics": {"cosine_similarity": cos, "mse": 0.0, "max_abs_diff": 0.0},
        }
    session = FakeSession(cached)

    req = BisectRequest(
        session_id="s1",
        metric=BisectMetric.COSINE_SIMILARITY,
        threshold=0.999,
        search_for=BisectSearchFor.ACCURACY_DROP,
    )

    queue = FakeQueue()
    job = await svc.start(
        request=req, graph_data=graph,
        queue_service=queue, session_service=session,
        broadcast=noop_broadcast,
    )
    assert job.status == BisectStatus.RUNNING

    # All nodes are cached — loop should complete without any enqueued tasks
    for _ in range(200):
        await asyncio.sleep(0.01)
        if not any(s.job.status == BisectStatus.RUNNING for s in svc._jobs.values()):
            break
    else:
        pytest.fail("Bisect did not converge within timeout (all cached)")

    assert len(queue.created_tasks) == 0, "No tasks should be enqueued for execution"
    assert len(queue.completed_tasks) > 0, "Reused tasks should be registered"
    for t in queue.completed_tasks:
        assert t.reused is True

    assert job.status == BisectStatus.DONE
    assert job.found_node == "op_5"


@pytest.mark.asyncio
async def test_bisect_with_some_cached_nodes():
    """Bisect works with a mix of cached and fresh nodes."""
    svc = BisectService()
    graph = _make_graph(10)

    # Only cache op_0..op_4 (the passing ones)
    cached = {}
    for i in range(5):
        name = f"op_{i}"
        cached[name] = {
            "task_id": f"orig_{name}",
            "session_id": "s1",
            "node_id": name,
            "node_name": name,
            "node_type": "Convolution",
            "status": "success",
            "metrics": {"cosine_similarity": 1.0, "mse": 0.0, "max_abs_diff": 0.0},
        }
    session = FakeSession(cached)

    req = BisectRequest(
        session_id="s1",
        metric=BisectMetric.COSINE_SIMILARITY,
        threshold=0.999,
        search_for=BisectSearchFor.ACCURACY_DROP,
    )

    queue = FakeQueue()
    job = await svc.start(
        request=req, graph_data=graph,
        queue_service=queue, session_service=session,
        broadcast=noop_broadcast,
    )

    # Feed results for non-cached nodes (op_5..op_9 fail)
    while any(s.job.status == BisectStatus.RUNNING for s in svc._jobs.values()):
        await asyncio.sleep(0.01)
        if queue.created_tasks:
            task = queue.created_tasks[-1]
            node_idx = int(task.node_name.split("_")[1])
            cos = 1.0 if node_idx < 5 else 0.5
            result = _make_task(task.task_id, task.node_name, TaskStatus.SUCCESS,
                                cos=cos, batch_id=task.batch_id)
            svc.on_task_complete(result)
            await asyncio.sleep(0.01)

    assert job.status == BisectStatus.DONE
    assert job.found_node == "op_5"
    assert len(queue.completed_tasks) > 0, "Some reused tasks should exist"
