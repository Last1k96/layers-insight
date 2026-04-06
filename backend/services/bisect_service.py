"""Bisection search service — binary search for accuracy drop or compilation failure."""
from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.schemas.bisect import (
    BisectJobInfo,
    BisectMetric,
    BisectSearchFor,
    BisectStatus,
    BisectStepInfo,
)
from backend.schemas.graph import GraphData
from backend.schemas.inference import AccuracyMetrics, InferenceTask, TaskStatus


def _topo_order(graph_data: GraphData) -> list[str]:
    """Return node IDs in topological order (already stored by OV's get_ordered_ops)."""
    return [n.id for n in graph_data.nodes]


def _nodes_between(
    graph_data: GraphData,
    start_node: Optional[str],
    end_node: Optional[str],
) -> list[str]:
    """Return node IDs between start and end (inclusive) in topological order.

    Skips Parameter and Result nodes.
    """
    topo = _topo_order(graph_data)
    node_types = {n.id: n.type for n in graph_data.nodes}

    if start_node and start_node in topo:
        start_idx = topo.index(start_node)
    else:
        start_idx = 0

    if end_node and end_node in topo:
        end_idx = topo.index(end_node)
    else:
        end_idx = len(topo) - 1

    return [
        nid for nid in topo[start_idx : end_idx + 1]
        if node_types.get(nid) not in ("Parameter", "Result")
    ]


@dataclass
class _JobState:
    """Per-job state for a single bisection search."""
    job: BisectJobInfo
    task: Optional[asyncio.Task] = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    step_done: asyncio.Event = field(default_factory=asyncio.Event)
    step_result: Optional[InferenceTask] = None
    pending_task_id: Optional[str] = None
    lo: int = 0
    hi: int = 0
    step: int = 0
    nodes: list[str] = field(default_factory=list)
    node_map: dict[str, Any] = field(default_factory=dict)
    request: Any = None
    graph_data: Optional[GraphData] = None
    steps_history: list[BisectStepInfo] = field(default_factory=list)


class BisectService:
    """Manages multiple concurrent bisection searches."""

    def __init__(self):
        self._jobs: dict[str, _JobState] = {}
        # Shared service references (set on first start, never cleared)
        self._queue_service: Any = None
        self._session_service: Any = None
        self._broadcast: Any = None

    # ── Properties ──

    @property
    def has_active_jobs(self) -> bool:
        return any(
            s.job.status in (BisectStatus.RUNNING, BisectStatus.PAUSED)
            for s in self._jobs.values()
        )

    @property
    def has_running_jobs(self) -> bool:
        return any(s.job.status == BisectStatus.RUNNING for s in self._jobs.values())

    @property
    def has_paused_jobs(self) -> bool:
        return any(s.job.status == BisectStatus.PAUSED for s in self._jobs.values())

    def get_jobs(self) -> list[BisectJobInfo]:
        """Return info for all tracked jobs."""
        return [s.job for s in self._jobs.values()]

    def get_job(self, job_id: str) -> Optional[BisectJobInfo]:
        state = self._jobs.get(job_id)
        return state.job if state else None

    # ── Task completion callback ──

    def on_task_complete(self, task: InferenceTask) -> None:
        """Called when an inference task finishes — routes to the correct job."""
        if not task.batch_id or not task.batch_id.startswith("bisect:"):
            return
        job_id = task.batch_id.split(":", 1)[1]
        state = self._jobs.get(job_id)
        if state is None:
            return  # Job was already stopped/removed
        state.step_result = task
        state.step_done.set()

    # ── Job lifecycle ──

    async def start(
        self,
        request: Any,
        graph_data: GraphData,
        queue_service: Any,
        session_service: Any,
        broadcast: Any,
    ) -> BisectJobInfo:
        """Start a new bisection search. Returns job info."""
        nodes = _nodes_between(graph_data, request.start_node, request.end_node)

        # When running in a sub-session, exclude grayed-out nodes (not in the cut model)
        if request.sub_session_id and session_service:
            sub_meta = session_service.get_sub_session_meta(
                request.session_id, request.sub_session_id
            )
            if sub_meta:
                grayed = set(sub_meta.get("grayed_nodes", []))
                nodes = [nid for nid in nodes if nid not in grayed]

        if len(nodes) < 2:
            detail = "Need at least 2 nodes in range for bisection"
            if request.sub_session_id:
                detail += " (after excluding nodes outside the sub-session model)"
            raise ValueError(detail)

        # Store shared refs (idempotent)
        self._queue_service = queue_service
        self._session_service = session_service
        self._broadcast = broadcast

        total_steps = max(1, math.ceil(math.log2(len(nodes))))
        job_id = str(uuid.uuid4())[:8]

        job = BisectJobInfo(
            job_id=job_id,
            session_id=request.session_id,
            status=BisectStatus.RUNNING,
            search_for=request.search_for,
            metric=request.metric,
            threshold=request.threshold,
            step=0,
            total_steps=total_steps,
            sub_session_id=request.sub_session_id,
        )

        state = _JobState(
            job=job,
            lo=0,
            hi=len(nodes) - 1,
            step=0,
            nodes=nodes,
            node_map={n.id: n for n in graph_data.nodes},
            request=request,
            graph_data=graph_data,
        )
        self._jobs[job_id] = state
        state.task = asyncio.create_task(self._run_loop(job_id))
        return job

    async def stop(self, job_id: str) -> Optional[BisectJobInfo]:
        """Stop a specific bisect job and clean up its child tasks."""
        state = self._jobs.get(job_id)
        if state is None:
            return None

        if state.job.status == BisectStatus.RUNNING:
            state.cancel_event.set()
            state.step_done.set()
            if state.task:
                try:
                    await asyncio.wait_for(state.task, timeout=5)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    state.task.cancel()

        # Cancel any waiting child tasks for this job
        batch_id = f"bisect:{job_id}"
        if self._queue_service:
            for task in list(self._queue_service.get_all_tasks()):
                if task.batch_id == batch_id and task.status == TaskStatus.WAITING:
                    await self._queue_service.cancel(task.task_id)

        state.job.status = BisectStatus.STOPPED
        await self._broadcast_job_status(state)
        del self._jobs[job_id]
        return state.job

    async def pause_all(self) -> None:
        """Pause all running bisect jobs (called from queue pause)."""
        for state in list(self._jobs.values()):
            if state.job.status == BisectStatus.RUNNING:
                await self._pause_job(state)

    async def resume_all(self) -> None:
        """Resume all paused bisect jobs (called from queue resume)."""
        for state in list(self._jobs.values()):
            if state.job.status == BisectStatus.PAUSED:
                await self._resume_job(state)

    async def _pause_job(self, state: _JobState) -> None:
        state.cancel_event.set()
        state.step_done.set()
        if state.task:
            try:
                await asyncio.wait_for(state.task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                state.task.cancel()
            state.task = None
        state.job.status = BisectStatus.PAUSED
        await self._broadcast_job_status(state)

    async def _resume_job(self, state: _JobState) -> None:
        state.cancel_event.clear()
        state.job.status = BisectStatus.RUNNING
        state.task = asyncio.create_task(self._run_loop(state.job.job_id))
        await self._broadcast_job_status(state)

    # ── Helpers ──

    @staticmethod
    def _evaluate_step(
        task: InferenceTask,
        search_for: BisectSearchFor,
        metric: BisectMetric,
        threshold: float,
    ) -> tuple[bool, Optional[float], Optional[str]]:
        """Evaluate whether a bisect step passed."""
        if search_for == BisectSearchFor.COMPILATION_FAILURE:
            if task.status == TaskStatus.FAILED:
                return False, None, task.error_detail or ""
            return True, None, None

        if task.status == TaskStatus.FAILED:
            return False, None, task.error_detail

        if not task.metrics:
            return False, None, "No metrics available"

        if metric == BisectMetric.COSINE_SIMILARITY:
            val = task.metrics.cosine_similarity
            passed = val >= threshold
        elif metric == BisectMetric.MSE:
            val = task.metrics.mse
            passed = val <= threshold
        elif metric == BisectMetric.MAX_ABS_DIFF:
            val = task.metrics.max_abs_diff
            passed = val <= threshold
        else:
            val = 0.0
            passed = False

        return passed, val, None

    @staticmethod
    def _build_task_from_metadata(task_data: dict) -> InferenceTask:
        """Build a minimal InferenceTask from session metadata for skip evaluation."""
        metrics = None
        if task_data.get("metrics"):
            m = task_data["metrics"]
            metrics = AccuracyMetrics(
                mse=m.get("mse", 0),
                max_abs_diff=m.get("max_abs_diff", 0),
                cosine_similarity=m.get("cosine_similarity", 0),
            )
        status_str = task_data.get("status", "failed")
        status = TaskStatus.SUCCESS if status_str == "success" else TaskStatus.FAILED
        return InferenceTask(
            task_id=task_data.get("task_id", ""),
            session_id=task_data.get("session_id", ""),
            node_id=task_data.get("node_id", ""),
            node_name=task_data.get("node_name", ""),
            node_type=task_data.get("node_type", ""),
            status=status,
            metrics=metrics,
            error_detail=task_data.get("error_detail"),
        )

    def _persist_job(self, state: _JobState) -> None:
        """Save bisect job to session metadata so it survives backend restart."""
        if self._session_service:
            try:
                self._session_service.save_bisect_job(
                    state.job.session_id,
                    state.job.job_id,
                    state.job.model_dump(mode="json"),
                )
            except Exception:
                pass

    async def _broadcast_job_status(self, state: _JobState) -> None:
        """Broadcast job status via WebSocket."""
        if self._broadcast and state.job.session_id:
            await self._broadcast(state.job.session_id, {
                "type": "bisect_job_status",
                **state.job.model_dump(),
            })

    # ── Main loop ──

    async def _run_loop(self, job_id: str) -> None:
        """Binary search loop for a specific job."""
        state = self._jobs.get(job_id)
        if state is None:
            return

        request = state.request
        lo = state.lo
        hi = state.hi
        step = state.step
        batch_id = f"bisect:{job_id}"

        try:
            # ── Handle resume with a pending (interrupted) task ──
            if state.pending_task_id is not None:
                if state.step_result is None:
                    state.step_done.clear()
                    await state.step_done.wait()

                    if state.cancel_event.is_set():
                        state.lo = lo
                        state.hi = hi
                        state.step = step
                        return

                result = state.step_result
                state.step_result = None
                state.pending_task_id = None

                if result is None:
                    state.job.status = BisectStatus.ERROR
                    state.job.error = "Inference result was lost"
                    await self._broadcast_job_status(state)
                    return

                mid = (lo + hi) // 2
                mid_node_id = state.nodes[mid]

                passed, metric_value, error = self._evaluate_step(
                    result, request.search_for, request.metric, request.threshold,
                )
                state.steps_history.append(BisectStepInfo(
                    node_name=result.node_name,
                    node_id=mid_node_id,
                    task_id=result.task_id if result.task_id else None,
                    metric_value=metric_value,
                    passed=passed,
                    error=error,
                ))

                if passed:
                    lo = mid + 1
                else:
                    hi = mid

            # ── Normal binary search loop ──
            while lo < hi:
                if state.cancel_event.is_set():
                    state.lo = lo
                    state.hi = hi
                    state.step = step
                    return

                mid = (lo + hi) // 2
                mid_node_id = state.nodes[mid]
                mid_node = state.node_map.get(mid_node_id)
                mid_name = mid_node.name if mid_node else mid_node_id
                mid_type = mid_node.type if mid_node else ""

                step += 1
                state.job.step = step
                state.job.current_node = mid_name
                await self._broadcast_job_status(state)

                # Check if node was already inferred
                existing_task_id = None
                if self._session_service:
                    existing_task_id = self._session_service.find_task_for_node(
                        request.session_id, mid_name, request.sub_session_id
                    )

                result = None
                if existing_task_id:
                    meta = self._session_service._read_metadata(request.session_id)
                    task_data = meta.get("tasks", {}).get(existing_task_id)
                    if task_data:
                        result = self._build_task_from_metadata(task_data)
                        result.node_id = mid_node_id
                        result.node_name = mid_name

                if result is None:
                    # Enqueue child task
                    state.step_done.clear()
                    state.step_result = None

                    task = self._queue_service.create_task(
                        session_id=request.session_id,
                        node_id=mid_node_id,
                        node_name=mid_name,
                        node_type=mid_type,
                    )
                    if request.sub_session_id:
                        task.sub_session_id = request.sub_session_id
                    task.batch_id = batch_id

                    state.pending_task_id = task.task_id
                    await self._queue_service.enqueue(task)

                    await state.step_done.wait()

                    if state.cancel_event.is_set():
                        state.lo = lo
                        state.hi = hi
                        state.step = step
                        return

                    state.pending_task_id = None
                    result = state.step_result
                    if result is None:
                        state.job.status = BisectStatus.ERROR
                        state.job.error = "Inference result was lost"
                        await self._broadcast_job_status(state)
                        return

                passed, metric_value, error = self._evaluate_step(
                    result, request.search_for, request.metric, request.threshold,
                )
                state.steps_history.append(BisectStepInfo(
                    node_name=mid_name,
                    node_id=mid_node_id,
                    task_id=result.task_id if result.task_id else None,
                    metric_value=metric_value,
                    passed=passed,
                    error=error,
                ))

                if passed:
                    lo = mid + 1
                else:
                    hi = mid

            # Save final state
            state.lo = lo
            state.hi = hi
            state.step = step

            # Bisection converged
            found_id = state.nodes[lo]
            found_node = state.node_map.get(found_id)
            found_name = found_node.name if found_node else found_id

            state.job.status = BisectStatus.DONE
            state.job.found_node = found_name
            state.job.step = step

            await self._broadcast_job_status(state)
            self._persist_job(state)

        except asyncio.CancelledError:
            state.lo = lo
            state.hi = hi
            state.step = step
            state.job.status = BisectStatus.PAUSED
        except Exception as e:
            state.lo = lo
            state.hi = hi
            state.step = step
            state.job.status = BisectStatus.ERROR
            state.job.error = str(e)
            await self._broadcast_job_status(state)
            self._persist_job(state)
