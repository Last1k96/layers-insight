"""Bisection search service — binary search for accuracy drop or compilation failure."""
from __future__ import annotations

import asyncio
import math
import uuid
from typing import Any, Optional

from backend.schemas.bisect import (
    BisectJobInfo,
    BisectMetric,
    BisectProgress,
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


class BisectService:
    """Manages a single bisection search at a time."""

    def __init__(self):
        self._progress = BisectProgress()
        self._job: Optional[BisectJobInfo] = None
        self._task: Optional[asyncio.Task] = None
        self._cancel_event = asyncio.Event()
        self._step_done = asyncio.Event()
        self._step_result: Optional[InferenceTask] = None

        # Persistent loop state for pause/resume
        self._lo: int = 0
        self._hi: int = 0
        self._step: int = 0
        self._pending_task_id: Optional[str] = None  # task we're waiting on (survives pause)
        self._nodes: list[str] = []
        self._node_map: dict[str, Any] = {}
        self._request: Any = None
        self._graph_data: Optional[GraphData] = None
        self._queue_service: Any = None
        self._session_service: Any = None
        self._broadcast: Any = None

    @property
    def progress(self) -> BisectProgress:
        return self._progress

    @property
    def job(self) -> Optional[BisectJobInfo]:
        return self._job

    @property
    def is_running(self) -> bool:
        return self._job is not None and self._job.status == BisectStatus.RUNNING

    @property
    def is_paused(self) -> bool:
        return self._job is not None and self._job.status == BisectStatus.PAUSED

    @property
    def is_active(self) -> bool:
        """True if a bisect job exists and is running or paused."""
        return self._job is not None and self._job.status in (
            BisectStatus.RUNNING, BisectStatus.PAUSED
        )

    def on_task_complete(self, task: InferenceTask) -> None:
        """Called when an inference task finishes — unblocks the bisect loop if relevant."""
        if task.batch_id == "bisect":
            self._step_result = task
            self._step_done.set()

    async def start(
        self,
        request: Any,
        graph_data: GraphData,
        queue_service: Any,
        session_service: Any,
        broadcast: Any,
    ) -> BisectJobInfo:
        """Start a bisection search. Returns job info."""
        if self.is_active:
            raise RuntimeError("A bisection is already running")

        nodes = _nodes_between(graph_data, request.start_node, request.end_node)
        if len(nodes) < 2:
            raise ValueError("Need at least 2 nodes in range for bisection")

        total_steps = max(1, math.ceil(math.log2(len(nodes))))
        job_id = str(uuid.uuid4())[:8]

        # Store persistent state
        self._nodes = nodes
        self._node_map = {n.id: n for n in graph_data.nodes}
        self._lo = 0
        self._hi = len(nodes) - 1
        self._step = 0
        self._request = request
        self._graph_data = graph_data
        self._queue_service = queue_service
        self._session_service = session_service
        self._broadcast = broadcast

        self._cancel_event.clear()

        self._job = BisectJobInfo(
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

        # Also keep the old progress for backward compat
        self._progress = BisectProgress(
            status=BisectStatus.RUNNING,
            session_id=request.session_id,
            search_for=request.search_for,
            metric=request.metric,
            threshold=request.threshold,
            range_start=nodes[0],
            range_end=nodes[-1],
            step=0,
            total_steps=total_steps,
        )

        self._task = asyncio.create_task(self._run_loop())
        return self._job

    async def pause(self) -> Optional[BisectJobInfo]:
        """Pause the bisect, preserving state for resume."""
        if not self.is_running:
            return self._job

        self._cancel_event.set()
        self._step_done.set()  # Unblock any waiting step

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None

        if self._job:
            self._job.status = BisectStatus.PAUSED
        self._progress.status = BisectStatus.PAUSED

        await self._broadcast_job_status()
        return self._job

    async def resume(self) -> Optional[BisectJobInfo]:
        """Resume a paused bisect from where it left off."""
        if not self.is_paused:
            return self._job

        self._cancel_event.clear()

        if self._job:
            self._job.status = BisectStatus.RUNNING
        self._progress.status = BisectStatus.RUNNING

        self._task = asyncio.create_task(self._run_loop())
        await self._broadcast_job_status()
        return self._job

    async def stop(self) -> Optional[BisectJobInfo]:
        """Stop the bisect entirely and clean up.

        Also cancels any waiting bisect child tasks still in the queue
        to prevent orphan tasks.
        """
        if self._job is None:
            return None

        if self.is_running:
            self._cancel_event.set()
            self._step_done.set()
            if self._task:
                try:
                    await asyncio.wait_for(self._task, timeout=5)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    self._task.cancel()
                self._task = None

        # Cancel any waiting bisect child tasks left in the queue
        if self._queue_service:
            for task in list(self._queue_service.get_all_tasks()):
                if task.batch_id == "bisect" and task.status == TaskStatus.WAITING:
                    await self._queue_service.cancel(task.task_id)

        job = self._job
        if job:
            job.status = BisectStatus.STOPPED

        self._progress.status = BisectStatus.STOPPED

        await self._broadcast_job_status()
        self._reset_state()
        return job

    def _reset_state(self) -> None:
        """Clear all bisect state."""
        self._job = None
        self._task = None
        self._nodes = []
        self._node_map = {}
        self._lo = 0
        self._hi = 0
        self._step = 0
        self._pending_task_id = None
        self._request = None
        self._graph_data = None
        self._queue_service = None
        self._session_service = None
        self._broadcast = None
        self._progress = BisectProgress()

    def _evaluate_step(
        self,
        task: InferenceTask,
        search_for: BisectSearchFor,
        metric: BisectMetric,
        threshold: float,
    ) -> tuple[bool, Optional[float], Optional[str]]:
        """Evaluate whether a bisect step passed.

        Returns (passed, metric_value, error_string).
        """
        if search_for == BisectSearchFor.COMPILATION_FAILURE:
            if task.status == TaskStatus.FAILED:
                error_detail = task.error_detail or ""
                return False, None, error_detail
            return True, None, None

        # accuracy_drop
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

    def _build_task_from_metadata(self, task_data: dict) -> InferenceTask:
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

    async def _broadcast_job_status(self) -> None:
        """Broadcast current job status via WebSocket."""
        if self._job and self._broadcast and self._job.session_id:
            await self._broadcast(self._job.session_id, {
                "type": "bisect_job_status",
                **self._job.model_dump(),
            })

    async def _run_loop(self) -> None:
        """The main bisection binary search loop. Reads from self._lo, self._hi, self._step.

        On resume after pause, if ``_pending_task_id`` is set, the loop skips
        creating a new task and waits for the already-enqueued one to finish.
        This prevents duplicate tasks when pause interrupts an executing step.
        """
        request = self._request
        lo = self._lo
        hi = self._hi
        step = self._step

        try:
            # ── Handle resume with a pending (interrupted) task ──
            if self._pending_task_id is not None:
                if self._step_result is None:
                    # Task hasn't completed yet — wait for it
                    self._step_done.clear()
                    await self._step_done.wait()

                    if self._cancel_event.is_set():
                        self._lo = lo
                        self._hi = hi
                        self._step = step
                        return

                result = self._step_result
                self._step_result = None
                self._pending_task_id = None

                if result is None:
                    if self._job:
                        self._job.status = BisectStatus.ERROR
                        self._job.error = "Inference result was lost"
                    self._progress.status = BisectStatus.ERROR
                    self._progress.error = "Inference result was lost"
                    await self._broadcast_job_status()
                    return

                mid = (lo + hi) // 2
                mid_node_id = self._nodes[mid]

                passed, metric_value, error = self._evaluate_step(
                    result, request.search_for, request.metric, request.threshold,
                )

                step_info = BisectStepInfo(
                    node_name=result.node_name,
                    node_id=mid_node_id,
                    task_id=result.task_id if result.task_id else None,
                    metric_value=metric_value,
                    passed=passed,
                    error=error,
                )
                self._progress.steps_history.append(step_info)

                if passed:
                    lo = mid + 1
                else:
                    hi = mid

            # ── Normal binary search loop ──
            while lo < hi:
                if self._cancel_event.is_set():
                    # Save state for resume
                    self._lo = lo
                    self._hi = hi
                    self._step = step
                    return

                mid = (lo + hi) // 2
                mid_node_id = self._nodes[mid]
                mid_node = self._node_map.get(mid_node_id)
                mid_name = mid_node.name if mid_node else mid_node_id
                mid_type = mid_node.type if mid_node else ""

                step += 1
                # Update job info
                if self._job:
                    self._job.step = step
                    self._job.current_node = mid_name

                self._progress.step = step
                self._progress.current_node = mid_name
                self._progress.range_start = self._nodes[lo]
                self._progress.range_end = self._nodes[hi]

                await self._broadcast_job_status()

                # Check if node was already inferred
                existing_task_id = None
                if self._session_service:
                    existing_task_id = self._session_service.find_task_for_node(
                        request.session_id, mid_name, request.sub_session_id
                    )

                result = None
                if existing_task_id:
                    # Use cached result — load from session metadata
                    meta = self._session_service._read_metadata(request.session_id)
                    task_data = meta.get("tasks", {}).get(existing_task_id)
                    if task_data:
                        result = self._build_task_from_metadata(task_data)
                        result.node_id = mid_node_id
                        result.node_name = mid_name

                if result is None:
                    # Need to run inference — enqueue child task
                    self._step_done.clear()
                    self._step_result = None

                    task = self._queue_service.create_task(
                        session_id=request.session_id,
                        node_id=mid_node_id,
                        node_name=mid_name,
                        node_type=mid_type,
                    )
                    if request.sub_session_id:
                        task.sub_session_id = request.sub_session_id
                    task.batch_id = "bisect"

                    self._pending_task_id = task.task_id
                    await self._queue_service.enqueue(task)

                    # Wait for the task to complete (or cancellation)
                    await self._step_done.wait()

                    if self._cancel_event.is_set():
                        self._lo = lo
                        self._hi = hi
                        self._step = step
                        return

                    self._pending_task_id = None
                    result = self._step_result
                    if result is None:
                        if self._job:
                            self._job.status = BisectStatus.ERROR
                            self._job.error = "Inference result was lost"
                        self._progress.status = BisectStatus.ERROR
                        self._progress.error = "Inference result was lost"
                        await self._broadcast_job_status()
                        return

                passed, metric_value, error = self._evaluate_step(
                    result, request.search_for, request.metric, request.threshold,
                )

                step_info = BisectStepInfo(
                    node_name=mid_name,
                    node_id=mid_node_id,
                    task_id=result.task_id if result.task_id else None,
                    metric_value=metric_value,
                    passed=passed,
                    error=error,
                )
                self._progress.steps_history.append(step_info)

                if passed:
                    lo = mid + 1
                else:
                    hi = mid

            # Save final state
            self._lo = lo
            self._hi = hi
            self._step = step

            # Bisection converged
            found_id = self._nodes[lo]
            found_node = self._node_map.get(found_id)
            found_name = found_node.name if found_node else found_id

            if self._job:
                self._job.status = BisectStatus.DONE
                self._job.found_node = found_name
                self._job.step = step

            self._progress.status = BisectStatus.DONE
            self._progress.found_node = found_name
            self._progress.range_start = found_name
            self._progress.range_end = found_name

            await self._broadcast_job_status()

        except asyncio.CancelledError:
            self._lo = lo
            self._hi = hi
            self._step = step
            if self._job:
                self._job.status = BisectStatus.PAUSED
            self._progress.status = BisectStatus.PAUSED
        except Exception as e:
            self._lo = lo
            self._hi = hi
            self._step = step
            if self._job:
                self._job.status = BisectStatus.ERROR
                self._job.error = str(e)
            self._progress.status = BisectStatus.ERROR
            self._progress.error = str(e)
            await self._broadcast_job_status()
