"""Bisection search service — graph-aware binary search for accuracy drop or compilation failure."""
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
from backend.utils.graph_utils import (
    build_reverse_adj,
    find_best_bisect_point,
    find_output_nodes,
    get_ancestors_in_set,
)


def _topo_order(graph_data: GraphData) -> list[str]:
    """Return node IDs in topological order (already stored by OV's get_ordered_ops)."""
    return [n.id for n in graph_data.nodes]


def _compute_search_candidates(
    graph_data: GraphData,
    reverse_adj: dict[str, set[str]],
    output_node: Optional[str],
    start_node: Optional[str],
    end_node: Optional[str],
) -> set[str]:
    """Compute the set of candidate nodes to bisect.

    - output_node: ancestors of this Result node's predecessor (per-output mode)
    - end_node: ancestors of this node (legacy from-node mode)
    - neither: all non-Parameter/Result nodes
    """
    node_types = {n.id: n.type for n in graph_data.nodes}
    skip_types = {"Parameter", "Result"}

    if output_node:
        # Find predecessor of the Result node
        preds = reverse_adj.get(output_node, set())
        if not preds:
            raise ValueError(f"Result node {output_node} has no predecessor")
        pred = next(iter(preds))
        # Get all ancestors of the predecessor (including itself)
        all_ancestors = get_ancestors_in_set(
            pred,
            {n.id for n in graph_data.nodes},
            reverse_adj,
        )
        return {nid for nid in all_ancestors if node_types.get(nid) not in skip_types}

    if end_node:
        all_node_ids = {n.id for n in graph_data.nodes}
        ancestors = get_ancestors_in_set(end_node, all_node_ids, reverse_adj)
        candidates = {nid for nid in ancestors if node_types.get(nid) not in skip_types}
        if start_node:
            # Also restrict to descendants of start_node
            topo = _topo_order(graph_data)
            if start_node in topo:
                start_idx = topo.index(start_node)
                allowed = set(topo[start_idx:])
                candidates = candidates & allowed
        return candidates

    # Full model: all non-Parameter/Result nodes
    return {n.id for n in graph_data.nodes if n.type not in skip_types}


@dataclass
class _JobState:
    """Per-job state for a single bisection search."""
    job: BisectJobInfo
    task: Optional[asyncio.Task] = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    step_done: asyncio.Event = field(default_factory=asyncio.Event)
    step_result: Optional[InferenceTask] = None
    pending_task_id: Optional[str] = None
    step: int = 0
    candidates: set[str] = field(default_factory=set)
    reverse_adj: dict[str, set[str]] = field(default_factory=dict)
    topo_order: list[str] = field(default_factory=list)
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

    def get_jobs(self, session_id: Optional[str] = None) -> list[BisectJobInfo]:
        """Return info for tracked jobs, optionally filtered by session."""
        jobs = [s.job for s in self._jobs.values()]
        if session_id:
            jobs = [j for j in jobs if j.session_id == session_id]
        return jobs

    def get_job(self, job_id: str) -> Optional[BisectJobInfo]:
        state = self._jobs.get(job_id)
        return state.job if state else None

    # ── Task completion callback ──

    def on_task_complete(self, task: InferenceTask) -> None:
        """Called when an inference task finishes — routes to the correct job."""
        if not task.batch_id or not task.batch_id.startswith("bisect:"):
            return
        if task.reused:
            return  # Reused tasks are handled synchronously in _infer_node
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
        reverse_adj = build_reverse_adj(graph_data.edges)
        candidates = _compute_search_candidates(
            graph_data, reverse_adj,
            request.output_node, request.start_node, request.end_node,
        )

        # When running in a sub-session, exclude grayed-out nodes
        if request.sub_session_id and session_service:
            sub_meta = session_service.get_sub_session_meta(
                request.session_id, request.sub_session_id
            )
            if sub_meta:
                grayed = set(sub_meta.get("grayed_nodes", []))
                candidates = candidates - grayed

        if len(candidates) < 2:
            detail = "Need at least 2 nodes in range for bisection"
            if request.sub_session_id:
                detail += " (after excluding nodes outside the sub-session model)"
            raise ValueError(detail)

        # Store shared refs (idempotent)
        self._queue_service = queue_service
        self._session_service = session_service
        self._broadcast = broadcast

        topo = _topo_order(graph_data)
        topo_filtered = [n for n in topo if n in candidates]
        total_steps = max(1, math.ceil(math.log2(len(candidates))))
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
            output_node=request.output_node,
        )

        state = _JobState(
            job=job,
            step=0,
            candidates=set(candidates),
            reverse_adj=reverse_adj,
            topo_order=topo_filtered,
            node_map={n.id: n for n in graph_data.nodes},
            request=request,
            graph_data=graph_data,
        )
        self._jobs[job_id] = state
        state.task = asyncio.create_task(self._run_loop(job_id))
        return job

    async def start_all_outputs(
        self,
        request: Any,
        graph_data: GraphData,
        queue_service: Any,
        session_service: Any,
        broadcast: Any,
    ) -> list[BisectJobInfo]:
        """Start one bisect job per model output. Returns list of job infos."""
        outputs = find_output_nodes(graph_data)
        if not outputs:
            raise ValueError("No output (Result) nodes found in graph")

        jobs: list[BisectJobInfo] = []
        for result_id, _pred_id in outputs:
            # Clone request with output_node set
            req_data = request.model_dump()
            req_data["output_node"] = result_id
            from backend.schemas.bisect import BisectRequest
            per_output_req = BisectRequest(**req_data)

            job = await self.start(
                request=per_output_req,
                graph_data=graph_data,
                queue_service=queue_service,
                session_service=session_service,
                broadcast=broadcast,
            )
            jobs.append(job)
        return jobs

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

    async def shutdown(self) -> None:
        """Cancel all running bisect tasks for clean shutdown."""
        for state in list(self._jobs.values()):
            state.cancel_event.set()
            state.step_done.set()
            if state.task:
                state.task.cancel()
                try:
                    await asyncio.wait_for(state.task, timeout=2)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        self._jobs.clear()

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

    # ── Inference helper ──

    async def _infer_node(
        self,
        state: _JobState,
        node_id: str,
        node_name: str,
        node_type: str,
        batch_id: str,
    ) -> Optional[InferenceTask]:
        """Check cache or enqueue inference for a node. Returns result task."""
        request = state.request

        # Check if node was already inferred
        if self._session_service:
            existing_task_id = self._session_service.find_task_for_node(
                request.session_id, node_name, request.sub_session_id
            )
            if existing_task_id:
                meta = self._session_service._read_metadata(request.session_id)
                task_data = meta.get("tasks", {}).get(existing_task_id)
                if task_data:
                    cached = self._build_task_from_metadata(task_data)
                    cached.node_id = node_id
                    cached.node_name = node_name

                    # Create a visible reused task entry in the bisect task list
                    reused_task = self._queue_service.create_task(
                        session_id=request.session_id,
                        node_id=node_id,
                        node_name=node_name,
                        node_type=node_type,
                    )
                    reused_task.batch_id = batch_id
                    reused_task.reused = True
                    reused_task.status = cached.status
                    reused_task.metrics = cached.metrics
                    reused_task.error_detail = cached.error_detail
                    reused_task.main_result = cached.main_result
                    reused_task.ref_result = cached.ref_result
                    if request.sub_session_id:
                        reused_task.sub_session_id = request.sub_session_id

                    await self._queue_service.add_completed_task(reused_task)
                    return reused_task

        # Enqueue child task
        state.step_done.clear()
        state.step_result = None

        task = self._queue_service.create_task(
            session_id=request.session_id,
            node_id=node_id,
            node_name=node_name,
            node_type=node_type,
        )
        if request.sub_session_id:
            task.sub_session_id = request.sub_session_id
        task.batch_id = batch_id

        state.pending_task_id = task.task_id
        await self._queue_service.enqueue(task)

        await state.step_done.wait()

        if state.cancel_event.is_set():
            return None  # Caller checks cancel_event

        state.pending_task_id = None
        return state.step_result

    # ── Main loop ──

    async def _run_loop(self, job_id: str) -> None:
        """Graph-aware bisection loop for a specific job."""
        state = self._jobs.get(job_id)
        if state is None:
            return

        request = state.request
        candidates = state.candidates
        step = state.step
        batch_id = f"bisect:{job_id}"

        try:
            # ── Handle resume with a pending (interrupted) task ──
            if state.pending_task_id is not None:
                if state.step_result is None:
                    state.step_done.clear()
                    await state.step_done.wait()

                    if state.cancel_event.is_set():
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

                mid_node_id = result.node_id

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
                    ancestors = get_ancestors_in_set(mid_node_id, candidates, state.reverse_adj)
                    candidates -= ancestors
                else:
                    candidates = get_ancestors_in_set(mid_node_id, candidates, state.reverse_adj)
                state.candidates = candidates

            # ── Step 0: per-output initial check ──
            if request.output_node and step == 0 and len(candidates) > 0:
                # Find the predecessor of the Result node
                preds = state.reverse_adj.get(request.output_node, set())
                output_pred = next(iter(preds)) if preds else None

                if output_pred:
                    pred_node = state.node_map.get(output_pred)
                    pred_name = pred_node.name if pred_node else output_pred
                    pred_type = pred_node.type if pred_node else ""

                    step += 1
                    state.job.step = step
                    state.job.current_node = pred_name
                    await self._broadcast_job_status(state)

                    result = await self._infer_node(
                        state, output_pred, pred_name, pred_type, batch_id,
                    )

                    if state.cancel_event.is_set():
                        state.step = step
                        return

                    if result is None:
                        state.job.status = BisectStatus.ERROR
                        state.job.error = "Inference result was lost"
                        await self._broadcast_job_status(state)
                        return

                    passed, metric_value, error = self._evaluate_step(
                        result, request.search_for, request.metric, request.threshold,
                    )
                    state.steps_history.append(BisectStepInfo(
                        node_name=pred_name,
                        node_id=output_pred,
                        task_id=result.task_id if result.task_id else None,
                        metric_value=metric_value,
                        passed=passed,
                        error=error,
                    ))

                    if passed:
                        # This output is fine — done in 1 inference
                        state.job.status = BisectStatus.DONE
                        state.job.found_node = None
                        state.job.step = step
                        await self._broadcast_job_status(state)
                        self._persist_job(state)
                        self._jobs.pop(job_id, None)
                        return

            # ── Main graph-aware bisection loop ──
            while len(candidates) > 1:
                if state.cancel_event.is_set():
                    state.step = step
                    return

                mid_node_id = find_best_bisect_point(
                    candidates, state.reverse_adj, state.topo_order,
                )
                mid_node = state.node_map.get(mid_node_id)
                mid_name = mid_node.name if mid_node else mid_node_id
                mid_type = mid_node.type if mid_node else ""

                step += 1
                state.job.step = step
                state.job.current_node = mid_name
                await self._broadcast_job_status(state)

                result = await self._infer_node(
                    state, mid_node_id, mid_name, mid_type, batch_id,
                )

                if state.cancel_event.is_set():
                    state.step = step
                    return

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
                    ancestors = get_ancestors_in_set(mid_node_id, candidates, state.reverse_adj)
                    candidates -= ancestors
                else:
                    candidates = get_ancestors_in_set(mid_node_id, candidates, state.reverse_adj)

                state.candidates = candidates

            # Save final state
            state.step = step

            # Bisection converged
            if candidates:
                found_id = next(iter(candidates))
                found_node = state.node_map.get(found_id)
                found_name = found_node.name if found_node else found_id
                state.job.found_node = found_name
            else:
                state.job.found_node = None

            state.job.status = BisectStatus.DONE
            state.job.step = step

            await self._broadcast_job_status(state)
            self._persist_job(state)
            self._jobs.pop(job_id, None)

        except asyncio.CancelledError:
            state.step = step
            state.candidates = candidates
            state.job.status = BisectStatus.PAUSED
        except Exception as e:
            state.step = step
            state.candidates = candidates
            state.job.status = BisectStatus.ERROR
            state.job.error = str(e)
            await self._broadcast_job_status(state)
            self._persist_job(state)
            self._jobs.pop(job_id, None)
