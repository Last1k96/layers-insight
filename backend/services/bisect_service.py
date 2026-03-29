"""Bisection search service — binary search for accuracy drop or compilation failure."""
from __future__ import annotations

import asyncio
import math
from typing import Any, Optional

from backend.schemas.bisect import (
    BisectMetric,
    BisectProgress,
    BisectSearchFor,
    BisectStatus,
    BisectStepInfo,
)
from backend.schemas.graph import GraphData, GraphEdge
from backend.schemas.inference import InferenceTask, TaskStatus


def _topo_order(graph_data: GraphData) -> list[str]:
    """Compute topological order of node IDs from graph data.

    The graph is already stored in topological order (from OV's get_ordered_ops),
    so we just return the node IDs in their existing order.
    """
    return [n.id for n in graph_data.nodes]


def _nodes_between(
    graph_data: GraphData,
    start_node: Optional[str],
    end_node: Optional[str],
) -> list[str]:
    """Return the list of node IDs between start and end (inclusive) in topological order.

    Skips Parameter and Result nodes as they are not meaningful inference targets.
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

    # Filter to only meaningful nodes (skip Parameter and Result)
    return [
        nid for nid in topo[start_idx : end_idx + 1]
        if node_types.get(nid) not in ("Parameter", "Result")
    ]


class BisectService:
    """Manages a single bisection search at a time."""

    def __init__(self):
        self._progress = BisectProgress()
        self._task: Optional[asyncio.Task] = None
        self._cancel_event = asyncio.Event()
        # Set when the current inference step finishes
        self._step_done = asyncio.Event()
        self._step_result: Optional[InferenceTask] = None

    @property
    def progress(self) -> BisectProgress:
        return self._progress

    @property
    def is_running(self) -> bool:
        return self._progress.status == BisectStatus.RUNNING

    def on_task_complete(self, task: InferenceTask) -> None:
        """Called when an inference task finishes — unblocks the bisect loop if relevant."""
        self._step_result = task
        self._step_done.set()

    async def start(
        self,
        request: Any,
        graph_data: GraphData,
        queue_service: Any,
        broadcast: Any,
    ) -> BisectProgress:
        """Start a bisection search. Returns the initial progress."""
        if self.is_running:
            raise RuntimeError("A bisection is already running")

        nodes = _nodes_between(graph_data, request.start_node, request.end_node)
        if len(nodes) < 2:
            raise ValueError("Need at least 2 nodes in range for bisection")

        total_steps = max(1, math.ceil(math.log2(len(nodes))))

        self._cancel_event.clear()
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

        self._task = asyncio.create_task(
            self._run_loop(
                nodes=nodes,
                request=request,
                graph_data=graph_data,
                queue_service=queue_service,
                broadcast=broadcast,
            )
        )

        return self._progress

    async def stop(self) -> BisectProgress:
        """Stop the running bisection."""
        if not self.is_running:
            return self._progress

        self._cancel_event.set()
        # Also unblock any waiting step
        self._step_done.set()

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

        self._progress.status = BisectStatus.STOPPED
        return self._progress

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
                # Compilation failures and crashes count as "failed"
                return False, None, error_detail
            # Success means compilation worked
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

    async def _run_loop(
        self,
        nodes: list[str],
        request: Any,
        graph_data: GraphData,
        queue_service: Any,
        broadcast: Any,
    ) -> None:
        """The main bisection binary search loop."""
        node_map = {n.id: n for n in graph_data.nodes}
        lo = 0
        hi = len(nodes) - 1
        step = 0

        try:
            while lo < hi:
                if self._cancel_event.is_set():
                    return

                mid = (lo + hi) // 2
                mid_node_id = nodes[mid]
                mid_node = node_map.get(mid_node_id)
                mid_name = mid_node.name if mid_node else mid_node_id
                mid_type = mid_node.type if mid_node else ""

                step += 1
                self._progress.step = step
                self._progress.current_node = mid_name
                self._progress.range_start = nodes[lo]
                self._progress.range_end = nodes[hi]

                # Broadcast progress
                await broadcast(request.session_id, {
                    "type": "bisect_progress",
                    "status": "running",
                    "range_start": nodes[lo],
                    "range_end": nodes[hi],
                    "current_node": mid_name,
                    "step": step,
                    "total_steps": self._progress.total_steps,
                })

                # Enqueue inference for the midpoint node
                self._step_done.clear()
                self._step_result = None

                task = queue_service.create_task(
                    session_id=request.session_id,
                    node_id=mid_node_id,
                    node_name=mid_name,
                    node_type=mid_type,
                )
                if request.sub_session_id:
                    task.sub_session_id = request.sub_session_id
                task.batch_id = "bisect"

                await queue_service.enqueue(task)

                # Wait for the task to complete (or cancellation)
                await self._step_done.wait()

                if self._cancel_event.is_set():
                    return

                result = self._step_result
                if result is None:
                    self._progress.status = BisectStatus.ERROR
                    self._progress.error = "Inference result was lost"
                    return

                passed, metric_value, error = self._evaluate_step(
                    result, request.search_for, request.metric, request.threshold,
                )

                step_info = BisectStepInfo(
                    node_name=mid_name,
                    node_id=mid_node_id,
                    task_id=result.task_id,
                    metric_value=metric_value,
                    passed=passed,
                    error=error,
                )
                self._progress.steps_history.append(step_info)

                if passed:
                    # This node is fine, problem is later
                    lo = mid + 1
                else:
                    # This node already has the problem
                    hi = mid

            # Bisection converged
            found_id = nodes[lo]
            found_node = node_map.get(found_id)
            found_name = found_node.name if found_node else found_id

            self._progress.status = BisectStatus.DONE
            self._progress.found_node = found_name
            self._progress.range_start = found_name
            self._progress.range_end = found_name

            await broadcast(request.session_id, {
                "type": "bisect_progress",
                "status": "done",
                "found_node": found_name,
                "step": step,
                "total_steps": self._progress.total_steps,
            })

        except asyncio.CancelledError:
            self._progress.status = BisectStatus.STOPPED
        except Exception as e:
            self._progress.status = BisectStatus.ERROR
            self._progress.error = str(e)
            await broadcast(request.session_id, {
                "type": "bisect_progress",
                "status": "error",
                "error": str(e),
                "step": step,
                "total_steps": self._progress.total_steps,
            })
