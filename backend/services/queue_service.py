"""Async task queue with sequential worker."""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, Optional

from backend.schemas.inference import InferenceTask, TaskStatus


class QueueService:
    """Manages inference task queue with sequential execution."""

    def __init__(self):
        self._queue: asyncio.Queue[InferenceTask] = asyncio.Queue()
        self._tasks: dict[str, InferenceTask] = {}
        self._task_order: list[str] = []  # Maintains insertion order
        self._worker_task: Optional[asyncio.Task] = None
        self._notify_callback: Optional[Callable] = None
        self._infer_callback: Optional[Callable] = None

    def set_callbacks(
        self,
        notify: Callable,
        infer: Callable,
    ) -> None:
        """Set callbacks for notifications and inference execution."""
        self._notify_callback = notify
        self._infer_callback = infer

    async def start_worker(self) -> None:
        """Start the background worker task."""
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop_worker(self) -> None:
        """Stop the background worker."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def enqueue(self, task: InferenceTask) -> InferenceTask:
        """Add a task to the queue."""
        self._tasks[task.task_id] = task
        self._task_order.append(task.task_id)
        await self._queue.put(task)
        await self._notify(task)
        return task

    def create_task(
        self,
        session_id: str,
        node_id: str,
        node_name: str,
        node_type: str,
    ) -> InferenceTask:
        """Create a new inference task."""
        task_id = str(uuid.uuid4())[:8]
        return InferenceTask(
            task_id=task_id,
            session_id=session_id,
            node_id=node_id,
            node_name=node_name,
            node_type=node_type,
            status=TaskStatus.WAITING,
        )

    async def reorder(self, task_ids: list[str]) -> None:
        """Reorder waiting tasks. Only affects waiting tasks."""
        # Rebuild queue with new order
        waiting = [tid for tid in task_ids if tid in self._tasks
                   and self._tasks[tid].status == TaskStatus.WAITING]

        # Drain and refill queue
        new_queue: asyncio.Queue[InferenceTask] = asyncio.Queue()

        # Keep non-waiting tasks that might be in the queue
        old_items = []
        while not self._queue.empty():
            try:
                old_items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # Add in new order
        for tid in waiting:
            await new_queue.put(self._tasks[tid])

        self._queue = new_queue

    async def rerun(self, task_id: str) -> Optional[InferenceTask]:
        """Re-enqueue a completed or failed task."""
        old_task = self._tasks.get(task_id)
        if old_task is None:
            return None

        new_task = self.create_task(
            session_id=old_task.session_id,
            node_id=old_task.node_id,
            node_name=old_task.node_name,
            node_type=old_task.node_type,
        )
        return await self.enqueue(new_task)

    async def cancel(self, task_id: str) -> bool:
        """Cancel a waiting task."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.WAITING:
            task.status = TaskStatus.FAILED
            task.error_detail = "Cancelled by user"
            await self._notify(task)
            return True
        return False

    def get_task(self, task_id: str) -> Optional[InferenceTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self, session_id: Optional[str] = None) -> list[InferenceTask]:
        """Get all tasks, optionally filtered by session."""
        tasks = list(self._tasks.values())
        if session_id:
            tasks = [t for t in tasks if t.session_id == session_id]
        return tasks

    async def _worker_loop(self) -> None:
        """Sequential worker: process one task at a time."""
        while True:
            task = await self._queue.get()

            # Skip cancelled tasks
            if task.status == TaskStatus.FAILED:
                self._queue.task_done()
                continue

            task.status = TaskStatus.EXECUTING
            await self._notify(task)

            try:
                if self._infer_callback:
                    result = await self._infer_callback(task)
                    # Result is the updated task
                    if isinstance(result, InferenceTask):
                        self._tasks[task.task_id] = result
                        await self._notify(result)
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error_detail = f"Worker error: {e}"
                await self._notify(task)
            finally:
                self._queue.task_done()

    async def _notify(self, task: InferenceTask) -> None:
        """Send task status notification."""
        if self._notify_callback:
            try:
                await self._notify_callback(task)
            except Exception:
                pass
