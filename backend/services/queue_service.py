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
        self._paused: bool = False
        self._pause_event: asyncio.Event = asyncio.Event()
        self._pause_event.set()  # Start unpaused (event is "set" = not blocked)
        self._executing_task_id: Optional[str] = None

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
            print(f"[shutdown:queue] cancelling worker (done={self._worker_task.done()})", flush=True)
            self._worker_task.cancel()
            try:
                await asyncio.wait_for(self._worker_task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                print("[shutdown:queue] worker cancel timed out", flush=True)
            print("[shutdown:queue] worker stopped", flush=True)
        else:
            print("[shutdown:queue] no worker to stop", flush=True)

    async def enqueue(self, task: InferenceTask) -> InferenceTask:
        """Add a task to the queue."""
        self._ensure_worker_alive()
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

    async def add_completed_task(self, task: InferenceTask) -> InferenceTask:
        """Register a pre-completed task in the store and notify, without enqueuing for execution."""
        self._tasks[task.task_id] = task
        self._task_order.append(task.task_id)
        await self._notify(task)
        return task

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
        new_task.sub_session_id = old_task.sub_session_id
        return await self.enqueue(new_task)

    @property
    def paused(self) -> bool:
        """Whether the queue worker is paused."""
        return self._paused

    async def pause(self, kill_callback: Optional[Callable] = None) -> Optional[str]:
        """Pause the queue. If a task is executing, kill it and re-queue at front.

        Args:
            kill_callback: Optional callable(task_id) -> bool to kill the running subprocess.

        Returns:
            The task_id of the re-queued task, or None.
        """
        self._paused = True
        self._pause_event.clear()

        requeued_id = None
        # If a task is currently executing, kill it and re-queue
        if self._executing_task_id:
            task = self._tasks.get(self._executing_task_id)
            if task and task.status == TaskStatus.EXECUTING:
                if kill_callback:
                    kill_callback(self._executing_task_id)
                # Reset task to waiting and push to front of queue
                task.status = TaskStatus.WAITING
                task.stage = None
                task.error_detail = None
                requeued_id = task.task_id

                # Drain existing queue, prepend re-queued task, re-add rest
                old_items = []
                while not self._queue.empty():
                    try:
                        old_items.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                await self._queue.put(task)
                for item in old_items:
                    await self._queue.put(item)

                await self._notify(task)

        return requeued_id

    async def resume(self) -> None:
        """Resume the queue worker."""
        self._paused = False
        self._pause_event.set()

    async def cancel_all(self) -> int:
        """Cancel all waiting tasks. Returns count of cancelled tasks."""
        cancelled = 0
        for task in list(self._tasks.values()):
            if task.status == TaskStatus.WAITING:
                task.status = TaskStatus.FAILED
                task.error_detail = "Cancelled"
                await self._notify(task)
                cancelled += 1

        # Drain the queue (items are now marked failed, worker will skip them)
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        return cancelled

    async def cancel(self, task_id: str) -> bool:
        """Cancel a waiting task."""
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.WAITING:
            task.status = TaskStatus.FAILED
            task.error_detail = "Cancelled by user"
            await self._notify(task)
            return True
        return False

    def is_deleted(self, task_id: str) -> bool:
        """Check if a task has been removed (deleted by user)."""
        return task_id not in self._tasks

    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the in-memory store."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        del self._tasks[task_id]
        if task_id in self._task_order:
            self._task_order.remove(task_id)
        return True

    def get_task(self, task_id: str) -> Optional[InferenceTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self, session_id: Optional[str] = None) -> list[InferenceTask]:
        """Get all tasks, optionally filtered by session."""
        tasks = list(self._tasks.values())
        if session_id:
            tasks = [t for t in tasks if t.session_id == session_id]
        return tasks

    def _ensure_worker_alive(self) -> None:
        """Restart the worker if it died unexpectedly."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        """Sequential worker: process one task at a time."""
        while True:
            # Wait until unpaused before dequeuing
            await self._pause_event.wait()

            task = await self._queue.get()

            # Skip cancelled tasks
            if task.status == TaskStatus.FAILED:
                self._queue.task_done()
                continue

            # Skip tasks deleted while waiting
            if task.task_id not in self._tasks:
                self._queue.task_done()
                continue

            # Check again after dequeue — we may have been paused between wait and get
            if self._paused:
                # Put it back and wait
                await self._queue.put(task)
                self._queue.task_done()
                continue

            self._executing_task_id = task.task_id
            task.status = TaskStatus.EXECUTING
            await self._notify(task)

            try:
                if self._infer_callback:
                    result = await self._infer_callback(task)
                    # Only store/notify if not deleted or paused during execution
                    if task.task_id in self._tasks and isinstance(result, InferenceTask):
                        if result.status != TaskStatus.WAITING:
                            self._tasks[task.task_id] = result
                            await self._notify(result)
            except Exception as e:
                if task.task_id in self._tasks and task.status != TaskStatus.WAITING:
                    task.status = TaskStatus.FAILED
                    task.error_detail = f"Worker error: {e}"
                    await self._notify(task)
            finally:
                self._executing_task_id = None
                self._queue.task_done()

    async def _notify(self, task: InferenceTask) -> None:
        """Send task status notification."""
        if self._notify_callback:
            try:
                await self._notify_callback(task)
            except Exception:
                pass
