"""Tests for queue service."""
import asyncio
import pytest

from backend.services.queue_service import QueueService
from backend.schemas.inference import InferenceTask, TaskStatus


@pytest.fixture
def queue_service():
    return QueueService()


class TestQueueService:
    @pytest.mark.asyncio
    async def test_create_task(self, queue_service):
        task = queue_service.create_task(
            session_id="s1",
            node_id="conv1",
            node_name="conv1",
            node_type="Convolution",
        )

        assert task.task_id is not None
        assert task.session_id == "s1"
        assert task.node_id == "conv1"
        assert task.status == TaskStatus.WAITING

    @pytest.mark.asyncio
    async def test_enqueue(self, queue_service):
        notifications = []
        queue_service.set_callbacks(
            notify=lambda t: notifications.append(t),
            infer=lambda t: t,
        )

        task = queue_service.create_task("s1", "n1", "conv1", "Convolution")
        result = await queue_service.enqueue(task)

        assert result.task_id == task.task_id
        assert len(notifications) == 1

    @pytest.mark.asyncio
    async def test_get_task(self, queue_service):
        queue_service.set_callbacks(notify=lambda t: None, infer=lambda t: t)

        task = queue_service.create_task("s1", "n1", "conv1", "Convolution")
        await queue_service.enqueue(task)

        retrieved = queue_service.get_task(task.task_id)
        assert retrieved is not None
        assert retrieved.task_id == task.task_id

    @pytest.mark.asyncio
    async def test_cancel_waiting(self, queue_service):
        queue_service.set_callbacks(notify=lambda t: None, infer=lambda t: t)

        task = queue_service.create_task("s1", "n1", "conv1", "Convolution")
        await queue_service.enqueue(task)

        result = await queue_service.cancel(task.task_id)
        assert result is True
        assert queue_service.get_task(task.task_id).status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_get_all_tasks_filtered(self, queue_service):
        queue_service.set_callbacks(notify=lambda t: None, infer=lambda t: t)

        t1 = queue_service.create_task("s1", "n1", "conv1", "Convolution")
        t2 = queue_service.create_task("s2", "n2", "relu1", "Relu")
        await queue_service.enqueue(t1)
        await queue_service.enqueue(t2)

        all_tasks = queue_service.get_all_tasks()
        assert len(all_tasks) == 2

        s1_tasks = queue_service.get_all_tasks(session_id="s1")
        assert len(s1_tasks) == 1
        assert s1_tasks[0].session_id == "s1"

    @pytest.mark.asyncio
    async def test_remove_task(self, queue_service):
        queue_service.set_callbacks(notify=lambda t: None, infer=lambda t: t)
        task = queue_service.create_task("s1", "n1", "conv1", "Convolution")
        await queue_service.enqueue(task)

        assert queue_service.remove_task(task.task_id) is True
        assert queue_service.get_task(task.task_id) is None
        assert queue_service.is_deleted(task.task_id) is True

    @pytest.mark.asyncio
    async def test_remove_nonexistent(self, queue_service):
        assert queue_service.remove_task("nonexistent") is False

    @pytest.mark.asyncio
    async def test_rerun_preserves_sub_session_id(self, queue_service):
        queue_service.set_callbacks(notify=lambda t: None, infer=lambda t: t)
        task = queue_service.create_task("s1", "n1", "conv1", "Convolution")
        task.sub_session_id = "sub1"
        await queue_service.enqueue(task)

        new_task = await queue_service.rerun(task.task_id)
        assert new_task is not None
        assert new_task.task_id != task.task_id
        assert new_task.sub_session_id == "sub1"
        assert new_task.node_name == "conv1"

    @pytest.mark.asyncio
    async def test_worker_skips_cancelled(self, queue_service):
        """Worker should skip tasks that were cancelled while waiting."""
        results = []

        async def mock_infer(task):
            results.append(task.task_id)
            task.status = TaskStatus.SUCCESS
            return task

        queue_service.set_callbacks(notify=lambda t: None, infer=mock_infer)

        t1 = queue_service.create_task("s1", "n1", "conv1", "Convolution")
        t2 = queue_service.create_task("s1", "n2", "relu1", "Relu")

        await queue_service.enqueue(t1)
        await queue_service.enqueue(t2)

        # Cancel first, let worker run
        await queue_service.cancel(t1.task_id)
        await queue_service.start_worker()

        # Give worker time to process
        await asyncio.sleep(0.1)
        await queue_service.stop_worker()

        # t1 was cancelled — only t2 should have been inferred
        assert t1.task_id not in results

    @pytest.mark.asyncio
    async def test_worker_skips_deleted(self, queue_service):
        """Worker should skip tasks that were deleted while waiting."""
        results = []

        async def mock_infer(task):
            results.append(task.task_id)
            task.status = TaskStatus.SUCCESS
            return task

        queue_service.set_callbacks(notify=lambda t: None, infer=mock_infer)

        t1 = queue_service.create_task("s1", "n1", "conv1", "Convolution")
        await queue_service.enqueue(t1)

        # Delete before worker processes
        queue_service.remove_task(t1.task_id)
        await queue_service.start_worker()

        await asyncio.sleep(0.1)
        await queue_service.stop_worker()

        assert t1.task_id not in results
