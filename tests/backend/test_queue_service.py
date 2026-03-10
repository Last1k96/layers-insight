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
