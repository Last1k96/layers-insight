"""Inference task routes."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.schemas.inference import EnqueueRequest, InferenceTask, ReorderRequest


class BatchEnqueueRequest(BaseModel):
    """Request to enqueue multiple nodes for inference."""
    session_id: str
    nodes: list[dict]  # list of {node_id, node_name, node_type}

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.post("/enqueue", response_model=InferenceTask)
async def enqueue_task(req: EnqueueRequest, request: Request) -> InferenceTask:
    """Enqueue a node for inference."""
    # Reject nodes that are grayed out (not in the cut sub-model)
    if req.sub_session_id:
        session_svc = request.app.state.session_service
        sub_meta = session_svc.get_sub_session_meta(req.session_id, req.sub_session_id)
        if sub_meta and req.node_name in sub_meta.get("grayed_nodes", []):
            raise HTTPException(status_code=400, detail="Node is not part of this sub-session's model")

    queue_svc = request.app.state.queue_service
    task = queue_svc.create_task(
        session_id=req.session_id,
        node_id=req.node_id,
        node_name=req.node_name,
        node_type=req.node_type,
    )
    if req.sub_session_id:
        task.sub_session_id = req.sub_session_id
    return await queue_svc.enqueue(task)


@router.post("/enqueue-batch")
async def enqueue_batch(req: BatchEnqueueRequest, request: Request) -> list[InferenceTask]:
    """Enqueue multiple nodes for inference."""
    queue_svc = request.app.state.queue_service
    batch_id = str(uuid.uuid4())[:8]
    tasks = []
    for node in req.nodes:
        task = queue_svc.create_task(
            session_id=req.session_id,
            node_id=node["node_id"],
            node_name=node["node_name"],
            node_type=node["node_type"],
        )
        task.batch_id = batch_id
        result = await queue_svc.enqueue(task)
        tasks.append(result)
    return tasks


@router.post("/pause")
async def pause_queue(request: Request) -> dict:
    """Pause the queue worker. Kills the currently executing task and re-queues it."""
    async with request.app.state.pause_resume_lock:
        queue_svc = request.app.state.queue_service

        # Idempotent: already paused → no-op
        if queue_svc.paused:
            return {"paused": True, "requeued_task_id": None}

        inference_svc = request.app.state.inference_service

        kill_fn = None
        if inference_svc:
            kill_fn = lambda tid: inference_svc.kill_current(tid)

        requeued_id = await queue_svc.pause(kill_callback=kill_fn)

        # Also pause all bisect jobs if any are running
        bisect_svc = request.app.state.bisect_service
        if bisect_svc and bisect_svc.has_running_jobs:
            await bisect_svc.pause_all()

        return {"paused": True, "requeued_task_id": requeued_id}


@router.post("/resume")
async def resume_queue(request: Request) -> dict:
    """Resume the queue worker."""
    async with request.app.state.pause_resume_lock:
        queue_svc = request.app.state.queue_service

        # Idempotent: already running → no-op
        if not queue_svc.paused:
            return {"paused": False}

        await queue_svc.resume()

        # Also resume all paused bisect jobs
        bisect_svc = request.app.state.bisect_service
        if bisect_svc and bisect_svc.has_paused_jobs:
            await bisect_svc.resume_all()

        return {"paused": False}


@router.post("/cancel-all")
async def cancel_all_tasks(request: Request) -> dict:
    """Cancel all waiting tasks."""
    queue_svc = request.app.state.queue_service
    count = await queue_svc.cancel_all()
    return {"cancelled": count}


@router.get("/queue-state")
async def get_queue_state(request: Request) -> dict:
    """Get current queue state (paused/running)."""
    queue_svc = request.app.state.queue_service
    return {"paused": queue_svc.paused}


@router.put("/reorder")
async def reorder_tasks(req: ReorderRequest, request: Request) -> dict:
    """Reorder queued tasks."""
    queue_svc = request.app.state.queue_service
    await queue_svc.reorder(req.task_ids)
    return {"reordered": True}


@router.post("/{task_id}/rerun", response_model=InferenceTask)
async def rerun_task(task_id: str, request: Request) -> InferenceTask:
    """Re-enqueue a completed or failed task."""
    queue_svc = request.app.state.queue_service
    task = await queue_svc.rerun(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/{task_id}", response_model=InferenceTask)
async def get_task(task_id: str, request: Request) -> InferenceTask:
    """Get task status (poll fallback)."""
    queue_svc = request.app.state.queue_service
    task = queue_svc.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.delete("/{task_id}")
async def delete_task(task_id: str, request: Request) -> dict:
    """Delete a task — cancels waiting, kills executing, removes completed."""
    queue_svc = request.app.state.queue_service
    session_svc = request.app.state.session_service
    task = queue_svc.get_task(task_id)

    if task is not None:
        # Task is in memory — handle live queue operations
        if task.status == "waiting":
            await queue_svc.cancel(task_id)
        elif task.status == "executing":
            inference_svc = request.app.state.inference_service
            if inference_svc:
                inference_svc.kill_current(task_id)

        queue_svc.remove_task(task_id)
        session_svc.delete_task(task.session_id, task_id)

        from backend.ws.handler import ws_manager
        await ws_manager.broadcast(task.session_id, {
            "type": "task_deleted",
            "task_id": task_id,
            "node_name": task.node_name,
        })
    else:
        # Task not in memory (e.g. after server restart) — try persisted data
        session_id = request.query_params.get("session_id", "")
        if not session_id:
            raise HTTPException(status_code=404, detail="Task not found")

        meta = session_svc._read_metadata(session_id)
        task_data = meta.get("tasks", {}).get(task_id)
        if task_data is None:
            raise HTTPException(status_code=404, detail="Task not found")

        node_name = task_data.get("node_name", "")
        session_svc.delete_task(session_id, task_id)

        from backend.ws.handler import ws_manager
        await ws_manager.broadcast(session_id, {
            "type": "task_deleted",
            "task_id": task_id,
            "node_name": node_name,
        })

    return {"deleted": True}
