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
    task = queue_svc.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status == "waiting":
        await queue_svc.cancel(task_id)
    elif task.status == "executing":
        inference_svc = request.app.state.inference_service
        if inference_svc:
            inference_svc.kill_current(task_id)
        # Worker loop will handle the killed process and mark it failed

    # Remove from queue service in-memory store
    queue_svc.remove_task(task_id)

    # Remove persisted task data and tensor files
    session_svc = request.app.state.session_service
    session_svc.delete_task(task.session_id, task_id)

    # Notify frontend to remove the task
    from backend.ws.handler import ws_manager
    await ws_manager.broadcast(task.session_id, {
        "type": "task_deleted",
        "task_id": task_id,
        "node_name": task.node_name,
    })

    return {"deleted": True}
