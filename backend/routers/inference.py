"""Inference task routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.schemas.inference import EnqueueRequest, InferenceTask, ReorderRequest

router = APIRouter(prefix="/api/inference", tags=["inference"])


@router.post("/enqueue", response_model=InferenceTask)
async def enqueue_task(req: EnqueueRequest, request: Request) -> InferenceTask:
    """Enqueue a node for inference."""
    queue_svc = request.app.state.queue_service
    task = queue_svc.create_task(
        session_id=req.session_id,
        node_id=req.node_id,
        node_name=req.node_name,
        node_type=req.node_type,
    )
    return await queue_svc.enqueue(task)


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
async def cancel_task(task_id: str, request: Request) -> dict:
    """Cancel a waiting task."""
    queue_svc = request.app.state.queue_service
    if await queue_svc.cancel(task_id):
        return {"cancelled": True}
    raise HTTPException(status_code=400, detail="Task not cancellable (not in waiting state)")
