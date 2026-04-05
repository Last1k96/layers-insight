"""Session management routes."""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from backend.schemas.session import (
    CloneEnqueueRequest,
    CloneRequest,
    CompareResponse,
    CutRequest,
    SessionConfig,
    SessionDetail,
    SessionInfo,
    SubSessionInfo,
)
from backend.services.model_cut_service import CutResult
from backend.utils.model_converter import convert_to_ir, detect_model_format

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _get_session_service(request: Request):
    return request.app.state.session_service


@router.post("", response_model=SessionInfo)
async def create_session(config: SessionConfig, request: Request) -> SessionInfo:
    """Create a new session. Non-IR models are converted to IR automatically."""
    svc = _get_session_service(request)

    model_path = Path(config.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=400, detail=f"Model path not found: {model_path}")
    try:
        fmt = detect_model_format(model_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if fmt == "ir":
        try:
            return svc.create_session(config)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Non-IR: convert to temporary IR, then pass converted files to session service
    ov_core = request.app.state.ov_core
    if ov_core is None:
        raise HTTPException(status_code=503, detail="OpenVINO not available — cannot convert model")

    tmp_dir = tempfile.mkdtemp(prefix="li_convert_")
    try:
        try:
            converted_xml = convert_to_ir(model_path, Path(tmp_dir), ov_core)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to convert {fmt} model to IR: {e}",
            )

        config = config.model_copy(update={
            "model_path": str(converted_xml),
            "original_format": fmt,
        })
        try:
            return svc.create_session(config, converted_dir=Path(tmp_dir))
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("", response_model=list[SessionInfo])
async def list_sessions(request: Request) -> list[SessionInfo]:
    """List all sessions."""
    svc = _get_session_service(request)
    return svc.list_sessions()


@router.get("/compare", response_model=CompareResponse)
async def compare_sessions(
    request: Request,
    session_a: str = Query(..., description="First session ID"),
    session_b: str = Query(..., description="Second session ID"),
) -> CompareResponse:
    """Compare inference results between two sessions node by node."""
    svc = _get_session_service(request)

    if svc.get_session(session_a) is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_a}' not found")
    if svc.get_session(session_b) is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_b}' not found")

    result = svc.compare_sessions(session_a, session_b)
    return CompareResponse(**result)


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str, request: Request) -> SessionDetail:
    """Get session detail."""
    svc = _get_session_service(request)
    detail = svc.get_session(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return detail


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    """Delete a session."""
    svc = _get_session_service(request)
    if svc.delete_session(session_id):
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="Session not found")


@router.post("/{session_id}/clone")
async def clone_session(session_id: str, req: CloneRequest, request: Request) -> dict:
    """Clone a session with optional overrides.

    Returns the new session info and a list of nodes that were inferred
    in the source session, ordered by worst accuracy first.
    """
    svc = _get_session_service(request)
    source = svc.get_session(session_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source session not found")

    overrides = req.model_dump(exclude_none=True)
    # Convert InputConfig objects to dicts for the service layer
    if overrides.get("inputs"):
        overrides["inputs"] = [inp.model_dump() for inp in req.inputs]

    result = svc.clone_session(session_id, overrides)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to clone session")

    new_info, inferred_nodes = result
    return {
        "session": new_info.model_dump(),
        "inferred_nodes": inferred_nodes,
    }


@router.post("/{session_id}/clone-enqueue")
async def clone_enqueue(session_id: str, req: CloneEnqueueRequest, request: Request) -> dict:
    """Batch-enqueue nodes from source session into a target session.

    Nodes are ordered by worst accuracy from the source session.
    """
    svc = _get_session_service(request)
    source = svc.get_session(session_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source session not found")

    target = svc.get_session(req.target_session_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Target session not found")

    # Get source session's inferred nodes for ordering
    source_meta = svc._read_metadata(session_id)
    inferred_nodes = svc._get_inferred_nodes_sorted(source_meta)
    node_order = {n["node_name"]: i for i, n in enumerate(inferred_nodes)}

    # Sort requested nodes by worst accuracy from source
    ordered_names = sorted(
        req.node_names,
        key=lambda name: node_order.get(name, len(inferred_nodes)),
    )

    # Build node info map from source tasks
    tasks = source_meta.get("tasks", {})
    node_info: dict[str, dict] = {}
    for task_data in tasks.values():
        if task_data.get("status") == "success":
            name = task_data.get("node_name")
            if name and name not in node_info:
                node_info[name] = {
                    "node_id": task_data.get("node_id", name),
                    "node_type": task_data.get("node_type", ""),
                }

    # Enqueue into target session's queue
    queue_svc = request.app.state.queue_service
    batch_id = str(uuid.uuid4())[:8]
    enqueued = []

    for name in ordered_names:
        info = node_info.get(name, {"node_id": name, "node_type": ""})
        task = queue_svc.create_task(
            session_id=req.target_session_id,
            node_id=info["node_id"],
            node_name=name,
            node_type=info["node_type"],
        )
        task.batch_id = batch_id
        result = await queue_svc.enqueue(task)
        enqueued.append(result.model_dump())

    return {"enqueued": len(enqueued), "batch_id": batch_id, "tasks": enqueued}


@router.post("/{session_id}/cut", response_model=SubSessionInfo)
async def cut_model(session_id: str, req: CutRequest, request: Request) -> SubSessionInfo:
    """Cut the model at a node, creating a sub-session."""
    svc = _get_session_service(request)
    session = svc.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    model_cut_svc = request.app.state.model_cut_service
    if model_cut_svc is None:
        raise HTTPException(status_code=503, detail="OpenVINO not available")

    model = request.app.state.models.get(session_id)
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")

    try:
        result: CutResult = model_cut_svc.perform_cut(
            session_svc=svc,
            session=session,
            session_id=session_id,
            model=model,
            node_name=req.node_name,
            cut_type=req.cut_type,
            input_precision=req.input_precision,
            parent_sub_session_id=req.parent_sub_session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Broadcast sub-session creation via WS (HTTP-layer concern)
    from backend.ws.handler import ws_manager
    await ws_manager.broadcast(session_id, {
        "type": "sub_session_created",
        "sub_session_id": result.sub_session.id,
        "parent_sub_session_id": req.parent_sub_session_id,
        "cut_type": result.effective_cut_type,
        "cut_node": req.node_name,
        "grayed_nodes": result.grayed_nodes,
        "ancestor_cuts": result.ancestor_cuts,
    })

    return result.sub_session


@router.get("/{session_id}/sub-sessions", response_model=list[SubSessionInfo])
async def list_sub_sessions(session_id: str, request: Request) -> list[SubSessionInfo]:
    """List sub-sessions for a session."""
    svc = _get_session_service(request)
    return svc.list_sub_sessions(session_id)


@router.delete("/{session_id}/sub-sessions/{sub_session_id}")
async def delete_sub_session(session_id: str, sub_session_id: str, request: Request) -> dict:
    """Delete a sub-session and all its descendants."""
    svc = _get_session_service(request)
    if svc.delete_sub_session(session_id, sub_session_id):
        return {"deleted": True, "sub_session_id": sub_session_id}
    raise HTTPException(status_code=404, detail="Sub-session not found")
