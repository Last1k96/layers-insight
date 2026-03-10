"""Session management routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.schemas.session import CutRequest, SessionConfig, SessionDetail, SessionInfo, SubSessionInfo

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _get_session_service(request: Request):
    return request.app.state.session_service


@router.post("", response_model=SessionInfo)
async def create_session(config: SessionConfig, request: Request) -> SessionInfo:
    """Create a new session."""
    svc = _get_session_service(request)
    return svc.create_session(config)


@router.get("", response_model=list[SessionInfo])
async def list_sessions(request: Request) -> list[SessionInfo]:
    """List all sessions."""
    svc = _get_session_service(request)
    return svc.list_sessions()


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
        if req.cut_type == "output":
            cut_model, grayed_nodes = model_cut_svc.make_output_node(model, req.node_name)
        elif req.cut_type == "input":
            # Find the main output .npy from the latest successful task for this node
            task_id = _find_task_for_node(svc, session_id, req.node_name)
            if task_id is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"No successful inference found for node '{req.node_name}'. Run inference first.",
                )
            npy_path = svc.get_tensor_path(session_id, task_id, "main_output")
            if npy_path is None:
                raise HTTPException(status_code=400, detail="Main output tensor not found")
            cut_model, input_data, grayed_nodes = model_cut_svc.make_input_node(
                model, req.node_name, str(npy_path), req.input_precision,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid cut_type: {req.cut_type}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Store cut model
    sub_session = svc.create_sub_session(
        session_id=session_id,
        cut_type=req.cut_type,
        cut_node=req.node_name,
        grayed_nodes=grayed_nodes,
    )

    # Cache the cut model for inference
    request.app.state.models[f"{session_id}:{sub_session.id}"] = cut_model

    # Broadcast sub-session creation via WS
    from backend.ws.handler import ws_manager
    await ws_manager.broadcast(session_id, {
        "type": "sub_session_created",
        "sub_session_id": sub_session.id,
        "cut_type": req.cut_type,
        "cut_node": req.node_name,
        "grayed_nodes": grayed_nodes,
    })

    return sub_session


def _find_task_for_node(svc, session_id: str, node_name: str) -> str | None:
    """Find the most recent successful task_id for a given node."""
    meta = svc._read_metadata(session_id)
    for task_id, task_data in reversed(list(meta.get("tasks", {}).items())):
        if task_data.get("node_name") == node_name and task_data.get("status") == "success":
            return task_id
    return None


@router.get("/{session_id}/sub-sessions", response_model=list[SubSessionInfo])
async def list_sub_sessions(session_id: str, request: Request) -> list[SubSessionInfo]:
    """List sub-sessions for a session."""
    svc = _get_session_service(request)
    return svc.list_sub_sessions(session_id)
