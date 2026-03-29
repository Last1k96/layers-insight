"""Session management routes."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from backend.schemas.session import CutRequest, SessionConfig, SessionDetail, SessionInfo, SubSessionInfo
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
    try:
        fmt = detect_model_format(model_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if fmt == "ir":
        return svc.create_session(config)

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
        return svc.create_session(config, converted_dir=Path(tmp_dir))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
