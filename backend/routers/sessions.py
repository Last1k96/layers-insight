"""Session management routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.schemas.session import SessionConfig, SessionDetail, SessionInfo

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
