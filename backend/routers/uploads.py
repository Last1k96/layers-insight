"""File upload routes for browser-side staging."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from backend.services.upload_service import UploadError, UploadService

router = APIRouter(prefix="/api/uploads", tags=["uploads"])


def _get_service(request: Request) -> UploadService:
    svc = getattr(request.app.state, "upload_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Upload service not available")
    return svc


class StagedFileResponse(BaseModel):
    staged_path: str
    original_filename: str
    size: int
    group_id: str
    warnings: list[str] = []


class GroupSummary(BaseModel):
    group_id: str
    files: list[dict]
    total_size: int
    mtime: float


def _ir_pair_warnings(svc: UploadService, group_id: str, kind: Optional[str], filename: str) -> list[str]:
    """Return validation hints for IR-pair completeness."""
    if kind != "model":
        return []
    if not filename.lower().endswith(".xml"):
        return []
    gdir = svc.uploads_dir / group_id
    if not gdir.exists():
        return []
    siblings = {f.name.lower() for f in gdir.iterdir() if f.is_file()}
    base = filename.lower()[: -len(".xml")]
    if f"{base}.bin" not in siblings:
        return ["missing .bin sibling"]
    return []


@router.post("", response_model=StagedFileResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    group_id: Optional[str] = Form(None),
    kind: Optional[str] = Form(None),
) -> StagedFileResponse:
    svc = _get_service(request)

    content_length: Optional[int] = None
    cl_header = request.headers.get("content-length")
    if cl_header is not None:
        try:
            content_length = int(cl_header)
        except ValueError:
            content_length = None

    try:
        staged = await svc.stage_upload(
            filename=file.filename or "uploaded.bin",
            reader=file.read,
            group_id=group_id,
            content_length=content_length,
        )
    except UploadError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))

    warnings = _ir_pair_warnings(svc, staged.group_id, kind, staged.original_filename)
    return StagedFileResponse(
        staged_path=staged.staged_path,
        original_filename=staged.original_filename,
        size=staged.size,
        group_id=staged.group_id,
        warnings=warnings,
    )


@router.get("", response_model=list[GroupSummary])
async def list_groups(request: Request) -> list[GroupSummary]:
    svc = _get_service(request)
    return [GroupSummary(**g) for g in svc.list_groups()]


@router.delete("/{group_id}")
async def delete_group(group_id: str, request: Request) -> dict:
    svc = _get_service(request)
    try:
        svc.delete_group(group_id)
    except UploadError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))
    return {"deleted": group_id}
