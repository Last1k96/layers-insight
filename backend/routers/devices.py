"""Device discovery routes."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/devices", tags=["devices"])


@router.get("")
async def list_devices(request: Request) -> list[str]:
    """List available OpenVINO devices."""
    ov_core = request.app.state.ov_core
    if ov_core is None:
        return ["CPU"]
    try:
        return ov_core.available_devices
    except Exception:
        return ["CPU"]
