"""Device discovery and app defaults routes."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["devices"])


@router.get("/devices")
async def list_devices(request: Request) -> list[str]:
    """List available OpenVINO devices."""
    ov_core = request.app.state.ov_core
    if ov_core is None:
        return ["CPU"]
    try:
        return ov_core.available_devices
    except Exception:
        return ["CPU"]


class AppDefaults(BaseModel):
    ov_path: Optional[str] = None
    model_path: Optional[str] = None
    input_path: Optional[str] = None
    main_device: str = "CPU"
    ref_device: str = "CPU"


@router.get("/defaults", response_model=AppDefaults)
async def get_defaults(request: Request) -> AppDefaults:
    """Return CLI-provided defaults for session creation form."""
    config = request.app.state.config
    return AppDefaults(
        ov_path=config.ov_path,
        model_path=config.model_path,
        input_path=config.input_path,
        main_device=config.main_device,
        ref_device=config.ref_device,
    )


class ModelInputInfo(BaseModel):
    name: str
    shape: list[int]
    element_type: str


@router.get("/model-inputs", response_model=list[ModelInputInfo])
async def get_model_inputs(
    request: Request,
    model_path: str = Query(..., description="Path to model .xml file"),
    ov_path: Optional[str] = Query(None, description="Custom OpenVINO path"),
) -> list[ModelInputInfo]:
    """Read a model file and return its input parameter info."""
    ov_core = request.app.state.ov_core
    if ov_core is None:
        raise HTTPException(status_code=503, detail="OpenVINO not available")

    xml_path = Path(model_path)
    if not xml_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")

    try:
        model = ov_core.read_model(str(xml_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read model: {e}")

    inputs = []
    for param in model.get_parameters():
        pshape = param.get_output_partial_shape(0)
        shape = list(pshape.get_shape()) if pshape.is_static else []
        inputs.append(ModelInputInfo(
            name=param.get_friendly_name(),
            shape=shape,
            element_type=str(param.get_output_element_type(0)),
        ))
    return inputs
