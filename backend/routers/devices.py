"""Device discovery and app defaults routes."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["devices"])


def _add_virtual_devices(devices: list[str]) -> list[str]:
    """Append virtual device names derived from real devices."""
    if "CPU" in devices and "CPU_fp16" not in devices:
        devices.append("CPU_fp16")
    return devices


@router.get("/devices")
async def list_devices(request: Request) -> list[str]:
    """List available OpenVINO devices."""
    ov_core = request.app.state.ov_core
    if ov_core is None:
        return ["CPU"]
    try:
        devices = list(ov_core.available_devices)
    except Exception:
        return ["CPU"]
    return _add_virtual_devices(devices)


class AppDefaults(BaseModel):
    ov_path: Optional[str] = None
    model_path: Optional[str] = None
    input_path: Optional[str] = None
    cli_inputs: list[str] = []
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
        cli_inputs=config.cli_inputs,
        main_device=config.main_device,
        ref_device=config.ref_device,
    )


class OvValidationResult(BaseModel):
    valid: bool
    devices: list[str]
    error: Optional[str] = None


@router.get("/validate-ov-path", response_model=OvValidationResult)
async def validate_ov_path(
    ov_path: Optional[str] = Query(None, description="Path to custom OpenVINO build"),
) -> OvValidationResult:
    """Validate an OV path by creating a fresh Core and registering plugins."""
    if not ov_path or not ov_path.strip():
        return OvValidationResult(valid=True, devices=["CPU"], error=None)

    ov_dir = Path(ov_path)
    if not ov_dir.exists():
        return OvValidationResult(valid=False, devices=["CPU"], error=f"Directory not found: {ov_path}")
    if not ov_dir.is_dir():
        return OvValidationResult(valid=False, devices=["CPU"], error=f"Not a directory: {ov_path}")

    plugins = list(ov_dir.glob("libopenvino_*_plugin.so"))
    if not plugins:
        return OvValidationResult(valid=False, devices=["CPU"], error=f"No OpenVINO plugins found in {ov_path}")

    try:
        import openvino as ov
        from backend.utils.ov_helpers import register_plugins

        core = ov.Core()
        devices = register_plugins(core, ov_path)
        return OvValidationResult(valid=True, devices=_add_virtual_devices(devices), error=None)
    except ImportError:
        return OvValidationResult(valid=False, devices=["CPU"], error="OpenVINO not installed")
    except Exception as e:
        return OvValidationResult(valid=False, devices=["CPU"], error=str(e))


class BrowseEntry(BaseModel):
    name: str
    path: str
    is_dir: bool


class BrowseResult(BaseModel):
    current: str
    parent: Optional[str]
    entries: list[BrowseEntry]


@router.get("/browse", response_model=BrowseResult)
async def browse_path(
    path: str = Query("", description="Directory to list"),
    mode: str = Query("file", description="'directory' or 'file'"),
) -> BrowseResult:
    """List directory contents for the file/directory picker."""
    if not path or not path.strip():
        path = str(Path.home())

    dir_path = Path(path).expanduser().resolve()
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")

    entries: list[BrowseEntry] = []
    try:
        for item in sorted(dir_path.iterdir(), key=lambda p: p.name.lower()):
            if item.name.startswith("."):
                continue
            if mode == "directory" and not item.is_dir():
                continue
            entries.append(BrowseEntry(name=item.name, path=str(item), is_dir=item.is_dir()))
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

    # Sort: directories first, then files
    entries.sort(key=lambda e: (not e.is_dir, e.name.lower()))

    parent = str(dir_path.parent) if dir_path.parent != dir_path else None

    return BrowseResult(current=str(dir_path), parent=parent, entries=entries)


class PathSuggestion(BaseModel):
    name: str
    path: str
    is_dir: bool


class PathSuggestResult(BaseModel):
    suggestions: list[PathSuggestion]


@router.get("/path-suggest", response_model=PathSuggestResult)
async def path_suggest(
    partial: str = Query("", description="Partial path to autocomplete"),
    mode: str = Query("file", description="'directory' or 'file'"),
) -> PathSuggestResult:
    """Suggest path completions for autocomplete."""
    if not partial or not partial.strip():
        return PathSuggestResult(suggestions=[])

    partial_path = Path(partial).expanduser()
    suggestions: list[PathSuggestion] = []

    # If partial is a valid directory, list its children
    if partial_path.is_dir() and partial.endswith("/"):
        try:
            for item in sorted(partial_path.iterdir(), key=lambda p: p.name.lower()):
                if item.name.startswith("."):
                    continue
                if mode == "directory" and not item.is_dir():
                    continue
                suggestions.append(PathSuggestion(name=item.name, path=str(item), is_dir=item.is_dir()))
                if len(suggestions) >= 20:
                    break
        except PermissionError:
            pass
    else:
        # Split into parent + prefix, match entries starting with prefix
        parent = partial_path.parent
        prefix = partial_path.name.lower()
        if parent.is_dir():
            try:
                for item in sorted(parent.iterdir(), key=lambda p: p.name.lower()):
                    if item.name.startswith("."):
                        continue
                    if not item.name.lower().startswith(prefix):
                        continue
                    if mode == "directory" and not item.is_dir():
                        continue
                    suggestions.append(PathSuggestion(name=item.name, path=str(item), is_dir=item.is_dir()))
                    if len(suggestions) >= 20:
                        break
            except PermissionError:
                pass

    # Sort: directories first, then files
    suggestions.sort(key=lambda s: (not s.is_dir, s.name.lower()))
    return PathSuggestResult(suggestions=suggestions)


class ModelInputInfo(BaseModel):
    name: str
    shape: list[int]
    element_type: str


class FrontendInfo(BaseModel):
    frontends: list[str]
    supported_formats: list[str]


_FRONTEND_FORMAT_MAP = {
    "onnx": [".onnx"],
    "tf": [".pb", ".pbtxt", "SavedModel"],
    "tflite": [".tflite"],
    "pytorch": [".pt", ".pth"],
    "paddle": [".pdmodel"],
}


@router.get("/frontends", response_model=FrontendInfo)
async def get_frontends(request: Request) -> FrontendInfo:
    """List available OpenVINO frontends and the model formats they support."""
    from backend.utils.model_converter import get_available_frontends

    ov_core = request.app.state.ov_core
    frontends = get_available_frontends(ov_core) if ov_core else []
    formats = [".xml"]  # IR is always supported
    for fe in frontends:
        formats.extend(_FRONTEND_FORMAT_MAP.get(fe, []))
    return FrontendInfo(frontends=frontends, supported_formats=sorted(set(formats)))


@router.get("/model-inputs", response_model=list[ModelInputInfo])
async def get_model_inputs(
    request: Request,
    model_path: str = Query(..., description="Path to model file"),
    ov_path: Optional[str] = Query(None, description="Custom OpenVINO path"),
) -> list[ModelInputInfo]:
    """Read a model file and return its input parameter info."""
    # Use a fresh Core with custom OV path if provided, otherwise fall back to app core
    if ov_path and ov_path.strip():
        try:
            import openvino as ov
            from backend.utils.ov_helpers import register_plugins

            ov_core = ov.Core()
            register_plugins(ov_core, ov_path)
        except ImportError:
            raise HTTPException(status_code=503, detail="OpenVINO not installed")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to initialize OV with path {ov_path}: {e}")
    else:
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
