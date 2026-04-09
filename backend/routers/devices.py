"""Device discovery and app defaults routes."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from backend.services.graph_service import _normalize_element_type

router = APIRouter(prefix="/api", tags=["devices"])

# Map OV's long element-type names to the short form accepted by compile_model
_OV_TYPE_SHORT: dict[str, str] = {
    "float32": "f32", "float16": "f16", "bfloat16": "bf16",
    "int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64",
    "uint8": "u8", "uint16": "u16", "uint32": "u32", "uint64": "u64",
}


def _normalize_ov_value(raw: object) -> str:
    """Normalize any OV property value to its shortest plain-string form.

    Handles all known representations:
      - pybind wrappers:  ``<Type: 'bfloat16'>``  → ``bf16``
                          ``<PerformanceMode: 'LATENCY'>``  → ``LATENCY``
      - Python enums:     ``PerformanceMode.LATENCY``  → ``LATENCY``
      - OV type names:    ``bfloat16``  → ``bf16``
      - plain strings:    ``LATENCY``  → ``LATENCY``  (passthrough)
    """
    s = str(raw)

    # 1) pybind wrapper: <ClassName: 'value'>
    m = re.search(r"<(\w+):\s*'([^']+)'>", s)
    if m:
        cls, inner = m.group(1), m.group(2)
        return _OV_TYPE_SHORT.get(inner, inner) if cls == "Type" else inner

    # 2) Python enum: ClassName.MEMBER
    if "." in s and not s.startswith("/") and not s.startswith("."):
        parts = s.rsplit(".", 1)
        # Enum class names are CamelCase — plain dotted paths (file paths) are not
        if parts[0] and parts[0][0].isupper() and parts[0].isidentifier():
            return _OV_TYPE_SHORT.get(parts[1], parts[1])

    # 3) Bare OV type name (e.g. value already stringified elsewhere)
    if s in _OV_TYPE_SHORT:
        return _OV_TYPE_SHORT[s]

    return s


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
    return devices


class DeviceProperty(BaseModel):
    name: str
    value: str
    type: str  # "bool", "string", "int", "enum"
    options: list[str] = []  # for enum type


@router.get("/device-config/{device_name}", response_model=list[DeviceProperty])
async def get_device_config(device_name: str, request: Request) -> list[DeviceProperty]:
    """Query available configuration properties for a device."""
    ov_core = request.app.state.ov_core
    if ov_core is None:
        return []

    try:
        supported = ov_core.get_property(device_name, "SUPPORTED_PROPERTIES")
    except Exception:
        return []

    # Known enum-like properties and their valid option values.
    # Source: OpenVINO 2024+ documentation / ov::hint, ov::log, ov::affinity enums.
    known_enum_options: dict[str, list[str]] = {
        "LOG_LEVEL": ["LOG_NONE", "LOG_ERROR", "LOG_WARNING", "LOG_INFO", "LOG_DEBUG", "LOG_TRACE"],
        "PERFORMANCE_HINT": ["", "LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"],
        "INFERENCE_PRECISION_HINT": ["f32", "f16", "bf16", "i8"],
        "SCHEDULING_CORE_TYPE": ["ANY_CORE", "PCORE_ONLY", "ECORE_ONLY"],
        "HINT_EXECUTION_MODE": ["PERFORMANCE", "ACCURACY"],
        "CACHE_MODE": ["OPTIMIZE_SIZE", "OPTIMIZE_SPEED"],
        "AFFINITY": ["CORE", "NUMA", "HYBRID_AWARE"],
    }

    # Properties to skip (internal, read-only, or cause issues)
    skip_props = {
        "SUPPORTED_PROPERTIES", "SUPPORTED_CONFIG_KEYS", "OPTIMIZATION_CAPABILITIES",
        "RANGE_FOR_ASYNC_INFER_REQUESTS", "RANGE_FOR_STREAMS", "FULL_DEVICE_NAME",
        "DEVICE_ARCHITECTURE", "DEVICE_TYPE", "DEVICE_GOPS", "DEVICE_UUID",
        "AVAILABLE_DEVICES", "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
        "CACHING_PROPERTIES", "LOADED_FROM_CACHE",
    }

    properties: list[DeviceProperty] = []
    for prop_name in supported:
        if prop_name in skip_props:
            continue
        try:
            value = ov_core.get_property(device_name, prop_name)
        except Exception:
            continue

        # Normalize OV wrappers / enums / type names to shortest plain string
        str_value = _normalize_ov_value(value)

        if prop_name in known_enum_options:
            prop_type = "enum"
            options = list(known_enum_options[prop_name])
            # Ensure current value is selectable even if not in known list
            if str_value not in options:
                options.insert(0, str_value)
        elif isinstance(value, bool):
            prop_type = "bool"
            options = []
        elif isinstance(value, int):
            prop_type = "int"
            options = []
        elif str_value.lower() in ("yes", "no", "true", "false"):
            prop_type = "bool"
            options = []
        else:
            prop_type = "string"
            options = []

        properties.append(DeviceProperty(
            name=prop_name,
            value=str_value,
            type=prop_type,
            options=options,
        ))

    return properties


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

    from backend.utils.ov_helpers import _PLUGIN_GLOB
    plugins = list(ov_dir.glob(_PLUGIN_GLOB))
    if not plugins:
        return OvValidationResult(valid=False, devices=["CPU"], error=f"No OpenVINO plugins found in {ov_path}")

    try:
        import openvino as ov
        from backend.utils.ov_helpers import register_plugins

        core = ov.Core()
        devices = register_plugins(core, ov_path)
        return OvValidationResult(valid=True, devices=devices, error=None)
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
    # Walk up to the nearest existing directory
    if not dir_path.is_dir():
        dir_path = dir_path.parent
    while not dir_path.exists() and dir_path != dir_path.parent:
        dir_path = dir_path.parent
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

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


class PathCheckResult(BaseModel):
    exists: bool
    is_file: bool = False
    is_dir: bool = False


@router.get("/check-path", response_model=PathCheckResult)
async def check_path(
    path: str = Query(..., description="Path to check"),
) -> PathCheckResult:
    """Check whether a file or directory exists at the given path."""
    p = Path(path).expanduser()
    if not p.exists():
        return PathCheckResult(exists=False)
    return PathCheckResult(exists=True, is_file=p.is_file(), is_dir=p.is_dir())


class ModelInputInfo(BaseModel):
    name: str
    shape: list[int | str]
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
        if pshape.is_static:
            shape = [d.get_length() for d in pshape]
        else:
            shape = [d.get_length() if d.is_static else "?" for d in pshape]
        inputs.append(ModelInputInfo(
            name=param.get_friendly_name(),
            shape=shape,
            element_type=_normalize_element_type(param.get_output_element_type(0)),
        ))
    return inputs
