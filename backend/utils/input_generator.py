"""Random and file-based input generation for inference."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


PRECISION_MAP = {
    "fp32": np.float32,
    "fp16": np.float16,
    "f32": np.float32,
    "f16": np.float16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "i8": np.int8,
    "bool": np.bool_,
}


def resolve_shape(shape: list[int | str], resolved: list[int] | None = None) -> list[int]:
    """Return concrete shape. Uses resolved values for dynamic dims."""
    result = []
    res_idx = 0
    for i, d in enumerate(shape):
        if isinstance(d, int):
            result.append(d)
        elif resolved and res_idx < len(resolved):
            result.append(resolved[res_idx])
            res_idx += 1
        else:
            raise ValueError(f"Dimension {i} is dynamic ('{d}') and no concrete value provided.")
    return result


def has_dynamic_dims(shape: list[int | str]) -> bool:
    """Check if a shape contains any dynamic dimensions."""
    return any(isinstance(d, str) for d in shape)


def validate_shape_bounds(shape: list[int], lower: list[int], upper: list[int]) -> None:
    """Validate that each dimension of shape is within [lower, upper] bounds."""
    for i, (val, lo, hi) in enumerate(zip(shape, lower, upper)):
        if val < lo or val > hi:
            raise ValueError(f"Dimension {i} value {val} is outside bounds [{lo}, {hi}]")


def generate_random_input(
    shape: list[int],
    precision: str = "fp32",
) -> np.ndarray:
    """Generate random input data for a given shape and precision."""
    dtype = PRECISION_MAP.get(precision, np.float32)
    rng = np.random.default_rng()
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 11, size=shape, dtype=dtype)
    return rng.random(shape).astype(dtype)


def load_input_from_file(path: str, shape: list[int] | None = None, precision: str = "fp32") -> np.ndarray:
    """Load input data from a file (.npy or raw binary)."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(str(p))
    if p.suffix in (".bin", ".raw"):
        dtype = PRECISION_MAP.get(precision, np.float32)
        data = np.fromfile(str(p), dtype=dtype)
        if shape:
            data = data.reshape(shape)
        return data
    # Try loading as image via PIL
    if p.suffix in (".png", ".jpg", ".jpeg", ".bmp"):
        try:
            from PIL import Image
        except ImportError:
            raise ValueError(f"PIL required for image loading: {p}")
        img = Image.open(str(p)).convert("RGB")
        if shape and len(shape) == 4:
            # Assume NCHW layout: [N, C, H, W]
            _, c, h, w = shape
            img = img.resize((w, h), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32)  # [H, W, C]
            arr = arr.transpose(2, 0, 1)  # [C, H, W]
            arr = np.expand_dims(arr, 0)  # [1, C, H, W]
            return arr
        return np.array(img, dtype=np.float32)
    raise ValueError(f"Unsupported input file format: {p.suffix}")


def prepare_inputs(
    model_params: list[dict[str, Any]],
    input_path: str | None = None,
    precision: str = "fp32",
    input_configs: list[dict[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    """Prepare inputs for all model parameters.

    Args:
        model_params: List of dicts with 'name', 'shape', 'element_type' keys.
        input_path: Path to input data or None for random (legacy fallback).
        precision: Default precision for random inputs (legacy fallback).
        input_configs: Per-input config list with 'name', 'data_type', 'source', 'path'.
    """
    # Build per-input config lookup
    config_map: dict[str, dict[str, Any]] = {}
    if input_configs:
        for cfg in input_configs:
            config_map[cfg["name"]] = cfg

    inputs = {}
    for param in model_params:
        name = param["name"]
        shape = param["shape"]

        cfg = config_map.get(name)
        if cfg:
            # Use per-input config
            dt = cfg.get("data_type", precision)
            if cfg.get("source") == "file" and cfg.get("path"):
                # For file inputs, shape comes from the file itself
                file_shape = None if has_dynamic_dims(shape) else shape
                tensor = load_input_from_file(cfg["path"], file_shape, dt)
                # Validate file input shape against bounds if provided
                lo = cfg.get("lower_bounds", [])
                hi = cfg.get("upper_bounds", [])
                if lo and hi:
                    validate_shape_bounds(list(tensor.shape), lo, hi)
                inputs[name] = tensor
            else:
                # Resolve dynamic shapes for random generation
                if has_dynamic_dims(shape):
                    concrete_shape = resolve_shape(shape, cfg.get("resolved_shape"))
                else:
                    concrete_shape = shape
                # Validate concrete shape against bounds if provided
                lo = cfg.get("lower_bounds", [])
                hi = cfg.get("upper_bounds", [])
                if lo and hi:
                    validate_shape_bounds(concrete_shape, lo, hi)
                inputs[name] = generate_random_input(concrete_shape, dt)
        elif input_path and input_path != "random":
            p = Path(input_path)
            if p.is_dir():
                # Look for matching file by param name
                for ext in (".npy", ".bin", ".png", ".jpg", ".jpeg", ".bmp"):
                    candidate = p / f"{name}{ext}"
                    if candidate.exists():
                        inputs[name] = load_input_from_file(str(candidate), shape, precision)
                        break
                else:
                    inputs[name] = generate_random_input(shape, precision)
            else:
                inputs[name] = load_input_from_file(str(p), shape, precision)
        else:
            inputs[name] = generate_random_input(shape, precision)
    return inputs
