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


def load_input_from_file(path: str, shape: list[int] | None = None) -> np.ndarray:
    """Load input data from a file (.npy or raw binary)."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(str(p))
    if p.suffix in (".bin", ".raw"):
        data = np.fromfile(str(p), dtype=np.float32)
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
                inputs[name] = load_input_from_file(cfg["path"], shape)
            else:
                inputs[name] = generate_random_input(shape, dt)
        elif input_path and input_path != "random":
            p = Path(input_path)
            if p.is_dir():
                # Look for matching file by param name
                for ext in (".npy", ".bin"):
                    candidate = p / f"{name}{ext}"
                    if candidate.exists():
                        inputs[name] = load_input_from_file(str(candidate), shape)
                        break
                else:
                    inputs[name] = generate_random_input(shape, precision)
            else:
                inputs[name] = load_input_from_file(str(p), shape)
        else:
            inputs[name] = generate_random_input(shape, precision)
    return inputs
