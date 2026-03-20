"""Shared OpenVINO helper utilities."""
from __future__ import annotations

import sys
from pathlib import Path


def register_plugins(core, ov_path: str | None) -> list[str]:
    """Register OV plugins from a custom build path and return available devices.

    If *ov_path* is ``None`` or empty the core is returned unchanged and its
    current ``available_devices`` list is returned.
    """
    if not ov_path:
        return list(core.available_devices)

    ov_lib_dir = Path(ov_path)
    for so_file in ov_lib_dir.glob("libopenvino_*_plugin.so"):
        name = so_file.stem
        parts = name.replace("libopenvino_", "").replace("_plugin", "")
        device_name = parts.upper().replace("INTEL_", "")
        if device_name not in core.available_devices:
            try:
                core.register_plugin(str(so_file), device_name)
            except Exception as e:
                print(f"  Could not register {device_name} plugin: {e}", file=sys.stderr)

    return list(core.available_devices)
