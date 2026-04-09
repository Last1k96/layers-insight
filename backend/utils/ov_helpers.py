"""Shared OpenVINO helper utilities."""
from __future__ import annotations

import sys
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"
# Linux: libopenvino_<device>_plugin.so   Windows: openvino_<device>_plugin.dll
_PLUGIN_GLOB = "openvino_*_plugin.dll" if _IS_WINDOWS else "libopenvino_*_plugin.so"
_LIB_PREFIX = "" if _IS_WINDOWS else "lib"


def register_plugins(core, ov_path: str | None) -> list[str]:
    """Register OV plugins from a custom build path and return available devices.

    If *ov_path* is ``None`` or empty the core is returned unchanged and its
    current ``available_devices`` list is returned.
    """
    if not ov_path:
        return list(core.available_devices)

    ov_lib_dir = Path(ov_path)
    for so_file in ov_lib_dir.glob(_PLUGIN_GLOB):
        name = so_file.stem
        parts = name.replace(f"{_LIB_PREFIX}openvino_", "").replace("_plugin", "")
        device_name = parts.upper().replace("INTEL_", "")
        if device_name == "TEMPLATE":
            continue
        if device_name not in core.available_devices:
            try:
                core.register_plugin(str(so_file), device_name)
            except Exception as e:
                # Silently skip "already registered" errors — this happens when
                # LD_LIBRARY_PATH already contains the OV build directory and
                # ov.Core() auto-discovers the plugins before we call this.
                if "already registered" in str(e):
                    continue
                print(f"  Could not register {device_name} plugin: {e}", file=sys.stderr)

    return list(core.available_devices)
