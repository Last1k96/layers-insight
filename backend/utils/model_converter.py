"""Model format detection and conversion to OpenVINO IR."""
from __future__ import annotations

from pathlib import Path

# Supported extensions mapped to format names
_EXT_MAP = {
    ".xml": "ir",
    ".onnx": "onnx",
    ".pb": "tf",
    ".pbtxt": "tf",
    ".tflite": "tflite",
    ".pt": "pytorch",
    ".pth": "pytorch",
}


def detect_model_format(path: Path) -> str:
    """Detect model format from file path.

    Returns one of: "ir", "onnx", "tf", "tflite", "pytorch", "saved_model".
    Raises ValueError if format is unrecognized.
    """
    if path.is_dir():
        # Check for TensorFlow SavedModel directory
        if (path / "saved_model.pb").exists():
            return "saved_model"
        raise ValueError(
            f"Directory '{path}' is not a recognized SavedModel "
            "(missing saved_model.pb)"
        )

    fmt = _EXT_MAP.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(
            f"Unrecognized model format: '{path.suffix}'. "
            f"Supported: {', '.join(sorted(_EXT_MAP.keys()))}, or SavedModel directory"
        )
    return fmt


def convert_to_ir(source_path: Path, output_dir: Path, ov_core) -> Path:
    """Convert a non-IR model to OpenVINO IR format.

    Reads the model via ov_core.read_model(), then saves as .xml/.bin
    into output_dir. Returns the path to the generated .xml file.
    """
    import openvino as ov

    model = ov_core.read_model(str(source_path))
    output_xml = output_dir / "model.xml"
    ov.save_model(model, str(output_xml))
    return output_xml


def get_available_frontends(ov_core) -> list[str]:
    """Query which model frontends are compiled into this OpenVINO build."""
    try:
        import openvino as ov
        fem = ov.frontend.FrontEndManager()
        return list(fem.get_available_front_ends())
    except Exception:
        return []
