"""Application configuration with CLI argument support."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    """Application settings, overridable via CLI args or env vars."""
    ov_path: Optional[str] = None
    model_path: Optional[str] = None
    input_path: Optional[str] = None  # path or "random"
    main_device: str = "CPU"
    ref_device: str = "CPU"
    port: int = 8000
    host: str = "0.0.0.0"
    sessions_dir: Path = Path("sessions")

    model_config = {"env_prefix": "LI_"}


def parse_cli_args() -> dict:
    """Parse CLI arguments, returning only explicitly provided values."""
    parser = argparse.ArgumentParser(description="Layers-Insight: Neural Network Graph Debugger")
    parser.add_argument("--ov-path", help="Path to OpenVINO binaries")
    parser.add_argument("--model", dest="model_path", help="Path to model .xml file")
    parser.add_argument("--input", dest="input_path", help="Input data path or 'random'")
    parser.add_argument("--main-device", help="Main inference device (e.g., GPU)")
    parser.add_argument("--ref-device", help="Reference inference device (e.g., CPU)")
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--host", help="Server host")
    parser.add_argument("--sessions-dir", help="Directory for session storage")

    args = parser.parse_args()
    result = {k: v for k, v in vars(args).items() if v is not None}
    # Resolve sessions_dir to absolute so CWD changes don't break paths
    if "sessions_dir" in result:
        result["sessions_dir"] = Path(result["sessions_dir"]).resolve()
    else:
        result["sessions_dir"] = Path("sessions").resolve()
    return result
