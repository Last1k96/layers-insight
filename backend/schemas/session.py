"""Session schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionConfig(BaseModel):
    """Configuration for a new session."""
    ov_path: Optional[str] = None
    model_path: str
    input_path: Optional[str] = None  # path or None for random
    main_device: str = "CPU"
    ref_device: str = "CPU"
    input_precision: str = "fp32"
    input_layout: str = "NCHW"


class SessionInfo(BaseModel):
    """Session metadata."""
    id: str
    model_path: str
    model_name: str
    created_at: str
    main_device: str
    ref_device: str
    task_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    # Phase 2 extensibility
    sub_sessions: list[Any] = []


class SessionDetail(BaseModel):
    """Full session detail including config."""
    id: str
    config: SessionConfig
    info: SessionInfo
    tasks: list[Any] = []  # list of InferenceTask summaries
