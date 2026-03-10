"""Session schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class InputConfig(BaseModel):
    """Per-input configuration for model inputs."""
    name: str
    shape: list[int] = []
    element_type: str = ""
    data_type: str = "fp32"  # precision to use for generation
    source: str = "random"  # "random" or "file"
    path: Optional[str] = None  # file path if source is "file"
    layout: str = ""  # e.g. "NCHW", "NHWC", "NCW", "NWC", etc.


class SessionConfig(BaseModel):
    """Configuration for a new session."""
    ov_path: Optional[str] = None
    model_path: str
    input_path: Optional[str] = None  # path or None for random (legacy, overridden by inputs)
    main_device: str = "CPU"
    ref_device: str = "CPU"
    input_precision: str = "fp32"
    input_layout: str = "NCHW"
    inputs: Optional[list[InputConfig]] = None  # per-input config


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
    sub_sessions: list["SubSessionInfo"] = []


class SubSessionInfo(BaseModel):
    """Sub-session created by model cutting."""
    id: str
    parent_id: str  # parent session or sub-session id
    cut_type: str  # "output" or "input"
    cut_node: str  # node name where cut was made
    grayed_nodes: list[str] = []
    created_at: str
    task_count: int = 0
    success_count: int = 0
    failed_count: int = 0


class CutRequest(BaseModel):
    """Request to cut the model at a node."""
    node_name: str
    cut_type: str  # "output" or "input"
    input_precision: str = "f16"  # for make_input_node


class SessionDetail(BaseModel):
    """Full session detail including config."""
    id: str
    config: SessionConfig
    info: SessionInfo
    tasks: list[Any] = []  # list of InferenceTask summaries
