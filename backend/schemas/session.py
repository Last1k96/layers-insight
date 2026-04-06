"""Session schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class InputConfig(BaseModel):
    """Per-input configuration for model inputs."""
    name: str
    shape: list[int | str] = []  # original shape from model, may have "?" for dynamic dims
    element_type: str = ""
    data_type: str = "fp32"  # precision to use for generation
    source: str = "random"  # "random" or "file"
    path: Optional[str] = None  # file path if source is "file"
    layout: str = ""  # e.g. "NCHW", "NHWC", "NCW", "NWC", etc.
    resolved_shape: list[int] = []  # user-specified concrete shape for random generation (dynamic dims)
    lower_bounds: list[int] = []  # per-dim lower bounds for model reshape
    upper_bounds: list[int] = []  # per-dim upper bounds for model reshape


class SessionConfig(BaseModel):
    """Configuration for a new session."""
    ov_path: Optional[str] = None
    model_path: str
    session_name: Optional[str] = None
    input_path: Optional[str] = None  # path or None for random (legacy, overridden by inputs)
    main_device: str = "CPU"
    ref_device: str = "CPU"
    input_precision: str = "fp32"
    input_layout: str = "NCHW"
    inputs: Optional[list[InputConfig]] = None  # per-input config
    original_format: Optional[str] = None  # original model format before conversion (e.g. "onnx")
    plugin_config: dict[str, str] = {}  # main device plugin properties
    ref_plugin_config: dict[str, str] = {}  # reference device plugin properties


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
    folder_size: int = 0
    # Phase 2 extensibility
    sub_sessions: list["SubSessionInfo"] = []


class SubSessionInfo(BaseModel):
    """Sub-session created by model cutting."""
    id: str
    parent_id: str  # parent session or sub-session id
    cut_type: str  # "output" or "input"
    cut_node: str  # node name where cut was made
    grayed_nodes: list[str] = []
    ancestor_cuts: list[dict] = []  # [{cut_node, cut_type}] for all ancestors
    created_at: str
    task_count: int = 0
    success_count: int = 0
    failed_count: int = 0


class RenameRequest(BaseModel):
    """Request to rename a session."""
    name: str


class CutRequest(BaseModel):
    """Request to cut the model at a node."""
    node_name: str
    cut_type: str  # "output", "input", or "input_random"
    input_precision: str = "f16"  # for make_input_node
    parent_sub_session_id: Optional[str] = None  # cut from a sub-session's model


class CloneRequest(BaseModel):
    """Request to clone a session with optional overrides."""
    model_path: Optional[str] = None
    main_device: Optional[str] = None
    ref_device: Optional[str] = None
    inputs: Optional[list[InputConfig]] = None
    plugin_config: Optional[dict[str, Any]] = None


class CloneEnqueueRequest(BaseModel):
    """Request to batch-enqueue nodes from a source session into a target session."""
    target_session_id: str
    node_names: list[str]


class CompareNodeResult(BaseModel):
    """Per-node comparison result across two sessions."""
    node_name: str
    node_type: str
    metrics_a: Optional[Any] = None  # AccuracyMetrics dict or None
    metrics_b: Optional[Any] = None
    delta_cosine: Optional[float] = None
    delta_mse: Optional[float] = None


class CompareSummary(BaseModel):
    """Summary statistics for a session comparison."""
    total_compared: int = 0
    improved: int = 0
    regressed: int = 0
    unchanged: int = 0
    only_in_a: int = 0
    only_in_b: int = 0


class CompareResponse(BaseModel):
    """Response for session comparison endpoint."""
    nodes: list[CompareNodeResult] = []
    summary: CompareSummary = CompareSummary()


class SessionDetail(BaseModel):
    """Full session detail including config."""
    id: str
    config: SessionConfig
    info: SessionInfo
    tasks: list[Any] = []  # list of InferenceTask summaries
