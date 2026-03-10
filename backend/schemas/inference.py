"""Inference task schemas."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Inference task status."""
    WAITING = "waiting"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"


class DeviceResult(BaseModel):
    """Result from a single device inference."""
    device: str
    output_shapes: list[list[int]] = []
    dtype: Optional[str] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean_val: Optional[float] = None
    std_val: Optional[float] = None


class AccuracyMetrics(BaseModel):
    """Accuracy comparison metrics between two devices."""
    mse: float
    max_abs_diff: float
    cosine_similarity: float


class InferenceTask(BaseModel):
    """A queued inference task."""
    task_id: str
    session_id: str
    node_id: str
    node_name: str
    node_type: str
    status: TaskStatus = TaskStatus.WAITING
    stage: Optional[str] = None
    error_detail: Optional[str] = None
    main_result: Optional[DeviceResult] = None
    ref_result: Optional[DeviceResult] = None
    metrics: Optional[AccuracyMetrics] = None
    # Phase 2 extensibility
    batch_id: Optional[str] = None
    sub_session_id: Optional[str] = None
    # Phase 3 extensibility
    bisect_id: Optional[str] = None


class EnqueueRequest(BaseModel):
    """Request to enqueue a node for inference."""
    session_id: str
    node_id: str
    node_name: str
    node_type: str


class ReorderRequest(BaseModel):
    """Request to reorder queued tasks."""
    task_ids: list[str]
