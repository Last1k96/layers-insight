"""Bisection search schemas."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class BisectSearchFor(str, Enum):
    ACCURACY_DROP = "accuracy_drop"
    COMPILATION_FAILURE = "compilation_failure"


class BisectMetric(str, Enum):
    COSINE_SIMILARITY = "cosine_similarity"
    MSE = "mse"
    MAX_ABS_DIFF = "max_abs_diff"


class BisectStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    STOPPED = "stopped"
    ERROR = "error"


class BisectRequest(BaseModel):
    """Request to start a bisection search."""
    session_id: str
    start_node: Optional[str] = None
    end_node: Optional[str] = None
    metric: BisectMetric = BisectMetric.COSINE_SIMILARITY
    threshold: float = 0.999
    search_for: BisectSearchFor = BisectSearchFor.ACCURACY_DROP
    sub_session_id: Optional[str] = None


class BisectStepInfo(BaseModel):
    """Info about a single bisection step."""
    node_name: str
    node_id: str
    task_id: Optional[str] = None
    metric_value: Optional[float] = None
    passed: Optional[bool] = None
    error: Optional[str] = None


class BisectProgress(BaseModel):
    """Current bisection state."""
    status: BisectStatus = BisectStatus.IDLE
    session_id: Optional[str] = None
    search_for: Optional[BisectSearchFor] = None
    metric: Optional[BisectMetric] = None
    threshold: Optional[float] = None
    range_start: Optional[str] = None
    range_end: Optional[str] = None
    current_node: Optional[str] = None
    step: int = 0
    total_steps: int = 0
    steps_history: list[BisectStepInfo] = []
    found_node: Optional[str] = None
    error: Optional[str] = None
