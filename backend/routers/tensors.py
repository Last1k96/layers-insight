"""Tensor data routes (Phase 2)."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/tensors", tags=["tensors"])

# Phase 2: GET /api/tensors/{task_id}/{output_name} -> binary tensor data
