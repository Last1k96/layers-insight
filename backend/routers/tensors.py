"""Tensor data routes for deep accuracy visualization."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

router = APIRouter(prefix="/api/tensors", tags=["tensors"])


@router.get("/{session_id}/{task_id}/{output_name}")
async def get_tensor(session_id: str, task_id: str, output_name: str, request: Request) -> Response:
    """Download tensor data as fp16 binary.

    The tensor is converted to float16 for efficient transfer.
    Client converts to Float32Array in browser.
    """
    svc = request.app.state.session_service
    tensor_path = svc.get_tensor_path(session_id, task_id, output_name)

    if tensor_path is None:
        raise HTTPException(status_code=404, detail="Tensor not found")

    data = np.load(str(tensor_path))

    # Convert to fp16 for efficient transfer
    fp16_data = data.astype(np.float16)

    # Include shape info in headers for client reconstruction
    shape_str = ",".join(str(d) for d in data.shape)
    dtype_str = str(data.dtype)

    return Response(
        content=fp16_data.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Tensor-Shape": shape_str,
            "X-Tensor-Dtype": dtype_str,
            "X-Tensor-Original-Dtype": dtype_str,
        },
    )


@router.get("/{session_id}/{task_id}/{output_name}/meta")
async def get_tensor_meta(session_id: str, task_id: str, output_name: str, request: Request) -> dict:
    """Get tensor metadata without downloading the data."""
    svc = request.app.state.session_service
    tensor_path = svc.get_tensor_path(session_id, task_id, output_name)

    if tensor_path is None:
        raise HTTPException(status_code=404, detail="Tensor not found")

    data = np.load(str(tensor_path))

    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "size_bytes": data.nbytes,
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
    }
