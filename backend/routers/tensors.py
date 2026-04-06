"""Tensor data routes for deep accuracy visualization."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

router = APIRouter(prefix="/api/tensors", tags=["tensors"])


class _ZipStreamBuffer:
    """Write-only buffer for streaming ZIP generation.

    Accumulates bytes written by zipfile and lets the caller drain them
    after each entry, so the response can be streamed incrementally.
    zipfile in write mode only needs write() and tell() — no seeking.
    """

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._pos = 0

    def write(self, data: bytes) -> int:
        self._chunks.append(data)
        self._pos += len(data)
        return len(data)

    def tell(self) -> int:
        return self._pos

    def flush(self) -> None:
        pass

    def drain(self) -> bytes:
        data = b"".join(self._chunks)
        self._chunks.clear()
        return data


# --- Export route MUST be registered before the wildcard {output_name} routes
#     so that "/export" is matched as a fixed segment, not as output_name. ---

@router.get("/{session_id}/{task_id}/export")
async def export_reproducer(session_id: str, task_id: str, request: Request) -> StreamingResponse:
    """Export a reproducer ZIP package for filing bug reports.

    The ZIP contains the cut model, raw binary input/output tensors,
    metadata JSON, and a helper script to convert between .npy and .bin.
    """
    svc = request.app.state.session_service
    task_data = svc.load_task_result(session_id, task_id)

    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")
    if task_data.get("status") != "success":
        raise HTTPException(status_code=400, detail="Can only export successful tasks")

    # Locate the task's tensor directory
    tensor_dir = svc.get_task_dir(session_id, task_id)
    if tensor_dir is None or not tensor_dir.exists():
        raise HTTPException(status_code=404, detail="Task output directory not found")

    # Load session config for device info
    session_detail = svc.get_session(session_id)
    if session_detail is None:
        raise HTTPException(status_code=404, detail="Session not found")

    node_name = task_data.get("node_name", "unknown")
    node_type = task_data.get("node_type", "")

    # Build info.json with all available data
    info: dict = {
        "node_name": node_name,
        "node_type": node_type,
        "model_name": session_detail.info.model_name,
        "main_device": session_detail.config.main_device,
        "ref_device": session_detail.config.ref_device,
        "metrics": task_data.get("metrics"),
        "inputs": [],
        "outputs": [],
        "session_config": {
            "ov_path": session_detail.config.ov_path,
            "model_path": svc._read_metadata(session_id).get("original_model_path", session_detail.config.model_path),
            "main_device": session_detail.config.main_device,
            "ref_device": session_detail.config.ref_device,
            "input_precision": session_detail.config.input_precision,
        },
    }

    # Add device output stats
    if task_data.get("main_result"):
        info["main_result"] = task_data["main_result"]
    if task_data.get("ref_result"):
        info["ref_result"] = task_data["ref_result"]

    # Add per-output breakdowns for multi-output nodes
    if task_data.get("per_output_metrics"):
        info["per_output_metrics"] = task_data["per_output_metrics"]
    if task_data.get("per_output_main_results"):
        info["per_output_main_results"] = task_data["per_output_main_results"]
    if task_data.get("per_output_ref_results"):
        info["per_output_ref_results"] = task_data["per_output_ref_results"]

    # Add plugin configs if non-empty
    if session_detail.config.plugin_config:
        info["session_config"]["plugin_config"] = session_detail.config.plugin_config
    if session_detail.config.ref_plugin_config:
        info["session_config"]["ref_plugin_config"] = session_detail.config.ref_plugin_config
    if session_detail.config.original_format:
        info["session_config"]["original_format"] = session_detail.config.original_format

    safe_name = node_name.replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _generate() -> Iterator[bytes]:
        buf = _ZipStreamBuffer()
        prefix = "reproducer/"

        with zipfile.ZipFile(buf, "w") as zf:
            # Model files — .bin stored (weights don't compress), .xml deflated
            for suffix in (".xml", ".bin"):
                model_file = tensor_dir / f"cut_model{suffix}"
                if model_file.exists():
                    compress = zipfile.ZIP_DEFLATED if suffix == ".xml" else zipfile.ZIP_STORED
                    zf.writestr(f"{prefix}cut_model{suffix}", model_file.read_bytes(), compress_type=compress)
                    yield buf.drain()

            # Input tensors (input_*.npy -> input_*.bin) — stored uncompressed
            for npy_path in sorted(tensor_dir.glob("input_*.npy")):
                arr = np.load(str(npy_path))
                bin_name = npy_path.stem + ".bin"
                zf.writestr(f"{prefix}{bin_name}", arr.tobytes(), compress_type=zipfile.ZIP_STORED)
                info["inputs"].append({
                    "name": npy_path.stem,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "file": bin_name,
                })
                yield buf.drain()

            # Output tensors — stored uncompressed (float tensors don't compress).
            # Handle both single-output (main_output.npy) and multi-output
            # (main_output_0.npy, ...) patterns.  Skip unnumbered file when
            # indexed files exist to avoid duplicates.
            for side in ("main_output", "ref_output"):
                indexed = sorted(tensor_dir.glob(f"{side}_[0-9]*.npy"))
                if indexed:
                    for npy_path in indexed:
                        arr = np.load(str(npy_path))
                        bin_name = npy_path.stem + ".bin"
                        zf.writestr(f"{prefix}{bin_name}", arr.tobytes(), compress_type=zipfile.ZIP_STORED)
                        info["outputs"].append({
                            "name": npy_path.stem,
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                            "file": bin_name,
                        })
                        yield buf.drain()
                else:
                    npy_path = tensor_dir / f"{side}.npy"
                    if npy_path.exists():
                        arr = np.load(str(npy_path))
                        bin_name = f"{side}.bin"
                        zf.writestr(f"{prefix}{bin_name}", arr.tobytes(), compress_type=zipfile.ZIP_STORED)
                        info["outputs"].append({
                            "name": side,
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                            "file": bin_name,
                        })
                        yield buf.drain()

            # info.json last (needs accumulated inputs/outputs lists)
            zf.writestr(f"{prefix}info.json", json.dumps(info, indent=2), compress_type=zipfile.ZIP_DEFLATED)
            yield buf.drain()

        # ZipFile.close() writes central directory
        tail = buf.drain()
        if tail:
            yield tail

    return StreamingResponse(
        _generate(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="reproducer_{safe_name}.zip"'},
    )


@router.get("/{session_id}/{task_id}/{output_name}")
async def get_tensor(session_id: str, task_id: str, output_name: str, request: Request) -> Response:
    """Download tensor data as fp16 binary.

    The tensor is converted to float16 for efficient transfer.
    Client converts to Float32Array in browser.

    Supports both legacy names (main_output, ref_output) and indexed names
    (main_output_0, ref_output_1, etc.).
    """
    svc = request.app.state.session_service
    tensor_path = svc.get_tensor_path(session_id, task_id, output_name)

    # Backward compat: main_output -> main_output_0, ref_output -> ref_output_0
    if tensor_path is None and output_name in ("main_output", "ref_output"):
        tensor_path = svc.get_tensor_path(session_id, task_id, f"{output_name}_0")

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

    # Backward compat: main_output -> main_output_0, ref_output -> ref_output_0
    if tensor_path is None and output_name in ("main_output", "ref_output"):
        tensor_path = svc.get_tensor_path(session_id, task_id, f"{output_name}_0")

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
