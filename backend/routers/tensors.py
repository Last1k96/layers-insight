"""Tensor data routes for deep accuracy visualization."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

router = APIRouter(prefix="/api/tensors", tags=["tensors"])


_CONVERT_NPY_SCRIPT = '''\
#!/usr/bin/env python3
"""Convert between .npy and raw .bin formats using info.json metadata.

Usage:
    python convert_npy.py to_npy       # .bin -> .npy (using info.json)
    python convert_npy.py to_bin       # .npy -> .bin (updates info.json)
"""
import json
import sys
from pathlib import Path

import numpy as np


def to_npy():
    """Convert .bin files to .npy using shapes/dtypes from info.json."""
    info = json.loads(Path("info.json").read_text())
    for entry in info.get("inputs", []) + info.get("outputs", []):
        bin_file = Path(entry["file"])
        if not bin_file.exists():
            print(f"  skip {bin_file} (not found)")
            continue
        arr = np.frombuffer(bin_file.read_bytes(), dtype=entry["dtype"]).reshape(entry["shape"])
        npy_path = bin_file.with_suffix(".npy")
        np.save(str(npy_path), arr)
        print(f"  {bin_file} -> {npy_path}")


def to_bin():
    """Convert .npy files to .bin and update info.json with shapes/dtypes."""
    info = json.loads(Path("info.json").read_text())
    for entry in info.get("inputs", []) + info.get("outputs", []):
        npy_path = Path(entry["file"]).with_suffix(".npy")
        if not npy_path.exists():
            print(f"  skip {npy_path} (not found)")
            continue
        arr = np.load(str(npy_path))
        entry["shape"] = list(arr.shape)
        entry["dtype"] = str(arr.dtype)
        bin_path = npy_path.with_suffix(".bin")
        bin_path.write_bytes(arr.tobytes())
        print(f"  {npy_path} -> {bin_path}")
    Path("info.json").write_text(json.dumps(info, indent=2))
    print("  info.json updated")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("to_npy", "to_bin"):
        print(__doc__)
        sys.exit(1)
    {"to_npy": to_npy, "to_bin": to_bin}[sys.argv[1]]()
'''


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

    # Build info.json
    info: dict = {
        "node_name": node_name,
        "node_type": node_type,
        "main_device": session_detail.config.main_device,
        "ref_device": session_detail.config.ref_device,
        "metrics": task_data.get("metrics"),
        "inputs": [],
        "outputs": [],
        "session_config": {
            "ov_path": session_detail.config.ov_path,
            "main_device": session_detail.config.main_device,
            "ref_device": session_detail.config.ref_device,
            "input_precision": session_detail.config.input_precision,
        },
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        prefix = "reproducer/"

        # Add cut model files (.xml and .bin)
        for suffix in (".xml", ".bin"):
            model_file = tensor_dir / f"cut_model{suffix}"
            if model_file.exists():
                zf.write(str(model_file), f"{prefix}cut_model{suffix}")

        # Add input tensors (input_*.npy -> input_*.bin)
        for npy_path in sorted(tensor_dir.glob("input_*.npy")):
            arr = np.load(str(npy_path))
            bin_name = npy_path.stem + ".bin"
            zf.writestr(f"{prefix}{bin_name}", arr.tobytes())
            info["inputs"].append({
                "name": npy_path.stem,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "file": bin_name,
            })

        # Add output tensors (main_output.npy, ref_output.npy -> .bin)
        for output_name in ("main_output", "ref_output"):
            npy_path = tensor_dir / f"{output_name}.npy"
            if npy_path.exists():
                arr = np.load(str(npy_path))
                bin_name = f"{output_name}.bin"
                zf.writestr(f"{prefix}{bin_name}", arr.tobytes())
                info["outputs"].append({
                    "name": output_name,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "file": bin_name,
                })

        # Write info.json
        zf.writestr(f"{prefix}info.json", json.dumps(info, indent=2))

        # Write convert_npy.py helper
        zf.writestr(f"{prefix}convert_npy.py", _CONVERT_NPY_SCRIPT)

    buf.seek(0)
    safe_name = node_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="reproducer_{safe_name}.zip"'},
    )


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
