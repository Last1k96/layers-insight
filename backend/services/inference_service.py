"""Inference execution service — runs OpenVINO in a subprocess for crash isolation."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.schemas.inference import AccuracyMetrics, DeviceResult, InferenceTask, TaskStatus


# Path to the subprocess worker script
_WORKER_SCRIPT = str(Path(__file__).parent.parent / "utils" / "inference_worker.py")


class InferenceService:
    """Handles graph cutting, inference, and accuracy comparison via subprocess."""

    def __init__(self, ov_core: Any, ov_path: str | None = None):
        self.core = ov_core
        self.ov_path = ov_path

    def cut_and_infer(
        self,
        model: Any,
        target_node_name: str,
        main_device: str,
        ref_device: str,
        model_path: str,
        input_path: Optional[str] = None,
        precision: str = "fp32",
        task: Optional[InferenceTask] = None,
        input_configs: Optional[list[dict]] = None,
    ) -> InferenceTask:
        """Cut the model at target node and run inference on both devices.

        Runs in a subprocess so OpenVINO C++ segfaults don't crash the server.
        """
        if task is None:
            task = InferenceTask(
                task_id="standalone",
                session_id="",
                node_id=target_node_name,
                node_name=target_node_name,
                node_type="",
            )

        task.status = TaskStatus.EXECUTING
        task.stage = "cutting_graph"

        with tempfile.TemporaryDirectory(prefix="li_infer_") as tmp_dir:
            # Build config for the subprocess worker
            worker_cfg = {
                "model_path": model_path,
                "node_name": target_node_name,
                "main_device": main_device,
                "ref_device": ref_device,
                "ov_path": self.ov_path,
                "input_path": input_path,
                "precision": precision,
                "input_configs": input_configs,
                "out_dir": tmp_dir,
            }

            try:
                proc = subprocess.run(
                    [sys.executable, _WORKER_SCRIPT],
                    input=json.dumps(worker_cfg),
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 min timeout
                )
            except subprocess.TimeoutExpired:
                task.status = TaskStatus.FAILED
                task.error_detail = "Inference timed out (300s)"
                return task

            # Subprocess crashed (segfault = -11, other signals are negative)
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "")[-500:]
                if proc.returncode < 0:
                    import signal
                    sig_name = signal.Signals(-proc.returncode).name
                    task.status = TaskStatus.FAILED
                    task.error_detail = (
                        f"OpenVINO crashed with {sig_name} (signal {-proc.returncode}) "
                        f"while processing node '{target_node_name}'. "
                        f"This node may not support cutting/inference on {main_device}."
                    )
                    if stderr_tail:
                        task.error_detail += f"\nstderr: {stderr_tail}"
                else:
                    task.status = TaskStatus.FAILED
                    task.error_detail = f"Worker exited with code {proc.returncode}"
                    if stderr_tail:
                        task.error_detail += f"\nstderr: {stderr_tail}"
                return task

            # Parse JSON result from stdout
            stdout = proc.stdout.strip()
            if not stdout:
                task.status = TaskStatus.FAILED
                task.error_detail = "Worker produced no output"
                if proc.stderr:
                    task.error_detail += f"\nstderr: {proc.stderr[-500:]}"
                return task

            try:
                result = json.loads(stdout)
            except json.JSONDecodeError as e:
                task.status = TaskStatus.FAILED
                task.error_detail = f"Worker output parse error: {e}\nstdout: {stdout[:500]}"
                return task

            # Check for error from the worker
            if "error" in result:
                task.status = TaskStatus.FAILED
                task.error_detail = result["error"]
                return task

            # Success — populate task
            task.main_result = DeviceResult(**result["main_result"])
            task.ref_result = DeviceResult(**result["ref_result"])
            task.metrics = AccuracyMetrics(**result["metrics"])
            task.status = TaskStatus.SUCCESS
            task.stage = None

            # Load numpy outputs
            main_output = None
            ref_output = None
            main_npy = Path(tmp_dir) / "main_output.npy"
            ref_npy = Path(tmp_dir) / "ref_output.npy"
            if main_npy.exists():
                main_output = np.load(str(main_npy))
            if ref_npy.exists():
                ref_output = np.load(str(ref_npy))

            return task, main_output, ref_output
