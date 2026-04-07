"""Inference execution service — runs OpenVINO in a subprocess for crash isolation."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from backend.schemas.inference import AccuracyMetrics, DeviceResult, InferenceTask, TaskStatus


# Path to the subprocess worker script
_WORKER_SCRIPT = str(Path(__file__).parent.parent / "utils" / "inference_worker.py")




class InferenceService:
    """Handles graph cutting, inference, and accuracy comparison via subprocess."""

    def __init__(self, ov_core: Any, ov_path: str | None = None):
        self.core = ov_core
        self.ov_path = ov_path
        self._current_proc: Optional[subprocess.Popen] = None
        self._current_task_id: Optional[str] = None

    def kill_current(self, task_id: str) -> bool:
        """Kill the running subprocess for the given task. Returns True if killed."""
        if self._current_task_id == task_id and self._current_proc is not None:
            try:
                self._current_proc.kill()
                return True
            except OSError:
                return False
        return False

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
        log_callback: Optional[Callable[[str, str, str], None]] = None,
        stage_callback: Optional[Callable[[str], None]] = None,
        ov_log_level: str = "WARNING",
        runtime_dir: Optional[str] = None,
        plugin_config: Optional[dict[str, str]] = None,
        ref_plugin_config: Optional[dict[str, str]] = None,
    ) -> InferenceTask:
        """Cut the model at target node and run inference on both devices.

        Runs in a subprocess so OpenVINO C++ segfaults don't crash the server.

        Args:
            log_callback: Optional callback(task_id, level, message) for real-time log streaming.
            ov_log_level: OpenVINO log level to set in subprocess (default: WARNING).
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

        def _log(level: str, msg: str) -> None:
            if log_callback:
                log_callback(task.task_id, level, msg)

        def _update_stage(stage: str) -> None:
            """Update task stage and notify via callback."""
            if task.stage != stage:
                task.stage = stage
                if stage_callback:
                    stage_callback(stage)

        # Use session's runtime/ dir if provided, otherwise fall back to temp dir
        if runtime_dir:
            import shutil as _shutil
            rd = Path(runtime_dir)
            if rd.exists():
                _shutil.rmtree(rd)
            rd.mkdir(parents=True, exist_ok=True)
            tmp_dir = runtime_dir
            use_runtime_dir = True
        else:
            tmp_dir = tempfile.mkdtemp(prefix="li_infer_")
            use_runtime_dir = False
        success = False
        try:
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
                "ov_log_level": ov_log_level,
                "plugin_config": plugin_config or {},
                "ref_plugin_config": ref_plugin_config or {},
            }

            stderr_lines: list[str] = []      # all lines
            raw_stderr_lines: list[str] = []  # non-JSON lines only (for error reporting)

            try:
                proc = subprocess.Popen(
                    [sys.executable, _WORKER_SCRIPT],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self._current_proc = proc
                self._current_task_id = task.task_id

                # Write config to stdin and close it
                proc.stdin.write(json.dumps(worker_cfg))
                proc.stdin.close()

                # Set up timeout
                timer = threading.Timer(300, proc.kill)
                timer.start()

                try:
                    # Stream stderr line by line using readline() for real-time output.
                    # (Iterating with `for line in proc.stderr` buffers internally
                    #  and won't yield lines until the buffer fills or EOF.)
                    while True:
                        line = proc.stderr.readline()
                        if not line:
                            break
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        stderr_lines.append(line)
                        try:
                            entry = json.loads(line)
                            if entry.get("type") == "stage":
                                _update_stage(entry["stage"])
                            else:
                                _log(entry.get("level", "info"), entry.get("msg", line))
                        except json.JSONDecodeError:
                            # Raw OV output or other stderr — keep for error reporting
                            # Filter out "already registered" noise from LD_LIBRARY_PATH overlap
                            if "already registered" not in line:
                                raw_stderr_lines.append(line)
                            _log("ov", line)

                    proc.wait()
                finally:
                    timer.cancel()

            except Exception as e:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                # Check if it was a timeout (proc killed by timer)
                task.status = TaskStatus.FAILED
                task.error_detail = f"Subprocess error: {e}"
                return task

            # Check if process was killed by timeout or pause
            if proc.returncode == -9:
                # If pause already reset task to WAITING, don't overwrite
                if task.status == TaskStatus.WAITING:
                    return task
                task.status = TaskStatus.FAILED
                task.error_detail = "Inference timed out (300s)"
                _log("error", "Inference timed out (300s)")
                return task

            # Subprocess crashed (segfault = -11, other signals are negative)
            if proc.returncode != 0:
                stderr_tail = "\n".join(raw_stderr_lines)[-500:] if raw_stderr_lines else ""
                if proc.returncode < 0:
                    import signal
                    sig_name = signal.Signals(-proc.returncode).name
                    task.status = TaskStatus.FAILED
                    last_stage_info = f" Last stage: {task.stage}" if task.stage else ""
                    task.error_detail = (
                        f"OpenVINO crashed with {sig_name} (signal {-proc.returncode}) "
                        f"while processing node '{target_node_name}'.{last_stage_info} "
                        f"This node may not support cutting/inference on {main_device}."
                    )
                    if stderr_tail:
                        task.error_detail += f"\nstderr: {stderr_tail}"
                else:
                    task.status = TaskStatus.FAILED
                    task.error_detail = f"Worker exited with code {proc.returncode}"
                    if stderr_tail:
                        task.error_detail += f"\nstderr: {stderr_tail}"
                _log("error", task.error_detail)
                return task

            # Parse JSON result from stdout
            stdout = proc.stdout.read().strip()
            if not stdout:
                task.status = TaskStatus.FAILED
                task.error_detail = "Worker produced no output"
                stderr_tail = "\n".join(raw_stderr_lines)[-500:] if raw_stderr_lines else ""
                if stderr_tail:
                    task.error_detail += f"\nstderr: {stderr_tail}"
                _log("error", task.error_detail)
                return task

            try:
                result = json.loads(stdout)
            except json.JSONDecodeError:
                # OpenVINO C++ may have written warnings to stdout before the
                # fd-level redirect took effect.  Try to find the JSON object
                # (always the last thing emitted) by scanning from the end.
                result = None
                for i in range(len(stdout) - 1, -1, -1):
                    if stdout[i] == '{':
                        try:
                            result = json.loads(stdout[i:])
                            break
                        except json.JSONDecodeError:
                            continue
                if result is None:
                    task.status = TaskStatus.FAILED
                    task.error_detail = f"Worker output parse error: no valid JSON in stdout\nstdout: {stdout[:500]}"
                    _log("error", task.error_detail)
                    return task

            # Check for error from the worker (already logged via stderr)
            if "error" in result:
                task.status = TaskStatus.FAILED
                task.error_detail = result["error"]
                return task

            # Success — populate task
            task.main_result = DeviceResult(**result["main_result"])
            task.ref_result = DeviceResult(**result["ref_result"])
            task.metrics = AccuracyMetrics(**result["metrics"])
            # Multi-output: parse per-output breakdowns if present
            if "per_output_metrics" in result:
                task.per_output_metrics = [AccuracyMetrics(**m) for m in result["per_output_metrics"]]
                task.per_output_main_results = [DeviceResult(**r) for r in result["per_output_main_results"]]
                task.per_output_ref_results = [DeviceResult(**r) for r in result["per_output_ref_results"]]
            task.status = TaskStatus.SUCCESS
            task.stage = None

            # Return the artifacts directory path — caller is responsible for cleanup
            success = True
            return task, tmp_dir
        finally:
            self._current_proc = None
            self._current_task_id = None
            if not success and not use_runtime_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
