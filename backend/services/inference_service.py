"""Inference execution service — graph cutting, compilation, and comparison."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from backend.schemas.inference import AccuracyMetrics, DeviceResult, InferenceTask, TaskStatus
from backend.utils.input_generator import prepare_inputs


class InferenceService:
    """Handles graph cutting, inference, and accuracy comparison."""

    def __init__(self, ov_core: Any):
        self.core = ov_core

    def cut_and_infer(
        self,
        model: Any,
        target_node_name: str,
        main_device: str,
        ref_device: str,
        input_path: Optional[str] = None,
        precision: str = "fp32",
        task: Optional[InferenceTask] = None,
    ) -> InferenceTask:
        """Cut the model at target node and run inference on both devices.

        All errors are captured in the task result, never raised.
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

        # Step 1: Find target op and cut model
        task.stage = "cutting_graph"
        try:
            target_op = self._find_op(model, target_node_name)
            if target_op is None:
                task.status = TaskStatus.FAILED
                task.stage = "cutting_graph"
                task.error_detail = f"Node '{target_node_name}' not found in model"
                return task

            new_outputs = [target_op.output(i) for i in range(target_op.get_output_size())]
            import openvino as ov
            cut_model = ov.Model(new_outputs, model.get_parameters(), f"cut_at_{target_node_name}")
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.stage = "cutting_graph"
            task.error_detail = str(e)
            return task

        # Step 2: Prepare inputs
        task.stage = "preparing_inputs"
        try:
            model_params = self._extract_params(model)
            inputs = prepare_inputs(model_params, input_path, precision)
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.stage = "preparing_inputs"
            task.error_detail = str(e)
            return task

        # Step 3: Run on main device
        task.stage = f"compiling_{main_device}"
        main_output, main_result, error = self._run_on_device(cut_model, main_device, inputs)
        if error:
            task.status = TaskStatus.FAILED
            task.error_detail = error
            return task
        task.main_result = main_result

        # Step 4: Run on reference device
        task.stage = f"compiling_{ref_device}"
        ref_output, ref_result, error = self._run_on_device(cut_model, ref_device, inputs)
        if error:
            task.status = TaskStatus.FAILED
            task.error_detail = error
            return task
        task.ref_result = ref_result

        # Step 5: Compute accuracy metrics
        task.stage = "computing_metrics"
        try:
            if main_output is not None and ref_output is not None:
                task.metrics = self._compute_metrics(main_output, ref_output)
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.stage = "computing_metrics"
            task.error_detail = str(e)
            return task

        task.status = TaskStatus.SUCCESS
        task.stage = None
        return task, main_output, ref_output

    def _find_op(self, model: Any, name: str) -> Any:
        """Find an operation by friendly name."""
        for op in model.get_ordered_ops():
            if op.get_friendly_name() == name:
                return op
        return None

    def _extract_params(self, model: Any) -> list[dict]:
        """Extract parameter info from model."""
        params = []
        for param in model.get_parameters():
            pshape = param.get_output_partial_shape(0)
            shape = list(pshape.get_shape()) if pshape.is_static else [1, 3, 224, 224]
            params.append({
                "name": param.get_friendly_name(),
                "shape": shape,
                "element_type": str(param.get_output_element_type(0)),
            })
        return params

    def _run_on_device(
        self, model: Any, device: str, inputs: dict
    ) -> tuple[Optional[np.ndarray], Optional[DeviceResult], Optional[str]]:
        """Run inference on a single device. Returns (output, result, error)."""
        try:
            compiled = self.core.compile_model(model, device)
        except Exception as e:
            return None, None, f"Compilation on {device} failed: {e}"

        try:
            infer_request = compiled.create_infer_request()
            infer_request.infer(inputs)

            # Get first output tensor
            output = infer_request.get_output_tensor(0).data.copy()

            result = DeviceResult(
                device=device,
                output_shapes=[list(output.shape)],
                dtype=str(output.dtype),
                min_val=float(np.min(output)),
                max_val=float(np.max(output)),
                mean_val=float(np.mean(output)),
                std_val=float(np.std(output)),
            )
            return output, result, None
        except Exception as e:
            return None, None, f"Inference on {device} failed: {e}"

    def _compute_metrics(self, main: np.ndarray, ref: np.ndarray) -> AccuracyMetrics:
        """Compute accuracy metrics between two output tensors."""
        main_flat = main.flatten().astype(np.float64)
        ref_flat = ref.flatten().astype(np.float64)

        diff = main_flat - ref_flat
        mse = float(np.mean(diff ** 2))
        max_abs_diff = float(np.max(np.abs(diff)))

        # Cosine similarity
        dot = np.dot(main_flat, ref_flat)
        norm_main = np.linalg.norm(main_flat)
        norm_ref = np.linalg.norm(ref_flat)
        if norm_main > 0 and norm_ref > 0:
            cosine_sim = float(dot / (norm_main * norm_ref))
        else:
            cosine_sim = 1.0 if np.allclose(main_flat, ref_flat) else 0.0

        return AccuracyMetrics(
            mse=mse,
            max_abs_diff=max_abs_diff,
            cosine_similarity=cosine_sim,
        )
