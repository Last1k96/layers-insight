"""Subprocess worker for isolated OpenVINO inference.

Runs model cutting + inference in a separate process so that a segfault
in the OpenVINO C++ layer does not kill the main server.

Protocol:
  stdin  ← JSON config (model_path, node_name, devices, inputs, ov_path)
  stdout → JSON result  (metrics, device_results, error)
  Temp dir for numpy outputs is passed in config and read back by caller.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np


def main() -> None:
    cfg = json.loads(sys.stdin.read())

    model_path: str = cfg["model_path"]
    node_name: str = cfg["node_name"]
    main_device: str = cfg["main_device"]
    ref_device: str = cfg["ref_device"]
    ov_path: str | None = cfg.get("ov_path")
    input_path: str | None = cfg.get("input_path")
    precision: str = cfg.get("precision", "fp32")
    input_configs: list[dict] | None = cfg.get("input_configs")
    out_dir: str = cfg["out_dir"]  # temp dir for numpy files

    try:
        import openvino as ov

        core = ov.Core()

        # Register plugins from custom OV build path
        if ov_path:
            ov_lib_dir = Path(ov_path)
            for so_file in ov_lib_dir.glob("libopenvino_*_plugin.so"):
                name = so_file.stem
                parts = name.replace("libopenvino_", "").replace("_plugin", "")
                device_name = parts.upper().replace("INTEL_", "")
                if device_name not in core.available_devices:
                    try:
                        core.register_plugin(str(so_file), device_name)
                    except Exception:
                        pass

        # Load model
        model = core.read_model(model_path)

        # Find target op
        target_op = None
        for op in model.get_ordered_ops():
            if op.get_friendly_name() == node_name:
                target_op = op
                break

        if target_op is None:
            _emit({"error": f"Node '{node_name}' not found in model"})
            return

        # Cut model
        new_outputs = [target_op.output(i) for i in range(target_op.get_output_size())]
        cut_model = ov.Model(new_outputs, model.get_parameters(), f"cut_at_{node_name}")

        # Prepare inputs
        from backend.utils.input_generator import prepare_inputs

        model_params = _extract_params(model)
        inputs = prepare_inputs(model_params, input_path, precision, input_configs)

        # Infer on main device
        main_out, main_result, main_err = _run_on_device(core, cut_model, main_device, inputs)
        if main_err:
            _emit({"error": main_err})
            return

        # Infer on reference device
        ref_out, ref_result, ref_err = _run_on_device(core, cut_model, ref_device, inputs)
        if ref_err:
            _emit({"error": ref_err})
            return

        # Compute metrics
        metrics = _compute_metrics(main_out, ref_out)

        # Save numpy outputs
        out_path = Path(out_dir)
        np.save(str(out_path / "main_output.npy"), main_out)
        np.save(str(out_path / "ref_output.npy"), ref_out)

        _emit({
            "main_result": main_result,
            "ref_result": ref_result,
            "metrics": metrics,
        })

    except Exception as e:
        _emit({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})


def _extract_params(model) -> list[dict]:
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


def _run_on_device(core, model, device: str, inputs: dict):
    try:
        compiled = core.compile_model(model, device)
    except Exception as e:
        return None, None, f"Compilation on {device} failed: {e}"
    try:
        req = compiled.create_infer_request()
        req.infer(inputs)
        output = req.get_output_tensor(0).data.copy()
        result = {
            "device": device,
            "output_shapes": [list(output.shape)],
            "dtype": str(output.dtype),
            "min_val": float(np.min(output)),
            "max_val": float(np.max(output)),
            "mean_val": float(np.mean(output)),
            "std_val": float(np.std(output)),
        }
        return output, result, None
    except Exception as e:
        return None, None, f"Inference on {device} failed: {e}"


def _compute_metrics(main: np.ndarray, ref: np.ndarray) -> dict:
    main_flat = main.flatten().astype(np.float64)
    ref_flat = ref.flatten().astype(np.float64)
    diff = main_flat - ref_flat
    mse = float(np.mean(diff ** 2))
    max_abs_diff = float(np.max(np.abs(diff)))
    dot = np.dot(main_flat, ref_flat)
    norm_main = np.linalg.norm(main_flat)
    norm_ref = np.linalg.norm(ref_flat)
    if norm_main > 0 and norm_ref > 0:
        cosine_sim = float(dot / (norm_main * norm_ref))
    else:
        cosine_sim = 1.0 if np.allclose(main_flat, ref_flat) else 0.0
    return {"mse": mse, "max_abs_diff": max_abs_diff, "cosine_similarity": cosine_sim}


def _emit(data: dict) -> None:
    json.dump(data, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
