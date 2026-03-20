"""Subprocess worker for isolated OpenVINO inference.

Runs model cutting + inference in a separate process so that a segfault
in the OpenVINO C++ layer does not kill the main server.

Protocol:
  stdin  ← JSON config (model_path, node_name, devices, inputs, ov_path)
  stdout → JSON result  (metrics, device_results, error)
  stderr → JSON log lines {"level":"info","msg":"..."}
  Temp dir for numpy outputs is passed in config and read back by caller.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np


def _log(level: str, msg: str) -> None:
    """Write a structured log line to stderr as JSON."""
    json.dump({"level": level, "msg": msg}, sys.stderr)
    sys.stderr.write("\n")
    sys.stderr.flush()


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
    ov_log_level: str = cfg.get("ov_log_level", "WARNING")

    # Set OV log level BEFORE importing openvino
    os.environ["OPENVINO_LOG_LEVEL"] = ov_log_level

    try:
        import gc

        _log("info", "Loading OpenVINO runtime...")
        import openvino as ov

        # --- Phase 1: Read model, cut subgraph, serialize to temp file ---
        # We use a dedicated Core for reading so that the compile Core starts
        # clean — avoids potential state leaks between read and compile phases.
        read_core = ov.Core()

        _log("info", f"Reading model: {Path(model_path).name}")
        model = read_core.read_model(model_path)
        _log("info", f"Model loaded: {len(list(model.get_ordered_ops()))} ops")

        # Find target op
        target_op = None
        for op in model.get_ordered_ops():
            if op.get_friendly_name() == node_name:
                target_op = op
                break

        if target_op is None:
            _log("error", f"Node '{node_name}' not found in model")
            _emit({"error": f"Node '{node_name}' not found in model"})
            return

        _log("info", f"Target node found: {node_name} (type: {target_op.get_type_name()})")

        # Cut model — only include parameters reachable from the target outputs
        _log("info", "Cutting model at target node...")
        new_outputs = target_op.outputs()
        reachable_params = _get_reachable_params(model, new_outputs)
        cut_model = ov.Model(new_outputs, reachable_params, f"cut_at_{node_name}")
        cut_model.validate_nodes_and_infer_types()
        _log("info", f"Cut model created: {len(reachable_params)} parameters, {len(new_outputs)} outputs")

        # Serialize cut model directly to out_dir so the task folder becomes
        # a self-contained reproducer (cut model + inputs + outputs).
        tmp_xml = str(Path(out_dir) / "cut_model.xml")
        ov.save_model(cut_model, tmp_xml)
        model_params = _extract_params(cut_model)
        _log("info", f"Serialized cut model to {tmp_xml}")

        # If either device is a virtual fp16 device, also save an fp16-compressed
        # copy.  Weight compression (compress_to_fp16) actually quantises constants
        # so even single-op subgraphs show precision differences.
        needs_fp16_model = _is_fp16_device(main_device) or _is_fp16_device(ref_device)
        fp16_xml = None
        if needs_fp16_model:
            fp16_xml = str(Path(out_dir) / "cut_model_fp16.xml")
            ov.save_model(cut_model, fp16_xml, compress_to_fp16=True)
            _log("info", "Saved fp16-compressed cut model")

        del cut_model, model, target_op, read_core
        gc.collect()

        # --- Phase 2: Reload model with fresh Core ---
        core = ov.Core()

        from backend.utils.ov_helpers import register_plugins
        register_plugins(core, ov_path)

        _log("info", f"Fresh core devices: {core.available_devices}")

        cut_model = core.read_model(tmp_xml)
        _log("info", f"Reloaded cut model: {len(list(cut_model.get_ordered_ops()))} ops")

        fp16_cut_model = None
        if fp16_xml:
            fp16_cut_model = core.read_model(fp16_xml)
            _log("info", f"Reloaded fp16 cut model: {len(list(fp16_cut_model.get_ordered_ops()))} ops")

        # Prepare inputs (use cut_model's params, not original model's)
        _log("info", "Preparing inputs...")
        from backend.utils.input_generator import prepare_inputs

        inputs = prepare_inputs(model_params, input_path, precision, input_configs)
        _log("info", f"Inputs prepared: {len(inputs)} tensors")

        # Save input tensors to out_dir for reproducibility
        for name, tensor in inputs.items():
            safe_name = name.replace("/", "_").replace("(", "_").replace(")", "_")
            np.save(str(Path(out_dir) / f"input_{safe_name}.npy"), tensor)

        # Infer on main device
        _log("info", f"Compiling model for {main_device}...")
        main_model = fp16_cut_model if _is_fp16_device(main_device) else cut_model
        main_out, main_result, main_err = _run_on_device(core, main_model, main_device, inputs)
        if main_err:
            _log("error", main_err)
            _emit({"error": main_err})
            return
        _log("info", f"Inference on {main_device} complete")

        # Infer on reference device
        _log("info", f"Compiling model for {ref_device}...")
        ref_model = fp16_cut_model if _is_fp16_device(ref_device) else cut_model
        ref_out, ref_result, ref_err = _run_on_device(core, ref_model, ref_device, inputs)
        if ref_err:
            _log("error", ref_err)
            _emit({"error": ref_err})
            return
        _log("info", f"Inference on {ref_device} complete")

        # Compute metrics
        _log("info", "Computing accuracy metrics...")
        metrics = _compute_metrics(main_out, ref_out)
        _log("info", f"Metrics: MSE={metrics['mse']:.6e}, cosine={metrics['cosine_similarity']:.6f}, max_diff={metrics['max_abs_diff']:.6e}")

        # Save numpy outputs
        out_path = Path(out_dir)
        np.save(str(out_path / "main_output.npy"), main_out)
        np.save(str(out_path / "ref_output.npy"), ref_out)

        _log("info", "Done — results ready")
        _emit({
            "main_result": main_result,
            "ref_result": ref_result,
            "metrics": metrics,
        })

    except Exception as e:
        _log("error", f"{type(e).__name__}: {e}")
        _emit({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})



def _get_reachable_params(model, target_outputs) -> list:
    """Walk backward from target outputs to find only reachable Parameter nodes."""
    # Build a name->Parameter map using the actual Parameter objects from the model.
    # We can't use id() because OV Python bindings create new wrapper objects each call.
    param_by_name = {p.get_friendly_name(): p for p in model.get_parameters()}
    visited = set()
    params = []
    stack = list(target_outputs)

    while stack:
        output = stack.pop()
        node = output.get_node()
        node_name = node.get_friendly_name()
        if node_name in visited:
            continue
        visited.add(node_name)

        # Check if this node is a Parameter by name match
        if node_name in param_by_name:
            params.append(param_by_name[node_name])
            continue

        for i in range(node.get_input_size()):
            source_output = node.input(i).get_source_output()
            stack.append(source_output)

    return params


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


_FP16_DEVICES = {"CPU_fp16"}


def _is_fp16_device(device: str) -> bool:
    """Check if device is a virtual fp16 device that uses weight compression."""
    return device in _FP16_DEVICES


def _parse_virtual_device(device: str) -> tuple[str, dict]:
    """Parse virtual device names into (actual_device, config).

    Supported virtual devices:
      CPU_fp16 → CPU with fp16-compressed weights (model-level, not runtime hint)
    """
    if device in _FP16_DEVICES:
        return ("CPU", {})
    return (device, {})


def _run_on_device(core, model, device: str, inputs: dict):
    actual_device, config = _parse_virtual_device(device)
    try:
        _log("info", f"Compiling on {device}...")
        compiled = core.compile_model(model, actual_device, config)
        _log("info", f"Compilation on {device} succeeded")
    except Exception as e:
        return None, None, f"Compilation on {device} failed: {e}"
    try:
        _log("info", f"Running inference on {device}...")
        req = compiled.create_infer_request()
        req.infer(inputs)
        output = req.get_output_tensor(0).data.copy()
        out64 = output.astype(np.float64)
        result = {
            "device": device,
            "output_shapes": [list(output.shape)],
            "dtype": str(output.dtype),
            "min_val": float(np.min(out64)),
            "max_val": float(np.max(out64)),
            "mean_val": float(np.mean(out64)),
            "std_val": float(np.std(out64)),
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


def _sanitize_for_json(obj):
    """Replace inf/nan with None so output is valid JSON for all parsers."""
    if isinstance(obj, float):
        import math
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _emit(data: dict) -> None:
    json.dump(_sanitize_for_json(data), sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
