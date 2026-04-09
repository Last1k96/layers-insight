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

from backend.utils.ov_graph_utils import get_reachable_params


def _log(level: str, msg: str) -> None:
    """Write a structured log line to stderr as JSON."""
    json.dump({"level": level, "msg": msg}, sys.stderr)
    sys.stderr.write("\n")
    sys.stderr.flush()


def _stage(name: str) -> None:
    """Write a stage update to stderr (parsed by inference_service, not shown in logs)."""
    json.dump({"type": "stage", "stage": name}, sys.stderr)
    sys.stderr.write("\n")
    sys.stderr.flush()


def _load_and_cut_model(cfg: dict) -> tuple:
    """Phase 1: Load model, find target op, cut to subgraph, save cut model.

    Returns (tmp_xml, model_params, input_configs).
    """
    import gc
    import openvino as ov

    model_path = cfg["model_path"]
    node_name = cfg["node_name"]
    main_device = cfg["main_device"]
    ref_device = cfg["ref_device"]
    out_dir = cfg["out_dir"]
    input_configs = cfg.get("input_configs")

    # Dedicated Core for reading — avoids state leaks into compile phase
    read_core = ov.Core()
    model = read_core.read_model(model_path)

    # Find target op
    target_op = None
    for op in model.get_ordered_ops():
        if op.get_friendly_name() == node_name:
            target_op = op
            break

    if target_op is None:
        raise ValueError(f"Node '{node_name}' not found in model")

    # Cut model — only include parameters reachable from the target outputs
    _stage("cutting_graph")
    new_outputs = target_op.outputs()
    reachable_params = get_reachable_params(model, new_outputs)
    cut_model = ov.Model(new_outputs, reachable_params, f"cut_at_{node_name}")
    cut_model.validate_nodes_and_infer_types()

    # Serialize cut model to out_dir (self-contained reproducer)
    tmp_xml = str(Path(out_dir) / "cut_model.xml")
    ov.save_model(cut_model, tmp_xml)
    model_params = _extract_params(cut_model)

    # Prune input_configs to only include parameters in the cut model
    if input_configs:
        cut_param_names = {p["name"] for p in model_params}
        input_configs = [c for c in input_configs if c["name"] in cut_param_names]

    del cut_model, model, target_op, read_core
    gc.collect()

    return tmp_xml, model_params, input_configs


def _prepare_inputs(cfg: dict, tmp_xml: str, input_configs: list[dict] | None):
    """Phase 2: Reload model with fresh Core, prepare input tensors.

    Returns (core, cut_model, inputs).
    """
    import openvino as ov
    from backend.utils.ov_helpers import register_plugins
    from backend.utils.input_generator import prepare_inputs

    ov_path = cfg.get("ov_path")
    input_path = cfg.get("input_path")
    precision = cfg.get("precision", "fp32")
    out_dir = cfg["out_dir"]

    core = ov.Core()
    register_plugins(core, ov_path)

    cut_model = core.read_model(tmp_xml)

    # Apply bounded reshape if input_configs have bounds (dynamic shapes)
    _apply_bounded_reshape(cut_model, input_configs)

    # Re-extract params after reshape (shapes may now be bounded/static)
    model_params = _extract_params(cut_model)

    _stage("preparing_inputs")
    inputs = prepare_inputs(model_params, input_path, precision, input_configs)

    # Save input tensors to out_dir for reproducibility
    for name, tensor in inputs.items():
        from backend.utils import sanitize_filename
        safe_name = sanitize_filename(name)
        np.save(str(Path(out_dir) / f"input_{safe_name}.npy"), tensor)

    return core, cut_model, inputs


def _run_inference(cfg: dict, core, cut_model, inputs: dict):
    """Phase 3: Compile on both devices, execute, collect ALL outputs.

    Returns (main_outs, main_results, ref_outs, ref_results) where each is a list
    (one entry per compiled model output). Raises RuntimeError on failure.
    """
    main_device = cfg["main_device"]
    ref_device = cfg["ref_device"]

    # Infer on main device
    _stage("compiling")
    main_plugin_config = cfg.get("plugin_config") or None
    main_outs, main_results, main_err = _run_on_device(core, cut_model, main_device, inputs, main_plugin_config)
    if main_err:
        raise RuntimeError(main_err)

    # Infer on reference device
    ref_plugin_config = cfg.get("ref_plugin_config") or None
    ref_outs, ref_results, ref_err = _run_on_device(core, cut_model, ref_device, inputs, ref_plugin_config)
    if ref_err:
        raise RuntimeError(ref_err)

    return main_outs, main_results, ref_outs, ref_results


def _save_outputs(cfg: dict, main_outs: list, ref_outs: list) -> None:
    """Phase 5: Save numpy output arrays to disk (all outputs)."""
    out_path = Path(cfg["out_dir"])
    num_outputs = len(main_outs)
    for i in range(num_outputs):
        np.save(str(out_path / f"main_output_{i}.npy"), main_outs[i])
        np.save(str(out_path / f"ref_output_{i}.npy"), ref_outs[i])
    # Backward compatibility: main_output.npy / ref_output.npy point to output 0
    np.save(str(out_path / "main_output.npy"), main_outs[0])
    np.save(str(out_path / "ref_output.npy"), ref_outs[0])


def main() -> None:
    # Preserve the real stdout for the JSON result, then redirect both the
    # Python-level sys.stdout AND the OS-level fd 1 to stderr.  The fd-level
    # redirect is critical: OpenVINO's C++ backends write warnings directly
    # to fd 1, bypassing Python's sys.stdout, which corrupts the JSON output.
    real_stdout_fd = os.dup(1)          # duplicate fd 1 before we clobber it
    os.dup2(2, 1)                       # fd 1 now points to stderr
    real_stdout = os.fdopen(real_stdout_fd, "w")  # wrap saved fd for _emit()
    sys.stdout = sys.stderr             # Python-level redirect (belt & suspenders)

    cfg = json.loads(sys.stdin.read())

    # Set OV log level BEFORE importing openvino
    os.environ["OPENVINO_LOG_LEVEL"] = cfg.get("ov_log_level", "WARNING")

    try:
        _stage("loading_model")

        # Phase 1: Load model, cut subgraph, serialize
        tmp_xml, model_params, input_configs = _load_and_cut_model(cfg)

        # Phase 2: Reload with fresh Core, prepare inputs
        core, cut_model, inputs = _prepare_inputs(
            cfg, tmp_xml, input_configs,
        )

        # Phase 3: Run inference on both devices (returns lists for multi-output)
        main_outs, main_results, ref_outs, ref_results = _run_inference(
            cfg, core, cut_model, inputs,
        )

        # Phase 4: Compute per-output accuracy metrics
        _stage("computing_metrics")
        num_outputs = len(main_outs)
        per_output_metrics: list[dict] = []
        for i in range(num_outputs):
            m = _compute_metrics(main_outs[i], ref_outs[i])
            per_output_metrics.append(m)

        # Aggregate: worst across outputs (highest MSE, highest max_abs_diff, lowest cosine)
        if num_outputs == 1:
            agg_metrics = per_output_metrics[0]
            _log("info", f"MSE={agg_metrics['mse']:.6e}  cosine={agg_metrics['cosine_similarity']:.6f}  max_diff={agg_metrics['max_abs_diff']:.6e}")
        else:
            for i, m in enumerate(per_output_metrics):
                _log("info", f"Output {i}: MSE={m['mse']:.6e}  cosine={m['cosine_similarity']:.6f}  max_diff={m['max_abs_diff']:.6e}")
            agg_metrics = {
                "mse": max(m["mse"] for m in per_output_metrics),
                "max_abs_diff": max(m["max_abs_diff"] for m in per_output_metrics),
                "cosine_similarity": min(m["cosine_similarity"] for m in per_output_metrics),
            }
            _log("info", f"Worst: MSE={agg_metrics['mse']:.6e}  cosine={agg_metrics['cosine_similarity']:.6f}  max_diff={agg_metrics['max_abs_diff']:.6e}")

        # Phase 5: Save outputs
        _save_outputs(cfg, main_outs, ref_outs)

        _stage("done")
        result_payload: dict = {
            "main_result": main_results[0],
            "ref_result": ref_results[0],
            "metrics": agg_metrics,
        }
        # Include per-output data when there are multiple outputs
        if num_outputs > 1:
            result_payload["per_output_metrics"] = per_output_metrics
            result_payload["per_output_main_results"] = main_results
            result_payload["per_output_ref_results"] = ref_results
        _emit(result_payload, _file=real_stdout)

    except ValueError as e:
        _log("error", str(e))
        _emit({"error": str(e)}, _file=real_stdout)
    except RuntimeError as e:
        _log("error", str(e))
        _emit({"error": str(e)}, _file=real_stdout)
    except Exception as e:
        _log("error", f"{type(e).__name__}: {e}")
        _emit({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}, _file=real_stdout)



def _extract_params(model) -> list[dict]:
    params = []
    for param in model.get_parameters():
        pshape = param.get_output_partial_shape(0)
        if pshape.is_static:
            shape = [d.get_length() for d in pshape]
        else:
            shape = [d.get_length() if d.is_static else "?" for d in pshape]
        params.append({
            "name": param.get_friendly_name(),
            "shape": shape,
            "element_type": str(param.get_output_element_type(0)),
        })
    return params


def _apply_bounded_reshape(model, input_configs):
    """Reshape model with bounded dimensions before compilation.

    For each input that has lower_bounds and upper_bounds, creates
    ov.Dimension(lo, hi) for dynamic dims and ov.Dimension(d) for static dims.
    """
    import openvino as ov

    if not input_configs:
        return
    reshape_map = {}
    for cfg in input_configs:
        lo = cfg.get("lower_bounds", [])
        hi = cfg.get("upper_bounds", [])
        if lo and hi:
            orig_shape = cfg.get("shape", [])
            dims = []
            for i, d in enumerate(orig_shape):
                if isinstance(d, str):  # dynamic dim
                    dims.append(ov.Dimension(lo[i], hi[i]))
                else:
                    dims.append(ov.Dimension(d))
            reshape_map[cfg["name"]] = ov.PartialShape(dims)
    if reshape_map:
        model.reshape(reshape_map)


def _run_on_device(core, model, device: str, inputs: dict, plugin_config: dict[str, str] | None = None):
    """Run inference and return ALL outputs.

    Returns:
        (outputs_list, per_output_results, error)
        - outputs_list: list[np.ndarray] — one array per compiled-model output
        - per_output_results: list[dict] — per-output device stats
        - error: str | None
    """
    config = plugin_config or {}
    try:
        compiled = core.compile_model(model, device, config)
    except Exception as e:
        return None, None, f"Compilation on {device} failed: {e}"
    try:
        req = compiled.create_infer_request()
        # Pass inputs by parameter order to avoid tensor-name vs friendly-name
        # mismatches that can occur after save_model/read_model cycles.
        input_list = [inputs[p.get_friendly_name()] for p in model.get_parameters()]
        req.infer(input_list)

        num_outputs = len(compiled.outputs)
        outputs_list: list = []
        per_output_results: list = []
        for i in range(num_outputs):
            output = req.get_output_tensor(i).data.copy()
            out64 = output.astype(np.float64)
            outputs_list.append(output)
            per_output_results.append({
                "device": device,
                "output_shapes": [list(output.shape)],
                "dtype": str(output.dtype),
                "min_val": float(np.min(out64)),
                "max_val": float(np.max(out64)),
                "mean_val": float(np.mean(out64)),
                "std_val": float(np.std(out64)),
            })

        return outputs_list, per_output_results, None
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


def _emit(data: dict, *, _file=None) -> None:
    out = _file or sys.stdout
    json.dump(_sanitize_for_json(data), out)
    out.flush()


if __name__ == "__main__":
    main()
