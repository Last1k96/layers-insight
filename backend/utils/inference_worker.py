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
        # We use a dedicated Core for reading because core.read_model() can
        # corrupt internal state for some plugins (e.g. TEMPLATE).  By
        # serializing the cut model and reloading with a fresh Core we avoid
        # SIGSEGV during compile_model.
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

        # Fold Constant→Convert and all-constant subexpressions to avoid
        # crashes in plugins that don't handle Convert(f16→f32) / Convert(i8→f32).
        n_folded = _fold_constant_subexpressions(cut_model)
        if n_folded > 0:
            _log("info", f"Folded {n_folded} constant subexpressions")

        # Rebuild FakeQuantize nodes — IR-deserialized FQ nodes crash some
        # plugins (e.g. TEMPLATE).  Fresh opset13 FQ nodes work fine.
        n_fq = _rebuild_fakequantize_nodes(cut_model)
        if n_fq > 0:
            _log("info", f"Rebuilt {n_fq} FakeQuantize nodes")

        if n_folded > 0 or n_fq > 0:
            cut_model = ov.Model(
                cut_model.get_results(), cut_model.get_parameters(),
                cut_model.get_friendly_name(),
            )
            cut_model.validate_nodes_and_infer_types()

        # Extract clean graph as pure numpy/python data, then destroy
        # the read Core to avoid Core-corruption issues with plugins
        # like TEMPLATE that crash when compile_model is called on a
        # Core that previously called read_model on FP16-INT8 IR.
        ops_data, result_target = _extract_graph_data(cut_model, node_name)
        model_params = _extract_params(cut_model)
        _log("info", f"Extracted {len(ops_data)} ops as pure data")

        del cut_model, model, target_op, read_core
        gc.collect()

        # --- Phase 2: Rebuild model with fresh Core ---
        core = ov.Core()

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

        _log("info", f"Fresh core devices: {core.available_devices}")

        cut_model = _rebuild_model_from_data(ov, ops_data, result_target)
        _log("info", f"Rebuilt model: {len(list(cut_model.get_ordered_ops()))} ops")

        # Prepare inputs (use cut_model's params, not original model's)
        _log("info", "Preparing inputs...")
        from backend.utils.input_generator import prepare_inputs

        inputs = prepare_inputs(model_params, input_path, precision, input_configs)
        _log("info", f"Inputs prepared: {len(inputs)} tensors")

        # Infer on main device
        _log("info", f"Compiling model for {main_device}...")
        main_out, main_result, main_err = _run_on_device(core, cut_model, main_device, inputs)
        if main_err:
            _log("error", main_err)
            _emit({"error": main_err})
            return
        _log("info", f"Inference on {main_device} complete")

        # Infer on reference device
        _log("info", f"Compiling model for {ref_device}...")
        ref_out, ref_result, ref_err = _run_on_device(core, cut_model, ref_device, inputs)
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


def _fold_constant_subexpressions(model):
    """Fold Constant→Convert chains and all-constant ops to avoid plugin crashes.

    Some plugins (e.g. TEMPLATE) crash on Convert(f16→f32) or Convert(i8→f32)
    during compilation.  These always originate from Constant→Convert patterns
    in IR-loaded FP16-INT8 models.  We pre-compute the conversion in numpy and
    replace the Convert with a new Constant of the target type.

    Also folds Multiply/Add/Subtract where both inputs are Constants.
    """
    import openvino as ov

    _NP_TYPES = {
        'float32': np.float32, 'float16': np.float16,
        'int32': np.int32, 'int64': np.int64,
        'int8_t': np.int8, 'uint8_t': np.uint8,
    }

    def _np_type(ov_et):
        s = str(ov_et)
        for k, v in _NP_TYPES.items():
            if k in s:
                return v
        return np.float32

    changed = True
    total = 0
    while changed:
        changed = False
        for op in list(model.get_ordered_ops()):
            t = op.get_type_name()
            if t in ('Parameter', 'Result', 'Constant'):
                continue
            n_in = op.get_input_size()
            if n_in == 0:
                continue
            all_const = all(
                op.input(i).get_source_output().get_node().get_type_name() == 'Constant'
                for i in range(n_in)
            )
            if not all_const:
                continue
            inputs_data = [
                op.input(i).get_source_output().get_node().get_data()
                for i in range(n_in)
            ]
            out_et = op.get_output_element_type(0)
            np_t = _np_type(out_et)
            try:
                if t == 'Convert':
                    r = inputs_data[0].astype(np_t)
                elif t == 'Multiply':
                    r = (inputs_data[0].astype(np.float64) * inputs_data[1].astype(np.float64)).astype(np_t)
                elif t == 'Add':
                    r = (inputs_data[0].astype(np.float64) + inputs_data[1].astype(np.float64)).astype(np_t)
                elif t == 'Subtract':
                    r = (inputs_data[0].astype(np.float64) - inputs_data[1].astype(np.float64)).astype(np_t)
                elif t == 'Concat':
                    axis = op.get_attributes().get('axis', 0)
                    r = np.concatenate(inputs_data, axis=axis).astype(np_t)
                elif t == 'Reshape':
                    r = inputs_data[0].reshape(inputs_data[1].tolist()).astype(np_t)
                elif t == 'Gather':
                    r = np.take(inputs_data[0], inputs_data[1],
                                axis=int(inputs_data[2])).astype(np_t)
                elif t == 'ShapeOf':
                    ps = op.input(0).get_source_output().get_node().get_output_partial_shape(0)
                    r = np.array(list(ps.get_shape()), dtype=np_t)
                else:
                    continue
                nc = ov.opset13.constant(r, dtype=out_et)
                for ti in list(op.output(0).get_target_inputs()):
                    ti.replace_source_output(nc.output(0))
                changed = True
                total += 1
            except Exception:
                pass

    return total


def _rebuild_fakequantize_nodes(model):
    """Rebuild FakeQuantize nodes using opset13 API.

    The TEMPLATE plugin (and potentially others) crashes on FakeQuantize
    nodes deserialized from IR.  Rebuilding them with fresh opset13 nodes
    — along with their constant inputs — avoids the crash while keeping
    identical semantics.
    """
    import openvino as ov

    count = 0
    for op in list(model.get_ordered_ops()):
        if op.get_type_name() != 'FakeQuantize':
            continue
        levels = op.get_attributes()['levels']
        data_in = op.input(0).get_source_output()
        new_inputs = [data_in]
        for i in range(1, 5):
            src = op.input(i).get_source_output().get_node()
            if src.get_type_name() == 'Constant':
                data = src.get_data().copy()
                # Reshape scalar (0-d) constants to [1] — some plugins
                # crash on 0-d FakeQuantize bound constants.
                if data.ndim == 0:
                    data = data.reshape([1])
                et = src.get_output_element_type(0)
                new_c = ov.opset13.constant(data, dtype=et)
                new_inputs.append(new_c.output(0))
            else:
                new_inputs.append(op.input(i).get_source_output())
        new_fq = ov.opset13.fake_quantize(
            new_inputs[0], new_inputs[1], new_inputs[2],
            new_inputs[3], new_inputs[4], levels,
        )
        for ti in list(op.output(0).get_target_inputs()):
            ti.replace_source_output(new_fq.output(0))
        count += 1
    return count


def _extract_graph_data(model, target_name: str) -> tuple[list[dict], str]:
    """Extract a clean OV model graph as pure python/numpy data.

    Returns (ops_data, result_target_name) where ops_data is a list of dicts
    describing each op (type, name, data/shape/attrs/inputs) and
    result_target_name is the name of the node feeding into the Result.
    """
    ops_data = []
    result_target = target_name
    for op in model.get_ordered_ops():
        t = op.get_type_name()
        name = op.get_friendly_name()
        entry = {"type": t, "name": name}
        if t == "Constant":
            entry["data"] = op.get_data().copy()
            entry["et"] = str(op.get_output_element_type(0))
        elif t == "Parameter":
            entry["shape"] = list(op.get_output_partial_shape(0).get_shape())
            entry["et"] = str(op.get_output_element_type(0))
        elif t == "Result":
            result_target = op.input(0).get_source_output().get_node().get_friendly_name()
            continue
        else:
            entry["attrs"] = op.get_attributes()
            entry["inputs"] = [
                op.input(i).get_source_output().get_node().get_friendly_name()
                for i in range(op.get_input_size())
            ]
            entry["out_et"] = str(op.get_output_element_type(0))
        ops_data.append(entry)
    return ops_data, result_target


def _rebuild_model_from_data(ov, ops_data: list[dict], result_target: str):
    """Rebuild an OV model from extracted pure data using opset13 API.

    This creates entirely fresh OV nodes with no references to any
    previously loaded IR, avoiding Core corruption issues.
    """
    _OV_ET_MAP = {
        "f32": ov.Type.f32, "f16": ov.Type.f16,
        "i32": ov.Type.i32, "i64": ov.Type.i64,
        "i8": ov.Type.i8, "u8": ov.Type.u8,
    }

    def _parse_et(et_str):
        for k, v in _OV_ET_MAP.items():
            if k in et_str:
                return v
        return ov.Type.f32

    node_map = {}
    params = []
    skipped = []

    for e in ops_data:
        t = e["type"]
        name = e["name"]
        if t == "Parameter":
            et = _parse_et(e.get("et", "f32"))
            p = ov.opset13.parameter(shape=e["shape"], dtype=et)
            p.set_friendly_name(name)
            node_map[name] = p.output(0)
            params.append(p)
        elif t == "Constant":
            data = e["data"]
            if data.ndim == 0:
                data = data.reshape([1])
            et = _parse_et(e.get("et", "f32"))
            c = ov.opset13.constant(data, dtype=et)
            node_map[name] = c.output(0)
        else:
            inputs = []
            missing = False
            for n in e["inputs"]:
                if n not in node_map:
                    missing = True
                    break
                inputs.append(node_map[n])
            if missing:
                skipped.append((t, name))
                continue
            a = e.get("attrs", {})
            try:
                node = _build_op(ov, t, inputs, a)
                if node is None:
                    skipped.append((t, name))
                    continue
                node_map[name] = node.output(0)
            except Exception:
                skipped.append((t, name))

    if skipped:
        _log("info", f"Skipped {len(skipped)} ops during rebuild: {skipped[:5]}")

    m = ov.Model([node_map[result_target]], params, f"rebuilt_{result_target}")
    m.validate_nodes_and_infer_types()
    return m


def _build_op(ov, op_type: str, inputs: list, attrs: dict):
    """Build a single opset13 op node from type, inputs and attributes."""
    if op_type == "FakeQuantize":
        return ov.opset13.fake_quantize(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], attrs["levels"],
        )
    elif op_type == "Convolution":
        return ov.opset13.convolution(
            inputs[0], inputs[1], attrs["strides"],
            attrs["pads_begin"], attrs["pads_end"], attrs["dilations"],
        )
    elif op_type == "GroupConvolution":
        return ov.opset13.group_convolution(
            inputs[0], inputs[1], attrs["strides"],
            attrs["pads_begin"], attrs["pads_end"], attrs["dilations"],
        )
    elif op_type == "Add":
        return ov.opset13.add(inputs[0], inputs[1])
    elif op_type == "Subtract":
        return ov.opset13.subtract(inputs[0], inputs[1])
    elif op_type == "Multiply":
        return ov.opset13.multiply(inputs[0], inputs[1])
    elif op_type == "Relu":
        return ov.opset13.relu(inputs[0])
    elif op_type == "Clamp":
        return ov.opset13.clamp(inputs[0], attrs["min"], attrs["max"])
    elif op_type == "MaxPool":
        return ov.opset13.max_pool(
            inputs[0], attrs["strides"], attrs["pads_begin"],
            attrs["pads_end"], attrs["kernel"],
        )
    elif op_type == "AvgPool":
        return ov.opset13.avg_pool(
            inputs[0], attrs["strides"], attrs["pads_begin"],
            attrs["pads_end"], attrs["kernel"],
            attrs.get("exclude-pad", True),
        )
    elif op_type == "MatMul":
        return ov.opset13.matmul(
            inputs[0], inputs[1],
            attrs.get("transpose_a", False), attrs.get("transpose_b", False),
        )
    elif op_type == "Reshape":
        return ov.opset13.reshape(inputs[0], inputs[1], attrs.get("special_zero", False))
    elif op_type == "Concat":
        return ov.opset13.concat(inputs, attrs["axis"])
    elif op_type == "Softmax":
        return ov.opset13.softmax(inputs[0], attrs.get("axis", 1))
    elif op_type == "Sigmoid":
        return ov.opset13.sigmoid(inputs[0])
    elif op_type == "Tanh":
        return ov.opset13.tanh(inputs[0])
    return None


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


def _run_on_device(core, model, device: str, inputs: dict):
    try:
        _log("info", f"Compiling on {device}...")
        compiled = core.compile_model(model, device)
        _log("info", f"Compilation on {device} succeeded")
    except Exception as e:
        return None, None, f"Compilation on {device} failed: {e}"
    try:
        _log("info", f"Running inference on {device}...")
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
