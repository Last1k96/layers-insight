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

        # Serialize cut model directly to out_dir so the task folder becomes
        # a self-contained reproducer (cut model + inputs + outputs).
        tmp_xml = str(Path(out_dir) / "cut_model.xml")
        ov.save_model(cut_model, tmp_xml)
        model_params = _extract_params(cut_model)
        _log("info", f"Serialized cut model to {tmp_xml}")

        del cut_model, model, target_op, read_core
        gc.collect()

        # --- Phase 2: Reload model with fresh Core ---
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

        cut_model = core.read_model(tmp_xml)
        _log("info", f"Reloaded cut model: {len(list(cut_model.get_ordered_ops()))} ops")

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
        main_model = _prepare_model_for_device(ov, cut_model, main_device)
        main_out, main_result, main_err = _run_on_device(core, main_model, main_device, inputs)
        if main_err:
            _log("error", main_err)
            _emit({"error": main_err})
            return
        _log("info", f"Inference on {main_device} complete")

        # Infer on reference device
        _log("info", f"Compiling model for {ref_device}...")
        ref_model = _prepare_model_for_device(ov, cut_model, ref_device)
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



def _prepare_model_for_device(ov, model, device: str):
    """Return a model suitable for the given device.

    The TEMPLATE plugin crashes on ANY IR-deserialized node.  Only models
    built entirely from fresh opset13 API calls work.  For TEMPLATE we
    extract the graph as pure data and rebuild every node from scratch.

    For well-behaved plugins (CPU, GPU, etc.) we return the original
    IR-loaded model so inference sees the exact original subgraph.
    """
    if device != "TEMPLATE":
        return model

    _log("info", f"Rebuilding model from scratch for {device}...")
    ops_data, result_target = _extract_graph_data(model)
    rebuilt = _rebuild_model_from_data(ov, ops_data, result_target)
    _log("info", f"Rebuilt model: {len(list(rebuilt.get_ordered_ops()))} ops "
         f"(original: {len(list(model.get_ordered_ops()))})")
    return rebuilt


def _extract_graph_data(model) -> tuple[list[dict], str]:
    """Extract an OV model graph as pure python/numpy data."""
    ops_data = []
    result_target = ""
    for op in model.get_ordered_ops():
        t = op.get_type_name()
        name = op.get_friendly_name()
        entry: dict = {"type": t, "name": name}
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
            entry["n_outputs"] = op.get_output_size()
            for oi in range(op.get_output_size()):
                entry[f"out_et_{oi}"] = str(op.get_output_element_type(oi))
        ops_data.append(entry)
    return ops_data, result_target


def _rebuild_model_from_data(ov, ops_data: list[dict], result_target: str):
    """Rebuild an OV model from extracted data using opset13 API."""
    _OV_ET = {
        "f32": ov.Type.f32, "f16": ov.Type.f16,
        "i32": ov.Type.i32, "i64": ov.Type.i64,
        "i8": ov.Type.i8, "u8": ov.Type.u8,
        "boolean": ov.Type.boolean,
    }

    def _et(s):
        for k, v in _OV_ET.items():
            if k in s:
                return v
        return ov.Type.f32

    # First pass: fold all-constant subexpressions so they become
    # single Constants (avoids needing to rebuild Convert, Reshape, etc.
    # on constant-only paths).
    _NP = {
        'float32': np.float32, 'float16': np.float16,
        'int32': np.int32, 'int64': np.int64,
        'int8_t': np.int8, 'uint8_t': np.uint8,
    }

    def _npt(et_str):
        for k, v in _NP.items():
            if k in et_str:
                return v
        return np.float32

    const_data: dict[str, tuple[np.ndarray, str]] = {}
    for e in ops_data:
        if e["type"] == "Constant":
            const_data[e["name"]] = (e["data"], e.get("et", "f32"))

    changed = True
    while changed:
        changed = False
        new_ops = []
        for e in ops_data:
            t = e["type"]
            if t in ("Parameter", "Constant"):
                new_ops.append(e)
                continue
            inputs = e.get("inputs", [])
            if not inputs or not all(n in const_data for n in inputs):
                new_ops.append(e)
                continue
            # All inputs are constants — try to fold
            inp = [const_data[n][0] for n in inputs]
            out_et_str = e.get("out_et_0", "f32")
            npt = _npt(out_et_str)
            r = None
            try:
                if t == 'Convert':
                    r = inp[0].astype(npt)
                elif t == 'Multiply':
                    r = (inp[0].astype(np.float64) * inp[1].astype(np.float64)).astype(npt)
                elif t == 'Add':
                    r = (inp[0].astype(np.float64) + inp[1].astype(np.float64)).astype(npt)
                elif t == 'Subtract':
                    r = (inp[0].astype(np.float64) - inp[1].astype(np.float64)).astype(npt)
                elif t == 'Concat':
                    r = np.concatenate(inp, axis=e["attrs"].get('axis', 0)).astype(npt)
                elif t == 'Reshape':
                    r = inp[0].reshape(inp[1].tolist()).astype(npt)
                elif t == 'Gather':
                    r = np.take(inp[0], inp[1], axis=int(inp[2])).astype(npt)
                elif t == 'ShapeOf':
                    src_e = next((x for x in ops_data if x["name"] == inputs[0]), None)
                    if src_e and src_e["type"] == "Constant":
                        r = np.array(list(src_e["data"].shape), dtype=npt)
            except Exception:
                pass
            if r is not None:
                const_data[e["name"]] = (r, out_et_str)
                new_ops.append({"type": "Constant", "name": e["name"], "data": r, "et": out_et_str})
                changed = True
            else:
                new_ops.append(e)
        ops_data = new_ops

    # Second pass: build fresh opset13 nodes
    node_map: dict[str, object] = {}
    params = []
    skipped = []

    for e in ops_data:
        t = e["type"]
        name = e["name"]
        if t == "Parameter":
            et = _et(e.get("et", "f32"))
            p = ov.opset13.parameter(shape=e["shape"], dtype=et)
            p.set_friendly_name(name)
            node_map[name] = p.output(0)
            params.append(p)
        elif t == "Constant":
            data = e["data"]
            if data.ndim == 0:
                data = data.reshape([1])
            et = _et(e.get("et", "f32"))
            c = ov.opset13.constant(data, dtype=et)
            node_map[name] = c.output(0)
        else:
            inputs = [node_map[n] for n in e["inputs"] if n in node_map]
            if len(inputs) != len(e["inputs"]):
                skipped.append((t, name))
                continue
            a = e.get("attrs", {})
            try:
                node = _build_op(ov, t, inputs, a)
                if node is None:
                    skipped.append((t, name))
                    continue
                node_map[name] = node.output(0)
            except Exception as exc:
                _log("info", f"Failed to build {t} '{name}': {exc}")
                skipped.append((t, name))

    if skipped:
        _log("info", f"Skipped {len(skipped)} ops during rebuild: {skipped[:10]}")

    if result_target not in node_map:
        raise RuntimeError(
            f"Cannot rebuild model for TEMPLATE: target node '{result_target}' "
            f"could not be built. Skipped ops: {skipped}"
        )

    m = ov.Model([node_map[result_target]], params, f"rebuilt_{result_target}")
    m.validate_nodes_and_infer_types()
    return m


def _ensure_i64(ov, inp):
    """If input is a float constant, cast its data to i64 and return a new constant."""
    node = inp.get_node()
    if node.get_type_name() == "Constant" and "int" not in str(node.get_output_element_type(0)):
        data = node.get_data().astype(np.int64)
        return ov.opset13.constant(data, dtype=ov.Type.i64).output(0)
    return inp


def _build_op(ov, op_type: str, inputs: list, attrs: dict):
    """Build a single opset13 op node from type, inputs and attributes."""
    o = ov.opset13
    if op_type == "FakeQuantize":
        return o.fake_quantize(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], attrs["levels"])
    elif op_type == "Convolution":
        return o.convolution(inputs[0], inputs[1], attrs["strides"], attrs["pads_begin"], attrs["pads_end"], attrs["dilations"])
    elif op_type == "GroupConvolution":
        return o.group_convolution(inputs[0], inputs[1], attrs["strides"], attrs["pads_begin"], attrs["pads_end"], attrs["dilations"])
    elif op_type == "Add":
        return o.add(inputs[0], inputs[1])
    elif op_type == "Subtract":
        return o.subtract(inputs[0], inputs[1])
    elif op_type == "Multiply":
        return o.multiply(inputs[0], inputs[1])
    elif op_type == "Divide":
        return o.divide(inputs[0], inputs[1])
    elif op_type == "Relu":
        return o.relu(inputs[0])
    elif op_type == "PRelu":
        return o.prelu(inputs[0], inputs[1])
    elif op_type == "Clamp":
        return o.clamp(inputs[0], attrs["min"], attrs["max"])
    elif op_type == "MaxPool":
        return o.max_pool(inputs[0], attrs["strides"], attrs["pads_begin"], attrs["pads_end"], attrs["kernel"])
    elif op_type == "AvgPool":
        return o.avg_pool(inputs[0], attrs["strides"], attrs["pads_begin"], attrs["pads_end"], attrs["kernel"], attrs.get("exclude-pad", True))
    elif op_type == "MatMul":
        return o.matmul(inputs[0], inputs[1], attrs.get("transpose_a", False), attrs.get("transpose_b", False))
    elif op_type == "Reshape":
        return o.reshape(inputs[0], _ensure_i64(ov, inputs[1]), attrs.get("special_zero", False))
    elif op_type == "Concat":
        return o.concat(inputs, attrs["axis"])
    elif op_type == "Softmax":
        return o.softmax(inputs[0], attrs.get("axis", 1))
    elif op_type == "Sigmoid":
        return o.sigmoid(inputs[0])
    elif op_type == "Tanh":
        return o.tanh(inputs[0])
    elif op_type == "Convert":
        _OV_ET = {"f32": ov.Type.f32, "f16": ov.Type.f16, "i32": ov.Type.i32, "i64": ov.Type.i64, "i8": ov.Type.i8, "u8": ov.Type.u8}
        dest = ov.Type.f32
        for k, v in _OV_ET.items():
            if k in attrs.get("destination_type", "f32"):
                dest = v; break
        return o.convert(inputs[0], dest)
    elif op_type == "Interpolate":
        mode = attrs.get("mode", "nearest")
        return o.interpolate(inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else inputs[1], mode)
    elif op_type == "ShapeOf":
        return o.shape_of(inputs[0])
    elif op_type == "Squeeze":
        return o.squeeze(inputs[0], inputs[1]) if len(inputs) > 1 else o.squeeze(inputs[0])
    elif op_type == "Unsqueeze":
        return o.unsqueeze(inputs[0], inputs[1])
    elif op_type == "Transpose":
        return o.transpose(inputs[0], _ensure_i64(ov, inputs[1]))
    elif op_type == "Gather":
        return o.gather(inputs[0], _ensure_i64(ov, inputs[1]), _ensure_i64(ov, inputs[2]))
    elif op_type == "StridedSlice":
        return o.strided_slice(
            inputs[0], _ensure_i64(ov, inputs[1]), _ensure_i64(ov, inputs[2]),
            _ensure_i64(ov, inputs[3]) if len(inputs) > 3 else _ensure_i64(ov, inputs[2]),
            attrs.get("begin_mask", []), attrs.get("end_mask", []),
        )
    elif op_type == "ReduceMean":
        return o.reduce_mean(inputs[0], inputs[1], attrs.get("keep_dims", False))
    elif op_type == "ReduceMax":
        return o.reduce_max(inputs[0], inputs[1], attrs.get("keep_dims", False))
    elif op_type == "ReduceSum":
        return o.reduce_sum(inputs[0], inputs[1], attrs.get("keep_dims", False))
    elif op_type == "Power":
        return o.power(inputs[0], inputs[1])
    elif op_type == "Sqrt":
        return o.sqrt(inputs[0])
    elif op_type == "Exp":
        return o.exp(inputs[0])
    elif op_type == "Log":
        return o.log(inputs[0])
    elif op_type == "Abs":
        return o.abs(inputs[0])
    elif op_type == "Negative":
        return o.negative(inputs[0])
    elif op_type == "Floor":
        return o.floor(inputs[0])
    elif op_type == "Ceiling":
        return o.ceiling(inputs[0])
    elif op_type == "Maximum":
        return o.maximum(inputs[0], inputs[1])
    elif op_type == "Minimum":
        return o.minimum(inputs[0], inputs[1])
    elif op_type == "Pad":
        pad_mode = attrs.get("pad_mode", "constant")
        if len(inputs) > 3:
            return o.pad(inputs[0], inputs[1], inputs[2], inputs[3], pad_mode)
        return o.pad(inputs[0], inputs[1], inputs[2], pad_mode)
    elif op_type == "Split":
        return o.split(inputs[0], inputs[1], attrs.get("num_splits", 2))
    elif op_type == "VariadicSplit":
        return o.variadic_split(inputs[0], inputs[1], inputs[2])
    elif op_type == "Broadcast":
        return o.broadcast(inputs[0], inputs[1])
    elif op_type == "Tile":
        return o.tile(inputs[0], inputs[1])
    elif op_type == "BatchNormInference":
        return o.batch_norm_inference(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], attrs.get("epsilon", 1e-5))
    elif op_type == "MVN":
        return o.mvn(inputs[0], inputs[1], attrs.get("normalize_variance", True), attrs.get("eps", 1e-9), attrs.get("eps_mode", "inside_sqrt"))
    elif op_type == "Swish":
        return o.swish(inputs[0])
    elif op_type == "HSigmoid":
        return o.hsigmoid(inputs[0])
    elif op_type == "HSwish":
        return o.hswish(inputs[0])
    elif op_type == "Mish":
        return o.mish(inputs[0])
    elif op_type == "SoftPlus":
        return o.softplus(inputs[0])
    elif op_type == "DetectionOutput":
        return o.detection_output(
            inputs[0], inputs[1], inputs[2],
            attrs,
        )
    elif op_type == "ROIPooling":
        return o.roi_pooling(inputs[0], inputs[1], attrs["output_size"], attrs["spatial_scale"], attrs.get("method", "max"))
    elif op_type == "PSROIPooling":
        return o.psroi_pooling(inputs[0], inputs[1], attrs["output_dim"], attrs["group_size"], attrs["spatial_scale"], attrs.get("spatial_bins_x", 1), attrs.get("spatial_bins_y", 1), attrs.get("mode", "average"))
    elif op_type == "RegionYolo":
        return o.region_yolo(inputs[0], attrs["coords"], attrs["classes"], attrs["num"], attrs.get("do_softmax", True), attrs.get("mask", []), attrs.get("axis", 1), attrs.get("end_axis", 3), attrs.get("anchors", []))
    elif op_type == "Elu":
        return o.elu(inputs[0], attrs.get("alpha", 1.0))
    elif op_type == "Selu":
        return o.selu(inputs[0], inputs[1], inputs[2])
    elif op_type == "Erf":
        return o.erf(inputs[0])
    elif op_type == "Equal":
        return o.equal(inputs[0], inputs[1])
    elif op_type == "NotEqual":
        return o.not_equal(inputs[0], inputs[1])
    elif op_type == "Greater":
        return o.greater(inputs[0], inputs[1])
    elif op_type == "GreaterEqual":
        return o.greater_equal(inputs[0], inputs[1])
    elif op_type == "Less":
        return o.less(inputs[0], inputs[1])
    elif op_type == "LessEqual":
        return o.less_equal(inputs[0], inputs[1])
    elif op_type == "Select":
        return o.select(inputs[0], inputs[1], inputs[2])
    elif op_type == "LogicalNot":
        return o.logical_not(inputs[0])
    elif op_type == "LogicalAnd":
        return o.logical_and(inputs[0], inputs[1])
    elif op_type == "LogicalOr":
        return o.logical_or(inputs[0], inputs[1])
    elif op_type == "PriorBox":
        return o.prior_box(inputs[0], inputs[1], attrs)
    elif op_type == "PriorBoxClustered":
        return o.prior_box_clustered(inputs[0], inputs[1], attrs)
    elif op_type == "NormalizeL2":
        return o.normalize_l2(inputs[0], inputs[1], attrs.get("eps", 1e-10), attrs.get("eps_mode", "add"))
    elif op_type == "Flatten":
        return o.reshape(inputs[0], _ensure_i64(ov, inputs[1]), False) if len(inputs) > 1 else None
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
