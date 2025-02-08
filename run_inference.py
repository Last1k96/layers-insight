#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np


def get_ov_core(openvino_bin):
    python_dir = os.path.join(openvino_bin, "python")
    if os.path.isdir(python_dir):
        if python_dir not in sys.path:
            sys.path.insert(0, python_dir)
    else:
        print(f"Warning: Python directory '{python_dir}' not found in the OpenVINO bin folder.")

    try:
        import openvino as ov
    except ImportError as e:
        print("Failed to import OpenVINO Inference Engine. "
              "Ensure that the python libraries from the provided OpenVINO build folder are accessible.\nError: " + str(
            e))
        return

    core = ov.Core()

    template_plugin = f"{openvino_bin}/libopenvino_template_plugin.so"
    if os.path.exists(template_plugin):
        core.register_plugin(template_plugin, "TEMPLATE")

    return ov, core


def get_available_plugins(openvino_bin):
    ov, core = get_ov_core(openvino_bin)

    return core.available_devices


def run_partial_inference(openvino_bin, model_xml, node_name, ref_plugin, main_plugin):
    ov, core = get_ov_core(openvino_bin)
    model = core.read_model(model=model_xml)

    intermediate_nodes = [op for op in model.get_ops() if op.get_friendly_name() == node_name]
    if len(intermediate_nodes) != 1:
        print(f"Failed to find one node '{node_name}'")
        return

    parameters = [inp.get_node() for inp in model.inputs]
    sub_model = ov.Model([intermediate_nodes[0]], parameters, "sub_model")

    inputs = [np.random.rand(*input_blob.shape).astype(np.float32) for input_blob in model.inputs]

    results = []
    for plugin in [main_plugin, ref_plugin]:
        compiled_model = core.compile_model(sub_model, plugin)
        inference_results = compiled_model(inputs)
        results.append(inference_results)

    def get_stats(data):
        res = str()
        res += "Min: " + str(np.min(main)) + "\r\n"
        res += "Max: " + str(np.max(main)) + "\r\n"
        res += "Mean: " + str(np.mean(main)) + "\r\n"
        res += "Std: " + str(np.std(main)) + "\r\n"
        return res

    stats = str()
    for main_key, ref_key in zip(results[0].keys(), results[1].keys()):
        main = results[0][main_key]
        ref = results[1][ref_key]

        stats += "Main plugin\r\n"
        stats += get_stats(main)
        stats += "Ref plugin\r\n"
        stats += get_stats(ref)

        diff = main - ref
        stats += "Difference\r\n"
        stats += get_stats(diff)

    return stats
