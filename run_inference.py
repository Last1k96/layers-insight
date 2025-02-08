#!/usr/bin/env python3

import os
import sys
import argparse
from typing import LiteralString

import numpy as np
import cv2
import re


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

def do_image_preprocessing(img, input_shape, layout, mean_vals, scale_vals, reverse_channels):
    if layout and layout == "nhwc":
        target_h, target_w = input_shape[1], input_shape[2]
    else:  # assume nchw
        target_h, target_w = input_shape[2], input_shape[3]
        
    img = cv2.resize(img, (target_w, target_h))
    if reverse_channels:
        img = img[..., ::-1]
    img = img.astype(np.float32)
    if mean_vals is not None:
        img = img - np.array(mean_vals)
    if scale_vals is not None:
        img = img / np.array(scale_vals)
    if not layout or layout == "nchw":
        img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def parse_shape(s):
    return [int(x.strip()) for x in s.strip("[]").split(",")]


def extract_layout(s):
    m = re.search(r'\((.*?)\)', s)
    return m.group(1).lower() if m else None


def parse_values(s):
    m = re.search(r'\[(.*?)\]', s)
    return [float(x) for x in m.group(1).split(",")] if m else None


def get_conversion_params(model_rt):
    conv = model_rt["conversion_parameters"] if "conversion_parameters" in model_rt else {}
    params = {}
    for key in conv.keys():
        params[key] = conv[key].astype(str)

    return params


def preprocess_image_for_model(img, model):
    model_rt = model.get_rt_info()
    conv_params = get_conversion_params(model_rt)

    input_shape = parse_shape(conv_params.get("input_shape", ""))
    layout = extract_layout(conv_params.get("layout", ""))
    mean_vals = parse_values(conv_params.get("mean_values", ""))
    scale_vals = parse_values(conv_params.get("scale_values", ""))
    reverse_channels = conv_params.get("reverse_input_channels", "false").lower() == "true"

    return do_image_preprocessing(img, input_shape, layout, mean_vals, scale_vals, reverse_channels)


def run_partial_inference(openvino_bin, model_xml, node_name, ref_plugin, main_plugin, input_path):
    ov, core = get_ov_core(openvino_bin)
    model = core.read_model(model=model_xml)

    # Define dimensions
    batch_size = 1
    seq_length = 384
    vocab_size = 30522  # Typical BERT vocab size

    # Generate random input_ids: integers in the range [0, vocab_size)
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_length), dtype=np.int32)

    # Generate attention_mask: all ones (shape: [1, 384])
    attention_mask = np.ones((batch_size, seq_length), dtype=np.int32)

    # Generate token_type_ids: all zeros (shape: [1, 384])
    token_type_ids = np.zeros((batch_size, seq_length), dtype=np.int32)

    # Bundle inputs in a list or dict as required by your inference code.
    # For example, if your model expects a list:
    inputs = [input_ids, attention_mask, token_type_ids]
    #img = cv2.imread(input_path)
    #if img is None:
    #    print("Failed to load image.")
    #    return

    #img = preprocess_image_for_model(img, model)

    intermediate_nodes = [op for op in model.get_ops() if op.get_friendly_name() == node_name]
    if len(intermediate_nodes) != 1:
        print(f"Failed to find one node '{node_name}'")
        return

    parameters = [inp.get_node() for inp in model.inputs]
    sub_model = ov.Model([intermediate_nodes[0]], parameters, "sub_model")

    #inputs = [img]

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
