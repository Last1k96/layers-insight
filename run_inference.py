#!/usr/bin/env python3

import os
import re
import sys

import cv2
import numpy as np
import dash_bootstrap_components as dbc
from dash import html

from metrics import comparison_metrics_table


# TODO extract to some kind of callback to run when openvino path is provided
def import_local_openvino(openvino_bin):
    python_dir = os.path.join(openvino_bin, "python")
    if os.path.isdir(python_dir):
        if python_dir not in sys.path:
            sys.path.insert(0, python_dir)
    else:
        print(f"Warning: Python directory '{python_dir}' not found in the OpenVINO bin folder.")


def get_ov_core(openvino_bin):
    import_local_openvino(openvino_bin)

    import openvino as ov
    core = ov.Core()

    template_plugin = f"{openvino_bin}/libopenvino_template_plugin.so"
    if os.path.exists(template_plugin):
        core.register_plugin(template_plugin, "TEMPLATE")

    return ov, core


def get_available_plugins(openvino_bin):
    ov, core = get_ov_core(openvino_bin)

    return core.available_devices


def get_input_names_from_model(model_path):
    pass


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


def configure_preprocessor(model, model_rt, img):
    # openvino dependencies should be available through developer's openvino binaries
    from openvino import Type, Layout
    from openvino.preprocess import PrePostProcessor, ResizeAlgorithm

    conv_params = get_conversion_params(model_rt)

    layout = extract_layout(conv_params.get("layout", ""))

    # TODO configure pre-post processor for multiple inputs
    # or just not use ppp due to lack of multi-image input models?
    ppp = PrePostProcessor(model)

    inp = ppp.input()

    # cv2 reads images in channel-minor layout
    inp.tensor().set_layout(Layout("NHWC"))
    inp.tensor().set_element_type(Type.u8)
    inp.tensor().set_shape(img.shape)

    inp.preprocess().convert_element_type(Type.f16)  # Should there be fp16? should it be configurable?

    input_shape = parse_shape(conv_params.get("input_shape", ""))
    height, width = (input_shape[1], input_shape[2]) if layout == "nhwc" else (input_shape[2], input_shape[3])
    inp.preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR, height, width)

    reverse_channels = conv_params.get("reverse_input_channels", "false").lower() == "true"
    if reverse_channels:
        inp.preprocess().reverse_channels()

    mean_vals = parse_values(conv_params.get("mean_values", ""))
    if mean_vals:
        inp.preprocess().mean(mean_vals)

    scale_vals = parse_values(conv_params.get("scale_values", ""))
    if scale_vals:
        inp.preprocess().scale(scale_vals)

    preprocessed_model = ppp.build()
    return preprocessed_model


def run_partial_inference(openvino_bin, model_xml, layer_name, ref_plugin, main_plugin, model_inputs, seed):
    ov, core = get_ov_core(openvino_bin)
    model = core.read_model(model=model_xml)

    intermediate_nodes = [op for op in model.get_ops() if op.get_friendly_name() == layer_name]
    if len(intermediate_nodes) != 1:
        print(f"Failed to find one node '{layer_name}'")
        return

    parameters = [inp.get_node() for inp in model.inputs]
    sub_model = ov.Model([intermediate_nodes[0]], parameters, "sub_model")

    # TODO support multiple inputs
    input_path = model_inputs[0]

    # model_input = model.inputs[0]
    # np.random.seed(hash(seed) % (2 ** 32))
    # shape = list(model_input.shape)
    # random_array = np.random.rand(*shape)

    # TODO figure out a way of differentiating between an image and a binary input to consider preprocessing
    img = cv2.imread(input_path, cv2.IMREAD_COLOR_RGB)
    if img is None:
        print(f"Error: Failed to load image: {input_path=}")
        return "Error: Failed to load image"

    img = np.expand_dims(img, axis=0)

    # Use runtime info from the original model because cut model loses that information for some reason
    model_rt = model.get_rt_info()
    sub_model = configure_preprocessor(sub_model, model_rt, img)

    inputs = [img]  # [img]


    # Define dimensions
    # batch_size = 1
    # seq_length = 384
    # vocab_size = 30522  # Typical BERT vocab size
    # input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_length), dtype=np.int32)
    # attention_mask = np.ones((batch_size, seq_length), dtype=np.int32)
    # token_type_ids = np.zeros((batch_size, seq_length), dtype=np.int32)
    # inputs = [input_ids, attention_mask, token_type_ids]

    results = []
    try:
        for plugin in [main_plugin, ref_plugin]:
            compiled_model = core.compile_model(sub_model, plugin)
            inference_results = compiled_model(inputs)
            results.append(inference_results)
    except Exception as e:
        print(e)

    # TODO multiple outputs
    for main_key, ref_key in zip(results[0].keys(), results[1].keys()):
        main = results[0][main_key]
        ref = results[1][ref_key]

        right_panel_div = html.Div([
            dbc.CardGroup([
                comparison_metrics_table(ref, main)
            ])
        ])

        return {"right-panel": right_panel_div,
                "main": main,
                "ref": ref}
