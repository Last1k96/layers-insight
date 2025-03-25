#!/usr/bin/env python3

import os
import re
import sys

import cv2
import numpy as np


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


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def configure_inputs_for_submodel(sub_model, model_rt, model_inputs, seed):
    from openvino import Type, Layout
    from openvino.preprocess import PrePostProcessor, ResizeAlgorithm

    # Create a PrePostProcessor instance for the sub-model.
    ppp = PrePostProcessor(sub_model)
    inputs = []

    for i, input_path in enumerate(model_inputs):
        model_input = sub_model.input(i)
        input_shape = list(model_input.get_shape())

        if not input_path or input_path.strip() == "":
            np.random.seed(hash(seed) % (2 ** 32))
            random_array = np.random.rand(*input_shape).astype(np.float32)
            inputs.append(random_array)
            ppp.input(i).tensor().set_shape(random_array.shape)
            ppp.input(i).tensor().set_element_type(Type.f16)
            continue

        if input_path.lower().endswith(IMAGE_EXTENSIONS):
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Error: Failed to load image: {input_path}")

            img = np.expand_dims(img, axis=0)
            inputs.append(img)

            inp = ppp.input(i)
            inp.tensor().set_layout(Layout("NHWC"))
            inp.tensor().set_element_type(Type.u8)
            inp.tensor().set_shape(img.shape)

            inp.preprocess().convert_element_type(Type.f16)

            conv_params = get_conversion_params(model_rt)
            layout = extract_layout(conv_params.get("layout", ""))
            if layout == "nhwc":
                height, width = input_shape[1], input_shape[2]
            else:
                height, width = input_shape[2], input_shape[3]
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
        else:
            data = np.fromfile(input_path, dtype=np.float32)  # Adjust dtype if needed.
            if data.size != np.prod(input_shape):
                raise ValueError(
                    f"Error: Binary file '{input_path}' size ({data.size}) does not match expected shape {input_shape}")

            data = data.reshape(input_shape)
            inputs.append(data)
            ppp.input(i).tensor().set_shape(data.shape)
            ppp.input(i).tensor().set_element_type(Type.f16)

    preprocessed_model = ppp.build()
    return inputs, preprocessed_model


def run_partial_inference(openvino_bin, model_xml, layer_name, ref_plugin, main_plugin, model_inputs, seed):
    ov, core = get_ov_core(openvino_bin)
    model = core.read_model(model=model_xml)

    intermediate_nodes = [op for op in model.get_ops() if op.get_friendly_name() == layer_name]
    if len(intermediate_nodes) != 1:
        raise ValueError(f"Failed to find one node '{layer_name}'")

    parameters = [inp.get_node() for inp in model.inputs]
    sub_model = ov.Model([intermediate_nodes[0]], parameters, "sub_model")

    model_rt = model.get_rt_info()

    inputs, preprocessed_model = configure_inputs_for_submodel(sub_model, model_rt, model_inputs, seed)

    results = []
    for plugin in [main_plugin, ref_plugin]:
        compiled_model = core.compile_model(preprocessed_model, plugin)
        inference_results = compiled_model(inputs)
        results.append(inference_results)

    # Process outputs (assuming a one-to-one correspondence between outputs).
    for main_key, ref_key in zip(results[0].keys(), results[1].keys()):
        main = results[0][main_key]
        ref = results[1][ref_key]
        return {"main": main,
                "ref": ref}
