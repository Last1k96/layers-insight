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
    # Use openvino library from local openvino bin folder. Should not depend on an openvino package.
    from openvino import Type, Layout
    from openvino.preprocess import PrePostProcessor, ResizeAlgorithm

    ppp = PrePostProcessor(sub_model)
    inputs = []

    for i, input_path in enumerate(model_inputs):
        model_input = sub_model.input(i)
        input_shape = list(model_input.get_shape())

        # If input_path is empty, generate a random input.
        if not input_path or input_path.strip() == "":
            np.random.seed(hash(seed) % (2 ** 32))
            random_array = np.random.rand(*input_shape).astype(np.float32)
            inputs.append(random_array)
            ppp.input(i).tensor().set_shape(random_array.shape)
            ppp.input(i).tensor().set_element_type(Type.f16)
            continue

        # Handle image inputs.
        if input_path.lower().endswith(IMAGE_EXTENSIONS):
            # Read the image using OpenCV (resulting in an image with shape [H, W, C]).
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Error: Failed to load image: {input_path}")

            # Get conversion parameters and expected layout from model metadata.
            conv_params = get_conversion_params(model_rt)

            input_rt = model_input.get_rt_info()
            expected_layout = input_rt["layout_0"].astype(str) if "layout_0" in input_rt else "[N,C,H,W]"

            # Determine target dimensions and prepare the image based on expected layout.
            if expected_layout and expected_layout.lower() == "[N,H,W,C]":
                target_height = input_shape[1]
                target_width = input_shape[2]
                resized_img = cv2.resize(img, (target_width, target_height))
                # Add a batch dimension.
                processed_img = np.expand_dims(resized_img, axis=0)
            else:
                target_height = input_shape[2]
                target_width = input_shape[3]
                resized_img = cv2.resize(img, (target_width, target_height))
                processed_img = np.expand_dims(resized_img, axis=0)
                processed_img = np.transpose(processed_img, (0, 3, 1, 2))

            inputs.append(processed_img)

            # Configure the input tensor using the expected layout.
            inp = ppp.input(i)
            inp.tensor().set_shape(processed_img.shape)
            inp.tensor().set_element_type(Type.u8)
            inp.tensor().set_layout(Layout(expected_layout))

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
            # For binary file inputs.
            data = np.fromfile(input_path, dtype=np.float32)  # Adjust dtype if needed.
            if data.size != np.prod(input_shape):
                raise ValueError(
                    f"Error: Binary file '{input_path}' size ({data.size}) does not match expected shape {input_shape}"
                )
            data = data.reshape(input_shape)
            inputs.append(data)
            ppp.input(i).tensor().set_shape(data.shape)
            ppp.input(i).tensor().set_element_type(Type.f16)

    # Build the preprocessed model.
    preprocessed_model = ppp.build()
    return inputs, preprocessed_model


def clean_empty_values(d):
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if v is not None and v != ""}


def run_partial_inference(openvino_bin, model_xml, layer_name, ref_plugin, main_plugin, model_inputs, seed,
                          plugins_config, cancel_event):
    ov, core, inputs, preprocessed_model = prepare_submodel_and_inputs(layer_name, model_inputs, model_xml,
                                                                       openvino_bin, seed)

    # Compile models for both plugins
    cm_main = core.compile_model(preprocessed_model, main_plugin,
                                 config=clean_empty_values(plugins_config.get(main_plugin, {})))
    cm_ref = core.compile_model(preprocessed_model, ref_plugin,
                                config=clean_empty_values(plugins_config.get(ref_plugin, {})))

    # Create and start async inference requests
    ir_main, ir_ref = cm_main.create_infer_request(), cm_ref.create_infer_request()
    ir_main.start_async(inputs)
    ir_ref.start_async(inputs)

    # Poll in small intervals to allow cancellation
    while not (ir_main.wait_for(10) and ir_ref.wait_for(10)):  # 10 ms
        if cancel_event.is_set():
            cancel_event.clear()
            ir_main.cancel()
            ir_ref.cancel()
            raise RuntimeError("Inference cancelled")

    # Collect results
    return [{"main": ir_main.results[m_key], "ref": ir_ref.results[r_key]}
            for m_key, r_key in zip(ir_main.results.keys(), ir_ref.results.keys())]


def prepare_submodel_and_inputs(layer_name, model_inputs, model_xml, openvino_bin, seed):
    ov, core = get_ov_core(openvino_bin)
    model = core.read_model(model=model_xml)

    # Find the target layer
    intermediate_nodes = [op for op in model.get_ops() if op.get_friendly_name() == layer_name]
    if len(intermediate_nodes) != 1:
        raise ValueError(f"Failed to find node '{layer_name}'")

    # Create submodel from the target layer to inputs
    sub_model = ov.Model(intermediate_nodes[0].outputs(),
                         [inp.get_node() for inp in model.inputs],
                         "sub_model")

    # Configure inputs and return everything needed for inference
    inputs, preprocessed_model = configure_inputs_for_submodel(sub_model, model.get_rt_info(), model_inputs, seed)
    return ov, core, inputs, preprocessed_model
