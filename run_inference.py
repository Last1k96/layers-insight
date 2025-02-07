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
            print(f"Added '{python_dir}' to sys.path.")
    else:
        print(f"Warning: Python directory '{python_dir}' not found in the OpenVINO bin folder.")

    try:
        import openvino as ov
    except ImportError as e:
        print("Failed to import OpenVINO Inference Engine. "
                 "Ensure that the python libraries from the provided OpenVINO build folder are accessible.\nError: " + str(e))
        return

    core = ov.Core()

    template_plugin = f"{openvino_bin}/libopenvino_template_plugin.so"
    if os.path.exists(template_plugin):
        core.register_plugin(template_plugin, "TEMPLATE")

    return core


def get_available_plugins(openvino_bin):
    core = get_ov_core(openvino_bin)

    return core.available_devices


def run_partial_inference(openvino_bin, model_xml, node_name, ref_plugin, main_plugin):
    core = get_ov_core(openvino_bin)
    model = core.read_model(model=model_xml)

    for op in model.get_ops():
        if op.get_friendly_name() == node_name:
            model.add_outputs(op.output(0))
            break
    else:
        return f"NODE NOT FOUND: {node_name}"

    model = core.compile_model(model, ref_plugin)
    inputs = [np.random.rand(*input_blob.shape).astype(np.float32) for input_blob in model.inputs]
    result = model(inputs)

    return result


def run_inference(openvino_bin, model_xml, device):
    core = get_ov_core(openvino_bin)

    print(f"Available devices: {core.available_devices}")

    print(f"Reading network from '{model_xml}'...")
    model = core.read_model(model=model_xml)

    print(f"Loading network on device '{device}'...")
    compiled_model = core.compile_model(model, device)

    # Retrieve the input blob name and its shape.
    input_blob = next(iter(model.inputs))
    input_shape = input_blob.shape
    print(f"Input blob: {input_blob} with shape {input_shape}")

    # Create dummy input data (using random values for demonstration).
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    print("Starting inference...")
    result = compiled_model(dummy_input)

    # Display the outputs.
    for output_data in result:
        print(f"Output blob: {output_data.any_name} has shape {output_data.shape}")

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on an OpenVINO IR model using Python libraries from a given OpenVINO build folder."
    )
    parser.add_argument(
        "--openvino_bin", type=str, required=False,
        help="Path to the OpenVINO build's bin folder containing plugin libraries and the python folder.",
        default="/home/mkurin/code/openvino/bin/intel64/Release"
    )
    parser.add_argument(
        "--model", type=str, required=False,
        help="Path to the OpenVINO IR .xml file.",
        default="/home/mkurin/models/age-gender-recognition-retail-0013/age-gender-recognition-retail-0013.xml"
    )
    parser.add_argument(
        "--device", type=str, default="CPU",
        help="Target device for inference. Default is CPU.",
    )
    args = parser.parse_args()

    # Run inference and capture the output.
    result = run_inference(args.openvino_bin, args.model, args.device)
    print("Inference completed successfully. Output keys:")
    print(list(result.keys()))

if __name__ == "__main__":
    main()
