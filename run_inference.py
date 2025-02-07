#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np

def setup_environment(openvino_bin):
    python_dir = os.path.join(openvino_bin, "python")
    if os.path.isdir(python_dir):
        if python_dir not in sys.path:
            sys.path.insert(0, python_dir)
            print(f"Added '{python_dir}' to sys.path.")
    else:
        print(f"Warning: Python directory '{python_dir}' not found in the OpenVINO bin folder.")


def get_available_plugins(openvino_bin):
    setup_environment(openvino_bin)

    try:
        import openvino as ov
    except ImportError as e:
        sys.exit("Failed to import OpenVINO Inference Engine. "
                 "Ensure that the python libraries from the provided OpenVINO build folder are accessible.\nError: " + str(e))

    core = ov.Core()
    # Example of registering a custom plugin if present
    template_plugin = f"{openvino_bin}/libopenvino_template_plugin.so"
    if os.path.exists(template_plugin):
        core.register_plugin(template_plugin, "TEMPLATE")

    # Return list of devices, e.g. ['CPU', 'GPU', 'TEMPLATE', ...]
    return core.available_devices


def run_inference(openvino_bin, model_xml, device):
    """
    Loads an IR model and runs inference on the given device.

    Args:
        openvino_bin (str): Path to the OpenVINO build's bin folder.
        model_xml (str): Path to the OpenVINO IR .xml file.
        device (str): Target device for inference (e.g., CPU, MYRIAD, GPU).

    Returns:
        dict: A dictionary containing the outputs from the inference.
    """
    # Set up environment variables and sys.path to use libraries from the build folder.
    setup_environment(openvino_bin)

    # Import the IECore class from the local OpenVINO python module.
    try:
        import openvino as ov
    except ImportError as e:
        sys.exit("Failed to import OpenVINO Inference Engine. "
                 "Ensure that the python libraries from the provided OpenVINO build folder are accessible.\nError: " + str(e))

    # Determine the binary weights file corresponding to the provided XML.
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    if not os.path.exists(model_xml) or not os.path.exists(model_bin):
        sys.exit(f"Model files not found. Please verify that both '{model_xml}' and '{model_bin}' exist.")

    # Create the Inference Engine core object.
    core = ov.Core()

    # Register TEMPLATE plugin
    template_plugin = f"{openvino_bin}/libopenvino_template_plugin.so"
    if os.path.exists(template_plugin):
        core.register_plugin(template_plugin, "TEMPLATE")

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
