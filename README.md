# LayerInsight

LayerInsight is a visualization tool for OpenVINO models that allows you to explore and visualize the internal layers of neural networks.

## Quick Start

A convenient `run.py` script has been provided to simplify the setup and execution process. This script works on all platforms (Windows, macOS, and Linux) and will:

1. Check if Python is installed
2. Create a virtual environment in a `.venv` folder (if it doesn't exist)
3. Install all dependencies using the setup file
4. Run the application with the arguments you provide

### Usage

```
chmod +x run.py
./run.py [arguments for main.py]
```

### Required Arguments

- `--openvino_bin`: Path to OpenVINO bin directory
- `--model`: Path to model XML file
- `--inputs`: Comma-separated list of input file paths

### Optional Arguments

- `--port`: Port to start the server on (default: 8050)
- `--debug`: Run the app in debug mode

### Examples

For Windows:
```
python run.py --openvino_bin "/path/to/openvino/build/Release" --model "path\to\model.xml" --inputs "path\to\input1.jpg,path\to\input2.jpg" --port 8080
```

For Linux/macOS:
```
python run.py --openvino_bin "/path/to/openvino/build/Release" --model "path/to/model.xml" --inputs "path/to/input1.jpg,path/to/input2.jpg" --port 8080
```

## Manual Setup

If you prefer to set up the environment manually:

1. Create a virtual environment:
   ```
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -e .
   ```

4. Run the application:
   ```
   python main.py [arguments]
   ```
   
## OpenVINO Requirements

LayerInsight requires OpenVINO with Python API support. When building OpenVINO from source, make sure to use the `ENABLE_PYTHON=ON` cmake flag:

```
cmake -DENABLE_PYTHON=ON ...
```

This ensures that the Python API is available, which is required for LayerInsight to interact with OpenVINO models.

## Features

- **Interactive Graph Visualization**: Navigate through your model's computational graph with an interactive visualization
- **Layer-by-Layer Analysis**: Inspect the outputs of each layer in the network
- **Plugin Comparison**: Compare outputs between different OpenVINO plugins (CPU, GPU, etc.)
- **Keyboard Navigation**:
  - Use **Arrow Up/Down** keys to navigate through the layers list
  - Use **Home/End** keys to jump to the first/last layer
  - Use **PageUp/PageDown** keys to move up/down by multiple layers
  - Hold **Ctrl** key while selecting a node to center the graph view on that node

## User Interface

LayerInsight provides a powerful interface for exploring neural network models:

### Main Components

- **Left Panel**: Contains the list of layers in the model. You can navigate through this list using the keyboard shortcuts mentioned above.
- **Center Panel**: Displays the computational graph of the model. You can:
  - Click on nodes to select them
  - Hold **Ctrl** while clicking to center the view on a node
  - Zoom in/out using the mouse wheel
  - Pan by clicking and dragging
- **Right Panel**: Shows detailed information about the selected layer, including:

## Important Notes

Here are some important tips, limitations, and known issues to be aware of when using LayerInsight:

### Tips and Useful Information

- **Plugin Logging**: You can pass the `LOG_INFO` parameter in the settings to a plugin to view compiler logs.
- **Image Pre-processing**: When performing inference with an image, LayerInsight automatically applies pre-processing (resize, interpolation, etc.) based on the meta information from the model.xml file. This is done using a pre-postprocessor that inserts additional operations in the network which are not shown in the main graph.
- **Layer Transformation**: When transforming a layer input into a model input, the layer's outputs are used to run inference. The application takes results from the main plugin. Note that the compounded error is not preserved between plugins, so the first couple of layers will show almost exact results.
- **Random Input Generation**: If no input path is provided, the application will generate random input data using a seed derived from the output folder name.
- **Inference Timeout**: The application has a 5-minute timeout for inference execution to prevent hanging on problematic models.
- **Subprocess Execution**: Inference is run in a separate subprocess to isolate OpenVINO execution and catch potential segfaults.

### Limitations

- **Visualization Performance**: Visualizations may take a few seconds to load after selection. Some visualizations can be particularly slow when the number of channels is large.
- **Layout and Precision**: Currently, you cannot configure input/output layout or precision explicitly. The output is in NCHW layout with fp32 precision.
- **Dynamic Shapes**: Dynamic shapes support is limited. There is no way to set the upper bounds for the inputs.
- **NHWC Format**: Visualizations will not work correctly for explicit NHWC shapes (commonly found in newer TensorFlow models).
- **Tensor Dimensions**: The reshape_to_3d function only supports tensors up to 5D. Tensors with more dimensions may not be visualized correctly.
- **Visualization Parameters**: Many visualization functions use hardcoded parameters (colorscales, thresholds, etc.) that cannot be customized by the user.

### Known Issues

- **Input Cutting Bug**: When cutting the input of the model, the first inference is skipped. You may need to click the node twice to get results.
- **Queue Clearing**: Clearing the queue sometimes leads to desynchronization between the left panel, graph, and results due to a race condition.
- **Correlation Calculation**: The channel correlation calculation may fail if the tensor contains NaN or infinite values, which are not fully handled in all visualizations.
