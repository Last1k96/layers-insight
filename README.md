# LayerInsight

LayerInsight is a visualization tool for OpenVINO models.

## Quick Start

A convenient `run.py` script has been provided to simplify the setup and execution process. This script works on all platforms (Windows, macOS, and Linux) and will:

1. Check if Python is installed
2. Create a virtual environment in a `.venv` folder (if it doesn't exist)
3. Install all dependencies using the setup file
4. Run the application with the arguments you provide

### Usage

```
python run.py [arguments for main.py]
```

On Unix-like systems (Linux, macOS), you may need to make the script executable first:
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
python run.py --openvino_bin "C:\Program Files (x86)\Intel\openvino_2021\bin" --model "path\to\model.xml" --inputs "path\to\input1.jpg,path\to\input2.jpg" --port 8080
```

For Linux/macOS:
```
python run.py --openvino_bin "/opt/intel/openvino/bin" --model "path/to/model.xml" --inputs "path/to/input1.jpg,path/to/input2.jpg" --port 8080
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
