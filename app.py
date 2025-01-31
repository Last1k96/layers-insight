# my_dash_cyto_app/app.py

import dash
from openvino_graph import parse_openvino_ir
from layout import create_layout
from callbacks import register_callbacks

def create_app(ir_path: str = None):
    """
    Create and configure the Dash app.
    :param ir_path: Path to an OpenVINO IR file (optional).
    """
    # 1. Instantiate the Dash app
    app = dash.Dash(__name__)

    # 2. Parse the IR to get Cytoscape elements (or use a default if None)
    if ir_path is None:
        elements = parse_openvino_ir("path/to/default.xml")  # or just parse_openvino_ir(None)
    else:
        elements = parse_openvino_ir(ir_path)

    # 3. Assign layout
    app.layout = create_layout(elements)

    # 4. Register all callbacks
    register_callbacks(app)

    return app
