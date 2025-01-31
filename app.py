# my_dash_ir_app/app.py

import dash
import dash_cytoscape as cyto

from openvino_graph import parse_openvino_ir
from layout import create_layout
from callbacks import register_callbacks

def create_app(ir_xml_path=None):
    """
    Create and configure the Dash app.
    :param ir_xml_path: Path to an OpenVINO IR .xml file.
    """

    # 1) Load the extra Cytoscape layouts, which includes 'dagre'
    cyto.load_extra_layouts()

    # 2) Create Dash app
    app = dash.Dash(__name__)

    # If no path provided, use a default or raise an error
    if ir_xml_path is None:
        ir_xml_path = "/home/mkurin/models/bert-large-uncased-whole-word-masking-squad-int8-0001/bert-large-uncased-whole-word-masking-squad-int8-0001.xml"
        # ir_xml_path = "/home/mkurin/models/age-gender-recognition-retail-0013/age-gender-recognition-retail-0013.xml"  # Replace with a valid path or handle differently

    # Parse IR => Cytoscape elements
    elements = parse_openvino_ir(ir_xml_path)

    # Assign layout
    app.layout = create_layout(elements)

    # Register callbacks
    register_callbacks(app)

    return app

