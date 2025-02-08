import uuid

import dash
import dash_cytoscape as cyto

from layout import create_layout
from callbacks import register_callbacks


def create_app(openvino_path, ir_xml_path, inputs_path):
    cyto.load_extra_layouts()
    app = dash.Dash(title="Layers Insight")

    app.layout = create_layout(openvino_path, ir_xml_path, inputs_path)
    register_callbacks(app)

    return app

