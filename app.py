import dash
import dash_cytoscape as cyto

from cache import process_tasks
from layout import create_layout
from callbacks import register_callbacks
import dash_bootstrap_components as dbc

import threading

def create_app(openvino_path, ir_xml_path, inputs_path):
    cyto.load_extra_layouts()
    app = dash.Dash(__name__, title="Layers Insight", external_stylesheets=[dbc.themes.SLATE])

    threading.Thread(target=process_tasks, daemon=True).start()

    app.layout = create_layout(openvino_path, ir_xml_path, inputs_path)
    register_callbacks(app)

    return app

