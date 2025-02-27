import dash
import dash_cytoscape as cyto
from dash import Output, Input, State

from cache import process_tasks
from layout import create_layout
from callbacks import register_callbacks, register_clientside_callbacks
import dash_bootstrap_components as dbc

import threading


def create_app(openvino_path, ir_xml_path, inputs_path):
    cyto.load_extra_layouts()
    app = dash.Dash(__name__, title="Layers Insight", external_stylesheets=[dbc.themes.DARKLY])
    app.layout = create_layout(openvino_path, ir_xml_path, inputs_path)

    register_callbacks(app)
    register_clientside_callbacks(app)

    threading.Thread(target=process_tasks, daemon=True).start()

    return app
