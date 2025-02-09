import dash
import dash_cytoscape as cyto
from dash.long_callback import DiskcacheLongCallbackManager

from layout import create_layout
from callbacks import register_callbacks
from cache import cache


def create_app(openvino_path, ir_xml_path, inputs_path):
    cyto.load_extra_layouts()
    long_callback_manager = DiskcacheLongCallbackManager(cache)
    app = dash.Dash(__name__, long_callback_manager=long_callback_manager, title="Layers Insight")

    app.layout = create_layout(openvino_path, ir_xml_path, inputs_path)
    register_callbacks(app)

    return app

