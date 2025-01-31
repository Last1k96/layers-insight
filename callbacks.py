# my_dash_ir_app/callbacks.py

from dash.dependencies import Input, Output
from partial_inference import run_partial_inference

def register_callbacks(app):
    """
    Register all Dash callbacks here.
    """
    @app.callback(
        Output('inference-output', 'children'),
        Input('ir-graph', 'tapNode')
    )
    def on_node_click(tapped_node):
        if tapped_node is None:
            return "Click a node to see partial inference results."
        node_id = tapped_node['data']['id']
        result = run_partial_inference(node_id)
        return f"Partial Inference result: {result}"
