# my_dash_cyto_app/callbacks.py

from dash.dependencies import Input, Output, State
from dash import dcc, html
import dash_cytoscape as cyto

from partial_inference import run_partial_inference


def register_callbacks(app):
    """
    Define and register all callbacks with the Dash app instance.
    """

    @app.callback(
        Output('inference-output', 'children'),
        Input('cytoscape-graph', 'tapNode')
    )
    def on_node_click(tapped_node):
        """
        Triggered when a user taps/clicks on a node in the Cytoscape graph.
        `tapped_node` is a dict with 'data': {'id': ..., 'label': ...}.
        """
        if tapped_node is None:
            return "Click on a node to see partial inference results."

        node_id = tapped_node['data']['id']
        result = run_partial_inference(node_id)
        return f"Partial Inference for node '{node_id}': {result}"
