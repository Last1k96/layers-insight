import os

from dash.dependencies import Input, Output, State
from partial_inference import run_partial_inference
from run_inference import get_available_plugins

def register_callbacks(app):
    """
    Register all Dash callbacks here.
    """
    @app.callback(
        Output('right-panel', 'children'),
        Input('ir-graph', 'tapNode'),
        prevent_initial_call=True
    )
    def on_node_click(tapped_node):
        print("on_node_click")
        if tapped_node is None:
            return "Click a node to see partial inference results."
        node_id = tapped_node['data']['id']
        for key in tapped_node:
            print(f"{key}: {tapped_node[key]}")
        result = run_partial_inference(node_id)
        return f"Partial Inference result: {result}"

    @app.callback(
        Output('available-plugins-list', 'children'),
        Output('reference-plugin-dropdown', 'options'),
        Output('other-plugin-dropdown', 'options'),
        Output('reference-plugin-dropdown', 'value'),
        Output('other-plugin-dropdown', 'value'),
        Input('find-plugins-btn', 'n_clicks'),
        State('openvino-bin-input', 'value'),
        prevent_initial_call=True
    )
    def find_plugins(n_clicks, openvino_bin):
        if not openvino_bin or not os.path.exists(openvino_bin):
            return ("Invalid OpenVINO bin path", [], [], None, None)

        devices = get_available_plugins(openvino_bin)  # e.g. ['CPU','GPU','TEMPLATE']
        if not devices:
            return ("No plugins found.", [], [], None, None)

        # Build dropdown options
        device_options = [{'label': d, 'value': d} for d in devices]

        # Choose default reference = CPU if present
        ref_value = 'CPU' if 'CPU' in devices else devices[0]

        # Choose default other plugin = first non-CPU if possible
        non_cpu_devices = [d for d in devices if d != 'CPU']
        other_value = non_cpu_devices[0] if len(non_cpu_devices) > 0 else ref_value

        return (
            "\n".join(devices),  # Text listing of plugins
            device_options,  # reference-plugin-dropdown options
            device_options,  # other-plugin-dropdown options
            ref_value,  # reference-plugin-dropdown value
            other_value  # other-plugin-dropdown value
        )
