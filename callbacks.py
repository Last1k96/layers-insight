import os

from dash.dependencies import Input, Output, State
from run_inference import get_available_plugins, run_partial_inference

def register_callbacks(app):
    """
    Register all Dash callbacks here.
    """
    @app.callback(
        Output('right-panel', 'children'),
        Input('ir-graph', 'tapNode'),
        State('openvino-bin-input', 'value'),
        State('model-xml-input', 'value'),
        State('reference-plugin-dropdown', 'value'),
        State('other-plugin-dropdown', 'value'),
        State('input-file-input', 'value'),
        prevent_initial_call=True
    )
    def on_node_click(tapped_node, openvino_bin, model_xml, ref_plugin, main_plugin, input_path):
        node_name = tapped_node['data']['layer_name']
        result = run_partial_inference(openvino_bin, model_xml, node_name, ref_plugin, main_plugin, input_path)
        return f"Partial Inference result: {result}"

    @app.callback(
        Output('available-plugins-list', 'children'),
        Output('reference-plugin-dropdown', 'options'),
        Output('other-plugin-dropdown', 'options'),
        Output('reference-plugin-dropdown', 'value'),
        Output('other-plugin-dropdown', 'value'),
        Input('find-plugins-btn', 'n_clicks'),
        State('openvino-bin-input', 'value'),
        prevent_initial_call=False
    )
    def find_plugins(n_clicks, openvino_bin):
        if not openvino_bin or not os.path.exists(openvino_bin):
            return "Invalid OpenVINO bin path", [], [], None, None

        devices = get_available_plugins(openvino_bin)
        if not devices:
            return "No plugins found.", [], [], None, None

        device_options = [{'label': d, 'value': d} for d in devices]

        ref_value = 'CPU' if 'CPU' in devices else devices[0]

        non_cpu_devices = [d for d in devices if d != 'CPU']
        other_value = non_cpu_devices[0] if len(non_cpu_devices) > 0 else ref_value

        return (
            "",  # Text listing of plugins
            device_options,  # reference-plugin-dropdown options
            device_options,  # other-plugin-dropdown options
            ref_value,  # reference-plugin-dropdown value
            other_value  # other-plugin-dropdown value
        )
