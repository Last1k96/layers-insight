import os

from dash import no_update, callback_context
from dash.dependencies import Input, Output, State
from run_inference import get_available_plugins, run_partial_inference

from cache import result_cache, task_queue, lock


def register_callbacks(app):

    @app.callback(
        Output('current-node-store', 'data'),
        Output('right-panel', 'children'),
        Output('ir-graph', 'elements'),  # Output for graph elements
        Input('ir-graph', 'tapNode'),
        State('openvino-bin-input', 'value'),
        State('model-xml-input', 'value'),
        State('reference-plugin-dropdown', 'value'),
        State('other-plugin-dropdown', 'value'),
        State('input-file-input', 'value'),
        State('current-node-store', 'data'),
        State('ir-graph', 'elements'),  # State for current graph elements
        prevent_initial_call=True
    )
    def on_node_click(clicked_node, openvino_bin, model_xml,
                               ref_plugin, main_plugin, input_path, current_node, elements):
        ctx = callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'ir-graph':  # Node Click
            if not clicked_node:
                return no_update, "Click a node", no_update

            layer_name = clicked_node['data']['layer_name']

            with lock:
                cached_result = result_cache.get(layer_name)

            if cached_result: #If already computed
                return layer_name, f"Partial Inference result: {cached_result}", update_node_style(elements, layer_name, 'green')

            if not all([openvino_bin, model_xml, ref_plugin, main_plugin, input_path]):
                return no_update, "Missing required parameters", no_update

            # Mark node as processing (orange border)
            elements = update_node_style(elements, layer_name, 'orange')

            task_queue.put((layer_name, openvino_bin, model_xml,
                            ref_plugin, main_plugin, input_path))

            return layer_name, "Processing...", elements  # Return updated elements

        return no_update, no_update, no_update



    def update_node_style(elements, layer_name, color):
        new_elements = []  # Create a new list to avoid modifying the original directly.
        for element in elements:
            new_element = element.copy() #Create a copy of the element
            if 'layer_name' in new_element['data'] and new_element['data']['layer_name'] == layer_name:
                new_element['data']['border_color'] = color

            new_elements.append(new_element)

        return new_elements

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
