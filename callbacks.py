import os

from dash import no_update, callback
from dash.dependencies import Input, Output, State
from run_inference import get_available_plugins, run_partial_inference

def register_callbacks(app):
    def update_node_style(elements, layer_name, color):
        new_elements = []  # Create a new list to avoid modifying the original directly.
        for element in elements:
            new_element = element.copy() #Create a copy of the element
            if 'layer_name' in new_element['data'] and new_element['data']['layer_name'] == layer_name:
                new_element['data']['border_color'] = color

            new_elements.append(new_element)

        return new_elements

    # Define the callback using background=True.
    # This callback is triggered when a node is tapped.
    @callback(
        Output('current-node-store', 'data'),
        Output('right-panel', 'children'),
        Output('ir-graph', 'elements'),
        Input('ir-graph', 'tapNode'),
        State('openvino-bin-input', 'value'),
        State('model-xml-input', 'value'),
        State('reference-plugin-dropdown', 'value'),
        State('other-plugin-dropdown', 'value'),
        State('input-file-input', 'value'),
        State('current-node-store', 'data'),
        State('ir-graph', 'elements'),
        background=True,  # This tells Dash to run the callback in the background.
        prevent_initial_call=True
    )
    def on_node_click(clicked_node, openvino_bin, model_xml,
                      ref_plugin, main_plugin, input_path, current_node, elements):
        # If no node was clicked, do nothing.
        if not clicked_node:
            return no_update, "Click a node", no_update

        layer_name = clicked_node['data']['layer_name']

        # Check that all required parameters are provided.
        if not all([openvino_bin, model_xml, ref_plugin, main_plugin, input_path]):
            return no_update, "Missing required parameters", no_update

        # Immediately update the node style to indicate processing (orange border).
        orange_elements = update_node_style(elements, layer_name, 'orange')

        # Now run the heavy work in the background.
        result = run_partial_inference(openvino_bin, model_xml, layer_name,
                                       ref_plugin, main_plugin, input_path)

        # After the work is done, update the node style to finished (green border).
        green_elements = update_node_style(orange_elements, layer_name, 'green')

        # Return the final outputs: current node store, inference result message, and updated graph elements.
        return layer_name, f"Partial Inference result: {result}", green_elements

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
