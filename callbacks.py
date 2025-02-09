import copy
import os

from dash import no_update, callback_context
from dash.dependencies import Input, Output, State
from run_inference import get_available_plugins

from cache import result_cache, task_queue, lock, processing_nodes


def register_callbacks(app):
    @app.callback(
        Output('right-panel', 'children'),
        Output('ir-graph', 'elements'),
        Output('last-clicked-node', 'data'),  # We'll update this when a node is clicked
        Input('ir-graph', 'tapNode'),
        Input('update-interval', 'n_intervals'),
        State('ir-graph', 'elements'),
        State('openvino-bin-input', 'value'),
        State('model-xml-input', 'value'),
        State('reference-plugin-dropdown', 'value'),
        State('other-plugin-dropdown', 'value'),
        State('input-file-input', 'value'),
        State('last-clicked-node', 'data'),  # We'll read the last clicked node state
        prevent_initial_call=True
    )
    def handle_node_click_and_interval(
            tap_node, n_intervals,
            elements,
            openvino_bin, model_xml,
            ref_plugin, main_plugin, input_path,
            last_clicked_node
    ):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'ir-graph':
            # User just clicked a node
            if not tap_node:
                # No node info
                return "Click a node", no_update, no_update

            layer_name = tap_node['data']['layer_name']

            with lock:
                cached_result = result_cache.get(layer_name)

            if cached_result:
                # We already have a result
                return f"Partial Inference result: {cached_result}", elements, layer_name

            if not all([openvino_bin, model_xml, ref_plugin, main_plugin, input_path]):
                return "Missing required parameters for inference", no_update, layer_name

            updated_elements = update_node_style(elements, layer_name, 'orange')

            with lock:
                processing_nodes.add(layer_name)

            task_queue.put((layer_name, openvino_bin, model_xml,
                            ref_plugin, main_plugin, input_path))

            # Set the last-clicked-node in the Store to this layer_name
            return "Processing...", updated_elements, layer_name

        elif triggered_id == 'update-interval':
            # Periodic check to see if any nodes are done
            have_updated_last_clicked = False
            with lock:
                if len(processing_nodes) == 0:
                    return no_update, no_update, no_update

                finished = []
                for processed_layer_name in list(processing_nodes):
                    if processed_layer_name in result_cache:
                        finished.append(processed_layer_name)

                if not finished:
                    return no_update, no_update, no_update

                # If we do get here, then the last clicked node has finished
                for processed_layer_name in finished:
                    result = result_cache[processed_layer_name]

                    color = 'green'
                    if isinstance(result, str) and result.startswith('Error:'):
                        color = 'red'

                    elements = update_node_style(elements, processed_layer_name, color)
                    processing_nodes.remove(processed_layer_name)
                    if processed_layer_name == last_clicked_node:
                        have_updated_last_clicked = True

            if have_updated_last_clicked:
                return f"Partial Inference result: {result}", elements, last_clicked_node
            else:
                return no_update, elements, last_clicked_node

        # Fallback
        return no_update, no_update, no_update



    def update_node_style(elements, layer_name, color):
        for element in elements:
            if 'layer_name' in element['data'] and element['data']['layer_name'] == layer_name:
                element['data']['border_color'] = color

        return elements

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
