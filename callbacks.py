import copy
import os

from dash import no_update, callback_context, exceptions
from dash.dependencies import Input, Output, State, ALL

from run_inference import get_available_plugins

from cache import result_cache, task_queue, lock, processing_nodes

def update_config(config: dict, model_xml=None, ov_bin_path=None, plugin1=None, plugin2=None, model_inputs=None):
    config.update({k: v for k, v in locals().items() if k != "config" and v is not None})

def register_callbacks(app):
    @app.callback(
        Output('right-panel', 'children'),
        Output('ir-graph', 'elements'),
        Output('last-clicked-node', 'data'),
        Input('ir-graph', 'tapNode'),
        Input('update-interval', 'n_intervals'),
        State('ir-graph', 'elements'),
        State('config-store', 'data'),  # <--- now read from config-store
        State('last-clicked-node', 'data'),
        prevent_initial_call=True
    )
    def handle_node_click_and_interval(tap_node, n_intervals, elements, config_data, last_clicked_node):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'ir-graph':
            # user clicked a node
            if not tap_node:
                return "Click a node", no_update, no_update

            layer_name = tap_node['data']['layer_name']

            # Check if we already have a cached result
            with lock:
                cached_result = result_cache.get(layer_name)
            if cached_result:
                return cached_result, elements, layer_name

            # Basic validation
            if not config_data:
                return "Config not set; please configure the model first.", no_update, layer_name

            # Extract the fields you need
            openvino_bin = config_data.get("ov_bin_path", "")
            model_xml = config_data.get("model_xml", "")
            ref_plugin = config_data.get("plugin1", "")
            main_plugin = config_data.get("plugin2", "")
            model_inputs = config_data.get("model_inputs", [])  # could be a list or dict

            # Are any mandatory fields missing?
            if not all([openvino_bin, model_xml, ref_plugin, main_plugin]):
                return "Missing required parameters for partial inference", no_update, layer_name

            # Mark the node as 'processing'
            updated_elements = update_node_style(elements, layer_name, 'orange')
            with lock:
                processing_nodes.add(layer_name)

            # Enqueue the entire config data plus the layer_name
            # This way, process_tasks() can see all the inputs
            task_queue.put((layer_name, config_data))

            return "Processing...", updated_elements, layer_name
        elif triggered_id == 'update-interval':
            # Periodic check to see if any nodes are done
            with lock:
                if not processing_nodes:
                    return no_update, no_update, no_update

                # Gather all nodes that have completed inference
                finished = [ln for ln in processing_nodes if ln in result_cache]

                if not finished:
                    return no_update, no_update, no_update

                last_node_result = None

                # Update each finished node's color and remove from 'processing_nodes'
                for processed_layer_name in finished:
                    result = result_cache[processed_layer_name]
                    color = 'green'

                    if isinstance(result, str) and result.startswith('Error:'):
                        color = 'red'

                    elements = update_node_style(elements, processed_layer_name, color)
                    processing_nodes.remove(processed_layer_name)

                    # If this finished node is the "last clicked" node, keep track of its result
                    if processed_layer_name == last_clicked_node:
                        last_node_result = result

            # If the last-clicked node just finished, display its result text
            if last_node_result is not None:
                return last_node_result, elements, last_clicked_node
            else:
                # We have updated some nodes, but not the last-clicked one
                return no_update, elements, last_clicked_node

        # Fallback
        return no_update, no_update, no_update

    @app.callback(
        Output("config-modal", "is_open"),
        [Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks")],
        [State("config-modal", "is_open")],
    )
    def toggle_modal(open_clicks, close_clicks, is_open):
        if open_clicks or close_clicks:
            return not is_open
        return is_open

    def update_node_style(elements, layer_name, color):
        for element in elements:
            if 'layer_name' in element['data'] and element['data']['layer_name'] == layer_name:
                element['data']['border_color'] = color

        return elements

    # @app.callback(
    #     Output("dynamic-input-fields", "children"),
    #     Input("model-xml-path", "value"),
    #     prevent_initial_call=True
    # )
    # def update_model_inputs(model_path):
    #     """Whenever the model-path input changes, rebuild the model input fields."""
    #     if not model_path:
    #         return []
    #     return build_model_input_fields(model_path)

    @app.callback(
        Output("plugin-store", "data"),
        Output('reference-plugin-dropdown', 'options'),
        Output('main-plugin-dropdown', 'options'),
        Output('reference-plugin-dropdown', 'value'),
        Output('main-plugin-dropdown', 'value'),
        Input('find-plugins-button', 'n_clicks'),
        State('ov-bin-path', 'value'),
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
            device_options,  # main-plugin-dropdown options
            ref_value,  # reference-plugin-dropdown value
            other_value  # main-plugin-dropdown value
        )


    @app.callback(
        Output("config-store", "data"),
        Input("close-modal", "n_clicks"),  # or "save-button", whichever you prefer
        [
            State("model-xml-path", "value"),
            State("ov-bin-path", "value"),
            State("reference-plugin-dropdown", "value"),
            State("main-plugin-dropdown", "value"),
            State({"type": "model-input", "name": ALL}, "value"),  # Grab ALL dynamic inputs
            State("config-store", "data"),
        ],
        prevent_initial_call=True
    )
    def save_config(
            n_clicks_close,
            model_xml,
            bin_path,
            ref_plugin,
            other_plugin,
            all_input_values,  # A list of strings
            current_data
    ):
        # If the close button wasn't clicked, don't update anything
        if not n_clicks_close:
            raise exceptions.PreventUpdate

        updated_data = current_data.copy() if current_data else {}
        #update_config(updated_data, model_xml, bin_path, ref_plugin, other_plugin, all_input_values)

        #print(f"{updated_data=}")
        # Store "static" fields
        updated_data["model_xml"] = model_xml
        updated_data["ov_bin_path"] = bin_path
        updated_data["plugin1"] = ref_plugin
        updated_data["plugin2"] = other_plugin

        # Store the dynamic input paths as a simple list
        # (Optionally, you can also store them with their names; see below)
        updated_data["model_inputs"] = all_input_values
        return updated_data
