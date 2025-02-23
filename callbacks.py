import base64
import copy
import io
import json
import os
import time

from dash import no_update, callback_context, exceptions, html
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from run_inference import get_available_plugins
from cache import result_cache, task_queue, processing_nodes
from visualization import plot_volume_tensor
from viz_bin_diff import plot_diagnostics


# ============================
# Helper Functions
# ============================

def update_config(config: dict,
                  model_xml=None,
                  ov_bin_path=None,
                  plugin1=None,
                  plugin2=None,
                  model_inputs=None):
    if model_xml is not None:
        config['model_xml'] = model_xml
    if ov_bin_path is not None:
        config['ov_bin_path'] = ov_bin_path
    if plugin1 is not None:
        config['plugin1'] = plugin1
    if plugin2 is not None:
        config['plugin2'] = plugin2
    if model_inputs is not None:
        config['model_inputs'] = model_inputs


def update_node_style(elements, layer_name, color):
    for element in elements:
        if 'data' in element and element['data'].get('layer_name') == layer_name:
            element['data']['border_color'] = color
    return elements


def update_selection(elements, selected_id):
    updated = []
    for element in elements:
        new_elem = copy.deepcopy(element)
        if "data" in new_elem:
            new_elem["selected"] = (new_elem["data"].get("id") == selected_id)
        updated.append(new_elem)
    return updated


# ============================
# Callback Registration
# ============================

def register_callbacks(app):
    # --- Callback: Update the Selected Node Store ---
    @app.callback(
        output=[Output("selected-node-store", "data")],
        inputs=[
            Input("ir-graph", "tapNode"),
            Input({'type': 'layer-button', 'node_id': ALL, 'layer_name': ALL}, "n_clicks")
        ],
        state=[State("ir-graph", "selectedNodeData")],
        prevent_initial_call=True
    )
    def update_selected_node(tap_node, button_clicks, selected_node_data):
        ctx = callback_context
        if not ctx.triggered:
            return [no_update]

        triggered_prop = ctx.triggered[0]["prop_id"]
        if triggered_prop.startswith("ir-graph") and tap_node:
            return [tap_node["data"].get("id")]
        elif "layer-button" in triggered_prop:
            button_id = json.loads(triggered_prop.split(".")[0])
            return [button_id.get("node_id")]
        return [no_update]


    # --- Callback: Handle Graph and Button Updates ---
    @app.callback(
        output=[
            Output('right-panel', 'children'),
            Output('ir-graph', 'elements'),
            Output('layer-name', 'children'),
            Output('left-panel', 'children')
        ],
        inputs=[
            Input('ir-graph', 'tapNode'),
            Input('update-interval', 'n_intervals'),
            Input({'type': 'layer-button', 'node_id': ALL, 'layer_name': ALL}, 'n_clicks'),
            Input("selected-node-store", "data")
        ],
        state=[
            State('ir-graph', 'elements'),
            State('config-store', 'data'),
            State('ir-graph', 'selectedNodeData'),
            State('left-panel', 'children')
        ],
        prevent_initial_call=True
    )
    def handle_updates(tap_node, n_intervals, button_clicks, selected_node_store,
                       elements, config_data, selected_node_data, left_panel):
        """
        Handle various events (graph node click, interval updates, left-panel button click)
        and update the right panel, graph elements, layer name, and left panel accordingly.
        """
        ctx = callback_context
        if not ctx.triggered:
            return no_update, elements, no_update, left_panel

        new_elements = copy.deepcopy(elements)
        right_panel_out = no_update
        layer_name_out = no_update
        left_panel_out = left_panel
        selected_id = None
        triggered_prop = ctx.triggered[0]['prop_id']

        # --- Case 1: Graph Node Click ---
        if triggered_prop.startswith('ir-graph'):
            if tap_node and 'data' in tap_node:
                layer_name = tap_node['data'].get('layer_name')
                node_id = tap_node['data'].get('id')
                cached_result = result_cache.get(layer_name)
                if cached_result:
                    right_panel_out = cached_result["right-panel"]
                    layer_name_out = layer_name
                else:
                    new_elements = update_node_style(new_elements, layer_name, 'orange')
                    right_panel_out = "Processing..."
                    layer_name_out = layer_name
                    processing_nodes.add(layer_name)
                    task_queue.put((node_id, layer_name, config_data))
                selected_id = node_id

        # --- Case 2: Interval Update ---
        elif triggered_prop.startswith('update-interval'):
            finished = [node for node in processing_nodes if node in result_cache]
            if finished:
                for processed_layer in finished:
                    result = result_cache[processed_layer]
                    is_error = isinstance(result, str) and result.startswith('Error:')
                    color = 'red' if is_error else 'green'

                    # Update node style and remove from processing set
                    new_elements = update_node_style(new_elements, processed_layer, color)
                    processing_nodes.remove(processed_layer)

                    # Create new button for the left panel
                    new_button = html.Button(
                        f"{processed_layer}",
                        id={
                            'type': 'layer-button',
                            'layer_name': processed_layer,
                            'node_id': result["node_id"]
                        },
                        n_clicks=0,
                        style={'display': 'block', 'width': "100%", "textAlign": "left"}
                    )
                    left_panel_out = left_panel_out or []
                    left_panel_out.append(new_button)

                    # Update right panel if the processed layer is the selected one
                    if (selected_node_data and isinstance(selected_node_data, list) and selected_node_data):
                        selected_layer_name = selected_node_data[0].get("layer_name")
                        if processed_layer == selected_layer_name:
                            right_panel_out = result["right-panel"] if not is_error else result
                            layer_name_out = processed_layer

            # Preserve current selection if any
            current_sel = next((el["data"].get("id") for el in new_elements if el.get("selected")), None)
            selected_id = current_sel if current_sel is not None else selected_node_store

        # --- Case 3: Left-Panel Button Click ---
        elif 'layer-button' in triggered_prop:
            button_id = json.loads(triggered_prop.split('.')[0])
            node_id = button_id.get('node_id')
            layer_name = button_id.get('layer_name')
            cached_result = result_cache.get(layer_name)
            right_panel_out = cached_result["right-panel"] if cached_result else no_update
            layer_name_out = layer_name
            selected_id = node_id

        else:
            selected_id = selected_node_store

        new_elements = update_selection(new_elements, selected_id)
        return right_panel_out, new_elements, layer_name_out, left_panel_out

    # --- Callback: Toggle Configuration Modal ---
    @app.callback(
        output=Output("config-modal", "is_open"),
        inputs=[
            Input("open-modal", "n_clicks"),
            Input("close-modal", "n_clicks")
        ],
        state=[State("config-modal", "is_open")]
    )
    def toggle_modal(open_clicks, close_clicks, is_open):
        """Toggle the configuration modal when open or close buttons are clicked."""
        if open_clicks or close_clicks:
            return not is_open
        return is_open

    # --- Callback: Find Plugins & Update Dropdowns ---
    @app.callback(
        output=[
            Output("plugin-store", "data"),
            Output('reference-plugin-dropdown', 'options'),
            Output('main-plugin-dropdown', 'options'),
            Output('reference-plugin-dropdown', 'value'),
            Output('main-plugin-dropdown', 'value')
        ],
        inputs=[Input('find-plugins-button', 'n_clicks')],
        state=[State('ov-bin-path', 'value')],
        prevent_initial_call=False
    )
    def find_plugins(n_clicks, openvino_bin):
        """
        Find available plugins based on the provided OpenVINO bin path
        and update dropdown options and default values.
        """
        if not openvino_bin or not os.path.exists(openvino_bin):
            return "Invalid OpenVINO bin path", [], [], None, None
        devices = get_available_plugins(openvino_bin)
        if not devices:
            return "No plugins found.", [], [], None, None

        device_options = [{'label': d, 'value': d} for d in devices]
        ref_value = 'CPU' if 'CPU' in devices else devices[0]
        non_cpu_devices = [d for d in devices if d != 'CPU']
        other_value = non_cpu_devices[0] if non_cpu_devices else ref_value
        return "", device_options, device_options, ref_value, other_value

    # --- Callback: Save Configuration ---
    @app.callback(
        output=Output("config-store", "data"),
        inputs=[Input("close-modal", "n_clicks")],
        state=[
            State("model-xml-path", "value"),
            State("ov-bin-path", "value"),
            State("reference-plugin-dropdown", "value"),
            State("main-plugin-dropdown", "value"),
            State({"type": "model-input", "name": ALL}, "value"),
            State("config-store", "data")
        ],
        prevent_initial_call=True
    )
    def save_config(n_clicks_close, model_xml, bin_path, ref_plugin, other_plugin,
                    all_input_values, current_data):
        """
        Save the configuration when the modal is closed.

        Raises:
            PreventUpdate: If the modal close button hasn't been clicked.
        """
        if not n_clicks_close:
            raise PreventUpdate

        updated_data = current_data.copy() if current_data else {}
        updated_data.update({
            "model_xml": model_xml,
            "ov_bin_path": bin_path,
            "plugin1": ref_plugin,
            "plugin2": other_plugin,
            "model_inputs": all_input_values
        })
        return updated_data

    # --- Callback: Toggle Visualization Modal & Update Visualizations ---
    @app.callback(
        output=[
            Output("visualization-modal", "is_open"),
            Output("vis-3d", "figure"),
            Output("vis-diagnostics", "children")
        ],
        inputs=[
            Input("visualization-button", "n_clicks"),
            Input("close-vis-modal", "n_clicks")
        ],
        state=[
            State("visualization-modal", "is_open"),
            State("layer-name", "children"),
            State('config-store', 'data')
        ]
    )
    def toggle_visualization_modal(n_open, n_close, is_open, layer_name, config):
        """
        Toggle the visualization modal. If opening, generate 3D and diagnostic visualizations.
        """
        ctx = callback_context
        if not ctx.triggered:
            return is_open, no_update, no_update

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == "visualization-button" and layer_name in result_cache:
            data = result_cache.get(layer_name, {})
            ref = data.get("ref")
            main = data.get("main")
            if ref is None or main is None:
                return is_open, no_update, no_update

            # Compute difference and generate 3D volume visualization
            diff = ref - main
            start_time = time.perf_counter()
            fig_3d = plot_volume_tensor(diff)
            print(f"fig_3d time: {time.perf_counter() - start_time:.6f} seconds")

            # Generate diagnostic plot and convert to image
            start_time = time.perf_counter()
            ref_plugin_name = config["plugin1"]
            main_plugin_name = config["plugin2"]
            diag_fig = plot_diagnostics(ref, main, ref_plugin_name, main_plugin_name)
            print(f"plot_diagnostics time: {time.perf_counter() - start_time:.6f} seconds")

            buf = io.BytesIO()
            diag_fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            encoded_diag = base64.b64encode(buf.getvalue()).decode("utf-8")
            diag_img = html.Img(
                src=f"data:image/png;base64,{encoded_diag}",
                style={"width": "100%", "display": "block", "margin": "0 auto"}
            )
            return True, fig_3d, diag_img

        elif triggered_id == "close-vis-modal":
            return False, None, None

        return
