import base64
import copy
import io
import os
import time

from dash import no_update, callback_context, exceptions, html
from dash.dependencies import Input, Output, State, ALL

from run_inference import get_available_plugins

from cache import result_cache, task_queue, lock, processing_nodes
from visualization import plot_volume_tensor
from viz_bin_diff import plot_diagnostics


def update_config(config: dict, model_xml=None, ov_bin_path=None, plugin1=None, plugin2=None, model_inputs=None):
    config.update({k: v for k, v in locals().items() if k != "config" and v is not None})


def register_callbacks(app):
    @app.callback(
        Output('right-panel', 'children'),
        Output('ir-graph', 'elements'),
        Output('layer-name', 'children'),
        Input('ir-graph', 'tapNode'),
        Input('update-interval', 'n_intervals'),
        State('ir-graph', 'elements'),
        State('config-store', 'data'),
        State('ir-graph', 'selectedNodeData'),
        prevent_initial_call=True
    )
    def handle_node_click_and_interval(tap_node, n_intervals, elements, config_data, selected_node_data):
        ctx = callback_context

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if triggered_id == 'ir-graph':
            layer_name = tap_node['data']['layer_name']

            with lock:
                cached_result = result_cache.get(layer_name)

            if cached_result:
                return cached_result["right-panel"], elements, layer_name

            if not config_data:
                return "Error: Config not set; please configure the model first.", no_update, no_update

            updated_elements = update_node_style(elements, layer_name, 'orange')
            with lock:
                processing_nodes.add(layer_name)

            task_queue.put((layer_name, config_data))

            return "Processing...", updated_elements, no_update
        elif triggered_id == 'update-interval':
            with lock:
                finished = [node for node in processing_nodes if node in result_cache]

                if not finished:
                    return no_update, no_update, no_update

                last_node_result = None

                for processed_layer_name in finished:
                    result = result_cache[processed_layer_name]
                    color = 'green'

                    is_error = isinstance(result, str) and result.startswith('Error:')
                    if is_error:
                        result_cache.pop(processed_layer_name)
                        color = 'red'

                    elements = update_node_style(elements, processed_layer_name, color)
                    processing_nodes.remove(processed_layer_name)

                    # If this finished node is the "last clicked" node, keep track of its
                    selected_layer_name = selected_node_data[0]["layer_name"] if len(selected_node_data) else None
                    if processed_layer_name == selected_layer_name:
                        if is_error:
                            last_node_result = result
                        else:
                            last_node_result = result["right-panel"]

            # If the last-clicked node just finished, display its result text
            if last_node_result is not None:
                return last_node_result, elements, selected_layer_name
            else:
                # We have updated some nodes, but not the last-clicked one
                return no_update, elements, no_update

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
        # update_config(updated_data, model_xml, bin_path, ref_plugin, other_plugin, all_input_values)

        # print(f"{updated_data=}")
        # Store "static" fields
        updated_data["model_xml"] = model_xml
        updated_data["ov_bin_path"] = bin_path
        updated_data["plugin1"] = ref_plugin
        updated_data["plugin2"] = other_plugin

        # Store the dynamic input paths as a simple list
        # (Optionally, you can also store them with their names; see below)
        updated_data["model_inputs"] = all_input_values
        return updated_data


    @app.callback(
        [Output("visualization-modal", "is_open"),
         Output("vis-3d", "figure"),
         Output("vis-diagnostics", "children")],
        [Input("visualization-button", "n_clicks"),
         Input("close-vis-modal", "n_clicks")],
        [State("visualization-modal", "is_open"),
         State("layer-name", "children")]
    )
    def toggle_visualization_modal(n_open, n_close, is_open, layer_name):
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

            # Generate the visualizations.
            diff = ref - main
            start_time = time.perf_counter()
            fig_3d = plot_volume_tensor(diff)
            print(f"fig_3d time: {time.perf_counter() - start_time:.6f} seconds")
            start_time = time.perf_counter()
            diag_fig = plot_diagnostics(ref, main)
            print(f"plot_diagnostics time: {time.perf_counter() - start_time:.6f} seconds")
            start_time = time.perf_counter()

            # Convert the matplotlib diagnostics figure to a PNG image.
            buf = io.BytesIO()
            diag_fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            encoded_diag = base64.b64encode(buf.getvalue()).decode("utf-8")
            diag_img = html.Img(
                src=f"data:image/png;base64,{encoded_diag}",
                style={"width": "100%", "display": "block", "margin": "0 auto"}
            )
            print(f"b64encode time: {time.perf_counter() - start_time:.6f} seconds")
            return True, fig_3d, diag_img

        elif triggered_id == "close-vis-modal":
            return False, no_update, no_update

        return is_open, no_update, no_update

    @app.callback(
        Output("tab-3d-content", "style"),
        Output("tab-diag-content", "style"),
        Input("vis-tabs", "value")
    )
    def toggle_tab_contents(active_tab):
        """
        Show the chosen tab, hide the other one
        dcc.Tabs unloads tabs from the DOM on changing tabs
        Rebuilding 3D graph is slow and doing it every time is unacceptable
        """
        if active_tab == "tab-3d":
            return {"display": "block"}, {"display": "none"}
        else:
            return {"display": "none"}, {"display": "block"}

