import base64
import copy
import io
import json
import os
import time
import bisect

from dash import no_update, callback_context, exceptions, html
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

from run_inference import get_available_plugins

from cache import result_cache, task_queue, processing_nodes, lock
from visualization import plot_volume_tensor
from viz_bin_diff import plot_diagnostics


class LockGuard:
    def __init__(self, lock):
        self._lock = lock
        self._lock.acquire()

    def __del__(self):
        # Called when the object is garbage-collected
        self._lock.release()


def update_config(config: dict, model_xml=None, ov_bin_path=None, plugin1=None, plugin2=None, model_inputs=None):
    config.update({k: v for k, v in locals().items() if k != "config" and v is not None})


def update_node_style(elements, node_id, color):
    for element in elements:
        if element['data'].get('id') == node_id:
            element['data']['border_color'] = color

    return elements


def set_selected_node_style(elements, node_id):
    for element in elements:
        if element["data"].get("id") == node_id:
            element["classes"] = "selected"
        else:
            element["classes"] = ""


def update_selection(elements, selected_id):
    updated = []
    for element in elements:
        new_elem = copy.deepcopy(element)
        if "data" in new_elem:
            new_elem["selected"] = (new_elem["data"].get("id") == selected_id)
        updated.append(new_elem)
    return updated


def register_callbacks(app):
    @app.callback(
        Output('just-finished-tasks-store', 'data'),
        Input('update-interval', 'n_intervals'),
        prevent_initial_call=True
    )
    def collect_finished_tasks(n_intervals):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        with lock:
            finished_nodes = [node for node in processing_nodes if node in result_cache]
            if not finished_nodes:
                return no_update

            for node_id in finished_nodes:
                processing_nodes.remove(node_id)

            return finished_nodes

    @app.callback(
        Output('selected-node-id-store', 'data'),
        Input('ir-graph', 'tapNode'),
        Input('selected-layer-index-store', 'data'),
        State('layer-store', 'data'),
        prevent_initial_call=True
    )
    def update_selected_node_id(tap_node, selected_layer_index, layers_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        selected_node_id = no_update

        if any(trigger.startswith('ir-graph') for trigger in triggers):
            selected_node_id = tap_node['data'].get('id')

        if any(trigger.startswith('selected-layer-index-store') for trigger in triggers):
            selected_layer = layers_list[selected_layer_index]
            selected_node_id = selected_layer["node_id"]

        return selected_node_id

    @app.callback(
        Output('ir-graph', 'elements'),
        Input('first-load', 'pathname'),
        Input('ir-graph', 'tapNode'),
        Input('just-finished-tasks-store', 'data'),
        Input('selected-layer-index-store', 'data'),
        State('ir-graph', 'elements'),
        State('config-store', 'data'),
        State('layer-store', 'data'),
        prevent_initial_call=True
    )
    def update_graph_elements(_, tap_node, finished_nodes, selected_layer_index, elements, config_data, layers_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        new_elements = copy.deepcopy(elements)

        if any(trigger.startswith('first-load') for trigger in triggers):
            with lock:
                for element in new_elements:
                    node_id = element['data'].get("id")

                    if node_id in result_cache:
                        element['data']['border_color'] = 'green'

                    if node_id in processing_nodes:
                        element['data']['border_color'] = 'yellow'

        if any(trigger.startswith('ir-graph') for trigger in triggers):
            with lock:
                node_id = tap_node['data'].get('id')

                if node_id not in result_cache:
                    layer_name = tap_node['data'].get('layer_name')
                    layer_type = tap_node['data'].get('type')
                    processing_nodes.add(node_id)
                    task_queue.put((node_id, layer_name, layer_type, config_data))
                    update_node_style(new_elements, node_id, 'orange')

                set_selected_node_style(new_elements, node_id)

        if any(trigger.startswith('just-finished-tasks-store') for trigger in triggers):
            with lock:
                # TODO do a proper error handling
                for element in new_elements:
                    if element['data'].get('id') in finished_nodes:
                        element['data']['border_color'] = 'green'

        if any(trigger.startswith('selected-layer-index-store') for trigger in triggers):
            selected_layer = layers_list[selected_layer_index]
            node_id = selected_layer["node_id"]

            set_selected_node_style(new_elements, node_id)

        return new_elements

    @app.callback(
        Output('layer-store', 'data'),
        Input('first-load', 'pathname'),
        Input('just-finished-tasks-store', 'data'),
        State('layer-store', 'data'),
        prevent_initial_call=True
    )
    def update_inferred_layers_list(_, finished_nodes, current_layers_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        layer_list_out = no_update

        if any(trigger.startswith('first-load') for trigger in triggers):
            with lock:
                layer_list_out = []
                for node_id, result in sorted(result_cache.items(), key=lambda item: int(item[1]["node_id"])):
                    layer_list_out.append({
                        "node_id": node_id,
                        "layer_name": result["layer_name"],
                        "layer_type": result["layer_type"]
                    })

        if any(trigger.startswith('just-finished-tasks-store') for trigger in triggers) and finished_nodes:
            with lock:
                layer_list_out = current_layers_list
                for node_id in finished_nodes:
                    result = result_cache[node_id]

                    list_of_ids = [int(item["node_id"]) for item in layer_list_out]
                    insertion_index = bisect.bisect_left(list_of_ids, int(node_id))
                    layer_list_out.insert(insertion_index, {
                        "node_id": node_id,
                        "layer_name": result['layer_name'],
                        "layer_type": result['layer_type']
                    })

        return layer_list_out

    @app.callback(
        Output('layer-list', 'children'),
        Input('layer-store', 'data'),
        Input('selected-layer-index-store', 'data'),
        prevent_initial_call=True
    )
    def render_layers(layers_list, selected_index):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        li_elements = []

        for i, layer in enumerate(layers_list):
            style = {'padding': '5px', 'marginBottom': '3px'}
            if i == selected_index:
                style.update({
                    'border': '1px solid black',
                    'background-color': 'darkgray',
                })

            li_elements.append(
                html.Li(
                    html.Div(
                        [
                            html.Span(layer['layer_type']),
                            html.Span(layer['layer_name'])
                        ],
                        style={'display': 'flex', 'justify-content': 'space-between'}
                    ),
                    id={'type': 'layer-li', 'index': i},
                    n_clicks=0,
                    style=style
                )
            )

        return li_elements

    @app.callback(
        Output('selected-layer-index-store', 'data'),
        Input("keyboard", "n_keydowns"),
        Input({'type': 'layer-li', 'index': ALL}, 'n_clicks'),
        Input('ir-graph', 'tapNode'),
        State("keyboard", "keydown"),
        State('selected-layer-index-store', 'data'),
        State('layer-store', 'data'),
        prevent_initial_call=True
    )
    def handle_keys_and_clicks(n_keydowns, li_n_clicks, tap_node, keydown, selected_layer_index, layers_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        new_index = no_update

        if any(trigger.startswith('ir-graph') for trigger in triggers):
            node_id = tap_node['data'].get('id')
            index = next((i for i, layer in enumerate(layers_list) if layer['node_id'] == node_id), None)

            if index is not None:
                new_index = index

        if any(trigger.startswith('keyboard') for trigger in triggers):
            num_layers = len(layers_list)
            if num_layers > 0:
                pressed_key = keydown.get('key')
                PAGE_STEP = 5

                if pressed_key == "ArrowUp":
                    new_index = max(0, selected_layer_index - 1)
                elif pressed_key == "ArrowDown":
                    new_index = min(num_layers - 1, selected_layer_index + 1)
                elif pressed_key == "Home":
                    new_index = 0
                elif pressed_key == "End":
                    new_index = num_layers - 1
                elif pressed_key == "PageUp":
                    new_index = max(0, selected_layer_index - PAGE_STEP)
                elif pressed_key == "PageDown":
                    new_index = min(num_layers - 1, selected_layer_index + PAGE_STEP)

        if any(trigger.endswith('.n_clicks') for trigger in triggers):
            first_trigger = next((trigger for trigger in triggers if trigger.endswith('.n_clicks')))

            # Avoid fantom click events when adding new elements to the layers list
            if not all(nc == 0 for nc in li_n_clicks):
                try:
                    id_part = first_trigger.split('.')[0]
                    triggered_id = json.loads(id_part)
                    new_index = triggered_id.get("index")
                except Exception:
                    pass

        return new_index

    # @app.callback(
    #     Output('right-panel-layer-name', 'children'),
    #     Output('right-panel', 'children'),
    #     Input('selected-node-id-store', 'data'),
    #     Input('selected-layer-index-store', 'data'),
    #     State('layer-store', 'data'),
    #     prevent_initial_call=True
    # )
    # def update_stats(selected_node_id, selected_layer_index, layers_list):
    #     ctx = callback_context
    #     if not ctx.triggered:
    #         return no_update, no_update
    #
    #     triggered_prop = ctx.triggered[0]['prop_id']
    #
    #     if triggered_prop.startswith('selected-node-id-store'):
    #         node_id = selected_node_id
    #         layer_name = "" # tap_node['data'].get('layer_name')
    #
    #         cached_result = result_cache.get(node_id)
    #         if cached_result:
    #             return cached_result["layer_name"], cached_result["right-panel"]
    #         else:
    #             return layer_name, "Processing..."
    #
    #     elif triggered_prop.startswith('selected-layer-index-store'):
    #         selected_layer = layers_list[selected_layer_index]
    #         node_id = selected_layer["node_id"]
    #         layer_name = selected_layer["layer_name"]
    #
    #         cached_result = result_cache.get(node_id)
    #         if cached_result:
    #             return layer_name, cached_result["right-panel"]
    #         else:
    #             return layer_name, "Processing..."
    #
    #     return no_update, no_update

    # @app.callback(
    #     Output('right-panel', 'children'),
    #     Output('ir-graph', 'elements'),
    #     Output('layer-name', 'children'),
    #     Output('layer-store', 'data'),
    #     Input('ir-graph', 'tapNode'),
    #     Input('update-interval', 'n_intervals'),
    #     Input("selected-node-id-store", "data"),
    #     Input('selected-layer-index-store', 'data'),
    #     State('ir-graph', 'elements'),
    #     State('config-store', 'data'),
    #     State('layer-store', 'data'),
    #     prevent_initial_call=True
    # )
    # def handle_updates(tap_node, n_intervals, selected_node_store,
    #                    selected_layer_index, elements, config_data, current_layer_list):
    #     ctx = callback_context
    #     if not ctx.triggered:
    #         return no_update, elements, no_update, no_update
    # 
    #     new_elements = copy.deepcopy(elements)
    #     right_panel_out = no_update
    #     layer_name_out = no_update
    #     layer_list_out = no_update
    #     selected_id = None
    # 
    #     triggered_prop = ctx.triggered[0]['prop_id']
    # 
    #     # Case 1: User clicked a node in the graph.
    #     if triggered_prop.startswith('ir-graph'):
    #         if tap_node and 'data' in tap_node:
    #             layer_name = tap_node['data'].get('layer_name')
    #             layer_type = tap_node['data'].get('type')
    #             node_id = tap_node['data'].get('id')
    #             cached_result = result_cache.get(node_id)
    #             if cached_result:
    #                 right_panel_out = cached_result["right-panel"]
    #                 layer_name_out = layer_name
    #             else:
    #                 new_elements = update_node_style(new_elements, node_id, 'orange')
    #                 right_panel_out = "Processing..."
    #                 layer_name_out = layer_name
    #                 processing_nodes.add(node_id)
    #                 task_queue.put((node_id, layer_name, layer_type, config_data))
    #             selected_id = node_id
    # 
    #     # Case 2: Update interval triggered.
    #     elif triggered_prop.startswith('update-interval'):
    #         finished = [node for node in processing_nodes if node in result_cache]
    #         if finished:
    #             # Update one finished node per interval update
    #             node_id = finished[0]
    #             result = result_cache[node_id]
    #             layer_name = result['layer_name']
    #             layer_type = result['layer_type']
    #             color = 'green'
    #             is_error = isinstance(result, str) and result.startswith('Error:')
    #             if is_error:
    #                 result_cache.pop(node_id)
    #                 color = 'red'
    #             new_elements = update_node_style(new_elements, node_id, color)
    #             processing_nodes.remove(node_id)
    # 
    #             layer_list_out = copy.deepcopy(current_layer_list) if current_layer_list is not None else []
    #             insertion_index = bisect.bisect_left(
    #                 [int(item["node_id"]) for item in layer_list_out],
    #                 int(node_id)
    #             )
    #             layer_list_out.insert(insertion_index, {
    #                 "node_id": node_id,
    #                 "layer_name": layer_name,
    #                 "layer_type": layer_type
    #             })
    # 
    #             # Always update the right panel and layer name when processing finishes.
    #             right_panel_out = result["right-panel"] if not is_error else result
    #             layer_name_out = layer_name
    # 
    #             # Preserve current selection from new_elements.
    #             current_sel = None
    #             for el in new_elements:
    #                 if "selected" in el and el["selected"]:
    #                     current_sel = el["data"].get("id")
    #                     break
    #             selected_id = current_sel if current_sel is not None else selected_node_store
    # 
    #     # Case 3: Selected layer index changed via keyboard arrow keys.
    #     elif triggered_prop == 'selected-layer-index-store.data':
    #         layer_list_copy = copy.deepcopy(current_layer_list)
    #         if layer_list_copy:
    #             if 0 <= selected_layer_index < len(layer_list_copy):
    #                 selected_layer = layer_list_copy[selected_layer_index]
    #                 node_id = selected_layer.get("node_id")
    #                 layer_name = selected_layer.get("layer_name")
    #                 cached_result = result_cache.get(node_id)
    #                 right_panel_out = cached_result["right-panel"] if cached_result else no_update
    #                 layer_name_out = layer_name
    #                 selected_id = node_id
    # 
    #     else:
    #         return no_update, no_update, no_update, no_update
    # 
    #     # Update selection in the graph elements so that only the selected node is marked.
    #     if selected_id is not None:
    #         new_elements = update_selection(new_elements, selected_id)
    # 
    #     return right_panel_out, new_elements, layer_name_out, layer_list_out
    # 
    # @app.callback(
    #     Output('layer-list', 'children'),
    #     Input('layer-store', 'data'),
    #     Input('selected-layer-index-store', 'data')
    # )
    # def render_layers(layers, selected_index):
    #     if not layers:
    #         return []
    # 
    #     li_elements = []
    #     for i, layer in enumerate(layers):
    #         print(f"{layer=}")
    #         is_selected = (i == selected_index)
    #         style = {'padding': '5px', 'marginBottom': '3px'}
    #         if is_selected:
    #             style.update({
    #                 'fontWeight': 'bold',
    #                 'backgroundColor': '#D3D3D3',
    #                 'border': '1px solid black'
    #             })
    # 
    #         li_elements.append(
    #             html.Li(
    #                 html.Div(
    #                     [
    #                         html.Span(layer['layer_type']),
    #                         html.Span(layer['layer_name'])
    #                     ],
    #                     style={'display': 'flex', 'justify-content': 'space-between'}
    #                 ),
    #                 id={'type': 'layer-li', 'index': i},
    #                 n_clicks=0,
    #                 style=style
    #             )
    #         )
    #     return li_elements
    # 
    # @app.callback(
    #     Output('selected-layer-index-store', 'data'),
    #     Input("keyboard", "n_keydowns"),
    #     Input({'type': 'layer-li', 'index': ALL}, 'n_clicks'),
    #     State("keyboard", "keydown"),
    #     State('selected-layer-index-store', 'data'),
    #     State('layer-store', 'data'),
    #     prevent_initial_call=True
    # )
    # def handle_keys_and_clicks(n_keydowns, li_n_clicks, keydown, current_index, layers):
    #     ctx = callback_context
    #     if not ctx.triggered:
    #         return no_update
    # 
    #     triggered_prop_id = ctx.triggered[0]['prop_id']
    #     if triggered_prop_id.startswith('{'):
    #         try:
    #             id_part = triggered_prop_id.split('.')[0]
    #             triggered_id = json.loads(id_part)
    #         except Exception:
    #             triggered_id = {}
    #     else:
    #         triggered_id = {}
    # 
    #     # If a list item was clicked, update the selection.
    #     if triggered_id.get("type") == "layer-li":
    #         return triggered_id.get("index")
    # 
    #     # For keyboard events, ensure keydown is provided.
    #     if not keydown:
    #         return no_update
    # 
    #     pressed_key = keydown.get('key')
    #     if pressed_key not in ("ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"):
    #         return no_update
    # 
    #     PAGE_STEP = 10  # Adjust this value as needed
    # 
    #     if pressed_key == "ArrowUp":
    #         new_index = max(0, current_index - 1)
    #     elif pressed_key == "ArrowDown":
    #         new_index = min(len(layers) - 1, current_index + 1)
    #     elif pressed_key == "Home":
    #         new_index = 0
    #     elif pressed_key == "End":
    #         new_index = len(layers) - 1
    #     elif pressed_key == "PageUp":
    #         new_index = max(0, current_index - PAGE_STEP)
    #     elif pressed_key == "PageDown":
    #         new_index = min(len(layers) - 1, current_index + PAGE_STEP)
    #     else:
    #         return no_update
    # 
    #     return new_index
    # 
    # @app.callback(
    #     Output("selected-node-id-store", "data"),
    #     Input("ir-graph", "tapNode"),
    #     prevent_initial_call=True
    # )
    # def update_selected_node(tap_node):
    #     ctx = callback_context
    #     if not ctx.triggered:
    #         return no_update
    #     triggered_prop = ctx.triggered[0]["prop_id"]
    #     if triggered_prop.startswith("ir-graph") and tap_node:
    #         return tap_node["data"].get("id")
    #     return no_update

    #######################################################################################################################

    @app.callback(
        Output("config-modal", "is_open"),
        Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks"),
        State("config-modal", "is_open"),
    )
    def toggle_modal(open_clicks, close_clicks, is_open):
        if open_clicks or close_clicks:
            return not is_open
        return is_open

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
            "",
            device_options,
            device_options,
            ref_value,
            other_value
        )

    @app.callback(
        Output("config-store", "data"),
        Input("close-modal", "n_clicks"),
        State("model-xml-path", "value"),
        State("ov-bin-path", "value"),
        State("reference-plugin-dropdown", "value"),
        State("main-plugin-dropdown", "value"),
        State({"type": "model-input", "name": ALL}, "value"),
        State("config-store", "data"),
        prevent_initial_call=True
    )
    def save_config(n_clicks_close, model_xml, bin_path, ref_plugin,
                    other_plugin, all_input_values, current_data):
        if not n_clicks_close:
            raise exceptions.PreventUpdate

        updated_data = current_data.copy() if current_data else {}
        updated_data["model_xml"] = model_xml
        updated_data["ov_bin_path"] = bin_path
        updated_data["plugin1"] = ref_plugin
        updated_data["plugin2"] = other_plugin
        updated_data["model_inputs"] = all_input_values
        return updated_data

    @app.callback(
        Output("visualization-modal", "is_open"),
        Output("vis-3d", "figure"),
        Output("vis-diagnostics", "children"),
        Input("visualization-button", "n_clicks"),
        Input("close-vis-modal", "n_clicks"),
        State("visualization-modal", "is_open"),
        State("selected-node-id-store", "data"),
        State('config-store', 'data')
    )
    def toggle_visualization_modal(n_open, n_close, is_open, node_id, config):
        ctx = callback_context
        if not ctx.triggered:
            return is_open, no_update, no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == "visualization-button" and node_id in result_cache:
            data = result_cache.get(node_id, {})
            ref = data.get("ref")
            main = data.get("main")
            if ref is None or main is None:
                return is_open, no_update, no_update

            diff = ref - main
            start_time = time.perf_counter()
            fig_3d = plot_volume_tensor(diff)
            print(f"fig_3d time: {time.perf_counter() - start_time:.6f} seconds")
            start_time = time.perf_counter()

            ref_plugin_name = config["plugin1"]
            main_plugin_name = config["plugin2"]
            diag_fig = plot_diagnostics(ref, main, ref_plugin_name, main_plugin_name)
            print(f"plot_diagnostics time: {time.perf_counter() - start_time:.6f} seconds")
            start_time = time.perf_counter()

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
            return False, None, None

        return is_open, no_update, no_update

    @app.callback(
        Output("tab-3d-content", "style"),
        Output("tab-diag-content", "style"),
        Input("vis-tabs", "value")
    )
    def toggle_tab_contents(active_tab):
        if active_tab == "tab-3d":
            return {"display": "block"}, {"display": "none"}
        else:
            return {"display": "none"}, {"display": "block"}
