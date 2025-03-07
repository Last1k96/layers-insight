import base64
import copy
import io
import json
import os
import bisect

from dash import no_update, callback_context, exceptions, html, dcc
from dash.dependencies import Input, Output, State, ALL
from run_inference import get_available_plugins

from cache import result_cache, task_queue, processing_layers, lock
from visualizations.new_cool_visualizations import animated_slices, isosurface_diff, parallel_coordinates_diff, \
    tensor_unfolding_diff, probabilistic_diff, interactive_tensor_diff_dashboard, \
    hierarchical_diff_visualization, tensor_network_visualization, channel_correlation_matrices, \
    gradient_flow_visualization, tensor_histogram_comparison, spectral_analysis, eigenvalue_comparison
from visualizations.visualization import plot_volume_tensor
from visualizations.viz_bin_diff import plot_diagnostics, reshape_to_3d


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
            finished_nodes = [node for node in processing_layers if node in result_cache]
            if not finished_nodes:
                return no_update

            for node_id in finished_nodes:
                processing_layers.pop(node_id)

            return finished_nodes

    @app.callback(
        Output('selected-node-id-store', 'data'),
        Output('selected-layer-name-store', 'data'),
        Input('ir-graph', 'tapNode'),
        Input('selected-layer-index-store', 'data'),
        State('layers-store', 'data'),
        prevent_initial_call=True
    )
    def update_selected_node_id(tap_node, selected_layer_index, layers_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        selected_node_id = no_update
        selected_layer_name = no_update

        if any(trigger.startswith('ir-graph') for trigger in triggers):
            selected_node_id = tap_node['data'].get('id')
            selected_layer_name = tap_node['data'].get('layer_name')

        if any(trigger.startswith('selected-layer-index-store') for trigger in triggers):
            selected_layer = layers_list[selected_layer_index]
            selected_node_id = selected_layer["node_id"]
            selected_layer_name = selected_layer["layer_name"]

        return selected_node_id, selected_layer_name

    @app.callback(
        Output('ir-graph', 'elements'),
        Input('first-load', 'pathname'),
        Input('ir-graph', 'tapNode'),
        Input('just-finished-tasks-store', 'data'),
        Input('selected-layer-index-store', 'data'),
        State('ir-graph', 'elements'),
        State('config-store', 'data'),
        State('layers-store', 'data'),
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

                    if node_id in processing_layers:
                        element['data']['border_color'] = 'yellow'

        if any(trigger.startswith('ir-graph') for trigger in triggers):
            with lock:
                node_id = tap_node['data'].get('id')

                if node_id not in result_cache:
                    layer_name = tap_node['data'].get('layer_name')
                    layer_type = tap_node['data'].get('type')

                    processing_layers[node_id] = {
                        "layer_name": layer_name,
                        "layer_type": layer_type
                    }

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
        Output('layers-store', 'data'),
        Output('clicked-graph-node-id-store', 'data'),
        Input('first-load', 'pathname'),
        Input('ir-graph', 'tapNode'),
        Input('just-finished-tasks-store', 'data'),
        State('layers-store', 'data'),
        prevent_initial_call=True
    )
    def update_layers_list(_, tap_node, finished_nodes, layers_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        layer_list_out = no_update
        clicked_graph_node_id = no_update

        if any(trigger.startswith('first-load') for trigger in triggers):
            with lock:
                layer_list_out = []

                for node_id, result in result_cache.items():
                    layer_list_out.append({
                        "node_id": node_id,
                        "layer_name": result["layer_name"],
                        "layer_type": result["layer_type"],
                        "done": True
                    })

                for node_id, result in processing_layers.items():
                    layer_list_out.append({
                        "node_id": node_id,
                        "layer_name": result["layer_name"],
                        "layer_type": result["layer_type"],
                        "done": False
                    })

                layer_list_out = sorted(layer_list_out, key=lambda item: int(item["node_id"]))

        if any(trigger.startswith('ir-graph') for trigger in triggers):
            with lock:
                layer_list_out = layers_list

                node_id = tap_node['data'].get('id')
                clicked_graph_node_id = node_id  # To trigger layer selection after new layer was added to the list
                if node_id not in result_cache:
                    layer_name = tap_node['data'].get('layer_name')
                    layer_type = tap_node['data'].get('type')

                    list_of_ids = [int(item["node_id"]) for item in layer_list_out]
                    insertion_index = bisect.bisect_left(list_of_ids, int(node_id))
                    layer_list_out.insert(insertion_index, {
                        "node_id": node_id,
                        "layer_name": layer_name,
                        "layer_type": layer_type,
                        "done": False
                    })

        if any(trigger.startswith('just-finished-tasks-store') for trigger in triggers) and finished_nodes:
            layer_list_out = layers_list

            for layer in layer_list_out:
                if layer["node_id"] in finished_nodes:
                    layer["done"] = True

        return layer_list_out, clicked_graph_node_id

    @app.callback(
        Output('layer-panel-list', 'children'),
        Input('selected-layer-index-store', 'data'),
        Input('layers-store', 'data'),
        State('layer-panel-list', 'children'),
        prevent_initial_call=True
    )
    def render_layers(selected_index, layers_list, rendered_layers):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        layer_index = 0
        if rendered_layers:
            for index, layer in enumerate(rendered_layers):
                background_color = layer["props"]["style"].get("backgroundColor", None)
                if background_color:
                    layer_index = index
                    break

        if any(trigger.startswith('selected-layer-index-store') for trigger in triggers):
            layer_index = selected_index

        li_elements = []

        for i, layer in enumerate(layers_list):
            color = '#4CAF50' if layer["done"] else '#BA8E23'
            style = {
                'color': color,
                'padding': '4px',
                'marginBottom': '2px',
            }
            if i == layer_index:
                style.update({
                    'backgroundColor': '#292E37',
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
        Input('clicked-graph-node-id-store', 'data'),
        State('layers-store', 'data'),
        State("keyboard", "keydown"),
        State('selected-layer-index-store', 'data'),
        prevent_initial_call=True
    )
    def update_selected_layer(n_keydowns, li_n_clicks, clicked_graph_node_id, layers_list, keydown,
                              selected_layer_index):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        new_index = no_update

        if any(trigger.startswith('clicked-graph-node-id-store') for trigger in triggers):
            for index, element in enumerate(layers_list):
                if element["node_id"] == clicked_graph_node_id:
                    new_index = index
                    break

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

    @app.callback(
        Output('right-panel-layer-name', 'children'),
        Output('right-panel', 'children'),
        Output('visualization-button', 'style'),
        Input('selected-node-id-store', 'data'),
        Input('selected-layer-index-store', 'data'),
        Input('just-finished-tasks-store', 'data'),
        State('layers-store', 'data'),
        State('selected-layer-name-store', 'data'),
        prevent_initial_call=True
    )
    def update_stats(selected_node_id, selected_layer_index, finished_nodes, layers_list, selected_layer_name):
        ctx = callback_context
        # If there is no trigger, return no update and hide the button.
        if not ctx.triggered:
            return no_update, no_update, {"display": "none"}

        triggers = [t['prop_id'] for t in ctx.triggered]
        node_id = None

        if any(trigger.startswith('selected-node-id-store') for trigger in triggers):
            node_id = selected_node_id

        if any(trigger.startswith('selected-layer-index-store') for trigger in triggers):
            selected_layer = layers_list[selected_layer_index]
            node_id = selected_layer["node_id"]

        if any(trigger.startswith('just-finished-tasks-store') for trigger in triggers) and finished_nodes:
            if selected_node_id in finished_nodes:
                node_id = selected_node_id

        if node_id is None:
            return no_update, no_update, {"display": "none"}

        cached_result = result_cache.get(node_id)
        if cached_result:
            # Show the button when a cached result exists.
            button_style = {"display": "block"}
            return selected_layer_name, cached_result["right-panel"], button_style
        else:
            # Hide the button while processing.
            button_style = {"display": "none"}
            return selected_layer_name, "Processing...", button_style

    #######################################################################################################################

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
        Input("config-modal", "is_open"),
        State("model-xml-path", "value"),
        State("ov-bin-path", "value"),
        State("reference-plugin-dropdown", "value"),
        State("main-plugin-dropdown", "value"),
        State({"type": "model-input", "name": ALL}, "value"),
        State("config-store", "data"),
        prevent_initial_call=True
    )
    def save_config(is_open, model_xml, bin_path, ref_plugin,
                    other_plugin, all_input_values, current_data):
        if is_open:
            return no_update

        updated_data = current_data.copy() if current_data else {}
        updated_data["model_xml"] = model_xml
        updated_data["ov_bin_path"] = bin_path
        updated_data["plugin1"] = ref_plugin
        updated_data["plugin2"] = other_plugin
        updated_data["model_inputs"] = all_input_values
        return updated_data

    @app.callback(
        Output("visualization-container", "children"),
        Output("last-selected-visualization", "data"),
        Output("store-figure", "data"),
        Input("visualization-modal", "is_open"),
        Input({"type": "visualization-btn", "index": ALL}, "n_clicks"),
        State("store-figure", "data"),
        State("last-selected-visualization", "data"),
        State("config-store", "data"),
        State("selected-node-id-store", "data"),
    )
    def select_visualization_type(is_open, btn_clicks, store_figure, last_selected_visualization, config, node_id):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Clear cache if the new node was clicked
        if store_figure and store_figure.get("node_id", None) != node_id:
            store_figure.clear()
            store_figure["node_id"] = node_id

        # Handle the modal case first
        default_viz_name = "viz1"
        if triggered_id == "visualization-modal":
            if is_open:
                selected_visualization = last_selected_visualization if last_selected_visualization is not None else default_viz_name
            else:
                return html.Div(), last_selected_visualization, no_update

        else:
            button_id = json.loads(triggered_id)
            selected_visualization = button_id["index"]

        data = result_cache.get(node_id, {})
        ref = data.get("ref")
        main = data.get("main")

        # Handle each visualization type
        if selected_visualization == "viz1":
            if "viz1" in store_figure:
                figure = store_figure["viz1"]
            else:
                diff = main - ref
                figure = plot_volume_tensor(diff)
                store_figure["viz1"] = figure

            return dcc.Graph(id="vis-graph", figure=figure,
                             style={'width': '100%',
                                    'height': 'calc(100vh - 150px)'}), selected_visualization, store_figure

        elif selected_visualization == "viz2":
            if "viz2" in store_figure:
                img = store_figure["viz2"]
            else:
                ref_plugin_name = config["plugin1"]
                main_plugin_name = config["plugin2"]
                figure = plot_diagnostics(ref, main, ref_plugin_name, main_plugin_name)
                buf = io.BytesIO()
                figure.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                encoded_diag = base64.b64encode(buf.getvalue()).decode("utf-8")
                img = html.Img(
                    src=f"data:image/png;base64,{encoded_diag}",
                    style={"width": "100%", "display": "block", "margin": "0 auto"}
                )
                store_figure["viz2"] = img

            return img, selected_visualization, store_figure

        elif selected_visualization == "viz3":
            if "viz3" in store_figure:
                animation_html = store_figure["viz3"]
            else:
                animation_html = animated_slices(ref, main, axis=0, fps=2)
                store_figure["viz3"] = animation_html

            return html.Div(
                html.Iframe(
                    srcDoc=animation_html,
                    style={
                        "width": "70%",
                        "height": "calc(100vh - 150px)",
                        "border": "none",
                        "minHeight": "600px"
                    }
                ),
                style={
                    "width": "100%",
                    "height": "calc(100vh - 150px)",  # Added height for vertical centering
                    "display": "flex",
                    "justifyContent": "center",  # Center horizontally
                    "alignItems": "center"  # Center vertically
                }
            ), selected_visualization, store_figure

        elif selected_visualization == "viz4":
            if "viz4" in store_figure:
                figure = store_figure["viz4"]
            else:
                ref = reshape_to_3d(ref)
                main = reshape_to_3d(main)
                figure = isosurface_diff(ref, main)
                store_figure["viz4"] = figure

            return dcc.Graph(id="vis-graph", figure=figure,
                             style={'width': '100%',
                                    'height': 'calc(100vh - 150px)'}), selected_visualization, store_figure

        elif selected_visualization == "viz8":
            if "viz8" in store_figure:
                figure = store_figure["viz8"]
            else:
                ref = reshape_to_3d(ref)
                main = reshape_to_3d(main)
                figure = interactive_tensor_diff_dashboard(ref, main)
                store_figure["viz8"] = figure

            return dcc.Graph(id="vis-graph", figure=figure,
                             style={'width': '100%',
                                    'height': 'calc(100vh - 150px)'}), selected_visualization, store_figure

        elif selected_visualization == "viz9":
            if "viz9" in store_figure:
                figure = store_figure["viz9"]
            else:
                ref = reshape_to_3d(ref)
                main = reshape_to_3d(main)
                figure = hierarchical_diff_visualization(ref, main)
                store_figure["viz9"] = figure

            return dcc.Graph(id="vis-graph", figure=figure,
                             style={'width': '100%',
                                    'height': 'calc(100vh - 150px)'}), selected_visualization, store_figure

        elif selected_visualization == "viz10":
            if "viz10" in store_figure:
                figure = store_figure["viz10"]
            else:
                ref = reshape_to_3d(ref)
                main = reshape_to_3d(main)
                figure = tensor_network_visualization(ref, main)
                store_figure["viz10"] = figure

            return dcc.Graph(id="vis-graph", figure=figure,
                             style={'width': '100%',
                                    'height': 'calc(100vh - 150px)'}), selected_visualization, store_figure

        elif selected_visualization == "viz12":
            if "viz12" in store_figure:
                figure = store_figure["viz12"]
            else:
                ref = reshape_to_3d(ref)
                main = reshape_to_3d(main)
                figure = channel_correlation_matrices(ref, main)
                store_figure["viz12"] = figure

            return dcc.Graph(id="vis-graph", figure=figure,
                             style={'width': '100%',
                                    'height': 'calc(100vh - 150px)'}), selected_visualization, store_figure

        elif selected_visualization == "viz13":
            if "viz13" in store_figure:
                figure = store_figure["viz13"]
            else:
                ref = reshape_to_3d(ref)
                main = reshape_to_3d(main)
                figure = gradient_flow_visualization(ref, main)
                store_figure["viz13"] = figure

            return dcc.Graph(id="vis-graph", figure=figure,
                             style={'width': '100%',
                                    'height': 'calc(100vh - 150px)'}), selected_visualization, store_figure

        elif selected_visualization == "viz14":
            if "viz14" in store_figure:
                figure = store_figure["viz14"]
            else:
                ref = reshape_to_3d(ref)
                main = reshape_to_3d(main)
                figure = tensor_histogram_comparison(ref, main)
                store_figure["viz14"] = figure

            return dcc.Graph(id="vis-graph", figure=figure,
                             style={'width': '100%',
                                    'height': 'calc(100vh - 150px)'}), selected_visualization, store_figure

        elif selected_visualization == "viz15":
            if "viz15" in store_figure:
                figure = store_figure["viz15"]
            else:
                ref = reshape_to_3d(ref)
                main = reshape_to_3d(main)
                figure = spectral_analysis(ref, main)
                store_figure["viz15"] = figure

            return dcc.Graph(id="vis-graph", figure=figure,
                             style={'width': '100%',
                                    'height': 'calc(100vh - 150px)'}), selected_visualization, store_figure

        elif selected_visualization == "viz16":
            if "viz16" in store_figure:
                figure = store_figure["viz16"]
            else:
                ref = reshape_to_3d(ref)
                main = reshape_to_3d(main)
                figure = eigenvalue_comparison(ref, main)
                store_figure["viz16"] = figure

            return dcc.Graph(id="vis-graph", figure=figure,
                             style={'width': '100%',
                                    'height': 'calc(100vh - 150px)'}), selected_visualization, store_figure

        return no_update, no_update, no_update

    @app.callback(
        Output("visualization-modal", "is_open"),
        Input("visualization-button", "n_clicks"),
        prevent_initial_call=True
    )
    def open_visualization_modal(n_open):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        return True


def register_clientside_callbacks(app):
    # Center on the node when Ctrl key is being held
    app.clientside_callback(
        """
        function(nodeId, keysPressed) {
            if (!keysPressed || !("Control" in keysPressed) || nodeId == null) {
                return;
            }
            if (window.cy) {
                const element = window.cy.getElementById(nodeId);
                if (element.length === 0) return; // Check if node exists
                const zoom = window.cy.zoom();
                const nodePos = element.position();
                const viewportCenterX = window.cy.width() / 2;
                const viewportCenterY = window.cy.height() / 2;
                const newPanX = viewportCenterX - (nodePos.x * zoom);
                const newPanY = viewportCenterY - (nodePos.y * zoom);
                window.cy.animate({
                    pan: { x: newPanX, y: newPanY }
                }, { duration: 150, easing: 'ease-in-out' });
            }
            return;
        }
        """,
        Input('selected-node-id-store', 'data'),
        State('keyboard', 'keys_pressed')
    )
