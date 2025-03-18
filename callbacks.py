import base64
import copy
import io
import json
import os
import bisect
from datetime import datetime
from pathlib import Path

import numpy as np
from dash import no_update, callback_context, html, dcc
from dash.dependencies import Input, Output, State, ALL
from run_inference import get_available_plugins

from cache import result_cache, task_queue, processing_layers
import cache
from visualizations.new_cool_visualizations import animated_slices, isosurface_diff, \
    interactive_tensor_diff_dashboard, \
    hierarchical_diff_visualization, tensor_network_visualization, channel_correlation_matrices
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
    config["datetime"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
        prevent_initial_call=True
    )
    def update_selected_node_id(tap_node, selected_layer_index):
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
            selected_layer = cache.layers_store_data[selected_layer_index]
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
        prevent_initial_call=True
    )
    def update_graph_elements(_, tap_node, finished_nodes, selected_layer_index, elements, config_data):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        if any(trigger.startswith('first-load') for trigger in triggers):
            for element in elements:
                node_id = element['data'].get("id")

                if node_id in result_cache:
                    element['data']['border_color'] = 'green'

                if node_id in processing_layers:
                    element['data']['border_color'] = 'yellow'

            cache.ir_graph_elements = elements
            return elements

        new_elements = cache.ir_graph_elements

        if any(trigger.startswith('ir-graph') for trigger in triggers):
            node_id = tap_node['data'].get('id')

            if node_id not in result_cache and not any(task[0] == node_id for task in task_queue.queue):
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
            # TODO do a proper error handling
            for element in new_elements:
                if element['data'].get('id') in finished_nodes:
                    element['data']['border_color'] = 'green'

        if any(trigger.startswith('selected-layer-index-store') for trigger in triggers):
            selected_layer = cache.layers_store_data[selected_layer_index]
            node_id = selected_layer["node_id"]

            set_selected_node_style(new_elements, node_id)

        return new_elements

    @app.callback(
        Output('layers-store', 'data'),
        Output('clicked-graph-node-id-store', 'data'),
        Input('first-load', 'pathname'),
        Input('ir-graph', 'tapNode'),
        Input('just-finished-tasks-store', 'data'),
        prevent_initial_call=True
    )
    def update_layers_list(_, tap_node, finished_nodes):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        if any(trigger.startswith('first-load') for trigger in triggers):
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
            return layer_list_out, no_update

        layer_list_out = cache.layers_store_data
        clicked_graph_node_id = no_update

        if any(trigger.startswith('ir-graph') for trigger in triggers):
            node_id = tap_node['data'].get('id')
            clicked_graph_node_id = node_id  # To trigger layer selection after new layer was added to the list
            if node_id not in result_cache and not any(layer["node_id"] == node_id for layer in layer_list_out):
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
        State("keyboard", "keydown"),
        State('selected-layer-index-store', 'data'),
        State("inference-settings-modal", "is_open"),
        State("visualization-modal", "is_open"),
        prevent_initial_call=True
    )
    def update_selected_layer(n_keydowns, li_n_clicks, clicked_graph_node_id, keydown,
                              selected_layer_index, is_settings_opened, is_visualization_opened):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        if is_settings_opened or is_visualization_opened:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        new_index = no_update

        if any(trigger.startswith('clicked-graph-node-id-store') for trigger in triggers):
            for index, element in enumerate(cache.layers_store_data):
                if element["node_id"] == clicked_graph_node_id:
                    new_index = index
                    break

        if any(trigger.startswith('keyboard') for trigger in triggers):
            num_layers = len(cache.layers_store_data)
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
        Output('save-outputs-button', 'style'),
        Input('selected-node-id-store', 'data'),
        Input('selected-layer-index-store', 'data'),
        Input('just-finished-tasks-store', 'data'),
        State('selected-layer-name-store', 'data'),
        prevent_initial_call=True
    )
    def update_stats(selected_node_id, selected_layer_index, finished_nodes, selected_layer_name):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, {'display': 'none'}, {'display': 'none'}

        triggers = [t['prop_id'] for t in ctx.triggered]
        node_id = None

        if any(trigger.startswith('selected-node-id-store') for trigger in triggers):
            node_id = selected_node_id

        if any(trigger.startswith('selected-layer-index-store') for trigger in triggers):
            selected_layer = cache.layers_store_data[selected_layer_index]
            node_id = selected_layer["node_id"]

        if any(trigger.startswith('just-finished-tasks-store') for trigger in triggers) and finished_nodes:
            if selected_node_id in finished_nodes:
                node_id = selected_node_id

        if node_id is None:
            return no_update, no_update, {'display': 'none'}, {'display': 'none'}

        cached_result = result_cache.get(node_id)
        if cached_result:
            # Show the button when a cached result exists.
            button_style = {'margin': '4px', 'display': 'block', 'width': 'calc(100% - 8px)'}
            return selected_layer_name, cached_result["right-panel"], button_style, button_style
        else:
            # Hide the button while processing.
            button_style = {'display': 'none'}
            return selected_layer_name, "Processing...", button_style, button_style

    #######################################################################################################################

    @app.callback(
        Output("plugin-store", "data"),
        Output('reference-plugin-dropdown', 'options'),
        Output('main-plugin-dropdown', 'options'),
        Input("ov-bin-path", "value"),
    )
    def find_plugins(openvino_bin):
        if not openvino_bin or not os.path.exists(openvino_bin):
            return [], [], []

        devices = get_available_plugins(openvino_bin)
        if not devices:
            return [], [], []

        device_options = [{'label': d, 'value': d} for d in devices]

        return (
            device_options,
            device_options,
            device_options
        )

    @app.callback(
        Output("inference-settings-modal", "is_open"),
        Output("config-store", "data"),
        Output("model-xml-path", "value"),
        Output("ov-bin-path", "value"),
        Output("reference-plugin-dropdown", "value"),
        Output("main-plugin-dropdown", "value"),
        Output("reference-plugin-dropdown", "placeholder"),
        Output("main-plugin-dropdown", "placeholder"),
        Output({"type": "model-input", "name": ALL}, "value"),
        Input("save-inference-config-button", "n_clicks"),
        Input("inference-settings-btn", "n_clicks"),
        State("config-store", "data"),
        State("model-xml-path", "value"),
        State("ov-bin-path", "value"),
        State("reference-plugin-dropdown", "value"),
        State("main-plugin-dropdown", "value"),
        State({"type": "model-input", "name": ALL}, "value"),
        prevent_initial_call=True
    )
    def save_config(save_btn_clicks, open_settings_btn_clicks, config, model_xml, bin_path, ref_plugin,
                    other_plugin, all_input_values):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        if any(trigger.startswith('inference-settings-btn') for trigger in triggers):
            return True, no_update, config["model_xml"], config["ov_bin_path"], None, None, config["plugin1"], config[
                "plugin2"], config["model_inputs"]

        if any(trigger.startswith('save-inference-config-button') for trigger in triggers):
            date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            plugin1 = ref_plugin if ref_plugin is not None else config["plugin1"]
            plugin2 = other_plugin if other_plugin is not None else config["plugin2"]

            config = {"datetime": date_time, "model_xml": model_xml, "ov_bin_path": bin_path,
                      "plugin1": plugin1, "plugin2": plugin2,
                      "model_inputs": all_input_values}

            return False, config, no_update, no_update, no_update, no_update, no_update, no_update, [no_update] * len(
                all_input_values)

    @app.callback(
        Output("notification-toast", "is_open"),
        Output("notification-toast", "children"),
        Input("save-outputs-button", "n_clicks"),
        State("config-store", "data"),
        State("selected-node-id-store", "data"),
        prevent_initial_call=True
    )
    def toggle_toast(n, config, node_id):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        folder_name = f"outputs/{config["datetime"]}"
        Path(f"{folder_name}").mkdir(parents=True, exist_ok=True)

        result = result_cache[node_id]
        layer_name = result["layer_name"].replace("/", "-")  # sanitize the layer name

        result["main"].tofile(f"{folder_name}/{int(node_id):04d}_{layer_name}.bin")
        result["ref"].tofile(f"{folder_name}/{int(node_id):04d}_{layer_name}_ref.bin")

        return True, f"Results are saved in {Path.cwd()}/{folder_name}"

    @app.callback(
        Output("visualization-buttons", "children"),
        Input("last-selected-visualization", "data"),
        State("visualization-buttons", "children"),
        prevent_initial_call=True
    )
    def update_visualization_button_selection(last_selected, viz_buttons):
        for btn in viz_buttons:
            btn_id = btn["props"]["id"]
            btn["props"]["active"] = (btn_id["index"] == last_selected)

        return viz_buttons

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
        prevent_initial_call=True
    )
    def select_visualization_type(is_open, btn_clicks, store_figure, last_selected_visualization, config, node_id):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        # Skip phantom click event on first loading
        if ctx.triggered[0]["value"] is None:
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
                viz_name = last_selected_visualization if last_selected_visualization is not None else default_viz_name
            else:
                return html.Div(), last_selected_visualization, no_update

        else:
            button_id = json.loads(triggered_id)
            viz_name = button_id["index"]

        data = result_cache.get(node_id, {})
        ref = reshape_to_3d(data.get("ref"))
        main = reshape_to_3d(data.get("main"))

        np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(main, nan=0.0, posinf=0.0, neginf=0.0)

        if viz_name == "viz1":
            if viz_name in store_figure:
                viz = store_figure[viz_name]
            else:
                diff = main - ref
                viz = plot_volume_tensor(diff)
                viz = dcc.Graph(id="vis-graph", figure=viz,
                                style={'width': '100%',
                                       'height': 'calc(100vh - 150px)'})
                store_figure[viz_name] = viz

            return viz, viz_name, store_figure

        elif viz_name == "viz2":
            if viz_name in store_figure:
                viz = store_figure[viz_name]
            else:
                ref_plugin_name = config["plugin1"]
                main_plugin_name = config["plugin2"]
                viz = plot_diagnostics(ref, main, ref_plugin_name, main_plugin_name)
                buf = io.BytesIO()
                viz.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                encoded_diag = base64.b64encode(buf.getvalue()).decode("utf-8")
                viz = html.Img(
                    src=f"data:image/png;base64,{encoded_diag}",
                    style={"width": "100%", "display": "block", "margin": "0 auto"}
                )
                store_figure[viz_name] = viz

            return viz, viz_name, store_figure

        elif viz_name == "viz3":
            if viz_name in store_figure:
                viz = store_figure[viz_name]
            else:
                viz = animated_slices(ref, main, axis=0, fps=2)
                store_figure[viz_name] = viz

            return html.Div(
                html.Iframe(
                    srcDoc=viz,
                    style={
                        "width": "100%",
                        "height": "100vh",
                        "border": "none",
                        "minHeight": "600px"
                    }
                ),
                style={
                    "width": "100%",
                    "height": "calc(100vh - 200px)",  # Added height for vertical centering
                    "display": "flex",
                    "justifyContent": "center",  # Center horizontally
                    "alignItems": "center"  # Center vertically
                }
            ), viz_name, store_figure

        elif viz_name == "viz4":
            if viz_name in store_figure:
                viz = store_figure[viz_name]
            else:
                viz = isosurface_diff(ref, main)
                store_figure[viz_name] = viz

            return dcc.Graph(id="vis-graph", figure=viz,
                             style={'width': '100%',
                                    'height': 'calc(100vh - 150px)'}), viz_name, store_figure

        elif viz_name == "viz9":
            if viz_name in store_figure:
                viz = store_figure[viz_name]
            else:
                viz = hierarchical_diff_visualization(ref, main)
                store_figure[viz_name] = viz

            return html.Div(
                html.Div(
                    dcc.Graph(
                        id="vis-graph",
                        figure=viz,
                        config={'responsive': True},
                        style={
                            "width": "100%",
                            "height": "100%"
                        }
                    ),
                    style={
                        "height": "100%",
                        "width": "100%",
                    }
                ),
                style={
                    "height": "calc(100vh - 150px)",
                    "display": "flex",
                    "justifyContent": "center",  # center horizontally
                    "alignItems": "center"  # center vertically
                }
            ), viz_name, store_figure

        elif viz_name == "viz10":
            if viz_name in store_figure:
                viz = store_figure[viz_name]
            else:
                viz = tensor_network_visualization(ref, main)
                store_figure[viz_name] = viz

            return html.Div(
                html.Div(
                    dcc.Graph(
                        id="vis-graph",
                        figure=viz,
                        config={'responsive': True},
                        style={
                            "width": "100%",
                            "height": "100%"
                        }
                    ),
                    style={
                        "width": "100%",
                        "height": "100%",
                    }
                ),
                style={
                    "height": "calc(100vh - 150px)",
                    "display": "flex",
                    "justifyContent": "center",  # center horizontally
                    "alignItems": "center"  # center vertically
                }
            ), viz_name, store_figure

        elif viz_name == "viz12":
            if viz_name in store_figure:
                viz = store_figure[viz_name]
            else:
                viz = channel_correlation_matrices(ref, main)
                store_figure[viz_name] = viz

            return html.Div(
                html.Div(
                    dcc.Graph(
                        id="vis-graph",
                        figure=viz,
                        config={'responsive': True},
                        style={
                            "width": "100%",
                            "height": "100%"
                        }
                    ),
                    style={
                        "width": "100%",
                        "aspectRatio": f"{7 / 2}",
                    }
                ),
                style={
                    "height": "calc(100vh - 150px)",
                    "display": "flex",
                    "justifyContent": "center",  # center horizontally
                    "alignItems": "center"  # center vertically
                }
            ), viz_name, store_figure

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
        function(nodeId, settingsOpened, visualizationOpened) {
            if (window.ctrlPressed === undefined) {
                window.ctrlPressed = false;
                document.addEventListener("keydown", function(e) {
                    if (e.key === "Control") {
                        window.ctrlPressed = true;
                    }
                });
                document.addEventListener("keyup", function(e) {
                    if (e.key === "Control") {
                        window.ctrlPressed = false;
                    }
                });
            }
        
            const isSettingsOpen = settingsOpened ?? false;
            const isVisualizationOpen = visualizationOpened ?? false;
            if (isSettingsOpen || isVisualizationOpen) {
                return;
            }
            
            if (!window.ctrlPressed || nodeId == null) {
                return;
            }
            
            if (window.cy) {
                const element = window.cy.getElementById(nodeId);
                if (element.length === 0) return; // Check if node exists

                const currentPan = window.cy.pan();
                
                const zoom = window.cy.zoom();
                const nodePos = element.position();
                const viewportCenterX = window.cy.width() / 2;
                const viewportCenterY = window.cy.height() / 2;
                const newPanX = viewportCenterX - (nodePos.x * zoom);
                const newPanY = viewportCenterY - (nodePos.y * zoom);
                
                const dx = newPanX - currentPan.x;
                const dy = newPanY - currentPan.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                const duration = Math.min(300, Math.max(100, distance));
                
                window.cy.stop();
                window.cy.animate({
                    pan: { x: newPanX, y: newPanY }
                }, { 
                    duration: duration,
                    easing: 'ease-in-out',
                    queue: false
                });
            }
            return;
        }
        """,
        Input('selected-node-id-store', 'data'),
        State("inference-settings-modal", "is_open"),
        State("visualization-modal", "is_open"),
    )
