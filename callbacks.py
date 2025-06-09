import ast
import base64
import copy
import io
import json
import os
import bisect
from pathlib import Path
from queue import Empty

import numpy as np
from dash import no_update, callback_context, html, dcc
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from layout import build_dynamic_stylesheet, update_config, read_openvino_ir, build_model_input_fields
from openvino_graph import parse_openvino_ir
from run_inference import get_available_plugins, prepare_submodel_and_inputs, get_ov_core
from colors import BorderColor

import cache
from visualizations.new_cool_visualizations import animated_slices, isosurface_diff, \
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


def update_border_color(elements, node_id, color):
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


def parse_prop_id(prop_id):
    """
    - If prop_id looks like "{…}.some_prop", parse the {…} as JSON or a Python literal.
    - Otherwise, if it’s "some-id.some_prop", split on the last '.' and return the id string.
    Returns: (id_value, trigger_name_or_None)
      • id_value will be a dict if the original began with "{", or a plain string otherwise.
      • trigger_name_or_None is the part after the final '.', or None if no dot‐suffix.
    """
    # 1) If it begins with "{", try to find the matching "}" and parse it.
    if prop_id.startswith("{"):
        i = prop_id.rfind("}")
        if i == -1:
            raise ValueError(f"Malformed prop_id (no closing brace): {prop_id!r}")
        literal = prop_id[: i + 1]
        remainder = prop_id[i + 1:].lstrip()
        trigger = remainder[1:].strip() if remainder.startswith(".") else None

        try:
            return json.loads(literal), trigger
        except json.JSONDecodeError:
            return ast.literal_eval(literal), trigger

    # 2) Otherwise, treat it as "some-id.some_prop" (or maybe just "some-id" with no dot).
    else:
        # Find the last dot
        j = prop_id.rfind(".")
        if j == -1:
            # No dot at all → no trigger suffix
            return prop_id, None
        else:
            return prop_id[:j], prop_id[j + 1:]


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

        finished_nodes = [node for node in cache.processing_layers if node in cache.result_cache]
        if not finished_nodes:
            return no_update

        for node_id in finished_nodes:
            cache.processing_layers.pop(node_id)

        return finished_nodes

    @app.callback(
        Output('selected-node-id-store', 'data'),
        Output('selected-layer-type-store', 'data'),
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
        selected_layer_type = no_update

        if any(trigger.startswith('ir-graph') for trigger in triggers):
            selected_node_id = tap_node['data'].get('id')
            selected_layer_type = tap_node['data'].get('layer_type')

        if any(trigger.startswith('selected-layer-index-store') for trigger in triggers):
            if selected_layer_index is None:
                return None, None

            selected_layer = cache.layers_store_data[selected_layer_index]
            selected_node_id = selected_layer["node_id"]
            selected_layer_type = selected_layer["layer_type"]

        return selected_node_id, selected_layer_type

    def cut_model_at_node_and_remove_unused(ov, core, model_xml, original_input_paths, node_to_cut, node_output_paths):
        model = core.read_model(model=model_xml)

        target_node = None
        for op in model.get_ordered_ops():
            if op.get_friendly_name() == node_to_cut:
                target_node = op
                break
        if target_node is None:
            raise RuntimeError("Could not find the target node in the model.")

        import openvino.opset14 as ops

        new_params = []
        for idx, output in enumerate(target_node.outputs()):
            new_param = ops.parameter(
                output.get_partial_shape(),
                output.get_element_type(),
                target_node.get_friendly_name() + f"_input_{idx}"
            )
            new_params.append(new_param)
            for target_input in list(output.get_target_inputs()):
                target_input.replace_source_output(new_param.output(0))

        original_params = [inp.get_node() for inp in model.inputs]
        all_params = original_params + new_params

        from openvino import Model
        model = Model(model.outputs, all_params, "sub_model")
        model.validate_nodes_and_infer_types()

        def get_used_parameters(model):
            used_params = set()
            visited = set()

            stack = [out.get_node() for out in model.outputs]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                if node.get_type_name() == "Parameter":
                    used_params.add(node)
                for inp in node.inputs():
                    source_node = inp.get_source_output().get_node()
                    if source_node not in visited:
                        stack.append(source_node)
            return used_params

        used_params = get_used_parameters(model)
        connected_params = [inp.get_node() for inp in model.inputs if inp.get_node() in used_params]

        model = Model(model.outputs, connected_params, "sub_model")
        model.validate_nodes_and_infer_types()

        orig_mapping = {}
        for orig_inp, orig_path in zip(original_params, original_input_paths):
            name = orig_inp.get_friendly_name()
            orig_mapping[name] = Path(orig_path).resolve()

        new_mapping = {}
        for new_inp, new_path in zip(new_params, node_output_paths):
            name = new_inp.get_friendly_name()
            orig_mapping[name] = Path(new_path).resolve()

        mapping = {**orig_mapping, **new_mapping}

        new_input_paths = []
        for param in connected_params:
            name = param.get_friendly_name()
            new_input_paths.append(mapping.get(name, ""))

        return model, new_input_paths

    @app.callback(
        Output('ir-graph', 'elements', allow_duplicate=True),
        Output('refresh-layout-trigger', 'data'),
        Input('model-path-after-cut', 'data'),
        State('refresh-layout-trigger', 'data'),
        prevent_initial_call=True
    )
    def clear_cache(model_after_cut, refresh_layout_trigger):
        with cache.lock:
            cache.result_cache.clear()
            cache.status_cache.clear()
            cache.processing_layers.clear()
            cache.layers_store_data.clear()

        elements = parse_openvino_ir(model_after_cut)

        cache.ir_graph_elements = elements
        return elements, refresh_layout_trigger + 1

    @app.callback(
        Output('config-store-after-cut', 'data'),
        Output('model-path-after-cut', 'data'),
        Output('transformed-node-name-store', 'data'),
        Input('transform-to-input-button', 'n_clicks'),
        State('config-store', 'data'),
        State('selected-node-id-store', 'data'),
        prevent_initial_call=True
    )
    def transform_layer_to_model_input(transform_to_input_btn, config, selected_node_id):
        # Stop any running inference tasks
        cache.cancel_event.set()

        # Clear the task queue
        while True:
            try:
                cache.task_queue.get_nowait()
                cache.task_queue.task_done()
            except Empty:
                break

        # Save outputs of current node in a subgraph folder
        layer = cache.result_cache[selected_node_id]
        node_to_cut = layer["layer_name"]

        input_paths = config["model_inputs"]
        model_xml = config["model_xml"]
        openvino_bin = config["ov_bin_path"]

        reproducer_folder = Path(config["output_folder"]) / "sub_networks" / "network0"
        Path(reproducer_folder).mkdir(parents=True, exist_ok=True)

        node_outputs = []

        # Save layer outputs to make them new model inputs
        for index, output in enumerate(layer["outputs"]):
            node_output = reproducer_folder / f"input_{index}.bin"
            output["main"].tofile(node_output)
            node_outputs.append(node_output)

        ov, core = get_ov_core(openvino_bin)
        new_model, new_input_paths = cut_model_at_node_and_remove_unused(ov, core, model_xml, input_paths, node_to_cut,
                                                                         node_outputs)

        xml_path = reproducer_folder / "model.xml"
        bin_path = reproducer_folder / "model.bin"
        ov.serialize(new_model, xml_path, bin_path)

        update_config(
            config=config,
            model_xml=str(xml_path.resolve()),
            model_inputs=[str(path) for path in new_input_paths]
        )

        # Find the actual name of the parameter node in the new graph
        # Parse the new model to get the actual parameter node name
        new_elements = parse_openvino_ir(str(xml_path.resolve()))
        parameter_node = None
        for element in new_elements:
            if 'data' in element and element['data'].get('type') == 'Parameter':
                parameter_node = element
                break

        transformed_node_name = parameter_node['data']['layer_name'] if parameter_node else "Parameter"

        return config, config["model_xml"], transformed_node_name

    @app.callback(
        Output('ir-graph', 'elements'),
        Input('first-load', 'pathname'),
        Input('ir-graph', 'tapNode'),
        Input('just-finished-tasks-store', 'data'),
        Input('selected-layer-index-store', 'data'),
        Input('restart-layer-button', 'n_clicks'),
        Input('model-path-after-cut', 'data'),
        Input('clear-queue-store', "data"),
        State('ir-graph', 'elements'),
        State('config-store', 'data'),
        State('plugins-config-store', 'data'),
        prevent_initial_call=True
    )
    def update_graph_elements(_, tap_node, finished_nodes, selected_layer_index, restart_layer_btn,
                              new_model_path, clear_queue, elements, config_data, plugins_config):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        if any(trigger.startswith('first-load') for trigger in triggers) or any(
                trigger.startswith('clear-queue-store') for trigger in triggers):
            for element in elements:
                node_id = element['data'].get("id")

                if node_id in cache.result_cache:
                    result = cache.result_cache[node_id]
                    if "error" in result:
                        element['data']['border_color'] = BorderColor.ERROR.value
                    else:
                        element['data']['border_color'] = BorderColor.SUCCESS.value
                elif node_id in cache.processing_layers:
                    element['data']['border_color'] = BorderColor.PROCESSING.value
                else:
                    element['data']['border_color'] = BorderColor.DEFAULT.value

            cache.ir_graph_elements = elements
            return elements

        new_elements = cache.ir_graph_elements

        if any(trigger.startswith('ir-graph') for trigger in triggers):
            node_id = tap_node['data'].get('id')

            if node_id not in cache.result_cache and not any(task[0] == node_id for task in cache.task_queue.queue):
                layer_name = tap_node['data'].get('layer_name')
                layer_type = tap_node['data'].get('type')

                cache.processing_layers[node_id] = {
                    "layer_name": layer_name,
                    "layer_type": layer_type
                }

                cache.task_queue.put((node_id, layer_name, layer_type, config_data, plugins_config))
                update_border_color(new_elements, node_id, BorderColor.PROCESSING.value)

            set_selected_node_style(new_elements, node_id)

        if any(trigger.startswith('just-finished-tasks-store') for trigger in triggers):
            for element in new_elements:
                node_id = element['data'].get("id")
                if node_id in finished_nodes:
                    result = cache.result_cache[node_id]
                    color = BorderColor.ERROR.value if "error" in result else BorderColor.SUCCESS.value
                    element['data']['border_color'] = color

        if any(trigger.startswith('selected-layer-index-store') for trigger in triggers):
            if selected_layer_index is None:
                node_id = None
            else:
                selected_layer = cache.layers_store_data[selected_layer_index]
                node_id = selected_layer["node_id"]

            set_selected_node_style(new_elements, node_id)

        if any(trigger.startswith('restart-layer-button') for trigger in triggers):
            selected_layer = cache.layers_store_data[selected_layer_index]
            node_id = selected_layer["node_id"]

            result = cache.result_cache.pop(node_id)

            layer_name = result["layer_name"]
            layer_type = result["layer_type"]

            cache.processing_layers[node_id] = {
                "layer_name": layer_name,
                "layer_type": layer_type
            }

            cache.task_queue.put((node_id, layer_name, layer_type, config_data, plugins_config))
            update_border_color(new_elements, node_id, BorderColor.PROCESSING.value)

        return new_elements

    @app.callback(
        [
            Output('layers-store', 'data'),
            Output('clicked-graph-node-id-store', 'data'),
            Output('metrics-store', 'data')
        ],
        Input('first-load', 'pathname'),
        Input('ir-graph', 'tapNode'),
        Input('just-finished-tasks-store', 'data'),
        Input('restart-layer-button', 'n_clicks'),
        Input('model-path-after-cut', 'data'),
        Input("clear-queue-store", "data"),
        State('selected-node-id-store', 'data'),
        State('metrics-store', 'data'),
        prevent_initial_call=True
    )
    def update_layers_list(_, tap_node, finished_nodes, restart_layers_btn, model_after_cut, clear_queue_btn, 
                           selected_node_id, metrics_store):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        if any(trigger.startswith('model-path-after-cut') for trigger in triggers):
            return [], None, {}

        if any(trigger.startswith('clear-queue-store') for trigger in triggers):
            cache.processing_layers.clear()
            # Only remove tasks with "running" status, keep finished tasks
            cache.layers_store_data = [layer for layer in cache.layers_store_data if layer.get("status") != "running"]
            return cache.layers_store_data, None, metrics_store

        # Function to calculate metrics for a result
        def calculate_metrics(result):
            if "error" in result or "outputs" not in result:
                return {}

            metrics_data = {}
            outputs_metrics = []

            for idx, output in enumerate(result["outputs"]):
                ref_data = output["ref"]
                main_data = output["main"]
                diff = ref_data - main_data

                # Create a metrics dictionary for this output
                output_metrics = {
                    # Basic metrics
                    "min": float(np.min(ref_data)),
                    "mean": float(np.mean(ref_data)),
                    "max": float(np.max(ref_data)),
                    "std": float(np.std(ref_data)),

                    # Advanced metrics
                    "MAE": float(np.mean(np.abs(diff))),
                    "MSE": float(np.mean(diff ** 2))
                }

                # Calculate RMSE and NRMSE
                rmse_value = float(np.sqrt(output_metrics["MSE"]))
                ref_range = float(np.max(ref_data) - np.min(ref_data))
                output_metrics["NRMSE"] = float(rmse_value / ref_range if ref_range != 0 else 0)

                # Calculate PSNR
                max_val = float(np.max(np.abs(ref_data)))
                output_metrics["PSNR"] = float('inf') if output_metrics["MSE"] == 0 else float(20 * np.log10(max_val) - 10 * np.log10(output_metrics["MSE"]))

                # Calculate Pearson correlation
                if np.std(ref_data) > 0 and np.std(main_data) > 0:
                    output_metrics["Pearson"] = float(np.corrcoef(ref_data.flatten(), main_data.flatten())[0, 1])
                else:
                    output_metrics["Pearson"] = 0.0

                outputs_metrics.append(output_metrics)

            metrics_data["outputs"] = outputs_metrics
            return metrics_data

        if any(trigger.startswith('first-load') for trigger in triggers) or any(
                trigger.startswith('clear-queue-store') for trigger in triggers):
            layer_list_out = []
            new_metrics_store = {}

            for node_id, result in cache.result_cache.items():
                if "error" in result:
                    layer_list_out.append({
                        "node_id": node_id,
                        "layer_name": result["layer_name"],
                        "layer_type": result["layer_type"],
                        "status": "error"
                    })
                else:
                    # Calculate metrics and store them in the metrics store
                    metrics_data = calculate_metrics(result)
                    new_metrics_store[node_id] = metrics_data

                    # For backward compatibility, also include a simplified version in the layer data
                    layer_metrics = {}
                    if "outputs" in metrics_data:
                        for idx, output_metrics in enumerate(metrics_data["outputs"]):
                            for metric_name, metric_value in output_metrics.items():
                                layer_metrics[f"{metric_name}_{idx}"] = metric_value

                    layer_list_out.append({
                        "node_id": node_id,
                        "layer_name": result["layer_name"],
                        "layer_type": result["layer_type"],
                        "status": "done",
                        "metrics": layer_metrics
                    })

            if any(trigger.startswith('first-load') for trigger in triggers):
                for node_id, result in cache.processing_layers.items():
                    layer_list_out.append({
                        "node_id": node_id,
                        "layer_name": result["layer_name"],
                        "layer_type": result["layer_type"],
                        "status": "running"
                    })

            layer_list_out = sorted(layer_list_out, key=lambda item: int(item["node_id"]))
            return layer_list_out, no_update, new_metrics_store

        layer_list_out = cache.layers_store_data
        clicked_graph_node_id = no_update
        new_metrics_store = metrics_store.copy() if metrics_store else {}

        if any(trigger.startswith('ir-graph') for trigger in triggers):
            node_id = tap_node['data'].get('id')
            clicked_graph_node_id = node_id  # To trigger layer selection after new layer was added to the list
            if node_id not in cache.result_cache and not any(layer["node_id"] == node_id for layer in layer_list_out):
                layer_name = tap_node['data'].get('layer_name')
                layer_type = tap_node['data'].get('type')

                list_of_ids = [int(item["node_id"]) for item in layer_list_out]
                insertion_index = bisect.bisect_left(list_of_ids, int(node_id))
                layer_list_out.insert(insertion_index, {
                    "node_id": node_id,
                    "layer_name": layer_name,
                    "layer_type": layer_type,
                    "status": "running"
                })

        if any(trigger.startswith('just-finished-tasks-store') for trigger in triggers) and finished_nodes:
            for layer in layer_list_out:
                node_id = layer["node_id"]
                if node_id in finished_nodes:
                    result = cache.result_cache[node_id]
                    status = "error" if "error" in result else "done"
                    layer["status"] = status

                    # If the layer is done, calculate and add metrics
                    if status == "done":
                        # Calculate metrics and store them in the metrics store
                        metrics_data = calculate_metrics(result)
                        new_metrics_store[node_id] = metrics_data

                        # For backward compatibility, also include a simplified version in the layer data
                        layer_metrics = {}
                        if "outputs" in metrics_data:
                            for idx, output_metrics in enumerate(metrics_data["outputs"]):
                                for metric_name, metric_value in output_metrics.items():
                                    layer_metrics[f"{metric_name}_{idx}"] = metric_value

                        layer["metrics"] = layer_metrics

        if any(trigger.startswith('restart-layer-button') for trigger in triggers):  # and finished_nodes
            for layer in layer_list_out:
                node_id = layer["node_id"]
                if node_id == selected_node_id:
                    layer["status"] = "running"
                    # Remove metrics when restarting a layer
                    if "metrics" in layer:
                        del layer["metrics"]
                    if node_id in new_metrics_store:
                        del new_metrics_store[node_id]

        return layer_list_out, clicked_graph_node_id, new_metrics_store

    @app.callback(
        Output('layer-panel-list', 'children'),
        Input('selected-layer-index-store', 'data'),
        Input('layers-store', 'data'),
        Input('factorio-grayed-out-operations', 'data'),
        State('layer-panel-list', 'children'),
        prevent_initial_call=True
    )
    def render_layers(selected_index, layers_list, grayed_out_operations, rendered_layers):
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
            if layer["status"] == "done":
                color = '#4CAF50'
            elif layer["status"] == "error":
                color = '#F05050'
            else:
                color = '#BA8E23'

            style = {
                'color': color,
                'padding': '4px',
                'marginBottom': '0px',
            }

            # Gray out layers that are in the grayed-out list
            if layer.get("layer_name", "") in grayed_out_operations:
                style.update({
                    'opacity': '0.5',
                })

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
        Input('model-path-after-cut', 'data'),
        Input("clear-queue-store", "data"),
        State("keyboard", "keydown"),
        State('selected-layer-index-store', 'data'),
        State("inference-settings-modal", "is_open"),
        State("visualization-modal", "is_open"),
        prevent_initial_call=True
    )
    def update_selected_layer(n_keydowns, li_n_clicks, clicked_graph_node_id, model_after_cut, clear_queue, keydown,
                              selected_layer_index, is_settings_opened, is_visualization_opened):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        if is_settings_opened or is_visualization_opened:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        new_index = no_update
        if any(trigger.startswith('model-path-after-cut') for trigger in triggers):
            return None

        if any(trigger.startswith('clear-queue-store') for trigger in triggers):
            return None

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
                    id_part, _ = parse_prop_id(first_trigger)
                    triggered_id = id_part
                    new_index = triggered_id.get("index")
                except Exception:
                    pass

        return new_index

    @app.callback(
        Output('right-panel-layer-name', 'children'),
        Output('right-panel-content', 'children'),
        Output('save-outputs-button', 'style'),
        Output('save-reproducer-button', 'style'),
        Output('transform-to-input-button', 'style'),
        Output('restart-layer-button', 'style'),
        Input('selected-node-id-store', 'data'),
        Input('selected-layer-index-store', 'data'),
        Input('just-finished-tasks-store', 'data'),
        Input('restart-layer-button', 'n_clicks'),
        State('selected-layer-type-store', 'data'),
        prevent_initial_call=True
    )
    def update_stats(selected_node_id, selected_layer_index, finished_nodes, restart_layer_btn, selected_layer_name):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update, no_update

        triggers = [t['prop_id'] for t in ctx.triggered]
        node_id = None

        show = {'margin': '4px', 'display': 'block', 'width': 'calc(100% - 8px)'}
        hide = {'display': 'none'}

        if any(trigger.startswith('restart-layer-button') for trigger in triggers):
            return selected_layer_name, "Processing...", hide, hide, hide, hide

        if any(trigger.startswith('selected-node-id-store') for trigger in triggers):
            node_id = selected_node_id

        if any(trigger.startswith('selected-layer-index-store') for trigger in triggers):
            if selected_layer_index is None:
                return "Layer's name", "", hide, hide, hide, hide
            selected_layer = cache.layers_store_data[selected_layer_index]
            node_id = selected_layer["node_id"]

        if any(trigger.startswith('just-finished-tasks-store') for trigger in triggers) and finished_nodes:
            if selected_node_id in finished_nodes:
                node_id = selected_node_id

        if node_id is None:
            return no_update, no_update, no_update, no_update, no_update, no_update

        cached_result = cache.result_cache.get(node_id)
        if cached_result:
            if "error" in cached_result:
                return selected_layer_name, cached_result["error"], hide, hide, hide, show
            else:
                right_panel = cache.status_cache[node_id]
                return selected_layer_name, right_panel, show, show, show, hide
        else:
            return selected_layer_name, "Processing...", hide, hide, hide, hide

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
        Output("model-input-paths", "children"),
        Input("save-inference-config-button", "n_clicks"),
        Input("inference-settings-btn", "n_clicks"),
        Input('config-store-after-cut', 'data'),
        State("config-store", "data"),
        State("model-xml-path", "value"),
        State("ov-bin-path", "value"),
        State("reference-plugin-dropdown", "value"),
        State("main-plugin-dropdown", "value"),
        State({"type": "model-input", "name": ALL}, "value"),
        State("plugins-config-store", "data"),
        prevent_initial_call=True
    )
    def save_config(save_btn_clicks, open_settings_btn_clicks, config_after_cut, config, model_xml, bin_path,
                    ref_plugin,
                    other_plugin, all_input_values, plugins_config):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        if any(trigger.startswith('inference-settings-btn') for trigger in triggers):
            model_inputs = read_openvino_ir(config["model_xml"])
            inputs_layout_div = build_model_input_fields(model_inputs, config["model_inputs"])
            # Get plugin values from config, defaulting to None if not present
            plugin1 = config.get("plugin1")
            plugin2 = config.get("plugin2")
            return True, no_update, config["model_xml"], config["ov_bin_path"], None, None, plugin1, plugin2, inputs_layout_div

        if any(trigger.startswith('save-inference-config-button') for trigger in triggers):
            plugin1 = ref_plugin if ref_plugin is not None else config.get("plugin1")
            plugin2 = other_plugin if other_plugin is not None else config.get("plugin2")

            update_config(
                config,
                model_xml,
                bin_path,
                plugin1,
                plugin2,
                all_input_values
            )

            # Save settings to settings/settings.json
            settings_dir = Path("settings")
            settings_dir.mkdir(exist_ok=True)
            settings_file = settings_dir / "settings.json"

            # Create settings object with selected plugins
            settings = {
                "plugins": {
                    "reference_plugin": plugin1,
                    "main_plugin": plugin2
                }
            }

            # Add plugin configurations if available, filtering out empty values
            if plugins_config and isinstance(plugins_config, dict):
                filtered_plugin_configs = {}
                for plugin_name, plugin_config in plugins_config.items():
                    if plugin_config:
                        filtered_config = {k: v for k, v in plugin_config.items() if v.strip() != ""}
                        if filtered_config:  # Only include plugins with at least one non-empty config
                            filtered_plugin_configs[plugin_name] = filtered_config

                settings["plugin_configs"] = filtered_plugin_configs

            # Save settings to file
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=4)

            model_inputs = read_openvino_ir(config["model_xml"])
            inputs_layout_div = build_model_input_fields(model_inputs, config["model_inputs"])

            return False, config, no_update, no_update, no_update, no_update, no_update, no_update, inputs_layout_div

        if any(trigger.startswith('config-store-after-cut') for trigger in triggers):
            c = config_after_cut
            model_inputs = read_openvino_ir(c["model_xml"])
            inputs_layout_div = build_model_input_fields(model_inputs, c["model_inputs"])
            return False, c, c["model_xml"], no_update, no_update, no_update, no_update, no_update, inputs_layout_div

        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    @app.callback(
        Output("plugins-config-store", "data"),
        Input({'type': 'plugin-config', 'plugin': ALL, 'param': ALL}, 'value'),
        State("config-plugin-dropdown", "value"),
        State("plugins-config-store", "data"),
        State({'type': 'plugin-config', 'plugin': ALL, 'param': ALL}, 'id')
    )
    def auto_save_plugin_config(values, selected_plugin, store_data, ids):
        # If there's no selected plugin or no input ids, do not update.
        if not selected_plugin or not ids:
            return store_data if store_data is not None else {}
        if store_data is None:
            store_data = {}

        # Save only the inputs that belong to the currently selected plugin and have non-empty values.
        current_config = {
            comp_id["param"]: value
            for value, comp_id in zip(values, ids)
            if comp_id.get("plugin") == selected_plugin and value.strip() != ""
        }

        store_data[selected_plugin] = current_config
        return store_data

    @app.callback(
        Output("plugin-config-table", "children"),
        Input("config-plugin-dropdown", "value"),
        State("ov-bin-path", "value"),
        State("plugins-config-store", "data")
    )
    def update_plugin_config_table(selected_plugin, openvino_bin, store_data):
        if not selected_plugin or not openvino_bin:
            return html.Div("Please select a plugin and ensure the OpenVINO bin folder is set.")
        try:
            ov, core = get_ov_core(openvino_bin)
            config_keys = core.get_property(selected_plugin, "SUPPORTED_PROPERTIES")
            if not config_keys:
                return html.Div("No configurable parameters available for this plugin.")
            current_values = {}
            if store_data is not None and selected_plugin in store_data:
                current_values = store_data[selected_plugin]
            rows = []
            for key in sorted(config_keys):
                default_value = current_values.get(key, "")
                rows.append(
                    html.Tr([
                        html.Td(key),
                        html.Td(
                            dbc.Input(
                                id={"type": "plugin-config", "plugin": selected_plugin, "param": key},
                                type="text",
                                value=default_value,
                                debounce=False
                            )
                        )
                    ])
                )
            return dbc.Table(
                html.Tbody(rows),
                bordered=True,
                striped=True,
                hover=True,
                size="sm"
            )
        except Exception as e:
            return html.Div(f"Error: {str(e)}")

    @app.callback(
        Output("notification-toast", "is_open"),
        Output("notification-toast", "children"),
        Input("save-outputs-button", "n_clicks"),
        Input('save-reproducer-button', 'n_clicks'),
        State("config-store", "data"),
        State("selected-node-id-store", "data"),
        prevent_initial_call=True
    )
    def toggle_toast(save_outputs_btn, save_reproducer_btn, config, node_id):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        result = cache.result_cache[node_id]
        layer_name = result["layer_name"]
        sanitized_layer_name = layer_name.replace("/", "-")  # sanitize the layer name

        if any(trigger.startswith('save-outputs-button') for trigger in triggers):
            outputs_folder = Path(config["output_folder"]) / "outputs"
            Path(outputs_folder).mkdir(parents=True, exist_ok=True)

            for index, output in enumerate(result["outputs"]):
                output["main"].tofile(f"{outputs_folder}/{int(node_id):04d}_{sanitized_layer_name}_{index}.bin")
                output["ref"].tofile(f"{outputs_folder}/{int(node_id):04d}_{sanitized_layer_name}_{index}_ref.bin")

            return True, f"Results are saved in {Path.cwd()}/{outputs_folder}"

        if any(trigger.startswith('save-reproducer-button') for trigger in triggers):
            ov, core, inputs, preprocessed_model = prepare_submodel_and_inputs(layer_name, config["model_inputs"],
                                                                               config["model_xml"],
                                                                               config["ov_bin_path"],
                                                                               config["output_folder"])

            reproducer_folder = Path(config["output_folder"]) / "reproducers" / sanitized_layer_name
            Path(reproducer_folder).mkdir(parents=True, exist_ok=True)

            xml_path = reproducer_folder / "model.xml"
            bin_path = reproducer_folder / "model.bin"
            ov.serialize(preprocessed_model, xml_path, bin_path)

            for index, input_data in enumerate(inputs):
                input_data.tofile(f"{reproducer_folder}/input_{index}.bin")

            for index, output in enumerate(result["outputs"]):
                output["main"].tofile(f"{reproducer_folder}/output_{index}.bin")
                output["ref"].tofile(f"{reproducer_folder}/output_{index}_ref.bin")

            return True, f"Reproducer is saved in {reproducer_folder}"

        return no_update, no_update

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
        State("visualization-output-id", "data"),
        prevent_initial_call=True
    )
    def select_visualization_type(is_open, btn_clicks, store_figure, last_selected_visualization, config, node_id,
                                  output_id):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        # Skip phantom click event on first loading
        if ctx.triggered[0]["value"] is None:
            return no_update, no_update, no_update

        triggered_id, _ = parse_prop_id(ctx.triggered[0]["prop_id"])

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
            try:
                # Check if triggered_id is already a dictionary
                if isinstance(triggered_id, dict):
                    button_id = triggered_id
                else:
                    # Fix potential JSON parsing issues by ensuring proper quotes
                    cleaned_triggered_id = triggered_id.replace('"{', '{').replace('}"', '}')
                    button_id = json.loads(cleaned_triggered_id)
                viz_name = button_id["index"]
            except Exception as e:
                print(f"Error parsing triggered_id: {triggered_id}, Error: {str(e)}")
                return html.Div("Error parsing visualization selection"), last_selected_visualization, no_update

        store_name = f"{output_id}{viz_name}"

        data = cache.result_cache.get(node_id, {})
        output = data["outputs"][output_id]
        ref = reshape_to_3d(output["ref"])
        main = reshape_to_3d(output["main"])

        np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(main, nan=0.0, posinf=0.0, neginf=0.0)

        if viz_name == "viz1":
            if store_name in store_figure:
                viz = store_figure[store_name]
            else:
                diff = main - ref
                viz = plot_volume_tensor(diff)
                viz = dcc.Graph(id="vis-graph", figure=viz,
                                style={'width': '100%',
                                       'height': 'calc(100vh - 150px)'})
                store_figure[store_name] = viz

            return viz, viz_name, store_figure

        elif viz_name == "viz2":
            if store_name in store_figure:
                viz = store_figure[store_name]
            else:
                ref_plugin_name = config.get("plugin1", "Reference")
                main_plugin_name = config.get("plugin2", "Main")
                viz = plot_diagnostics(ref, main, ref_plugin_name, main_plugin_name)
                buf = io.BytesIO()
                viz.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                encoded_diag = base64.b64encode(buf.getvalue()).decode("utf-8")
                viz = html.Img(
                    src=f"data:image/png;base64,{encoded_diag}",
                    style={"width": "100%", "display": "block", "margin": "0 auto"}
                )
                store_figure[store_name] = viz

            return viz, viz_name, store_figure

        elif viz_name == "viz3":
            if store_name in store_figure:
                viz = store_figure[store_name]
            else:
                viz = animated_slices(ref, main, axis=0, fps=2)
                store_figure[store_name] = viz

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
            if store_name in store_figure:
                viz = store_figure[store_name]
            else:
                viz = isosurface_diff(ref, main)
                store_figure[store_name] = viz

            return dcc.Graph(id="vis-graph", figure=viz,
                             style={'width': '100%',
                                    'height': 'calc(100vh - 150px)'}), viz_name, store_figure

        elif viz_name == "viz9":
            if store_name in store_figure:
                viz = store_figure[store_name]
            else:
                viz = hierarchical_diff_visualization(ref, main)
                store_figure[store_name] = viz

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
            if store_name in store_figure:
                viz = store_figure[store_name]
            else:
                viz = tensor_network_visualization(ref, main)
                store_figure[store_name] = viz

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
            if store_name in store_figure:
                viz = store_figure[store_name]
            else:
                viz = channel_correlation_matrices(ref, main)
                store_figure[store_name] = viz

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

        return no_update, no_update, no_update

    @app.callback(
        Output("visualization-modal", "is_open"),
        Output("visualization-output-id", "data"),
        Input({"type": "visualization-button", "index": ALL}, "n_clicks"),
        prevent_initial_call=True
    )
    def open_visualization_modal(n_clicks_list):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        if all(nc is None or nc == 0 for nc in n_clicks_list):
            raise PreventUpdate

        # Get the id of the button that triggered the callback
        try:
            trigger_id, _ = parse_prop_id(ctx.triggered[0]["prop_id"])
            triggered_id = trigger_id
            index = triggered_id["index"]
        except Exception as e:
            print(f"Error parsing trigger_id: {ctx.triggered[0]["prop_id"]}, Error: {str(e)}")
            return False, no_update

        return True, index

    @app.callback(
        Output("clear-queue-store", "data"),
        Input("clear-queue-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_queue(_):
        cache.processing_layers.clear()
        cache.cancel_event.set()

        while True:
            try:
                cache.task_queue.get_nowait()
                cache.task_queue.task_done()
            except Empty:
                break

        # Clear the cancel event after emptying the queue
        # to ensure the first task after clearing is not skipped
        cache.cancel_event.clear()

        return True

    # Helper function to create file browser content
    def create_file_browser_content(path, selected_file=None):
        if not os.path.exists(path):
            return html.Div("Path does not exist")

        # Create the current path display
        path_display = html.Div(
            [
                html.Span(path),
            ],
            style={"marginBottom": "10px"}
        )

        # Get directories and files in the current path
        try:
            items = os.listdir(path)
            directories = [item for item in items if os.path.isdir(os.path.join(path, item))]
            files = [item for item in items if os.path.isfile(os.path.join(path, item))]

            # Sort directories and files alphabetically
            directories.sort()
            files.sort()

            # Create parent directory entry (..)
            parent_directory_item = [
                html.Li(
                    dbc.Button(
                        "📁 ..",
                        id={"type": "file-browser-item", "index": ".."},
                        color="link",
                        style={"textAlign": "left", "width": "100%"}
                    ),
                    style={"marginBottom": "5px"}
                )
            ]

            # Create list items for directories
            directory_items = [
                html.Li(
                    dbc.Button(
                        f"📁 {directory}",
                        id={"type": "file-browser-item", "index": directory},
                        color="link",
                        style={"textAlign": "left", "width": "100%"}
                    ),
                    style={"marginBottom": "5px"}
                )
                for directory in directories
            ]

            # Create list items for files
            file_items = [
                html.Li(
                    dbc.Button(
                        f"📄 {file}",
                        id={"type": "file-browser-item", "index": file},
                        color="link",
                        style={
                            "textAlign": "left",
                            "width": "100%",
                            "backgroundColor": "#007bff" if selected_file and os.path.basename(
                                selected_file) == file else "transparent",
                            "color": "white" if selected_file and os.path.basename(selected_file) == file else "inherit"
                        }
                    ),
                    style={"marginBottom": "5px"}
                )
                for file in files
            ]

            # Combine parent directory, directories, and files
            items_list = html.Ul(
                parent_directory_item + directory_items + file_items,
                style={"listStyleType": "none", "padding": 0, "maxHeight": "400px", "overflowY": "auto"}
            )

            return html.Div([path_display, items_list])

        except Exception as e:
            return html.Div(f"Error: {str(e)}")

    # File browser callbacks
    @app.callback(
        [
            Output("file-browser-modal", "is_open"),
            Output("file-browser-current-path", "data"),
            Output("file-browser-target", "data"),
            Output("file-browser-content", "children"),
            Output("file-browser-mode", "data"),
            Output("file-browser-selected-file", "data"),
            Output("file-browser-header", "children"),
        ],
        [
            Input("browse-ov-bin-path", "n_clicks"),
            Input("browse-model-xml-path", "n_clicks"),
            Input({"type": "browse-model-input", "name": ALL}, "n_clicks"),
            Input("file-browser-select", "n_clicks"),
            Input({"type": "file-browser-item", "index": ALL}, "n_clicks"),
        ],
        [
            State("file-browser-modal", "is_open"),
            State("file-browser-current-path", "data"),
            State("file-browser-target", "data"),
            State("file-browser-mode", "data"),
            State("file-browser-selected-file", "data"),
            State("ov-bin-path", "value"),
            State("model-xml-path", "value"),
            State({"type": "model-input", "name": ALL}, "value"),
            State({"type": "model-input", "name": ALL}, "id"),
        ],
        prevent_initial_call=True,
    )
    def handle_file_browser(
            browse_ov_btn, browse_model_btn, browse_input_btns, select_btn, item_clicks,
            is_open, current_path, target, mode, selected_file, ov_bin_path, model_path, input_paths, input_ids
    ):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update

        trigger_id, _ = parse_prop_id(ctx.triggered[0]["prop_id"])

        # Check if the trigger is actually a button click, not just a property change
        if trigger_id == "browse-ov-bin-path" and browse_ov_btn is not None:
            # Start with the current path if it exists, otherwise use C:\ or user's home
            path = ov_bin_path if ov_bin_path and os.path.exists(ov_bin_path) else os.path.expanduser("~")
            if not os.path.isdir(path):
                path = os.path.dirname(path) if os.path.exists(os.path.dirname(path)) else os.path.expanduser("~")

            return True, path, "ov-bin-path", create_file_browser_content(path), "directory", "", "Select Directory"

        # Handle model XML path browse button
        if trigger_id == "browse-model-xml-path" and browse_model_btn is not None:
            # Start with the current path if it exists, otherwise use user's home
            path = model_path if model_path and os.path.exists(model_path) else os.path.expanduser("~")
            if os.path.isfile(path):
                path = os.path.dirname(path)
            elif not os.path.isdir(path):
                path = os.path.dirname(path) if os.path.exists(os.path.dirname(path)) else os.path.expanduser("~")

            return True, path, "model-xml-path", create_file_browser_content(path), "file", "", "Select File"

        # Handle model input path browse buttons
        if isinstance(trigger_id, dict) and trigger_id.get("type") == "browse-model-input" and any(browse_input_btns):
            input_name = trigger_id.get("name")
            # Find the corresponding input path
            input_path = ""
            for i, input_id in enumerate(input_ids):
                if input_id.get("name") == input_name:
                    input_path = input_paths[i] if i < len(input_paths) else ""
                    break

            # Start with the current path if it exists, otherwise use user's home
            path = input_path if input_path and os.path.exists(input_path) else os.path.expanduser("~")
            if os.path.isfile(path):
                path = os.path.dirname(path)
            elif not os.path.isdir(path):
                path = os.path.dirname(path) if os.path.exists(os.path.dirname(path)) else os.path.expanduser("~")

            target_id = {"type": "model-input", "name": input_name}
            return True, path, target_id, create_file_browser_content(path), "file", "", "Select File"

        # Handle file/directory selection
        if isinstance(trigger_id, dict) or (isinstance(trigger_id, str) and trigger_id.startswith("{")):
            # A file or directory was clicked
            try:
                # Use parse_prop_id to get the item_id
                item_id, _ = parse_prop_id(ctx.triggered[0]["prop_id"])

                # Check if the item_id has the expected structure (has an "index" key)
                if not isinstance(item_id, dict) or "index" not in item_id:
                    # This is not a file browser item click, so skip this section
                    raise PreventUpdate

                item_index = item_id["index"]

                # Special handling for ".." (parent directory)
                if item_index == "..":
                    parent_path = os.path.dirname(current_path)
                    header_text = "Select Directory" if mode == "directory" else "Select File"
                    return True, parent_path, target, create_file_browser_content(
                        parent_path), mode, selected_file, header_text

                # Get the path of the clicked item
                path = os.path.join(current_path, item_index)
            except PreventUpdate:
                raise
            except Exception as e:
                # Print the error for debugging
                print(f"Error parsing trigger_id: {ctx.triggered[0]['prop_id']}, Error: {str(e)}")

                # If we can't parse the JSON, try to parse it again with parse_prop_id
                try:
                    parsed_id, _ = parse_prop_id(trigger_id)
                    if parsed_id == ".." or (isinstance(parsed_id, dict) and parsed_id.get("index") == ".."):
                        parent_path = os.path.dirname(current_path)
                        return True, parent_path, target, create_file_browser_content(parent_path), mode, selected_file
                except:
                    pass
                # If it's not the ".." button and we can't parse the JSON, just return no update
                return no_update, no_update, no_update, no_update, no_update, no_update

            # If it's a directory, navigate to it
            if os.path.isdir(path):
                header_text = "Select Directory" if mode == "directory" else "Select File"
                return True, path, target, create_file_browser_content(path), mode, selected_file, header_text

            # If it's a file and we're in file mode, select it
            if os.path.isfile(path) and mode == "file":
                return True, current_path, target, create_file_browser_content(current_path,
                                                                               path), mode, path, "Select File"

            # If it's a file but we're in directory mode, do nothing
            header_text = "Select Directory" if mode == "directory" else "Select File"
            return True, current_path, target, create_file_browser_content(
                current_path), mode, selected_file, header_text

        # Handle select button
        if trigger_id == "file-browser-select":
            # In directory mode, select the current directory
            if mode == "directory":
                return False, current_path, target, no_update, mode, current_path, no_update

            # In file mode, select the selected file
            if mode == "file" and selected_file:
                return False, current_path, target, no_update, mode, selected_file, no_update

            # No selection
            header_text = "Select Directory" if mode == "directory" else "Select File"
            return True, current_path, target, create_file_browser_content(
                current_path), mode, selected_file, header_text

        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    @app.callback(
        [
            Output("ov-bin-path", "value", allow_duplicate=True),
            Output("model-xml-path", "value", allow_duplicate=True),
            Output({"type": "model-input", "name": ALL}, "value", allow_duplicate=True),
        ],
        Input("file-browser-selected-file", "data"),
        [
            State("file-browser-target", "data"),
            State({"type": "model-input", "name": ALL}, "id"),
        ],
        prevent_initial_call=True,
    )
    def update_path_input(selected_path, target, input_ids):
        if not selected_path or not target:
            raise PreventUpdate

        # For OpenVINO bin path
        if target == "ov-bin-path":
            return selected_path, no_update, [no_update] * len(input_ids)

        # For model XML path
        if target == "model-xml-path":
            return no_update, selected_path, [no_update] * len(input_ids)

        # For model input paths
        if isinstance(target, dict) and target.get("type") == "model-input":
            target_name = target.get("name")
            input_updates = []
            for input_id in input_ids:
                if input_id.get("name") == target_name:
                    input_updates.append(selected_path)
                else:
                    input_updates.append(no_update)
            return no_update, no_update, input_updates

        raise PreventUpdate


def register_clientside_callbacks(app):
    app.clientside_callback(
        """
        function(pathname) {
            function applyPercentagePan() {
                if (window.cy) {
                    const panX = window.innerWidth * 0.30;

                    window.cy.pan({
                        x: panX,
                        y: 0
                    });
                }
            }

            // Apply pan on initial load
            applyPercentagePan();

            // Set up resize handler if not already done
            if (!window.panHandlerInitialized) {
                window.addEventListener('resize', applyPercentagePan);
                window.panHandlerInitialized = true;
            }

            return null;
        }
        """,
        Output('dummy-output', 'data', allow_duplicate=True),
        Input('first-load', 'pathname'),
        prevent_initial_call=True
    )

    # Panel resize functionality
    app.clientside_callback(
        """
        function(dummy) {
            // Initialize the resize functionality only once
            if (!window.panelResizeInitialized) {
                window.panelResizeInitialized = true;

                function initPanelResize() {
                    const leftPanel = document.getElementById('left-panel');
                    const rightPanel = document.getElementById('right-panel');
                    const leftHandle = document.getElementById('left-panel-resize-handle');
                    const rightHandle = document.getElementById('right-panel-resize-handle');

                    if (!leftPanel || !rightPanel || !leftHandle || !rightHandle) {
                        // If elements aren't ready yet, try again later
                        setTimeout(initPanelResize, 100);
                        return;
                    }

                    let isLeftDragging = false;
                    let isRightDragging = false;
                    let startX = 0;
                    let startWidth = 0;

                    // Left panel resize
                    leftHandle.addEventListener('mousedown', function(e) {
                        isLeftDragging = true;
                        startX = e.clientX;
                        startWidth = parseFloat(getComputedStyle(leftPanel).width);
                        document.body.style.cursor = 'col-resize';
                        e.preventDefault();
                    });

                    // Right panel resize
                    rightHandle.addEventListener('mousedown', function(e) {
                        isRightDragging = true;
                        startX = e.clientX;
                        startWidth = parseFloat(getComputedStyle(rightPanel).width);
                        document.body.style.cursor = 'col-resize';
                        e.preventDefault();
                    });

                    document.addEventListener('mousemove', function(e) {
                        if (isLeftDragging) {
                            const width = startWidth + (e.clientX - startX);
                            const minWidth = 150;
                            const maxWidth = window.innerWidth * 0.5;

                            if (width >= minWidth && width <= maxWidth) {
                                leftPanel.style.width = width + 'px';
                            }
                        } else if (isRightDragging) {
                            const width = startWidth - (e.clientX - startX);
                            const minWidth = 150;
                            const maxWidth = window.innerWidth * 0.5;

                            if (width >= minWidth && width <= maxWidth) {
                                rightPanel.style.width = width + 'px';
                            }
                        }
                    });

                    document.addEventListener('mouseup', function() {
                        isLeftDragging = false;
                        isRightDragging = false;
                        document.body.style.cursor = '';
                    });
                }

                // Initialize the resize functionality
                initPanelResize();

                // Re-initialize on window resize
                window.addEventListener('resize', function() {
                    initPanelResize();
                });
            }

            return null;
        }
        """,
        Output('dummy-output', 'data'),
        Input('first-load', 'pathname')
    );

    # Manual layout refresh function
    app.clientside_callback(
        """
        function(trigger) {
            if (window.cy) {
                // Run the layout algorithm manually
                window.cy.layout({
                    name: 'dagre',
                    directed: true,
                    rankDir: 'TB',
                    nodeSep: 25,
                    rankSep: 50,
                    fit: false
                }).run();

                return trigger;
            }
            return trigger;
        }
        """,
        Output('refresh-layout-trigger', 'data', allow_duplicate=True),
        Input('refresh-layout-trigger', 'data'),
        prevent_initial_call=True
    )

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

    # Center on the transformed node after graph transformation
    app.clientside_callback(
        """
        function(transformedNodeName, elements, settingsOpened, visualizationOpened) {
            if (!transformedNodeName || !elements || elements.length === 0) {
                return;
            }

            const isSettingsOpen = settingsOpened ?? false;
            const isVisualizationOpen = visualizationOpened ?? false;
            if (isSettingsOpen || isVisualizationOpen) {
                return;
            }

            if (window.cy) {
                // Find the node with the matching layer_name
                let targetNode = null;
                for (const element of elements) {
                    if (element.data && element.data.layer_name === transformedNodeName) {
                        targetNode = element;
                        break;
                    }
                }

                if (!targetNode) return; // No matching node found

                const nodeId = targetNode.data.id;
                const element = window.cy.getElementById(nodeId);
                if (element.length === 0) return; // Check if node exists in cytoscape

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
        Input('transformed-node-name-store', 'data'),
        State('ir-graph', 'elements'),
        State("inference-settings-modal", "is_open"),
        State("visualization-modal", "is_open"),
    )
