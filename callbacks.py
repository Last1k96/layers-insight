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
from colors import BORDER_COLORS, BorderColorType

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


def update_node_style(elements, node_id, color):
    for element in elements:
        if element['data'].get('id') == node_id:
            element['data']['border_color'] = color

    return elements


def set_selected_node_style(elements, node_id):
    # Get the type of the selected node
    selected_type = None
    for element in elements:
        if element["data"].get("id") == node_id:
            selected_type = element["data"].get("type")
            break

    # Set classes based on node type
    for element in elements:
        if element["data"].get("id") == node_id:
            element["classes"] = "selected"
        elif selected_type and element["data"].get("type") != selected_type:
            element["classes"] = "selected-different-type"
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
        Output('ir-graph', 'stylesheet'),
        Input('just-finished-tasks-store', 'data'),
        Input('selected-layer-index-store', 'data'),
        Input('restart-layer-button', 'n_clicks'),
        Input('ir-graph', 'tapNode'),
        State('ir-graph', 'stylesheet'),
        State('ir-graph', 'elements'),
        prevent_initial_call=True
    )
    def update_node_stylesheet(finished_nodes, selected_layer_index, restart_layer_btn, tap_node, current_stylesheet, elements):
        ctx = callback_context
        if not ctx.triggered:
            return no_update

        # Start with the base stylesheet
        if current_stylesheet is None:
            # If no stylesheet is provided, use a default one
            stylesheet = build_dynamic_stylesheet(cache.ir_graph_elements)
        else:
            # Use the current stylesheet as a base
            stylesheet = current_stylesheet

        # Update node border colors based on elements' data attributes
        for element in elements:
            if 'data' in element and 'id' in element['data'] and 'border_color' in element['data']:
                node_id = element['data']['id']
                color = element['data']['border_color']

                # Check if a selector for this node already exists
                node_selector = f'node[id = "{node_id}"]'

                # Find if this selector already exists in the stylesheet
                selector_exists = False
                for style in stylesheet:
                    if style.get('selector') == node_selector:
                        # Update the existing style
                        style['style']['border-color'] = color
                        selector_exists = True
                        break

                # If the selector doesn't exist, add a new style
                if not selector_exists:
                    stylesheet.append({
                        'selector': node_selector,
                        'style': {
                            'border-color': color
                        }
                    })

        # Update the selected node style
        if hasattr(cache, 'selected_node_id') and cache.selected_node_id is not None:
            # Add styles for the selected node classes
            selected_class_exists = False
            selected_different_type_class_exists = False

            for style in stylesheet:
                if style.get('selector') == 'node.selected':
                    selected_class_exists = True
                if style.get('selector') == 'node.selected-different-type':
                    selected_different_type_class_exists = True
                if selected_class_exists and selected_different_type_class_exists:
                    break

            if not selected_class_exists:
                stylesheet.append({
                    'selector': 'node.selected',
                    'style': {
                        'background-color': BORDER_COLORS[BorderColorType.ERROR],
                        'z-index': 9999  # Ensure selected node is on top
                    }
                })

            if not selected_different_type_class_exists:
                stylesheet.append({
                    'selector': 'node.selected-different-type',
                    'style': {
                        'background-color': BORDER_COLORS[BorderColorType.SELECTED_DIFFERENT_TYPE],
                        'z-index': 9998  # Just below the selected node
                    }
                })

            # We don't need to add a style for the specific selected node
            # because the class selector will handle it, and the class is
            # applied to the node in the set_selected_node_style function

        return stylesheet

    @app.callback(
        Output('ir-graph', 'layout'),
        Output('layout-reset-interval', 'disabled'),
        Output('layout-reset-interval', 'n_intervals'),
        Input('model-path-after-cut', 'data'),
        prevent_initial_call=True
    )
    def refresh_graph_layout(model_path_after_cut):
        """
        Manually refresh the graph layout when the model is modified.
        This is needed because autoRefreshLayout is set to False to prevent
        unnecessary layout recalculations during normal operation.
        """
        if not model_path_after_cut:
            return no_update, no_update, no_update, no_update

        import time
        timestamp = int(time.time())  # Current Unix timestamp in seconds

        layout = {
            'name': 'dagre',
            'directed': True,
            'rankDir': 'TB',
            'nodeSep': 25,
            'rankSep': 50,
            'fit': True,  # Change to True to force a complete layout refresh
            'animate': True,  # Add animation to make the refresh more visible
            'randomize': timestamp,  # Use timestamp to ensure the layout is seen as new
            'refresh': True,  # Explicitly request a refresh
        }

        # Enable the interval component to trigger a layout reset after a delay
        # Also reset the interval component's n_intervals to 0
        return layout, False, 0

    @app.callback(
        Output('ir-graph', 'layout', allow_duplicate=True),
        Output('layout-reset-interval', 'disabled', allow_duplicate=True),
        Input('layout-reset-interval', 'n_intervals'),
        prevent_initial_call=True
    )
    def reset_layout_after_delay(n_intervals):
        """
        Reset the layout to its original configuration after a delay.
        This ensures that the graph is properly redrawn and then returns to its original state.
        """
        if n_intervals is None or n_intervals < 1:
            return no_update, no_update

        print(f"[DEBUG] Resetting graph layout after delay (n_intervals: {n_intervals})")

        # Return the original layout configuration and disable the interval
        layout = {
            'name': 'dagre',
            'directed': True,
            'rankDir': 'TB',
            'nodeSep': 25,
            'rankSep': 50,
            'fit': False,  # Reset to False to match the original configuration
        }

        return layout, True

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
        Input('model-path-after-cut', 'data'),
        prevent_initial_call=True
    )
    def clear_cache(model_after_cut):
        with cache.lock:
            cache.result_cache.clear()
            cache.status_cache.clear()
            cache.processing_layers.clear()
            cache.layers_store_data.clear()

        elements = parse_openvino_ir(model_after_cut)

        cache.ir_graph_elements = elements
        return elements

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
                trigger.startswith('clear-queue-store') for trigger in triggers) or any(
                trigger.startswith('model-path-after-cut') for trigger in triggers):
            for element in elements:
                node_id = element['data'].get("id")

                if node_id in cache.result_cache:
                    result = cache.result_cache[node_id]
                    if "error" in result:
                        element['data']['border_color'] = BORDER_COLORS[BorderColorType.ERROR]
                    else:
                        element['data']['border_color'] = BORDER_COLORS[BorderColorType.SUCCESS]
                elif node_id in cache.processing_layers:
                    element['data']['border_color'] = BORDER_COLORS[BorderColorType.PROCESSING]
                else:
                    element['data']['border_color'] = BORDER_COLORS[BorderColorType.DEFAULT]

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
                update_node_style(new_elements, node_id, BORDER_COLORS[BorderColorType.SELECTED])

            set_selected_node_style(new_elements, node_id)

        if any(trigger.startswith('just-finished-tasks-store') for trigger in triggers):
            for element in new_elements:
                node_id = element['data'].get("id")
                if node_id in finished_nodes:
                    result = cache.result_cache[node_id]
                    color = BORDER_COLORS[BorderColorType.ERROR] if "error" in result else BORDER_COLORS[BorderColorType.SUCCESS]
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
            update_node_style(new_elements, node_id, BORDER_COLORS[BorderColorType.SELECTED])

        return new_elements

    @app.callback(
        Output('layers-store', 'data'),
        Output('clicked-graph-node-id-store', 'data'),
        Input('first-load', 'pathname'),
        Input('ir-graph', 'tapNode'),
        Input('just-finished-tasks-store', 'data'),
        Input('restart-layer-button', 'n_clicks'),
        Input('model-path-after-cut', 'data'),
        Input("clear-queue-store", "data"),
        State('selected-node-id-store', 'data'),
        prevent_initial_call=True
    )
    def update_layers_list(_, tap_node, finished_nodes, restart_layers_btn, model_after_cut, selected_node_id,
                           clear_queue_btn):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        triggers = [t['prop_id'] for t in ctx.triggered]
        print(f"{triggers=}")

        if any(trigger.startswith('model-path-after-cut') for trigger in triggers):
            return [], None

        if any(trigger.startswith('clear-queue-store') for trigger in triggers):
            cache.processing_layers.clear()
            # Only remove tasks with "running" status, keep finished tasks
            cache.layers_store_data = [layer for layer in cache.layers_store_data if layer.get("status") != "running"]
            return cache.layers_store_data, None

        if any(trigger.startswith('first-load') for trigger in triggers) or any(
                trigger.startswith('clear-queue-store') for trigger in triggers):
            layer_list_out = []

            for node_id, result in cache.result_cache.items():
                if "error" in result:
                    layer_list_out.append({
                        "node_id": node_id,
                        "layer_name": result["layer_name"],
                        "layer_type": result["layer_type"],
                        "status": "error"
                    })
                else:
                    layer_list_out.append({
                        "node_id": node_id,
                        "layer_name": result["layer_name"],
                        "layer_type": result["layer_type"],
                        "status": "done"
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
            return layer_list_out, no_update

        layer_list_out = cache.layers_store_data
        clicked_graph_node_id = no_update

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

        if any(trigger.startswith('restart-layer-button') for trigger in triggers):  # and finished_nodes
            for layer in layer_list_out:
                node_id = layer["node_id"]
                if node_id == selected_node_id:
                    layer["status"] = "running"

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

        print(f"{len(cache.result_cache)=}")
        print(f"{cache.processing_layers=}")
        print(f"{cache.status_cache=}")
        print(f"{cache.layers_store_data=}")
        print()

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
                    id_part = first_trigger.split('.')[0]
                    triggered_id = json.loads(id_part)
                    new_index = triggered_id.get("index")
                except Exception:
                    pass

        return new_index

    @app.callback(
        Output('right-panel-layer-name', 'children'),
        Output('right-panel', 'children'),
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
        prevent_initial_call=True
    )
    def save_config(save_btn_clicks, open_settings_btn_clicks, config_after_cut, config, model_xml, bin_path,
                    ref_plugin,
                    other_plugin, all_input_values):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        triggers = [t['prop_id'] for t in ctx.triggered]

        if any(trigger.startswith('inference-settings-btn') for trigger in triggers):
            model_inputs = read_openvino_ir(config["model_xml"])
            inputs_layout_div = build_model_input_fields(model_inputs, config["model_inputs"])
            return True, no_update, config["model_xml"], config["ov_bin_path"], None, None, config["plugin1"], config[
                "plugin2"], inputs_layout_div

        if any(trigger.startswith('save-inference-config-button') for trigger in triggers):
            plugin1 = ref_plugin if ref_plugin is not None else config["plugin1"]
            plugin2 = other_plugin if other_plugin is not None else config["plugin2"]

            update_config(
                config,
                model_xml,
                bin_path,
                plugin1,
                plugin2,
                all_input_values
            )

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
        # Save only the inputs that belong to the currently selected plugin.
        current_config = {
            comp_id["param"]: value
            for value, comp_id in zip(values, ids)
            if comp_id.get("plugin") == selected_plugin
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
        triggered_id = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
        index = triggered_id["index"]

        return True, index

    @app.callback(
        Output("clear-queue-store", "data"),
        Input("clear-queue-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_queue(_):
        # cache.processing_layers.clear()
        cache.cancel_event.set()

        while True:
            try:
                cache.task_queue.get_nowait()
                cache.task_queue.task_done()
            except Empty:
                break

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
