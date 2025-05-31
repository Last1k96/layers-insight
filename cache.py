from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor

from dash import html
import dash_bootstrap_components as dbc

from metrics import comparison_metrics_table
from run_inference import run_partial_inference

# Shared resources
result_cache = {}
status_cache = {}
processing_layers = {}
task_queue = Queue()
cancel_event = threading.Event()

# Global variables to avoid race conditions with Dash dcc.Store elements
# when the same element is used as both Output and State in callbacks
layers_store_data = []
ir_graph_elements = {}

lock = threading.Lock()


# Start a background processing thread
def process_tasks():
    while True:
        task = task_queue.get()  # blocking
        if task is None:  # graceful shutdown
            break

        # If the button was pressed *before* we started this task,
        # just skip it and reset the flag for the next loop.
        if cancel_event.is_set():
            cancel_event.clear()
            task_queue.task_done()
            continue

        node_id, layer_name, layer_type, config, plugins_config = task

        outputs = []
        result = {"node_id": node_id, "layer_name": layer_name, "layer_type": layer_type}

        try:
            outputs = run_partial_inference(
                openvino_bin=config.get("ov_bin_path"),
                model_xml=config.get("model_xml"),
                layer_name=layer_name,
                ref_plugin=config.get("plugin1"),
                main_plugin=config.get("plugin2"),
                model_inputs=config.get("model_inputs", []),
                seed=config["output_folder"],
                plugins_config=plugins_config,
                cancel_event=cancel_event
            )

            # Create comparison metrics table for successful inference
            right_panel_div = html.Div([
                dbc.CardGroup([
                    comparison_metrics_table(output["ref"], output["main"], idx),
                ], style={"marginLeft": "8px"})
                for idx, output in enumerate(outputs)
            ])
            status_cache[node_id] = right_panel_div
            result["outputs"] = outputs

        except RuntimeError as e:
            if str(e) == "Inference cancelled":
                # Skip this task if inference was cancelled
                continue
        except Exception as e:
            print(e)
            result["error"] = str(e)

        # Result dictionary is already populated with node_id, layer_name, layer_type
        # and outputs (if successful)

        with lock:
            result_cache[node_id] = result

        task_queue.task_done()
