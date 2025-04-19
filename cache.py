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

# The problem with Dash dcc.Store elements is when you have Output and State having the same element
# the State object could have stale data if the callback for reading the State is enqueued right after
# the state was updated with Output. State captures the data as the callback was created, thus if it was
# created while you were updating the State, you will lose the updated state in the next callback.
# Use global variables for cases where we are both reading the State and Output-ing the same state in the same callback
layers_store_data = []
ir_graph_elements = {}

lock = threading.Lock()


# Start a background processing thread
def process_tasks():
    while True:
        task = task_queue.get()  # blocking get()
        if task is None:
            break

        node_id, layer_name, layer_type, config = task

        exception_str = ""
        outputs = []

        try:
            outputs = run_partial_inference(
                openvino_bin=config.get("ov_bin_path"),
                model_xml=config.get("model_xml"),
                layer_name=layer_name,
                ref_plugin=config.get("plugin1"),
                main_plugin=config.get("plugin2"),
                model_inputs=config.get("model_inputs", []),
                seed=config["output_folder"]
            )
        except Exception as e:
            exception_str = str(e)
            print(e)

        result = {}

        if exception_str:
            result = {"error": exception_str}
        else:
            right_panel_div = html.Div([
                dbc.CardGroup([
                    comparison_metrics_table(output["ref"], output["main"], idx),
                ], style={"marginLeft": "8px"})
                for idx, output in enumerate(outputs)
            ])
            status_cache[node_id] = right_panel_div

        result["node_id"] = node_id
        result["layer_name"] = layer_name
        result["layer_type"] = layer_type
        result["outputs"] = outputs

        with lock:
            result_cache[node_id] = result

        task_queue.task_done()
