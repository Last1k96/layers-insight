from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor

from run_inference import run_partial_inference

# Shared resources
result_cache = {}
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

# Start background processing thread
def process_tasks():
    while True:
        task = task_queue.get() # blocking get()
        if task is None:
            break
        try:
            node_id, layer_name, layer_type, config_data = task

            openvino_bin = config_data.get("ov_bin_path")
            model_xml = config_data.get("model_xml")
            ref_plugin = config_data.get("plugin1")
            main_plugin = config_data.get("plugin2")
            model_inputs = config_data.get("model_inputs", [])

            result = run_partial_inference(
                openvino_bin=openvino_bin,
                model_xml=model_xml,
                layer_name=layer_name,
                ref_plugin=ref_plugin,
                main_plugin=main_plugin,
                model_inputs=model_inputs
            )

            result["node_id"] = node_id
            result["layer_name"] = layer_name
            result["layer_type"] = layer_type

            with lock:
                result_cache[node_id] = result

        except Exception as e:
            with lock:
                result_cache[node_id] = f"Error: {str(e)}" # TODO better errors, replace the cache on re-run
        finally:
            task_queue.task_done()