from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor

from run_inference import run_partial_inference

# Shared resources
result_cache = {}
processing_nodes = set()
task_queue = Queue()
lock = threading.Lock()


# Start background processing thread
def process_tasks():
    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            # task is now (layer_name, config_data)
            node_id, layer_name, config_data = task

            openvino_bin = config_data.get("ov_bin_path")
            model_xml = config_data.get("model_xml")
            ref_plugin = config_data.get("plugin1")
            main_plugin = config_data.get("plugin2")
            # This could be a list of input paths or a dict {input_name: path}
            model_inputs = config_data.get("model_inputs", [])

            # Call your partial inference function
            # Modify run_partial_inference(...) to accept model_inputs
            result = run_partial_inference(
                openvino_bin=openvino_bin,
                model_xml=model_xml,
                layer_name=layer_name,
                ref_plugin=ref_plugin,
                main_plugin=main_plugin,
                model_inputs=model_inputs
            )

            result["node_id"] = node_id

            with lock:
                result_cache[layer_name] = result

        except Exception as e:
            with lock:
                result_cache[layer_name] = f"Error: {str(e)}" # TODO better errors, replace the cache on re-run
        finally:
            task_queue.task_done()