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
            layer_name, openvino_bin, model_xml, ref_plugin, main_plugin, input_path = task
            result = run_partial_inference(openvino_bin, model_xml, layer_name, ref_plugin, main_plugin, input_path)
            with lock:
                result_cache[layer_name] = result
        except Exception as e:
            with lock:
                result_cache[layer_name] = f"Error: {str(e)}"
        finally:
            task_queue.task_done()