import dash
import dash_cytoscape as cyto

from layout import create_layout
from callbacks import register_callbacks

from run_inference import run_partial_inference

from cache import result_cache, task_queue, lock, executor
import threading

def create_app(openvino_path, ir_xml_path, inputs_path):
    cyto.load_extra_layouts()
    app = dash.Dash(title="Layers Insight")

    # Start background processing thread
    def process_tasks():
        while True:
            task = task_queue.get()
            if task is None:
                break
            try:
                node_id, openvino_bin, model_xml, ref_plugin, main_plugin, input_path = task
                result = run_partial_inference(openvino_bin, model_xml, node_id, ref_plugin, main_plugin, input_path)
                with lock:
                    result_cache[node_id] = result
            except Exception as e:
                with lock:
                    result_cache[node_id] = f"Error: {str(e)}"
            finally:
                task_queue.task_done()

    threading.Thread(target=process_tasks, daemon=True).start()

    app.layout = create_layout(openvino_path, ir_xml_path, inputs_path)
    register_callbacks(app)

    return app

