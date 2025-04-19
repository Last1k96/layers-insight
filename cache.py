import multiprocessing as mp
import queue
import threading
import time
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

from dash import html
import dash_bootstrap_components as dbc

from metrics import comparison_metrics_table
from run_inference import run_partial_inference

# Shared resources
status_cache: dict = {}
result_cache: dict = {}
processing_layers = {}

task_queue: "queue.Queue[tuple]" = queue.Queue()
current_proc: mp.Process | None = None
lock = threading.Lock()

# The problem with Dash dcc.Store elements is when you have Output and State having the same element
# the State object could have stale data if the callback for reading the State is enqueued right after
# the state was updated with Output. State captures the data as the callback was created, thus if it was
# created while you were updating the State, you will lose the updated state in the next callback.
# Use global variables for cases where we are both reading the State and Output-ing the same state in the same callback
layers_store_data = []
ir_graph_elements = {}

def _run_task_in_subprocess(task: tuple, result_q: mp.Queue):
    """Runs inside *its own* process."""
    node_id, layer_name, layer_type, cfg, plug_cfg = task
    try:
        outputs = run_partial_inference(
            model_xml=cfg["model_xml"],
            layer_name=layer_name,
            ref_plugin=cfg["plugin1"],
            main_plugin=cfg["plugin2"],
            model_inputs=cfg.get("model_inputs", []),
            openvino_bin=cfg.get("ov_bin_path"),
            seed=cfg.get("output_folder"),
            plugins_config=plug_cfg,
        )
        result_q.put(("ok", node_id, layer_name, layer_type, outputs))
    except Exception as exc:  # propagate any error back to the main process
        result_q.put(("err", node_id, layer_name, layer_type, str(exc)))


# ── 3.  Background thread that feeds tasks to subprocesses ───────────────────
def process_tasks() -> None:
    """Forever pulls tasks off the queue and runs each in its own process."""
    global current_proc
    while True:
        task = task_queue.get()          # blocks until a task is available
        if task is None:                 # sentinel => shut the thread down
            break

        result_q: mp.Queue = mp.Queue()
        current_proc = mp.Process(
            target=_run_task_in_subprocess, args=(task, result_q)
        )
        current_proc.start()

        # Wait until the subprocess signals completion or error
        status, node_id, layer_name, layer_type, payload = result_q.get()
        current_proc.join()              # reap the child process
        current_proc = None

        # ─ Update your caches exactly as you did before ──────────────────
        with lock:
            if status == "ok":
                outputs = payload
                status_cache[node_id] = html.Div(
                    [dbc.Card("Success")], style={"marginLeft": "8px"}
                )
                result_cache[node_id] = {
                    "node_id": node_id,
                    "layer_name": layer_name,
                    "layer_type": layer_type,
                    "outputs": outputs,
                }
            else:
                exception_str = payload
                status_cache[node_id] = html.Div(
                    [dbc.Card(f"Error: {exception_str}")],
                    style={"marginLeft": "8px"},
                )
                result_cache[node_id] = {
                    "node_id": node_id,
                    "layer_name": layer_name,
                    "layer_type": layer_type,
                    "error": exception_str,
                    "outputs": [],
                }

        task_queue.task_done()


# Kick off the background worker
threading.Thread(target=process_tasks, daemon=True).start()
