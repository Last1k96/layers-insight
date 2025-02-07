import uuid

import dash
import dash_cytoscape as cyto

from layout import create_layout
from callbacks import register_callbacks

workspaces = {}

def create_new_workspace(label, model_xml, model_bin, input_file, ref_plugin, other_plugin):
    """
    Creates a unique workspace entry and returns the new workspace ID.
    """
    wksp_id = str(uuid.uuid4())  # unique ID
    workspaces[wksp_id] = {
        "label": label,  # user-friendly name
        "model_xml_path": model_xml,
        "model_bin_path": model_bin,
        "input_path": input_file,
        "ref_plugin": ref_plugin,
        "other_plugin": other_plugin,
        # If you want to store partial inference results in memory:
        "results": {
            ref_plugin: {},
            other_plugin: {}
        }
    }
    return wksp_id


def delete_workspace(wksp_id):
    """
    Safely delete a workspace from the global dict.
    """
    if wksp_id in workspaces:
        del workspaces[wksp_id]


def create_app(openvino_path, ir_xml_path):
    cyto.load_extra_layouts()
    app = dash.Dash(title="Layers Insight")

    app.layout = create_layout(openvino_path, ir_xml_path)
    register_callbacks(app)

    return app

