from email.policy import default

import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import os, sys

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 1. Configuration Modal
# --------------------------------------------------------------------
config_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("OpenVINO Configuration")),
        dbc.ModalBody(
            [
                # Model XML Path
                dbc.Label("Path to model.xml"),
                dbc.Input(type="text", id="model-xml-path", placeholder="Enter path to model.xml"),
                html.Br(),

                # Input files (example: two separate fields)
                dbc.Label("Input File #1"),
                dbc.Input(type="text", id="input-file-1", placeholder="Path to input file #1"),
                html.Br(),
                dbc.Label("Input File #2"),
                dbc.Input(type="text", id="input-file-2", placeholder="Path to input file #2"),
                html.Br(),

                # OpenVINO bin folder
                dbc.Label("Path to OpenVINO bin folder"),
                dbc.Input(type="text", id="ov-bin-path", placeholder="Path to OpenVINO bin/ folder", value="/home/mkurin/code/openvino/bin/intel64/Release"),
                html.Br(),

                # Find Plugins Button
                dbc.Button("Find Plugins", id="find-plugins-button", color="primary", n_clicks=0),
                html.Br(),
                html.Br(),

                # Two Dropdowns for plugin selection
                dbc.Label("Plugin #1"),
                dcc.Dropdown(
                    id="plugin1-dropdown",
                    options=[],
                    value=None,
                    placeholder="Select plugin #1"
                ),
                html.Br(),
                dbc.Label("Plugin #2"),
                dcc.Dropdown(
                    id="plugin2-dropdown",
                    options=[],
                    value=None,
                    placeholder="Select plugin #2"
                ),
            ]
        ),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
        ),
    ],
    id="config-modal",
    is_open=False,
    size="lg",  # you can adjust modal size if needed
)

# 2. Hidden Stores
# --------------------------------------------------------------------
# Store for discovered plugins (updated after "Find Plugins" is clicked)
plugin_store = dcc.Store(id="plugin-store", data=[])

# Store for final configuration parameters
config_store = dcc.Store(id="config-store", data={})

# 3. Layout
# --------------------------------------------------------------------
open_button = dbc.Button(
    "Open Config",
    id="open-modal",
    color="primary",
    n_clicks=0,
    style={"position": "absolute", "top": "20px", "left": "20px"}
)

app.layout = html.Div(
    [
        open_button,
        config_modal,
        plugin_store,
        config_store,
        html.Div("Main application content goes here!", style={"margin-top": "80px"}),
        # Display stored config for debugging/demo
        html.Hr(),
        html.H4("Current Configuration:"),
        html.Pre(id="display-config", style={"whiteSpace": "pre-wrap"}),
    ],
    style={"position": "relative", "height": "100vh"}
)

# 4. Callbacks
# --------------------------------------------------------------------

# (A) Toggle modal open/close
@app.callback(
    Output("config-modal", "is_open"),
    [Input("open-modal", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("config-modal", "is_open")],
)
def toggle_modal(open_clicks, close_clicks, modal_is_open):
    if open_clicks or close_clicks:
        return not modal_is_open
    return modal_is_open

def import_local_openvino(openvino_bin):
    python_dir = os.path.join(openvino_bin, "python")
    if os.path.isdir(python_dir):
        if python_dir not in sys.path:
            sys.path.insert(0, python_dir)
    else:
        print(f"Warning: Python directory '{python_dir}' not found in the OpenVINO bin folder.")

def get_ov_core(openvino_bin):
    import_local_openvino(openvino_bin)

    import openvino as ov
    core = ov.Core()

    template_plugin = f"{openvino_bin}/libopenvino_template_plugin.so"
    if os.path.exists(template_plugin):
        core.register_plugin(template_plugin, "TEMPLATE")

    return ov, core


def get_available_plugins(openvino_bin):
    ov, core = get_ov_core(openvino_bin)

    return core.available_devices

# (B) Find plugins callback
@app.callback(
    [
        Output("plugin-store", "data"),
        Output("plugin1-dropdown", "options"),
        Output("plugin2-dropdown", "options"),
        Output("plugin1-dropdown", "value"),
        Output("plugin2-dropdown", "value"),
    ],
    Input("find-plugins-button", "n_clicks"),
    State("ov-bin-path", "value"),
    prevent_initial_call=True
)
def find_plugins(n_clicks, ov_bin_path):
    if not ov_bin_path or not os.path.exists(ov_bin_path):
        # Return empty plugin list or handle error
        return [], [], [], None, None

    discovered_plugins = get_available_plugins(ov_bin_path)
    plugin_options = [{"label": p, "value": p} for p in discovered_plugins]

    # Default for plugin #1
    if "CPU" in discovered_plugins:
        plugin1_value = "CPU"
    else:
        plugin1_value = discovered_plugins[0] if discovered_plugins else None

    # Default for plugin #2 (non-CPU if possible)
    non_cpu_plugins = [p for p in discovered_plugins if p != "CPU"]
    plugin2_value = non_cpu_plugins[0] if non_cpu_plugins else None

    return [
        discovered_plugins,  # plugin-store.data
        plugin_options,  # plugin1-dropdown.options
        plugin_options,  # plugin2-dropdown.options
        plugin1_value,  # plugin1-dropdown.value
        plugin2_value,  # plugin2-dropdown.value
    ]


# (D) Save configuration to config_store on modal close
@app.callback(
    Output("config-store", "data"),
    Input("close-modal", "n_clicks"),
    [
        State("model-xml-path", "value"),
        State("input-file-1", "value"),
        State("input-file-2", "value"),
        State("ov-bin-path", "value"),
        State("plugin1-dropdown", "value"),
        State("plugin2-dropdown", "value"),
        State("config-store", "data"),
    ],
    prevent_initial_call=True
)
def save_config(n_clicks_close, model_xml, input_file_1, input_file_2, ov_bin_path,
                plugin1, plugin2, current_data):
    """
    When the user closes the modal, store the current config in the config-store.
    """
    updated_data = current_data.copy()
    updated_data["model_xml"] = model_xml
    updated_data["input_file_1"] = input_file_1
    updated_data["input_file_2"] = input_file_2
    updated_data["ov_bin_path"] = ov_bin_path
    updated_data["plugin1"] = plugin1
    updated_data["plugin2"] = plugin2
    return updated_data

# (E) Display current config
@app.callback(
    Output("display-config", "children"),
    Input("config-store", "data")
)
def display_config(data):
    """Show config as a JSON-ish string in the UI for demo purposes."""
    return str(data)

if __name__ == "__main__":
    app.run_server(debug=True)