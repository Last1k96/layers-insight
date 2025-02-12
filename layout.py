import os
import random
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
from openvino_graph import parse_openvino_ir
from known_ops import OPENVINO_OP_COLORS_DARK
import dash_bootstrap_components as dbc
from dash_split_pane import DashSplitPane

from run_inference import get_available_plugins
from callbacks import update_config


def build_dynamic_stylesheet(elements):
    # Base node/edge styles
    stylesheet = [
        {
            'selector': 'node',
            'style': {
                'content': 'data(display_label)',
                'width': 'label',
                'height': 'label',
                'shape': 'round-rectangle',
                'border-width': '2',
                'border-color': 'data(border_color)',
                'padding': '6px',
                'font-size': '12px',
                'font-family': 'sans-serif',
                'text-valign': 'center',
                'text-halign': 'center',
                'background-color': 'primary',
                'background-width': '2',
                'color': '#fff',
            }
        },
        {
            'selector': 'edge',
            'style': {
                'width': 2,
                'line-color': '#888',
                'target-arrow-color': '#888',
                'target-arrow-shape': 'triangle',
                'curve-style': 'straight',
                'label': 'data(display_label)',
                'color': '#DCDCDC',

                'text-outline-opacity': '0.8',
                'text-outline-width': '2',
                'text-outline-color': '#222',

                'text-background-color': '#888',
                'text-background-opacity': '0',
                'text-background-shape': 'round-rectangle',
                'text-background-padding': '2px',
                'font-family': 'sans-serif',
                'font-size': '8',
            }
        }
    ]

    # Collect distinct op types
    op_types = set()
    for el in elements:
        if 'data' in el and 'type' in el['data']:
            op_types.add(el['data']['type'])

    used_random_colors = {}
    for op_type in op_types:
        # Check if op_type is in OPENVINO_OP_COLORS_DARK
        if op_type in OPENVINO_OP_COLORS_DARK:
            color = OPENVINO_OP_COLORS_DARK[op_type]
        else:
            # Fallback: generate random subdued color
            if op_type not in used_random_colors:
                r = random.randint(40, 160)
                g = random.randint(40, 160)
                b = random.randint(40, 160)
                used_random_colors[op_type] = f"#{r:02X}{g:02X}{b:02X}"
            color = used_random_colors[op_type]

        # Add a rule for that op_type
        stylesheet.append({
            'selector': f'node[type="{op_type}"]',
            'style': {
                'background-color': color
            }
        })

    return stylesheet


def read_openvino_ir(model_xml_path):
    from openvino import Core
    core = Core()
    model = core.read_model(model_xml_path)  # Read the IR model from disk

    inputs_info = []
    for input_node in model.inputs:
        input_name = input_node.get_any_name()  # Get the input tensor name
        input_shape = list(input_node.shape)  # Convert shape to a Python list
        inputs_info.append({"name": input_name, "shape": input_shape})

    return inputs_info


def build_model_input_fields(model_path, inputs_path):
    model_inputs = read_openvino_ir(model_path)

    if len(inputs_path) == 0 or len(inputs_path) != len(model_inputs):
        inputs_path = [""] * len(model_inputs)

    components = []
    for index, (model_input, input_path) in enumerate(zip(model_inputs, inputs_path), start=1):
        name = model_input["name"]
        shape = model_input["shape"]

        components.append(
            dbc.Label(f"Input #{index}: '{name}' with shape {shape}")
        )
        # Use a *pattern-matching* dictionary for `id`
        components.append(
            dbc.Input(
                id={"type": "model-input", "name": name},  # <--- Pattern-matching ID
                type="text",
                placeholder=f"Enter path for {name}",
                value=input_path,
            )
        )
        components.append(html.Br())

    return html.Div(components)


def create_layout(openvino_path, ir_xml_path, inputs_path):
    elements = parse_openvino_ir(ir_xml_path)
    dynamic_stylesheet = build_dynamic_stylesheet(elements)

    if openvino_path and os.path.exists(openvino_path):
        discovered_plugins = get_available_plugins(openvino_path)
    else:
        discovered_plugins = []

    if "CPU" in discovered_plugins:
        plugin1_value = "CPU"
    else:
        plugin1_value = discovered_plugins[0] if discovered_plugins else None

    non_cpu_plugins = [p for p in discovered_plugins if p != "CPU"]
    plugin2_value = non_cpu_plugins[0] if non_cpu_plugins else None

    initial_config = {}
    update_config(
        initial_config,
        ir_xml_path,
        openvino_path,
        plugin1_value,
        plugin2_value,
        inputs_path
    )

    config_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Inference Configuration")),
            dbc.ModalBody(
                [
                    # Model XML Path
                    dbc.Label("Path to model.xml"),
                    dbc.Input(id="model-xml-path", value=ir_xml_path, placeholder="Enter path to model.xml"),
                    html.Br(),

                    # Input files (example: two separate fields)
                    build_model_input_fields(ir_xml_path, inputs_path),

                    # OpenVINO bin folder
                    dbc.Label("Path to OpenVINO bin folder"),
                    dbc.Input(id="ov-bin-path", value=openvino_path, placeholder="Path to OpenVINO bin/ folder"),
                    html.Br(),

                    # Find Plugins Button
                    dbc.Button("Find Plugins", id="find-plugins-button", color="primary", n_clicks=0),
                    html.Br(),
                    html.Br(),

                    # Two Dropdowns for plugin selection
                    dbc.Label("Reference Plugin"),
                    dcc.Dropdown(
                        id="reference-plugin-dropdown",
                        options=discovered_plugins,
                        value=plugin1_value,
                        clearable=False,
                    ),
                    html.Br(),
                    dbc.Label("Main Plugin"),
                    dcc.Dropdown(
                        id="main-plugin-dropdown",
                        options=discovered_plugins,
                        value=plugin2_value,
                        clearable=False,
                    ),
                ]
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
            ),
        ],
        id="config-modal",
        is_open=False,  # Modal starts closed
    )

    open_button = dbc.Button(
        "⚙",
        id="open-modal",
        color="primary",
        n_clicks=0,
        style={"position": "absolute", "top": "20px", "left": "20px"}
    )

    plugin_store = dcc.Store(id="plugin-store", data=discovered_plugins)
    config_store = dcc.Store(id="config-store", data=initial_config)
    return html.Div(
        [
            # Resizable panel, positioned on top of the graph.
            DashSplitPane(
                split="vertical",
                size="80%",  # use 'size' instead of 'defaultSize'
                children=[
                    # First pane: e.g. your Cytoscape component
                    cyto.Cytoscape(
                        id='ir-graph',
                        elements=elements,
                        style={"width": "100%", "height": "100%", 'color': 'primary'},
                        layout={
                            'name': 'dagre',
                            'directed': True,
                            'rankDir': 'TB',
                            'nodeSep': 25,
                            'rankSep': 50,
                        },
                        autoungrabify=True,
                        wheelSensitivity=0.2,
                        stylesheet=dynamic_stylesheet
                    ),
                    # Second pane: your right panel
                    html.Div([
                        html.H3(id="layer-name", children=["Layer Name"]),
                        html.Div(
                            "Panel Content",
                            id='right-panel',
                            style={'padding': '10px', 'height': '100%', 'overflow': 'auto', 'color': 'secondary'},
                        )
                    ])

                ]
            ),
            # Other components (if needed) can be positioned as desired.
            open_button,
            config_modal,
            plugin_store,
            config_store,
            dcc.Interval(id='update-interval', interval=500, n_intervals=0),
        ],
        className="main-container",
        # style={
        #     "position": "relative",
        #     "width": "100vw",
        #     "height": "100vh",
        #     "background-color": "#404040"
        # }
    )

