import os
import random
from dash import dcc, html, Input, Output, State
from dash_extensions import Keyboard
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
                'background-width': '2',
                'color': '#fff'
            }
        },
        {
            'selector': 'edge',
            'style': {
                'width': '1',
                'line-color': '#888',
                'target-arrow-color': '#888',
                'target-arrow-shape': 'triangle',
                'curve-style': 'straight',
                'label': 'data(display_label)',
                'color': '#DCDCDC',

                'text-outline-opacity': '1',
                'text-outline-width': '1',
                'text-outline-color': '#222',

                'text-background-color': '#888',
                'text-background-opacity': '0',
                'text-background-shape': 'round-rectangle',
                'text-background-padding': '2px',
                'font-family': 'sans-serif',
                'font-size': '8'
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

    stylesheet.append({
        'selector': 'node.selected', # .selected is a custom class, :selected builtin bugs when resetting manually
        'style': {
            'background-color': 'red',
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

    # Configuration modal with inline styles for a light background.
    config_modal = dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle("Inference Configuration"),
            ),
            dbc.ModalBody(
                [
                    dbc.Label("Path to model.xml"),
                    dbc.Input(
                        id="model-xml-path",
                        value=ir_xml_path,
                        placeholder="Enter path to model.xml"
                    ),
                    html.Br(),
                    build_model_input_fields(ir_xml_path, inputs_path),
                    dbc.Label("Path to OpenVINO bin folder"),
                    dbc.Input(
                        id="ov-bin-path",
                        value=openvino_path,
                        placeholder="Path to OpenVINO bin/ folder"
                    ),
                    html.Br(),
                    dbc.Button("Find Plugins", id="find-plugins-button", color="dark", n_clicks=0),
                    html.Br(), html.Br(),
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
                ],
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0, color="dark"),
            ),
        ],
        id="config-modal",
        is_open=False,
    )

    # Visualization modal with inline light background styles.
    visualization_modal = dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle("Visualizations"),
                style={'backgroundColor': '#f0f0f0'}
            ),
            dbc.ModalBody(
                [
                    dcc.Tabs(
                        id="vis-tabs",
                        value="tab-3d",
                        children=[
                            dcc.Tab(label="3D Volume", value="tab-3d"),
                            dcc.Tab(label="Diagnostics", value="tab-diag")
                        ]
                    ),
                    html.Div(
                        id="tab-3d-content",
                        style={"display": "block"},
                        children=[
                            dcc.Graph(
                                id="vis-3d",
                                style={'width': 'calc(100vw - 50px)', 'height': 'calc(100vh - 150px)'}
                            )
                        ]
                    ),
                    html.Div(
                        id="tab-diag-content",
                        style={"display": "none"},
                        children=[
                            html.Div(
                                id="vis-diagnostics",
                                style={
                                    "textAlign": "center",
                                    "overflowY": "auto",
                                    "maxHeight": "80vh"
                                }
                            )
                        ]
                    ),
                ],
                style={'backgroundColor': '#f0f0f0'}
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-vis-modal", className="ml-auto"),
                style={'backgroundColor': '#f0f0f0'}
            )
        ],
        id="visualization-modal",
        fullscreen=True,
        is_open=False,
    )

    plugin_store = dcc.Store(id="plugin-store", data=discovered_plugins)
    config_store = dcc.Store(id="config-store", data=initial_config)

    graph_container = cyto.Cytoscape(
        id='ir-graph',
        elements=elements,
        style={
            "width": "100%",
            "height": "100%",
            "backgroundColor": "#404040",
            "cursor": "default",
        },
        layout={
            'name': 'dagre',
            'directed': True,
            'rankDir': 'TB',
            'nodeSep': 25,
            'rankSep': 50,
        },
        autoungrabify=True,
        autoRefreshLayout=True,  # Required for JavaScript access
        wheelSensitivity=0.2,
        stylesheet=dynamic_stylesheet
    )

    open_button = dbc.Button(
        "Inference settings",
        id="open-modal",
        color="dark",
        n_clicks=0,
        className="w-100",
    )

    left_pane = html.Div([
        open_button,

        html.H3(children=["Inferred layers"]),
        Keyboard(
            id="keyboard",
            captureKeys=["ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown", "Control"],
        ),
        html.Ul(
            id='layer-list',
            style={'padding': '10px', 'height': '100%', 'overflow': 'auto'},
        ),
    ])

    right_pane = html.Div([
        html.H3(id='right-panel-layer-name', children=["Layer's Status"]),
        html.Div(
            id='right-panel',
            style={'padding': '10px', 'height': '100%', 'overflow': 'auto'},
        ),
        dbc.Button(
            "Visualization",
            id="visualization-button",
            color="secondary",
            className="w-100",
            style={"display": "none"}
        )
    ])

    graph_and_right = DashSplitPane(
        split="vertical",
        size="20%",
        primary="first",
        children=[
            left_pane,
            graph_container,
        ]
    )

    dash_pane = DashSplitPane(
        split="vertical",
        size="20%",
        primary="second",
        children=[
            graph_and_right,
            right_pane,
        ]
    )

    return html.Div(
        [
            dash_pane,

            config_modal,
            plugin_store,
            config_store,
            visualization_modal,

            dcc.Location(id='first-load', refresh=False),
            dcc.Store(id='selected-layer-index-store', data=-1),
            dcc.Store(id='layer-store', data = []),
            dcc.Store(id='just-finished-tasks-store', data=[]),

            # To preserve selected layer info and update Layer Status on interval trigger
            dcc.Store(id='selected-layer-name-store', data=""),
            dcc.Store(id='selected-node-id-store', data=None),

            html.Div(id='dummy-output'),  # dummy output for the clientside callback
            html.Div(id='center-node-trigger', style={'display': 'none'}),

            dcc.Interval(id='update-interval', interval=1000, n_intervals=0),
        ],
    )
