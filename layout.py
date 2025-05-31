import os
import random
from datetime import datetime
from pathlib import Path

from dash import dcc, html, Input, Output, State
from dash_extensions import Keyboard
import dash_cytoscape as cyto

from openvino_graph import parse_openvino_ir
from known_ops import OPENVINO_OP_COLORS_DARK
from colors import BorderColor
import dash_bootstrap_components as dbc

from run_inference import get_available_plugins


def update_config(config: dict, model_xml=None, ov_bin_path=None, plugin1=None, plugin2=None, model_inputs=None):
    config.update({k: v for k, v in locals().items() if k != "config" and v is not None})
    p = Path(config["model_xml"])
    model_name = p.stem
    output_path = Path(f"dump/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{model_name}")
    config["output_folder"] = str(output_path.resolve())


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
        'selector': 'node.selected',  # .selected is a custom class, :selected builtin bugs when resetting manually
        'style': {
            'background-color': BorderColor.ERROR.value,
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


def build_model_input_fields(model_inputs, inputs_path):
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
                id={"type": "model-input", "name": name},
                type="text",
                placeholder=f"Enter input path (fill with random noize if empty)",
                value=input_path,
            )
        )
        components.append(html.Br())

    return components


def create_layout(openvino_path, model_path, inputs_path):
    elements = parse_openvino_ir(model_path)
    dynamic_stylesheet = build_dynamic_stylesheet(elements)

    if openvino_path and os.path.exists(openvino_path):
        discovered_plugins = list(get_available_plugins(openvino_path))
    else:
        discovered_plugins = []

    if "CPU" in discovered_plugins:
        plugin1_value = "CPU"
    else:
        plugin1_value = discovered_plugins[0] if discovered_plugins else None

    non_cpu_plugins = [p for p in discovered_plugins if p != "CPU"]
    plugin2_value = non_cpu_plugins[0] if non_cpu_plugins else None

    model_inputs = read_openvino_ir(model_path)

    if len(inputs_path) == 0 or len(inputs_path) != len(model_inputs):
        inputs_path = [""] * len(model_inputs)

    initial_config = {}
    update_config(
        initial_config,
        model_path,
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
                dbc.Tabs([
                    dbc.Tab(
                        [
                            dbc.Label("Path to OpenVINO bin folder"),
                            dbc.Input(
                                id="ov-bin-path",
                                value=openvino_path,
                                placeholder="Path to OpenVINO bin/ folder"
                            ),
                            html.Br(),
                            dbc.Label("Reference Plugin"),
                            dcc.Dropdown(
                                id="reference-plugin-dropdown",
                                options=discovered_plugins,
                                clearable=False,
                            ),
                            html.Br(),
                            dbc.Label("Main Plugin"),
                            dcc.Dropdown(
                                id="main-plugin-dropdown",
                                options=discovered_plugins,
                                clearable=False,
                            ),
                        ],
                        label="OpenVINO",
                        className="p-3"
                    ),
                    dbc.Tab(
                        [
                            dbc.Label("Path to model.xml"),
                            dbc.Input(
                                id="model-xml-path",
                                value=model_path,
                                placeholder="Enter path to model.xml"
                            ),
                            html.Br(),
                            html.Div(
                                id="model-input-paths",
                                children=build_model_input_fields(model_inputs, inputs_path)
                            ),
                        ],
                        label="Model",
                        className="p-3"  # Adds padding for better spacing
                    ),
                    dbc.Tab(
                        [
                            dbc.Label("Plugin"),
                            dcc.Dropdown(
                                id="config-plugin-dropdown",
                                options=discovered_plugins,
                                clearable=False,
                            ),
                            html.Br(),
                            html.Div(id="plugin-config-table"),  # Table gets built dynamically
                            dcc.Store(id="plugins-config-store", data={})
                        ],
                        label="Plugin Config",
                        className="p-3"
                    )
                ])
            ),
            dbc.ModalFooter(
                dbc.Button("Save", id="save-inference-config-button", n_clicks=0),
            ),
        ],
        id="inference-settings-modal",
        is_open=False,
        size="xl",
    )

    # Visualization modal with inline light background styles.
    visualization_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Visualizations")),
            dbc.ModalBody(
                html.Div(
                    style={
                        "display": "flex",
                        "height": "100%",  # use full height of modal body
                        "overflow": "hidden"  # disable scrolling on the container
                    },
                    children=[
                        # Left side: Visualization container that scrolls internally
                        html.Div(
                            id="visualization-container",
                            children=[],  # your image(s) will be added here
                            style={
                                "flex": "1",
                                "overflowY": "auto",  # vertical scroll for tall image
                                "overflowX": "hidden"  # hide horizontal scrollbar
                            },
                        ),
                        # Right side: Fixed width button column
                        html.Div(
                            id="visualization-buttons",
                            children=[
                                dbc.Button("Volumetric", id={"type": "visualization-btn", "index": "viz1"},
                                           className="mb-1 w-100"),
                                dbc.Button("Isosurfaces", id={"type": "visualization-btn", "index": "viz4"},
                                           className="mb-1 w-100"),
                                dbc.Button("Bubble Rings", id={"type": "visualization-btn", "index": "viz10"},
                                           className="mb-1 w-100"),
                                dbc.Button("Per-channel slider", id={"type": "visualization-btn", "index": "viz3"},
                                           className="mb-1 w-100"),
                                dbc.Button("Per-channel unrolled", id={"type": "visualization-btn", "index": "viz2"},
                                           className="mb-1 w-100"),
                                dbc.Button("Hierarchical View", id={"type": "visualization-btn", "index": "viz9"},
                                           className="mb-1 w-100"),
                                dbc.Button("Correlation", id={"type": "visualization-btn", "index": "viz12"},
                                           className="mb-1 w-100"),
                            ],
                            style={
                                "width": "200px",  # fixed width for the buttons column
                                "display": "flex",
                                "flexDirection": "column"
                            }
                        )
                    ]
                ),
                style={
                    "padding": 0,
                    "height": "100vh",  # ensure modal body fills the viewport
                    "overflow": "hidden"  # disable modal-level scrolling
                }
            ),
        ],
        id="visualization-modal",
        fullscreen=True,
        is_open=False,
    )

    plugin_store = dcc.Store(id="plugin-store", data=discovered_plugins)
    config_store = dcc.Store(id="config-store", data=initial_config)

    graph_container = cyto.Cytoscape(
        id='ir-graph',
        className="main-graph",
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
            'fit': False,
        },
        autoungrabify=True,
        autoRefreshLayout=False,
        wheelSensitivity=0.2,
        stylesheet=dynamic_stylesheet
    )

    left_pane = html.Div([
        dbc.Button(
            "Inference settings",
            id="inference-settings-btn",
            color="dark",
            n_clicks=0,
            className="w-100",
            style={'margin-bottom': '10px'}
        ),
        Keyboard(
            id="keyboard",
            captureKeys=["ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"],
        ),
        html.Div(
            html.Ul(
                id='layer-panel-list',
                style={'padding': '2px'}
            ),
            style={
                'overflowY': 'auto',
                'flex': '1'
            }
        ),
        dbc.Button(
            "Clear Queue",
            id="clear-queue-btn",
            color="dark",
            n_clicks=0,
            className="w-100",
        ),
    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'height': '100%'
    })

    right_pane = html.Div([
        html.H5(id='right-panel-layer-name', children=["Layer's Status"]),
        html.Div(
            children=[
                html.Div(
                    id='right-panel-content',
                    style={'height': '100%', 'overflow': 'auto'},
                ),
                dbc.Button(
                    "Save outputs",
                    id="save-outputs-button",
                    color="secondary",
                    style={'display': 'none'}  # will be updated by update_stats callback
                ),
                dbc.Button(
                    "Save reproducer",
                    id="save-reproducer-button",
                    color="secondary",
                    style={'display': 'none'}  # will be updated by update_stats callback
                ),
                dbc.Button(
                    "Transform this layer into model input",
                    id="transform-to-input-button",
                    color="secondary",
                    style={'display': 'none'}  # will be updated by update_stats callback
                ),
                dbc.Button(
                    "Restart layer",
                    id="restart-layer-button",
                    color="secondary",
                    style={'display': 'none'}  # will be updated by update_stats callback
                )
            ],
            style={
                'height': '100%',
                'overflowY': 'auto'
            }
        )
    ])

    notification_toast = dbc.Toast(
        "Notification message",
        id="notification-toast",
        header="Notification",
        is_open=False,
        dismissable=True,
        duration=5000,
        style={
            "position": "fixed",
            "right": "calc(15% + 20px)",  # Position to the left of the right panel
            "top": "10px",
            "width": "300px",
            "zIndex": 1000,
        }
    )

    # Left panel (resizable)
    left_panel = html.Div(
        [
            html.Div(
                left_pane,
                className="panel-content"
            ),
            html.Div(
                className="resize-handle",
                id="left-panel-resize-handle"
            )
        ],
        className="side-panel left-panel",
        id="left-panel"
    )

    # Right panel (resizable) with notification toast
    right_panel = html.Div(
        [
            html.Div(
                [right_pane, notification_toast],
                className="panel-content"
            ),
            html.Div(
                className="resize-handle",
                id="right-panel-resize-handle"
            )
        ],
        className="side-panel right-panel",
        id="right-panel"
    )

    return html.Div(
        [
            graph_container,
            left_panel,
            right_panel,

            config_modal,
            plugin_store,
            config_store,

            dcc.Store(id='store-figure', data={}),
            dcc.Store(id='update-visualization-on-open'),
            dcc.Store(id='update-visualization-on-close'),
            dcc.Store(id='last-selected-visualization', data=None),
            dcc.Store(id='visualization-output-id', data=None),
            visualization_modal,

            dcc.Location(id='first-load', refresh=False),
            dcc.Store(id='selected-layer-index-store', data=-1),
            dcc.Store(id='layers-store', data=[]),
            dcc.Store(id='just-finished-tasks-store', data=[]),

            # To preserve selected layer info and update Layer Status on interval trigger
            dcc.Store(id='selected-layer-type-store', data=""),
            dcc.Store(id='selected-node-id-store', data=None),

            # To signal updates after the network was cut
            dcc.Store(id='config-store-after-cut', data={}),
            dcc.Store(id='model-path-after-cut', data=""),

            dcc.Store(id='clicked-graph-node-id-store'),  # to break circular dependency
            dcc.Store(id='clear-queue-store', data=False),

            dcc.Store(id='transformed-node-name-store', data=None),  # to store the name of the node that was transformed
            dcc.Store('dummy-output', data=None),
            dcc.Store(id='refresh-layout-trigger', data=0),  # to trigger manual layout refresh

            dcc.Interval(id='update-interval', interval=1000, n_intervals=0),
        ],
            style={
                'height': '100vh',
                'overflow': 'hidden',
                'position': 'fixed',
                'width': '100%',
                'top': '0',
                'left': '0'
            }
    )
