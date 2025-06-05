import os
import random
from datetime import datetime
from pathlib import Path

from dash import dcc, html
from dash_extensions import Keyboard
import dash_cytoscape as cyto

from openvino_graph import parse_openvino_ir
from known_ops import OPENVINO_OP_COLORS_DARK
from colors import BorderColor
import dash_bootstrap_components as dbc

from run_inference import get_available_plugins


def update_config(config: dict, model_xml=None, ov_bin_path=None, plugin1=None, plugin2=None, model_inputs=None):
    if model_xml is not None:
        config["model_xml"] = model_xml
    if ov_bin_path is not None:
        config["ov_bin_path"] = ov_bin_path
    if plugin1 is not None:
        config["plugin1"] = plugin1
    if plugin2 is not None:
        config["plugin2"] = plugin2
    if model_inputs is not None:
        config["model_inputs"] = model_inputs

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

    # Collect distinct op types and add style rules
    used_random_colors = {}
    for el in elements:
        if 'data' in el and 'type' in el['data']:
            op_type = el['data']['type']
            if op_type in OPENVINO_OP_COLORS_DARK:
                color = OPENVINO_OP_COLORS_DARK[op_type]
            else:
                if op_type not in used_random_colors:
                    r, g, b = random.randint(40, 160), random.randint(40, 160), random.randint(40, 160)
                    used_random_colors[op_type] = f"#{r:02X}{g:02X}{b:02X}"
                color = used_random_colors[op_type]

            # Add rule only if not already added
            selector = f'node[type="{op_type}"]'
            if not any(rule['selector'] == selector for rule in stylesheet):
                stylesheet.append({
                    'selector': selector,
                    'style': {'background-color': color}
                })

    stylesheet.append({
        'selector': 'node.selected',  # .selected is a custom class, :selected builtin bugs when resetting manually
        'style': {
            'background-color': BorderColor.SELECTED.value,
        }
    })

    return stylesheet


def read_openvino_ir(model_xml_path):
    from openvino import Core
    core = Core()
    model = core.read_model(model_xml_path)
    return [{"name": node.get_any_name(), "shape": list(node.shape)} for node in model.inputs]


def build_model_input_fields(model_inputs, inputs_path):
    components = []
    for index, (model_input, input_path) in enumerate(zip(model_inputs, inputs_path), start=1):
        name, shape = model_input["name"], model_input["shape"]
        components.extend([
            dbc.Label(f"Input #{index}: '{name}' with shape {shape}"),
            dbc.Input(
                id={"type": "model-input", "name": name},
                type="text",
                placeholder=f"Enter input path (fill with random noize if empty)",
                value=input_path,
            ),
            html.Br()
        ])
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

    # Configuration modal
    config_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Inference Configuration")),
            dbc.ModalBody(
                dbc.Tabs([
                    dbc.Tab(
                        [
                            dbc.Label("Path to OpenVINO bin folder"),
                            dbc.InputGroup([
                                dbc.Input(id="ov-bin-path", value=openvino_path,
                                          placeholder="Path to OpenVINO bin/ folder"),
                                dbc.Button("Browse", id="browse-ov-bin-path", color="secondary"),
                            ]),
                            html.Br(),
                            dbc.Label("Reference Plugin"),
                            dcc.Dropdown(id="reference-plugin-dropdown", options=discovered_plugins, clearable=False),
                            html.Br(),
                            dbc.Label("Main Plugin"),
                            dcc.Dropdown(id="main-plugin-dropdown", options=discovered_plugins, clearable=False),
                        ],
                        label="OpenVINO",
                        className="p-3"
                    ),
                    dbc.Tab(
                        [
                            dbc.Label("Path to model.xml"),
                            dbc.Input(id="model-xml-path", value=model_path, placeholder="Enter path to model.xml"),
                            html.Br(),
                            html.Div(id="model-input-paths",
                                     children=build_model_input_fields(model_inputs, inputs_path)),
                        ],
                        label="Model",
                        className="p-3"
                    ),
                    dbc.Tab(
                        [
                            dbc.Label("Plugin"),
                            dcc.Dropdown(id="config-plugin-dropdown", options=discovered_plugins, clearable=False),
                            html.Br(),
                            html.Div(id="plugin-config-table"),
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

    # Visualization modal
    visualization_buttons = [
        dbc.Button(label, id={"type": "visualization-btn", "index": viz_id}, className="mb-1 w-100")
        for label, viz_id in [
            ("Volumetric", "viz1"),
            ("Isosurfaces", "viz4"),
            ("Bubble Rings", "viz10"),
            ("Per-channel slider", "viz3"),
            ("Per-channel unrolled", "viz2"),
            ("Hierarchical View", "viz9"),
            ("Correlation", "viz12")
        ]
    ]

    visualization_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Visualizations")),
            dbc.ModalBody(
                html.Div(
                    style={"display": "flex", "height": "100%", "overflow": "hidden"},
                    children=[
                        html.Div(
                            id="visualization-container",
                            children=[],
                            style={"flex": "1", "overflowY": "auto", "overflowX": "hidden"},
                        ),
                        html.Div(
                            id="visualization-buttons",
                            children=visualization_buttons,
                            style={"width": "200px", "display": "flex", "flexDirection": "column"}
                        )
                    ]
                ),
                style={"padding": 0, "height": "100vh", "overflow": "hidden"}
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
        style={"width": "100%", "height": "100%", "backgroundColor": "#404040", "cursor": "default"},
        layout={'name': 'dagre', 'directed': True, 'rankDir': 'TB', 'nodeSep': 25, 'rankSep': 50, 'fit': False},
        autoungrabify=True,
        autoRefreshLayout=False,
        wheelSensitivity=0.2,
        stylesheet=dynamic_stylesheet
    )

    left_pane = html.Div([
        dbc.Button("Inference settings", id="inference-settings-btn", color="dark", n_clicks=0,
                   className="w-100", style={'margin': '0'}),
        html.Div([
            Keyboard(id="keyboard", captureKeys=["ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"]),
            html.Div(
                html.Ul(id='layer-panel-list', style={'padding': '2px'}),
                style={'overflowY': 'auto', 'flex': '1'}
            ),
        ], style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'overflow': 'hidden'}),
        dbc.Button("Clear Queue", id="clear-queue-btn", color="dark", n_clicks=0,
                   className="w-100", style={'margin': '0'}),
    ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '0'})

    # Define buttons with common properties
    action_buttons = [
        ("Save outputs", "save-outputs-button"),
        ("Save reproducer", "save-reproducer-button"),
        ("Transform this layer into model input", "transform-to-input-button"),
        ("Restart layer", "restart-layer-button")
    ]

    right_pane = html.Div([
        html.H5(id='right-panel-layer-name', children=["Layer's Status"]),
        html.Div(
            children=[
                html.Div(id='right-panel-content', style={'height': '100%', 'overflow': 'auto'}),
                *[dbc.Button(text, id=btn_id, color="secondary", style={'display': 'none'})
                  for text, btn_id in action_buttons]
            ],
            style={'height': '100%', 'overflowY': 'auto'}
        )
    ])

    notification_toast = dbc.Toast(
        "Notification message",
        id="notification-toast",
        header="Notification",
        is_open=False,
        dismissable=True,
        duration=5000,
        style={"position": "fixed", "right": "calc(15% + 20px)", "top": "10px", "width": "300px", "zIndex": 1000}
    )

    # Panels (resizable)
    left_panel = html.Div(
        [
            html.Div(left_pane, className="panel-content"),
            html.Div(className="resize-handle", id="left-panel-resize-handle")
        ],
        className="side-panel left-panel",
        id="left-panel"
    )

    right_panel = html.Div(
        [
            html.Div([right_pane, notification_toast], className="panel-content"),
            html.Div(className="resize-handle", id="right-panel-resize-handle")
        ],
        className="side-panel right-panel",
        id="right-panel"
    )

    # Group related stores for better organization
    visualization_stores = [
        dcc.Store(id='store-figure', data={}),
        dcc.Store(id='update-visualization-on-open'),
        dcc.Store(id='update-visualization-on-close'),
        dcc.Store(id='last-selected-visualization', data=None),
        dcc.Store(id='visualization-output-id', data=None),
    ]

    layer_selection_stores = [
        dcc.Store(id='selected-layer-index-store', data=-1),
        dcc.Store(id='layers-store', data=[]),
        dcc.Store(id='just-finished-tasks-store', data=[]),
        dcc.Store(id='selected-layer-type-store', data=""),
        dcc.Store(id='selected-node-id-store', data=None),
    ]

    model_update_stores = [
        dcc.Store(id='config-store-after-cut', data={}),
        dcc.Store(id='model-path-after-cut', data=""),
        dcc.Store(id='transformed-node-name-store', data=None),
        dcc.Store(id='clicked-graph-node-id-store'),
        dcc.Store(id='clear-queue-store', data=False),
        dcc.Store('dummy-output', data=None),
        dcc.Store(id='refresh-layout-trigger', data=0),
    ]

    # File browser modal for directory selection
    file_browser_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Select Directory")),
            dbc.ModalBody(
                [
                    html.Div(id="file-browser-content"),
                    dcc.Store(id="file-browser-current-path", data=""),
                    dcc.Store(id="file-browser-target", data=""),
                    dcc.Store(id="file-browser-mode", data="directory"),
                    dcc.Store(id="file-browser-selected-file", data=""),
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button("Cancel", id="file-browser-cancel", className="me-2"),
                    dbc.Button("Select", id="file-browser-select", color="primary"),
                ]
            ),
        ],
        id="file-browser-modal",
        is_open=False,
        size="lg",
    )

    return html.Div(
        [
            graph_container,
            left_panel,
            right_panel,
            config_modal,
            plugin_store,
            config_store,
            visualization_modal,
            file_browser_modal,
            dcc.Location(id='first-load', refresh=False),
            *visualization_stores,
            *layer_selection_stores,
            *model_update_stores,
            dcc.Interval(id='update-interval', interval=1000, n_intervals=0),
        ],
        style={'height': '100vh', 'overflow': 'hidden', 'position': 'fixed', 'width': '100%', 'top': '0', 'left': '0'}
    )
