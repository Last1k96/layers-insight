import random
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
from known_ops import OPENVINO_OP_COLORS_DARK

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
                'border-width': '1.5',
                'border-color': '#000',
                'padding': '6px',
                'font-size': '12px',
                'font-family': 'sans-serif',
                'text-valign': 'center',
                'text-halign': 'center',
                'background-color': '#666',
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

                'text-outline-opacity': '1',
                'text-outline-width': '2',
                'text-outline-color': '#303436',

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


def create_layout(elements):
    dynamic_stylesheet = build_dynamic_stylesheet(elements)

    return html.Div([
        # Cytoscape Graph (Left Side)
        cyto.Cytoscape(
            id='ir-graph',
            elements=elements,
            style={"position": "absolute", "top": 0, "left": 0, "width": "100%", "height": "100%"},
            layout={
                'name': 'dagre',
                'directed': True,
                'rankDir': 'TB',
                'nodeSep': 25,
                'rankSep': 50,
            },
            # autoungrabify=True,
            wheelSensitivity=0.2,  # Adjusts the zoom speed when scrolling
            stylesheet=dynamic_stylesheet
        ),
        html.Div([
            html.H3("Left Panel (e.g., Workspace Selection)"),
            html.H3("Select / Delete Workspace"),
            dcc.Dropdown(id='workspace-select-dropdown', options=[], value=None),
            html.Button("Delete Selected Workspace", id='delete-workspace-btn', style={"marginTop": "5px"}),

            dcc.Input(id='workspace-label-input', type='text', placeholder='Friendly name', value='My Workspace'),
            dcc.Input(id='model-xml-input', type='text', placeholder='Model XML Path', value=''),
            dcc.Input(id='input-file-input', type='text', placeholder='Input file', value=''),

            html.Div("Reference Plugin:"),
            dcc.Dropdown(id='reference-plugin-dropdown', options=[], value=None, clearable=False),

            html.Div("Other Plugin:"),
            dcc.Dropdown(id='other-plugin-dropdown', options=[], value=None, clearable=False),

            html.Button("Create Workspace", id='create-workspace-btn'),
            html.Label("OpenVINO bin folder:"),
            dcc.Input(id='openvino-bin-input', type='text', placeholder='/path/to/openvino_bin', value=''),
            html.Button("Find Plugins", id='find-plugins-btn'),
            html.Div("Available Plugins:", style={"marginTop": "10px"}),
            html.Div(id='available-plugins-list', style={"whiteSpace": "pre-line"}),

            # The drag handle can be its own sub-div on the right edge
            html.Div(id="left-drag-handle", className="drag-handle-left")
        ],
            id="left-panel",
            className="panel-left"),
        html.Div([
            html.H3("Right Panel (e.g., Partial Inference Output)"),
            html.Div("Your inference results or analysis here..."),

            # The drag handle on the left edge
            html.Div(id="right-drag-handle", className="drag-handle-right")
        ],
            id="right-panel",
            className="panel-right"
        ),
    ], className="main-container",
        style={"position": "relative", "width": "100vw", "height": "100vh", "background-color": "#404040"})


