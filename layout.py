# my_dash_ir_app/layout.py

import random
from dash import html
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
                'border-width': 2,
                'border-color': '#111',
                'padding': '10px',
                'font-size': '12px',
                'text-wrap': 'wrap',
                'text-max-width': 80,
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
                'target-arrow-shape': 'triangle'
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
            style={
                'height': '100%',
                'width': '100%',
                'minWidth': '0',  # Critical for resize
                'minHeight': '0'  # Critical for resize
            },
            layout={
                'name': 'dagre',
                'directed': True,
                'rankDir': 'TB',
                'nodeSep': 25,
                'rankSep': 50,
            },
            autoungrabify=True,
            stylesheet=dynamic_stylesheet
        ),
        html.Div(id='splitter', className='splitter'),
        html.Div(id='inference-output')
    ], className="main-container")

