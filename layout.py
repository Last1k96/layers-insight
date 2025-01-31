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
                'border-radius': '6px',
                'border-width': 1,
                'border-color': '#333',
                'padding': '10px',
                'font-size': '12px',
                'text-wrap': 'wrap',
                'text-max-width': 80,
                'text-valign': 'center',
                'text-halign': 'center',

                # We'll default to a medium gray if we don't find a color
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

    # Notice we are not setting backgroundColor or color inline here,
    # because we handle that globally in assets/style.css
    return html.Div([
        html.H3("IR Graph: Dark Color Palette"),
        cyto.Cytoscape(
            id='ir-graph',
            elements=elements,
            style={'width': '100%', 'height': '800px'},
            layout={
                'name': 'dagre',
                'directed': True,
                'rankDir': 'TB',
                'nodeSep': 25,
                'rankSep': 50,
            },
            autoungrabify=True,  # Automatically disable grabbing
            stylesheet=dynamic_stylesheet
        ),
        html.Div(id='inference-output', style={'fontSize': '16px'})
    ])
