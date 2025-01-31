# my_dash_ir_app/layout.py

from dash import html
import dash_cytoscape as cyto

def create_layout(elements):
    """
    Return Dash layout with a DAG-based ('dagre') layout,
    and node styling that auto-sizes around the label text
    (rounded corners, margin, color-coded by type).
    """
    return html.Div([
        html.H3("IR Graph (Netron-like style: auto-sized round rectangles)"),

        # Cytoscape graph
        cyto.Cytoscape(
            id='ir-graph',
            elements=elements,
            style={'width': '100%', 'height': '800px'},

            # DAG layout
            layout={
                'name': 'dagre',
                'rankDir': 'TB',   # 'TB' = top->bottom, or 'LR' for left->right
                'nodeSep': 40,
                'rankSep': 80
            },

            # Stylesheet
            stylesheet=[

                # (1) Default node style: auto-sized "round-rectangle" with wrapped text
                {
                    'selector': 'node',
                    'style': {
                        # Show the label stored in data['display_label']
                        'content': 'data(display_label)',

                        # Auto-fit the node size to its label text
                        'width': 'label',
                        'height': 'label',

                        'shape': 'round-rectangle',
                        'border-radius': '6px',
                        'border-width': 1,
                        'border-color': '#333',

                        # Provide space around the text
                        'padding': '10px',

                        # Text styling
                        'font-size': '12px',
                        'text-wrap': 'wrap',
                        'text-max-width': 80,   # break text after ~80px
                        'text-valign': 'center',
                        'text-halign': 'center',

                        # Default background/text color
                        'background-color': '#777',
                        'color': '#fff',
                    }
                },

                # (2) Convolution = Blue
                {
                    'selector': 'node[type="Convolution"]',
                    'style': {
                        'background-color': '#337ab7',
                    }
                },

                # (3) Add = Dark Gray
                {
                    'selector': 'node[type="Add"]',
                    'style': {
                        'background-color': '#424242',
                    }
                },

                # (4) ReLU = Red
                {
                    'selector': 'node[type="ReLU"]',
                    'style': {
                        'background-color': '#c0392b',
                    }
                },

                # (5) MaxPool = Green
                {
                    'selector': 'node[type="MaxPool"]',
                    'style': {
                        'background-color': '#239B56',
                    }
                },

                # (6) Parameter (e.g., "data") = Gray
                {
                    'selector': 'node[type="Parameter"]',
                    'style': {
                        'background-color': '#888',
                    }
                },

                # (7) Edges
                {
                    'selector': 'edge',
                    'style': {
                        'width': 2,
                        'line-color': '#bbb',
                        'target-arrow-color': '#bbb',
                        'target-arrow-shape': 'triangle',
                        'arrow-scale': 1
                    }
                }
            ]
        ),

        # Div for inference or logging output
        html.Div(
            id='inference-output',
            style={'marginTop': 10, 'fontSize': '16px'}
        )
    ])
