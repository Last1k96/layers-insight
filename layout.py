# my_dash_ir_app/layout.py

from dash import html
import dash_cytoscape as cyto

def create_layout(elements):
    """
    Return Dash layout with a DAG-based layout ('dagre'),
    plus custom stylesheet for Netron-like rounded rectangles and colors.
    """
    return html.Div([
        html.H3("IR Graph (Netron-like Style)"),

        cyto.Cytoscape(
            id='ir-graph',
            elements=elements,
            style={'width': '100%', 'height': '800px'},

            layout={
                'name': 'dagre',
                'rankDir': 'TB',  # 'TB' = top->bottom, 'LR' = left->right
                'nodeSep': 40,
                'rankSep': 80
            },

            stylesheet=[
                # Default node style
                {
                    'selector': 'node',
                    'style': {
                        'shape': 'round-rectangle',
                        'content': 'data(display_label)',
                        'font-size': '12px',
                        'text-wrap': 'wrap',
                        'text-max-width': 80,
                        'background-color': '#777',
                        'color': '#fff',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'border-width': 1,
                        'border-color': '#333',
                        'border-radius': '6px',
                        'padding': '6px'
                    }
                },
                # Convolution = Blue
                {
                    'selector': 'node[type="Convolution"]',
                    'style': {
                        'background-color': '#337ab7',  # netron-like blue
                    }
                },
                # Add = Dark Gray
                {
                    'selector': 'node[type="Add"]',
                    'style': {
                        'background-color': '#424242',
                    }
                },
                # ReLU = Red
                {
                    'selector': 'node[type="ReLU"]',
                    'style': {
                        'background-color': '#c0392b',
                    }
                },
                # MaxPool = Green
                {
                    'selector': 'node[type="MaxPool"]',
                    'style': {
                        'background-color': '#239B56',
                    }
                },
                # Parameter (e.g. data) = Gray
                {
                    'selector': 'node[type="Parameter"]',
                    'style': {
                        'background-color': '#888',
                    }
                },

                # Similarly for other ops: Mul, SoftMax, Concat, etc.
                # Example:
                # {
                #   'selector': 'node[type="SoftMax"]',
                #   'style': { 'background-color': '#9B59B6' }
                # },

                # Edges
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
        html.Div(id='inference-output', style={'marginTop': 10, 'fontSize': '16px'})
    ])
