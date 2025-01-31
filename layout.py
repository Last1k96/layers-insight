# my_dash_ir_app/layout.py

from dash import html
import dash_cytoscape as cyto

def create_layout(elements):
    """
    Return the Dash layout for our IR graph using the 'dagre' layout.
    No need for manual HTML script tags; we'll use cyto.load_extra_layouts() in app.py.
    """
    return html.Div([
        html.H3("OpenVINO IR Graph with DAG Layout (Dagre)"),

        cyto.Cytoscape(
            id='ir-graph',
            elements=elements,
            style={'width': '100%', 'height': '800px'},

            # Use the 'dagre' layout
            layout={
                'name': 'dagre',
                'rankDir': 'TB',  # 'TB' = top->bottom, 'LR' = left->right
                'nodeSep': 50,
                'rankSep': 100
            },
            stylesheet=[
                {
                    'selector': 'node',
                    'style': {
                        'content': 'data(label)',
                        'font-size': '12px',
                        'text-wrap': 'wrap',
                        'text-max-width': 80,
                        'background-color': '#0074D9',
                        'color': '#fff',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'border-width': 1,
                        'border-color': '#333',
                        'padding': '10px'
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'width': 2,
                        'line-color': '#999',
                        'target-arrow-color': '#999',
                        'target-arrow-shape': 'triangle'
                    }
                }
            ]
        ),

        html.Div(
            id='inference-output',
            style={'marginTop': 10, 'fontSize': '16px'}
        )
    ])
