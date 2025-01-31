import dash_cytoscape as cyto
from dash import html

def create_layout(elements):
    """
    Return the Dash layout, given a list of Cytoscape elements.
    """
    return html.Div([
        # The Cytoscape graph
        cyto.Cytoscape(
            id='cytoscape-graph',
            elements=elements,
            style={'width': '100%', 'height': '600px'},
            layout={'name': 'grid'},  # or 'breadthfirst', 'circle', etc.
            stylesheet=[
                {
                    'selector': 'node',
                    'style': {
                        'content': 'data(label)',
                        'background-color': '#0074D9',
                        'color': '#fff',
                        'text-valign': 'center'
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'width': 3,
                        'line-color': '#ccc',
                        'target-arrow-color': '#ccc',
                        'target-arrow-shape': 'triangle'
                    }
                }
            ]
        ),
        # A place to show inference results or logs
        html.Div(id='inference-output', style={'marginTop': 10, 'fontSize': '16px'})
    ])
