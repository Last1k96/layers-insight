import dash
import dash_cytoscape as cyto
from dash import Output, Input, State

from cache import process_tasks
from layout import create_layout
from callbacks import register_callbacks
import dash_bootstrap_components as dbc

import threading


def create_app(openvino_path, ir_xml_path, inputs_path):
    cyto.load_extra_layouts()
    app = dash.Dash(__name__, title="Layers Insight", external_stylesheets=[dbc.themes.DARKLY])
    app.layout = create_layout(openvino_path, ir_xml_path, inputs_path)

    register_callbacks(app)

    app.clientside_callback(
        """
        function(nodeId, keysPressed) {
            if (!keysPressed) {
                return null;
            }
            if (!("Control" in keysPressed)) {
                return null;
            }
            if (nodeId === undefined || nodeId === null) {
                return null;
            }
            if (window.cy) {
                const element = window.cy.getElementById(nodeId);
                const zoom = window.cy.zoom();
                
                const nodePos = element.position();
                
                const viewportCenterX = window.cy.width() / 2;
                const viewportCenterY = window.cy.height() / 2;
                
                const newPanX = viewportCenterX - (nodePos.x * zoom);
                const newPanY = viewportCenterY - (nodePos.y * zoom);
                
                window.cy.animate({
                    pan: { x: newPanX, y: newPanY }
                }, {
                    duration: 180,   // Duration in milliseconds.
                    easing: 'ease-in-out'
                }); 
            }
            return null;
        }
        """,
        Output('dummy-output', 'children'),
        Input('selected-node-id-store', 'data'),
        State('keyboard', 'keys_pressed')
    )

    threading.Thread(target=process_tasks, daemon=True).start()

    return app
