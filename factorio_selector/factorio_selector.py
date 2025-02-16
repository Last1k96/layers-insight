import json

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, MATCH, ALL

from evaluate_expr import build_expression_function

# Sample dictionary of possible variables:
options = {
    "name1": 5,
    "name2": 10,
    "name3": 15
}

# List of operators:
operators = [
    ">",
    "<",
    ">=",
    "<=",
    "==",
    "!=",
]

height = 25
element_height = 40

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dcc.Store(id='global-element-index', data=0),
    dcc.Store(id='global-toggle-index', data=0),
    dbc.Button("Add Condition",
               id="add-condition-btn",
               style={"marginLeft": "70px"},
               n_clicks=0),
    html.Div(
        style={"display": "flex", "marginBottom": "10px", "width": "100px"},
        children=[
            # Left column: for AND/OR toggle buttons.
            html.Div(id="toggle-container", style={"position": "relative", "width": "60px", "marginTop": "26px"}),
            # Right column: main condition rows.
            html.Div(id="conditions-container", style={"flex": "1", "paddingLeft": "10px"})
        ]
    ),

    html.Hr(),
    html.Div("Resulting condition data:"),
    html.Pre(id="debug-output", style={"whiteSpace": "pre-wrap"}),
])


@app.callback(
    [
        Output("conditions-container", "children", allow_duplicate=True),
        Output("toggle-container", "children", allow_duplicate=True),
        Output("global-element-index", "data"),
        Output("global-toggle-index", "data")
    ],
    Input("add-condition-btn", "n_clicks"),
    Input("global-element-index", "data"),
    Input("global-toggle-index", "data"),
    State("conditions-container", "children"),
    State("toggle-container", "children"),
    prevent_initial_call=True
)
def add_condition(n_clicks, global_element_index, global_toggle_index, current_children, toggles):
    if current_children is None:
        current_children = []

    row_id = f"condition-row-{global_element_index}"
    global_element_index += 1

    row_layout = html.Div(
        className="condition-row-container",
        id=row_id,
        style={"width": "600px", "height": element_height, "position": "relative"},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center"},
                children=[
                    dcc.Dropdown(
                        id={"type": "variable-dropdown", "index": row_id},
                        options=[{"label": k, "value": k} for k in options.keys()],
                        placeholder="Select variable",
                        value=next(iter(options)),
                        style={"width": "100%", "height": height, "marginRight": "4px"},
                        clearable=False,
                    ),
                    dcc.Dropdown(
                        id={"type": "operator-dropdown", "index": row_id},
                        options=[{"label": op, "value": op} for op in operators],
                        placeholder="",
                        value=">",
                        style={"width": "60px", "height": height, "marginRight": "4px"},
                        clearable = False,
                    ),
                    dcc.Input(
                        id={"type": "value-input", "index": row_id},
                        type="text",
                        placeholder="Value",
                        value=0.01,
                        style={"width": "100px", "height": height + 10, "marginRight": "4px", "marginTop": "10px"}
                    ),
                    dbc.Button(
                        "𐌗",
                        id={"type": "remove-condition", "index": row_id},
                        n_clicks=0,
                        color="danger",
                        size="sm",
                        style={"width": "30px", "height": height + 4, "marginTop": "10px"}
                    )
                ]
            )
        ]
    )

    current_children.append(row_layout)

    if not toggles:
        toggles = []

    if len(current_children) > 1:
        and_or_button = dbc.Button(
            "AND",
            id={"type": "logic-operator", "index": global_toggle_index},
            n_clicks=0,
            style={
                "height": f"{element_height - 5}px",
                "left": "5px",
                "width": "60px",
                "marginTop": "5px",
                "marginLeft": "5px",
            }
        )
        global_toggle_index += 1

        toggles.append(and_or_button)

    return current_children, toggles, global_element_index, global_toggle_index


@app.callback(
    Output({"type": "logic-operator", "index": MATCH}, "children"),
    Input({"type": "logic-operator", "index": MATCH}, "n_clicks"),
    State({"type": "logic-operator", "index": MATCH}, "children"),
    prevent_initial_call=True
)
def toggle_logic_operator(n_clicks, current_label):
    return "OR" if current_label == "AND" else "AND"


@app.callback(
    Output("debug-output", "children"),
    Input({"type": "logic-operator", "index": ALL}, "children"),
    Input({"type": "variable-dropdown", "index": ALL}, "value"),
    Input({"type": "operator-dropdown", "index": ALL}, "value"),
    Input({"type": "value-input", "index": ALL}, "value"),
    prevent_initial_call=True
)
def show_condition_data(logic_ops, variable_values, operator_values, input_values):
    if not variable_values:
        return str(), str(), str()

    tokens = [{
        "variable": variable_values[0],
        "operator": operator_values[0],
        "value": input_values[0]
    }]

    for logic_op, var_sel, op_sel, val_sel in zip(logic_ops, variable_values[1:], operator_values[1:],
                                                  input_values[1:]):
        tokens.append({
            "logic_op": logic_op,
        })
        tokens.append({
            "variable": variable_values[0],
            "operator": operator_values[0],
            "value": input_values[0]
        })

    expr = build_expression_function(tokens)
    expr_evaluated = expr(options)
    return str(options), str(tokens), str(expr_evaluated)


@app.callback(
    [
        Output("conditions-container", "children", allow_duplicate=True),
        Output("toggle-container", "children", allow_duplicate=True)
    ],
    Input({"type": "remove-condition", "index": ALL}, "n_clicks"),
    State("conditions-container", "children"),
    State("toggle-container", "children"),
    prevent_initial_call=True
)
def remove_condition(n_clicks_list, rows, toggles):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    if all(nc == 0 for nc in n_clicks_list):
        raise dash.exceptions.PreventUpdate

    # Find which remove button was clicked
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    triggered_id = json.loads(triggered_id)  # Convert string back to dict
    remove_index = triggered_id["index"]

    remove_id = [i for i, row in enumerate(rows) if row["props"]["id"] == remove_index][0]
    rows.pop(remove_id)

    if toggles:
        remove_toggle_id = max(0, remove_id - 1)
        toggles.pop(remove_toggle_id)

    return rows, toggles


if __name__ == "__main__":
    app.run_server(debug=True)
