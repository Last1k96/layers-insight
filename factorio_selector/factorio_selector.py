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
    "<",
    ">",
    "<=",
    ">=",
    "==",
    "!=",
]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    # Some embedded CSS for the fixed row height and basic styling.
    dcc.Markdown('''
    <style>
      .condition-row {
          height: 50px;
          display: flex;
          align-items: center;
          border-bottom: 1px solid #ccc;
          padding-left: 5px;
      }
    </style>
    ''', dangerously_allow_html=True),

    dbc.Button("Add Condition", id="add-condition-btn", n_clicks=0),

    html.Div(
        style={"display": "flex"},
        children=[
            # Left column: for AND/OR toggle buttons.
            html.Div(id="toggle-container", style={"position": "relative", "width": "60px"}),
            # Right column: main condition rows.
            html.Div(id="conditions-container", style={"flex": "1", "paddingLeft": "10px"})
        ]
    ),

    html.Hr(),
    html.Div("Resulting condition data:"),
    html.Pre(id="debug-output", style={"whiteSpace": "pre-wrap"})
])


@app.callback(
    [
        Output("conditions-container", "children"),
        Output("toggle-container", "children")
    ],
    Input("add-condition-btn", "n_clicks"),
    State("conditions-container", "children"),
    State("toggle-container", "children")
)
def add_condition(n_clicks, current_children, toggles):
    if current_children is None:
        current_children = []

    row_id = f"condition-row-{len(current_children)}"

    height = 25
    element_height = 40
    row_layout = html.Div(
        className="condition-row-container",
        id=row_id,
        style={"height": element_height},
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center"},
                children=[
                    dcc.Dropdown(
                        id={"type": "variable-dropdown", "index": row_id},
                        options=[{"label": k, "value": k} for k in options.keys()],
                        placeholder="Select variable",
                        style={"width": "150px", "height": height, "marginRight": "10px"}
                    ),
                    dcc.Dropdown(
                        id={"type": "operator-dropdown", "index": row_id},
                        options=[{"label": op, "value": op} for op in operators],
                        placeholder="",
                        style={"width": "60px", "height": height, "marginRight": "10px"}
                    ),
                    dcc.Input(
                        id={"type": "value-input", "index": row_id},
                        type="text",
                        placeholder="Value",
                        style={"width": "100px", "height": height + 10, "marginTop": "10px"}
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
            id={"type": "logic-operator", "index": len(current_children)},
            n_clicks=0,
            style={
                "position": "absolute",
                "top": f"{len(toggles) * element_height + element_height // 2 + 5}px",
                "height": f"{element_height - 5}px",
                "left": "5px",
                "width": "60px"
            }
        )

        toggles.append(and_or_button)

    return current_children, toggles


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


if __name__ == "__main__":
    app.run_server(debug=True)
