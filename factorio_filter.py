import json
import re

import dash
from dash import dcc, html, Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc

from factorio_selector.evaluate_expr import build_expression_function
from metrics import METRIC_INFO

# List of operators for different variable types
OPERATORS = {
    "metrics": [">", "<", ">=", "<=", "==", "!="],
    "name": ["~="],
    "type": ["~="]
}

# Element styling
HEIGHT = 25
ELEMENT_HEIGHT = 40

def create_factorio_selector():
    """Create the Factorio Selector component for the left panel."""
    return html.Div([
        dcc.Store(id='factorio-element-index', data=0),
        dcc.Store(id='factorio-toggle-index', data=0),
        html.Div(
            style={"display": "flex", "width": "100%", "marginBottom": "10px"},
            children=[
                # Left column: for AND/OR toggle buttons
                html.Div(id="factorio-toggle-container", style={"width": "60px", "marginRight": "5px"}),
                # Right column: main condition rows
                html.Div(id="factorio-conditions-container", style={"flex": "1"})
            ]
        ),
        dbc.Button("Add condition",
                  id="factorio-add-condition-btn",
                  n_clicks=0,
                  className="w-100",
                  style={'margin': '0'})
    ])

def register_factorio_callbacks(app):
    """Register all callbacks for the Factorio Selector."""

    @app.callback(
        [
            Output("factorio-conditions-container", "children", allow_duplicate=True),
            Output("factorio-toggle-container", "children", allow_duplicate=True),
            Output("factorio-element-index", "data"),
            Output("factorio-toggle-index", "data")
        ],
        Input("factorio-add-condition-btn", "n_clicks"),
        State("factorio-element-index", "data"),
        State("factorio-toggle-index", "data"),
        State("factorio-conditions-container", "children"),
        State("factorio-toggle-container", "children"),
        prevent_initial_call=True
    )
    def add_condition(n_clicks, global_element_index, global_toggle_index, current_children, toggles):
        if current_children is None:
            current_children = []

        row_id = f"factorio-condition-row-{global_element_index}"
        global_element_index += 1

        # Create variable options: Name, Type, and all metrics
        variable_options = [
            {"label": "Name", "value": "name"},
            {"label": "Type", "value": "type"}
        ]

        # Add metrics from METRIC_INFO
        for metric_key in METRIC_INFO:
            full_name, _ = METRIC_INFO[metric_key]
            variable_options.append({"label": full_name, "value": f"metrics.{metric_key}"})

        row_layout = html.Div(
            className="factorio-condition-row-container",
            id=row_id,
            style={"width": "100%", "height": ELEMENT_HEIGHT, "position": "relative"},
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center"},
                    children=[
                        dcc.Dropdown(
                            id={"type": "factorio-variable-dropdown", "index": row_id},
                            options=variable_options,
                            placeholder="Select variable",
                            value=variable_options[0]["value"],
                            style={"width": "100%", "height": HEIGHT, "marginRight": "4px"},
                            clearable=False,
                        ),
                        dcc.Dropdown(
                            id={"type": "factorio-operator-dropdown", "index": row_id},
                            options=[{"label": "~=", "value": "~="}],  # Default for "name"
                            placeholder="",
                            value="~=",
                            style={"width": "60px", "height": HEIGHT, "marginRight": "4px"},
                            clearable=False,
                        ),
                        dcc.Input(
                            id={"type": "factorio-value-input", "index": row_id},
                            type="text",
                            placeholder="Value",
                            value="",
                            style={"width": "100px", "height": HEIGHT + 10, "marginRight": "4px", "marginTop": "10px"}
                        ),
                        dbc.Button(
                            "𐌗",
                            id={"type": "factorio-remove-condition", "index": row_id},
                            n_clicks=0,
                            color="danger",
                            size="sm",
                            style={"width": "30px", "height": HEIGHT + 4, "marginTop": "10px"}
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
                id={"type": "factorio-logic-operator", "index": global_toggle_index},
                n_clicks=0,
                style={
                    "height": f"{ELEMENT_HEIGHT - 5}px",
                    "width": "60px",
                    "marginTop": "5px",
                    "position": "relative",
                    "top": f"{ELEMENT_HEIGHT / 2}px",
                }
            )
            global_toggle_index += 1

            toggles.append(and_or_button)

        return current_children, toggles, global_element_index, global_toggle_index

    @app.callback(
        Output({"type": "factorio-logic-operator", "index": MATCH}, "children"),
        Input({"type": "factorio-logic-operator", "index": MATCH}, "n_clicks"),
        State({"type": "factorio-logic-operator", "index": MATCH}, "children"),
        prevent_initial_call=True
    )
    def toggle_logic_operator(n_clicks, current_label):
        return "OR" if current_label == "AND" else "AND"

    @app.callback(
        Output({"type": "factorio-operator-dropdown", "index": MATCH}, "options"),
        Output({"type": "factorio-operator-dropdown", "index": MATCH}, "value"),
        Input({"type": "factorio-variable-dropdown", "index": MATCH}, "value"),
        prevent_initial_call=True
    )
    def update_operator_options(variable_value):
        if variable_value.startswith("metrics."):
            operators = OPERATORS["metrics"]
            default_value = ">"
        elif variable_value == "name":
            operators = OPERATORS["name"]
            default_value = "~="
        elif variable_value == "type":
            operators = OPERATORS["type"]
            default_value = "~="
        else:
            operators = OPERATORS["metrics"]
            default_value = ">"

        options = [{"label": op, "value": op} for op in operators]
        return options, default_value

    @app.callback(
        [
            Output("factorio-conditions-container", "children", allow_duplicate=True),
            Output("factorio-toggle-container", "children", allow_duplicate=True)
        ],
        Input({"type": "factorio-remove-condition", "index": ALL}, "n_clicks"),
        State("factorio-conditions-container", "children"),
        State("factorio-toggle-container", "children"),
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
            if remove_toggle_id < len(toggles):
                toggles.pop(remove_toggle_id)

        return rows, toggles

    @app.callback(
        Output("dummy-output", "data", allow_duplicate=True),
        Input({"type": "factorio-logic-operator", "index": ALL}, "children"),
        Input({"type": "factorio-variable-dropdown", "index": ALL}, "value"),
        Input({"type": "factorio-operator-dropdown", "index": ALL}, "value"),
        Input({"type": "factorio-value-input", "index": ALL}, "value"),
        State("layers-store", "data"),
        prevent_initial_call=True
    )
    def apply_filter(logic_ops, variable_values, operator_values, input_values, layers):
        if not variable_values or not layers:
            return dash.no_update

        # Build tokens for the expression function
        tokens = []
        if variable_values:
            tokens.append({
                "variable": variable_values[0],
                "operator": operator_values[0],
                "value": input_values[0]
            })

            for i in range(len(logic_ops)):
                if i + 1 < len(variable_values):
                    tokens.append({
                        "logic_op": logic_ops[i],
                    })
                    tokens.append({
                        "variable": variable_values[i + 1],
                        "operator": operator_values[i + 1],
                        "value": input_values[i + 1]
                    })

        # If no tokens, return without filtering
        if not tokens:
            return dash.no_update

        # Create a function to evaluate operations against the filter
        expr = build_expression_function_with_regex(tokens)

        # Apply filter to layers and print matching operations
        filtered_operations = []
        for layer in layers:
            layer_data = {
                "name": layer.get("name", ""),
                "type": layer.get("type", ""),
                "metrics": layer.get("metrics", {})
            }

            if expr(layer_data):
                filtered_operations.append(layer_data["name"])

        # Print filtered operations to console
        if filtered_operations:
            print("Filtered operations:", filtered_operations)

        return dash.no_update

def build_expression_function_with_regex(tokens):
    """Build an expression function that supports regex comparison for name and type."""
    if not tokens:
        return lambda value: False

    def evaluate_condition(condition, values):
        var_name = condition['variable']
        op_str = condition['operator']
        comp_value = condition['value']

        # Handle metrics.X path
        if var_name.startswith("metrics."):
            metric_name = var_name.split(".", 1)[1]
            if metric_name in values.get("metrics", {}):
                actual_value = values["metrics"][metric_name]
            else:
                return False
        else:
            # Handle name and type
            if var_name in values:
                actual_value = values[var_name]
            else:
                return False

        # Handle regex comparison for name and type
        if op_str == "~=":
            try:
                pattern = re.compile(comp_value, re.IGNORECASE)
                return bool(pattern.search(str(actual_value)))
            except re.error:
                # If regex is invalid, do a simple case-insensitive substring match
                return comp_value.lower() in str(actual_value).lower()
        else:
            # For numeric comparisons
            try:
                comp_value = float(comp_value)
                actual_value = float(actual_value)

                if op_str == ">":
                    return actual_value > comp_value
                elif op_str == "<":
                    return actual_value < comp_value
                elif op_str == ">=":
                    return actual_value >= comp_value
                elif op_str == "<=":
                    return actual_value <= comp_value
                elif op_str == "==":
                    return actual_value == comp_value
                elif op_str == "!=":
                    return actual_value != comp_value
                else:
                    return False
            except (ValueError, TypeError):
                return False

    def expression_function(values):
        groups = []
        current_group = [evaluate_condition(tokens[0], values)]
        i = 1

        while i < len(tokens):
            logic_op = tokens[i]['logic_op'].upper()
            i += 1
            cond_result = evaluate_condition(tokens[i], values)
            if logic_op == 'AND':
                current_group.append(cond_result)
            elif logic_op == 'OR':
                groups.append(current_group)
                current_group = [cond_result]
            else:
                raise ValueError(f"Unknown logical operator: {logic_op}")
            i += 1

        groups.append(current_group)

        return any(all(group) for group in groups)

    return expression_function
