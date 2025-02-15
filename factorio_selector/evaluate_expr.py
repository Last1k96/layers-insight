import operator

ops = {
    '<': operator.lt,
    '>': operator.gt,
    '<=': operator.le,
    '>=': operator.ge,
    '==': operator.eq,
    '!=': operator.ne,
}


def evaluate_condition(condition, values):
    var_name = condition['variable']
    op_str = condition['operator']
    comp_value = condition['value']

    actual_value = values[var_name]

    comp_value = type(actual_value)(comp_value)

    return ops[op_str](actual_value, comp_value)


def build_expression_function(tokens):
    empty_func = lambda value: str()
    if not tokens:
        return empty_func

    for token in tokens:
        if any(v is None for k, v in token.items()):
            return empty_func

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


def run_tests():
    # Test 1: Single condition evaluation
    tokens = [
        {'variable': 'a', 'operator': '==', 'value': 'True'}
    ]
    expr = build_expression_function(tokens)
    assert expr({'a': True}) is True, "Test 1a failed: Expected True when a is True"
    assert expr({'a': False}) is False, "Test 1b failed: Expected False when a is False"

    # Test 2: AND-only evaluation
    tokens = [
        {'variable': 'a', 'operator': '==', 'value': 'True'},
        {'logic_op': 'AND'},
        {'variable': 'b', 'operator': '==', 'value': 'True'}
    ]
    expr = build_expression_function(tokens)
    assert expr({'a': True, 'b': True}) is True, "Test 2a failed: Expected True when both a and b are True"
    assert expr({'a': True, 'b': False}) is False, "Test 2b failed: Expected False when b is False"
    assert expr({'a': False, 'b': True}) is False, "Test 2c failed: Expected False when a is False"

    # Test 3: OR-only evaluation
    tokens = [
        {'variable': 'a', 'operator': '==', 'value': 'True'},
        {'logic_op': 'OR'},
        {'variable': 'b', 'operator': '==', 'value': 'True'}
    ]
    expr = build_expression_function(tokens)
    assert expr({'a': False, 'b': True}) is True, "Test 3a failed: Expected True when b is True"
    assert expr({'a': True, 'b': False}) is True, "Test 3b failed: Expected True when a is True"
    assert expr({'a': False, 'b': False}) is False, "Test 3c failed: Expected False when both are False"

    # Test 4: Mixed AND/OR with AND precedence: A OR B AND C
    # Expected grouping: A OR (B AND C)
    tokens = [
        {'variable': 'A', 'operator': '==', 'value': 'True'},
        {'logic_op': 'OR'},
        {'variable': 'B', 'operator': '==', 'value': 'True'},
        {'logic_op': 'AND'},
        {'variable': 'C', 'operator': '==', 'value': 'True'}
    ]
    expr = build_expression_function(tokens)
    # Case 4a: A=True, B=False, C=False -> True OR (False AND False) = True
    assert expr({'A': True, 'B': False, 'C': False}) is True, "Test 4a failed: Expected True"
    # Case 4b: A=False, B=True, C=False -> False OR (True AND False) = False
    assert expr({'A': False, 'B': True, 'C': False}) is False, "Test 4b failed: Expected False"
    # Case 4c: A=False, B=True, C=True -> False OR (True AND True) = True
    assert expr({'A': False, 'B': True, 'C': True}) is True, "Test 4c failed: Expected True"

    # Test 5: More complex: A AND B OR C AND D
    # Expected grouping with AND precedence: (A AND B) OR (C AND D)
    tokens = [
        {'variable': 'A', 'operator': '==', 'value': 'True'},
        {'logic_op': 'AND'},
        {'variable': 'B', 'operator': '==', 'value': 'True'},
        {'logic_op': 'OR'},
        {'variable': 'C', 'operator': '==', 'value': 'True'},
        {'logic_op': 'AND'},
        {'variable': 'D', 'operator': '==', 'value': 'True'}
    ]
    expr = build_expression_function(tokens)
    # Case 5a: A=True, B=True, C=False, D=True -> (True AND True) OR (False AND True) = True OR False = True
    assert expr({'A': True, 'B': True, 'C': False, 'D': True}) is True, "Test 5a failed: Expected True"
    # Case 5b: A=False, B=True, C=True, D=True -> (False AND True) OR (True AND True) = False OR True = True
    assert expr({'A': False, 'B': True, 'C': True, 'D': True}) is True, "Test 5b failed: Expected True"
    # Case 5c: A=True, B=False, C=False, D=False -> (True AND False) OR (False AND False) = False OR False = False
    assert expr({'A': True, 'B': False, 'C': False, 'D': False}) is False, "Test 5c failed: Expected False"

    # Test 6: Expression with multiple OR's and AND's: A OR B OR C AND D AND E
    # Expected grouping: A OR B OR (C AND D AND E)
    tokens = [
        {'variable': 'A', 'operator': '==', 'value': 'True'},
        {'logic_op': 'OR'},
        {'variable': 'B', 'operator': '==', 'value': 'True'},
        {'logic_op': 'OR'},
        {'variable': 'C', 'operator': '==', 'value': 'True'},
        {'logic_op': 'AND'},
        {'variable': 'D', 'operator': '==', 'value': 'True'},
        {'logic_op': 'AND'},
        {'variable': 'E', 'operator': '==', 'value': 'True'}
    ]
    expr = build_expression_function(tokens)
    # Case 6a: A=False, B=False, C=True, D=True, E=True -> False OR False OR (True AND True AND True) = True
    assert expr({'A': False, 'B': False, 'C': True, 'D': True, 'E': True}) is True, "Test 6a failed: Expected True"
    # Case 6b: A=False, B=False, C=True, D=True, E=False -> False OR False OR (True AND True AND False) = False
    assert expr({'A': False, 'B': False, 'C': True, 'D': True, 'E': False}) is False, "Test 6b failed: Expected False"
    # Case 6c: A=True, B=False, C=False, D=False, E=False -> True OR ... = True
    assert expr({'A': True, 'B': False, 'C': False, 'D': False, 'E': False}) is True, "Test 6c failed: Expected True"

    print("All tests passed successfully.")


if __name__ == "__main__":
    run_tests()