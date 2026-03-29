"""Shared OpenVINO graph utilities."""
from __future__ import annotations

from typing import Any


def get_reachable_params(model: Any, target_outputs) -> list:
    """Walk backward from target outputs to find only reachable Parameter nodes.

    Uses name-based matching -- OV Python bindings may create new wrapper
    objects each call, so id() is unreliable.
    """
    param_by_name = {p.get_friendly_name(): p for p in model.get_parameters()}
    visited: set[str] = set()
    params: list = []
    stack = list(target_outputs)

    while stack:
        output = stack.pop()
        node = output.get_node()
        node_name = node.get_friendly_name()
        if node_name in visited:
            continue
        visited.add(node_name)

        if node_name in param_by_name:
            params.append(param_by_name[node_name])
            continue

        for i in range(node.get_input_size()):
            source_output = node.input(i).get_source_output()
            stack.append(source_output)

    return params
