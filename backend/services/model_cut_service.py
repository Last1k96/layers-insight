"""Model cutting service — make output/input node, create sub-sessions."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np


class ModelCutService:
    """Handles model cutting and sub-session creation."""

    def __init__(self, ov_core: Any):
        self.core = ov_core

    def make_output_node(self, model: Any, target_node_name: str) -> tuple[Any, list[str]]:
        """Cut model so target node becomes the only output.

        Returns (cut_model, grayed_node_ids) where grayed_node_ids are
        nodes downstream of the target that are no longer in the model.
        """
        import openvino as ov

        target_op = self._find_op(model, target_node_name)
        if target_op is None:
            raise ValueError(f"Node '{target_node_name}' not found in model")

        new_outputs = target_op.outputs()
        reachable_params = self._get_reachable_params(model, new_outputs)
        cut_model = ov.Model(new_outputs, reachable_params, f"output_at_{target_node_name}")
        cut_model.validate_nodes_and_infer_types()

        # Determine grayed-out nodes (in original but not in cut model)
        cut_node_names = {op.get_friendly_name() for op in cut_model.get_ordered_ops()}
        all_node_names = {op.get_friendly_name() for op in model.get_ordered_ops()}
        grayed_nodes = sorted(all_node_names - cut_node_names)

        return cut_model, grayed_nodes

    def make_input_node(
        self,
        model_path: str,
        target_node_name: str,
        input_npy_path: str,
        precision: str = "f16",
    ) -> tuple[Any, np.ndarray, list[str]]:
        """Cut model so target node becomes a new input parameter.

        Reads a fresh model from disk so the original in-memory model is
        never mutated.  The saved main output .npy from a previous
        inference is used as the input data for the new parameter.

        Returns (cut_model, input_data, grayed_node_ids).
        """
        import openvino as ov

        # Read a fresh copy — never mutate the cached original
        model = self.core.read_model(model_path)

        target_op = self._find_op(model, target_node_name)
        if target_op is None:
            raise ValueError(f"Node '{target_node_name}' not found in model")

        # Capture all node names BEFORE mutation
        all_node_names = {op.get_friendly_name() for op in model.get_ordered_ops()}

        # Load the saved output to use as input data
        input_data = np.load(input_npy_path)

        # Use the target node's actual output element type so the new
        # Parameter is type-compatible with downstream consumers.
        ov_type = target_op.get_output_element_type(0)

        # Create new Parameter to replace the target node's output
        new_param = ov.opset13.parameter(
            shape=ov.PartialShape(list(input_data.shape)),
            dtype=ov_type,
        )
        new_param.set_friendly_name(f"Parameter({target_node_name})")

        # Replace target's output consumers with the new parameter
        target_output = target_op.output(0)
        new_param_output = new_param.output(0)

        for target_input in list(target_output.get_target_inputs()):
            target_input.replace_source_output(new_param_output)

        # Collect downstream nodes to build new model
        # Get original results that are still reachable
        original_results = []
        for result in model.get_results():
            original_results.append(result.output(0))

        # Find all reachable parameters — the new_param plus any original
        # Parameters still reachable via paths that don't cross the cut point
        reachable_params = self._get_reachable_params(model, original_results)
        # Add new_param (it won't be in model.get_parameters())
        all_params = [new_param] + [p for p in reachable_params if id(p) != id(new_param)]

        cut_model = ov.Model(original_results, all_params, f"input_at_{target_node_name}")

        cut_model.validate_nodes_and_infer_types()

        # Determine grayed-out nodes (upstream of cut point, no longer in model)
        cut_node_names = {op.get_friendly_name() for op in cut_model.get_ordered_ops()}
        grayed_nodes = sorted(all_node_names - cut_node_names)

        return cut_model, input_data, grayed_nodes

    def _get_reachable_params(self, model: Any, target_outputs) -> list:
        """Walk backward from target outputs to find only reachable Parameter nodes."""
        # Use name-based matching — OV Python bindings may create new wrapper
        # objects each call, so id() is unreliable.
        param_by_name = {p.get_friendly_name(): p for p in model.get_parameters()}
        visited = set()
        params = []
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

    def _find_op(self, model: Any, name: str) -> Any:
        """Find an operation by friendly name."""
        for op in model.get_ordered_ops():
            if op.get_friendly_name() == name:
                return op
        return None
