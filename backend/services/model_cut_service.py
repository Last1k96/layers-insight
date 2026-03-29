"""Model cutting service — make output/input node, create sub-sessions."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from backend.utils.ov_graph_utils import get_reachable_params


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
        reachable_params = get_reachable_params(model, new_outputs)
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
        input_data = np.load(input_npy_path)
        return self._build_input_cut(model_path, target_node_name, input_data)

    def make_input_node_random(
        self,
        model_path: str,
        target_node_name: str,
        precision: str = "f16",
    ) -> tuple[Any, np.ndarray, list[str]]:
        """Cut model so target node becomes a new input parameter with random data.

        Like make_input_node but generates random data from the node's
        shape/dtype instead of loading from .npy.

        Returns (cut_model, input_data, grayed_node_ids).
        """
        import openvino as ov
        from backend.utils.input_generator import generate_random_input, PRECISION_MAP

        model = self.core.read_model(model_path)
        target_op = self._find_op(model, target_node_name)
        if target_op is None:
            raise ValueError(f"Node '{target_node_name}' not found in model")

        # Get shape — must be static
        partial_shape = target_op.get_output_partial_shape(0)
        if partial_shape.is_dynamic:
            dim_str = ", ".join(str(d) if d.is_static else "?" for d in partial_shape)
            raise ValueError(
                f"Node '{target_node_name}' has dynamic shape [{dim_str}] — "
                f"cannot generate random data. Use 'input' cut type with a file instead."
            )
        shape = [d.get_length() for d in partial_shape]

        # Map OV element type to precision string for generate_random_input
        ov_type = target_op.get_output_element_type(0)
        ov_type_str = ov_type.to_dtype().name  # e.g. 'float32', 'int64'
        # Reverse-map numpy dtype name to our precision key
        dtype_to_precision = {v.__name__: k for k, v in PRECISION_MAP.items()}
        # numpy dtype names: 'float32', 'float16', 'int32', 'int64', 'uint8', 'int8', 'bool_'
        prec = dtype_to_precision.get(ov_type_str, precision)

        input_data = generate_random_input(shape, prec)
        return self._build_input_cut(model_path, target_node_name, input_data)

    def _build_input_cut(
        self,
        model_path: str,
        target_node_name: str,
        input_data: np.ndarray,
    ) -> tuple[Any, np.ndarray, list[str]]:
        """Shared logic for input cuts: replace target node with a Parameter.

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

        # Use the target node's actual output element type so the new
        # Parameter is type-compatible with downstream consumers.
        ov_type = target_op.get_output_element_type(0)

        # Create new Parameter to replace the target node's output
        new_param = ov.opset13.parameter(
            shape=ov.PartialShape(list(input_data.shape)),
            dtype=ov_type,
        )
        new_param.set_friendly_name(target_node_name)

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
        reachable_params = get_reachable_params(model, original_results)
        # Add new_param (it won't be in model.get_parameters())
        all_params = [new_param] + [p for p in reachable_params if id(p) != id(new_param)]

        cut_model = ov.Model(original_results, all_params, f"input_at_{target_node_name}")

        cut_model.validate_nodes_and_infer_types()

        # Determine grayed-out nodes (upstream of cut point, no longer in model)
        cut_node_names = {op.get_friendly_name() for op in cut_model.get_ordered_ops()}
        grayed_nodes = sorted(all_node_names - cut_node_names)

        return cut_model, input_data, grayed_nodes

    def _find_op(self, model: Any, name: str) -> Any:
        """Find an operation by friendly name."""
        for op in model.get_ordered_ops():
            if op.get_friendly_name() == name:
                return op
        return None
