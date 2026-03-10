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

        new_outputs = [target_op.output(i) for i in range(target_op.get_output_size())]
        cut_model = ov.Model(new_outputs, model.get_parameters(), f"output_at_{target_node_name}")

        # Determine grayed-out nodes (in original but not in cut model)
        cut_node_names = {op.get_friendly_name() for op in cut_model.get_ordered_ops()}
        all_node_names = {op.get_friendly_name() for op in model.get_ordered_ops()}
        grayed_nodes = sorted(all_node_names - cut_node_names)

        return cut_model, grayed_nodes

    def make_input_node(
        self,
        model: Any,
        target_node_name: str,
        input_npy_path: str,
        precision: str = "f16",
    ) -> tuple[Any, np.ndarray, list[str]]:
        """Cut model so target node becomes a new input parameter.

        The saved main output .npy from a previous inference is used as
        the input data for the new parameter.

        Returns (cut_model, input_data, grayed_node_ids).
        """
        import openvino as ov

        target_op = self._find_op(model, target_node_name)
        if target_op is None:
            raise ValueError(f"Node '{target_node_name}' not found in model")

        # Load the saved output to use as input data
        input_data = np.load(input_npy_path)

        # Map precision string to OV element type
        precision_map = {
            "f16": ov.Type.f16,
            "fp16": ov.Type.f16,
            "f32": ov.Type.f32,
            "fp32": ov.Type.f32,
            "i32": ov.Type.i32,
            "i64": ov.Type.i64,
            "u8": ov.Type.u8,
        }
        ov_type = precision_map.get(precision, ov.Type.f16)

        # Create new Parameter to replace the target node's output
        new_param = ov.opset13.parameter(
            shape=ov.PartialShape(list(input_data.shape)),
            dtype=ov_type,
        )
        new_param.set_friendly_name(f"cut_input_{target_node_name}")

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

        cut_model = ov.Model(original_results, [new_param], f"input_at_{target_node_name}")

        # Determine grayed-out nodes (upstream of cut point, no longer in model)
        cut_node_names = {op.get_friendly_name() for op in cut_model.get_ordered_ops()}
        all_node_names = {op.get_friendly_name() for op in model.get_ordered_ops()}
        grayed_nodes = sorted(all_node_names - cut_node_names)

        return cut_model, input_data, grayed_nodes

    def _find_op(self, model: Any, name: str) -> Any:
        """Find an operation by friendly name."""
        for op in model.get_ordered_ops():
            if op.get_friendly_name() == name:
                return op
        return None
