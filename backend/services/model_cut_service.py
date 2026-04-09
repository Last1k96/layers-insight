"""Model cutting service — make output/input node, create sub-sessions."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.utils.ov_graph_utils import get_reachable_params


@dataclass
class CutResult:
    """Result of a model cut operation, used by the router to broadcast and respond."""
    sub_session: Any  # SubSessionInfo
    grayed_nodes: list[str]
    ancestor_cuts: list[dict]
    effective_cut_type: str


class ModelCutService:
    """Handles model cutting and sub-session creation."""

    def __init__(self, ov_core: Any):
        self.core = ov_core

    def perform_cut(
        self,
        session_svc: Any,
        session: Any,
        session_id: str,
        model: Any,
        node_name: str,
        cut_type: str,
        input_precision: str = "f16",
        parent_sub_session_id: Optional[str] = None,
    ) -> CutResult:
        """Perform the full model-cut workflow: cut, compute grayed nodes, create sub-session, save artifacts.

        This orchestrates the entire cut operation that was previously in the router:
        1. Resolve parent sub-session metadata (for chained cuts)
        2. Perform the OV model cut (output / input / input_random)
        3. Compute accumulated grayed nodes and ancestor chain
        4. Create the sub-session via session_svc
        5. Save the cut model and input configs to disk

        Returns a CutResult that the router can use for the WS broadcast and HTTP response.

        Raises ValueError for cut logic errors, KeyError-like errors for missing data.
        """
        # --- Step 1: Resolve parent sub-session metadata ---
        parent_sub, parent_sub_resolved, parent_grayed, \
            parent_input_configs_rel, parent_ancestor_cuts = \
            self._resolve_parent_metadata(session_svc, session_id, parent_sub_session_id)

        # --- Step 2: Perform the OV model cut ---
        source_model_path = session.config.model_path
        cut_model_obj, input_data, new_grayed = self._perform_ov_cut(
            session_svc=session_svc,
            session_id=session_id,
            model=model,
            node_name=node_name,
            cut_type=cut_type,
            input_precision=input_precision,
            source_model_path=source_model_path,
            parent_sub_session_id=parent_sub_session_id,
            parent_sub_resolved=parent_sub_resolved,
        )

        # Normalize cut_type: input_random behaves identically to input downstream
        effective_cut_type = "input" if cut_type == "input_random" else cut_type

        # --- Step 3: Compute accumulated grayed nodes and ancestor chain ---
        accumulated_grayed, ancestor_cuts = self._compute_grayed_and_ancestors(
            new_grayed=new_grayed,
            parent_grayed=parent_grayed,
            parent_ancestor_cuts=parent_ancestor_cuts,
            parent_sub=parent_sub,
        )

        # --- Step 4: Create the sub-session ---
        sub_session = session_svc.create_sub_session(
            session_id=session_id,
            cut_type=effective_cut_type,
            cut_node=node_name,
            grayed_nodes=accumulated_grayed,
            parent_sub_session_id=parent_sub_session_id,
            ancestor_cuts=ancestor_cuts,
        )

        # --- Step 5: Save cut model and input artifacts ---
        self._save_cut_artifacts(
            session_svc=session_svc,
            session_id=session_id,
            sub_session=sub_session,
            cut_model_obj=cut_model_obj,
            effective_cut_type=effective_cut_type,
            node_name=node_name,
            input_data=input_data,
            new_grayed=new_grayed,
            parent_input_configs_rel=parent_input_configs_rel,
        )

        return CutResult(
            sub_session=sub_session,
            grayed_nodes=accumulated_grayed,
            ancestor_cuts=ancestor_cuts,
            effective_cut_type=effective_cut_type,
        )

    # ------------------------------------------------------------------
    # Internal helpers for perform_cut
    # ------------------------------------------------------------------

    def _resolve_parent_metadata(
        self,
        session_svc: Any,
        session_id: str,
        parent_sub_session_id: Optional[str],
    ) -> tuple[Optional[dict], Optional[dict], list[str], list[dict], list[dict]]:
        """Resolve parent sub-session metadata for chained cuts.

        Returns (parent_sub, parent_sub_resolved, parent_grayed,
                 parent_input_configs_rel, parent_ancestor_cuts).
        """
        if not parent_sub_session_id:
            return None, None, [], [], []

        parent_sub = session_svc.get_sub_session_meta(session_id, parent_sub_session_id)
        parent_sub_resolved = session_svc.get_sub_session_meta_resolved(session_id, parent_sub_session_id)
        if parent_sub is None:
            raise ValueError("Parent sub-session not found")

        parent_grayed = parent_sub.get("grayed_nodes", [])
        parent_input_configs_rel = parent_sub.get("input_configs", [])
        parent_ancestor_cuts = parent_sub.get("ancestor_cuts", [])

        return parent_sub, parent_sub_resolved, parent_grayed, parent_input_configs_rel, parent_ancestor_cuts

    def _perform_ov_cut(
        self,
        session_svc: Any,
        session_id: str,
        model: Any,
        node_name: str,
        cut_type: str,
        input_precision: str,
        source_model_path: str,
        parent_sub_session_id: Optional[str],
        parent_sub_resolved: Optional[dict],
    ) -> tuple[Any, Optional[np.ndarray], list[str]]:
        """Execute the OV model cut. Returns (cut_model, input_data_or_None, new_grayed)."""
        input_data = None

        if cut_type == "output":
            if parent_sub_session_id and parent_sub_resolved:
                import openvino as ov
                parent_model_path = parent_sub_resolved.get("model_path")
                if not parent_model_path:
                    raise ValueError("Parent sub-session has no model")
                source_model = ov.Core().read_model(parent_model_path)
                cut_model_obj, new_grayed = self.make_output_node(source_model, node_name)
            else:
                cut_model_obj, new_grayed = self.make_output_node(model, node_name)

        elif cut_type in ("input", "input_random"):
            if parent_sub_session_id and parent_sub_resolved:
                ov_model_path = parent_sub_resolved.get("model_path", source_model_path)
            else:
                ov_model_path = source_model_path

            if cut_type == "input":
                task_id = session_svc.find_task_for_node(session_id, node_name, parent_sub_session_id)
                if task_id is None:
                    raise ValueError(
                        f"No successful inference found for node '{node_name}'. Run inference first."
                    )
                npy_path = session_svc.get_tensor_path(session_id, task_id, "main_output")
                if npy_path is None:
                    raise ValueError("Main output tensor not found")
                cut_model_obj, input_data, new_grayed = self.make_input_node(
                    ov_model_path, node_name, str(npy_path), input_precision,
                )
            else:
                cut_model_obj, input_data, new_grayed = self.make_input_node_random(
                    ov_model_path, node_name, input_precision,
                )
        else:
            raise ValueError(f"Invalid cut_type: {cut_type}")

        return cut_model_obj, input_data, new_grayed

    @staticmethod
    def _compute_grayed_and_ancestors(
        new_grayed: list[str],
        parent_grayed: list[str],
        parent_ancestor_cuts: list[dict],
        parent_sub: Optional[dict],
    ) -> tuple[list[str], list[dict]]:
        """Compute accumulated grayed nodes and the ancestor_cuts chain."""
        new_grayed_set = set(new_grayed)

        # Track which ancestor input-cut nodes are still reachable
        still_reachable_cuts = set()
        for ac in parent_ancestor_cuts:
            if ac["cut_type"] == "input" and ac["cut_node"] not in new_grayed_set:
                still_reachable_cuts.add(ac["cut_node"])
        if parent_sub and parent_sub.get("cut_type") == "input":
            if parent_sub["cut_node"] not in new_grayed_set:
                still_reachable_cuts.add(parent_sub["cut_node"])

        accumulated_grayed = list(
            (set(parent_grayed) | set(new_grayed)) - still_reachable_cuts
        )

        ancestor_cuts = parent_ancestor_cuts + [{
            "cut_node": parent_sub["cut_node"],
            "cut_type": parent_sub["cut_type"],
        }] if parent_sub else []

        return accumulated_grayed, ancestor_cuts

    def _save_cut_artifacts(
        self,
        session_svc: Any,
        session_id: str,
        sub_session: Any,
        cut_model_obj: Any,
        effective_cut_type: str,
        node_name: str,
        input_data: Optional[np.ndarray],
        new_grayed: list[str],
        parent_input_configs_rel: list[dict],
    ) -> None:
        """Serialize the cut model and input artifacts to the sub-session directory."""
        import openvino as ov

        sub_dir = session_svc._session_path(session_id) / "sub_sessions" / sub_session.id
        cut_model_abs = str(sub_dir / "cut_model.xml")
        ov.save_model(cut_model_obj, cut_model_abs)

        # Replace cut_model.bin with symlink to root session model.bin
        cut_bin = sub_dir / "cut_model.bin"
        meta = session_svc._read_metadata(session_id)
        model_xml_rel = meta["config"]["model_path"]
        session_path = session_svc._session_path(session_id)
        model_bin = session_path / Path(model_xml_rel).with_suffix(".bin")
        if cut_bin.exists() and model_bin.exists():
            cut_bin.unlink()
            cut_bin.symlink_to(model_bin.resolve())

        rel_cut_model = f"sub_sessions/{sub_session.id}/cut_model.xml"

        if effective_cut_type == "input":
            self._save_input_cut_artifacts(
                session_svc, session_id, sub_session, sub_dir,
                node_name, input_data, rel_cut_model, parent_input_configs_rel,
            )
        else:
            self._save_output_cut_artifacts(
                session_svc, session_id, sub_session, sub_dir,
                rel_cut_model, new_grayed, parent_input_configs_rel,
            )

    @staticmethod
    def _save_input_cut_artifacts(
        session_svc: Any,
        session_id: str,
        sub_session: Any,
        sub_dir: Path,
        node_name: str,
        input_data: Optional[np.ndarray],
        rel_cut_model: str,
        parent_input_configs_rel: list[dict],
    ) -> None:
        """Save .npy input and accumulated input_configs for an input cut."""
        inputs_dir = sub_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)

        param_name = node_name
        from backend.utils import sanitize_filename
        safe_filename = sanitize_filename(param_name)
        npy_save_path = str(inputs_dir / f"{safe_filename}.npy")
        np.save(npy_save_path, input_data)

        rel_npy_path = f"sub_sessions/{sub_session.id}/inputs/{safe_filename}.npy"
        new_config = {"name": param_name, "source": "file", "path": rel_npy_path}
        accumulated_configs = parent_input_configs_rel + [new_config]

        session_svc.update_sub_session_meta(session_id, sub_session.id, {
            "model_path": rel_cut_model,
            "input_configs": accumulated_configs,
        })

    @staticmethod
    def _save_output_cut_artifacts(
        session_svc: Any,
        session_id: str,
        sub_session: Any,
        sub_dir: Path,
        rel_cut_model: str,
        new_grayed: list[str],
        parent_input_configs_rel: list[dict],
    ) -> None:
        """Copy parent input files and save config for an output cut."""
        new_grayed_set = set(new_grayed)
        copied_configs: list[dict] = []

        from backend.utils import sanitize_filename
        for cfg in parent_input_configs_rel:
            if cfg.get("source") != "file" or not cfg.get("path"):
                copied_configs.append(cfg)
                continue
            if cfg["name"] in new_grayed_set:
                continue
            inputs_dir = sub_dir / "inputs"
            inputs_dir.mkdir(exist_ok=True)
            src_abs = str(session_svc._session_path(session_id) / cfg["path"])
            safe_filename = sanitize_filename(cfg["name"])
            dst_rel = f"sub_sessions/{sub_session.id}/inputs/{safe_filename}.npy"
            dst_abs = str(session_svc._session_path(session_id) / dst_rel)
            shutil.copy2(src_abs, dst_abs)
            copied_configs.append({**cfg, "path": dst_rel})

        session_svc.update_sub_session_meta(session_id, sub_session.id, {
            "model_path": rel_cut_model,
            "input_configs": copied_configs,
        })

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
