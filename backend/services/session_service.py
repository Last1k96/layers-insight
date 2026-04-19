"""Session management and persistence service."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.schemas.session import SessionConfig, SessionDetail, SessionInfo


def _link_or_copy(src: Path, dst: Path) -> None:
    """Replace *dst* with a symlink, hard link, or copy of *src* (first that works)."""
    dst.unlink(missing_ok=True)
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
from backend.utils.input_generator import (
    generate_random_input,
    has_dynamic_dims,
    resolve_shape,
    validate_shape_bounds,
)


class SessionService:
    """Manages session lifecycle and persistence."""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir.resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _metadata_path(self, session_id: str) -> Path:
        return self._session_path(session_id) / "metadata.json"

    def _to_session_rel(self, session_id: str, abs_path: Path) -> str:
        """Convert an absolute path to a session-relative path string (always forward slashes)."""
        return abs_path.relative_to(self._session_path(session_id)).as_posix()

    def _resolve_path(self, session_id: str, rel_path: str) -> Path:
        """Resolve a session-relative path to absolute."""
        return self._session_path(session_id) / rel_path

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use as a folder name."""
        from backend.utils import sanitize_filename
        return sanitize_filename(name)

    def get_session_folder_size(self, session_id: str) -> int:
        """Calculate total disk usage of a session folder in bytes."""
        session_path = self._session_path(session_id)
        if not session_path.exists():
            return 0
        total = 0
        for f in session_path.rglob("*"):
            try:
                total += f.lstat().st_size
            except OSError:
                pass
        return total

    def create_session(self, config: SessionConfig, converted_dir: Optional[Path] = None) -> SessionInfo:
        """Create a new session directory with metadata.

        If converted_dir is provided, moves pre-converted .xml/.bin from that
        temp directory into the session folder instead of copying from original path.
        """
        original_xml = Path(config.model_path)
        model_name = config.session_name or original_xml.stem
        now = datetime.now(timezone.utc)
        safe_model = self._sanitize_name(model_name)
        session_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{safe_model}"
        # Ensure uniqueness
        base_id = session_id
        counter = 2
        while self._session_path(session_id).exists():
            session_id = f"{base_id}_{counter}"
            counter += 1
        session_path = self._session_path(session_id)
        session_path.mkdir(parents=True)
        (session_path / "tasks").mkdir()
        (session_path / "output").mkdir()
        (session_path / "runtime").mkdir()

        if converted_dir is not None:
            # Move pre-converted IR files from temp dir into session
            for f in converted_dir.iterdir():
                if f.is_file() and f.suffix in (".xml", ".bin"):
                    shutil.move(f, session_path / f.name)
            rel_model_path = "model.xml"
        else:
            # Copy model .xml and .bin into session (self-contained)
            local_xml = session_path / original_xml.name
            shutil.copy2(original_xml, local_xml)

            original_bin = original_xml.with_suffix(".bin")
            if original_bin.exists():
                local_bin = session_path / original_bin.name
                shutil.copy2(original_bin, local_bin)

            # Store session-relative model path (just the filename)
            rel_model_path = original_xml.name
        config = config.model_copy(update={"model_path": rel_model_path})

        # Generate and persist random inputs so they are reused across inferences
        if config.inputs:
            inputs_dir = session_path / "inputs"
            inputs_dir.mkdir(exist_ok=True)
            updated_inputs = []
            for inp in config.inputs:
                if inp.source == "file" and inp.path:
                    # User-provided file: copy into session inputs dir
                    src = Path(inp.path)
                    if not src.exists():
                        raise FileNotFoundError(
                            f"Input file not found for '{inp.name}': {inp.path}"
                        )
                    dst = inputs_dir / src.name
                    shutil.copy2(src, dst)
                    # Store as session-relative path (always forward slashes)
                    updated_inputs.append(inp.model_copy(update={
                        "path": Path("inputs", src.name).as_posix(),
                    }))
                elif inp.shape:
                    # Random source with known shape: generate and save
                    # Resolve dynamic dims if present
                    if has_dynamic_dims(inp.shape):
                        if not inp.resolved_shape:
                            # Can't generate random without concrete dims — defer to inference time
                            updated_inputs.append(inp)
                            continue
                        concrete = resolve_shape(inp.shape, inp.resolved_shape)
                        # Validate against bounds if provided
                        if inp.lower_bounds and inp.upper_bounds:
                            validate_shape_bounds(concrete, inp.lower_bounds, inp.upper_bounds)
                    else:
                        concrete = [d for d in inp.shape if isinstance(d, int)]
                    safe_name = self._sanitize_name(inp.name)
                    npy_path = inputs_dir / f"{safe_name}.npy"
                    data = generate_random_input(concrete, inp.data_type)
                    np.save(str(npy_path), data)
                    updated_inputs.append(inp.model_copy(update={
                        "source": "file",
                        "path": Path("inputs", f"{safe_name}.npy").as_posix(),
                    }))
                else:
                    # No shape yet — will be generated at inference time
                    updated_inputs.append(inp)
            config = config.model_copy(update={"inputs": updated_inputs})

        info = SessionInfo(
            id=session_id,
            model_path=rel_model_path,
            model_name=model_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            main_device=config.main_device,
            ref_device=config.ref_device,
        )

        metadata = {
            "schema_version": 2,
            "info": info.model_dump(),
            "config": config.model_dump(),
            "original_model_path": str(original_xml),
            "tasks": {},
            "sub_sessions": [],
        }
        self._write_metadata(session_id, metadata)
        return info

    def list_sessions(self) -> list[SessionInfo]:
        """List all sessions."""
        sessions = []
        if not self.sessions_dir.exists():
            return sessions
        for p in sorted(self.sessions_dir.iterdir()):
            if p.is_dir() and (p / "metadata.json").exists():
                try:
                    meta = self._read_metadata(p.name)
                    info_data = dict(meta["info"])
                    # Recompute counts from actual tasks dict
                    tasks = meta.get("tasks", {})
                    info_data["task_count"] = len(tasks)
                    info_data["success_count"] = sum(1 for t in tasks.values() if t.get("status") == "success")
                    # Resolve model_path to absolute for consumers
                    if "model_path" in info_data:
                        info_data["model_path"] = str(
                            self._resolve_path(p.name, info_data["model_path"])
                        )
                    info_data["folder_size"] = self.get_session_folder_size(p.name)
                    sessions.append(SessionInfo(**info_data))
                except Exception:
                    continue
        return sessions

    def get_session(self, session_id: str) -> Optional[SessionDetail]:
        """Get full session detail with resolved absolute paths."""
        meta_path = self._metadata_path(session_id)
        if not meta_path.exists():
            return None
        meta = self._read_metadata(session_id)

        # Resolve session-relative paths to absolute for consumers
        config_data = dict(meta["config"])
        config_data["model_path"] = str(self._resolve_path(session_id, config_data["model_path"]))
        if config_data.get("inputs"):
            resolved_inputs = []
            for inp in config_data["inputs"]:
                if inp.get("path"):
                    inp = dict(inp)
                    inp["path"] = str(self._resolve_path(session_id, inp["path"]))
                resolved_inputs.append(inp)
            config_data["inputs"] = resolved_inputs

        info_data = dict(meta["info"])
        info_data["model_path"] = str(self._resolve_path(session_id, info_data["model_path"]))
        # Recompute counts from actual tasks dict
        tasks = meta.get("tasks", {})
        info_data["task_count"] = len(tasks)
        info_data["success_count"] = sum(1 for t in tasks.values() if t.get("status") == "success")

        return SessionDetail(
            id=session_id,
            config=SessionConfig(**config_data),
            info=SessionInfo(**info_data),
            tasks=list(tasks.values()),
        )

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data."""
        session_path = self._session_path(session_id)
        if session_path.exists():
            shutil.rmtree(session_path)
            return True
        return False

    def save_graph_cache(self, session_id: str, graph_data: dict) -> None:
        """Save graph data with positions to session directory."""
        path = self._session_path(session_id) / "graph_cache.json"
        self._atomic_write(path, json.dumps(graph_data, default=str))

    def load_graph_cache(self, session_id: str) -> Optional[dict]:
        """Load cached graph data."""
        path = self._session_path(session_id) / "graph_cache.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def save_task_result(
        self,
        session_id: str,
        task_id: str,
        task_data: dict,
        artifacts_dir: Optional[str] = None,
        sub_session_id: Optional[str] = None,
    ) -> None:
        """Save task result metadata and output tensors.

        If artifacts_dir is provided, all files from that directory are moved
        into the task's tensor directory, creating a self-contained reproducer
        (cut model + inputs + outputs).

        If sub_session_id is provided, tensors are stored under the sub-session's
        own tensors/ directory instead of the root session's.
        """
        # Update metadata
        meta = self._read_metadata(session_id)
        meta["tasks"][task_id] = task_data

        # Update counts
        tasks = meta["tasks"]
        meta["info"]["task_count"] = len(tasks)
        meta["info"]["success_count"] = sum(1 for t in tasks.values() if t.get("status") == "success")
        meta["info"]["failed_count"] = sum(1 for t in tasks.values() if t.get("status") == "failed")
        self._write_metadata(session_id, meta)

        # Move artifacts (cut model, inputs, outputs) to task tensor dir
        # Name the folder after the node for easy discovery
        if artifacts_dir:
            node_name = task_data.get("node_name", task_id)

            # Determine base tensors directory
            if sub_session_id:
                tensors_base = self._session_path(session_id) / "sub_sessions" / sub_session_id / "output"
            else:
                tensors_base = self._session_path(session_id) / "output"
            tensors_base.mkdir(parents=True, exist_ok=True)

            folder_name = self._unique_tensor_folder_in(tensors_base, node_name)
            task_data["tensor_dir"] = folder_name
            if sub_session_id:
                task_data["tensor_base"] = Path("sub_sessions", sub_session_id, "output").as_posix()
            meta["tasks"][task_id] = task_data
            self._write_metadata(session_id, meta)

            task_dir = tensors_base / folder_name
            task_dir.mkdir(parents=True, exist_ok=True)
            # Only persist output tensors — cut model and inputs are
            # regenerated on demand during export to save disk space.
            src = Path(artifacts_dir)
            for f in src.iterdir():
                if f.is_file() and f.suffix == ".npy" and f.stem.startswith(("main_output", "ref_output")):
                    shutil.move(str(f), str(task_dir / f.name))

    def load_task_result(self, session_id: str, task_id: str) -> dict:
        """Load task result metadata."""
        meta = self._read_metadata(session_id)
        return meta.get("tasks", {}).get(task_id, {})

    def delete_task(self, session_id: str, task_id: str) -> bool:
        """Delete a task's metadata and tensor files."""
        meta = self._read_metadata(session_id)
        task_data = meta.get("tasks", {}).get(task_id)
        if task_data is None:
            return False

        # Remove tensor directory if it exists
        folder_name = task_data.get("tensor_dir")
        if folder_name:
            tensor_base = task_data.get("tensor_base")
            if tensor_base:
                tensor_dir = self._session_path(session_id) / tensor_base / folder_name
            else:
                tensor_dir = self._session_path(session_id) / "output" / folder_name
            if tensor_dir.exists():
                shutil.rmtree(tensor_dir, ignore_errors=True)

        # Remove from metadata
        del meta["tasks"][task_id]
        tasks = meta["tasks"]
        meta["info"]["task_count"] = len(tasks)
        meta["info"]["success_count"] = sum(1 for t in tasks.values() if t.get("status") == "success")
        meta["info"]["failed_count"] = sum(1 for t in tasks.values() if t.get("status") == "failed")
        self._write_metadata(session_id, meta)
        return True

    def find_task_for_node(
        self,
        session_id: str,
        node_name: str,
        sub_session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Find the most recent successful task_id for a node.

        When sub_session_id is given, searches tasks with matching sub_session_id first,
        then walks up the parent chain, then falls back to root session tasks.
        """
        meta = self._read_metadata(session_id)
        tasks = meta.get("tasks", {})

        if sub_session_id:
            # Search in the given sub-session first, then walk up ancestors
            current_sub_id: Optional[str] = sub_session_id
            while current_sub_id:
                for task_id, task_data in reversed(list(tasks.items())):
                    if (task_data.get("node_name") == node_name
                            and task_data.get("status") == "success"
                            and task_data.get("sub_session_id") == current_sub_id
                            and not task_data.get("reused")):
                        return task_id
                # Walk up to parent sub-session
                sub_meta = self.get_sub_session_meta(session_id, current_sub_id)
                if sub_meta:
                    parent_id = sub_meta.get("parent_id", "")
                    # If parent_id is the root session id, break to fall through
                    if parent_id == session_id:
                        break
                    current_sub_id = parent_id
                else:
                    break

        # Fall back to root session tasks (no sub_session_id)
        for task_id, task_data in reversed(list(tasks.items())):
            if (task_data.get("node_name") == node_name
                    and task_data.get("status") == "success"
                    and not task_data.get("sub_session_id")
                    and not task_data.get("reused")):
                return task_id
        return None

    def create_sub_session(
        self,
        session_id: str,
        cut_type: str,
        cut_node: str,
        grayed_nodes: list[str],
        parent_sub_session_id: Optional[str] = None,
        ancestor_cuts: Optional[list[dict]] = None,
    ):
        """Create a sub-session within an existing session."""
        from backend.schemas.session import SubSessionInfo

        safe_cut = self._sanitize_name(cut_node)
        sub_id = safe_cut
        # Ensure uniqueness among existing sub-sessions
        meta = self._read_metadata(session_id)
        existing_ids = {s["id"] for s in meta.get("sub_sessions", [])}
        base_id = sub_id
        counter = 2
        while sub_id in existing_ids:
            sub_id = f"{base_id}_{counter}"
            counter += 1
        sub_path = self._session_path(session_id) / "sub_sessions" / sub_id
        sub_path.mkdir(parents=True, exist_ok=True)
        (sub_path / "output").mkdir(exist_ok=True)
        (sub_path / "runtime").mkdir(exist_ok=True)

        sub_info = SubSessionInfo(
            id=sub_id,
            parent_id=parent_sub_session_id or session_id,
            cut_type=cut_type,
            cut_node=cut_node,
            grayed_nodes=grayed_nodes,
            ancestor_cuts=ancestor_cuts or [],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Add to session metadata (reuse meta from uniqueness check above)
        meta.setdefault("sub_sessions", []).append(sub_info.model_dump())
        meta["info"]["sub_sessions"] = meta["sub_sessions"]
        self._write_metadata(session_id, meta)

        return sub_info

    def get_sub_session_meta(self, session_id: str, sub_session_id: str) -> Optional[dict]:
        """Get raw metadata for a specific sub-session (paths are session-relative)."""
        meta = self._read_metadata(session_id)
        for s in meta.get("sub_sessions", []):
            if s.get("id") == sub_session_id:
                return s
        return None

    def get_sub_session_meta_resolved(self, session_id: str, sub_session_id: str) -> Optional[dict]:
        """Get sub-session metadata with paths resolved to absolute."""
        raw = self.get_sub_session_meta(session_id, sub_session_id)
        if raw is None:
            return None
        resolved = dict(raw)
        if resolved.get("model_path"):
            resolved["model_path"] = str(self._resolve_path(session_id, resolved["model_path"]))
        if resolved.get("input_configs"):
            resolved_configs = []
            for cfg in resolved["input_configs"]:
                cfg = dict(cfg)
                if cfg.get("path"):
                    cfg["path"] = str(self._resolve_path(session_id, cfg["path"]))
                resolved_configs.append(cfg)
            resolved["input_configs"] = resolved_configs
        return resolved

    def update_sub_session_meta(self, session_id: str, sub_session_id: str, updates: dict) -> None:
        """Update sub-session metadata with additional fields (e.g. model_path, input_configs)."""
        meta = self._read_metadata(session_id)
        for s in meta.get("sub_sessions", []):
            if s.get("id") == sub_session_id:
                s.update(updates)
                break
        self._write_metadata(session_id, meta)

    def set_tight_mode(self, session_id: str, sub_session_id: str, enabled: bool) -> None:
        self.update_sub_session_meta(session_id, sub_session_id, {"tight_mode": enabled})

    def _sub_session_dir(self, session_id: str, sub_session_id: str) -> Path:
        return self._session_path(session_id) / "sub_sessions" / sub_session_id

    def save_sub_session_tight_graph(
        self, session_id: str, sub_session_id: str, graph_data: dict,
    ) -> str:
        """Persist the sub-session's standalone tight graph to its own folder.

        Returns the session-relative path so it can be stored in metadata.
        """
        sub_dir = self._sub_session_dir(session_id, sub_session_id)
        sub_dir.mkdir(parents=True, exist_ok=True)
        path = sub_dir / "tight_graph.json"
        self._atomic_write(path, json.dumps(graph_data, default=str))
        return Path("sub_sessions", sub_session_id, "tight_graph.json").as_posix()

    def load_sub_session_tight_graph(
        self, session_id: str, sub_session_id: str,
    ) -> Optional[dict]:
        """Load the persisted tight graph for a sub-session, or None."""
        sub_meta = self.get_sub_session_meta(session_id, sub_session_id)
        if sub_meta is None:
            return None
        rel = sub_meta.get("tight_graph_path")
        if not rel:
            return None
        path = self._resolve_path(session_id, rel)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def list_sub_sessions(self, session_id: str) -> list:
        """List sub-sessions for a session."""
        from backend.schemas.session import SubSessionInfo
        meta = self._read_metadata(session_id)
        results = []
        for s in meta.get("sub_sessions", []):
            payload = dict(s)
            has_graph = bool(s.get("tight_graph_path"))
            payload["has_tight_layout"] = has_graph
            # Migration fallback: pre-tight_mode sub-sessions with a cached
            # tight layout default to ON so users don't lose their view state.
            payload["tight_mode"] = s.get("tight_mode", has_graph)
            results.append(SubSessionInfo(**payload))
        return results

    def delete_sub_session(self, session_id: str, sub_session_id: str) -> bool:
        """Delete a sub-session and all its descendants, files, and associated tasks."""
        meta = self._read_metadata(session_id)
        subs = meta.get("sub_sessions", [])

        # Find all descendant sub-session IDs (BFS)
        to_delete = {sub_session_id}
        changed = True
        while changed:
            changed = False
            for s in subs:
                if s["id"] not in to_delete and s.get("parent_id") in to_delete:
                    to_delete.add(s["id"])
                    changed = True

        if sub_session_id not in {s["id"] for s in subs}:
            return False

        # Remove sub-session metadata entries
        meta["sub_sessions"] = [s for s in subs if s["id"] not in to_delete]
        meta["info"]["sub_sessions"] = meta["sub_sessions"]

        # Remove tasks associated with deleted sub-sessions
        tasks = meta.get("tasks", {})
        meta["tasks"] = {
            tid: t for tid, t in tasks.items()
            if t.get("sub_session_id") not in to_delete
        }

        # Update counts
        tasks = meta["tasks"]
        meta["info"]["task_count"] = len(tasks)
        meta["info"]["success_count"] = sum(1 for t in tasks.values() if t.get("status") == "success")
        meta["info"]["failed_count"] = sum(1 for t in tasks.values() if t.get("status") == "failed")

        self._write_metadata(session_id, meta)

        # Delete sub-session directories from disk
        for sid in to_delete:
            sub_dir = self._session_path(session_id) / "sub_sessions" / sid
            if sub_dir.exists():
                shutil.rmtree(sub_dir)

        return True

    def get_tensor_path(self, session_id: str, task_id: str, output_name: str) -> Optional[Path]:
        """Get path to a saved tensor file."""
        meta = self._read_metadata(session_id)
        task_data = meta.get("tasks", {}).get(task_id, {})
        folder_name = task_data.get("tensor_dir", task_id)
        # Check if tensors are in a sub-session directory
        tensor_base = task_data.get("tensor_base")
        if tensor_base:
            path = self._session_path(session_id) / tensor_base / folder_name / f"{output_name}.npy"
        else:
            path = self._session_path(session_id) / "output" / folder_name / f"{output_name}.npy"
        return path if path.exists() else None

    def get_task_dir(self, session_id: str, task_id: str) -> Optional[Path]:
        """Get the directory containing a task's artifacts (model, tensors)."""
        meta = self._read_metadata(session_id)
        task_data = meta.get("tasks", {}).get(task_id, {})
        if not task_data:
            return None
        folder_name = task_data.get("tensor_dir", task_id)
        tensor_base = task_data.get("tensor_base")
        if tensor_base:
            path = self._session_path(session_id) / tensor_base / folder_name
        else:
            path = self._session_path(session_id) / "output" / folder_name
        return path if path.exists() else None

    def _unique_tensor_folder_in(self, tensors_dir: Path, node_name: str) -> str:
        """Return a unique folder name under given tensors dir based on node_name."""
        safe = self._sanitize_name(node_name)
        if not (tensors_dir / safe).exists():
            return safe
        i = 2
        while (tensors_dir / f"{safe}_{i}").exists():
            i += 1
        return f"{safe}_{i}"

    def _read_metadata(self, session_id: str) -> dict:
        return json.loads(self._metadata_path(session_id).read_text())

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Write content to file atomically via temp file + rename."""
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise

    def clone_session(
        self,
        source_session_id: str,
        overrides: dict,
    ) -> Optional[tuple["SessionInfo", list[dict]]]:
        """Clone a session with optional overrides.

        Returns (new_session_info, inferred_nodes_sorted_by_worst_accuracy) or None.
        inferred_nodes is a list of dicts with node_name, node_type, metrics.
        """
        source_meta = self._read_metadata(source_session_id)
        source_config_data = dict(source_meta["config"])
        source_session_path = self._session_path(source_session_id)

        # Resolve source model path to absolute
        source_model_abs = str(source_session_path / source_config_data["model_path"])

        # Build new config from source, applying overrides
        new_config_data = dict(source_config_data)
        # Restore model_path to absolute for create_session
        new_config_data["model_path"] = source_model_abs

        if overrides.get("model_path"):
            new_config_data["model_path"] = overrides["model_path"]
        if overrides.get("main_device"):
            new_config_data["main_device"] = overrides["main_device"]
        if overrides.get("ref_device"):
            new_config_data["ref_device"] = overrides["ref_device"]
        if overrides.get("inputs") is not None:
            new_config_data["inputs"] = overrides["inputs"]
        elif new_config_data.get("inputs"):
            # Resolve input paths to absolute from source session
            resolved_inputs = []
            for inp in new_config_data["inputs"]:
                inp = dict(inp)
                if inp.get("path") and not Path(inp["path"]).is_absolute():
                    inp["path"] = str(source_session_path / inp["path"])
                resolved_inputs.append(inp)
            new_config_data["inputs"] = resolved_inputs

        from backend.schemas.session import SessionConfig, InputConfig
        new_config = SessionConfig(**new_config_data)

        # Create the new session
        new_info = self.create_session(new_config)

        # Store source_session_id in the new session's metadata
        new_meta = self._read_metadata(new_info.id)
        new_meta["source_session_id"] = source_session_id
        if overrides.get("plugin_config"):
            new_meta["plugin_config"] = overrides["plugin_config"]
        self._write_metadata(new_info.id, new_meta)

        # Collect inferred nodes from source session, sorted by worst accuracy
        inferred_nodes = self._get_inferred_nodes_sorted(source_meta)

        return new_info, inferred_nodes

    def _get_inferred_nodes_sorted(self, meta: dict) -> list[dict]:
        """Extract successfully inferred nodes from metadata, sorted by worst accuracy (lowest cosine first)."""
        tasks = meta.get("tasks", {})
        nodes = []
        seen_names = set()
        for task_data in tasks.values():
            if task_data.get("status") != "success":
                continue
            node_name = task_data.get("node_name")
            if not node_name or node_name in seen_names:
                continue
            seen_names.add(node_name)
            metrics = task_data.get("metrics")
            nodes.append({
                "node_name": node_name,
                "node_type": task_data.get("node_type", ""),
                "node_id": task_data.get("node_id", ""),
                "metrics": metrics,
            })

        # Sort by worst accuracy: lowest cosine_similarity first
        def sort_key(n):
            m = n.get("metrics")
            if m and m.get("cosine_similarity") is not None:
                return m["cosine_similarity"]
            return 2.0  # Nodes without metrics go last
        nodes.sort(key=sort_key)
        return nodes

    def get_source_session_id(self, session_id: str) -> Optional[str]:
        """Get the source session ID if this session was cloned."""
        meta = self._read_metadata(session_id)
        return meta.get("source_session_id")

    def compare_sessions(self, session_a_id: str, session_b_id: str) -> dict:
        """Compare inference results between two sessions.

        Returns a dict with 'nodes' (list of per-node comparisons) and 'summary'.
        """
        meta_a = self._read_metadata(session_a_id)
        meta_b = self._read_metadata(session_b_id)

        # Build node -> best task metrics maps
        def build_node_map(meta: dict) -> dict:
            tasks = meta.get("tasks", {})
            node_map: dict[str, dict] = {}
            for task_data in tasks.values():
                if task_data.get("status") != "success":
                    continue
                name = task_data.get("node_name")
                if not name:
                    continue
                # Keep the latest successful task per node
                node_map[name] = {
                    "node_type": task_data.get("node_type", ""),
                    "metrics": task_data.get("metrics"),
                }
            return node_map

        map_a = build_node_map(meta_a)
        map_b = build_node_map(meta_b)

        all_names = set(map_a.keys()) | set(map_b.keys())

        TOLERANCE = 0.0001
        nodes = []
        summary = {
            "total_compared": 0,
            "improved": 0,
            "regressed": 0,
            "unchanged": 0,
            "only_in_a": 0,
            "only_in_b": 0,
        }

        for name in sorted(all_names):
            in_a = name in map_a
            in_b = name in map_b

            if in_a and not in_b:
                summary["only_in_a"] += 1
                nodes.append({
                    "node_name": name,
                    "node_type": map_a[name]["node_type"],
                    "metrics_a": map_a[name]["metrics"],
                    "metrics_b": None,
                    "delta_cosine": None,
                    "delta_mse": None,
                })
                continue
            if in_b and not in_a:
                summary["only_in_b"] += 1
                nodes.append({
                    "node_name": name,
                    "node_type": map_b[name]["node_type"],
                    "metrics_a": None,
                    "metrics_b": map_b[name]["metrics"],
                    "delta_cosine": None,
                    "delta_mse": None,
                })
                continue

            # Both present
            summary["total_compared"] += 1
            metrics_a = map_a[name]["metrics"]
            metrics_b = map_b[name]["metrics"]

            delta_cosine = None
            delta_mse = None

            if metrics_a and metrics_b:
                cos_a = metrics_a.get("cosine_similarity")
                cos_b = metrics_b.get("cosine_similarity")
                mse_a = metrics_a.get("mse")
                mse_b = metrics_b.get("mse")

                if cos_a is not None and cos_b is not None:
                    delta_cosine = cos_b - cos_a  # positive = improved
                if mse_a is not None and mse_b is not None:
                    delta_mse = mse_b - mse_a  # negative = improved

                # Classify: improved = cosine went up (or mse went down)
                if delta_cosine is not None:
                    if delta_cosine > TOLERANCE:
                        summary["improved"] += 1
                    elif delta_cosine < -TOLERANCE:
                        summary["regressed"] += 1
                    else:
                        summary["unchanged"] += 1
                else:
                    summary["unchanged"] += 1
            else:
                summary["unchanged"] += 1

            nodes.append({
                "node_name": name,
                "node_type": map_a[name]["node_type"],
                "metrics_a": metrics_a,
                "metrics_b": metrics_b,
                "delta_cosine": delta_cosine,
                "delta_mse": delta_mse,
            })

        return {"nodes": nodes, "summary": summary}

    def save_bisect_job(self, session_id: str, job_id: str, job_data: dict) -> None:
        """Persist bisect job info to session metadata (survives backend restart)."""
        meta = self._read_metadata(session_id)
        if "bisect_jobs" not in meta:
            meta["bisect_jobs"] = {}
        meta["bisect_jobs"][job_id] = job_data
        self._write_metadata(session_id, meta)

    def load_bisect_jobs(self, session_id: str) -> dict[str, dict]:
        """Load all persisted bisect jobs from session metadata."""
        meta = self._read_metadata(session_id)
        if "bisect_jobs" in meta:
            return meta["bisect_jobs"]
        # Legacy single-job format
        if "bisect_job" in meta:
            job = meta["bisect_job"]
            return {job["job_id"]: job} if "job_id" in job else {}
        return {}

    def merge_bisect_tasks(self, session_id: str, job_id: str) -> int:
        """Clear batch_id from bisect tasks so they appear as regular tasks.

        Deduplicates by node_name: if a node already has a non-bisect task,
        the bisect copy is deleted instead of merged.
        Returns the number of tasks affected.
        """
        meta = self._read_metadata(session_id)
        batch_id = f"bisect:{job_id}"
        tasks = meta.get("tasks", {})

        # Collect node_names that already have a non-bisect task
        existing_nodes = set()
        for td in tasks.values():
            if td.get("batch_id") != batch_id and not td.get("batch_id", "").startswith("bisect"):
                existing_nodes.add(td.get("node_name"))

        count = 0
        to_delete = []
        for task_id, task_data in tasks.items():
            if task_data.get("batch_id") == batch_id:
                if task_data.get("node_name") in existing_nodes:
                    to_delete.append(task_id)
                else:
                    task_data.pop("batch_id", None)
                count += 1
        for task_id in to_delete:
            del tasks[task_id]
        if count > 0:
            self._write_metadata(session_id, meta)
        return count

    def clear_bisect_job(self, session_id: str, job_id: str) -> None:
        """Remove a specific persisted bisect job from session metadata."""
        meta = self._read_metadata(session_id)
        changed = False
        if "bisect_jobs" in meta and job_id in meta["bisect_jobs"]:
            del meta["bisect_jobs"][job_id]
            changed = True
        # Also clean up legacy key if present
        if "bisect_job" in meta:
            if meta["bisect_job"].get("job_id") == job_id:
                del meta["bisect_job"]
                changed = True
        if changed:
            self._write_metadata(session_id, meta)

    def rename_session(self, session_id: str, new_name: str) -> Optional[str]:
        """Rename a session's display name and, when possible, its folder on disk.

        The session id has the form ``YYYYMMDD_HHMMSS_{safe_name}`` — the timestamp
        prefix stays put while ``{safe_name}`` is recomputed from ``new_name``.
        Tasks, bisect jobs, and sub-sessions referencing the old id in metadata
        are rewritten.

        Returns the resulting session id (unchanged if the name sanitizes to the
        same suffix or the id has no parseable timestamp), or ``None`` if the
        session does not exist.
        """
        import re

        old_path = self._session_path(session_id)
        if not old_path.exists():
            return None

        ts_match = re.match(r"^(\d{8}_\d{6})_", session_id)
        if ts_match:
            timestamp = ts_match.group(1)
            new_id = f"{timestamp}_{self._sanitize_name(new_name)}"
            if new_id != session_id:
                base_new = new_id
                counter = 2
                while self._session_path(new_id).exists():
                    new_id = f"{base_new}_{counter}"
                    counter += 1
                old_path.rename(self._session_path(new_id))
        else:
            new_id = session_id

        meta = self._read_metadata(new_id)
        meta["info"]["id"] = new_id
        meta["info"]["model_name"] = new_name

        if new_id != session_id:
            for t in meta.get("tasks", {}).values():
                if t.get("session_id") == session_id:
                    t["session_id"] = new_id
            for j in meta.get("bisect_jobs", {}).values():
                if j.get("session_id") == session_id:
                    j["session_id"] = new_id
            for s in meta.get("sub_sessions", []):
                if s.get("parent_id") == session_id:
                    s["parent_id"] = new_id

        self._write_metadata(new_id, meta)
        return new_id

    def _write_metadata(self, session_id: str, metadata: dict) -> None:
        self._atomic_write(
            self._metadata_path(session_id),
            json.dumps(metadata, indent=2, default=str),
        )
