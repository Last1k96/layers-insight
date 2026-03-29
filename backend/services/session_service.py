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
        """Convert an absolute path to a session-relative path string."""
        return str(abs_path.relative_to(self._session_path(session_id)))

    def _resolve_path(self, session_id: str, rel_path: str) -> Path:
        """Resolve a session-relative path to absolute."""
        return self._session_path(session_id) / rel_path

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for use as a folder name."""
        import re
        safe = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
        safe = re.sub(r'[^\w\-.]', '_', safe)
        # Collapse multiple underscores
        safe = re.sub(r'_+', '_', safe).strip('_')
        return safe or "unnamed"

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
        model_name = original_xml.stem
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
                    shutil.move(str(f), str(session_path / f.name))
            rel_model_path = "model.xml"
        else:
            # Copy model .xml into session, symlink .bin weights
            local_xml = session_path / original_xml.name
            shutil.copy2(str(original_xml), str(local_xml))

            original_bin = original_xml.with_suffix(".bin")
            if original_bin.exists():
                local_bin = session_path / original_bin.name
                local_bin.symlink_to(original_bin.resolve())

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
                    if src.exists():
                        dst = inputs_dir / src.name
                        shutil.copy2(str(src), str(dst))
                        # Store as session-relative path
                        updated_inputs.append(inp.model_copy(update={
                            "path": f"inputs/{src.name}",
                        }))
                    else:
                        updated_inputs.append(inp)
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
                    safe_name = inp.name.replace("/", "_").replace("\\", "_")
                    npy_path = inputs_dir / f"{safe_name}.npy"
                    data = generate_random_input(concrete, inp.data_type)
                    np.save(str(npy_path), data)
                    updated_inputs.append(inp.model_copy(update={
                        "source": "file",
                        "path": f"inputs/{safe_name}.npy",
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

        return SessionDetail(
            id=session_id,
            config=SessionConfig(**config_data),
            info=SessionInfo(**info_data),
            tasks=list(meta.get("tasks", {}).values()),
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
                task_data["tensor_base"] = f"sub_sessions/{sub_session_id}/output"
            meta["tasks"][task_id] = task_data
            self._write_metadata(session_id, meta)

            task_dir = tensors_base / folder_name
            task_dir.mkdir(parents=True, exist_ok=True)
            src = Path(artifacts_dir)
            for f in src.iterdir():
                if f.is_file() and f.name.endswith("_output.npy"):
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
                            and task_data.get("sub_session_id") == current_sub_id):
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
                    and not task_data.get("sub_session_id")):
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

    def list_sub_sessions(self, session_id: str) -> list:
        """List sub-sessions for a session."""
        from backend.schemas.session import SubSessionInfo
        meta = self._read_metadata(session_id)
        return [SubSessionInfo(**s) for s in meta.get("sub_sessions", [])]

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
        safe = node_name.replace("/", "_").replace("\\", "_")
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

    def _write_metadata(self, session_id: str, metadata: dict) -> None:
        self._atomic_write(
            self._metadata_path(session_id),
            json.dumps(metadata, indent=2, default=str),
        )
