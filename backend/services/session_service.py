"""Session management and persistence service."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.schemas.session import SessionConfig, SessionDetail, SessionInfo


class SessionService:
    """Manages session lifecycle and persistence."""

    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _metadata_path(self, session_id: str) -> Path:
        return self._session_path(session_id) / "metadata.json"

    def create_session(self, config: SessionConfig) -> SessionInfo:
        """Create a new session directory with metadata."""
        session_id = str(uuid.uuid4())[:8]
        session_path = self._session_path(session_id)
        session_path.mkdir(parents=True)
        (session_path / "tasks").mkdir()
        (session_path / "tensors").mkdir()

        # Copy model .xml into session, symlink .bin weights
        original_xml = Path(config.model_path)
        model_name = original_xml.stem
        local_xml = session_path / original_xml.name
        shutil.copy2(str(original_xml), str(local_xml))

        original_bin = original_xml.with_suffix(".bin")
        if original_bin.exists():
            local_bin = session_path / original_bin.name
            local_bin.symlink_to(original_bin.resolve())

        # Point config at the session-local copy
        config = config.model_copy(update={"model_path": str(local_xml)})

        info = SessionInfo(
            id=session_id,
            model_path=str(local_xml),
            model_name=model_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            main_device=config.main_device,
            ref_device=config.ref_device,
        )

        metadata = {
            "schema_version": 1,
            "info": info.model_dump(),
            "config": config.model_dump(),
            "tasks": {},
            "sub_sessions": [],  # Phase 2 extensibility
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
                    sessions.append(SessionInfo(**meta["info"]))
                except Exception:
                    continue
        return sessions

    def get_session(self, session_id: str) -> Optional[SessionDetail]:
        """Get full session detail."""
        meta_path = self._metadata_path(session_id)
        if not meta_path.exists():
            return None
        meta = self._read_metadata(session_id)
        return SessionDetail(
            id=session_id,
            config=SessionConfig(**meta["config"]),
            info=SessionInfo(**meta["info"]),
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
        path.write_text(json.dumps(graph_data, default=str))

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
    ) -> None:
        """Save task result metadata and output tensors.

        If artifacts_dir is provided, all files from that directory are moved
        into the task's tensor directory, creating a self-contained reproducer
        (cut model + inputs + outputs).
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
            folder_name = self._unique_tensor_folder(session_id, node_name)
            task_data["tensor_dir"] = folder_name
            meta["tasks"][task_id] = task_data
            self._write_metadata(session_id, meta)

            task_dir = self._session_path(session_id) / "tensors" / folder_name
            task_dir.mkdir(parents=True, exist_ok=True)
            src = Path(artifacts_dir)
            for f in src.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(task_dir / f.name))

    def load_task_result(self, session_id: str, task_id: str) -> dict:
        """Load task result metadata."""
        meta = self._read_metadata(session_id)
        return meta.get("tasks", {}).get(task_id, {})

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

        sub_id = str(uuid.uuid4())[:8]
        sub_path = self._session_path(session_id) / "sub_sessions" / sub_id
        sub_path.mkdir(parents=True, exist_ok=True)
        (sub_path / "tensors").mkdir(exist_ok=True)

        sub_info = SubSessionInfo(
            id=sub_id,
            parent_id=parent_sub_session_id or session_id,
            cut_type=cut_type,
            cut_node=cut_node,
            grayed_nodes=grayed_nodes,
            ancestor_cuts=ancestor_cuts or [],
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Add to session metadata
        meta = self._read_metadata(session_id)
        meta.setdefault("sub_sessions", []).append(sub_info.model_dump())
        meta["info"]["sub_sessions"] = meta["sub_sessions"]
        self._write_metadata(session_id, meta)

        return sub_info

    def get_sub_session_meta(self, session_id: str, sub_session_id: str) -> Optional[dict]:
        """Get metadata for a specific sub-session."""
        meta = self._read_metadata(session_id)
        for s in meta.get("sub_sessions", []):
            if s.get("id") == sub_session_id:
                return s
        return None

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

    def get_tensor_path(self, session_id: str, task_id: str, output_name: str) -> Optional[Path]:
        """Get path to a saved tensor file."""
        # Look up the tensor folder name from task metadata
        meta = self._read_metadata(session_id)
        task_data = meta.get("tasks", {}).get(task_id, {})
        folder_name = task_data.get("tensor_dir", task_id)
        path = self._session_path(session_id) / "tensors" / folder_name / f"{output_name}.npy"
        return path if path.exists() else None

    def _unique_tensor_folder(self, session_id: str, node_name: str) -> str:
        """Return a unique folder name under tensors/ based on node_name."""
        # Sanitize: replace path-unsafe chars with underscores
        safe = node_name.replace("/", "_").replace("\\", "_")
        tensors_dir = self._session_path(session_id) / "tensors"
        if not (tensors_dir / safe).exists():
            return safe
        # Append incrementing suffix
        i = 2
        while (tensors_dir / f"{safe}_{i}").exists():
            i += 1
        return f"{safe}_{i}"

    def _read_metadata(self, session_id: str) -> dict:
        return json.loads(self._metadata_path(session_id).read_text())

    def _write_metadata(self, session_id: str, metadata: dict) -> None:
        self._metadata_path(session_id).write_text(
            json.dumps(metadata, indent=2, default=str)
        )
