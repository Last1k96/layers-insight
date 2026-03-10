"""Session management and persistence service."""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

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

        model_name = Path(config.model_path).stem

        info = SessionInfo(
            id=session_id,
            model_path=config.model_path,
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
        main_output: Optional[np.ndarray] = None,
        ref_output: Optional[np.ndarray] = None,
    ) -> None:
        """Save task result metadata and output tensors."""
        # Update metadata
        meta = self._read_metadata(session_id)
        meta["tasks"][task_id] = task_data

        # Update counts
        tasks = meta["tasks"]
        meta["info"]["task_count"] = len(tasks)
        meta["info"]["success_count"] = sum(1 for t in tasks.values() if t.get("status") == "success")
        meta["info"]["failed_count"] = sum(1 for t in tasks.values() if t.get("status") == "failed")
        self._write_metadata(session_id, meta)

        # Save tensors
        task_dir = self._session_path(session_id) / "tensors" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        if main_output is not None:
            np.save(str(task_dir / "main_output.npy"), main_output)
        if ref_output is not None:
            np.save(str(task_dir / "ref_output.npy"), ref_output)

    def load_task_result(self, session_id: str, task_id: str) -> dict:
        """Load task result metadata."""
        meta = self._read_metadata(session_id)
        return meta.get("tasks", {}).get(task_id, {})

    def create_sub_session(
        self,
        session_id: str,
        cut_type: str,
        cut_node: str,
        grayed_nodes: list[str],
    ):
        """Create a sub-session within an existing session."""
        from backend.schemas.session import SubSessionInfo

        sub_id = str(uuid.uuid4())[:8]
        sub_path = self._session_path(session_id) / "sub_sessions" / sub_id
        sub_path.mkdir(parents=True, exist_ok=True)
        (sub_path / "tensors").mkdir(exist_ok=True)

        sub_info = SubSessionInfo(
            id=sub_id,
            parent_id=session_id,
            cut_type=cut_type,
            cut_node=cut_node,
            grayed_nodes=grayed_nodes,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        # Add to session metadata
        meta = self._read_metadata(session_id)
        meta.setdefault("sub_sessions", []).append(sub_info.model_dump())
        meta["info"]["sub_sessions"] = meta["sub_sessions"]
        self._write_metadata(session_id, meta)

        return sub_info

    def list_sub_sessions(self, session_id: str) -> list:
        """List sub-sessions for a session."""
        from backend.schemas.session import SubSessionInfo
        meta = self._read_metadata(session_id)
        return [SubSessionInfo(**s) for s in meta.get("sub_sessions", [])]

    def get_tensor_path(self, session_id: str, task_id: str, output_name: str) -> Optional[Path]:
        """Get path to a saved tensor file."""
        path = self._session_path(session_id) / "tensors" / task_id / f"{output_name}.npy"
        return path if path.exists() else None

    def _read_metadata(self, session_id: str) -> dict:
        return json.loads(self._metadata_path(session_id).read_text())

    def _write_metadata(self, session_id: str, metadata: dict) -> None:
        self._metadata_path(session_id).write_text(
            json.dumps(metadata, indent=2, default=str)
        )
