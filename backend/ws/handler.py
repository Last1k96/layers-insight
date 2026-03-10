"""WebSocket endpoint and message dispatch."""
from __future__ import annotations

import json
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from backend.schemas.inference import InferenceTask


class ConnectionManager:
    """Manages WebSocket connections per session."""

    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        if session_id not in self._connections:
            self._connections[session_id] = set()
        self._connections[session_id].add(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket) -> None:
        if session_id in self._connections:
            self._connections[session_id].discard(websocket)
            if not self._connections[session_id]:
                del self._connections[session_id]

    async def broadcast(self, session_id: str, message: dict) -> None:
        """Broadcast message to all connections in a session."""
        if session_id not in self._connections:
            return
        dead = []
        for ws in self._connections[session_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections[session_id].discard(ws)

    async def send_task_status(self, task: InferenceTask) -> None:
        """Broadcast task status update."""
        message = {
            "type": "task_status",
            "task_id": task.task_id,
            "node_id": task.node_id,
            "node_name": task.node_name,
            "status": task.status.value,
            "stage": task.stage,
            "error_detail": task.error_detail,
            "metrics": task.metrics.model_dump() if task.metrics else None,
            "main_result": task.main_result.model_dump() if task.main_result else None,
            "ref_result": task.ref_result.model_dump() if task.ref_result else None,
        }
        await self.broadcast(task.session_id, message)


# Global instance
ws_manager = ConnectionManager()
