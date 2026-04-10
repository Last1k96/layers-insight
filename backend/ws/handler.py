"""WebSocket endpoint and message dispatch."""
from __future__ import annotations

import math
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from backend.schemas.inference import InferenceTask


def _sanitize_for_json(obj):
    """Replace float inf/nan with None for JavaScript JSON.parse compatibility."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


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

    async def close_all(self) -> None:
        """Close all WebSocket connections (used during shutdown)."""
        total = sum(len(s) for s in self._connections.values())
        print(f"[shutdown:ws] closing {total} connections across {len(self._connections)} sessions", flush=True)
        for session_id in list(self._connections):
            for ws in list(self._connections.get(session_id, [])):
                try:
                    print(f"[shutdown:ws]   closing ws for {session_id}...", flush=True)
                    await asyncio.wait_for(ws.close(), timeout=3)
                    print(f"[shutdown:ws]   closed.", flush=True)
                except asyncio.TimeoutError:
                    print(f"[shutdown:ws]   close timed out, forcing.", flush=True)
                except Exception as e:
                    print(f"[shutdown:ws]   close error: {e}", flush=True)
        self._connections.clear()
        print("[shutdown:ws] all cleared", flush=True)

    async def send_task_status(self, task: InferenceTask) -> None:
        """Broadcast task status update."""
        message = {
            "type": "task_status",
            "task_id": task.task_id,
            "session_id": task.session_id,
            "node_id": task.node_id,
            "node_name": task.node_name,
            "node_type": task.node_type,
            "status": task.status.value,
            "stage": task.stage,
            "error_detail": task.error_detail,
            "metrics": task.metrics.model_dump() if task.metrics else None,
            "main_result": task.main_result.model_dump() if task.main_result else None,
            "ref_result": task.ref_result.model_dump() if task.ref_result else None,
            "batch_id": task.batch_id,
            "sub_session_id": task.sub_session_id,
            "per_output_metrics": [m.model_dump() for m in task.per_output_metrics] if task.per_output_metrics else None,
            "per_output_main_results": [r.model_dump() for r in task.per_output_main_results] if task.per_output_main_results else None,
            "per_output_ref_results": [r.model_dump() for r in task.per_output_ref_results] if task.per_output_ref_results else None,
            "reused": task.reused,
        }
        await self.broadcast(task.session_id, _sanitize_for_json(message))


# Global instance
ws_manager = ConnectionManager()
