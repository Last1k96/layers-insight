"""WebSocket integration tests."""
from __future__ import annotations

import pytest
from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient

from backend.ws.handler import ConnectionManager


class TestWebSocket:
    def test_connect_disconnect(self, test_session):
        """WebSocket connect and disconnect without errors."""
        manager = ConnectionManager()

        app = FastAPI()

        @app.websocket("/ws/{session_id}")
        async def ws_endpoint(websocket: WebSocket, session_id: str):
            await manager.connect(session_id, websocket)
            try:
                while True:
                    await websocket.receive_json()
            except Exception:
                manager.disconnect(session_id, websocket)

        client = TestClient(app)
        with client.websocket_connect(f"/ws/{test_session.id}") as ws:
            # Just connecting and disconnecting should work
            pass

    def test_broadcast(self, test_session):
        """Connected client receives broadcast messages."""
        manager = ConnectionManager()
        received = []

        app = FastAPI()

        @app.websocket("/ws/{session_id}")
        async def ws_endpoint(websocket: WebSocket, session_id: str):
            await manager.connect(session_id, websocket)
            # Send a message back immediately
            await manager.broadcast(session_id, {"type": "test", "data": "hello"})
            try:
                while True:
                    await websocket.receive_json()
            except Exception:
                manager.disconnect(session_id, websocket)

        client = TestClient(app)
        with client.websocket_connect(f"/ws/{test_session.id}") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "test"
            assert msg["data"] == "hello"
