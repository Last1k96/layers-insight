"""Shared test fixtures."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI

from backend.config import AppConfig
from backend.routers import devices, graph, inference, sessions, tensors
from backend.services.queue_service import QueueService
from backend.services.session_service import SessionService
from backend.schemas.session import SessionConfig


@pytest.fixture
def sample_graph_data():
    """Minimal graph data for testing."""
    return {
        "nodes": [
            {"id": "param_0", "name": "input", "type": "Parameter", "shape": [1, 3, 224, 224]},
            {"id": "conv_0", "name": "conv1", "type": "Convolution", "shape": [1, 64, 112, 112]},
            {"id": "relu_0", "name": "relu1", "type": "Relu", "shape": [1, 64, 112, 112]},
            {"id": "result_0", "name": "output", "type": "Result", "shape": [1, 64, 112, 112]},
        ],
        "edges": [
            {"source": "param_0", "target": "conv_0", "source_port": 0, "target_port": 0},
            {"source": "conv_0", "target": "relu_0", "source_port": 0, "target_port": 0},
            {"source": "relu_0", "target": "result_0", "source_port": 0, "target_port": 0},
        ],
    }


@pytest.fixture
def sample_model_files(tmp_path):
    """Create dummy model XML/BIN files and return the XML path."""
    model_xml = tmp_path / "models" / "test_model.xml"
    model_xml.parent.mkdir(parents=True, exist_ok=True)
    model_xml.write_text("<model/>")
    model_xml.with_suffix(".bin").write_bytes(b"\x00" * 16)
    return model_xml


@pytest.fixture
def sample_config(sample_model_files):
    """SessionConfig pointing at the dummy model files."""
    return SessionConfig(
        model_path=str(sample_model_files),
        main_device="CPU",
        ref_device="CPU",
    )


@pytest.fixture
def test_app(tmp_path, sample_model_files):
    """Create a FastAPI app with mocked OV, real services against tmp_path."""
    config = AppConfig(
        sessions_dir=tmp_path / "sessions",
        model_path=str(sample_model_files),
    )

    app = FastAPI(title="Layers-Insight-Test")
    app.state.config = config
    app.state.ov_core = None  # No OV in tests
    app.state.models = {}
    app.state.inference_service = None
    app.state.model_cut_service = None

    session_service = SessionService(config.sessions_dir)
    app.state.session_service = session_service

    queue_service = QueueService()
    queue_service.set_callbacks(
        notify=AsyncMock(),
        infer=AsyncMock(),
    )
    app.state.queue_service = queue_service

    app.include_router(devices.router)
    app.include_router(sessions.router)
    app.include_router(graph.router)
    app.include_router(inference.router)
    app.include_router(tensors.router)

    return app


@pytest.fixture
def async_client(test_app):
    """httpx.AsyncClient wrapping the test app."""
    import httpx
    transport = httpx.ASGITransport(app=test_app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def test_session(test_app, sample_config):
    """Pre-create a session and return its SessionInfo."""
    svc = test_app.state.session_service
    return svc.create_session(sample_config)
