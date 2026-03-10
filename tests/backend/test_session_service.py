"""Tests for session service."""
import json
import pytest
import tempfile
from pathlib import Path

import numpy as np

from backend.services.session_service import SessionService
from backend.schemas.session import SessionConfig


@pytest.fixture
def session_service(tmp_path):
    return SessionService(tmp_path / "sessions")


@pytest.fixture
def sample_config():
    return SessionConfig(
        model_path="/path/to/model.xml",
        main_device="CPU",
        ref_device="GPU",
    )


class TestSessionService:
    def test_create_session(self, session_service, sample_config):
        info = session_service.create_session(sample_config)

        assert info.id is not None
        assert info.model_name == "model"
        assert info.main_device == "CPU"
        assert info.ref_device == "GPU"

    def test_list_sessions(self, session_service, sample_config):
        session_service.create_session(sample_config)
        session_service.create_session(sample_config)

        sessions = session_service.list_sessions()
        assert len(sessions) == 2

    def test_get_session(self, session_service, sample_config):
        info = session_service.create_session(sample_config)

        detail = session_service.get_session(info.id)
        assert detail is not None
        assert detail.id == info.id
        assert detail.config.model_path == "/path/to/model.xml"

    def test_delete_session(self, session_service, sample_config):
        info = session_service.create_session(sample_config)
        assert session_service.delete_session(info.id)
        assert session_service.get_session(info.id) is None

    def test_delete_nonexistent(self, session_service):
        assert not session_service.delete_session("nonexistent")

    def test_graph_cache(self, session_service, sample_config):
        info = session_service.create_session(sample_config)
        graph_data = {"nodes": [{"id": "n1"}], "edges": []}

        session_service.save_graph_cache(info.id, graph_data)
        loaded = session_service.load_graph_cache(info.id)

        assert loaded == graph_data

    def test_task_result_persistence(self, session_service, sample_config):
        info = session_service.create_session(sample_config)

        task_data = {"task_id": "t1", "status": "success", "node_name": "conv1"}
        main_output = np.random.randn(1, 64, 112, 112).astype(np.float32)
        ref_output = np.random.randn(1, 64, 112, 112).astype(np.float32)

        session_service.save_task_result(
            info.id, "t1", task_data,
            main_output=main_output,
            ref_output=ref_output,
        )

        # Verify task metadata
        result = session_service.load_task_result(info.id, "t1")
        assert result["task_id"] == "t1"
        assert result["status"] == "success"

        # Verify tensor files
        tensor_dir = session_service._session_path(info.id) / "tensors" / "t1"
        assert (tensor_dir / "main_output.npy").exists()
        assert (tensor_dir / "ref_output.npy").exists()

    def test_create_sub_session(self, session_service, sample_config):
        info = session_service.create_session(sample_config)

        sub = session_service.create_sub_session(
            session_id=info.id,
            cut_type="output",
            cut_node="conv1",
            grayed_nodes=["relu1", "result_0"],
        )

        assert sub.id is not None
        assert sub.parent_id == info.id
        assert sub.cut_type == "output"
        assert sub.cut_node == "conv1"
        assert sub.grayed_nodes == ["relu1", "result_0"]

        # Verify persisted
        subs = session_service.list_sub_sessions(info.id)
        assert len(subs) == 1
        assert subs[0].id == sub.id

    def test_get_tensor_path(self, session_service, sample_config):
        info = session_service.create_session(sample_config)

        # No tensor yet
        assert session_service.get_tensor_path(info.id, "t1", "main_output") is None

        # Save a tensor
        session_service.save_task_result(
            info.id, "t1", {"status": "success"},
            main_output=np.zeros((1, 3)),
        )

        path = session_service.get_tensor_path(info.id, "t1", "main_output")
        assert path is not None
        assert path.exists()

    def test_task_count_update(self, session_service, sample_config):
        info = session_service.create_session(sample_config)

        session_service.save_task_result(info.id, "t1", {"status": "success"})
        session_service.save_task_result(info.id, "t2", {"status": "failed"})
        session_service.save_task_result(info.id, "t3", {"status": "success"})

        detail = session_service.get_session(info.id)
        assert detail.info.task_count == 3
        assert detail.info.success_count == 2
        assert detail.info.failed_count == 1
