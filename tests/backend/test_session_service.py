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
def sample_config(tmp_path):
    # Create a real model file so create_session can copy it
    model_xml = tmp_path / "models" / "model.xml"
    model_xml.parent.mkdir(parents=True, exist_ok=True)
    model_xml.write_text("<model/>")
    model_xml.with_suffix(".bin").write_bytes(b"\x00" * 16)
    return SessionConfig(
        model_path=str(model_xml),
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
        assert detail.config.model_path.endswith("model.xml")
        assert info.id in detail.config.model_path  # path is inside session dir

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

    def test_task_result_persistence(self, session_service, sample_config, tmp_path):
        info = session_service.create_session(sample_config)

        task_data = {"task_id": "t1", "status": "success", "node_name": "conv1"}

        # Create a temporary artifacts directory simulating worker output
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        main_output = np.random.randn(1, 64, 112, 112).astype(np.float32)
        ref_output = np.random.randn(1, 64, 112, 112).astype(np.float32)
        np.save(str(artifacts_dir / "main_output.npy"), main_output)
        np.save(str(artifacts_dir / "ref_output.npy"), ref_output)
        (artifacts_dir / "cut_model.xml").write_text("<model/>")
        (artifacts_dir / "cut_model.bin").write_bytes(b"\x00" * 16)
        np.save(str(artifacts_dir / "input_data.npy"), np.zeros((1, 3)))

        session_service.save_task_result(
            info.id, "t1", task_data,
            artifacts_dir=str(artifacts_dir),
        )

        # Verify task metadata
        result = session_service.load_task_result(info.id, "t1")
        assert result["task_id"] == "t1"
        assert result["status"] == "success"

        # Verify all artifact files were moved — folder is named after the node
        output_dir = session_service._session_path(info.id) / "output" / "conv1"
        assert (output_dir / "main_output.npy").exists()
        assert (output_dir / "ref_output.npy").exists()
        assert (output_dir / "cut_model.xml").exists()
        assert (output_dir / "cut_model.bin").exists()
        assert (output_dir / "input_data.npy").exists()

        # Verify get_tensor_path resolves via task_id -> tensor_dir mapping
        path = session_service.get_tensor_path(info.id, "t1", "main_output")
        assert path is not None
        assert "conv1" in str(path)

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

    def test_get_tensor_path(self, session_service, sample_config, tmp_path):
        info = session_service.create_session(sample_config)

        # No tensor yet
        assert session_service.get_tensor_path(info.id, "t1", "main_output") is None

        # Save via artifacts_dir
        artifacts_dir = tmp_path / "artifacts2"
        artifacts_dir.mkdir()
        np.save(str(artifacts_dir / "main_output.npy"), np.zeros((1, 3)))

        session_service.save_task_result(
            info.id, "t1", {"status": "success", "node_name": "relu1"},
            artifacts_dir=str(artifacts_dir),
        )

        path = session_service.get_tensor_path(info.id, "t1", "main_output")
        assert path is not None
        assert path.exists()
        assert "relu1" in str(path)

    def test_duplicate_node_name_folders(self, session_service, sample_config, tmp_path):
        """Re-inferring the same node creates a new uniquely-named folder."""
        info = session_service.create_session(sample_config)

        for i, task_id in enumerate(["t1", "t2"]):
            artifacts_dir = tmp_path / f"dup_artifacts_{i}"
            artifacts_dir.mkdir()
            np.save(str(artifacts_dir / "main_output.npy"), np.zeros((1, 3)))
            session_service.save_task_result(
                info.id, task_id,
                {"status": "success", "node_name": "conv1"},
                artifacts_dir=str(artifacts_dir),
            )

        output_dir = session_service._session_path(info.id) / "output"
        assert (output_dir / "conv1").exists()
        assert (output_dir / "conv1_2").exists()

        # Both resolve correctly via task_id
        assert session_service.get_tensor_path(info.id, "t1", "main_output") is not None
        assert session_service.get_tensor_path(info.id, "t2", "main_output") is not None

    def test_get_sub_session_meta(self, session_service, sample_config):
        info = session_service.create_session(sample_config)
        sub = session_service.create_sub_session(
            session_id=info.id, cut_type="output",
            cut_node="conv1", grayed_nodes=["relu1"],
        )

        meta = session_service.get_sub_session_meta(info.id, sub.id)
        assert meta is not None
        assert meta["id"] == sub.id
        assert meta["cut_type"] == "output"

    def test_get_sub_session_meta_not_found(self, session_service, sample_config):
        info = session_service.create_session(sample_config)
        assert session_service.get_sub_session_meta(info.id, "nonexistent") is None

    def test_update_sub_session_meta(self, session_service, sample_config):
        info = session_service.create_session(sample_config)
        sub = session_service.create_sub_session(
            session_id=info.id, cut_type="input",
            cut_node="conv1", grayed_nodes=["param0"],
        )

        session_service.update_sub_session_meta(info.id, sub.id, {
            "model_path": "/tmp/cut_model.xml",
            "input_configs": [{"name": "cut_input_conv1", "source": "file", "path": "/tmp/in.npy"}],
        })

        meta = session_service.get_sub_session_meta(info.id, sub.id)
        assert meta["model_path"] == "/tmp/cut_model.xml"
        assert len(meta["input_configs"]) == 1
        assert meta["input_configs"][0]["name"] == "cut_input_conv1"

    def test_task_count_update(self, session_service, sample_config):
        info = session_service.create_session(sample_config)

        session_service.save_task_result(info.id, "t1", {"status": "success"})
        session_service.save_task_result(info.id, "t2", {"status": "failed"})
        session_service.save_task_result(info.id, "t3", {"status": "success"})

        detail = session_service.get_session(info.id)
        assert detail.info.task_count == 3
        assert detail.info.success_count == 2
        assert detail.info.failed_count == 1

    def test_chained_sub_sessions(self, session_service, sample_config):
        """Sub-sessions can chain: S1 -> S2, with parent_sub_session_id."""
        info = session_service.create_session(sample_config)

        s1 = session_service.create_sub_session(
            session_id=info.id,
            cut_type="input",
            cut_node="conv1",
            grayed_nodes=["param0"],
        )
        assert s1.parent_id == info.id
        assert s1.ancestor_cuts == []

        s2 = session_service.create_sub_session(
            session_id=info.id,
            cut_type="input",
            cut_node="conv2",
            grayed_nodes=["param0", "conv1_upstream"],
            parent_sub_session_id=s1.id,
            ancestor_cuts=[{"cut_node": "conv1", "cut_type": "input"}],
        )
        assert s2.parent_id == s1.id
        assert s2.ancestor_cuts == [{"cut_node": "conv1", "cut_type": "input"}]

        # Verify persisted
        subs = session_service.list_sub_sessions(info.id)
        assert len(subs) == 2
        s2_loaded = [s for s in subs if s.id == s2.id][0]
        assert s2_loaded.parent_id == s1.id
        assert s2_loaded.ancestor_cuts == [{"cut_node": "conv1", "cut_type": "input"}]

    def test_find_task_for_node_root(self, session_service, sample_config):
        """find_task_for_node returns root tasks when no sub_session_id given."""
        info = session_service.create_session(sample_config)
        session_service.save_task_result(info.id, "t1", {
            "status": "success", "node_name": "conv1",
        })
        assert session_service.find_task_for_node(info.id, "conv1") == "t1"
        assert session_service.find_task_for_node(info.id, "relu1") is None

    def test_find_task_for_node_sub_session(self, session_service, sample_config):
        """find_task_for_node searches sub-session tasks first, then falls back to root."""
        info = session_service.create_session(sample_config)

        # Root task
        session_service.save_task_result(info.id, "t_root", {
            "status": "success", "node_name": "conv1",
        })

        # Sub-session task
        s1 = session_service.create_sub_session(
            session_id=info.id, cut_type="input",
            cut_node="relu0", grayed_nodes=["param0"],
        )
        session_service.save_task_result(info.id, "t_sub", {
            "status": "success", "node_name": "conv1", "sub_session_id": s1.id,
        })

        # With sub_session_id, should find the sub-session task
        assert session_service.find_task_for_node(info.id, "conv1", s1.id) == "t_sub"

        # Node only in root — should fall back
        session_service.save_task_result(info.id, "t_root2", {
            "status": "success", "node_name": "bn1",
        })
        assert session_service.find_task_for_node(info.id, "bn1", s1.id) == "t_root2"

    def test_find_task_for_node_ancestor_walk(self, session_service, sample_config):
        """find_task_for_node walks up the parent chain."""
        info = session_service.create_session(sample_config)

        s1 = session_service.create_sub_session(
            session_id=info.id, cut_type="input",
            cut_node="conv1", grayed_nodes=["param0"],
        )
        session_service.save_task_result(info.id, "t_s1", {
            "status": "success", "node_name": "relu1", "sub_session_id": s1.id,
        })

        s2 = session_service.create_sub_session(
            session_id=info.id, cut_type="input",
            cut_node="conv2", grayed_nodes=["param0", "conv1"],
            parent_sub_session_id=s1.id,
            ancestor_cuts=[{"cut_node": "conv1", "cut_type": "input"}],
        )

        # s2 has no task for relu1, but s1 (parent) does
        assert session_service.find_task_for_node(info.id, "relu1", s2.id) == "t_s1"

    def test_sub_session_cascade_delete(self, session_service, sample_config, tmp_path):
        """Deleting a parent sub-session cascades to children."""
        info = session_service.create_session(sample_config)

        parent = session_service.create_sub_session(
            session_id=info.id, cut_type="output",
            cut_node="conv1", grayed_nodes=["relu1"],
        )
        child = session_service.create_sub_session(
            session_id=info.id, cut_type="input",
            cut_node="relu1", grayed_nodes=["param0"],
            parent_sub_session_id=parent.id,
        )

        # Add tasks to both sub-sessions
        session_service.save_task_result(info.id, "tp", {
            "status": "success", "node_name": "conv1", "sub_session_id": parent.id,
        })
        session_service.save_task_result(info.id, "tc", {
            "status": "success", "node_name": "relu1", "sub_session_id": child.id,
        })

        # Delete parent — child should also be removed
        assert session_service.delete_sub_session(info.id, parent.id) is True

        subs = session_service.list_sub_sessions(info.id)
        assert len(subs) == 0

        # Tasks associated with deleted sub-sessions should also be removed
        detail = session_service.get_session(info.id)
        assert detail.info.task_count == 0

    def test_delete_task_removes_metadata_and_tensors(self, session_service, sample_config, tmp_path):
        """delete_task removes both metadata and tensor directory."""
        info = session_service.create_session(sample_config)

        artifacts_dir = tmp_path / "del_artifacts"
        artifacts_dir.mkdir()
        np.save(str(artifacts_dir / "main_output.npy"), np.zeros((1, 3)))

        session_service.save_task_result(
            info.id, "t1",
            {"status": "success", "node_name": "conv1"},
            artifacts_dir=str(artifacts_dir),
        )

        # Verify output dir exists
        tensor_dir = session_service._session_path(info.id) / "output" / "conv1"
        assert tensor_dir.exists()

        # Delete task
        assert session_service.delete_task(info.id, "t1") is True

        # Tensor dir should be gone
        assert not tensor_dir.exists()

        # Metadata should be gone
        result = session_service.load_task_result(info.id, "t1")
        assert result == {}

        # Counts should be updated
        detail = session_service.get_session(info.id)
        assert detail.info.task_count == 0

    def test_get_sub_session_meta_resolved_paths(self, session_service, sample_config):
        """get_sub_session_meta_resolved returns absolute paths."""
        info = session_service.create_session(sample_config)
        sub = session_service.create_sub_session(
            session_id=info.id, cut_type="input",
            cut_node="conv1", grayed_nodes=["param0"],
        )

        session_service.update_sub_session_meta(info.id, sub.id, {
            "model_path": "sub_sessions/conv1/cut_model.xml",
            "input_configs": [
                {"name": "input", "source": "file", "path": "sub_sessions/conv1/inputs/input.npy"},
            ],
        })

        resolved = session_service.get_sub_session_meta_resolved(info.id, sub.id)
        assert resolved is not None
        # Paths should be absolute
        assert resolved["model_path"].startswith("/")
        assert resolved["input_configs"][0]["path"].startswith("/")

    def test_clone_session_basic(self, session_service, sample_config):
        """Clone creates a new session from the source."""
        info = session_service.create_session(sample_config)
        result = session_service.clone_session(info.id, {})
        assert result is not None
        new_info, inferred_nodes = result
        assert new_info.id != info.id
        assert new_info.model_name == info.model_name
        assert inferred_nodes == []

    def test_clone_session_with_overrides(self, session_service, sample_config):
        """Clone applies device overrides."""
        info = session_service.create_session(sample_config)
        result = session_service.clone_session(info.id, {
            "main_device": "GPU",
            "ref_device": "GPU",
        })
        new_info, _ = result
        assert new_info.main_device == "GPU"
        assert new_info.ref_device == "GPU"

    def test_clone_preserves_source_id(self, session_service, sample_config):
        """Cloned session stores source_session_id."""
        info = session_service.create_session(sample_config)
        result = session_service.clone_session(info.id, {})
        new_info, _ = result
        assert session_service.get_source_session_id(new_info.id) == info.id

    def test_clone_returns_inferred_nodes_sorted(self, session_service, sample_config):
        """Clone returns inferred nodes sorted by worst accuracy."""
        info = session_service.create_session(sample_config)
        session_service.save_task_result(info.id, "t1", {
            "status": "success", "node_name": "conv1",
            "metrics": {"cosine_similarity": 0.99, "mse": 0.001, "max_abs_diff": 0.01},
        })
        session_service.save_task_result(info.id, "t2", {
            "status": "success", "node_name": "relu1",
            "metrics": {"cosine_similarity": 0.5, "mse": 0.1, "max_abs_diff": 0.5},
        })

        result = session_service.clone_session(info.id, {})
        _, inferred_nodes = result
        assert len(inferred_nodes) == 2
        assert inferred_nodes[0]["node_name"] == "relu1"  # worst first
        assert inferred_nodes[1]["node_name"] == "conv1"

    def test_compare_sessions_basic(self, session_service, sample_config):
        """Compare two sessions returns correct deltas and summary."""
        s1 = session_service.create_session(sample_config)
        s2 = session_service.create_session(sample_config)

        session_service.save_task_result(s1.id, "t1", {
            "status": "success", "node_name": "conv1",
            "metrics": {"cosine_similarity": 0.90, "mse": 0.01, "max_abs_diff": 0.1},
        })
        session_service.save_task_result(s2.id, "t2", {
            "status": "success", "node_name": "conv1",
            "metrics": {"cosine_similarity": 0.95, "mse": 0.005, "max_abs_diff": 0.05},
        })

        result = session_service.compare_sessions(s1.id, s2.id)
        assert result["summary"]["total_compared"] == 1
        assert result["summary"]["improved"] == 1
        conv1 = result["nodes"][0]
        assert conv1["delta_cosine"] == pytest.approx(0.05, abs=1e-6)
        assert conv1["delta_mse"] == pytest.approx(-0.005, abs=1e-6)

    def test_compare_sessions_only_in_one(self, session_service, sample_config):
        """Nodes only in one session are counted correctly."""
        s1 = session_service.create_session(sample_config)
        s2 = session_service.create_session(sample_config)

        session_service.save_task_result(s1.id, "t1", {
            "status": "success", "node_name": "conv1",
            "metrics": {"cosine_similarity": 0.9, "mse": 0.01, "max_abs_diff": 0.1},
        })
        session_service.save_task_result(s2.id, "t2", {
            "status": "success", "node_name": "relu1",
            "metrics": {"cosine_similarity": 0.95, "mse": 0.005, "max_abs_diff": 0.05},
        })

        result = session_service.compare_sessions(s1.id, s2.id)
        assert result["summary"]["only_in_a"] == 1
        assert result["summary"]["only_in_b"] == 1
        assert result["summary"]["total_compared"] == 0

    def test_compare_sessions_unchanged(self, session_service, sample_config):
        """Nodes with tiny delta are counted as unchanged."""
        s1 = session_service.create_session(sample_config)
        s2 = session_service.create_session(sample_config)

        session_service.save_task_result(s1.id, "t1", {
            "status": "success", "node_name": "conv1",
            "metrics": {"cosine_similarity": 0.9999, "mse": 0.001, "max_abs_diff": 0.01},
        })
        session_service.save_task_result(s2.id, "t2", {
            "status": "success", "node_name": "conv1",
            "metrics": {"cosine_similarity": 0.99995, "mse": 0.0011, "max_abs_diff": 0.011},
        })

        result = session_service.compare_sessions(s1.id, s2.id)
        assert result["summary"]["unchanged"] == 1
