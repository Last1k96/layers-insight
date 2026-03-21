"""Tests for dynamic shape support across the pipeline."""
from __future__ import annotations

import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from backend.schemas.graph import GraphNode, NodeInput, GraphData
from backend.schemas.session import InputConfig, SessionConfig
from backend.services.session_service import SessionService
from backend.utils.input_generator import (
    resolve_shape,
    has_dynamic_dims,
    validate_shape_bounds,
    generate_random_input,
    prepare_inputs,
)


# ── Schema tests ──────────────────────────────────────────────────────


class TestGraphSchemasDynamicShapes:
    def test_graph_node_static_shape(self):
        node = GraphNode(id="n1", name="n1", type="Conv", shape=[1, 3, 224, 224])
        assert node.shape == [1, 3, 224, 224]

    def test_graph_node_dynamic_shape(self):
        node = GraphNode(id="n1", name="n1", type="Conv", shape=["?", 3, 224, 224])
        assert node.shape == ["?", 3, 224, 224]

    def test_graph_node_mixed_shape(self):
        node = GraphNode(id="n1", name="n1", type="Conv", shape=["?", 3, "?", 224])
        assert node.shape == ["?", 3, "?", 224]

    def test_graph_node_no_shape(self):
        node = GraphNode(id="n1", name="n1", type="Conv")
        assert node.shape is None

    def test_node_input_dynamic_shape(self):
        ni = NodeInput(name="src", port=0, shape=["?", 3, 224, 224])
        assert ni.shape == ["?", 3, 224, 224]


class TestInputConfigSchema:
    def test_static_shape(self):
        cfg = InputConfig(name="input", shape=[1, 3, 224, 224])
        assert cfg.shape == [1, 3, 224, 224]
        assert cfg.resolved_shape == []
        assert cfg.lower_bounds == []
        assert cfg.upper_bounds == []

    def test_dynamic_shape_with_bounds(self):
        cfg = InputConfig(
            name="input",
            shape=["?", 3, 224, 224],
            resolved_shape=[4],
            lower_bounds=[1, 3, 224, 224],
            upper_bounds=[16, 3, 224, 224],
        )
        assert cfg.shape == ["?", 3, 224, 224]
        assert cfg.resolved_shape == [4]
        assert cfg.lower_bounds == [1, 3, 224, 224]
        assert cfg.upper_bounds == [16, 3, 224, 224]

    def test_all_dynamic_dims(self):
        cfg = InputConfig(
            name="input",
            shape=["?", "?"],
            resolved_shape=[8, 16],
            lower_bounds=[1, 1],
            upper_bounds=[32, 64],
        )
        assert cfg.shape == ["?", "?"]
        assert cfg.resolved_shape == [8, 16]


# ── Helper function tests ─────────────────────────────────────────────


class TestHasDynamicDims:
    def test_static_shape(self):
        assert has_dynamic_dims([1, 3, 224, 224]) is False

    def test_dynamic_shape(self):
        assert has_dynamic_dims(["?", 3, 224, 224]) is True

    def test_all_dynamic(self):
        assert has_dynamic_dims(["?", "?"]) is True

    def test_empty_shape(self):
        assert has_dynamic_dims([]) is False


class TestResolveShape:
    def test_all_static(self):
        result = resolve_shape([1, 3, 224, 224])
        assert result == [1, 3, 224, 224]

    def test_single_dynamic_dim(self):
        result = resolve_shape(["?", 3, 224, 224], resolved=[4])
        assert result == [4, 3, 224, 224]

    def test_multiple_dynamic_dims(self):
        result = resolve_shape(["?", 3, "?", 224], resolved=[8, 112])
        assert result == [8, 3, 112, 224]

    def test_all_dynamic(self):
        result = resolve_shape(["?", "?", "?"], resolved=[1, 10, 20])
        assert result == [1, 10, 20]

    def test_no_resolved_raises(self):
        with pytest.raises(ValueError, match="dynamic.*no concrete value"):
            resolve_shape(["?", 3], resolved=None)

    def test_insufficient_resolved_raises(self):
        with pytest.raises(ValueError, match="dynamic.*no concrete value"):
            resolve_shape(["?", "?", 3], resolved=[4])

    def test_static_dims_ignored_in_resolved(self):
        """Static dims don't consume resolved values."""
        result = resolve_shape([1, "?", 3, "?"], resolved=[10, 20])
        assert result == [1, 10, 3, 20]


class TestValidateShapeBounds:
    def test_within_bounds(self):
        validate_shape_bounds([4, 3, 224, 224], [1, 3, 224, 224], [16, 3, 224, 224])

    def test_at_lower_bound(self):
        validate_shape_bounds([1, 3], [1, 3], [16, 3])

    def test_at_upper_bound(self):
        validate_shape_bounds([16, 3], [1, 3], [16, 3])

    def test_below_lower_bound_raises(self):
        with pytest.raises(ValueError, match="outside bounds"):
            validate_shape_bounds([0, 3], [1, 3], [16, 3])

    def test_above_upper_bound_raises(self):
        with pytest.raises(ValueError, match="outside bounds"):
            validate_shape_bounds([32, 3], [1, 3], [16, 3])

    def test_middle_dim_out_of_bounds(self):
        with pytest.raises(ValueError, match="Dimension 1"):
            validate_shape_bounds([4, 0, 224], [1, 1, 224], [16, 10, 224])


# ── prepare_inputs with dynamic shapes ─────────────────────────────


class TestPrepareInputsDynamic:
    def test_random_with_resolved_shape(self):
        params = [{"name": "input", "shape": ["?", 3, 8, 8]}]
        configs = [
            {
                "name": "input",
                "data_type": "fp32",
                "resolved_shape": [2],
                "lower_bounds": [1, 3, 8, 8],
                "upper_bounds": [16, 3, 8, 8],
            }
        ]
        inputs = prepare_inputs(params, input_configs=configs)
        assert inputs["input"].shape == (2, 3, 8, 8)
        assert inputs["input"].dtype == np.float32

    def test_random_without_resolved_raises(self):
        params = [{"name": "input", "shape": ["?", 3]}]
        configs = [{"name": "input", "data_type": "fp32"}]
        with pytest.raises(ValueError, match="dynamic.*no concrete value"):
            prepare_inputs(params, input_configs=configs)

    def test_random_out_of_bounds_raises(self):
        params = [{"name": "input", "shape": ["?", 3]}]
        configs = [
            {
                "name": "input",
                "data_type": "fp32",
                "resolved_shape": [32],
                "lower_bounds": [1, 3],
                "upper_bounds": [16, 3],
            }
        ]
        with pytest.raises(ValueError, match="outside bounds"):
            prepare_inputs(params, input_configs=configs)

    def test_file_input_with_bounds_validation(self, tmp_path):
        data = np.zeros((4, 3), dtype=np.float32)
        npy = tmp_path / "input.npy"
        np.save(str(npy), data)

        params = [{"name": "input", "shape": ["?", 3]}]
        configs = [
            {
                "name": "input",
                "source": "file",
                "path": str(npy),
                "lower_bounds": [1, 3],
                "upper_bounds": [16, 3],
            }
        ]
        inputs = prepare_inputs(params, input_configs=configs)
        assert inputs["input"].shape == (4, 3)

    def test_file_input_out_of_bounds_raises(self, tmp_path):
        data = np.zeros((32, 3), dtype=np.float32)
        npy = tmp_path / "input.npy"
        np.save(str(npy), data)

        params = [{"name": "input", "shape": ["?", 3]}]
        configs = [
            {
                "name": "input",
                "source": "file",
                "path": str(npy),
                "lower_bounds": [1, 3],
                "upper_bounds": [16, 3],
            }
        ]
        with pytest.raises(ValueError, match="outside bounds"):
            prepare_inputs(params, input_configs=configs)

    def test_static_shape_ignores_bounds(self):
        """Static shapes work as before, bounds are not checked if not provided."""
        params = [{"name": "input", "shape": [1, 3, 8, 8]}]
        configs = [{"name": "input", "data_type": "fp32"}]
        inputs = prepare_inputs(params, input_configs=configs)
        assert inputs["input"].shape == (1, 3, 8, 8)

    def test_multiple_inputs_mixed_static_dynamic(self, tmp_path):
        """One static input, one dynamic input."""
        data = np.ones((4, 3), dtype=np.float32)
        npy = tmp_path / "file_input.npy"
        np.save(str(npy), data)

        params = [
            {"name": "static_in", "shape": [1, 3]},
            {"name": "dynamic_in", "shape": ["?", 3, 8, 8]},
        ]
        configs = [
            {"name": "static_in", "data_type": "fp32"},
            {
                "name": "dynamic_in",
                "data_type": "fp32",
                "resolved_shape": [2],
                "lower_bounds": [1, 3, 8, 8],
                "upper_bounds": [8, 3, 8, 8],
            },
        ]
        inputs = prepare_inputs(params, input_configs=configs)
        assert inputs["static_in"].shape == (1, 3)
        assert inputs["dynamic_in"].shape == (2, 3, 8, 8)


# ── Session service with dynamic shapes ────────────────────────────


@pytest.fixture
def session_service(tmp_path):
    return SessionService(tmp_path / "sessions")


@pytest.fixture
def dynamic_config(tmp_path):
    model_xml = tmp_path / "models" / "model.xml"
    model_xml.parent.mkdir(parents=True, exist_ok=True)
    model_xml.write_text("<model/>")
    model_xml.with_suffix(".bin").write_bytes(b"\x00" * 16)
    return SessionConfig(
        model_path=str(model_xml),
        main_device="CPU",
        ref_device="CPU",
        inputs=[
            InputConfig(
                name="input_0",
                shape=["?", 3, 224, 224],
                element_type="f32",
                data_type="fp32",
                source="random",
                layout="NCHW",
                resolved_shape=[1],
                lower_bounds=[1, 3, 224, 224],
                upper_bounds=[16, 3, 224, 224],
            ),
        ],
    )


class TestSessionServiceDynamic:
    def test_create_session_dynamic_random_generates_input(
        self, session_service, dynamic_config
    ):
        """Random source with resolved_shape generates and saves .npy."""
        info = session_service.create_session(dynamic_config)
        detail = session_service.get_session(info.id)
        assert detail is not None

        # Input should have been generated and source changed to "file"
        inp = detail.config.inputs[0]
        assert inp.source == "file"
        assert inp.path is not None
        assert inp.path.endswith(".npy")
        # The file should exist
        from pathlib import Path
        assert Path(inp.path).exists()
        # Load and verify shape
        data = np.load(inp.path)
        assert data.shape == (1, 3, 224, 224)

    def test_create_session_dynamic_no_resolved_defers(
        self, session_service, tmp_path
    ):
        """Dynamic shape without resolved_shape defers generation."""
        model_xml = tmp_path / "models" / "model.xml"
        model_xml.parent.mkdir(parents=True, exist_ok=True)
        model_xml.write_text("<model/>")
        model_xml.with_suffix(".bin").write_bytes(b"\x00" * 16)

        config = SessionConfig(
            model_path=str(model_xml),
            main_device="CPU",
            ref_device="CPU",
            inputs=[
                InputConfig(
                    name="input_0",
                    shape=["?", 3, 224, 224],
                    element_type="f32",
                    data_type="fp32",
                    source="random",
                    layout="NCHW",
                    # No resolved_shape, no bounds
                ),
            ],
        )

        info = session_service.create_session(config)
        detail = session_service.get_session(info.id)
        # Without resolved_shape, input should remain as-is (deferred)
        inp = detail.config.inputs[0]
        assert inp.source == "random"
        assert inp.path is None

    def test_create_session_dynamic_bounds_validation_error(
        self, session_service, tmp_path
    ):
        """resolved_shape outside bounds raises ValueError."""
        model_xml = tmp_path / "models" / "model.xml"
        model_xml.parent.mkdir(parents=True, exist_ok=True)
        model_xml.write_text("<model/>")
        model_xml.with_suffix(".bin").write_bytes(b"\x00" * 16)

        config = SessionConfig(
            model_path=str(model_xml),
            main_device="CPU",
            ref_device="CPU",
            inputs=[
                InputConfig(
                    name="input_0",
                    shape=["?", 3],
                    element_type="f32",
                    data_type="fp32",
                    source="random",
                    resolved_shape=[32],
                    lower_bounds=[1, 3],
                    upper_bounds=[16, 3],
                ),
            ],
        )

        with pytest.raises(ValueError, match="outside bounds"):
            session_service.create_session(config)

    def test_create_session_static_inputs_unchanged(
        self, session_service, tmp_path
    ):
        """Static-shape inputs are not affected by dynamic shape code paths."""
        model_xml = tmp_path / "models" / "model.xml"
        model_xml.parent.mkdir(parents=True, exist_ok=True)
        model_xml.write_text("<model/>")
        model_xml.with_suffix(".bin").write_bytes(b"\x00" * 16)

        config = SessionConfig(
            model_path=str(model_xml),
            main_device="CPU",
            ref_device="CPU",
            inputs=[
                InputConfig(
                    name="input_0",
                    shape=[1, 3, 8, 8],
                    element_type="f32",
                    data_type="fp32",
                    source="random",
                    layout="NCHW",
                ),
            ],
        )

        info = session_service.create_session(config)
        detail = session_service.get_session(info.id)
        inp = detail.config.inputs[0]
        assert inp.source == "file"
        assert inp.path.endswith(".npy")


# ── Graph service — dynamic shape extraction ──────────────────────


def _make_mock_op_dynamic(name, op_type, output_shape_dims=None, inputs=None):
    """Create a mock OV op that supports dynamic partial shapes.

    output_shape_dims: list of (value, is_static) tuples, e.g.
       [(None, False), (3, True), (224, True), (224, True)] for [?, 3, 224, 224]
    """
    op = MagicMock()
    op.get_friendly_name.return_value = name
    op.get_type_name.return_value = op_type
    op.get_output_size.return_value = 1

    pshape = MagicMock()
    if output_shape_dims is not None:
        all_static = all(is_static for _, is_static in output_shape_dims)
        pshape.is_static = all_static
        if all_static:
            pshape.get_shape.return_value = [v for v, _ in output_shape_dims]

        # Iteration support
        dims = []
        for val, is_static in output_shape_dims:
            dim = MagicMock()
            dim.is_static = is_static
            if is_static:
                dim.get_length.return_value = val
            else:
                dim.get_length.side_effect = RuntimeError("dynamic dim")
            dim.__str__ = lambda self, v=val, s=is_static: str(v) if s else "?"
            dims.append(dim)
        pshape.__iter__ = lambda self: iter(dims)
    else:
        pshape.is_static = True
        pshape.get_shape.return_value = [1]
        dim = MagicMock()
        dim.is_static = True
        dim.get_length.return_value = 1
        pshape.__iter__ = lambda self: iter([dim])

    output_mock = MagicMock()
    output_mock.get_partial_shape.return_value = pshape
    output_mock.get_element_type.return_value = "f32"
    op.output.return_value = output_mock

    # Input
    in_size = 0 if inputs is None else len(inputs)
    op.get_input_size.return_value = in_size
    if inputs:
        def get_input(idx):
            inp = MagicMock()
            source_out = MagicMock()
            source_node = MagicMock()
            source_node.get_friendly_name.return_value = inputs[idx]
            source_out.get_node.return_value = source_node
            source_out.get_index.return_value = 0
            inp.get_source_output.return_value = source_out
            return inp
        op.input = get_input

    op.get_attributes.return_value = {}
    return op


class TestGraphServiceDynamic:
    def test_dynamic_shape_extraction(self):
        """Dynamic dims should appear as '?' in the extracted graph."""
        from backend.services.graph_service import extract_graph

        param_op = _make_mock_op_dynamic(
            "input", "Parameter",
            output_shape_dims=[(None, False), (3, True), (224, True), (224, True)],
        )
        conv_op = _make_mock_op_dynamic(
            "conv1", "Convolution",
            output_shape_dims=[(None, False), (64, True), (112, True), (112, True)],
            inputs=["input"],
        )
        result_op = _make_mock_op_dynamic(
            "output", "Result",
            inputs=["conv1"],
        )

        # Wire up source node references
        ops = [param_op, conv_op, result_op]
        op_by_name = {op.get_friendly_name(): op for op in ops}
        for op in ops:
            if op.get_input_size() == 0:
                continue
            original_input = op.input
            def make_fn(orig, lookup):
                def get_input(idx):
                    inp = orig(idx)
                    src_name = inp.get_source_output().get_node().get_friendly_name()
                    if src_name in lookup:
                        inp.get_source_output().get_node.return_value = lookup[src_name]
                    return inp
                return get_input
            op.input = make_fn(original_input, op_by_name)

        model = MagicMock()
        model.get_ordered_ops.return_value = ops

        graph = extract_graph(model)

        param_node = next(n for n in graph.nodes if n.name == "input")
        assert param_node.shape == ["?", 3, 224, 224]

        conv_node = next(n for n in graph.nodes if n.name == "conv1")
        assert conv_node.shape == ["?", 64, 112, 112]

    def test_static_shape_extraction_unchanged(self):
        """Static shapes still extracted as ints."""
        from backend.services.graph_service import extract_graph

        param_op = _make_mock_op_dynamic(
            "input", "Parameter",
            output_shape_dims=[(1, True), (3, True), (224, True), (224, True)],
        )
        result_op = _make_mock_op_dynamic("output", "Result", inputs=["input"])

        ops = [param_op, result_op]
        op_by_name = {op.get_friendly_name(): op for op in ops}
        original_input = result_op.input
        def make_fn(orig, lookup):
            def get_input(idx):
                inp = orig(idx)
                src_name = inp.get_source_output().get_node().get_friendly_name()
                if src_name in lookup:
                    inp.get_source_output().get_node.return_value = lookup[src_name]
                return inp
            return get_input
        result_op.input = make_fn(original_input, op_by_name)

        model = MagicMock()
        model.get_ordered_ops.return_value = ops

        graph = extract_graph(model)

        param_node = next(n for n in graph.nodes if n.name == "input")
        assert param_node.shape == [1, 3, 224, 224]
        # All ints, no strings
        assert all(isinstance(d, int) for d in param_node.shape)


# ── Inference worker — _extract_params and _apply_bounded_reshape ──


class TestInferenceWorkerDynamic:
    def test_extract_params_static(self):
        from backend.utils.inference_worker import _extract_params

        param = MagicMock()
        param.get_friendly_name.return_value = "input"
        pshape = MagicMock()
        pshape.is_static = True
        dim1 = MagicMock(); dim1.get_length.return_value = 1; dim1.is_static = True
        dim2 = MagicMock(); dim2.get_length.return_value = 3; dim2.is_static = True
        pshape.__iter__ = lambda self: iter([dim1, dim2])
        param.get_output_partial_shape.return_value = pshape
        param.get_output_element_type.return_value = "f32"

        model = MagicMock()
        model.get_parameters.return_value = [param]

        params = _extract_params(model)
        assert params == [{"name": "input", "shape": [1, 3], "element_type": "f32"}]

    def test_extract_params_dynamic(self):
        from backend.utils.inference_worker import _extract_params

        param = MagicMock()
        param.get_friendly_name.return_value = "input"
        pshape = MagicMock()
        pshape.is_static = False

        dim_dyn = MagicMock(); dim_dyn.is_static = False
        dim_static = MagicMock(); dim_static.is_static = True; dim_static.get_length.return_value = 3
        pshape.__iter__ = lambda self: iter([dim_dyn, dim_static])
        param.get_output_partial_shape.return_value = pshape
        param.get_output_element_type.return_value = "f32"

        model = MagicMock()
        model.get_parameters.return_value = [param]

        params = _extract_params(model)
        assert params == [{"name": "input", "shape": ["?", 3], "element_type": "f32"}]

    def test_apply_bounded_reshape_no_bounds(self):
        from backend.utils.inference_worker import _apply_bounded_reshape

        model = MagicMock()
        # No bounds — reshape should not be called
        _apply_bounded_reshape(model, [{"name": "input", "shape": [1, 3]}])
        model.reshape.assert_not_called()

    def test_apply_bounded_reshape_none_configs(self):
        from backend.utils.inference_worker import _apply_bounded_reshape

        model = MagicMock()
        _apply_bounded_reshape(model, None)
        model.reshape.assert_not_called()

    def test_apply_bounded_reshape_with_bounds(self):
        from backend.utils.inference_worker import _apply_bounded_reshape

        model = MagicMock()
        configs = [{
            "name": "input",
            "shape": ["?", 3, 224, 224],
            "lower_bounds": [1, 3, 224, 224],
            "upper_bounds": [16, 3, 224, 224],
        }]

        mock_ov = MagicMock()
        mock_dim = MagicMock()
        mock_ov.Dimension.return_value = mock_dim
        mock_pshape = MagicMock()
        mock_ov.PartialShape.return_value = mock_pshape

        with patch.dict("sys.modules", {"openvino": mock_ov}):
            _apply_bounded_reshape(model, configs)

        # Should call Dimension(1, 16) for dynamic dim, Dimension(3) etc for static
        assert mock_ov.Dimension.call_count == 4
        # First call is for the dynamic dim: Dimension(1, 16)
        mock_ov.Dimension.assert_any_call(1, 16)
        # Static dims: Dimension(3), Dimension(224), Dimension(224)
        mock_ov.Dimension.assert_any_call(3)
        mock_ov.Dimension.assert_any_call(224)

        mock_ov.PartialShape.assert_called_once()
        model.reshape.assert_called_once_with({"input": mock_pshape})


# ── Model cut service — dynamic shape error ────────────────────────


class TestModelCutDynamicShape:
    def test_make_input_node_random_dynamic_error_message(self):
        """Dynamic shape should produce a helpful error message."""
        from backend.services.model_cut_service import ModelCutService

        core = MagicMock()
        svc = ModelCutService(core)

        target_op = MagicMock()
        target_op.get_friendly_name.return_value = "conv1"
        target_op.get_type_name.return_value = "Convolution"

        # Create a dynamic partial shape
        pshape = MagicMock()
        pshape.is_dynamic = True
        dim_dyn = MagicMock()
        dim_dyn.is_static = False
        dim_dyn.__str__ = lambda self: "?"
        dim_static = MagicMock()
        dim_static.is_static = True
        dim_static.__str__ = lambda self: "64"
        pshape.__iter__ = lambda self: iter([dim_dyn, dim_static])
        target_op.get_output_partial_shape.return_value = pshape

        model = MagicMock()
        model.get_ordered_ops.return_value = [target_op]
        core.read_model.return_value = model

        with pytest.raises(ValueError, match="dynamic shape.*input.*cut type"):
            svc.make_input_node_random("/fake/model.xml", "conv1")


# ── API-level test for model inputs with dynamic shapes ────────────


class TestModelInputsAPIDynamic:
    @pytest.mark.asyncio
    async def test_model_inputs_returns_dynamic_shapes(self, tmp_path):
        """GET /api/model-inputs should return '?' for dynamic dims."""
        import httpx
        from fastapi import FastAPI
        from backend.routers import devices
        from backend.config import AppConfig

        # Create a real file so the existence check passes
        model_xml = tmp_path / "model.xml"
        model_xml.write_text("<model/>")

        app = FastAPI()

        # Mock OV core that returns a model with dynamic shapes
        mock_core = MagicMock()
        mock_model = MagicMock()

        param = MagicMock()
        param.get_friendly_name.return_value = "input"
        pshape = MagicMock()
        pshape.is_static = False

        dim_dyn = MagicMock()
        dim_dyn.is_static = False
        dim_static1 = MagicMock()
        dim_static1.is_static = True
        dim_static1.get_length.return_value = 3
        dim_static2 = MagicMock()
        dim_static2.is_static = True
        dim_static2.get_length.return_value = 224
        dim_static3 = MagicMock()
        dim_static3.is_static = True
        dim_static3.get_length.return_value = 224
        pshape.__iter__ = lambda self: iter([dim_dyn, dim_static1, dim_static2, dim_static3])

        param.get_output_partial_shape.return_value = pshape
        param.get_output_element_type.return_value = "f32"
        mock_model.get_parameters.return_value = [param]
        mock_core.read_model.return_value = mock_model

        app.state.config = AppConfig()
        app.state.ov_core = mock_core
        app.include_router(devices.router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/model-inputs",
                params={"model_path": str(model_xml)},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "input"
        assert data[0]["shape"] == ["?", 3, 224, 224]
        assert data[0]["element_type"] == "f32"

    @pytest.mark.asyncio
    async def test_model_inputs_static_shapes_are_ints(self, tmp_path):
        """Static shapes should be returned as ints, not strings."""
        import httpx
        from fastapi import FastAPI
        from backend.routers import devices
        from backend.config import AppConfig

        model_xml = tmp_path / "model.xml"
        model_xml.write_text("<model/>")

        app = FastAPI()

        mock_core = MagicMock()
        mock_model = MagicMock()

        param = MagicMock()
        param.get_friendly_name.return_value = "input"
        pshape = MagicMock()
        pshape.is_static = True

        dims = []
        for v in [1, 3, 224, 224]:
            d = MagicMock()
            d.is_static = True
            d.get_length.return_value = v
            dims.append(d)
        pshape.__iter__ = lambda self: iter(dims)

        param.get_output_partial_shape.return_value = pshape
        param.get_output_element_type.return_value = "f32"
        mock_model.get_parameters.return_value = [param]
        mock_core.read_model.return_value = mock_model

        app.state.config = AppConfig()
        app.state.ov_core = mock_core
        app.include_router(devices.router)

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/api/model-inputs",
                params={"model_path": str(model_xml)},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["shape"] == [1, 3, 224, 224]
        # All ints
        assert all(isinstance(d, int) for d in data[0]["shape"])
