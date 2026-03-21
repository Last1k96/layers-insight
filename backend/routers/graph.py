"""Graph data routes."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.services.graph_service import (
    apply_layout,
    compute_layout,
    extract_graph,
    load_model,
    search_nodes,
)

router = APIRouter(prefix="/api/sessions/{session_id}/graph", tags=["graph"])


def _build_reshape_map(config):
    """Build an OV reshape map from session input configs.

    Returns (reshape_map, has_dynamic) or (None, False) if nothing to reshape.
    """
    if not config.inputs:
        return None, False

    try:
        import openvino as ov
    except ImportError:
        return None, False

    has_dynamic = False
    reshape_map = {}
    for inp in config.inputs:
        if not any(isinstance(d, str) for d in inp.shape):
            continue
        has_dynamic = True

        dims = []
        res_idx = 0
        for d in inp.shape:
            if isinstance(d, str):
                if inp.resolved_shape and res_idx < len(inp.resolved_shape):
                    dims.append(ov.Dimension(inp.resolved_shape[res_idx]))
                    res_idx += 1
                elif inp.lower_bounds and inp.upper_bounds:
                    lo_idx = len(dims)
                    if lo_idx < len(inp.lower_bounds) and lo_idx < len(inp.upper_bounds):
                        dims.append(ov.Dimension(inp.lower_bounds[lo_idx], inp.upper_bounds[lo_idx]))
                    else:
                        return None, True
                else:
                    return None, True
            else:
                dims.append(ov.Dimension(d))

        reshape_map[inp.name] = ov.PartialShape(dims)

    return (reshape_map if reshape_map else None), has_dynamic


def _compute_propagated_shapes(ov_core, model_path: str, config) -> dict[str, list[int]] | None:
    """Reshape a fresh model copy and extract propagated shapes for all ops.

    Returns {node_name: [dim, ...]} or None if not applicable.
    """
    reshape_map, has_dynamic = _build_reshape_map(config)
    if not has_dynamic or not reshape_map:
        return None

    try:
        model_copy = ov_core.read_model(model_path)
        model_copy.reshape(reshape_map)

        shapes: dict[str, list[int]] = {}
        for op in model_copy.get_ordered_ops():
            name = op.get_friendly_name()
            if op.get_output_size() > 0:
                pshape = op.output(0).get_partial_shape()
                if pshape.is_static:
                    shapes[name] = [d.get_length() for d in pshape]
                else:
                    shapes[name] = [
                        d.get_length() if d.is_static else -1 for d in pshape
                    ]
        return shapes
    except Exception:
        return None


@router.get("")
async def get_graph(session_id: str, request: Request) -> JSONResponse:
    """Get full graph data with positions and colors."""
    session_svc = request.app.state.session_service
    session = session_svc.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Ensure model is loaded for inference (even if graph is cached)
    ov_core = request.app.state.ov_core
    if session_id not in request.app.state.models and ov_core is not None:
        try:
            model = load_model(session.config.model_path, ov_core)
            request.app.state.models[session_id] = model
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")

    # Check for cached graph
    cached = session_svc.load_graph_cache(session_id)
    if cached:
        return JSONResponse(content=cached)

    if ov_core is None:
        raise HTTPException(status_code=503, detail="OpenVINO not available")

    model = request.app.state.models.get(session_id)
    if model is None:
        raise HTTPException(status_code=400, detail="Failed to load model")

    graph_data = extract_graph(model)
    positions = await compute_layout(graph_data)
    graph_data = apply_layout(graph_data, positions)

    # Cache for future requests
    graph_dict = graph_data.model_dump()

    # Compute propagated shapes from resolved input dims (on a model copy)
    propagated = _compute_propagated_shapes(ov_core, session.config.model_path, session.config)
    if propagated:
        graph_dict["propagated_shapes"] = propagated

    session_svc.save_graph_cache(session_id, graph_dict)

    return JSONResponse(content=graph_dict)


@router.get("/search")
async def search_graph(session_id: str, q: str, request: Request) -> list[dict]:
    """Search nodes by name or type."""
    session_svc = request.app.state.session_service
    cached = session_svc.load_graph_cache(session_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Graph not loaded yet")

    from backend.schemas.graph import GraphData
    graph_data = GraphData(**cached)
    results = search_nodes(graph_data, q)
    return [r.model_dump() for r in results[:50]]  # Limit results


MAX_PREVIEW_ELEMENTS = 1000


@router.get("/constant/{node_name:path}")
async def get_constant_data(
    session_id: str, node_name: str, request: Request,
) -> JSONResponse:
    """Get data from a Constant node by name."""
    model = request.app.state.models.get(session_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")

    for op in model.get_ordered_ops():
        if op.get_friendly_name() == node_name and op.get_type_name() == "Constant":
            try:
                data: np.ndarray = op.get_data()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to read data: {e}")

            flat = data.flatten()
            total = int(flat.size)
            truncated = total > MAX_PREVIEW_ELEMENTS
            preview = flat[:MAX_PREVIEW_ELEMENTS]

            return JSONResponse(content={
                "name": node_name,
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "total_elements": total,
                "truncated": truncated,
                "stats": {
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                },
                "data": preview.tolist(),
            })

    raise HTTPException(status_code=404, detail=f"Constant node '{node_name}' not found")
