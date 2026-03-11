"""Graph data routes."""
from __future__ import annotations

import asyncio

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


@router.get("")
async def get_graph(session_id: str, request: Request) -> JSONResponse:
    """Get full graph data with positions and colors."""
    session_svc = request.app.state.session_service
    session = session_svc.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check for cached graph
    cached = session_svc.load_graph_cache(session_id)
    if cached:
        return JSONResponse(content=cached)

    # Load model and extract graph
    ov_core = request.app.state.ov_core
    if ov_core is None:
        raise HTTPException(status_code=503, detail="OpenVINO not available")

    try:
        model = load_model(session.config.model_path, ov_core)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")

    # Store model in app state for inference
    request.app.state.models[session_id] = model

    graph_data = extract_graph(model)
    positions = await compute_layout(graph_data)
    graph_data = apply_layout(graph_data, positions)

    # Cache for future requests
    graph_dict = graph_data.model_dump()
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
