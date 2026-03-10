"""Graph data routes."""
from __future__ import annotations

import asyncio

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
