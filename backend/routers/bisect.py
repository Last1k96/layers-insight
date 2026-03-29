"""Bisection search routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.schemas.bisect import BisectProgress, BisectRequest
from backend.schemas.graph import GraphData

router = APIRouter(prefix="/api/inference/bisect", tags=["bisect"])


@router.post("", response_model=BisectProgress)
async def start_bisection(req: BisectRequest, request: Request) -> BisectProgress:
    """Start a bisection search for accuracy drop or compilation failure."""
    bisect_svc = request.app.state.bisect_service
    if bisect_svc is None:
        raise HTTPException(status_code=503, detail="Bisect service not available")

    if bisect_svc.is_running:
        raise HTTPException(status_code=409, detail="A bisection is already running")

    # Load graph data for topological ordering
    session_svc = request.app.state.session_service
    cached = session_svc.load_graph_cache(req.session_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Graph not loaded. Open the graph first.")

    graph_data = GraphData(**cached)

    queue_svc = request.app.state.queue_service

    from backend.ws.handler import ws_manager
    try:
        progress = await bisect_svc.start(
            request=req,
            graph_data=graph_data,
            queue_service=queue_svc,
            broadcast=ws_manager.broadcast,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return progress


@router.post("/stop", response_model=BisectProgress)
async def stop_bisection(request: Request) -> BisectProgress:
    """Stop the running bisection search."""
    bisect_svc = request.app.state.bisect_service
    if bisect_svc is None:
        raise HTTPException(status_code=503, detail="Bisect service not available")

    return await bisect_svc.stop()


@router.get("/status", response_model=BisectProgress)
async def get_bisection_status(request: Request) -> BisectProgress:
    """Get the current bisection state."""
    bisect_svc = request.app.state.bisect_service
    if bisect_svc is None:
        raise HTTPException(status_code=503, detail="Bisect service not available")

    return bisect_svc.progress
