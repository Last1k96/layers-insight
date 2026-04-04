"""Bisection search routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.schemas.bisect import BisectJobInfo, BisectProgress, BisectRequest
from backend.schemas.graph import GraphData

router = APIRouter(prefix="/api/inference/bisect", tags=["bisect"])


@router.post("", response_model=BisectJobInfo)
async def start_bisection(req: BisectRequest, request: Request) -> BisectJobInfo:
    """Start a bisection search for accuracy drop or compilation failure."""
    bisect_svc = request.app.state.bisect_service
    if bisect_svc is None:
        raise HTTPException(status_code=503, detail="Bisect service not available")

    if bisect_svc.is_active:
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
        job = await bisect_svc.start(
            request=req,
            graph_data=graph_data,
            queue_service=queue_svc,
            session_service=session_svc,
            broadcast=ws_manager.broadcast,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return job


@router.post("/stop", response_model=BisectJobInfo)
async def stop_bisection(request: Request) -> BisectJobInfo:
    """Stop the running bisection search entirely."""
    bisect_svc = request.app.state.bisect_service
    if bisect_svc is None:
        raise HTTPException(status_code=503, detail="Bisect service not available")

    job = await bisect_svc.stop()
    if job is None:
        raise HTTPException(status_code=404, detail="No bisect job active")
    return job


@router.get("/status")
async def get_bisection_status(request: Request) -> dict:
    """Get the current bisection state."""
    bisect_svc = request.app.state.bisect_service
    if bisect_svc is None:
        raise HTTPException(status_code=503, detail="Bisect service not available")

    job = bisect_svc.job
    if job:
        return job.model_dump()
    return {"status": "idle"}
