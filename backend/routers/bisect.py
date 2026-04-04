"""Bisection search routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.schemas.bisect import BisectJobInfo, BisectRequest
from backend.schemas.graph import GraphData

router = APIRouter(prefix="/api/inference/bisect", tags=["bisect"])


@router.post("", response_model=BisectJobInfo)
async def start_bisection(req: BisectRequest, request: Request) -> BisectJobInfo:
    """Start a bisection search for accuracy drop or compilation failure."""
    async with request.app.state.pause_resume_lock:
        bisect_svc = request.app.state.bisect_service
        if bisect_svc is None:
            raise HTTPException(status_code=503, detail="Bisect service not available")

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
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return job


@router.post("/{job_id}/stop", response_model=BisectJobInfo)
async def stop_bisection(job_id: str, request: Request) -> BisectJobInfo:
    """Stop a specific bisection job."""
    async with request.app.state.pause_resume_lock:
        bisect_svc = request.app.state.bisect_service
        if bisect_svc is None:
            raise HTTPException(status_code=503, detail="Bisect service not available")

        job = await bisect_svc.stop(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Bisect job not found")
        return job


@router.get("/status")
async def get_bisection_status(request: Request, session_id: str | None = None) -> dict:
    """Get all bisection jobs (active + persisted)."""
    bisect_svc = request.app.state.bisect_service
    if bisect_svc is None:
        raise HTTPException(status_code=503, detail="Bisect service not available")

    jobs = [j.model_dump() for j in bisect_svc.get_jobs()]

    # Also include persisted jobs from session metadata (for reload after backend restart)
    if session_id:
        session_svc = request.app.state.session_service
        persisted = session_svc.load_bisect_jobs(session_id)
        active_ids = {j["job_id"] for j in jobs}
        for jid, jdata in persisted.items():
            if jid not in active_ids:
                jobs.append(jdata)

    return {"jobs": jobs}


@router.delete("/{job_id}")
async def dismiss_bisect_job(job_id: str, request: Request, session_id: str) -> dict:
    """Clear a persisted bisect job from session metadata (after merge/discard)."""
    session_svc = request.app.state.session_service
    session_svc.clear_bisect_job(session_id, job_id)
    return {"cleared": True}
