"""Session management routes."""
from __future__ import annotations

import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Iterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.schemas.session import (
    CloneEnqueueRequest,
    CloneRequest,
    CompareResponse,
    CutRequest,
    RenameRequest,
    SessionConfig,
    SessionDetail,
    SessionInfo,
    SubSessionInfo,
)
from backend.services.model_cut_service import CutResult
from backend.utils import sanitize_filename
from backend.utils.model_converter import convert_to_ir, detect_model_format
from backend.utils.zip_stream import ZipStreamBuffer

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _get_session_service(request: Request):
    return request.app.state.session_service


@router.post("", response_model=SessionInfo)
async def create_session(config: SessionConfig, request: Request) -> SessionInfo:
    """Create a new session. Non-IR models are converted to IR automatically."""
    svc = _get_session_service(request)

    model_path = Path(config.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=400, detail=f"Model path not found: {model_path}")
    try:
        fmt = detect_model_format(model_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if fmt == "ir":
        try:
            return svc.create_session(config)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Non-IR: convert to temporary IR, then pass converted files to session service
    ov_core = request.app.state.ov_core
    if ov_core is None:
        raise HTTPException(status_code=503, detail="OpenVINO not available — cannot convert model")

    tmp_dir = tempfile.mkdtemp(prefix="li_convert_")
    try:
        try:
            converted_xml = convert_to_ir(model_path, Path(tmp_dir), ov_core)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to convert {fmt} model to IR: {e}",
            )

        config = config.model_copy(update={
            "model_path": str(converted_xml),
            "original_format": fmt,
        })
        try:
            return svc.create_session(config, converted_dir=Path(tmp_dir))
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("", response_model=list[SessionInfo])
async def list_sessions(request: Request) -> list[SessionInfo]:
    """List all sessions."""
    svc = _get_session_service(request)
    return svc.list_sessions()


@router.get("/compare", response_model=CompareResponse)
async def compare_sessions(
    request: Request,
    session_a: str = Query(..., description="First session ID"),
    session_b: str = Query(..., description="Second session ID"),
) -> CompareResponse:
    """Compare inference results between two sessions node by node."""
    svc = _get_session_service(request)

    if svc.get_session(session_a) is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_a}' not found")
    if svc.get_session(session_b) is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_b}' not found")

    result = svc.compare_sessions(session_a, session_b)
    return CompareResponse(**result)


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str, request: Request) -> SessionDetail:
    """Get session detail."""
    svc = _get_session_service(request)
    detail = svc.get_session(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return detail


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict:
    """Delete a session."""
    svc = _get_session_service(request)
    if svc.delete_session(session_id):
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="Session not found")


@router.patch("/{session_id}/rename")
async def rename_session(session_id: str, req: RenameRequest, request: Request) -> dict:
    """Rename a session and its on-disk folder.

    If renaming changes the session id (new name sanitizes to a different
    suffix), in-memory state keyed by the old id is remapped to the new id and
    the new id is returned so the client can update its references.
    """
    svc = _get_session_service(request)
    new_name = req.name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    app = request.app
    queue_svc = app.state.queue_service
    bisect_svc = app.state.bisect_service

    # Reject while work is actively running — the executing subprocess has
    # absolute paths baked into its argv, and pulling the folder out from
    # under it would break the run.
    executing_id = queue_svc._executing_task_id
    if executing_id:
        exec_task = queue_svc.get_task(executing_id)
        if exec_task and exec_task.session_id == session_id:
            raise HTTPException(
                status_code=409,
                detail="Cannot rename while an inference is executing. Pause the queue first.",
            )

    from backend.schemas.bisect import BisectStatus
    for job in bisect_svc.get_jobs(session_id):
        if job.status == BisectStatus.RUNNING:
            raise HTTPException(
                status_code=409,
                detail="Cannot rename while a bisect job is running. Pause or stop it first.",
            )

    async with app.state.pause_resume_lock:
        try:
            new_id = svc.rename_session(session_id, new_name)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Failed to rename folder: {e}")
        if new_id is None:
            raise HTTPException(status_code=404, detail="Session not found")

        if new_id != session_id:
            from backend.ws.handler import ws_manager

            models = app.state.models
            if session_id in models:
                models[new_id] = models.pop(session_id)

            if session_id in ws_manager._connections:
                ws_manager._connections[new_id] = ws_manager._connections.pop(session_id)

            for task in queue_svc.get_all_tasks(session_id):
                task.session_id = new_id

            for job_state in bisect_svc._jobs.values():
                if job_state.job.session_id == session_id:
                    job_state.job.session_id = new_id
                    if job_state.request is not None:
                        try:
                            job_state.request.session_id = new_id
                        except Exception:
                            pass

    return {"renamed": True, "name": new_name, "id": new_id}


@router.post("/{session_id}/clone")
async def clone_session(session_id: str, req: CloneRequest, request: Request) -> dict:
    """Clone a session with optional overrides.

    Returns the new session info and a list of nodes that were inferred
    in the source session, ordered by worst accuracy first.
    """
    svc = _get_session_service(request)
    source = svc.get_session(session_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source session not found")

    overrides = req.model_dump(exclude_none=True)
    # Convert InputConfig objects to dicts for the service layer
    if overrides.get("inputs"):
        overrides["inputs"] = [inp.model_dump() for inp in req.inputs]

    result = svc.clone_session(session_id, overrides)
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to clone session")

    new_info, inferred_nodes = result
    return {
        "session": new_info.model_dump(),
        "inferred_nodes": inferred_nodes,
    }


@router.post("/{session_id}/clone-enqueue")
async def clone_enqueue(session_id: str, req: CloneEnqueueRequest, request: Request) -> dict:
    """Batch-enqueue nodes from source session into a target session.

    Nodes are ordered by worst accuracy from the source session.
    """
    svc = _get_session_service(request)
    source = svc.get_session(session_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source session not found")

    target = svc.get_session(req.target_session_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Target session not found")

    # Get source session's inferred nodes for ordering
    source_meta = svc._read_metadata(session_id)
    inferred_nodes = svc._get_inferred_nodes_sorted(source_meta)
    node_order = {n["node_name"]: i for i, n in enumerate(inferred_nodes)}

    # Sort requested nodes by worst accuracy from source
    ordered_names = sorted(
        req.node_names,
        key=lambda name: node_order.get(name, len(inferred_nodes)),
    )

    # Build node info map from source tasks
    tasks = source_meta.get("tasks", {})
    node_info: dict[str, dict] = {}
    for task_data in tasks.values():
        if task_data.get("status") == "success":
            name = task_data.get("node_name")
            if name and name not in node_info:
                node_info[name] = {
                    "node_id": task_data.get("node_id", name),
                    "node_type": task_data.get("node_type", ""),
                }

    # Enqueue into target session's queue
    queue_svc = request.app.state.queue_service
    batch_id = str(uuid.uuid4())[:8]
    enqueued = []

    for name in ordered_names:
        info = node_info.get(name, {"node_id": name, "node_type": ""})
        task = queue_svc.create_task(
            session_id=req.target_session_id,
            node_id=info["node_id"],
            node_name=name,
            node_type=info["node_type"],
        )
        task.batch_id = batch_id
        result = await queue_svc.enqueue(task)
        enqueued.append(result.model_dump())

    return {"enqueued": len(enqueued), "batch_id": batch_id, "tasks": enqueued}


@router.post("/{session_id}/cut", response_model=SubSessionInfo)
async def cut_model(session_id: str, req: CutRequest, request: Request) -> SubSessionInfo:
    """Cut the model at a node, creating a sub-session."""
    svc = _get_session_service(request)
    session = svc.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    model_cut_svc = request.app.state.model_cut_service
    if model_cut_svc is None:
        raise HTTPException(status_code=503, detail="OpenVINO not available")

    model = request.app.state.models.get(session_id)
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")

    try:
        result: CutResult = model_cut_svc.perform_cut(
            session_svc=svc,
            session=session,
            session_id=session_id,
            model=model,
            node_name=req.node_name,
            cut_type=req.cut_type,
            input_precision=req.input_precision,
            parent_sub_session_id=req.parent_sub_session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Broadcast sub-session creation via WS (HTTP-layer concern)
    from backend.ws.handler import ws_manager
    await ws_manager.broadcast(session_id, {
        "type": "sub_session_created",
        "sub_session_id": result.sub_session.id,
        "parent_sub_session_id": req.parent_sub_session_id,
        "cut_type": result.effective_cut_type,
        "cut_node": req.node_name,
        "grayed_nodes": result.grayed_nodes,
        "ancestor_cuts": result.ancestor_cuts,
    })

    return result.sub_session


@router.get("/{session_id}/sub-sessions", response_model=list[SubSessionInfo])
async def list_sub_sessions(session_id: str, request: Request) -> list[SubSessionInfo]:
    """List sub-sessions for a session."""
    svc = _get_session_service(request)
    return svc.list_sub_sessions(session_id)


@router.delete("/{session_id}/sub-sessions/{sub_session_id}")
async def delete_sub_session(session_id: str, sub_session_id: str, request: Request) -> dict:
    """Delete a sub-session and all its descendants."""
    svc = _get_session_service(request)
    if svc.delete_sub_session(session_id, sub_session_id):
        return {"deleted": True, "sub_session_id": sub_session_id}
    raise HTTPException(status_code=404, detail="Sub-session not found")


@router.get("/{session_id}/sub-sessions/{sub_session_id}/export")
async def export_sub_session(
    session_id: str,
    sub_session_id: str,
    request: Request,
) -> StreamingResponse:
    """Stream a ZIP of the sub-session's cut model plus every input it needs.

    Inputs are materialised the same way inference does: root session
    inputs are merged with the sub-session's overrides, then
    ``prepare_inputs`` is called over the cut model's parameters so
    file-backed inputs are loaded and random ones are regenerated.

    Layout inside the ZIP:
        sub_session_<cut_node>/
            cut_model.xml
            cut_model.bin
            input_<name>.bin         (raw bytes, one per cut-model parameter)
            info.json
    """
    from backend.utils.input_generator import prepare_inputs

    svc = _get_session_service(request)
    session = svc.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    sub_meta = svc.get_sub_session_meta(session_id, sub_session_id)
    if sub_meta is None:
        raise HTTPException(status_code=404, detail="Sub-session not found")

    model_rel = sub_meta.get("model_path")
    if not model_rel:
        raise HTTPException(status_code=404, detail="Sub-session cut model not on disk")

    session_path = svc._session_path(session_id)
    model_xml_abs = session_path / model_rel
    model_bin_abs = model_xml_abs.with_suffix(".bin")
    if not model_xml_abs.exists():
        raise HTTPException(status_code=404, detail="Sub-session cut model not on disk")

    ov_core = request.app.state.ov_core
    if ov_core is None:
        raise HTTPException(status_code=503, detail="OpenVINO not available")

    # Merge root + sub-session input configs exactly like the inference
    # path does (backend/main.py:108-126). Sub-session overrides are
    # appended last so they win in prepare_inputs' name lookup.
    merged_configs: list[dict] = []
    if session.config.inputs:
        merged_configs.extend(inp.model_dump() for inp in session.config.inputs)

    resolved_sub = svc.get_sub_session_meta_resolved(session_id, sub_session_id) or {}
    for cfg in resolved_sub.get("input_configs", []):
        merged_configs.append(dict(cfg))

    # Read the cut model so we can enumerate the exact parameters it
    # needs. Reshape to any concrete bounds on the root inputs so
    # dynamic dims materialise to fixed tensors before generation.
    try:
        cut_model = ov_core.read_model(str(model_xml_abs))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to read cut model: {err}")

    model_params: list[dict] = []
    for param in cut_model.get_parameters():
        pshape = param.get_output_partial_shape(0)
        shape = [d.get_length() if d.is_static else "?" for d in pshape]
        model_params.append({
            "name": param.get_friendly_name(),
            "shape": shape,
            "element_type": str(param.get_output_element_type(0)),
        })
    del cut_model

    try:
        inputs = prepare_inputs(
            model_params,
            input_path=None,
            precision=session.config.input_precision or "fp32",
            input_configs=merged_configs,
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to prepare inputs: {err}")

    cut_node = sub_meta.get("cut_node", "unknown")
    cut_type = sub_meta.get("cut_type", "unknown")
    info: dict = {
        "sub_session_id": sub_session_id,
        "cut_node": cut_node,
        "cut_type": cut_type,
        "parent_id": sub_meta.get("parent_id"),
        "ancestor_cuts": sub_meta.get("ancestor_cuts", []),
        "model_name": session.info.model_name,
        "main_device": session.config.main_device,
        "ref_device": session.config.ref_device,
        "inputs": [],
        "session_config": {
            "ov_path": session.config.ov_path,
            "main_device": session.config.main_device,
            "ref_device": session.config.ref_device,
            "input_precision": session.config.input_precision,
        },
    }

    safe_cut = sanitize_filename(cut_node)

    def _generate() -> Iterator[bytes]:
        buf = ZipStreamBuffer()
        prefix = f"sub_session_{safe_cut}/"

        with zipfile.ZipFile(buf, "w") as zf:
            # Model files — .bin is already compact weights, .xml is text.
            zf.writestr(
                f"{prefix}cut_model.xml",
                model_xml_abs.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            yield buf.drain()
            if model_bin_abs.exists():
                zf.writestr(
                    f"{prefix}cut_model.bin",
                    model_bin_abs.read_bytes(),
                    compress_type=zipfile.ZIP_STORED,
                )
                yield buf.drain()

            # Emit every input the cut model actually needs.
            for name, tensor in inputs.items():
                safe_name = sanitize_filename(name)
                bin_name = f"input_{safe_name}.bin"
                zf.writestr(
                    f"{prefix}{bin_name}",
                    tensor.tobytes(),
                    compress_type=zipfile.ZIP_STORED,
                )
                info["inputs"].append({
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "file": bin_name,
                })
                yield buf.drain()

            zf.writestr(
                f"{prefix}info.json",
                json.dumps(info, indent=2),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            yield buf.drain()

        tail = buf.drain()
        if tail:
            yield tail

    filename = f"sub_session_{safe_cut}.zip"
    return StreamingResponse(
        _generate(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{session_id}/export")
async def export_session(
    session_id: str,
    request: Request,
) -> StreamingResponse:
    """Stream a ZIP of the full (uncut) model plus every input it needs.

    Mirrors ``export_sub_session`` but without any sub-session merging —
    inputs come solely from the session config.
    """
    from backend.utils.input_generator import prepare_inputs

    svc = _get_session_service(request)
    session = svc.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    model_xml_abs = Path(session.config.model_path)
    model_bin_abs = model_xml_abs.with_suffix(".bin")
    if not model_xml_abs.exists():
        raise HTTPException(status_code=404, detail="Session model not on disk")

    ov_core = request.app.state.ov_core
    if ov_core is None:
        raise HTTPException(status_code=503, detail="OpenVINO not available")

    input_configs: list[dict] = []
    if session.config.inputs:
        input_configs.extend(inp.model_dump() for inp in session.config.inputs)

    try:
        model = ov_core.read_model(str(model_xml_abs))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to read model: {err}")

    model_params: list[dict] = []
    for param in model.get_parameters():
        pshape = param.get_output_partial_shape(0)
        shape = [d.get_length() if d.is_static else "?" for d in pshape]
        model_params.append({
            "name": param.get_friendly_name(),
            "shape": shape,
            "element_type": str(param.get_output_element_type(0)),
        })
    del model

    try:
        inputs = prepare_inputs(
            model_params,
            input_path=None,
            precision=session.config.input_precision or "fp32",
            input_configs=input_configs,
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Failed to prepare inputs: {err}")

    info: dict = {
        "session_id": session_id,
        "model_name": session.info.model_name,
        "main_device": session.config.main_device,
        "ref_device": session.config.ref_device,
        "inputs": [],
        "session_config": {
            "ov_path": session.config.ov_path,
            "main_device": session.config.main_device,
            "ref_device": session.config.ref_device,
            "input_precision": session.config.input_precision,
        },
    }

    safe_name = sanitize_filename(session.info.model_name or "model")

    def _generate() -> Iterator[bytes]:
        buf = ZipStreamBuffer()
        prefix = f"{safe_name}/"

        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                f"{prefix}model.xml",
                model_xml_abs.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            yield buf.drain()
            if model_bin_abs.exists():
                zf.writestr(
                    f"{prefix}model.bin",
                    model_bin_abs.read_bytes(),
                    compress_type=zipfile.ZIP_STORED,
                )
                yield buf.drain()

            for name, tensor in inputs.items():
                safe_input = sanitize_filename(name)
                bin_name = f"input_{safe_input}.bin"
                zf.writestr(
                    f"{prefix}{bin_name}",
                    tensor.tobytes(),
                    compress_type=zipfile.ZIP_STORED,
                )
                info["inputs"].append({
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "file": bin_name,
                })
                yield buf.drain()

            zf.writestr(
                f"{prefix}info.json",
                json.dumps(info, indent=2),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            yield buf.drain()

        tail = buf.drain()
        if tail:
            yield tail

    filename = f"{safe_name}.zip"
    return StreamingResponse(
        _generate(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{session_id}/sub-sessions/{sub_session_id}/relayout")
async def relayout_sub_session(
    session_id: str,
    sub_session_id: str,
    request: Request,
) -> dict:
    """Build the sub-session's standalone "tight" graph.

    The tight graph is a self-contained ``GraphData`` — a subset of the
    session graph (non-grayed nodes + their edges) with freshly computed
    positions/waypoints. It is persisted as ``sub_sessions/{ssid}/tight_graph.json``
    so the frontend can load it as an independent graph, with no position
    coupling to the full model's layout.

    Response: the full ``GraphData`` dict plus ``tight_mode: True``.
    """
    from backend.schemas.graph import GraphData, GraphNode, GraphEdge
    from backend.services.graph_service import (
        apply_layout,
        compute_block_aware_layout,
        compute_elk_layout,
        compute_layout,
        should_use_block_layout,
    )

    svc = _get_session_service(request)
    session = svc.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    sub_meta = svc.get_sub_session_meta(session_id, sub_session_id)
    if sub_meta is None:
        raise HTTPException(status_code=404, detail="Sub-session not found")

    cached = svc.load_graph_cache(session_id)
    if not cached:
        raise HTTPException(status_code=400, detail="Graph not loaded yet")

    grayed = set(sub_meta.get("grayed_nodes", []))
    all_nodes = cached.get("nodes", [])
    all_edges = cached.get("edges", [])

    visible_nodes = [n for n in all_nodes if n.get("id") not in grayed]
    if not visible_nodes:
        raise HTTPException(status_code=400, detail="Sub-session has no visible nodes")

    visible_ids = {n["id"] for n in visible_nodes}
    visible_edges = [
        e for e in all_edges
        if e.get("source") in visible_ids and e.get("target") in visible_ids
    ]

    # Reconstruct a GraphData with only the visible subset. Position/size
    # fields stay at defaults since those are exactly what we're computing.
    sub_graph = GraphData(
        nodes=[GraphNode(**n) for n in visible_nodes],
        edges=[GraphEdge(**e) for e in visible_edges],
    )

    # Match the same resolution the initial graph route does.
    layout_mode = session.config.layout_mode
    if layout_mode == "auto":
        if session.config.use_elk_layout:
            layout_mode = "elk"
        elif should_use_block_layout(sub_graph):
            layout_mode = "block"
        else:
            layout_mode = "dag"

    try:
        if layout_mode == "block":
            result = await compute_block_aware_layout(sub_graph)
        elif layout_mode == "elk":
            result = await compute_elk_layout(sub_graph)
        else:
            result = await compute_layout(sub_graph)
    except RuntimeError as err:
        raise HTTPException(status_code=500, detail=f"Layout failed: {err}")

    laid_out = apply_layout(sub_graph, result)
    tight_graph = laid_out.model_dump()

    # Preserve the subset of propagated_shapes for the visible nodes.
    propagated = cached.get("propagated_shapes")
    if propagated:
        visible_names = {n.get("name") for n in visible_nodes}
        tight_graph["propagated_shapes"] = {
            name: shape for name, shape in propagated.items()
            if name in visible_names
        }

    rel_path = svc.save_sub_session_tight_graph(
        session_id, sub_session_id, tight_graph,
    )

    # Drop any legacy positions-only payload from older schemas — the
    # standalone tight_graph.json is now the single source of truth.
    svc.update_sub_session_meta(session_id, sub_session_id, {
        "tight_graph_path": rel_path,
        "tight_mode": True,
        "tight_layout": None,
    })

    return {"graph": tight_graph, "tight_mode": True}


@router.get("/{session_id}/sub-sessions/{sub_session_id}/tight-graph")
async def get_sub_session_tight_graph(
    session_id: str,
    sub_session_id: str,
    request: Request,
) -> dict:
    """Return the persisted standalone tight graph for a sub-session."""
    svc = _get_session_service(request)
    sub_meta = svc.get_sub_session_meta(session_id, sub_session_id)
    if sub_meta is None:
        raise HTTPException(status_code=404, detail="Sub-session not found")
    graph = svc.load_sub_session_tight_graph(session_id, sub_session_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Tight graph not computed yet")
    return graph


class TightModeRequest(BaseModel):
    enabled: bool


@router.post("/{session_id}/sub-sessions/{sub_session_id}/tight-mode")
async def set_sub_session_tight_mode(
    session_id: str,
    sub_session_id: str,
    body: TightModeRequest,
    request: Request,
) -> dict:
    """Toggle tight-mode view preference for a sub-session.

    Does not compute a layout. If enabling without a cached tight layout,
    responds 400 and the client is expected to POST /relayout first.
    """
    svc = _get_session_service(request)
    sub_meta = svc.get_sub_session_meta(session_id, sub_session_id)
    if sub_meta is None:
        raise HTTPException(status_code=404, detail="Sub-session not found")

    if body.enabled and not sub_meta.get("tight_graph_path"):
        raise HTTPException(
            status_code=400,
            detail="No cached tight graph; POST /relayout first",
        )

    svc.set_tight_mode(session_id, sub_session_id, body.enabled)
    return {"tight_mode": body.enabled}
