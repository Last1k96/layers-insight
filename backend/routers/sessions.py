"""Session management routes."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from backend.schemas.session import CutRequest, SessionConfig, SessionDetail, SessionInfo, SubSessionInfo
from backend.utils.model_converter import convert_to_ir, detect_model_format

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _get_session_service(request: Request):
    return request.app.state.session_service


@router.post("", response_model=SessionInfo)
async def create_session(config: SessionConfig, request: Request) -> SessionInfo:
    """Create a new session. Non-IR models are converted to IR automatically."""
    svc = _get_session_service(request)

    model_path = Path(config.model_path)
    try:
        fmt = detect_model_format(model_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if fmt == "ir":
        return svc.create_session(config)

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
        return svc.create_session(config, converted_dir=Path(tmp_dir))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("", response_model=list[SessionInfo])
async def list_sessions(request: Request) -> list[SessionInfo]:
    """List all sessions."""
    svc = _get_session_service(request)
    return svc.list_sessions()


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

    # Resolve source model and parent metadata for chained cuts
    parent_sub = None
    parent_sub_resolved = None
    parent_grayed: list[str] = []
    parent_input_configs: list[dict] = []
    parent_input_configs_rel: list[dict] = []
    parent_ancestor_cuts: list[dict] = []

    if req.parent_sub_session_id:
        parent_sub = svc.get_sub_session_meta(session_id, req.parent_sub_session_id)
        parent_sub_resolved = svc.get_sub_session_meta_resolved(session_id, req.parent_sub_session_id)
        if parent_sub is None:
            raise HTTPException(status_code=400, detail="Parent sub-session not found")
        parent_grayed = parent_sub.get("grayed_nodes", [])
        parent_input_configs = parent_sub_resolved.get("input_configs", [])
        parent_input_configs_rel = parent_sub.get("input_configs", [])
        parent_ancestor_cuts = parent_sub.get("ancestor_cuts", [])

    try:
        if req.cut_type == "output":
            if req.parent_sub_session_id and parent_sub_resolved:
                # Read the parent sub-session's cut model (need absolute path)
                import openvino as ov
                parent_model_path = parent_sub_resolved.get("model_path")
                if not parent_model_path:
                    raise HTTPException(status_code=400, detail="Parent sub-session has no model")
                source_model = ov.Core().read_model(parent_model_path)
                cut_model, new_grayed = model_cut_svc.make_output_node(source_model, req.node_name)
            else:
                cut_model, new_grayed = model_cut_svc.make_output_node(model, req.node_name)
        elif req.cut_type in ("input", "input_random"):
            # Use parent sub-session's resolved model path or root model path
            if req.parent_sub_session_id and parent_sub_resolved:
                source_model_path = parent_sub_resolved.get("model_path", session.config.model_path)
            else:
                source_model_path = session.config.model_path

            if req.cut_type == "input":
                # Find the main output .npy for this node
                task_id = svc.find_task_for_node(session_id, req.node_name, req.parent_sub_session_id)
                if task_id is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No successful inference found for node '{req.node_name}'. Run inference first.",
                    )
                npy_path = svc.get_tensor_path(session_id, task_id, "main_output")
                if npy_path is None:
                    raise HTTPException(status_code=400, detail="Main output tensor not found")

                cut_model, input_data, new_grayed = model_cut_svc.make_input_node(
                    source_model_path, req.node_name, str(npy_path), req.input_precision,
                )
            else:
                cut_model, input_data, new_grayed = model_cut_svc.make_input_node_random(
                    source_model_path, req.node_name, req.input_precision,
                )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid cut_type: {req.cut_type}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # After cutting, input_random behaves identically to input —
    # normalize so all downstream code only needs to handle "input".
    effective_cut_type = "input" if req.cut_type == "input_random" else req.cut_type

    # Track which ancestor input-cut nodes are still reachable
    # (present as a Parameter in the new cut model, i.e. NOT grayed).
    new_grayed_set = set(new_grayed)
    still_reachable_cuts = set()
    for ac in parent_ancestor_cuts:
        if ac["cut_type"] == "input":
            if ac["cut_node"] not in new_grayed_set:
                still_reachable_cuts.add(ac["cut_node"])
    # Also check the parent's own cut node
    if parent_sub and parent_sub.get("cut_type") == "input":
        if parent_sub["cut_node"] not in new_grayed_set:
            still_reachable_cuts.add(parent_sub["cut_node"])

    # Accumulate grayed nodes from parent, but exclude ancestor input-cut
    # nodes that are still reachable in the new sub-model
    accumulated_grayed = list(
        (set(parent_grayed) | set(new_grayed)) - still_reachable_cuts
    )

    # Compute ancestor_cuts chain
    ancestor_cuts = parent_ancestor_cuts + [{
        "cut_node": parent_sub["cut_node"],
        "cut_type": parent_sub["cut_type"],
    }] if parent_sub else []

    # Store cut model
    sub_session = svc.create_sub_session(
        session_id=session_id,
        cut_type=effective_cut_type,
        cut_node=req.node_name,
        grayed_nodes=accumulated_grayed,
        parent_sub_session_id=req.parent_sub_session_id,
        ancestor_cuts=ancestor_cuts,
    )

    # Serialize cut model to sub-session directory for subprocess inference
    import openvino as ov

    sub_dir = svc._session_path(session_id) / "sub_sessions" / sub_session.id
    cut_model_abs = str(sub_dir / "cut_model.xml")
    ov.save_model(cut_model, cut_model_abs)

    # Store session-relative paths in metadata
    rel_cut_model = f"sub_sessions/{sub_session.id}/cut_model.xml"

    # For input cuts, save the .npy input and store accumulated input_configs
    if effective_cut_type == "input":
        inputs_dir = sub_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        import numpy as np
        param_name = req.node_name
        safe_filename = param_name.replace("/", "_").replace("\\", "_")
        npy_save_path = str(inputs_dir / f"{safe_filename}.npy")
        np.save(npy_save_path, input_data)

        # Store session-relative path for input config
        rel_npy_path = f"sub_sessions/{sub_session.id}/inputs/{safe_filename}.npy"
        new_config = {"name": param_name, "source": "file", "path": rel_npy_path}
        accumulated_configs = parent_input_configs_rel + [new_config]

        svc.update_sub_session_meta(session_id, sub_session.id, {
            "model_path": rel_cut_model,
            "input_configs": accumulated_configs,
        })
    else:
        # For output cuts, copy parent's input .npy files into this sub-session
        # so each sub-session is self-contained. Only copy inputs that are still
        # reachable (not grayed out by this cut).
        import shutil
        copied_configs = []
        for cfg in parent_input_configs_rel:
            if cfg.get("source") != "file" or not cfg.get("path"):
                copied_configs.append(cfg)
                continue
            # Skip inputs for nodes that got grayed out by this cut
            if cfg["name"] in new_grayed_set:
                continue
            inputs_dir = sub_dir / "inputs"
            inputs_dir.mkdir(exist_ok=True)
            src_abs = str(svc._session_path(session_id) / cfg["path"])
            safe_filename = cfg["name"].replace("/", "_").replace("\\", "_")
            dst_rel = f"sub_sessions/{sub_session.id}/inputs/{safe_filename}.npy"
            dst_abs = str(svc._session_path(session_id) / dst_rel)
            shutil.copy2(src_abs, dst_abs)
            copied_configs.append({**cfg, "path": dst_rel})

        svc.update_sub_session_meta(session_id, sub_session.id, {
            "model_path": rel_cut_model,
            "input_configs": copied_configs,
        })

    # Broadcast sub-session creation via WS
    from backend.ws.handler import ws_manager
    await ws_manager.broadcast(session_id, {
        "type": "sub_session_created",
        "sub_session_id": sub_session.id,
        "parent_sub_session_id": req.parent_sub_session_id,
        "cut_type": effective_cut_type,
        "cut_node": req.node_name,
        "grayed_nodes": accumulated_grayed,
        "ancestor_cuts": ancestor_cuts,
    })

    return sub_session


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
