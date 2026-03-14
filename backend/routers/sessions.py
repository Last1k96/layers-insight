"""Session management routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.schemas.session import CutRequest, SessionConfig, SessionDetail, SessionInfo, SubSessionInfo

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _get_session_service(request: Request):
    return request.app.state.session_service


@router.post("", response_model=SessionInfo)
async def create_session(config: SessionConfig, request: Request) -> SessionInfo:
    """Create a new session."""
    svc = _get_session_service(request)
    return svc.create_session(config)


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
    parent_grayed: list[str] = []
    parent_input_configs: list[dict] = []
    parent_ancestor_cuts: list[dict] = []

    if req.parent_sub_session_id:
        parent_sub = svc.get_sub_session_meta(session_id, req.parent_sub_session_id)
        if parent_sub is None:
            raise HTTPException(status_code=400, detail="Parent sub-session not found")
        parent_grayed = parent_sub.get("grayed_nodes", [])
        parent_input_configs = parent_sub.get("input_configs", [])
        parent_ancestor_cuts = parent_sub.get("ancestor_cuts", [])

    try:
        if req.cut_type == "output":
            if req.parent_sub_session_id and parent_sub:
                # Read the parent sub-session's cut model
                import openvino as ov
                parent_model_path = parent_sub.get("model_path")
                if not parent_model_path:
                    raise HTTPException(status_code=400, detail="Parent sub-session has no model")
                source_model = ov.Core().read_model(parent_model_path)
                cut_model, new_grayed = model_cut_svc.make_output_node(source_model, req.node_name)
            else:
                cut_model, new_grayed = model_cut_svc.make_output_node(model, req.node_name)
        elif req.cut_type == "input":
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

            # Use parent sub-session's model path or root model path
            if req.parent_sub_session_id and parent_sub:
                source_model_path = parent_sub.get("model_path", session.config.model_path)
            else:
                source_model_path = session.config.model_path

            cut_model, input_data, new_grayed = model_cut_svc.make_input_node(
                source_model_path, req.node_name, str(npy_path), req.input_precision,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid cut_type: {req.cut_type}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Normalize sub-model node names back to original graph names.
    # When cutting a sub-model, Parameter(X) nodes created by prior input cuts
    # appear with that wrapper name — map them back so the frontend can match.
    normalized_grayed = []
    for name in new_grayed:
        if name.startswith("Parameter(") and name.endswith(")"):
            normalized_grayed.append(name[len("Parameter("):-1])
        else:
            normalized_grayed.append(name)

    # Accumulate grayed nodes from parent
    accumulated_grayed = list(set(parent_grayed + normalized_grayed))

    # Compute ancestor_cuts chain
    ancestor_cuts = parent_ancestor_cuts + [{
        "cut_node": parent_sub["cut_node"],
        "cut_type": parent_sub["cut_type"],
    }] if parent_sub else []

    # Store cut model
    sub_session = svc.create_sub_session(
        session_id=session_id,
        cut_type=req.cut_type,
        cut_node=req.node_name,
        grayed_nodes=accumulated_grayed,
        parent_sub_session_id=req.parent_sub_session_id,
        ancestor_cuts=ancestor_cuts,
    )

    # Serialize cut model to sub-session directory for subprocess inference
    import openvino as ov

    sub_dir = svc._session_path(session_id) / "sub_sessions" / sub_session.id
    cut_model_path = str(sub_dir / "cut_model.xml")
    ov.save_model(cut_model, cut_model_path)

    # For input cuts, save the .npy input and store accumulated input_configs
    if req.cut_type == "input":
        inputs_dir = sub_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        import numpy as np
        param_name = f"Parameter({req.node_name})"
        safe_filename = param_name.replace("/", "_").replace("\\", "_")
        npy_save_path = str(inputs_dir / f"{safe_filename}.npy")
        np.save(npy_save_path, input_data)

        new_config = {"name": param_name, "source": "file", "path": npy_save_path}
        accumulated_configs = parent_input_configs + [new_config]

        svc.update_sub_session_meta(session_id, sub_session.id, {
            "model_path": cut_model_path,
            "input_configs": accumulated_configs,
        })
    else:
        svc.update_sub_session_meta(session_id, sub_session.id, {
            "model_path": cut_model_path,
            "input_configs": parent_input_configs,
        })

    # Broadcast sub-session creation via WS
    from backend.ws.handler import ws_manager
    await ws_manager.broadcast(session_id, {
        "type": "sub_session_created",
        "sub_session_id": sub_session.id,
        "parent_sub_session_id": req.parent_sub_session_id,
        "cut_type": req.cut_type,
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
