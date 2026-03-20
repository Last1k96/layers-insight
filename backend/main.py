"""FastAPI application entry point."""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from backend.config import AppConfig, parse_cli_args
from backend.routers import devices, graph, inference, sessions, tensors
from backend.services.inference_service import InferenceService
from backend.utils.ov_helpers import register_plugins
from backend.services.queue_service import QueueService
from backend.services.session_service import SessionService
from backend.ws.handler import ws_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init services on startup, cleanup on shutdown."""
    config = app.state.config

    # Initialize OpenVINO Core
    ov_core = None
    try:
        import openvino as ov
        ov_core = ov.Core()

        # Register plugins from custom OV build path if provided
        register_plugins(ov_core, config.ov_path)

        print(f"OpenVINO initialized. Available devices: {ov_core.available_devices}")

        from backend.utils.model_converter import get_available_frontends
        frontends = get_available_frontends(ov_core)
        if frontends:
            print(f"Available model frontends: {', '.join(frontends)}")
        else:
            print("No model frontends detected (only IR format supported)")
    except ImportError:
        print("WARNING: OpenVINO not installed. Running in UI-only mode.", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: OpenVINO init failed: {e}. Running in UI-only mode.", file=sys.stderr)

    app.state.ov_core = ov_core
    app.state.models = {}  # session_id -> ov.Model cache

    # Initialize services
    session_service = SessionService(config.sessions_dir)
    app.state.session_service = session_service

    inference_service = InferenceService(ov_core, ov_path=config.ov_path) if ov_core else None
    app.state.inference_service = inference_service

    from backend.services.model_cut_service import ModelCutService
    model_cut_service = ModelCutService(ov_core) if ov_core else None
    app.state.model_cut_service = model_cut_service

    queue_service = QueueService()
    app.state.queue_service = queue_service

    # Set up queue callbacks
    async def on_task_notify(task):
        await ws_manager.send_task_status(task)

    async def on_infer(task):
        if inference_service is None:
            task.status = "failed"
            task.error_detail = "OpenVINO not available"
            return task

        model = app.state.models.get(task.session_id)
        if model is None:
            task.status = "failed"
            task.error_detail = "Model not loaded. Open the graph first."
            return task

        session = session_service.get_session(task.session_id)
        if session is None:
            task.status = "failed"
            task.error_detail = "Session not found"
            return task

        # Pass per-input configs if available
        input_configs = None
        if session.config.inputs:
            input_configs = [inp.model_dump() for inp in session.config.inputs]

        # Sub-session routing: use cut model path and merged input configs
        infer_model_path = session.config.model_path
        if task.sub_session_id:
            sub_meta = session_service.get_sub_session_meta_resolved(task.session_id, task.sub_session_id)
            if sub_meta:
                if sub_meta.get("model_path"):
                    infer_model_path = sub_meta["model_path"]
                if sub_meta.get("input_configs"):
                    # Merge: sub-session input configs take priority
                    sub_cfgs = sub_meta["input_configs"]
                    if input_configs:
                        input_configs = input_configs + sub_cfgs
                    else:
                        input_configs = sub_cfgs

        # Set up real-time log streaming via WebSocket
        loop = asyncio.get_running_loop()

        def log_callback(task_id: str, level: str, message: str) -> None:
            from datetime import datetime, timezone
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast(task.session_id, {
                    "type": "inference_log",
                    "task_id": task_id,
                    "node_name": task.node_name,
                    "level": level,
                    "message": message,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }),
                loop,
            )

        def stage_callback(stage: str) -> None:
            """Send a task_status update when the inference stage changes."""
            asyncio.run_coroutine_threadsafe(
                ws_manager.send_task_status(task),
                loop,
            )

        log_callback(task.task_id, "info", f"Task started for node '{task.node_name}'")

        result = await asyncio.to_thread(
            inference_service.cut_and_infer,
            model=model,
            target_node_name=task.node_name,
            main_device=session.config.main_device,
            ref_device=session.config.ref_device,
            model_path=infer_model_path,
            input_path=session.config.input_path,
            precision=session.config.input_precision,
            task=task,
            input_configs=input_configs,
            log_callback=log_callback,
            stage_callback=stage_callback,
        )

        # If the task was deleted while executing, skip saving results
        if queue_service.is_deleted(task.task_id):
            if isinstance(result, tuple):
                import shutil
                shutil.rmtree(result[1], ignore_errors=True)
            return task

        # Handle tuple result (task, artifacts_dir)
        if isinstance(result, tuple):
            updated_task, artifacts_dir = result
            try:
                session_service.save_task_result(
                    session_id=task.session_id,
                    task_id=task.task_id,
                    task_data=updated_task.model_dump(),
                    artifacts_dir=artifacts_dir,
                    sub_session_id=task.sub_session_id,
                )
            finally:
                import shutil
                shutil.rmtree(artifacts_dir, ignore_errors=True)
            log_callback(task.task_id, "info", f"Task completed for node '{task.node_name}'")
            return updated_task
        else:
            # Error case — result is just the task
            session_service.save_task_result(
                session_id=task.session_id,
                task_id=task.task_id,
                task_data=result.model_dump(),
            )
            log_callback(task.task_id, "error", f"Task failed for node '{task.node_name}': {result.error_detail}")
            return result

    queue_service.set_callbacks(notify=on_task_notify, infer=on_infer)
    await queue_service.start_worker()

    print(f"\n  Open in browser: http://localhost:{config.port}\n")
    yield

    # Shutdown
    await queue_service.stop_worker()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    cli_args = parse_cli_args()
    config = AppConfig(**cli_args)

    app = FastAPI(title="Layers-Insight", lifespan=lifespan)
    app.state.config = config

    # Include routers
    app.include_router(devices.router)
    app.include_router(sessions.router)
    app.include_router(graph.router)
    app.include_router(inference.router)
    app.include_router(tensors.router)

    # WebSocket endpoint
    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        await ws_manager.connect(session_id, websocket)
        try:
            while True:
                data = await websocket.receive_json()
                # Handle client messages
                msg_type = data.get("type")
                if msg_type == "cancel_task":
                    task_id = data.get("task_id")
                    if task_id:
                        queue_svc = app.state.queue_service
                        await queue_svc.cancel(task_id)
        except WebSocketDisconnect:
            ws_manager.disconnect(session_id, websocket)
        except Exception:
            ws_manager.disconnect(session_id, websocket)

    # Serve frontend static files (production build)
    frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

    return app


# Module entry point
app = create_app()

if __name__ == "__main__":
    import uvicorn
    config = app.state.config
    uvicorn.run(
        "backend.main:app",
        host=config.host,
        port=config.port,
        reload=False,
    )
