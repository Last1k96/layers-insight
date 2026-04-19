# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Layers-Insight is a neural-network accuracy debugger for OpenVINO. An NPU developer loads a model, views its graph, runs layer-by-layer inference on a main device (e.g. NPU) and a reference device (e.g. CPU), and compares outputs to localize where accuracy degrades. FastAPI backend + Svelte 5 frontend + a custom WebGPU graph renderer (no Sigma/Three).

## Running & building

The entry point handles everything (local Node.js, Python venv, deps, frontend build, launch):

```bash
./start.py --ov-path /path/to/openvino/bin/intel64/Release --model model.xml --input random
# --no-https to disable TLS; --port to change; --main-device/--ref-device for device split
```

Dev loop without rebuilding the bundle on every change:

```bash
make dev-backend     # uvicorn backend.main:app --reload on :8000
make dev-frontend    # vite dev on :5173, proxies /api and /ws to :8000
```

Tests:

```bash
make test                                          # all backend tests (201+)
source .venv/bin/activate && pytest tests/backend/test_inference_service.py -v
source .venv/bin/activate && pytest tests/api/test_inference_api.py::test_name -v
cd frontend && npm test                            # vitest (happy-dom)
cd frontend && npm test -- path/to/file.test.ts    # single file
cd frontend && npm run build                       # production bundle into frontend/dist/
```

The prod bundle in `frontend/dist/` is served by FastAPI's `StaticFiles` mount, so after a backend-only change you don't need to rebuild. After a frontend-only change in `--reload` mode, run `cd frontend && npm run build` (or use `npm run dev` with the Vite proxy).

## OpenVINO rules (non-obvious)

- **Never rely on system OpenVINO.** All OV capabilities must come from the user's `--ov-path` (e.g. `/home/last1k/code/openvino/bin/intel64/Release`). `start.py` / `start.sh` prepend this to `LD_LIBRARY_PATH` and `PYTHONPATH`.
- Don't use internal classes like `ov.runtime.op.Parameter` — they may not exist in custom builds. To identify Parameter nodes, compare identity against `model.get_parameters()` with `id()`, not `isinstance`.
- Inference always runs in a subprocess (`backend/utils/inference_worker.py`) to isolate C++ segfaults from the FastAPI process.
- Inference errors **never raise** out of services; they are captured in `InferenceTask.error_detail` and surfaced to the client via WebSocket.

## Backend architecture

FastAPI app factory in `backend/main.py`. `lifespan()` initializes singletons and attaches them to `app.state`:

- `session_service` — file-backed store (`sessions/{id}/metadata.json` + `tensors/{task_id}/*.npy`). Metadata writes are atomic (temp file + `os.replace`).
- `inference_service` — cuts the model (`ov.Model([target_op.output(0)], model.get_parameters())`) and dual-runs main/ref devices via subprocess.
- `model_cut_service` — creates sub-session cut models for deep-dive inference.
- `queue_service` — single asyncio worker that pops tasks and calls the injected `on_infer` callback via `asyncio.to_thread` (OV releases the GIL).
- `bisect_service` — multi-job binary search over topological order. Child tasks use `batch_id = "bisect:{job_id}"`; completion is routed by prefix, not by live service state, to avoid races during pause transitions.
- `upload_service` — chunked uploads with TTL sweeper.
- `ws_manager` (`backend/ws/handler.py`) — per-session broadcast; `send_task_status` and ad-hoc messages like `inference_log`.

Services are accessed in routers via `request.app.state.<name>`. Routers live in `backend/routers/{sessions,graph,inference,bisect,tensors,devices,uploads}.py`. Keep routers thin — they validate and delegate.

Graph layout runs through a Node.js subprocess that shells out to `elkjs` (`backend/utils/elk_layout.js`). Layout results are cached on the session.

Inference task model (`backend/schemas/inference.py`) has extensibility hooks that are in active use: `batch_id`, `sub_session_id`, `bisect_id`. `metadata.json` carries a `schema_version` and migrates legacy `bisect_job` → `bisect_jobs` on read.

## Frontend architecture

Svelte 5 **runes** syntax (`$state`, `$derived`). Stores are classes exported as singletons from `frontend/src/lib/stores/*.svelte.ts`:

- `session`, `graph`, `queue`, `tensors`, `config`, `metrics` (IndexedDB cache via `idb`), `bisect`, `log`, `upload`, `advancedFilter`.

View router in `App.svelte`: picker → new-session → main. The main graph view is `frontend/src/lib/graph/` which contains both the custom WebGPU renderer (`webgpu/`) and interaction logic.

The deep-accuracy visualization system under `frontend/src/lib/accuracy/` has its own WebGPU pipelines (`accuracy/webgpu/`) for voxel/overlay rendering of tensor comparisons. It is code-split into its own chunk (`manualChunks` in `vite.config.ts`).

Vite config sets `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers. **Do not remove them** — they unlock 5 µs `performance.now()` precision (instead of Chrome's 100 µs Spectre clamp), which the bench harness depends on.

## WebGPU renderer invariants

The renderer in `frontend/src/lib/graph/webgpu/` has hard-won perf characteristics. Breaking any of these regresses frame time by 20-100x.

- **Pan/zoom must not call `scheduleRefresh()`.** Camera-only changes go through `rebuildCameraDependentParts()`, which fires only when text fade thresholds cross or when an edge is selected.
- **Edge geometry is built once per graph**, in `setGraph()`. `EdgeGeometry { positions, edgeRanges, vertexCount }` is cached. Color updates swap a separate color buffer via `useCachedEdgeColors()` — they do not re-tessellate.
- `hexToRgb` must return from the module-level `Map` cache, not recompute.
- `_lastHoveredEdge` must track the actual current hovered edge, not get reset to `-1` — otherwise every accuracy toggle triggers a spurious edge color rebuild.
- Preallocated scratch arrays (`nodeDataScratch`, `glyphDataScratch`) are reused across frames.

If asked to optimize the renderer: run the bench first (`frontend/perf.html` via the `webgpu-bench` skill), don't re-investigate hotspots already addressed. Known post-optimization numbers on `cached-real` (496 nodes / 600 edges): `accuracyToggle` ~3.7 ms, pan/zoom/idle ~0.1 ms.

## Testing notes

- Backend tests use `pytest` + `pytest-asyncio` + `httpx`. Fixtures are in `tests/conftest.py`.
- API tests (`tests/api/`) exercise full FastAPI app via TestClient; service tests (`tests/backend/`) hit services directly.
- There is no mocked OpenVINO — tests that need it are skipped when OV isn't importable. Tests that only exercise graph utils, queue state, sessions, uploads, DAG layout, etc. run without OV.

## Environment quirks

- WSL2; no sudo. `start.py` downloads Node.js 20 into `.node/` and a venv into `.venv/` — don't try to use system Node or a global Python install.
- pyproject.toml uses `setuptools.build_meta` (not `backends._legacy`, which doesn't exist). `@sveltejs/vite-plugin-svelte` v5 is required for Vite 6.
- HTTPS is on by default; `start.py` auto-generates self-signed certs via `backend/certs.py`. Pass `--no-https` to disable.

## Debugging a running server

The backend exposes diagnostic endpoints that are handy when something's stuck:

```bash
curl -s http://localhost:8000/api/inference/queue-state        # {"paused": bool}
curl -sX POST http://localhost:8000/api/inference/resume
curl -s http://localhost:8000/api/inference/bisect/status      # all bisect jobs
curl -sX POST http://localhost:8000/api/inference/bisect/{job_id}/stop
```

## Roadmap

Strategic direction lives in `ROADMAP.md` (16 ordered items, with the first three phases of the product already implemented). Consult it before proposing large features — item ordering encodes dependency assumptions.
