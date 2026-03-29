# Layers-Insight: Strategic Analysis & Roadmap

## Context

Layers-Insight is a neural network accuracy debugger for OpenVINO. An NPU developer loads a model, visualizes its graph, runs layer-by-layer inference on both the NPU (main device) and CPU (reference device), and compares outputs to find where accuracy degrades. The tool has a FastAPI backend, Svelte 5 frontend, WebGPU graph rendering, and a rich deep-accuracy visualization system.

---

## Bugs to Fix

### Bug 1: `Math.min(...fp32)` stack overflow
- **Problem**: `frontend/src/lib/stores/tensors.svelte.ts:51-52` calls `Math.min(...fp32)` and `Math.max(...fp32)` which spreads the entire Float32Array as function arguments. For tensors with >100K elements, this throws "Maximum call stack size exceeded" because JavaScript has a call stack argument limit.
- **Fix**: Replace with a simple loop: `let min = Infinity, max = -Infinity; for (let i = 0; i < fp32.length; i++) { if (fp32[i] < min) min = fp32[i]; if (fp32[i] > max) max = fp32[i]; }`
- **Files**: `frontend/src/lib/stores/tensors.svelte.ts` (lines 51-52)

### Bug 2: Non-atomic metadata writes
- **Problem**: `backend/services/session_service.py` writes `metadata.json` directly via `json.dump()`. If the server crashes mid-write (e.g., OOM, power loss), the JSON is truncated/corrupted and the session is unrecoverable.
- **Fix**: Write to a temp file in the same directory (`metadata.json.tmp`), then `os.replace()` (atomic on POSIX). Pattern: `with open(tmp, 'w') as f: json.dump(...); os.replace(tmp, target)`.
- **Files**: `backend/services/session_service.py` -- find all `_write_metadata` or equivalent calls

### Bug 3: Code duplication `_get_reachable_params`
- **Problem**: Identical backward-walk logic exists in both `backend/services/model_cut_service.py` and `backend/utils/inference_worker.py`. If one is updated and the other isn't, behavior diverges.
- **Fix**: Extract to a shared module (e.g., `backend/utils/ov_graph_utils.py`) and import from both files.
- **Files**: `backend/services/model_cut_service.py`, `backend/utils/inference_worker.py` -- search for `_get_reachable_params`

---

## Improvements to Existing Features

### Improve 1: Accuracy columns in queue panel
- **Problem**: Queue panel shows task status (waiting/executing/success/failed) but no accuracy numbers. User must click each node individually to see if accuracy is bad.
- **Fix**: Add MSE and cosine_similarity columns to the queue task list. Data already available in `task.metrics`. Allow sorting by these columns. Color-code values (green=good, yellow=warning, red=bad).
- **Files**: `frontend/src/lib/panels/QueuePanel.svelte` (add columns), `frontend/src/lib/stores/types.ts` (AccuracyMetrics already has the fields)

### Improve 2: Persistent accuracy toggle (not Alt-hold)
- **Problem**: The accuracy overlay (coloring nodes by MSE on the graph) only activates while holding Alt. This is a momentary view -- user can't interact with nodes while holding Alt.
- **Fix**: Add a toggle button in the graph toolbar. When toggled on, persist `accuracyViewActive = true` in graph store. Support coloring by cosine_similarity and max_abs_diff too (dropdown selector). Keep Alt-hold as a quick preview.
- **Files**: `frontend/src/lib/stores/graph.svelte.ts` (`accuracyViewActive`), `frontend/src/lib/graph/webgpu/WebGPURenderer.ts` (reads the flag), `frontend/src/lib/components/MainView.svelte` or wherever the toolbar lives (add toggle button)

### Improve 3: Multi-output capture
- **Problem**: `inference_worker.py` only captures `req.get_output_tensor(0)`. Nodes with multiple outputs (Split, TopK, NonMaxSuppression) lose secondary outputs. Metrics are computed only on first output.
- **Fix**: Iterate over all compiled model outputs. Save each as `main_output_0.npy`, `main_output_1.npy`, etc. Compute metrics per-output. Return all in task result. Frontend shows per-output tabs or selector.
- **Files**: `backend/utils/inference_worker.py` (output capture loop), `backend/schemas/inference.py` (task result schema for multiple outputs), `frontend/src/lib/panels/NodeStatus.svelte` (output selector)

### Improve 4: Batch queue intelligence
- **Problem**: Current batch queue does BFS traversal with stride -- topology-blind. Misses side branches. User must manually select start node and configure.
- **Fix**: Add modes: (a) "Infer all un-inferred nodes" (no node selection needed), (b) "Infer all nodes of type X" (filter by op type), (c) "Infer along critical path" (longest path from input to output). Add a top-level "Infer All" button.
- **Files**: `frontend/src/lib/panels/BatchQueue.svelte` (new mode selector), `backend/routers/inference.py` (batch endpoint already exists), `frontend/src/lib/components/MainView.svelte` (top-level button)

### Improve 5: Graph filtering
- **Problem**: Only substring search available. For a 3000-node model, user can't quickly find "all Convolutions with MSE > 0.001" or "all un-inferred nodes."
- **Fix**: Add a filter bar with: op type dropdown (from op_categories), accuracy range slider, status checkboxes (inferred/not inferred/failed), shape text filter. Filtered-out nodes should be dimmed (like grayed nodes), not hidden.
- **Files**: `frontend/src/lib/graph/GraphSearch.svelte` (extend or create `GraphFilter.svelte`), `frontend/src/lib/stores/graph.svelte.ts` (filter state), `frontend/src/lib/graph/webgpu/WebGPURenderer.ts` (dimming logic, similar to grayed nodes)

### Improve 6: Sub-session navigation UX
- **Problem**: Switching sub-sessions updates grayed nodes but doesn't center the view on the cut point. User must manually scroll/search for it.
- **Fix**: When switching sub-session, auto-center (animate pan/zoom) to the cut node. The cut node name is available in `sub_session.cut_node`.
- **Files**: `frontend/src/lib/panels/SubSessionNav.svelte` (on click handler), `frontend/src/lib/graph/panZoom.ts` (has `centerOnNode` or similar)

---

## New Features

### Feature 1: Automated bisection mode
- **Problem**: Finding the first accuracy-dropping layer in a 500-layer model requires manual node-by-node inference. This is the #1 time sink for NPU developers.
- **How**: Binary search the topological order. Given start (model input) and end (selected node or model output), infer the midpoint node. If cosine similarity >= threshold, search second half; else search first half. Converges in log2(N) steps (~9 for 500 layers).
- **Backend**: New endpoint `POST /api/inference/bisect` that accepts `{session_id, start_node, end_node, threshold, sub_session_id}`. Runs the loop server-side, broadcasting progress via WebSocket. Returns the first failing node.
- **Frontend**: Bisect panel or modal: select start/end nodes (or use defaults), set threshold, start button. Show binary search progress visualization (which range is being searched, current midpoint).
- **Files**: New `backend/routers/bisect.py` or extend `inference.py`, new `frontend/src/lib/panels/BisectPanel.svelte` (stub already exists), `frontend/src/lib/stores/graph.svelte.ts` (bisect state)

### Feature 2: Accuracy summary dashboard
- **Problem**: No overview of accuracy across the whole model. User must click nodes one by one.
- **How**: A dedicated view showing: (a) worst-N nodes by each metric, (b) accuracy distribution histogram (cosine > 0.999, 0.99-0.999, < 0.99), (c) error trend chart along topological order ("waveform" view).
- **Backend**: New endpoint `GET /api/sessions/{id}/accuracy-summary` that aggregates all task results for the session.
- **Frontend**: New component `AccuracySummary.svelte` with Chart.js or canvas-drawn charts. Accessible from session header or toolbar.
- **Files**: New `backend/routers/accuracy.py`, new `frontend/src/lib/panels/AccuracySummary.svelte`

### Feature 3: Accuracy threshold configuration
- **Problem**: User sees raw numbers (MSE=0.00234, cosine=0.9987) but must mentally judge if they're acceptable. No visual distinction between "fine" and "problematic" nodes.
- **How**: User-configurable thresholds: `{cosine_warning: 0.999, cosine_fail: 0.99, mse_warning: 0.001, mse_fail: 0.01}`. Applied to: graph node coloring (green/yellow/red), queue panel (icon/color), accuracy view (pass/fail badge). Store in config store, persist to localStorage.
- **Files**: `frontend/src/lib/stores/config.svelte.ts` (already has `globalThreshold`/`categoryThresholds` -- extend), `frontend/src/lib/graph/webgpu/nodesPipeline.ts` (node coloring), `frontend/src/lib/panels/QueuePanel.svelte`, `frontend/src/lib/panels/NodeStatus.svelte`

### Feature 4: Cross-session comparison
- **Problem**: No way to compare results between sessions (e.g., "same model, NPU driver v1 vs. v2" or "before vs. after optimization").
- **How**: New view: select two sessions, match nodes by name, show side-by-side metrics table with delta column. Highlight improved/regressed nodes.
- **Files**: New `frontend/src/lib/views/SessionCompare.svelte`, extend `backend/routers/sessions.py` with comparison endpoint

### Feature 5: Reproducibility export
- **Problem**: When an NPU developer finds a bad layer, they need to file a bug report with the driver team. Currently must manually gather files.
- **How**: One-click "Export reproducer" button on NodeStatus for successful tasks. Backend zips: cut_model.xml/.bin + input .npy files + main_output.npy + ref_output.npy + metrics JSON + environment info (OV version, device, driver). Returns as downloadable .zip.
- **Files**: New endpoint `GET /api/tensors/{session_id}/{task_id}/export`, `frontend/src/lib/panels/NodeStatus.svelte` (add export button)

### Feature 6: Real input data support
- **Problem**: Random inputs can mask or create accuracy issues that don't appear with real data. NPU accuracy problems are often data-dependent.
- **How**: In session creation, support: (a) image folder path (loads images one by one), (b) .npy directory (each file = one input), (c) calibration dataset. Store dataset reference in session config. Allow re-running inference with different inputs from the dataset.
- **Files**: `backend/utils/input_generator.py` (extend), `frontend/src/lib/views/NewSession.svelte` (dataset picker), `backend/services/session_service.py` (dataset config)

---

## Refactoring

### Refactor 1: Cut endpoint -> service
- **Problem**: `backend/routers/sessions.py:84-268` has 185 lines of business logic (OV model manipulation, file I/O, sub-session state) in the router.
- **Fix**: Move to `model_cut_service.py` or new `sub_session_service.py`. Router should only do request validation and call service method.
- **Files**: `backend/routers/sessions.py`, `backend/services/model_cut_service.py`

### Refactor 2: Inference worker decomposition
- **Problem**: `inference_worker.py` (359 lines) handles everything in one function.
- **Fix**: Split into: `load_and_cut_model()`, `prepare_inputs()`, `run_inference()`, `compute_metrics()`, `save_results()`. Each function testable independently.
- **Files**: `backend/utils/inference_worker.py`

### Refactor 3: Frontend store reactivity
- **Problem**: `new Map([...this.cache, [key, value]])` creates O(n) copies on every update.
- **Fix**: Use Svelte 5's native Map/Set tracking or batch updates into fewer copy operations.
- **Files**: `frontend/src/lib/stores/tensors.svelte.ts`, `frontend/src/lib/stores/queue.svelte.ts`, `frontend/src/lib/stores/graph.svelte.ts`

---

## Implementation Order

1. **Bug fixes** (immediate): Bug 1 (Math.min), Bug 2 (atomic writes), Bug 3 (dedup)
2. **Quick wins** (days): Improve 1 (queue columns), Improve 2 (accuracy toggle), Improve 6 (sub-session center)
3. **Core value** (weeks): Feature 3 (thresholds), Feature 1 (bisection), Feature 2 (summary dashboard)
4. **Workflow** (weeks): Improve 4 (batch intelligence), Improve 5 (filtering), Feature 5 (export)
5. **Scale** (months): Feature 4 (cross-session), Feature 6 (real data), Refactoring
