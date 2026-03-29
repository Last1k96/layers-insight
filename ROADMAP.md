# Layers-Insight: Strategic Analysis & Roadmap

## Context

Layers-Insight is a neural network accuracy debugger for OpenVINO. An NPU developer loads a model, visualizes its graph, runs layer-by-layer inference on both the NPU (main device) and CPU (reference device), and compares outputs to find where accuracy degrades. The tool has a FastAPI backend, Svelte 5 frontend, WebGPU graph rendering, and a rich deep-accuracy visualization system.

---

## Bugs (DONE)

All three bugs have been fixed:
- ~~Bug 1: `Math.min(...fp32)` stack overflow~~ -- replaced with loop
- ~~Bug 2: Non-atomic metadata writes~~ -- atomic write via temp file + `os.replace()`
- ~~Bug 3: Code duplication `_get_reachable_params`~~ -- extracted to `backend/utils/ov_graph_utils.py`

---

## Roadmap (ordered by priority and dependency)

### 1. Refactor: Cut endpoint -> service
- **Problem**: `backend/routers/sessions.py:84-268` has 185 lines of business logic (OV model manipulation, file I/O, sub-session state) in a router function.
- **Fix**: Move to `model_cut_service.py` or a new `sub_session_service.py`. Router should only validate the request and call the service.
- **Why first**: Bisection mode and clone workflow both need clean cut logic. Refactoring now prevents duplication later.
- **Files**: `backend/routers/sessions.py`, `backend/services/model_cut_service.py`

### 2. Refactor: Inference worker decomposition
- **Problem**: `inference_worker.py` (330 lines) handles model loading, cutting, input prep, two-device inference, metrics, and JSON output in one large function.
- **Fix**: Split into: `load_and_cut_model()`, `prepare_inputs()`, `run_inference()`, `compute_metrics()`, `save_results()`. Each function testable independently.
- **Why now**: Multi-output capture, plugin config, and bisection all extend the worker. Decomposing first makes those additions clean.
- **Files**: `backend/utils/inference_worker.py`

### 3. Refactor: Frontend store reactivity
- **Problem**: `new Map([...this.cache, [key, value]])` creates O(n) copies on every update. With large models and frequent WebSocket updates during batch inference, this becomes sluggish.
- **Fix**: Use Svelte 5's native Map/Set tracking or batch updates into fewer copy operations.
- **Why now**: Batch inference, bisection, and real-time queue updates all generate rapid store mutations.
- **Files**: `frontend/src/lib/stores/tensors.svelte.ts`, `frontend/src/lib/stores/queue.svelte.ts`, `frontend/src/lib/stores/graph.svelte.ts`

### 4. Unified accuracy system (thresholds + persistent toggle + metric selector)
Merges the old "accuracy threshold config" and "persistent accuracy toggle" into one cohesive system.

- **Problem**: The accuracy overlay only works while holding Alt (can't interact with the graph). Raw metric numbers require mental interpretation. No consistent coloring across views.
- **Fix**: A unified accuracy configuration that drives coloring everywhere:
  - **Persistent toggle button** on graph toolbar (plus keyboard shortcut, e.g. `A`) to enable accuracy coloring mode. Alt-hold remains as a quick preview.
  - **Metric selector dropdown** next to the toggle: cosine similarity (default), MSE, max_abs_diff.
  - **Range controls** in the dropdown to define the color gradient (e.g., cosine 0.99-1.0 maps red-to-green).
  - **Consistent coloring** across: graph node fill (accuracy mode), node outline (normal mode), queue panel metric text, and accuracy-colored edges in Alt view.
  - **Text rendering**: In accuracy mode, node labels drawn with outline/halo for legibility on top of colored backgrounds. Non-inferred node names hidden (current behavior).
  - **Multi-output edge coloring**: In accuracy mode, each edge uses the accuracy of the specific output port it represents, not the node's worst-case. A Split with one good and one bad output shows green and red edges respectively.
  - Thresholds persist to localStorage.
- **Files**: `frontend/src/lib/stores/config.svelte.ts` (threshold state), `frontend/src/lib/graph/webgpu/nodesPipeline.ts` (node coloring), `frontend/src/lib/graph/webgpu/edgesPipeline.ts` (edge coloring), `frontend/src/lib/graph/webgpu/textPipeline.ts` (text outline), `frontend/src/lib/panels/QueuePanel.svelte`, `frontend/src/lib/panels/NodeStatus.svelte`, `frontend/src/lib/components/MainView.svelte` (toolbar)

### 5. Multi-output capture
- **Problem**: `inference_worker.py` only captures `output(0)`. Multi-output nodes (Split, TopK, NonMaxSuppression) lose secondary outputs. Accuracy problems on non-first outputs are silently missed.
- **Fix**:
  - **Backend**: Iterate over all compiled model outputs. Save each as `main_output_0.npy`, `main_output_1.npy`, etc. Compute metrics per-output.
  - **NodeStatus**: Show all outputs stacked vertically, each with its own metrics (MSE, cosine, per-device stats) and its own "Deep Accuracy View" button. No tabs or toggles -- all visible at once. Single-output nodes look unchanged.
  - **Queue panel**: Show the worst metric across all outputs so nothing slips through.
  - **Accuracy view**: Opens for one specific output (whichever "Deep Accuracy View" button was clicked). No selector inside.
- **Files**: `backend/utils/inference_worker.py` (output capture loop), `backend/schemas/inference.py` (per-output metrics), `frontend/src/lib/panels/NodeStatus.svelte`, `frontend/src/lib/panels/QueuePanel.svelte`

### 6. Accuracy columns in queue panel with sorting
- **Problem**: After batch inference, finding problematic nodes requires clicking through each task individually.
- **Fix**: Add cosine and MSE columns to each task row (shown for completed tasks only). Color-coded using the unified threshold ranges from item 4. Sortable columns:
  - **Node name column (default)**: topological order (matches graph top-to-bottom). Topological index stored per task from `graphData.nodes` array order.
  - **Cosine column**: sort by cosine similarity (ascending = worst first).
  - **MSE column**: sort by MSE (descending = worst first).
  - **Click same header again**: toggle ascending/descending.
  - For multi-output nodes, columns show worst metric across outputs.
- **Files**: `frontend/src/lib/panels/QueuePanel.svelte`, `frontend/src/lib/stores/queue.svelte.ts` (sort state)

### 7. Sub-session fit-to-view + scroll wheel speed
- **Problem**: Switching sub-sessions updates grayed nodes but doesn't move the camera. The developer must manually find the relevant area. Mouse scroll zoom is too slow for navigating large graphs.
- **Fix**:
  - **Fit-to-view**: When switching sub-sessions, compute the bounding box of all non-grayed nodes and animate the camera to fit that bounding box in the viewport with padding. For root session, fit the entire graph.
  - **Scroll wheel speed**: Increase zoom factor per wheel tick (~2x current speed).
- **Files**: `frontend/src/lib/panels/SubSessionNav.svelte` (click handler), `frontend/src/lib/graph/panZoom.ts` (fit-to-bounds method, zoom speed constant)

### 8. Cancel all + Start/Pause queue control
- **Problem**: No way to cancel a large batch at once. No way to pause mid-batch to inspect intermediate results and then resume. Essential for the clone-and-compare workflow.
- **Fix**:
  - **Cancel all button** in queue panel header. Cancels all waiting tasks in one click.
  - **Start/Pause toggle** in queue panel header. Pause semantics: if a task is currently executing, cancel it and re-queue it at the top. All waiting tasks freeze. Resume picks up from the top.
  - Queue state: `running` (default) or `paused`. New tasks can still be enqueued while paused -- they just don't execute until resumed.
- **Files**: `frontend/src/lib/panels/QueuePanel.svelte`, `backend/services/queue_service.py` (pause state, cancel-all endpoint), `backend/routers/inference.py` (new endpoints)

### 9. Keyboard shortcuts system
- **Problem**: Common actions require mouse interaction. No shortcut for accuracy toggle, search, or other frequent operations.
- **Fix**: A shortcuts system with configurable keybindings. Initial shortcuts:
  - `A`: Toggle accuracy view
  - `Ctrl+F`: Search (already exists)
  - `Escape`: Close panels/overlays
  - Additional shortcuts defined as features are built.
  - Shortcut reference accessible via `?` key.
- **Files**: New `frontend/src/lib/shortcuts.ts`, integration into relevant components

### 10. Bisection mode
- **Problem**: Finding the first accuracy-dropping layer in a 500-layer model requires manual node-by-node inference. The #1 time sink for NPU developers.
- **How**: Binary search the topological order. Fully automatic by default.
  - **Config panel**: Search target (accuracy drop / compilation failure), metric selector, threshold, node range (defaults: model input -> model output, or select two nodes).
  - **Algorithm**: Infer midpoint, if result passes threshold search second half, if fails search first half. For compilation failure, check = "did it compile?" Converges in ~log2(N) steps.
  - **Progress visualization**: Highlight current search range on graph, show narrowing animation. Each midpoint inference is a regular task visible in the queue.
  - **Completion**: Auto-select the found node, open NodeStatus. Developer can inspect all intermediate results in the queue.
  - **Controls**: Stop/cancel button. Results inspectable after cancellation.
  - **Non-monotonic accuracy**: Since accuracy can improve after a bad layer (e.g., normalization), the algorithm searches for the **first** node below threshold. If midpoint is bad, search first half (problem started earlier or here). If midpoint is good, search second half.
- **Files**: New `backend/routers/bisect.py` or extend `inference.py`, `frontend/src/lib/panels/BisectPanel.svelte` (stub exists), `frontend/src/lib/stores/graph.svelte.ts` (bisect state), WebSocket messages for bisect progress

### 11. Batch queue intelligence
- **Problem**: Current batch queue is a BFS walk from a selected node with stride. Misses side branches, requires manual setup, no "just infer everything" option.
- **Fix**: Mode tabs in the batch modal:
  - **All nodes**: Infer every node in topological order (or every Nth with stride). No start node needed. Accessible from a toolbar "Infer All" button.
  - **By type**: Pick op types from a checklist (Convolution, MatMul, etc.), infer only those.
  - **Un-inferred only**: Infer nodes that don't have results yet. Fill gaps after a partial batch.
  - **From selection**: Current behavior (forward/backward from a node with stride). Kept as one of the modes.
- **Files**: `frontend/src/lib/panels/BatchQueue.svelte` (mode tabs), `frontend/src/lib/components/MainView.svelte` (toolbar button)

### 12. Plugin configuration
- **Problem**: NPU developers debug accuracy by toggling compilation passes, precision hints, and plugin-specific options. Currently no way to set these -- model compiles with defaults.
- **Fix**: In session creation form (and clone form), a collapsible "Plugin configuration" section. When a device is selected, query available config properties via OpenVINO API (`ov_core.get_property()`, supported properties). Display as a form: toggles for booleans, dropdowns for enums, text fields for strings/numbers. Show defaults. Config stored in session metadata, passed to `compile_model()`.
- **Files**: `backend/routers/devices.py` (new endpoint for device properties), `frontend/src/lib/views/NewSession.svelte` (config section), `backend/utils/inference_worker.py` (pass config to `compile_model()`), `backend/schemas/session.py` (config in session metadata)

### 13. Clone session & compare workflow
- **Problem**: Iterative debugging requires running the same model with different parameters (inputs, devices, plugin config) and comparing results. Sessions are deliberately immutable. No comparison tooling exists.
- **Fix**: A complete "clone & compare" workflow:
  - **Clone session**: "Clone" button on session. Opens the new session form pre-filled with all current settings. Developer changes what they want (inputs, device, plugin config, etc.). Creates a new session.
  - **Auto-queue**: The clone auto-enqueues all nodes that were inferred in the source session, **ordered by worst accuracy first** from the source session's results.
  - **Start/Pause integration**: Developer hits Start, inference runs worst-first. Developer can Pause at any point to compare.
  - **Session diff view**: Select source and clone sessions, see node-by-node comparison. Matching by node name (guaranteed overlap since same model). Delta column colored: green = improved, red = regressed, gray = unchanged. Summary: "142 compared, 38 improved, 3 regressed, 101 unchanged."
  - **Resume**: Developer can resume inference after comparing, adding more data points.
- **Files**: `frontend/src/lib/views/SessionPicker.svelte` (clone button), `frontend/src/lib/views/NewSession.svelte` (pre-fill from source), `backend/routers/sessions.py` (clone endpoint with auto-queue), new `frontend/src/lib/views/SessionCompare.svelte`

### 14. Reproducibility export
- **Problem**: Filing a bug report to the driver team requires manually gathering cut model, inputs, outputs, and environment info from the session directory.
- **Fix**: "Export reproducer" button in NodeStatus for successful tasks. Backend generates a `.zip`:
  ```
  reproducer/
    cut_model.xml
    cut_model.bin
    input_0.bin / input_0.npy
    main_output.bin / main_output.npy
    ref_output.bin / ref_output.npy
    info.json   (device, OV version, metrics, node name, session config)
  ```
  - **Format note**: NPU driver developers typically work with raw `.bin` files, not `.npy`. Options: (a) export as raw `.bin` with shape/dtype in `info.json`, (b) include a conversion script, (c) offer both formats. Decision deferred to implementation.
- **Files**: New endpoint `GET /api/tensors/{session_id}/{task_id}/export`, `frontend/src/lib/panels/NodeStatus.svelte` (export button)

### 15. Graph filtering / Factorio-style search
- **Problem**: For large models, the developer needs to filter inferred nodes by op type, accuracy range, status, etc.
- **Fix**: Extend the existing inferred nodes search into a Factorio-style filter engine. Keep Ctrl+F simple (node name/type substring). The inferred nodes search gets an extended view with combinatorial filter rules: op type, accuracy range (using thresholds), status, shape. Non-matching nodes dimmed on graph (spatial context preserved).
- **Priority**: Lower -- may not be needed if accuracy toggle + sorted queue cover the use case well enough.
- **Files**: Extend inferred nodes search component, `frontend/src/lib/stores/graph.svelte.ts` (filter state), `frontend/src/lib/graph/webgpu/WebGPURenderer.ts` (dimming)

### 16. Accuracy summary dashboard
- **Problem**: No single overview of accuracy across the whole model.
- **What**: Worst-N table, distribution histogram, topological waveform chart.
- **Priority**: Low -- accuracy toggle + sorted queue may cover this. Implement if there's still a gap after items 4-6 are done.
- **Files**: New `frontend/src/lib/panels/AccuracySummary.svelte`, new `backend/routers/accuracy.py`
