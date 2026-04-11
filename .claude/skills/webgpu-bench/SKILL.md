---
name: webgpu-bench
description: This skill should be used when the user asks to "run the WebGPU bench", "benchmark the renderer", "find the hotspot", "profile the graph rendering", "what's slow in the render path", "where is the frame time going", "measure frame phases", or mentions perf.html, FrameStats, PerfHud, the synthetic graph generator, or hotspot identification. Provides the procedural workflow for running the layers-insight WebGPU bench harness, reading per-phase hotspot tables, and turning a single bench run into an actionable optimization target.
version: 0.2.0
---

# WebGPU Renderer Bench Workflow

This skill encodes the procedural knowledge for measuring the layers-insight
WebGPU renderer's performance. The harness lives at `frontend/perf.html` and
covers both the main graph and the WebGPU minimap.

The output is **hotspot-centric**: every scenario emits a per-phase table
sorted by percentage of CPU frame time, so a single run is enough to point at
the slowest part of the frame. There is no before/after diff workflow — the
percentage column is the comparison.

## When to Run the Bench

Run the bench any time the user wants to:

- Find which part of the frame is dominating CPU time (the headline use case).
- Reproduce a "feels laggy" report on a midrange laptop and see where the time
  is going.
- Validate that an optimization actually moved the dominant phase down (re-run
  the same scenario; the fraction will shift).
- Measure the marginal CPU cost of the WebGPU minimap (`minimapCost` scenario).
- Sanity-check the renderer at large synthetic sizes (1k–50k nodes) without a
  real backend session.

The harness is fully self-contained — no backend, no real session data
required. Datasets are bundled (real cached graph fixture) or generated
(synthetic 1k–50k node DAGs).

## Where Things Live

| Purpose | Path |
| --- | --- |
| Bench page | `frontend/perf.html` |
| Bench entry | `frontend/src/perf-main.ts` |
| Runner + formatters | `frontend/src/lib/perf/runner.ts` |
| Scenarios | `frontend/src/lib/perf/scenarios.ts` |
| FrameStats / phases | `frontend/src/lib/perf/instrumentation.ts` |
| Synthetic generator | `frontend/src/lib/perf/syntheticGraph.ts` |
| Real fixture | `frontend/src/lib/perf/fixtures/cached_graph.json` |
| WebGPU renderer (instrumented) | `frontend/src/lib/graph/webgpu/WebGPURenderer.ts` |
| WebGPU minimap target | `frontend/src/lib/graph/webgpu/MinimapTarget.ts` |
| In-app HUD | `frontend/src/lib/perf/PerfHud.svelte` |

## How to Run the Bench

The bench needs a real browser to access WebGPU, so end-to-end automation is
impossible from a CLI agent. Follow this two-actor workflow.

### Step 1: Start the dev server in the background

```bash
cd /home/last1k/code/layers-insight/frontend && \
  source ~/.nvm/nvm.sh && nvm use 20 >/dev/null 2>&1 && \
  npm run dev
```

Use `run_in_background: true` so the agent keeps control. Wait for Vite to
print `Local:   http://localhost:5173/` before instructing the user.

### Step 2: Open the harness

Tell the user to open `http://localhost:5173/perf.html`. The page exposes:

- A **dataset** dropdown: `cached-real` (496 nodes / 600 edges, the bundled
  fixture) plus `synthetic-1k`, `-5k`, `-10k`, `-25k`, `-50k`.
- A **scenarios** checklist (all selected by default). Each scenario is
  documented below.
- A "**Disable minimap**" checkbox for sanity comparisons.
- A "**Run**" button. Per-scenario hotspot tables stream into the log as the
  run progresses.
- A "**Download bench-result.json**" link that appears when the run finishes.

### Step 3: Read the hotspot tables

Each scenario produces a block like this in the log:

```
── panSweep ──
  cpu mean=14.20 ms  p50=13.80  p95=22.40  max=28.10  fps≈70.4
  gpu mean=2.10 ms  p50=2.00  p95=3.40

  phase                       % frame   mean ms      p50      p95  count
  ----------------------------------------------------------------
  appearance.edgeColorRebuild   45.2%      6.42     6.30     8.10     60
  render.encode                 22.8%      3.24     3.20     4.00     60
  appearance.nodeBuild          14.5%      2.06     2.00     2.40     60
  appearance.textRebuild         8.1%      1.15     1.10     1.40     60
  appearance.nodeUpload          3.2%      0.45     0.45     0.55     60
  render.minimap                 2.1%      0.30     0.30     0.45     60
  render.submit                  1.8%      0.26     0.25     0.35     60
  appearance.ghostRebuild        0.4%      0.05     0.05     0.10     60
  ----------------------------------------------------------------
  unaccounted                    1.9%      0.27
```

Read it like this:

- **% frame** is the share of total CPU frame time consumed by that phase
  across the scenario. **This is the hotspot column.** If one phase dominates,
  optimize it first.
- **mean ms** is per-firing average. Use this when comparing two phases of
  similar % share.
- **p50 / p95** are per-firing percentiles. A phase with high p95 / low p50
  is bursty — investigate the spikes separately.
- **count** is how many frames the phase fired in. A phase that ate 35% of
  frame time but fired in only 3 of 60 frames is a different problem than one
  that fires every frame.
- **unaccounted** is `cpuTotal - sum(phases)`. Usually <5%; bookkeeping +
  WebGPU API overhead. If it grows large, add finer-grained markers.

### Step 4: Decide what to fix

Use this rule of thumb:

| Top phase | First thing to look at |
| --- | --- |
| `appearance.nodeBuild` | The per-node loop in `WebGPURenderer.ts:updateAppearance` (lines ~257-367). Mostly `Float32Array` writes + `hexToRgb` + override checks. |
| `appearance.edgeRebuild` | `rebuildEdges` (around `WebGPURenderer.ts:776`) and `buildEdgeGeometry` / `buildPathGeometry` in `edgesPipeline.ts`. B-spline tessellation is the heavy lift. |
| `appearance.edgeColorRebuild` | The "needsRebuild" branch in `updateAppearance` (lines ~385-409). Calls `buildEdgeGeometry` again with a per-edge color callback — same tessellation cost as a full rebuild. |
| `appearance.textRebuild` | `rebuildText` in `WebGPURenderer.ts` — glyph instance assembly is O(nodes × label length). |
| `appearance.ghostRebuild` | `updateGhosts` — usually negligible unless an edge is selected. |
| `render.encode` | Bind group setup + draw call dispatch. Look at draw call counts and buffer reallocation. |
| `render.minimap` | The minimap pass. Run with "Disable minimap" to confirm and quantify the delta. |
| `render.submit` | Almost always small (<1ms). If big, the GPU is back-pressuring. |

The PerfHud overlay shows the same hotspot ranking live during real
interaction (`?perfHud=1`). Use it to confirm the bench-derived ranking
matches what happens under organic input.

### Step 5: Save the JSON if archival matters

The downloaded JSON contains everything in the log plus environment metadata
(`adapterInfo`, `userAgent`, GPU timestamp support). Useful when:

- Posting numbers in an issue or PR description.
- Sharing a midrange-laptop run with someone on a beefy desktop.
- Looking up old runs without re-running the bench.

The schema is documented in `references/result-schema.md` and is small enough
to read with `jq` directly. There is intentionally no diff tool — comparison
is the percentage column inside a single run.

## Available Scenarios

These match the scenario keys in `frontend/src/lib/perf/scenarios.ts`. Each
exercises a real production code path, not a parallel reimplementation.

| Scenario | What it stresses | Frames |
| --- | --- | --- |
| `idle` | Pure draw cost on a static frame. Use for the floor — phases reflect minimum-work-per-frame. | 30 |
| `panSweep` | 60-step circular pan. Camera matrix uploads + redraw, no graph data changes. | 60 |
| `zoomSweep` | 60-step 0.2× → 5× → 0.2× zoom anchored at canvas center. Same as panSweep but exercises the zoom-dependent text fade path too. | 60 |
| `hoverSweep` | Hover-test + appearance rebuild path across 50 evenly spaced nodes. The frame-by-frame hover changes force `updateAppearance` every step. | 50 |
| `accuracyToggle` | Flips `graphStore.accuracyViewActive` 20 times. Exercises the expensive `rebuildEdges` path that re-tessellates every edge. | 20 |
| `selectionThrash` | Selects 100 different nodes in sequence. Exercises selection-driven appearance rebuilds and edge color rebuilds. | 100 |
| `minimapCost` | `panSweep` with the minimap suspended vs. active. Notes carry the deltaCpuP50 in ms and percent. | 120 |

When investigating a specific complaint, narrow the scenario set rather than
running everything: e.g. only `idle` + `panSweep` for "scrolling feels janky",
only `accuracyToggle` for "Alt-key flicker is slow", only `hoverSweep` for
"hover lag".

## Phase Reference

The instrumented phases inside the renderer:

| Phase | Lives in | Notes |
| --- | --- | --- |
| `appearance.total` | `updateAppearance` body | Wraps the entire CPU rebuild call. Use to compare against `render.total`. |
| `appearance.edgeRebuild` | `rebuildEdges()` calls | Fires only when accuracy view toggles or new metrics arrive. |
| `appearance.nodeBuild` | The per-node loop building `Float32Array` instance data | Always fires. |
| `appearance.nodeUpload` | `updateNodeInstances` | The `device.queue.writeBuffer` call. Usually tiny. |
| `appearance.edgeColorRebuild` | The "needsRebuild" branch for hover/select/grayed | Re-tessellates the entire edge list. Often a hidden hotspot. |
| `appearance.textRebuild` | `rebuildText()` | Per-glyph instance assembly. |
| `appearance.ghostRebuild` | `updateGhosts()` | Off-screen edge endpoint indicators. |
| `render.total` | `render()` body | Wraps encoding + minimap pass + submit. |
| `render.encode` | beginRenderPass → pass.end | Pure WebGPU command encoding. |
| `render.minimap` | `MinimapTarget.draw(encoder)` | Only fires when minimap attached. |
| `render.submit` | `device.queue.submit` | Should be tiny. |

Phases are tracked by `FrameStats` in
`frontend/src/lib/perf/instrumentation.ts` using preallocated `Float64Array`
ring buffers — zero allocation per frame, ~50ns of overhead per marker.

## Live Spot-Checks (PerfHud)

For ad-hoc measurements during real interaction (not scripted scenarios),
append `?perfHud=1` to any app URL. The overlay shows rolling FPS, CPU/GPU
p50/p95, and the **top 5 phases by % of frame time**, refreshed at 4 Hz.
Use this to confirm a fix improved the dominant phase under organic input.

## Adding a New Scenario

When the user asks to add a new scenario:

1. Add the function to `frontend/src/lib/perf/scenarios.ts` following the
   existing `ScenarioFn` shape (`async (ctx) => ScenarioResult`).
2. Register it in the `SCENARIOS` map and append the name to
   `DEFAULT_SCENARIO_ORDER` (or leave off the default order if it's
   opt-in only).
3. The bench page picks up new scenarios automatically — no UI changes
   needed.
4. Drive the renderer through real APIs (`panZoom.setState`, `graphStore`
   mutations, `setHoveredNode`) so the scenario measures production code
   paths, not parallel implementations.
5. Use `await ctx.renderer.forceRender()` to render-and-wait between steps.
6. Call `ctx.stats.reset()` at the start so the scenario's snapshot doesn't
   include warmup frames from the previous one.

## Adding a New Phase

When the user asks to instrument a finer-grained subphase:

1. Pick a dotted name following the existing convention
   (`appearance.subphase` or `render.subphase`).
2. Add `fs?.beginPhase('appearance.foo')` / `fs?.endPhase('appearance.foo')`
   around the code in `WebGPURenderer.ts`. `fs` is the local copy of
   `this.frameStats`.
3. Phases are auto-registered on first use (up to `MAX_PHASES = 64` in
   `instrumentation.ts`), no enum to update.
4. Avoid overlapping (nested) phases with the same name — sibling phases are
   fine. The `.total` parent phases (`appearance.total`, `render.total`)
   intentionally span multiple sub-phases.

## Cautions

- The bench tears down and rebuilds the renderer between datasets. Don't
  expect to "switch dataset mid-run".
- The WebGPU minimap shares node/edge buffers with the main renderer; if
  measuring main-graph cost in isolation, use the "Disable minimap" checkbox
  or the `minimapCost` scenario.
- GPU timestamps land 1–2 frames after the actual frame because `mapAsync`
  is asynchronous. The first sample after `stats.reset()` may show
  `gpuP50: null` even when the feature is supported. CPU phase times are
  always accurate.
- Scenarios run sequentially with explicit `forceRender()` between steps —
  do not run two scenarios in parallel; they share the same renderer.
- Don't read absolute CPU times across machines for thresholds. The
  **percentages** are portable; the **absolute milliseconds** are not.
- When optimizing, look for phases >20% of frame time first. Phases <5%
  are noise unless they fire bursty (high p95 / low p50).

## Additional Resources

- **`references/result-schema.md`** — full TypeScript-derived schema of
  BenchResult, ScenarioResult, HotspotProfile, and PhaseAggregate for parsing
  the downloaded JSON.
