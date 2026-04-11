# BenchResult JSON Schema

The bench harness emits one `BenchResult` object per run. This document is the
canonical schema, derived from `frontend/src/lib/perf/runner.ts` and
`frontend/src/lib/perf/instrumentation.ts`. Use it when parsing bench output
without needing to read the TypeScript source.

All numbers are in milliseconds unless noted. All `*Ms` fields are
`performance.now()`-based wall-clock unless they end in `gpu*Ms`, which is
derived from WebGPU timestamp queries (originally nanoseconds, converted to
ms).

The output is **hotspot-centric**: each scenario contains a sorted list of
`PhaseAggregate` entries with a `fractionOfFrame` value. Read that column
first; everything else is supporting context.

## Top-level: BenchResult

```jsonc
{
  "env": {
    "userAgent": "Mozilla/5.0 ...",
    "devicePixelRatio": 1.5,
    "viewport": { "width": 1920, "height": 1080 },
    "timestamp": "2026-04-11T19:30:00.000Z",
    "adapterInfo": {
      "vendor": "intel" | "nvidia" | "amd" | "apple" | "...",
      "architecture": "...",
      "device": "...",
      "description": "..."
    },
    "supportsGpuTimestamps": true
  },
  "dataset": {
    "label": "cached-real" | "synthetic-1k" | "synthetic-5k" | ...,
    "nodeCount": 496,
    "edgeCount": 600
  },
  "coldLoad": {
    "modelBuildMs": 0,
    "gpuInitMs": 820.4,
    "firstAppearanceMs": 0,
    "firstRenderMs": 14.2,
    "totalMs": 834.6
  },
  "scenarios": [ /* ScenarioResult[] */ ]
}
```

### Notes on `coldLoad`

- `gpuInitMs` covers the entire `initRenderer()` call (graph model build,
  WebGPU device acquisition, pipeline creation, fitToView, first appearance).
- `firstRenderMs` is just the warmup `forceRender()` call done after init,
  before any scenarios run.
- `modelBuildMs` and `firstAppearanceMs` are reserved for future fine-grained
  splits — currently always 0.

### Notes on `env.adapterInfo`

The fields come from the WebGPU `GPUAdapterInfo` interface, which is
implementation-defined. Some browsers redact `device` and `description`
behind a flag for fingerprinting protection. Do not rely on any field
being non-empty.

## ScenarioResult

```jsonc
{
  "name": "panSweep",
  "durationMs": 1240.5,
  "frames": 60,
  "profile": { /* HotspotProfile */ },
  "notes": { /* free-form, scenario-specific */ }
}
```

- `durationMs` is the full wall-clock time spent inside the scenario function
  (including any setup and cleanup, not just render time).
- `frames` is the number of forced renders the scenario performed.
- `profile` is the `FrameStats.snapshot()` result over the scenario's frame
  window — see HotspotProfile below.
- `notes` is set only by scenarios that emit auxiliary data:
  - `minimapCost` sets `notes.withMinimap`, `notes.withoutMinimap`,
    `notes.deltaCpuP50Ms`, and `notes.deltaCpuP50Pct` so the minimap's
    marginal cost is directly readable without re-running the scenario.

## HotspotProfile

```jsonc
{
  "frames": 60,
  "windowMs": 1180.3,
  "fps": 50.8,

  "totalCpuMs": 852.0,
  "meanCpuMs": 14.2,
  "cpuP50": 13.8,
  "cpuP95": 22.4,
  "cpuMin": 11.0,
  "cpuMax": 28.1,

  "totalGpuMs": 126.0,
  "meanGpuMs": 2.1,
  "gpuP50": 2.0,
  "gpuP95": 3.4,

  "phases": [ /* PhaseAggregate[] sorted by fractionOfFrame desc */ ],

  "totalAccountedMs": 836.0,
  "unaccountedMs": 16.0,
  "unaccountedFraction": 0.019
}
```

| Field | Meaning |
| --- | --- |
| `frames` | Number of frame samples in the window |
| `windowMs` | Wall-clock time spanned by those samples |
| `fps` | `frames / windowMs * 1000`. Synthetic since frames are forced; relative metric only |
| `totalCpuMs` | Sum of per-frame CPU times across all frames in the window |
| `meanCpuMs` | Per-frame mean CPU time |
| `cpuP50/P95/Min/Max` | Distribution of per-frame CPU times |
| `totalGpuMs` | Sum of GPU pass times for frames where `timestamp-query` data was available; `null` if none |
| `meanGpuMs/gpuP50/P95` | Distribution over frames with valid GPU samples |
| `phases` | Per-phase aggregates sorted by `fractionOfFrame` descending |
| `totalAccountedMs` | Sum of all phases' `totalMs` — the part of `totalCpuMs` covered by markers |
| `unaccountedMs` | `totalCpuMs - totalAccountedMs`. The bookkeeping/overhead remainder |
| `unaccountedFraction` | `unaccountedMs / totalCpuMs` in [0, 1]. Should usually be <0.05 |

### GPU sample availability

`gpuP50` (and friends) are `null` when:

- The adapter does not support the `timestamp-query` feature.
- The window is too short for any GPU readback to have completed
  (typically the first 1–2 frames after `stats.reset()`).

When some frames in the window have GPU data and others don't, only the
frames with valid GPU samples contribute to `gpuP50/P95`. CPU phase times are
always accurate regardless of GPU support.

## PhaseAggregate

```jsonc
{
  "name": "appearance.edgeColorRebuild",
  "count": 60,
  "totalMs": 385.2,
  "meanMs": 6.42,
  "p50": 6.30,
  "p95": 8.10,
  "min": 5.80,
  "max": 9.20,
  "fractionOfFrame": 0.452
}
```

| Field | Meaning |
| --- | --- |
| `name` | Dotted phase name (`<group>.<phase>`) |
| `count` | Number of frames in which this phase fired at least once |
| `totalMs` | Sum of phase times across all firings |
| `meanMs` | `totalMs / count` |
| `p50/p95/min/max` | Distribution of per-firing times (only across the frames where it fired) |
| `fractionOfFrame` | `totalMs / HotspotProfile.totalCpuMs`, in [0, 1] |

### Choosing what to optimize

Sort by `fractionOfFrame` descending and look at the top 1–3 entries. Use
this guide:

- **Top phase >40% of frame** — clear win. Optimize this first.
- **Top phase 20–40%** — worth optimizing, but expect diminishing returns.
- **Top phase <20%** — frame time is spread out. Look at `unaccountedFraction`
  and consider adding finer-grained markers before optimizing anything.
- **Phase with `count < frames` but high `meanMs`** — bursty. Investigate
  whether the spike pattern matches what users complain about.
- **`fractionOfFrame` differs wildly between two scenarios for the same
  phase** — that phase is sensitive to whatever the scenario stresses (camera
  changes vs. data changes vs. hover). Pick the scenario closest to the user
  complaint and optimize against that.

## Reading with jq

Quick recipes for parsing the JSON without writing code:

```bash
# Top hotspot per scenario
jq '.scenarios[] | { name, top: .profile.phases[0] }' bench-result.json

# All phases above 10% of frame for the panSweep scenario
jq '.scenarios[] | select(.name == "panSweep")
   | .profile.phases | map(select(.fractionOfFrame > 0.1))' bench-result.json

# Cold-load timings
jq '.coldLoad' bench-result.json

# Minimap marginal cost
jq '.scenarios[] | select(.name == "minimapCost") | .notes' bench-result.json
```
