# Bench fixtures

## cached_graph.json

A frozen snapshot of a real OpenVINO graph after ELK layout, used as the
"realistic" dataset for the WebGPU bench harness.

- **Origin**: copied from `sessions/20260409_205644_model/graph_cache.json`
  (date in the path is project-time, not real-world time)
- **Nodes**: 496
- **Edges**: 600
- **Schema**: matches `GraphData` from `frontend/src/lib/stores/types.ts`
  (`nodes[]` with x/y/width/height/color/category, `edges[]` with
  source/target/waypoints)

This file is bundled into the perf entry point so the harness can run with
no backend dependency. If you regenerate it from a different model, update
the counts above so the next reader knows what they're benchmarking against.

For larger workloads, the bench harness also exposes a synthetic generator
in `../syntheticGraph.ts`.
