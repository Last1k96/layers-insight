/**
 * Scripted scenarios for the WebGPU bench harness.
 *
 * Each scenario drives the live renderer through a deterministic sequence of
 * camera moves / store mutations / hover events, then returns a HotspotProfile
 * pulled from the renderer's FrameStats. The runner formats the per-scenario
 * hotspots as a table sorted by % of frame time.
 *
 * Scenarios talk to the renderer through the same APIs the live app uses
 * (PanZoom.setState, graphStore mutations, hitGrid hover) so we benchmark the
 * production hot paths, not parallel reimplementations.
 */
import type { WebGPURenderer } from '../graph/webgpu/WebGPURenderer';
import type { PanZoom } from '../graph/panZoom';
import type { GraphData } from '../stores/types';
import { graphStore } from '../stores/graph.svelte';
import { setHoveredNode, refreshRenderer, getMinimapTarget } from '../graph/renderer';
import type { FrameStats, HotspotProfile } from './instrumentation';

export interface BenchContext {
  renderer: WebGPURenderer;
  panZoom: PanZoom;
  graphData: GraphData;
  stats: FrameStats;
  log: (msg: string) => void;
}

export interface ScenarioResult {
  name: string;
  durationMs: number;
  frames: number;
  profile: HotspotProfile;
  /** Free-form notes a scenario can attach (e.g. minimap-on-vs-off deltas). */
  notes?: Record<string, unknown>;
}

export type ScenarioFn = (ctx: BenchContext) => Promise<ScenarioResult>;

// ---- Helpers ----

async function nextFrame(): Promise<void> {
  return new Promise(r => requestAnimationFrame(() => r()));
}

async function renderAndWait(renderer: WebGPURenderer): Promise<void> {
  await renderer.forceRender();
}

// ---- Scenarios ----

export const idleScenario: ScenarioFn = async (ctx) => {
  ctx.log('idle: forcing 30 redraws of a static frame');
  ctx.stats.reset();
  const FRAMES = 30;
  const t0 = performance.now();
  for (let i = 0; i < FRAMES; i++) {
    ctx.renderer.markDirty();
    await renderAndWait(ctx.renderer);
    await nextFrame();
  }
  const dur = performance.now() - t0;
  return {
    name: 'idle',
    durationMs: dur,
    frames: FRAMES,
    profile: ctx.stats.snapshot(),
  };
};

export const panSweepScenario: ScenarioFn = async (ctx) => {
  ctx.log('panSweep: 60 scripted pan steps over a fixed circular path');
  ctx.stats.reset();
  const STEPS = 60;
  const radius = 400;
  const cx = ctx.panZoom.translateX;
  const cy = ctx.panZoom.translateY;
  const scale = ctx.panZoom.ratio;

  const t0 = performance.now();
  for (let i = 0; i < STEPS; i++) {
    const a = (i / STEPS) * Math.PI * 2;
    const tx = cx + Math.cos(a) * radius;
    const ty = cy + Math.sin(a) * radius;
    ctx.panZoom.setState({ tx, ty, scale });
    await renderAndWait(ctx.renderer);
  }
  const dur = performance.now() - t0;

  // Restore original camera
  ctx.panZoom.setState({ tx: cx, ty: cy, scale });
  await renderAndWait(ctx.renderer);

  return {
    name: 'panSweep',
    durationMs: dur,
    frames: STEPS,
    profile: ctx.stats.snapshot({ windowFrames: STEPS }),
  };
};

export const zoomSweepScenario: ScenarioFn = async (ctx) => {
  ctx.log('zoomSweep: 60 scripted zoom steps from 0.2× → 5× → 0.2×');
  ctx.stats.reset();
  const STEPS = 60;
  const startScale = ctx.panZoom.ratio;
  const startTx = ctx.panZoom.translateX;
  const startTy = ctx.panZoom.translateY;

  // Zoom anchored at canvas center
  const w = ctx.renderer.canvas.clientWidth || 800;
  const h = ctx.renderer.canvas.clientHeight || 600;
  const cx = w / 2;
  const cy = h / 2;

  const t0 = performance.now();
  for (let i = 0; i < STEPS; i++) {
    const t = i / (STEPS - 1);
    // Triangle wave: 0 → 1 → 0 across the sweep
    const tri = t < 0.5 ? t * 2 : (1 - t) * 2;
    const minS = 0.2, maxS = 5.0;
    const newScale = minS * Math.pow(maxS / minS, tri);

    const tx = cx - (cx - startTx) * (newScale / startScale);
    const ty = cy - (cy - startTy) * (newScale / startScale);
    ctx.panZoom.setState({ tx, ty, scale: newScale });
    await renderAndWait(ctx.renderer);
  }
  const dur = performance.now() - t0;

  ctx.panZoom.setState({ tx: startTx, ty: startTy, scale: startScale });
  await renderAndWait(ctx.renderer);

  return {
    name: 'zoomSweep',
    durationMs: dur,
    frames: STEPS,
    profile: ctx.stats.snapshot({ windowFrames: STEPS }),
  };
};

export const hoverSweepScenario: ScenarioFn = async (ctx) => {
  ctx.log('hoverSweep: simulate hover over 50 nodes');
  ctx.stats.reset();
  const nodes = ctx.graphData.nodes;
  const SAMPLES = Math.min(50, nodes.length);
  const stride = Math.max(1, Math.floor(nodes.length / SAMPLES));
  const picked: typeof nodes = [];
  for (let i = 0; i < nodes.length && picked.length < SAMPLES; i += stride) {
    picked.push(nodes[i]);
  }

  const t0 = performance.now();
  for (const n of picked) {
    const cx = n.x + n.width / 2;
    const cy = n.y + n.height / 2;
    const id = ctx.renderer.hitGrid.query(cx, cy);
    setHoveredNode(id);
    refreshRenderer();
    await renderAndWait(ctx.renderer);
  }
  setHoveredNode(null);
  refreshRenderer();
  await renderAndWait(ctx.renderer);
  const dur = performance.now() - t0;

  return {
    name: 'hoverSweep',
    durationMs: dur,
    frames: SAMPLES,
    profile: ctx.stats.snapshot({ windowFrames: SAMPLES }),
  };
};

export const accuracyToggleScenario: ScenarioFn = async (ctx) => {
  ctx.log('accuracyToggle: flip accuracyViewActive 20 times (rebuilds edges)');
  ctx.stats.reset();
  const TOGGLES = 20;
  const original = graphStore.accuracyViewActive;

  const t0 = performance.now();
  for (let i = 0; i < TOGGLES; i++) {
    graphStore.accuracyViewActive = i % 2 === 0;
    refreshRenderer();
    await renderAndWait(ctx.renderer);
  }
  const dur = performance.now() - t0;
  graphStore.accuracyViewActive = original;
  refreshRenderer();
  await renderAndWait(ctx.renderer);

  return {
    name: 'accuracyToggle',
    durationMs: dur,
    frames: TOGGLES,
    profile: ctx.stats.snapshot({ windowFrames: TOGGLES }),
  };
};

export const selectionThrashScenario: ScenarioFn = async (ctx) => {
  ctx.log('selectionThrash: select 100 nodes in sequence');
  ctx.stats.reset();
  const nodes = ctx.graphData.nodes;
  const SAMPLES = Math.min(100, nodes.length);
  const stride = Math.max(1, Math.floor(nodes.length / SAMPLES));

  const t0 = performance.now();
  for (let i = 0; i < SAMPLES; i++) {
    const node = nodes[(i * stride) % nodes.length];
    graphStore.selectedNodeId = node.id;
    refreshRenderer();
    await renderAndWait(ctx.renderer);
  }
  const dur = performance.now() - t0;
  graphStore.selectedNodeId = null;
  refreshRenderer();
  await renderAndWait(ctx.renderer);

  return {
    name: 'selectionThrash',
    durationMs: dur,
    frames: SAMPLES,
    profile: ctx.stats.snapshot({ windowFrames: SAMPLES }),
  };
};

export const minimapCostScenario: ScenarioFn = async (ctx) => {
  ctx.log('minimapCost: pan sweep with minimap suspended vs active');
  const target = getMinimapTarget();
  if (!target) {
    ctx.log('  (no minimap attached — running active-only)');
    const r = await panSweepScenario(ctx);
    return { ...r, name: 'minimapCost', notes: { skipped: 'no minimap target' } };
  }

  // ── With minimap ──
  target.setSuspended(false);
  ctx.stats.reset();
  const withResult = await panSweepScenario(ctx);

  // ── Without minimap ──
  target.setSuspended(true);
  ctx.stats.reset();
  const withoutResult = await panSweepScenario(ctx);
  target.setSuspended(false);

  const withCpu = withResult.profile.cpuP50;
  const withoutCpu = withoutResult.profile.cpuP50;

  return {
    name: 'minimapCost',
    durationMs: withResult.durationMs + withoutResult.durationMs,
    frames: withResult.frames + withoutResult.frames,
    profile: withResult.profile,
    notes: {
      withMinimap: {
        cpuP50: withCpu,
        cpuP95: withResult.profile.cpuP95,
        gpuP50: withResult.profile.gpuP50,
      },
      withoutMinimap: {
        cpuP50: withoutCpu,
        cpuP95: withoutResult.profile.cpuP95,
        gpuP50: withoutResult.profile.gpuP50,
      },
      deltaCpuP50Ms: withCpu - withoutCpu,
      deltaCpuP50Pct: withoutCpu > 0 ? ((withCpu - withoutCpu) / withoutCpu) * 100 : 0,
    },
  };
};

export const SCENARIOS: Record<string, ScenarioFn> = {
  idle: idleScenario,
  panSweep: panSweepScenario,
  zoomSweep: zoomSweepScenario,
  hoverSweep: hoverSweepScenario,
  accuracyToggle: accuracyToggleScenario,
  selectionThrash: selectionThrashScenario,
  minimapCost: minimapCostScenario,
};

export const DEFAULT_SCENARIO_ORDER: string[] = [
  'idle',
  'panSweep',
  'zoomSweep',
  'hoverSweep',
  'selectionThrash',
  'accuracyToggle',
  'minimapCost',
];
