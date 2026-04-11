/**
 * Top-level orchestrator for the WebGPU bench harness.
 *
 * Loads a dataset (cached real fixture or synthetic), boots the renderer +
 * minimap into a hidden host element, walks the requested scenarios, and
 * produces a structured BenchResult that the perf.html page can format and
 * download. Output is hotspot-centric: per scenario, phases are sorted by
 * percentage of CPU frame time so a single run is enough to point at the
 * slowest part of the frame.
 */
import type { GraphData } from '../stores/types';
import {
  initRenderer,
  destroyRenderer,
  getCamera,
  getGPURenderer,
  attachMinimap,
} from '../graph/renderer';
import { FrameStats, type HotspotProfile, type PhaseAggregate } from './instrumentation';
import {
  SCENARIOS,
  DEFAULT_SCENARIO_ORDER,
  type BenchContext,
  type ScenarioResult,
} from './scenarios';
import { generateSyntheticGraph, SYNTHETIC_PRESETS } from './syntheticGraph';
import cachedGraphFixture from './fixtures/cached_graph.json';

export interface DatasetDescriptor {
  label: string;
  load: () => Promise<GraphData>;
}

export interface BenchConfig {
  dataset: DatasetDescriptor;
  scenarios?: string[];
  /** Set to true to disable the minimap before scenarios run (sanity check). */
  disableMinimap?: boolean;
  /** Logical canvas size for the offscreen host. Defaults to 1280×800. */
  hostWidth?: number;
  hostHeight?: number;
  log?: (msg: string) => void;
}

export interface ColdLoadTimings {
  modelBuildMs: number;
  gpuInitMs: number;
  firstAppearanceMs: number;
  firstRenderMs: number;
  totalMs: number;
}

export interface BenchResultEnv {
  userAgent: string;
  devicePixelRatio: number;
  viewport: { width: number; height: number };
  timestamp: string;
  adapterInfo?: Record<string, unknown>;
  supportsGpuTimestamps: boolean;
}

export interface BenchResult {
  env: BenchResultEnv;
  dataset: { label: string; nodeCount: number; edgeCount: number };
  coldLoad: ColdLoadTimings;
  scenarios: ScenarioResult[];
}

export const DATASETS: DatasetDescriptor[] = [
  {
    label: 'cached-real',
    load: async () => cachedGraphFixture as unknown as GraphData,
  },
  ...SYNTHETIC_PRESETS.map(p => ({
    label: p.label,
    load: async () => generateSyntheticGraph(p.opts),
  })),
];

export function getDataset(label: string): DatasetDescriptor | undefined {
  return DATASETS.find(d => d.label === label);
}

async function gatherEnv(supportsGpuTimestamps: boolean): Promise<BenchResultEnv> {
  let adapterInfo: Record<string, unknown> | undefined;
  try {
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        const info = adapter.info as unknown as Record<string, unknown>;
        adapterInfo = {
          vendor: info?.vendor ?? null,
          architecture: info?.architecture ?? null,
          device: info?.device ?? null,
          description: info?.description ?? null,
        };
      }
    }
  } catch {
    // Best-effort only
  }

  return {
    userAgent: navigator.userAgent,
    devicePixelRatio: window.devicePixelRatio || 1,
    viewport: { width: window.innerWidth, height: window.innerHeight },
    timestamp: new Date().toISOString(),
    adapterInfo,
    supportsGpuTimestamps,
  };
}

function makeHost(width: number, height: number): HTMLDivElement {
  const host = document.createElement('div');
  host.style.position = 'fixed';
  host.style.left = '0';
  host.style.top = '0';
  host.style.width = `${width}px`;
  host.style.height = `${height}px`;
  host.style.pointerEvents = 'none';
  host.style.opacity = '0.001';
  host.style.zIndex = '-1';
  document.body.appendChild(host);
  return host;
}

function makeMinimap(): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.style.position = 'fixed';
  canvas.style.left = '0';
  canvas.style.bottom = '0';
  canvas.style.width = '200px';
  canvas.style.height = '150px';
  canvas.style.pointerEvents = 'none';
  canvas.style.opacity = '0.001';
  canvas.style.zIndex = '-1';
  document.body.appendChild(canvas);
  return canvas;
}

export async function runBench(config: BenchConfig): Promise<BenchResult> {
  const log = config.log ?? ((msg: string) => console.log('[bench]', msg));
  const hostWidth = config.hostWidth ?? 1280;
  const hostHeight = config.hostHeight ?? 800;

  log(`Loading dataset: ${config.dataset.label}`);
  const t0 = performance.now();
  const graphData = await config.dataset.load();
  const tDataLoaded = performance.now();
  log(`  ${graphData.nodes.length} nodes, ${graphData.edges.length} edges (${(tDataLoaded - t0).toFixed(0)} ms)`);

  const host = makeHost(hostWidth, hostHeight);
  const minimapCanvas = makeMinimap();

  log('Initializing renderer...');
  const tInitStart = performance.now();
  await initRenderer(host, graphData);
  const tInitEnd = performance.now();

  const renderer = getGPURenderer();
  const panZoom = getCamera();
  if (!renderer || !panZoom) {
    document.body.removeChild(host);
    document.body.removeChild(minimapCanvas);
    throw new Error('[bench] renderer or panZoom is null after initRenderer');
  }

  const stats = new FrameStats();
  renderer.setFrameStats(stats);

  if (!config.disableMinimap) {
    log('Attaching minimap...');
    attachMinimap(minimapCanvas);
  }

  log('Warmup render...');
  const tWarmStart = performance.now();
  await renderer.forceRender();
  const tWarmEnd = performance.now();

  const env = await gatherEnv(renderer.supportsGpuTimestamps);

  const coldLoad: ColdLoadTimings = {
    modelBuildMs: 0,
    gpuInitMs: tInitEnd - tInitStart,
    firstAppearanceMs: 0,
    firstRenderMs: tWarmEnd - tWarmStart,
    totalMs: tWarmEnd - tInitStart,
  };

  const scenarioNames = config.scenarios ?? DEFAULT_SCENARIO_ORDER;
  const scenarioResults: ScenarioResult[] = [];

  const ctx: BenchContext = {
    renderer,
    panZoom,
    graphData,
    stats,
    log,
  };

  for (const name of scenarioNames) {
    const fn = SCENARIOS[name];
    if (!fn) {
      log(`Skipping unknown scenario: ${name}`);
      continue;
    }
    try {
      log('');
      log(formatScenarioHeader(name));
      const result = await fn(ctx);
      scenarioResults.push(result);
      log(formatHotspotTable(result.profile, '  '));
      if (result.notes) {
        log('  notes: ' + JSON.stringify(result.notes));
      }
    } catch (e) {
      log(`  ${name} failed: ${(e as Error).message}`);
    }
  }

  // Cleanup
  renderer.setFrameStats(null);
  destroyRenderer();
  if (host.parentNode) host.parentNode.removeChild(host);
  if (minimapCanvas.parentNode) minimapCanvas.parentNode.removeChild(minimapCanvas);

  return {
    env,
    dataset: {
      label: config.dataset.label,
      nodeCount: graphData.nodes.length,
      edgeCount: graphData.edges.length,
    },
    coldLoad,
    scenarios: scenarioResults,
  };
}

// ── Formatting ─────────────────────────────────────────────────────────────

export function formatScenarioHeader(name: string): string {
  return `── ${name} ──`;
}

/**
 * Format a HotspotProfile as a fixed-width table sorted by % of frame time.
 * Used by the bench page log and the summary panel.
 */
export function formatHotspotTable(profile: HotspotProfile, indent = ''): string {
  if (profile.frames === 0) {
    return `${indent}(no frames captured)`;
  }

  const lines: string[] = [];
  const cpuLine = `cpu mean=${profile.meanCpuMs.toFixed(2)} ms  p50=${profile.cpuP50.toFixed(2)}  p95=${profile.cpuP95.toFixed(2)}  max=${profile.cpuMax.toFixed(2)}  fps≈${profile.fps.toFixed(1)}`;
  lines.push(indent + cpuLine);

  if (profile.gpuP50 != null) {
    const gpuLine = `gpu mean=${profile.meanGpuMs?.toFixed(2)} ms  p50=${profile.gpuP50.toFixed(2)}  p95=${profile.gpuP95?.toFixed(2)}`;
    lines.push(indent + gpuLine);
  } else {
    lines.push(indent + 'gpu n/a (timestamp-query not supported)');
  }
  lines.push('');

  // Phase table
  if (profile.phases.length === 0) {
    lines.push(indent + '(no phases recorded — instrumentation off?)');
    return lines.join('\n');
  }

  const NAME_W = 28;
  const header = pad('phase', NAME_W) + rpad('% frame', 9) + rpad('mean ms', 10) + rpad('p50', 9) + rpad('p95', 9) + rpad('count', 7);
  lines.push(indent + header);
  lines.push(indent + '-'.repeat(header.length));

  for (const p of profile.phases) {
    const pct = (p.fractionOfFrame * 100).toFixed(1) + '%';
    const row = pad(p.name, NAME_W)
      + rpad(pct, 9)
      + rpad(p.meanMs.toFixed(2), 10)
      + rpad(p.p50.toFixed(2), 9)
      + rpad(p.p95.toFixed(2), 9)
      + rpad(String(p.count), 7);
    lines.push(indent + row);
  }

  // Unaccounted footer
  const unaccPct = (profile.unaccountedFraction * 100).toFixed(1) + '%';
  const unaccMean = (profile.unaccountedMs / profile.frames).toFixed(2);
  lines.push(indent + '-'.repeat(header.length));
  lines.push(indent + pad('unaccounted', NAME_W) + rpad(unaccPct, 9) + rpad(unaccMean, 10));

  return lines.join('\n');
}

export function summarize(result: BenchResult): string {
  const lines: string[] = [];
  lines.push(`Dataset: ${result.dataset.label} (${result.dataset.nodeCount} nodes, ${result.dataset.edgeCount} edges)`);
  lines.push(`Cold load: ${result.coldLoad.totalMs.toFixed(0)} ms total (init ${result.coldLoad.gpuInitMs.toFixed(0)} + first render ${result.coldLoad.firstRenderMs.toFixed(0)})`);
  lines.push(`GPU timestamps: ${result.env.supportsGpuTimestamps ? 'supported' : 'not available'}`);
  lines.push('');

  // Top hotspot per scenario at a glance
  lines.push('Top hotspot per scenario:');
  for (const s of result.scenarios) {
    const top = s.profile.phases[0];
    if (top) {
      lines.push(`  ${rpad(s.name, 20)} ${rpad((top.fractionOfFrame * 100).toFixed(1) + '%', 8)} ${top.name}  (mean ${top.meanMs.toFixed(2)} ms, cpu ${s.profile.meanCpuMs.toFixed(2)} ms)`);
    } else {
      lines.push(`  ${rpad(s.name, 20)} (no phases)`);
    }
  }
  lines.push('');

  for (const s of result.scenarios) {
    lines.push(formatScenarioHeader(s.name));
    lines.push(formatHotspotTable(s.profile, '  '));
    if (s.notes) lines.push('  notes: ' + JSON.stringify(s.notes));
    lines.push('');
  }
  return lines.join('\n');
}

// ── tiny string helpers ────────────────────────────────────────────────────

function pad(s: string, w: number): string {
  if (s.length >= w) return s;
  return s + ' '.repeat(w - s.length);
}

function rpad(s: string, w: number): string {
  if (s.length >= w) return s.padStart(w);
  return ' '.repeat(w - s.length) + s;
}

// Keep PhaseAggregate referenced in case downstream consumers want to import the type via runner.
export type { PhaseAggregate };
