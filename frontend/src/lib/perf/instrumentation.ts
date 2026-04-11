/**
 * Phase-based frame profiler for the WebGPU renderer.
 *
 * Each frame is divided into named phases (e.g. "appearance.nodeBuild",
 * "render.encode") that are timed with `performance.now()` markers. The
 * snapshot aggregates phases across the captured frame window into a
 * HotspotProfile sorted by percentage of total frame time, so a single bench
 * run is enough to point at the slowest part of the frame.
 *
 * Storage is preallocated typed arrays — zero allocation per frame, cheap
 * enough to leave on permanently in dev (PerfHud).
 */

const MAX_PHASES = 64;
const DEFAULT_CAPACITY = 240;

export interface PhaseAggregate {
  /** Phase name, e.g. "appearance.nodeBuild" */
  name: string;
  /** Number of frames in which this phase fired at least once */
  count: number;
  /** Sum of phase times across all firings (ms) */
  totalMs: number;
  /** Mean ms per firing (totalMs / count) */
  meanMs: number;
  p50: number;
  p95: number;
  min: number;
  max: number;
  /** Share of the window's total CPU frame time, in [0, 1] */
  fractionOfFrame: number;
}

export interface HotspotProfile {
  /** Number of frame samples in the window */
  frames: number;
  /** Wall-clock duration covered by the window (ms) */
  windowMs: number;
  /** Synthetic FPS = frames / windowMs */
  fps: number;
  /** Total CPU frame time across all frames (ms) */
  totalCpuMs: number;
  /** Mean CPU frame time per frame (ms) */
  meanCpuMs: number;
  cpuP50: number;
  cpuP95: number;
  cpuMin: number;
  cpuMax: number;
  /** Total GPU pass time across frames where timestamp-query data was available */
  totalGpuMs: number | null;
  meanGpuMs: number | null;
  gpuP50: number | null;
  gpuP95: number | null;
  /** Phases sorted by fractionOfFrame descending */
  phases: PhaseAggregate[];
  /** Sum of all phase totals (ms) — when this is well below totalCpuMs the rest is bookkeeping */
  totalAccountedMs: number;
  /** totalCpuMs - totalAccountedMs (ms) */
  unaccountedMs: number;
  /** unaccountedMs / totalCpuMs */
  unaccountedFraction: number;
}

export class FrameStats {
  readonly capacity: number;

  // ---- Phase registration ----
  private phaseNames: string[] = [];
  private phaseIds = new Map<string, number>();

  // ---- Per-frame ring buffer ----
  /** Per-frame, per-phase milliseconds. Indexed [frameSlot * MAX_PHASES + phaseId]. */
  private phaseData: Float64Array;
  /** Per-frame total CPU ms */
  private cpuTotal: Float64Array;
  /** Per-frame GPU pass ms (or 0 if invalid) */
  private gpuMs: Float64Array;
  /** Per-frame validity flag for gpuMs */
  private gpuValid: Uint8Array;
  /** Per-frame wall-clock start time (performance.now) */
  private frameStartTimes: Float64Array;

  private head = 0;
  private filled = 0;

  // ---- In-flight frame state ----
  private frameStart = 0;
  /** performance.now() at the start of each currently-open phase, indexed by phaseId */
  private phaseOpenAt: Float64Array;
  /** Accumulated ms for each phase in the current frame, indexed by phaseId */
  private phaseAccum: Float64Array;

  constructor(capacity = DEFAULT_CAPACITY) {
    this.capacity = capacity;
    this.phaseData = new Float64Array(capacity * MAX_PHASES);
    this.cpuTotal = new Float64Array(capacity);
    this.gpuMs = new Float64Array(capacity);
    this.gpuValid = new Uint8Array(capacity);
    this.frameStartTimes = new Float64Array(capacity);
    this.phaseOpenAt = new Float64Array(MAX_PHASES);
    this.phaseAccum = new Float64Array(MAX_PHASES);
  }

  // ---- Phase registration ----

  private idFor(name: string): number {
    let id = this.phaseIds.get(name);
    if (id !== undefined) return id;
    if (this.phaseNames.length >= MAX_PHASES) return -1;
    id = this.phaseNames.length;
    this.phaseNames.push(name);
    this.phaseIds.set(name, id);
    return id;
  }

  // ---- Frame lifecycle ----

  /**
   * Open a frame window. Idempotent — calling twice without an intervening
   * endFrame() is a no-op so the renderer can call beginFrame() at the top
   * of both updateAppearance() and render() without worrying about who got
   * there first. Phase markers fire freely between begin and end and roll
   * into the same frame slot.
   */
  beginFrame(): void {
    if (this.frameStart > 0) return;
    this.frameStart = performance.now();
  }

  /**
   * Mark the start of a named phase. Subsequent endPhase(name) closes it.
   * Phases must not nest with the same name; sibling/non-overlapping phases
   * are fine. Markers are accepted any time — if no frame is open, they
   * accumulate into the next frame to be flushed.
   */
  beginPhase(name: string): void {
    const id = this.idFor(name);
    if (id < 0) return;
    this.phaseOpenAt[id] = performance.now();
  }

  endPhase(name: string): void {
    const id = this.phaseIds.get(name);
    if (id === undefined) return;
    const start = this.phaseOpenAt[id];
    if (start === 0) return;
    this.phaseAccum[id] += performance.now() - start;
    this.phaseOpenAt[id] = 0;
  }

  /** Add a phase contribution directly (e.g. when timing was measured outside the frame). */
  markPhase(name: string, ms: number): void {
    if (ms <= 0) return;
    const id = this.idFor(name);
    if (id < 0) return;
    this.phaseAccum[id] += ms;
  }

  endFrame(gpuMs: number | null = null): void {
    if (this.frameStart === 0) return;
    // Report the larger of wall-clock and the sum of phase accumulators.
    // The sum is more accurate when phases ran across multiple JS tasks
    // (e.g. updateAppearance via RAF, then render() in another RAF).
    const wallClockMs = performance.now() - this.frameStart;
    let phaseSum = 0;
    for (let i = 0; i < this.phaseNames.length; i++) phaseSum += this.phaseAccum[i];
    const cpuMs = phaseSum > wallClockMs ? phaseSum : wallClockMs;

    const slot = this.head;
    this.frameStartTimes[slot] = this.frameStart;
    this.cpuTotal[slot] = cpuMs;
    this.gpuMs[slot] = gpuMs ?? 0;
    this.gpuValid[slot] = gpuMs == null ? 0 : 1;

    const baseOff = slot * MAX_PHASES;
    for (let i = 0; i < this.phaseNames.length; i++) {
      this.phaseData[baseOff + i] = this.phaseAccum[i];
    }
    for (let i = this.phaseNames.length; i < MAX_PHASES; i++) {
      this.phaseData[baseOff + i] = 0;
    }

    this.head = (slot + 1) % this.capacity;
    if (this.filled < this.capacity) this.filled++;

    // Clear accumulators for the next frame
    this.phaseAccum.fill(0);
    this.phaseOpenAt.fill(0);
    this.frameStart = 0;
  }

  // ---- Inspection ----

  reset(): void {
    this.head = 0;
    this.filled = 0;
    this.frameStart = 0;
    this.phaseAccum.fill(0);
    this.phaseOpenAt.fill(0);
  }

  get registeredPhases(): readonly string[] { return this.phaseNames; }

  /**
   * Aggregate the most-recent N frames (or all available, or those within
   * a wall-clock window) into a HotspotProfile.
   */
  snapshot(opts: { windowFrames?: number; windowMs?: number } = {}): HotspotProfile {
    if (this.filled === 0) return emptyProfile();

    const total = this.filled;
    const want = opts.windowFrames != null ? Math.min(total, opts.windowFrames) : total;
    const cutoff = opts.windowMs != null ? performance.now() - opts.windowMs : -Infinity;

    // Walk newest-to-oldest, collect frame indices in window
    const slots: number[] = [];
    let firstT = Infinity;
    let lastT = -Infinity;
    for (let i = 0; i < want; i++) {
      const slot = (this.head - 1 - i + this.capacity) % this.capacity;
      const t = this.frameStartTimes[slot];
      if (t < cutoff) break;
      slots.push(slot);
      if (t < firstT) firstT = t;
      if (t > lastT) lastT = t;
    }

    if (slots.length === 0) return emptyProfile();

    // CPU + GPU aggregates
    const cpuValues: number[] = new Array(slots.length);
    const gpuValues: number[] = [];
    let totalCpu = 0;
    for (let i = 0; i < slots.length; i++) {
      const c = this.cpuTotal[slots[i]];
      cpuValues[i] = c;
      totalCpu += c;
      if (this.gpuValid[slots[i]]) gpuValues.push(this.gpuMs[slots[i]]);
    }
    const cpuAgg = aggregate(cpuValues);

    let totalGpu: number | null = null;
    let gpuAgg: ReturnType<typeof aggregate> | null = null;
    if (gpuValues.length > 0) {
      totalGpu = gpuValues.reduce((s, v) => s + v, 0);
      gpuAgg = aggregate(gpuValues);
    }

    // Per-phase aggregates
    const phases: PhaseAggregate[] = [];
    let totalAccounted = 0;
    for (let phaseId = 0; phaseId < this.phaseNames.length; phaseId++) {
      const samples: number[] = [];
      let phaseTotal = 0;
      for (const slot of slots) {
        const v = this.phaseData[slot * MAX_PHASES + phaseId];
        if (v > 0) {
          samples.push(v);
          phaseTotal += v;
        }
      }
      if (phaseTotal === 0) continue;
      const a = aggregate(samples);
      phases.push({
        name: this.phaseNames[phaseId],
        count: samples.length,
        totalMs: phaseTotal,
        meanMs: phaseTotal / samples.length,
        p50: a.p50,
        p95: a.p95,
        min: a.min,
        max: a.max,
        fractionOfFrame: totalCpu > 0 ? phaseTotal / totalCpu : 0,
      });
      totalAccounted += phaseTotal;
    }
    phases.sort((a, b) => b.fractionOfFrame - a.fractionOfFrame);

    const windowMs = Math.max(0, lastT - firstT);
    const unaccounted = Math.max(0, totalCpu - totalAccounted);

    return {
      frames: slots.length,
      windowMs,
      fps: windowMs > 0 ? (slots.length / windowMs) * 1000 : 0,
      totalCpuMs: totalCpu,
      meanCpuMs: cpuAgg.mean,
      cpuP50: cpuAgg.p50,
      cpuP95: cpuAgg.p95,
      cpuMin: cpuAgg.min,
      cpuMax: cpuAgg.max,
      totalGpuMs: totalGpu,
      meanGpuMs: gpuAgg ? gpuAgg.mean : null,
      gpuP50: gpuAgg ? gpuAgg.p50 : null,
      gpuP95: gpuAgg ? gpuAgg.p95 : null,
      phases,
      totalAccountedMs: totalAccounted,
      unaccountedMs: unaccounted,
      unaccountedFraction: totalCpu > 0 ? unaccounted / totalCpu : 0,
    };
  }
}

function emptyProfile(): HotspotProfile {
  return {
    frames: 0,
    windowMs: 0,
    fps: 0,
    totalCpuMs: 0,
    meanCpuMs: 0,
    cpuP50: 0,
    cpuP95: 0,
    cpuMin: 0,
    cpuMax: 0,
    totalGpuMs: null,
    meanGpuMs: null,
    gpuP50: null,
    gpuP95: null,
    phases: [],
    totalAccountedMs: 0,
    unaccountedMs: 0,
    unaccountedFraction: 0,
  };
}

function aggregate(values: number[]): { p50: number; p95: number; min: number; max: number; mean: number } {
  if (values.length === 0) return { p50: 0, p95: 0, min: 0, max: 0, mean: 0 };
  const sorted = values.slice().sort((a, b) => a - b);
  const sum = sorted.reduce((s, v) => s + v, 0);
  return {
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean: sum / sorted.length,
  };
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const t = idx - lo;
  return sorted[lo] * (1 - t) + sorted[hi] * t;
}
