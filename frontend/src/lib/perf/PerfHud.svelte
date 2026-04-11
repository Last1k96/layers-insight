<script lang="ts">
  /**
   * Floating performance overlay. Mounts only when ?perfHud=1 is in the URL.
   * Subscribes to the WebGPU renderer's FrameStats at 4 Hz and shows rolling
   * FPS, frame time, GPU time, and the top 5 phases by % of frame time so the
   * slowest part of the frame is always visible.
   */
  import { onMount, onDestroy } from 'svelte';
  import { onRendererReady } from '../graph/renderer';
  import { FrameStats, type HotspotProfile } from './instrumentation';
  import type { WebGPURenderer } from '../graph/webgpu/WebGPURenderer';

  const enabled = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('perfHud');

  let renderer: WebGPURenderer | null = null;
  let stats: FrameStats | null = null;
  let profile = $state<HotspotProfile | null>(null);
  let pollHandle: number | null = null;
  let unsubscribeReady: (() => void) | null = null;

  const TOP_N = 5;

  onMount(() => {
    if (!enabled) return;
    unsubscribeReady = onRendererReady((r) => {
      renderer = r;
      stats = new FrameStats(240);
      r.setFrameStats(stats);
    });
    pollHandle = window.setInterval(() => {
      if (stats) profile = stats.snapshot({ windowMs: 1000 });
    }, 250);
  });

  onDestroy(() => {
    if (pollHandle !== null) clearInterval(pollHandle);
    if (renderer && stats) renderer.setFrameStats(null);
    if (unsubscribeReady) unsubscribeReady();
  });

  function fmt(n: number | null | undefined, digits = 2): string {
    if (n == null || !Number.isFinite(n)) return '–';
    return n.toFixed(digits);
  }
</script>

{#if enabled}
  <div class="perf-hud">
    <div class="title">PERF HOTSPOTS</div>
    {#if !profile || profile.frames === 0}
      <div class="dim small">idle</div>
    {:else}
      <div class="row"><span>fps</span><b>{fmt(profile.fps, 1)}</b></div>
      <div class="row"><span>cpu p50</span><b>{fmt(profile.cpuP50)} ms</b></div>
      <div class="row"><span>cpu p95</span><b>{fmt(profile.cpuP95)} ms</b></div>
      {#if profile.gpuP50 != null}
        <div class="row"><span>gpu p50</span><b>{fmt(profile.gpuP50)} ms</b></div>
      {:else}
        <div class="row dim"><span>gpu</span><b>n/a</b></div>
      {/if}

      <div class="divider"></div>
      <div class="phases-header"><span>phase</span><span>%</span><span>mean</span></div>
      {#if profile.phases.length === 0}
        <div class="small dim">no phases recorded</div>
      {:else}
        {#each profile.phases.slice(0, TOP_N) as p (p.name)}
          <div class="phase-row">
            <span class="phase-name">{p.name}</span>
            <span class="phase-pct">{(p.fractionOfFrame * 100).toFixed(1)}%</span>
            <span class="phase-ms">{fmt(p.meanMs)}</span>
          </div>
        {/each}
        {#if profile.unaccountedFraction > 0.01}
          <div class="phase-row dim">
            <span class="phase-name">(unaccounted)</span>
            <span class="phase-pct">{(profile.unaccountedFraction * 100).toFixed(1)}%</span>
            <span class="phase-ms">–</span>
          </div>
        {/if}
      {/if}
    {/if}
  </div>
{/if}

<style>
  .perf-hud {
    position: fixed;
    top: 12px;
    right: 12px;
    z-index: 9999;
    background: rgba(15, 17, 25, 0.92);
    border: 1px solid #3A3F56;
    border-radius: 6px;
    padding: 8px 10px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
    color: #e5e7eb;
    min-width: 240px;
    pointer-events: none;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  }
  .row {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    line-height: 1.45;
  }
  .title {
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 0.08em;
    color: #8E94B0;
    margin-bottom: 4px;
  }
  .dim { opacity: 0.65; }
  .small { font-size: 10px; }
  b { font-weight: 500; }
  .divider {
    height: 1px;
    background: #3A3F56;
    margin: 6px 0 4px 0;
  }
  .phases-header {
    display: grid;
    grid-template-columns: 1fr 50px 50px;
    font-size: 10px;
    color: #8E94B0;
    margin-bottom: 2px;
  }
  .phases-header span:nth-child(2),
  .phases-header span:nth-child(3) {
    text-align: right;
  }
  .phase-row {
    display: grid;
    grid-template-columns: 1fr 50px 50px;
    line-height: 1.4;
  }
  .phase-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .phase-pct, .phase-ms {
    text-align: right;
  }
</style>
