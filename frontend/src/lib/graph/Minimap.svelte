<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { graphStore } from '../stores/graph.svelte';
  import {
    getCamera,
    getGPURenderer,
    onRendererReady,
    attachMinimap,
    detachMinimap,
    getMinimapTarget,
  } from './renderer';
  import type { MinimapTarget as MinimapTargetType } from './webgpu/MinimapTarget';

  let canvas: HTMLCanvasElement = $state()!;
  let collapsed = $state(localStorage.getItem('minimap-collapsed') === 'true');
  let target: MinimapTargetType | null = null;
  let unsubReady: (() => void) | null = null;

  const MIN_W = 150, MIN_H = 100, MAX_W = 600, MAX_H = 500;
  let mmWidth = $state(parseInt(localStorage.getItem('minimap-w') ?? '200') || 200);
  let mmHeight = $state(parseInt(localStorage.getItem('minimap-h') ?? '150') || 150);

  // Resize drag state
  let isResizing = false;
  let resizeStartX = 0;
  let resizeStartY = 0;
  let resizeStartW = 0;
  let resizeStartH = 0;

  function tryAttach() {
    if (!canvas || collapsed) return;
    const r = getGPURenderer();
    if (!r) return;
    if (target) return; // already attached
    target = attachMinimap(canvas);
  }

  function handleResizeDown(e: MouseEvent) {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    isResizing = true;
    resizeStartX = e.clientX;
    resizeStartY = e.clientY;
    resizeStartW = mmWidth;
    resizeStartH = mmHeight;
    window.addEventListener('mousemove', handleResizeMove);
    window.addEventListener('mouseup', handleResizeUp);
  }

  function handleResizeMove(e: MouseEvent) {
    if (!isResizing) return;
    // Dragging left/up increases size (anchored bottom-right)
    mmWidth = Math.max(MIN_W, Math.min(MAX_W, resizeStartW + (resizeStartX - e.clientX)));
    mmHeight = Math.max(MIN_H, Math.min(MAX_H, resizeStartH + (resizeStartY - e.clientY)));
    if (target) target.handleResize();
    const r = getGPURenderer();
    if (r) r.markDirty();
  }

  function handleResizeUp() {
    isResizing = false;
    localStorage.setItem('minimap-w', String(mmWidth));
    localStorage.setItem('minimap-h', String(mmHeight));
    window.removeEventListener('mousemove', handleResizeMove);
    window.removeEventListener('mouseup', handleResizeUp);
  }

  // ── Click-to-pan ──

  let isDragging = false;

  function minimapToGraph(e: MouseEvent): { x: number; y: number } | null {
    const t = target ?? getMinimapTarget();
    if (!t) return null;
    const rect = canvas.getBoundingClientRect();
    return t.minimapToGraph(e.clientX - rect.left, e.clientY - rect.top);
  }

  function moveCameraTo(graphX: number, graphY: number) {
    const panZoom = getCamera();
    const gpu = getGPURenderer();
    if (!panZoom || !gpu) return;
    const viewW = gpu.canvas.clientWidth || 800;
    const viewH = gpu.canvas.clientHeight || 600;
    const tx = viewW / 2 - graphX * panZoom.ratio;
    const ty = viewH / 2 - graphY * panZoom.ratio;
    panZoom.setState({ tx, ty, scale: panZoom.ratio });
  }

  function handleMouseDown(e: MouseEvent) {
    if (e.button !== 0) return;
    e.preventDefault();
    isDragging = true;
    const pt = minimapToGraph(e);
    if (pt) moveCameraTo(pt.x, pt.y);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }

  function handleMouseMove(e: MouseEvent) {
    if (!isDragging) return;
    const pt = minimapToGraph(e);
    if (pt) moveCameraTo(pt.x, pt.y);
  }

  function handleMouseUp() {
    isDragging = false;
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', handleMouseUp);
  }

  function handleWheel(e: WheelEvent) {
    e.preventDefault();
    e.stopPropagation();
    const panZoom = getCamera();
    const gpu = getGPURenderer();
    if (!panZoom || !gpu) return;

    const pt = minimapToGraph(e as unknown as MouseEvent);
    if (!pt) return;

    const factor = e.deltaY < 0 ? 1.25 : 1 / 1.25;
    const newScale = Math.max(0.01, Math.min(50, panZoom.ratio * factor));

    const viewW = gpu.canvas.clientWidth || 800;
    const viewH = gpu.canvas.clientHeight || 600;
    const tx = viewW / 2 - pt.x * newScale;
    const ty = viewH / 2 - pt.y * newScale;
    panZoom.setState({ tx, ty, scale: newScale });
  }

  onMount(() => {
    // Attach as soon as both the canvas and renderer are ready
    unsubReady = onRendererReady(() => tryAttach());

    // Re-attach if collapse toggles or graph data changes
    const stop = $effect.root(() => {
      $effect(() => {
        // Touch reactive deps so this re-runs
        const _v = graphStore.cameraVersion;
        void _v;
        if (graphStore.graphData && canvas && !collapsed) {
          tryAttach();
          // Ensure size matches new mmWidth/mmHeight
          if (target) target.handleResize();
          const r = getGPURenderer();
          if (r) r.markDirty();
        }
      });
    });

    return () => {
      stop();
    };
  });

  onDestroy(() => {
    if (unsubReady) { unsubReady(); unsubReady = null; }
    detachMinimap();
    target = null;
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', handleMouseUp);
    window.removeEventListener('mousemove', handleResizeMove);
    window.removeEventListener('mouseup', handleResizeUp);
  });

  // When collapsed flips on, detach so we don't waste GPU work
  $effect(() => {
    if (collapsed) {
      detachMinimap();
      target = null;
    } else if (canvas) {
      tryAttach();
    }
  });
</script>

<div class="mm-root">
  {#if !collapsed}
    <div class="mm-frame" style="width: {mmWidth}px; height: {mmHeight}px;">
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div class="mm-resize" onmousedown={handleResizeDown} aria-label="Resize minimap">
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
          <path d="M2 10L10 10L10 2" fill="none" stroke="currentColor" stroke-width="1.5"/>
          <path d="M5 10L10 10L10 5" fill="none" stroke="currentColor" stroke-width="1.5"/>
        </svg>
      </div>
      <canvas
        bind:this={canvas}
        class="mm-canvas"
        style="width: {mmWidth}px; height: {mmHeight}px;"
        onmousedown={handleMouseDown}
        onwheel={handleWheel}
      ></canvas>
    </div>
  {/if}
  <button
    class="mm-toggle"
    class:mm-toggle--collapsed={collapsed}
    onclick={() => { collapsed = !collapsed; localStorage.setItem('minimap-collapsed', String(collapsed)); }}
    title={collapsed ? 'Show minimap' : 'Hide minimap'}
  >
    <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true">
      <rect x="2" y="2" width="12" height="12" rx="1.5" />
      <path d="M6 6h4v4H6z" fill="currentColor" stroke="none" />
    </svg>
    {collapsed ? 'Map' : 'Hide map'}
  </button>
</div>

<style>
  .mm-root {
    position: absolute;
    bottom: 14px;
    right: 14px;
    z-index: 20;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
  }
  .mm-frame {
    position: relative;
    overflow: hidden;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md) var(--radius-md) 0 0;
    box-shadow: var(--shadow-elevated);
  }
  .mm-resize {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
    padding: 1px;
    border-radius: var(--radius-xs);
    color: var(--text-muted-soft);
    cursor: nw-resize;
    transform: rotate(180deg);
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: color var(--dur-fast) ease, background var(--dur-fast) ease;
  }
  .mm-resize:hover {
    color: var(--accent);
    background: var(--accent-bg-soft);
  }
  .mm-canvas {
    position: absolute;
    inset: 0;
    cursor: crosshair;
  }
  .mm-toggle {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 9px;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted-strong);
    background: var(--bg-panel);
    border: 1px solid var(--border-color);
    border-radius: 0 0 var(--radius-md) var(--radius-md);
    transition: color var(--dur-fast) ease, background var(--dur-fast) ease, border-color var(--dur-fast) ease;
  }
  .mm-toggle--collapsed {
    border-radius: var(--radius-md);
  }
  .mm-toggle:hover {
    color: var(--text-primary);
    background: var(--bg-menu);
    border-color: var(--accent-border);
  }
</style>
