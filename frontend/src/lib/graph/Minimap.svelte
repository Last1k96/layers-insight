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

<div class="absolute bottom-4 right-4 z-20 flex flex-col items-end">
  {#if !collapsed}
    <div
      class="relative border border-[--border-color] rounded-t bg-[--bg-primary] overflow-hidden shadow-lg"
      style="width: {mmWidth}px; height: {mmHeight}px;"
    >
      <!-- Resize handle (top-left corner) -->
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div
        class="absolute top-0 left-0 w-3 h-3 z-10 cursor-nw-resize"
        onmousedown={handleResizeDown}
      >
        <svg class="w-3 h-3 text-muted-soft rotate-180" viewBox="0 0 12 12">
          <path d="M2 10L10 10L10 2" fill="none" stroke="currentColor" stroke-width="1.5"/>
          <path d="M5 10L10 10L10 5" fill="none" stroke="currentColor" stroke-width="1.5"/>
        </svg>
      </div>
      <canvas
        bind:this={canvas}
        class="absolute inset-0 cursor-crosshair"
        style="width: {mmWidth}px; height: {mmHeight}px;"
        onmousedown={handleMouseDown}
        onwheel={handleWheel}
      ></canvas>
    </div>
  {/if}
  <button
    class="px-2 py-0.5 text-xs text-muted hover:text-content-primary bg-[--bg-panel] border border-[--border-color] {collapsed ? 'rounded' : 'rounded-b border-t-0'} hover:bg-[--bg-menu] transition-colors"
    onclick={() => { collapsed = !collapsed; localStorage.setItem('minimap-collapsed', String(collapsed)); }}
  >
    {collapsed ? 'Map' : 'Hide'}
  </button>
</div>
