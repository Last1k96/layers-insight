<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { getCamera, getGPURenderer, getNodeSize } from './renderer';

  let canvas: HTMLCanvasElement = $state()!;
  let collapsed = $state(localStorage.getItem('minimap-collapsed') === 'true');
  let cleanupFn: (() => void) | null = null;

  const MIN_W = 150, MIN_H = 100, MAX_W = 600, MAX_H = 500;
  let mmWidth = $state(parseInt(localStorage.getItem('minimap-w') ?? '200') || 200);
  let mmHeight = $state(parseInt(localStorage.getItem('minimap-h') ?? '150') || 150);

  // Resize drag state
  let isResizing = false;
  let resizeStartX = 0;
  let resizeStartY = 0;
  let resizeStartW = 0;
  let resizeStartH = 0;

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
    // Update canvas buffer to match new size
    if (canvas) {
      canvas.width = mmWidth;
      canvas.height = mmHeight;
    }
    scheduleDraw();
  }

  function handleResizeUp() {
    isResizing = false;
    localStorage.setItem('minimap-w', String(mmWidth));
    localStorage.setItem('minimap-h', String(mmHeight));
    window.removeEventListener('mousemove', handleResizeMove);
    window.removeEventListener('mouseup', handleResizeUp);
  }

  // Cached bounds/scale from last draw — reused by handleClick
  let cachedScale = 1;
  let cachedOffsetX = 0;
  let cachedOffsetY = 0;
  let hasCachedBounds = false;

  // RAF coalescing
  let rafPending = false;

  function scheduleDraw() {
    if (!rafPending) {
      rafPending = true;
      requestAnimationFrame(() => {
        rafPending = false;
        draw();
      });
    }
  }

  function draw() {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const graphData = graphStore.graphData;
    const panZoom = getCamera();
    const gpu = getGPURenderer();
    if (!graphData || !panZoom || !gpu) return;

    ctx.clearRect(0, 0, mmWidth, mmHeight);

    // Build node lookup map and cache sizes in one pass
    const nodeMap = new Map<string, typeof graphData.nodes[0]>();
    const sizeMap = new Map<string, { width: number; height: number }>();
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    for (const node of graphData.nodes) {
      nodeMap.set(node.id, node);
      const size = getNodeSize(node.id);
      sizeMap.set(node.id, size);
      minX = Math.min(minX, node.x);
      minY = Math.min(minY, node.y);
      maxX = Math.max(maxX, node.x + size.width);
      maxY = Math.max(maxY, node.y + size.height);
    }

    if (!isFinite(minX)) { hasCachedBounds = false; return; }

    const gw = maxX - minX + 40;
    const gh = maxY - minY + 40;
    const scale = Math.min(mmWidth / gw, mmHeight / gh);
    const offsetX = (mmWidth - gw * scale) / 2 - minX * scale + 20 * scale;
    const offsetY = (mmHeight - gh * scale) / 2 - minY * scale + 20 * scale;

    // Cache for handleClick
    cachedScale = scale;
    cachedOffsetX = offsetX;
    cachedOffsetY = offsetY;
    hasCachedBounds = true;

    // Draw edges
    ctx.strokeStyle = '#3A3F56';
    ctx.lineWidth = 0.5;
    for (const edge of graphData.edges) {
      const sn = nodeMap.get(edge.source);
      const tn = nodeMap.get(edge.target);
      if (!sn || !tn) continue;
      const ss = sizeMap.get(edge.source)!;
      const ts = sizeMap.get(edge.target)!;
      ctx.beginPath();
      ctx.moveTo((sn.x + ss.width / 2) * scale + offsetX, (sn.y + ss.height / 2) * scale + offsetY);
      ctx.lineTo((tn.x + ts.width / 2) * scale + offsetX, (tn.y + ts.height / 2) * scale + offsetY);
      ctx.stroke();
    }

    // Draw nodes
    for (const node of graphData.nodes) {
      const size = sizeMap.get(node.id)!;
      const x = node.x * scale + offsetX;
      const y = node.y * scale + offsetY;
      const w = Math.max(2, size.width * scale);
      const h = Math.max(1, size.height * scale);
      ctx.fillStyle = node.color;
      ctx.fillRect(x, y, w, h);
    }

    // Draw viewport rectangle
    const elRect = gpu.canvas.getBoundingClientRect();
    const viewW = elRect.width || 800;
    const viewH = elRect.height || 600;

    const topLeft = panZoom.viewportToGraph(0, 0);
    const bottomRight = panZoom.viewportToGraph(viewW, viewH);

    const vx = topLeft.x * scale + offsetX;
    const vy = topLeft.y * scale + offsetY;
    const vw = (bottomRight.x - topLeft.x) * scale;
    const vh = (bottomRight.y - topLeft.y) * scale;

    ctx.strokeStyle = '#4C8DFF';
    ctx.lineWidth = 2;
    ctx.strokeRect(vx, vy, vw, vh);
    ctx.fillStyle = 'rgba(76, 141, 255, 0.08)';
    ctx.fillRect(vx, vy, vw, vh);
  }

  // Drag state
  let isDragging = false;

  function minimapToGraph(e: MouseEvent): { x: number; y: number } | null {
    if (!hasCachedBounds) return null;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    return {
      x: (mx - cachedOffsetX) / cachedScale,
      y: (my - cachedOffsetY) / cachedScale,
    };
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
    if (!panZoom || !gpu || !hasCachedBounds) return;

    const pt = minimapToGraph(e as unknown as MouseEvent);
    if (!pt) return;

    const factor = e.deltaY < 0 ? 1.25 : 1 / 1.25;
    const newScale = Math.max(0.01, Math.min(50, panZoom.ratio * factor));

    // Zoom centered on the graph point under the minimap cursor
    const viewW = gpu.canvas.clientWidth || 800;
    const viewH = gpu.canvas.clientHeight || 600;
    const tx = viewW / 2 - pt.x * newScale;
    const ty = viewH / 2 - pt.y * newScale;
    panZoom.setState({ tx, ty, scale: newScale });
  }

  onMount(() => {
    const unwatch = $effect.root(() => {
      $effect(() => {
        // Read cameraVersion so the effect re-runs when the renderer initializes
        const _cv = graphStore.cameraVersion;
        if (graphStore.graphData && canvas && !collapsed) {
          // Clean up previous listener before setting up new one
          if (cleanupFn) {
            cleanupFn();
            cleanupFn = null;
          }
          const cam = getCamera();
          if (cam) {
            cam.on('updated', scheduleDraw);
            cleanupFn = () => cam.off('updated', scheduleDraw);
          }
          requestAnimationFrame(() => draw());
        }
      });
    });

    return () => {
      unwatch();
    };
  });

  onDestroy(() => {
    if (cleanupFn) {
      cleanupFn();
      cleanupFn = null;
    }
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', handleMouseUp);
    window.removeEventListener('mousemove', handleResizeMove);
    window.removeEventListener('mouseup', handleResizeUp);
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
        <svg class="w-3 h-3 text-gray-500 rotate-180" viewBox="0 0 12 12">
          <path d="M2 10L10 10L10 2" fill="none" stroke="currentColor" stroke-width="1.5"/>
          <path d="M5 10L10 10L10 5" fill="none" stroke="currentColor" stroke-width="1.5"/>
        </svg>
      </div>
      <canvas
        bind:this={canvas}
        class="absolute inset-0 cursor-crosshair"
        width={mmWidth}
        height={mmHeight}
        onmousedown={handleMouseDown}
        onwheel={handleWheel}
      ></canvas>
    </div>
  {/if}
  <button
    class="px-2 py-0.5 text-xs text-gray-400 bg-[--bg-panel] border border-[--border-color] {collapsed ? 'rounded' : 'rounded-b border-t-0'} hover:bg-[--bg-menu] transition-colors"
    onclick={() => { collapsed = !collapsed; localStorage.setItem('minimap-collapsed', String(collapsed)); }}
  >
    {collapsed ? 'Map' : 'Hide'}
  </button>
</div>
