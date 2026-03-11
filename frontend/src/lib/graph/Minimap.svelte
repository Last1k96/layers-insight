<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { getCamera, getGPURenderer, getNodeSize } from './renderer';

  let canvas: HTMLCanvasElement;
  let collapsed = $state(false);
  let cleanupFn: (() => void) | null = null;

  const MINIMAP_WIDTH = 200;
  const MINIMAP_HEIGHT = 150;

  function draw() {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const graphData = graphStore.graphData;
    const panZoom = getCamera();
    const gpu = getGPURenderer();
    if (!graphData || !panZoom || !gpu) return;

    canvas.width = MINIMAP_WIDTH;
    canvas.height = MINIMAP_HEIGHT;
    ctx.clearRect(0, 0, MINIMAP_WIDTH, MINIMAP_HEIGHT);

    // Compute graph bounds
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const node of graphData.nodes) {
      const size = getNodeSize(node.id);
      minX = Math.min(minX, node.x);
      minY = Math.min(minY, node.y);
      maxX = Math.max(maxX, node.x + size.width);
      maxY = Math.max(maxY, node.y + size.height);
    }

    if (!isFinite(minX)) return;

    const gw = maxX - minX + 40;
    const gh = maxY - minY + 40;
    const scale = Math.min(MINIMAP_WIDTH / gw, MINIMAP_HEIGHT / gh);
    const offsetX = (MINIMAP_WIDTH - gw * scale) / 2 - minX * scale + 20 * scale;
    const offsetY = (MINIMAP_HEIGHT - gh * scale) / 2 - minY * scale + 20 * scale;

    // Draw edges
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 0.5;
    for (const edge of graphData.edges) {
      const sn = graphData.nodes.find(n => n.id === edge.source);
      const tn = graphData.nodes.find(n => n.id === edge.target);
      if (!sn || !tn) continue;
      const ss = getNodeSize(edge.source);
      const ts = getNodeSize(edge.target);
      ctx.beginPath();
      ctx.moveTo((sn.x + ss.width / 2) * scale + offsetX, (sn.y + ss.height / 2) * scale + offsetY);
      ctx.lineTo((tn.x + ts.width / 2) * scale + offsetX, (tn.y + ts.height / 2) * scale + offsetY);
      ctx.stroke();
    }

    // Draw nodes
    for (const node of graphData.nodes) {
      const size = getNodeSize(node.id);
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

    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 2;
    ctx.strokeRect(vx, vy, vw, vh);
    ctx.fillStyle = 'rgba(59, 130, 246, 0.08)';
    ctx.fillRect(vx, vy, vw, vh);
  }

  function handleClick(e: MouseEvent) {
    const panZoom = getCamera();
    const gpu = getGPURenderer();
    const graphData = graphStore.graphData;
    if (!panZoom || !gpu || !graphData) return;

    // Compute same bounds/scale as draw()
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const node of graphData.nodes) {
      const size = getNodeSize(node.id);
      minX = Math.min(minX, node.x);
      minY = Math.min(minY, node.y);
      maxX = Math.max(maxX, node.x + size.width);
      maxY = Math.max(maxY, node.y + size.height);
    }
    if (!isFinite(minX)) return;

    const gw = maxX - minX + 40;
    const gh = maxY - minY + 40;
    const scale = Math.min(MINIMAP_WIDTH / gw, MINIMAP_HEIGHT / gh);
    const offsetX = (MINIMAP_WIDTH - gw * scale) / 2 - minX * scale + 20 * scale;
    const offsetY = (MINIMAP_HEIGHT - gh * scale) / 2 - minY * scale + 20 * scale;

    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const graphX = (clickX - offsetX) / scale;
    const graphY = (clickY - offsetY) / scale;

    const viewW = gpu.canvas.clientWidth || 800;
    const viewH = gpu.canvas.clientHeight || 600;
    const tx = viewW / 2 - graphX * panZoom.ratio;
    const ty = viewH / 2 - graphY * panZoom.ratio;

    panZoom.animate({ tx, ty }, 200);
  }

  onMount(() => {
    const unwatch = $effect.root(() => {
      $effect(() => {
        if (graphStore.graphData && canvas && !collapsed) {
          const cam = getCamera();
          if (cam) {
            const handler = () => draw();
            cam.on('updated', handler);
            cleanupFn = () => cam.off('updated', handler);
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
  });
</script>

<div class="absolute bottom-4 right-4 z-10">
  <button
    class="absolute -top-7 right-0 px-2 py-0.5 text-xs text-gray-400 bg-gray-800 border border-gray-700 rounded-t hover:bg-gray-700 transition-colors"
    onclick={() => collapsed = !collapsed}
  >
    {collapsed ? 'Map' : 'Hide'}
  </button>

  {#if !collapsed}
    <div
      class="relative border border-gray-700 rounded bg-gray-900 overflow-hidden shadow-lg"
      style="width: {MINIMAP_WIDTH}px; height: {MINIMAP_HEIGHT}px;"
    >
      <canvas
        bind:this={canvas}
        class="absolute inset-0 cursor-crosshair"
        width={MINIMAP_WIDTH}
        height={MINIMAP_HEIGHT}
        onclick={handleClick}
      ></canvas>
    </div>
  {/if}
</div>
