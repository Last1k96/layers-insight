<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Sigma from 'sigma';
  import Graph from 'graphology';
  import { graphStore } from '../stores/graph.svelte';
  import { getSigma, getGraph } from './renderer';

  let container: HTMLDivElement;
  let overlayCanvas: HTMLCanvasElement;
  let minimapSigma: Sigma | null = null;
  let minimapGraph: Graph | null = null;
  let collapsed = $state(false);
  let cameraCleanup: (() => void) | null = null;

  const MINIMAP_WIDTH = 200;
  const MINIMAP_HEIGHT = 150;

  function initMinimap() {
    destroyMinimap();

    const mainGraph = getGraph();
    const graphData = graphStore.graphData;
    if (!mainGraph || !graphData || !container) return;

    // Create a separate graph for the minimap
    minimapGraph = new Graph();

    for (const node of graphData.nodes) {
      minimapGraph.addNode(node.id, {
        x: node.x,
        y: node.y,
        size: 2,
        color: node.color,
        type: 'circle',
      });
    }

    for (const edge of graphData.edges) {
      try {
        minimapGraph.addEdge(edge.source, edge.target, {
          color: '#374151',
          size: 0.5,
          type: 'line',
        });
      } catch {
        // Skip duplicates
      }
    }

    minimapSigma = new Sigma(minimapGraph, container, {
      renderLabels: false,
      renderEdgeLabels: false,
      defaultEdgeColor: '#374151',
      defaultEdgeType: 'line',
      allowInvalidContainer: true,
      enableCameraRotation: false,
      // Disable interactions — we handle click manually
      enableCameraZooming: false,
      enableCameraPanning: false,
    } as any);

    // Fit the minimap to show entire graph
    minimapSigma.getCamera().setState({ x: 0.5, y: 0.5, ratio: 1, angle: 0 });

    // Listen to main camera changes to update viewport rectangle
    const mainSigma = getSigma();
    if (mainSigma) {
      const mainCamera = mainSigma.getCamera();
      const handler = () => drawViewportRect();
      mainCamera.on('updated', handler);
      cameraCleanup = () => mainCamera.off('updated', handler);
    }

    // Initial draw
    requestAnimationFrame(() => drawViewportRect());
  }

  function drawViewportRect() {
    if (!overlayCanvas || !minimapSigma) return;

    const mainSigma = getSigma();
    if (!mainSigma) return;

    const ctx = overlayCanvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match display
    overlayCanvas.width = MINIMAP_WIDTH;
    overlayCanvas.height = MINIMAP_HEIGHT;
    ctx.clearRect(0, 0, MINIMAP_WIDTH, MINIMAP_HEIGHT);

    // Get the main camera's visible rectangle in graph coords
    const viewRect = mainSigma.viewRectangle();
    // viewRect: { x1, y1, x2, y2, height } — corners in graph space

    // Convert graph coords to minimap viewport coords
    const tl = minimapSigma.graphToViewport({ x: viewRect.x1, y: viewRect.y1 });
    const br = minimapSigma.graphToViewport({ x: viewRect.x2, y: viewRect.y2 });

    const x = Math.min(tl.x, br.x);
    const y = Math.min(tl.y, br.y);
    const w = Math.abs(br.x - tl.x);
    const h = Math.abs(br.y - tl.y);

    // Draw viewport rectangle
    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    // Semi-transparent fill
    ctx.fillStyle = 'rgba(59, 130, 246, 0.08)';
    ctx.fillRect(x, y, w, h);
  }

  function handleClick(e: MouseEvent) {
    if (!minimapSigma) return;

    const mainSigma = getSigma();
    if (!mainSigma) return;

    const rect = overlayCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Convert minimap viewport coords to graph coords
    const graphCoords = minimapSigma.viewportToGraph({ x, y });

    // Navigate main camera
    const mainCamera = mainSigma.getCamera();
    mainCamera.animate(
      { x: graphCoords.x, y: graphCoords.y },
      { duration: 200 },
    );
  }

  function destroyMinimap() {
    if (cameraCleanup) {
      cameraCleanup();
      cameraCleanup = null;
    }
    if (minimapSigma) {
      minimapSigma.kill();
      minimapSigma = null;
    }
    minimapGraph = null;
  }

  onMount(() => {
    const unwatch = $effect.root(() => {
      $effect(() => {
        if (graphStore.graphData && container) {
          // Small delay to ensure main renderer is ready
          requestAnimationFrame(() => initMinimap());
        }
      });
    });

    return () => {
      unwatch();
    };
  });

  onDestroy(() => {
    destroyMinimap();
  });
</script>

<div class="absolute bottom-4 right-4 z-10">
  <!-- Toggle button -->
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
      <!-- Sigma.js minimap container -->
      <div
        bind:this={container}
        class="absolute inset-0"
      ></div>

      <!-- Overlay canvas for viewport rectangle + click handling -->
      <canvas
        bind:this={overlayCanvas}
        class="absolute inset-0 cursor-crosshair"
        width={MINIMAP_WIDTH}
        height={MINIMAP_HEIGHT}
        onclick={handleClick}
      ></canvas>
    </div>
  {/if}
</div>
