<script lang="ts">
  import { renderDiagnostics, type DiagLayout } from './diagRenderer';
  import { ALL_COLORMAP_OPTIONS, type ColormapName } from './tensorUtils';

  let { main, ref, shape, mainLabel = 'Main', refLabel = 'Reference' }: {
    main: Float32Array;
    ref: Float32Array;
    shape: number[];
    mainLabel?: string;
    refLabel?: string;
  } = $props();

  let canvas: HTMLCanvasElement;
  let container: HTMLDivElement;
  let containerWidth = $state(800);

  let colormap: ColormapName = $state('viridis');
  let zoomLevel = $state(1);
  let layout: DiagLayout | null = $state(null);

  // Tooltip state
  let tooltipVisible = $state(false);
  let tooltipX = $state(0);
  let tooltipY = $state(0);
  let tooltipText = $state('');

  $effect(() => {
    if (!container) return;
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) {
        containerWidth = entry.contentRect.width;
      }
    });
    obs.observe(container);
    return () => obs.disconnect();
  });

  $effect(() => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    layout = renderDiagnostics(ctx, canvas, main, ref, shape, containerWidth, mainLabel, refLabel, colormap);
  });

  function handleMouseMove(e: MouseEvent) {
    if (!layout || !canvas) {
      tooltipVisible = false;
      return;
    }

    const rect = canvas.getBoundingClientRect();
    // Account for zoom: the canvas is scaled via CSS transform, so
    // getBoundingClientRect() returns the scaled size. We need coords
    // in the unscaled canvas pixel space.
    const mx = (e.clientX - rect.left) / zoomLevel;
    const my = (e.clientY - rect.top) / zoomLevel;

    const { panelW, panelH, blocksPerRow, channelCount, blockW, blockH, blockGap, labelH, gap } = layout;

    // Determine which block column and row the mouse is in
    const cellW = blockW + blockGap;
    const cellH = blockH + blockGap;
    const col = Math.floor(mx / cellW);
    const row = Math.floor(my / cellH);

    if (col < 0 || col >= blocksPerRow || row < 0) {
      tooltipVisible = false;
      return;
    }

    const channel = row * blocksPerRow + col;
    if (channel >= channelCount) {
      tooltipVisible = false;
      return;
    }

    // Local coordinates within the block
    const bx = col * cellW;
    const by = row * cellH;
    const lx = mx - bx;
    const ly = my - by;

    // Check if within the label row at the top
    if (ly < labelH) {
      tooltipVisible = false;
      return;
    }

    const panelY = ly - labelH;
    let panelType: string | null = null;

    // Top row: ref (left) | main (right)
    if (panelY >= 0 && panelY < panelH) {
      if (lx >= 0 && lx < panelW) {
        panelType = refLabel;
      } else if (lx >= panelW + gap && lx < panelW + gap + panelW) {
        panelType = mainLabel;
      }
    }
    // Bottom row: diff (left) | density (right)
    else if (panelY >= panelH + gap && panelY < panelH + gap + panelH) {
      if (lx >= 0 && lx < panelW) {
        panelType = 'Diff';
      } else if (lx >= panelW + gap && lx < panelW + gap + panelW) {
        panelType = 'Density';
      }
    }

    if (!panelType) {
      tooltipVisible = false;
      return;
    }

    tooltipText = `Ch ${channel} \u2014 ${panelType}`;
    tooltipX = e.clientX + 12;
    tooltipY = e.clientY + 12;
    tooltipVisible = true;
  }

  function handleMouseLeave() {
    tooltipVisible = false;
  }

  function zoomIn() {
    zoomLevel = Math.min(zoomLevel + 0.5, 5);
  }

  function zoomOut() {
    zoomLevel = Math.max(zoomLevel - 0.5, 0.5);
  }

  function zoomReset() {
    zoomLevel = 1;
  }
</script>

<div bind:this={container} class="w-full">
  <!-- Controls bar -->
  <div class="flex items-center gap-3 mb-2 text-xs text-gray-300">
    <label class="flex items-center gap-1">
      Colormap
      <select
        bind:value={colormap}
        class="bg-gray-800 border border-gray-600 rounded px-1.5 py-0.5 text-xs text-gray-200"
      >
        {#each ALL_COLORMAP_OPTIONS as opt}
          <option value={opt.value}>{opt.label}</option>
        {/each}
      </select>
    </label>

    <span class="border-l border-gray-600 h-4"></span>

    <div class="flex items-center gap-1">
      <span>Zoom</span>
      <button
        onclick={zoomOut}
        class="bg-gray-700 hover:bg-gray-600 rounded px-1.5 py-0.5 text-xs"
        title="Zoom out"
      >&minus;</button>
      <span class="w-10 text-center">{Math.round(zoomLevel * 100)}%</span>
      <button
        onclick={zoomIn}
        class="bg-gray-700 hover:bg-gray-600 rounded px-1.5 py-0.5 text-xs"
        title="Zoom in"
      >+</button>
      <button
        onclick={zoomReset}
        class="bg-gray-700 hover:bg-gray-600 rounded px-1.5 py-0.5 text-xs"
        title="Reset zoom"
      >Reset</button>
    </div>
  </div>

  <!-- Canvas wrapper with scroll and zoom -->
  <div class="overflow-auto max-h-[80vh] border border-gray-700 rounded">
    <canvas
      bind:this={canvas}
      class="block"
      style="transform: scale({zoomLevel}); transform-origin: top left;"
      onmousemove={handleMouseMove}
      onmouseleave={handleMouseLeave}
    ></canvas>
  </div>
</div>

<!-- Tooltip (fixed position, follows cursor) -->
{#if tooltipVisible}
  <div
    class="fixed z-50 pointer-events-none bg-gray-900 border border-gray-600 rounded px-2 py-1 text-xs text-gray-200 shadow-lg"
    style="left: {tooltipX}px; top: {tooltipY}px;"
  >
    {tooltipText}
  </div>
{/if}
