<script lang="ts">
  import { renderDiagnostics, computeChannelMetrics, type DiagLayout, type ChannelMetrics, type DiagOptions } from './diagRenderer';
  import { getSpatialDims, formatValue, drawColorbar, ALL_COLORMAP_OPTIONS, type ColormapName } from './tensorUtils';
  import { rangeScroll } from './rangeScroll';
  import { keyboardNav } from './keyboardNav';

  let { main, ref, shape, mainLabel = 'Main', refLabel = 'Reference' }: {
    main: Float32Array;
    ref: Float32Array;
    shape: number[];
    mainLabel?: string;
    refLabel?: string;
  } = $props();

  let canvas: HTMLCanvasElement;
  let expandedCanvas: HTMLCanvasElement;
  let container: HTMLDivElement;
  let containerWidth = $state(800);

  // Controls
  let cmapDiff: ColormapName = $state('coolwarm');
  let cmapDensity: ColormapName = $state('viridis');
  let signedDensity = $state(true);
  let highlightWorst = $state(true);
  let zoomLevel = $state(1);
  let batch = $state(0);
  let sortBy = $state<'index' | 'cosSim' | 'meanDiff' | 'maxDiff'>('index');

  // Expanded view
  let expandedChannel = $state<number | null>(null);

  // Layout from renderer
  let layout: DiagLayout | null = $state(null);

  // Tooltip state
  let tooltipVisible = $state(false);
  let tooltipX = $state(0);
  let tooltipY = $state(0);
  let tooltipData = $state<{
    ch: number;
    panel: string;
    row?: number;
    col?: number;
    refVal?: number;
    mainVal?: number;
    diff?: number;
    metrics?: ChannelMetrics;
  } | null>(null);

  let dims = $derived(getSpatialDims(shape));

  // Compute metrics for sort/filter (separate from render)
  let allMetrics = $derived(computeChannelMetrics(main, ref, shape, batch));

  let channelOrder = $derived.by((): number[] => {
    if (sortBy === 'index') return Array.from({ length: dims.channels }, (_, i) => i);
    const sorted = [...allMetrics].sort((a, b) => {
      switch (sortBy) {
        case 'cosSim': return a.cosSim - b.cosSim; // worst first
        case 'meanDiff': return b.meanAbsDiff - a.meanAbsDiff; // worst first
        case 'maxDiff': return b.maxAbsDiff - a.maxAbsDiff; // worst first
        default: return a.ch - b.ch;
      }
    });
    return sorted.map(m => m.ch);
  });

  let diagOptions = $derived<DiagOptions>({
    colormaps: { diff: cmapDiff, density: cmapDensity },
    signedDensity,
    channelOrder: expandedChannel !== null ? [expandedChannel] : channelOrder,
    batch,
    highlightWorst: expandedChannel === null && highlightWorst,
  });

  // ResizeObserver
  $effect(() => {
    if (!container) return;
    const obs = new ResizeObserver((entries) => {
      for (const entry of entries) containerWidth = entry.contentRect.width;
    });
    obs.observe(container);
    return () => obs.disconnect();
  });

  // Render grid view
  $effect(() => {
    if (expandedChannel !== null) return;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    layout = renderDiagnostics(ctx, canvas, main, ref, shape, containerWidth, mainLabel, refLabel, diagOptions);
  });

  // Render expanded view
  $effect(() => {
    if (expandedChannel === null) return;
    if (!expandedCanvas) return;
    const ctx = expandedCanvas.getContext('2d');
    if (!ctx) return;
    const expandedLayout = renderDiagnostics(ctx, expandedCanvas, main, ref, shape, containerWidth, mainLabel, refLabel, diagOptions);
    if (expandedLayout) layout = expandedLayout;

    // Draw colorbars below the panels
    if (expandedLayout) {
      const { panelW, panelH, labelH: lh, gap: g } = expandedLayout;
      const cbY = lh + panelH * 2 + g + 8;
      const cbW = Math.min(panelW - 10, 200);
      // Diff colorbar (below bottom-left)
      drawColorbar(ctx, 5, cbY, cbW, 10, cmapDiff, -expandedLayout.globalDiffMax, expandedLayout.globalDiffMax);
      // Density colorbar (below bottom-right)
      const oxDensity = panelW + g;
      drawColorbar(ctx, oxDensity + 5, cbY, cbW, 10, cmapDensity, 0, 1);
    }
  });

  // Tooltip: reverse-map hover coords to tensor values
  function handleMouseMove(e: MouseEvent) {
    if (!layout || (!canvas && !expandedCanvas)) {
      tooltipVisible = false;
      return;
    }
    const activeCanvas = expandedChannel !== null ? expandedCanvas : canvas;
    if (!activeCanvas) { tooltipVisible = false; return; }

    const rect = activeCanvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / zoomLevel;
    const my = (e.clientY - rect.top) / zoomLevel;

    const { panelW, panelH, blocksPerRow, blockW, blockH, blockGap, labelH, gap, channelOrder: order, xMap, yMap, H, W, channelMetrics } = layout;

    const cellW = blockW + blockGap;
    const cellH = blockH + blockGap;
    const col = Math.floor(mx / cellW);
    const row = Math.floor(my / cellH);

    if (col < 0 || col >= blocksPerRow || row < 0) { tooltipVisible = false; return; }

    const displayIdx = row * blocksPerRow + col;
    if (displayIdx >= order.length) { tooltipVisible = false; return; }
    const ch = order[displayIdx];

    const bx = col * cellW;
    const by = row * cellH;
    const lx = mx - bx;
    const ly = my - by;

    if (ly < labelH) { tooltipVisible = false; return; }

    const panelY = ly - labelH;
    let panel: string | null = null;
    let inTopRow = false;
    let inBottomRow = false;

    if (panelY >= 0 && panelY < panelH) {
      inTopRow = true;
      if (lx >= 0 && lx < panelW) panel = refLabel;
      else if (lx >= panelW + gap && lx < panelW + gap + panelW) panel = mainLabel;
    } else if (panelY >= panelH + gap && panelY < panelH + gap + panelH) {
      inBottomRow = true;
      if (lx >= 0 && lx < panelW) panel = 'Diff';
      else if (lx >= panelW + gap && lx < panelW + gap + panelW) panel = 'Density';
    }

    if (!panel) { tooltipVisible = false; return; }

    // Compute tensor coordinates
    const panelLocalX = lx >= panelW + gap ? lx - panelW - gap : lx;
    const panelLocalY = inTopRow ? panelY : panelY - panelH - gap;
    const px = Math.floor(panelLocalX);
    const py = Math.floor(panelLocalY);

    const metrics = channelMetrics[displayIdx];

    if (panel !== 'Density' && px >= 0 && px < panelW && py >= 0 && py < panelH) {
      const srcX = xMap[px];
      const srcY = yMap[py];
      const batchStride = layout.channelCount * H * W;
      const offset = batch * batchStride + ch * H * W + srcY * W + srcX;
      const rv = ref[offset];
      const mv = main[offset];
      tooltipData = { ch, panel, row: srcY, col: srcX, refVal: rv, mainVal: mv, diff: mv - rv, metrics };
    } else {
      tooltipData = { ch, panel, metrics };
    }

    tooltipX = e.clientX + 12;
    tooltipY = e.clientY + 12;
    tooltipVisible = true;
  }

  function handleMouseLeave() { tooltipVisible = false; }

  function handleClick(e: MouseEvent) {
    if (expandedChannel !== null) return; // already expanded
    if (!layout || !canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / zoomLevel;
    const my = (e.clientY - rect.top) / zoomLevel;

    const cellW = layout.blockW + layout.blockGap;
    const cellH = layout.blockH + layout.blockGap;
    const col = Math.floor(mx / cellW);
    const row = Math.floor(my / cellH);

    if (col < 0 || col >= layout.blocksPerRow || row < 0) return;
    const displayIdx = row * layout.blocksPerRow + col;
    if (displayIdx >= layout.channelOrder.length) return;

    expandedChannel = layout.channelOrder[displayIdx];
  }

  function collapseExpanded() { expandedChannel = null; }

  // Expanded channel navigation
  function stepExpandedChannel(delta: number) {
    if (expandedChannel === null) return;
    const idx = channelOrder.indexOf(expandedChannel);
    if (idx < 0) return;
    const next = idx + delta;
    if (next >= 0 && next < channelOrder.length) {
      expandedChannel = channelOrder[next];
    }
  }

  // Expanded channel metrics
  let expandedMetrics = $derived.by((): ChannelMetrics | null => {
    if (expandedChannel === null) return null;
    return allMetrics.find(m => m.ch === expandedChannel) ?? null;
  });

  function zoomIn() { zoomLevel = Math.min(zoomLevel + 0.5, 5); }
  function zoomOut() { zoomLevel = Math.max(zoomLevel - 0.5, 0.5); }
  function zoomReset() { zoomLevel = 1; }
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div bind:this={container} class="w-full h-full flex flex-col gap-2" tabindex="0" use:keyboardNav={{
  onResetZoom: zoomReset,
  onNextChannel: () => stepExpandedChannel(1),
  onPrevChannel: () => stepExpandedChannel(-1),
  onNextBatch: () => { if (batch < dims.batches - 1) batch++; },
  onPrevBatch: () => { if (batch > 0) batch--; },
}}>
  <!-- Controls bar -->
  <div class="flex flex-wrap items-center gap-3 text-xs text-gray-300 shrink-0">
    {#if dims.batches > 1}
      <label class="flex items-center gap-1">
        Batch
        <input use:rangeScroll type="range" min="0" max={dims.batches - 1} bind:value={batch} class="w-16" />
        <span class="w-4 text-center">{batch}</span>
      </label>
      <span class="border-l border-gray-600 h-4"></span>
    {/if}

    <label class="flex items-center gap-1">
      Sort
      <select bind:value={sortBy} class="bg-gray-800 border border-gray-600 rounded px-1.5 py-0.5 text-xs text-gray-200">
        <option value="index">Index</option>
        <option value="cosSim">Worst Cosine</option>
        <option value="meanDiff">Mean |Diff|</option>
        <option value="maxDiff">Max |Diff|</option>
      </select>
    </label>

    <span class="border-l border-gray-600 h-4"></span>

    <label class="flex items-center gap-1">
      Diff
      <select bind:value={cmapDiff} class="bg-gray-800 border border-gray-600 rounded px-1.5 py-0.5 text-xs text-gray-200">
        {#each ALL_COLORMAP_OPTIONS as opt}
          <option value={opt.value}>{opt.label}</option>
        {/each}
      </select>
    </label>
    <label class="flex items-center gap-1">
      Density
      <select bind:value={cmapDensity} class="bg-gray-800 border border-gray-600 rounded px-1.5 py-0.5 text-xs text-gray-200">
        {#each ALL_COLORMAP_OPTIONS as opt}
          <option value={opt.value}>{opt.label}</option>
        {/each}
      </select>
    </label>

    <span class="border-l border-gray-600 h-4"></span>

    <label class="flex items-center gap-1.5 text-gray-400">
      <input type="checkbox" bind:checked={signedDensity} /> Signed density
    </label>
    <label class="flex items-center gap-1.5 text-gray-400">
      <input type="checkbox" bind:checked={highlightWorst} /> Highlight worst
    </label>

    <span class="border-l border-gray-600 h-4"></span>

    <div class="flex items-center gap-1">
      <span>Zoom</span>
      <button onclick={zoomOut} class="bg-gray-700 hover:bg-gray-600 rounded px-1.5 py-0.5 text-xs" title="Zoom out">&minus;</button>
      <span class="w-10 text-center">{Math.round(zoomLevel * 100)}%</span>
      <button onclick={zoomIn} class="bg-gray-700 hover:bg-gray-600 rounded px-1.5 py-0.5 text-xs" title="Zoom in">+</button>
      <button onclick={zoomReset} class="bg-gray-700 hover:bg-gray-600 rounded px-1.5 py-0.5 text-xs" title="Reset zoom">Reset</button>
    </div>

    <span class="text-gray-500 ml-auto">{dims.channels} channels | {dims.height}&times;{dims.width}</span>
  </div>

  <!-- Expanded view header -->
  {#if expandedChannel !== null && expandedMetrics}
    <div class="flex items-center gap-3 text-xs shrink-0 bg-surface-base rounded px-3 py-2 border border-edge">
      <button onclick={collapseExpanded} class="text-gray-400 hover:text-white px-2 py-0.5 border border-edge rounded">
        &larr; Grid
      </button>
      <span class="font-medium text-content-primary">Channel {expandedChannel}</span>
      <span class={expandedMetrics.cosSim > 0.999 ? 'text-green-400' : expandedMetrics.cosSim > 0.99 ? 'text-yellow-400' : 'text-red-400'}>
        cos={expandedMetrics.cosSim.toFixed(6)}
      </span>
      <span class="text-gray-400">mean|diff|={formatValue(expandedMetrics.meanAbsDiff)}</span>
      <span class="text-gray-400">max|diff|={formatValue(expandedMetrics.maxAbsDiff)}</span>
      <span class="text-gray-600 text-[10px] ml-auto">[/] prev/next channel &middot; Esc grid</span>
    </div>
  {/if}

  <!-- Canvas -->
  <div class="flex-1 overflow-auto border border-gray-700 rounded min-h-0">
    {#if expandedChannel !== null}
      <canvas
        bind:this={expandedCanvas}
        class="block cursor-crosshair"
        style="transform: scale({zoomLevel}); transform-origin: top left;"
        onmousemove={handleMouseMove}
        onmouseleave={handleMouseLeave}
      ></canvas>
    {:else}
      <canvas
        bind:this={canvas}
        class="block cursor-pointer"
        style="transform: scale({zoomLevel}); transform-origin: top left;"
        onmousemove={handleMouseMove}
        onmouseleave={handleMouseLeave}
        onclick={handleClick}
      ></canvas>
    {/if}
  </div>
</div>

<!-- Escape to collapse -->
<svelte:window onkeydown={(e) => { if (e.key === 'Escape' && expandedChannel !== null) { e.stopPropagation(); expandedChannel = null; } }} />

<!-- Tooltip -->
{#if tooltipVisible && tooltipData}
  <div
    class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg max-w-[300px]"
    style="left: {tooltipX}px; top: {tooltipY}px;"
  >
    <div class="font-medium text-gray-300 mb-1">Ch {tooltipData.ch} — {tooltipData.panel}</div>
    {#if tooltipData.row !== undefined && tooltipData.col !== undefined}
      <div class="font-mono text-gray-500">[{tooltipData.row}, {tooltipData.col}]</div>
      <div><span class="text-gray-400">{refLabel}:</span> {formatValue(tooltipData.refVal!)}</div>
      <div><span class="text-gray-400">{mainLabel}:</span> {formatValue(tooltipData.mainVal!)}</div>
      <div><span class="text-gray-400">diff:</span> <span class={tooltipData.diff! > 0 ? 'text-blue-400' : tooltipData.diff! < 0 ? 'text-red-400' : ''}>{formatValue(tooltipData.diff!)}</span></div>
    {/if}
    {#if tooltipData.metrics}
      <div class="border-t border-gray-700 mt-1 pt-1 text-gray-500">
        cos={tooltipData.metrics.cosSim.toFixed(4)} max|d|={formatValue(tooltipData.metrics.maxAbsDiff)}
      </div>
    {/if}
  </div>
{/if}
