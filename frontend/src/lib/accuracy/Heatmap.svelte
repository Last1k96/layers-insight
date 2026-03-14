<script lang="ts">
  import {
    getSpatialDims,
    extractSlice,
    valueToImageData,
    formatValue,
    drawColorbar,
    computeStats,
    type ColormapName,
  } from './tensorUtils';

  let {
    diff,
    main,
    ref,
    shape,
  }: {
    diff: Float32Array | null;
    main?: Float32Array | null;
    ref?: Float32Array | null;
    shape: number[];
  } = $props();

  let canvas: HTMLCanvasElement;
  let sliceIndex = $state(0);
  let channelIndex = $state(0);
  let colormap: ColormapName = $state('blueGreenRed');
  let globalNorm = $state(false);

  // Hover state
  let hoverX = $state(-1);
  let hoverY = $state(-1);
  let showTooltip = $state(false);
  let tooltipScreenX = $state(0);
  let tooltipScreenY = $state(0);

  // Zoom/pan state
  let zoom = $state(1);
  let panX = $state(0);
  let panY = $state(0);
  let dragging = $state(false);
  let dragStartX = 0;
  let dragStartY = 0;
  let panStartX = 0;
  let panStartY = 0;

  let dims = $derived(getSpatialDims(shape));

  // Global range for normalization
  let globalRange = $derived.by((): [number, number] | undefined => {
    if (!globalNorm || !diff) return undefined;
    const stats = computeStats(diff);
    return [stats.min, stats.max];
  });

  // Extract current slice
  let sliceData = $derived.by(() => {
    if (!diff) return null;
    return extractSlice(diff, shape, sliceIndex, channelIndex);
  });

  let mainSlice = $derived.by(() => {
    if (!main) return null;
    return extractSlice(main, shape, sliceIndex, channelIndex);
  });

  let refSlice = $derived.by(() => {
    if (!ref) return null;
    return extractSlice(ref, shape, sliceIndex, channelIndex);
  });

  // Offscreen image
  let offscreenImage = $derived.by(() => {
    if (!sliceData) return null;
    return valueToImageData(sliceData.data, sliceData.w, sliceData.h, colormap, globalRange);
  });

  // Auto-fit: compute base scale so image fills canvas
  let baseScale = $derived.by(() => {
    if (!sliceData || !canvas) return 1;
    const displayW = canvas.clientWidth;
    const displayH = canvas.clientHeight;
    if (!displayW || !displayH || !sliceData.w || !sliceData.h) return 1;
    return Math.min(displayW / sliceData.w, displayH / sliceData.h);
  });

  function redraw() {
    if (!canvas || !offscreenImage || !sliceData) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const displayW = canvas.clientWidth;
    const displayH = canvas.clientHeight;
    canvas.width = displayW;
    canvas.height = displayH;

    // Draw offscreen image at native resolution, then scale with zoom/pan
    const offscreen = new OffscreenCanvas(sliceData.w, sliceData.h);
    const offCtx = offscreen.getContext('2d')!;
    offCtx.putImageData(offscreenImage, 0, 0);

    // Effective scale = baseScale * zoom, center the image
    const effectiveScale = baseScale * zoom;
    const imgW = sliceData.w * effectiveScale;
    const imgH = sliceData.h * effectiveScale;
    const offsetX = (displayW - sliceData.w * baseScale) / 2 + panX;
    const offsetY = (displayH - sliceData.h * baseScale) / 2 + panY;

    ctx.clearRect(0, 0, displayW, displayH);
    ctx.setTransform(effectiveScale, 0, 0, effectiveScale, offsetX, offsetY);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0);
    ctx.resetTransform();

    // Draw crosshair
    if (showTooltip && hoverX >= 0 && hoverY >= 0) {
      const sx = hoverX * effectiveScale + offsetX;
      const sy = hoverY * effectiveScale + offsetY;
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(sx + 0.5, 0);
      ctx.lineTo(sx + 0.5, displayH);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, sy + 0.5);
      ctx.lineTo(displayW, sy + 0.5);
      ctx.stroke();
    }

    // Draw colorbar
    const stats = sliceData ? computeStats(sliceData.data) : null;
    if (stats) {
      const range = globalRange || [stats.min, stats.max];
      drawColorbar(ctx, 10, displayH - 30, Math.min(200, displayW - 20), 12, colormap, range[0], range[1]);
    }
  }

  $effect(() => {
    // Track all reactive deps
    offscreenImage;
    zoom;
    panX;
    panY;
    showTooltip;
    hoverX;
    hoverY;
    redraw();
  });

  // Event handlers
  function screenToData(clientX: number, clientY: number): [number, number] {
    if (!canvas || !sliceData) return [-1, -1];
    const rect = canvas.getBoundingClientRect();
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;
    const displayW = canvas.clientWidth;
    const displayH = canvas.clientHeight;
    const offsetX = (displayW - sliceData.w * baseScale) / 2 + panX;
    const offsetY = (displayH - sliceData.h * baseScale) / 2 + panY;
    const effectiveScale = baseScale * zoom;
    const dataX = (sx - offsetX) / effectiveScale;
    const dataY = (sy - offsetY) / effectiveScale;
    return [dataX, dataY];
  }

  function handleWheel(e: WheelEvent) {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    // Zoom centered on cursor — panX/panY are offsets from centered position
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const displayW = canvas.clientWidth;
    const displayH = canvas.clientHeight;
    // Center offset before zoom
    const cxOff = (displayW - (sliceData?.w ?? 1) * baseScale) / 2;
    const cyOff = (displayH - (sliceData?.h ?? 1) * baseScale) / 2;
    // Adjust pan so point under cursor stays fixed
    panX = cx - (cx - cxOff - panX) * factor - cxOff;
    panY = cy - (cy - cyOff - panY) * factor - cyOff;
    zoom *= factor;
  }

  function handleMouseDown(e: MouseEvent) {
    if (e.button !== 0) return;
    dragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    panStartX = panX;
    panStartY = panY;
  }

  function handleMouseMove(e: MouseEvent) {
    if (dragging) {
      panX = panStartX + (e.clientX - dragStartX);
      panY = panStartY + (e.clientY - dragStartY);
      return;
    }
    const [dx, dy] = screenToData(e.clientX, e.clientY);
    const ix = Math.floor(dx);
    const iy = Math.floor(dy);
    if (sliceData && ix >= 0 && ix < sliceData.w && iy >= 0 && iy < sliceData.h) {
      hoverX = ix;
      hoverY = iy;
      tooltipScreenX = e.clientX;
      tooltipScreenY = e.clientY;
      showTooltip = true;
    } else {
      showTooltip = false;
    }
  }

  function handleMouseUp() { dragging = false; }
  function handleMouseLeave() { dragging = false; showTooltip = false; }

  // Reset zoom/pan when slice changes
  $effect(() => {
    const _s = shape;
    zoom = 1;
    panX = 0;
    panY = 0;
  });

  const colormapOptions: { value: ColormapName; label: string }[] = [
    { value: 'blueGreenRed', label: 'Blue-Green-Red' },
    { value: 'viridis', label: 'Viridis' },
    { value: 'coolwarm', label: 'Coolwarm' },
    { value: 'magma', label: 'Magma' },
  ];
</script>

<svelte:window onmouseup={handleMouseUp} />

<div class="flex flex-col gap-4 relative h-full">
  <!-- Controls -->
  <div class="flex flex-wrap gap-4 items-center text-xs">
    {#if dims.batches > 1}
      <label class="flex items-center gap-2">
        <span class="text-gray-400">Batch:</span>
        <input type="range" min="0" max={dims.batches - 1} bind:value={sliceIndex} class="w-24" />
        <span class="text-gray-300 w-6">{sliceIndex}</span>
      </label>
    {/if}
    {#if dims.channels > 1}
      <label class="flex items-center gap-2">
        <span class="text-gray-400">Channel:</span>
        <input type="range" min="0" max={dims.channels - 1} bind:value={channelIndex} class="w-32" />
        <span class="text-gray-300 w-8">{channelIndex}/{dims.channels}</span>
      </label>
    {/if}
    <label class="flex items-center gap-2">
      <span class="text-gray-400">Colormap:</span>
      <select
        bind:value={colormap}
        class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300"
      >
        {#each colormapOptions as opt}
          <option value={opt.value}>{opt.label}</option>
        {/each}
      </select>
    </label>
    <label class="flex items-center gap-1.5 text-gray-400">
      <input type="checkbox" bind:checked={globalNorm} />
      Global norm
    </label>
    <span class="text-gray-500">
      Shape: [{shape.join(', ')}] | Slice: {dims.height}x{dims.width}
    </span>
  </div>

  <!-- Canvas -->
  <div class="flex-1 flex justify-center bg-surface-base rounded-lg p-4 overflow-hidden min-h-0">
    <canvas
      bind:this={canvas}
      class="w-full h-full cursor-crosshair"
      style="image-rendering: pixelated;"
      onwheel={handleWheel}
      onmousedown={handleMouseDown}
      onmousemove={handleMouseMove}
      onmouseleave={handleMouseLeave}
    ></canvas>
  </div>

  <!-- Tooltip -->
  {#if showTooltip && hoverX >= 0 && hoverY >= 0 && sliceData}
    {@const idx = hoverY * sliceData.w + hoverX}
    <div
      class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
      style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
    >
      <div class="font-mono text-gray-400">[{hoverY}, {hoverX}]</div>
      {#if refSlice}
        <div><span class="text-gray-400">Ref:</span> {formatValue(refSlice.data[idx])}</div>
      {/if}
      {#if mainSlice}
        <div><span class="text-gray-400">Main:</span> {formatValue(mainSlice.data[idx])}</div>
      {/if}
      <div><span class="text-gray-400">Diff:</span> {formatValue(sliceData.data[idx])}</div>
    </div>
  {/if}
</div>
