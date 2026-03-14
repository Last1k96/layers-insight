<script lang="ts">
  import { getSpatialDims, extractSlice, computeStats, COLORMAPS } from './tensorUtils';

  let { main, ref, shape }: {
    main: Float32Array;
    ref: Float32Array;
    shape: number[];
  } = $props();

  let canvas: HTMLCanvasElement;
  let container: HTMLDivElement;
  let containerWidth = $state(800);
  let rendering = $state(false);
  let renderProgress = $state(0);

  let dims = $derived(getSpatialDims(shape));

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

  const BLOCKS_PER_ROW = 4;
  const DENSITY_BINS = 32;
  const CHANNELS_PER_FRAME = 16;

  const coolwarmLUT = COLORMAPS.coolwarm;
  const viridisLUT = COLORMAPS.viridis;

  // Pre-compute power-0.4 LUT: maps 0..255 -> 0..255
  const POW04_LUT = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    POW04_LUT[i] = (Math.pow(i / 255, 0.4) * 255 + 0.5) | 0;
  }

  $effect(() => {
    if (!canvas) return;
    const C = dims.channels;
    const H = dims.height;
    const W = dims.width;
    if (C === 0 || H === 0 || W === 0) return;

    const _containerWidth = containerWidth;

    const gap = 2;
    const blockGap = 8;
    const labelH = 16;
    const panelW = Math.max(16, ((_containerWidth - (BLOCKS_PER_ROW - 1) * blockGap) / BLOCKS_PER_ROW / 2) | 0);
    const panelH = Math.max(16, (panelW * H / W + 0.5) | 0);
    const blockW = panelW * 2 + gap;
    const blockH = panelH * 2 + gap + labelH;
    const rows = Math.ceil(C / BLOCKS_PER_ROW);
    const totalW = BLOCKS_PER_ROW * blockW + (BLOCKS_PER_ROW - 1) * blockGap;
    const totalH = rows * blockH + (rows - 1) * blockGap;

    canvas.width = totalW;
    canvas.height = totalH;
    canvas.style.width = `${totalW}px`;
    canvas.style.height = `${totalH}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, totalW, totalH);

    // Global stats — single pass
    let globalDiffMax = 0;
    for (let i = 0; i < main.length; i++) {
      const d = main[i] - ref[i];
      const ad = d < 0 ? -d : d;
      if (ad > globalDiffMax) globalDiffMax = ad;
    }
    if (globalDiffMax === 0) globalDiffMax = 1;
    const invDiffMax = 1 / globalDiffMax;

    const globalRefStats = computeStats(ref);
    const globalMainStats = computeStats(main);
    const grayMin = Math.min(globalRefStats.min, globalMainStats.min);
    const graySpan = Math.max(globalRefStats.max, globalMainStats.max) - grayMin || 1;
    const invGraySpan = 255 / graySpan;

    const mainMin = globalMainStats.min;
    const mainSpan = globalMainStats.max - mainMin || 1;
    const invMainSpan = (DENSITY_BINS - 1) / mainSpan;
    const invDiffBins = (DENSITY_BINS - 1) / globalDiffMax;

    // Precompute sample maps: for each output pixel, which source pixel to read
    const xMap = new Int32Array(panelW);
    const yMap = new Int32Array(panelH);
    for (let x = 0; x < panelW; x++) xMap[x] = (x * W / panelW) | 0;
    for (let y = 0; y < panelH; y++) yMap[y] = (y * H / panelH) | 0;

    // Reusable buffers at panel resolution (not tensor resolution!)
    const panelSize = panelW * panelH;
    const panelImgData = ctx.createImageData(panelW, panelH);
    const panelPx = panelImgData.data;

    const DB2 = DENSITY_BINS * DENSITY_BINS;
    const densityBuf = new Float32Array(DB2);
    const densityImgData = ctx.createImageData(DENSITY_BINS, DENSITY_BINS);
    const densityPx = densityImgData.data;

    // Offscreen for density (needs scaling from DENSITY_BINS to panelW)
    const offDensity = new OffscreenCanvas(DENSITY_BINS, DENSITY_BINS);
    const offDensityCtx = offDensity.getContext('2d')!;

    rendering = true;
    renderProgress = 0;
    let cancelled = false;
    let frameId = 0;

    function renderBatch(startCh: number) {
      if (cancelled) return;
      const endCh = Math.min(startCh + CHANNELS_PER_FRAME, C);

      for (let c = startCh; c < endCh; c++) {
        const row = (c / BLOCKS_PER_ROW) | 0;
        const col = c % BLOCKS_PER_ROW;
        const bx = col * (blockW + blockGap);
        const by = row * (blockH + blockGap);
        const panelY = by + labelH;

        const refSlice = extractSlice(ref, shape, 0, c).data;
        const mainSlice = extractSlice(main, shape, 0, c).data;

        // Top-left: Ref grayscale — render at panel resolution via nearest-neighbor
        for (let py = 0; py < panelH; py++) {
          const srcRow = yMap[py] * W;
          const dstRow = py * panelW;
          for (let px = 0; px < panelW; px++) {
            const g = ((refSlice[srcRow + xMap[px]] - grayMin) * invGraySpan) | 0;
            const b4 = (dstRow + px) << 2;
            panelPx[b4] = g;
            panelPx[b4 + 1] = g;
            panelPx[b4 + 2] = g;
            panelPx[b4 + 3] = 255;
          }
        }
        ctx.putImageData(panelImgData, bx, panelY);

        // Top-right: Main grayscale
        for (let py = 0; py < panelH; py++) {
          const srcRow = yMap[py] * W;
          const dstRow = py * panelW;
          for (let px = 0; px < panelW; px++) {
            const g = ((mainSlice[srcRow + xMap[px]] - grayMin) * invGraySpan) | 0;
            const b4 = (dstRow + px) << 2;
            panelPx[b4] = g;
            panelPx[b4 + 1] = g;
            panelPx[b4 + 2] = g;
            panelPx[b4 + 3] = 255;
          }
        }
        ctx.putImageData(panelImgData, bx + panelW + gap, panelY);

        // Bottom-left: Signed diff with coolwarm
        for (let py = 0; py < panelH; py++) {
          const srcRow = yMap[py] * W;
          const dstRow = py * panelW;
          for (let px = 0; px < panelW; px++) {
            const si = srcRow + xMap[px];
            const norm = (mainSlice[si] - refSlice[si]) * invDiffMax * 127.5 + 127.5;
            const idx = (norm < 0 ? 0 : norm > 255 ? 255 : norm | 0) * 3;
            const b4 = (dstRow + px) << 2;
            panelPx[b4] = coolwarmLUT[idx];
            panelPx[b4 + 1] = coolwarmLUT[idx + 1];
            panelPx[b4 + 2] = coolwarmLUT[idx + 2];
            panelPx[b4 + 3] = 255;
          }
        }
        ctx.putImageData(panelImgData, bx, panelY + panelH + gap);

        // Bottom-right: 2D density histogram
        densityBuf.fill(0);
        // Accumulate at full resolution for accuracy
        const chSize = H * W;
        for (let i = 0; i < chSize; i++) {
          const mv = mainSlice[i];
          const d = mv - refSlice[i];
          let xi = ((mv - mainMin) * invMainSpan) | 0;
          let yi = ((d < 0 ? -d : d) * invDiffBins) | 0;
          if (xi >= DENSITY_BINS) xi = DENSITY_BINS - 1;
          if (yi >= DENSITY_BINS) yi = DENSITY_BINS - 1;
          densityBuf[(DENSITY_BINS - 1 - yi) * DENSITY_BINS + xi]++;
        }

        let maxDensity = 1;
        for (let i = 0; i < DB2; i++) {
          if (densityBuf[i] > maxDensity) maxDensity = densityBuf[i];
        }
        const invMaxDensity = 255 / maxDensity;

        for (let i = 0; i < DB2; i++) {
          // Use precomputed pow(0.4) LUT
          const lin = (densityBuf[i] * invMaxDensity) | 0;
          const idx = POW04_LUT[lin > 255 ? 255 : lin] * 3;
          const b4 = i << 2;
          densityPx[b4] = viridisLUT[idx];
          densityPx[b4 + 1] = viridisLUT[idx + 1];
          densityPx[b4 + 2] = viridisLUT[idx + 2];
          densityPx[b4 + 3] = 255;
        }
        offDensityCtx.putImageData(densityImgData, 0, 0);
        ctx.drawImage(offDensity, bx + panelW + gap, panelY + panelH + gap, panelW, panelH);

        // Labels
        ctx.fillStyle = '#9ca3af';
        ctx.font = '11px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(`Ch ${c}`, bx + blockW / 2, by);

        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = '9px monospace';
        ctx.textAlign = 'left';
        ctx.fillText('ref', bx + 2, panelY + 2);
        ctx.fillText('main', bx + panelW + gap + 2, panelY + 2);
        ctx.fillText('diff', bx + 2, panelY + panelH + gap + 2);
        ctx.fillText('density', bx + panelW + gap + 2, panelY + panelH + gap + 2);
      }

      renderProgress = ((endCh / C) * 100 + 0.5) | 0;

      if (endCh < C) {
        frameId = requestAnimationFrame(() => renderBatch(endCh));
      } else {
        rendering = false;
      }
    }

    frameId = requestAnimationFrame(() => renderBatch(0));

    return () => {
      cancelled = true;
      cancelAnimationFrame(frameId);
    };
  });
</script>

<div bind:this={container} class="w-full">
  {#if rendering}
    <div class="text-xs text-gray-400 mb-2">Rendering channels... {renderProgress}%</div>
  {/if}
  <canvas bind:this={canvas} class="block mx-auto"></canvas>
</div>
