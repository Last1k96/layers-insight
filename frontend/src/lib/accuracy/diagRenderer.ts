// Optimized diagnostics renderer — all channels in a single putImageData call.
// Uses Uint32 LUTs, merged loops, inline density, and batched labels.

import { getSpatialDims, extractSlice, computeStats, COLORMAPS } from './tensorUtils';

const BLOCKS_PER_ROW = 4;
const DENSITY_BINS = 64;
const DB2 = DENSITY_BINS * DENSITY_BINS;

// ---------------------------------------------------------------------------
// Uint32 LUTs — pack RGBA into single 32-bit value (little-endian: 0xAABBGGRR)
// ---------------------------------------------------------------------------

function buildU32LUT(lut3: Uint8Array): Uint32Array {
  const out = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    const b = i * 3;
    out[i] = 0xFF000000 | (lut3[b + 2] << 16) | (lut3[b + 1] << 8) | lut3[b];
  }
  return out;
}

const coolwarmU32 = buildU32LUT(COLORMAPS.coolwarm);
const viridisU32 = buildU32LUT(COLORMAPS.viridis);

// Grayscale Uint32 LUT
const grayU32 = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  grayU32[i] = 0xFF000000 | (i << 16) | (i << 8) | i;
}

// Pre-compute power-0.4 LUT: maps 0..255 -> 0..255
const POW04_LUT = new Uint8Array(256);
for (let i = 0; i < 256; i++) {
  POW04_LUT[i] = (Math.pow(i / 255, 0.4) * 255 + 0.5) | 0;
}

// Background color #111827 in ABGR little-endian
const BG_U32 = 0xFF271811;

export interface DiagCanvas {
  width: number;
  height: number;
  style: { width: string; height: string };
}

export function renderDiagnostics(
  ctx: CanvasRenderingContext2D,
  canvas: DiagCanvas,
  main: Float32Array,
  ref: Float32Array,
  shape: number[],
  containerWidth: number,
  mainLabel = 'main',
  refLabel = 'ref',
): void {
  const dims = getSpatialDims(shape);
  const C = dims.channels;
  const H = dims.height;
  const W = dims.width;
  if (C === 0 || H === 0 || W === 0) return;

  const gap = 2;
  const blockGap = 8;
  const labelH = 16;
  const panelW = Math.max(16, ((containerWidth - (BLOCKS_PER_ROW - 1) * blockGap) / BLOCKS_PER_ROW / 2) | 0);
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

  // --- Global stats (single pass) ---
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

  // Precompute sample maps
  const xMap = new Int32Array(panelW);
  const yMap = new Int32Array(panelH);
  for (let x = 0; x < panelW; x++) xMap[x] = (x * W / panelW) | 0;
  for (let y = 0; y < panelH; y++) yMap[y] = (y * H / panelH) | 0;

  // --- Single full-canvas ImageData ---
  const imgData = new ImageData(totalW, totalH);
  const px32 = new Uint32Array(imgData.data.buffer);
  px32.fill(BG_U32);

  // Density upscale maps (for each pixel in panelW/panelH, which density bin)
  const dxMap = new Int32Array(panelW);
  const dyMap = new Int32Array(panelH);
  for (let x = 0; x < panelW; x++) dxMap[x] = (x * DENSITY_BINS / panelW) | 0;
  for (let y = 0; y < panelH; y++) dyMap[y] = (y * DENSITY_BINS / panelH) | 0;

  // Reusable density buffer
  const densityBuf = new Float32Array(DB2);

  // Label collection for batched text rendering
  const chLabels: { text: string; x: number; y: number }[] = [];
  const panelLabels: { text: string; x: number; y: number }[] = [];

  const chSize = H * W;
  const stride = chSize > 4096 ? ((chSize / 4096) | 0) || 1 : 1;

  for (let c = 0; c < C; c++) {
    const row = (c / BLOCKS_PER_ROW) | 0;
    const col = c % BLOCKS_PER_ROW;
    const bx = col * (blockW + blockGap);
    const by = row * (blockH + blockGap);
    const panelY = by + labelH;

    const refSlice = extractSlice(ref, shape, 0, c).data;
    const mainSlice = extractSlice(main, shape, 0, c).data;

    // --- Merged loop: ref gray + main gray + diff coolwarm ---
    const oxRef = bx;
    const oxMain = bx + panelW + gap;
    const oyTop = panelY;
    const oxDiff = bx;
    const oyBottom = panelY + panelH + gap;

    for (let py = 0; py < panelH; py++) {
      const srcRow = yMap[py] * W;
      const canvasRowTop = (oyTop + py) * totalW;
      const canvasRowBot = (oyBottom + py) * totalW;

      for (let px = 0; px < panelW; px++) {
        const si = srcRow + xMap[px];
        const rv = refSlice[si];
        const mv = mainSlice[si];

        // Ref grayscale
        let g = ((rv - grayMin) * invGraySpan) | 0;
        if (g < 0) g = 0; else if (g > 255) g = 255;
        px32[canvasRowTop + oxRef + px] = grayU32[g];

        // Main grayscale
        g = ((mv - grayMin) * invGraySpan) | 0;
        if (g < 0) g = 0; else if (g > 255) g = 255;
        px32[canvasRowTop + oxMain + px] = grayU32[g];

        // Diff coolwarm
        let norm = (mv - rv) * invDiffMax * 127.5 + 127.5;
        let ci = norm | 0;
        if (ci < 0) ci = 0; else if (ci > 255) ci = 255;
        px32[canvasRowBot + oxDiff + px] = coolwarmU32[ci];
      }
    }

    // --- Density histogram (inline) ---
    densityBuf.fill(0);
    for (let i = 0; i < chSize; i += stride) {
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

    // Pre-color density bins into Uint32
    const densityColored = new Uint32Array(DB2);
    for (let i = 0; i < DB2; i++) {
      const lin = (densityBuf[i] * invMaxDensity) | 0;
      densityColored[i] = viridisU32[POW04_LUT[lin > 255 ? 255 : lin]];
    }

    // Nearest-neighbor upscale directly into full ImageData
    const oxDensity = bx + panelW + gap;
    const oyDensity = panelY + panelH + gap;

    for (let py = 0; py < panelH; py++) {
      const srcBinRow = dyMap[py] * DENSITY_BINS;
      const canvasRow = (oyDensity + py) * totalW + oxDensity;
      for (let px = 0; px < panelW; px++) {
        px32[canvasRow + px] = densityColored[srcBinRow + dxMap[px]];
      }
    }

    // Collect labels
    chLabels.push({ text: `Ch ${c}`, x: bx + blockW / 2, y: by });
    panelLabels.push(
      { text: refLabel, x: bx + 2, y: panelY + 2 },
      { text: mainLabel, x: oxMain + 2, y: panelY + 2 },
      { text: 'diff', x: bx + 2, y: oyBottom + 2 },
      { text: 'density', x: oxDensity + 2, y: oyDensity + 2 },
    );
  }

  // --- Single putImageData ---
  ctx.putImageData(imgData, 0, 0);

  // --- Batched labels ---
  ctx.fillStyle = '#9ca3af';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const l of chLabels) {
    ctx.fillText(l.text, l.x, l.y);
  }

  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  for (const l of panelLabels) {
    ctx.fillText(l.text, l.x, l.y);
  }
}
