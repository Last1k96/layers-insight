// Optimized diagnostics renderer — all channels in a single putImageData call.
// Uses Uint32 LUTs, merged loops, inline density, and batched labels.

import { getSpatialDims, extractSlice, computeStats, COLORMAPS, type ColormapName } from './tensorUtils';

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

export interface ChannelMetrics {
  ch: number;
  cosSim: number;
  meanAbsDiff: number;
  maxAbsDiff: number;
}

export interface DiagLayout {
  panelW: number;
  panelH: number;
  blocksPerRow: number;
  channelCount: number;
  blockW: number;
  blockH: number;
  blockGap: number;
  labelH: number;
  gap: number;
  pad: number;
  channelMetrics: ChannelMetrics[];
  channelOrder: number[];
  xMap: Int32Array;
  yMap: Int32Array;
  grayMin: number;
  graySpan: number;
  globalDiffMax: number;
  mainMin: number;
  mainSpan: number;
  H: number;
  W: number;
}

export interface DiagOptions {
  colormaps?: { gray?: ColormapName; diff?: ColormapName; density?: ColormapName };
  signedDensity?: boolean;
  channelOrder?: number[];
  batch?: number;
  highlightWorst?: boolean;
}

/**
 * Compute per-channel metrics without rendering.
 * Used to compute sort order before calling renderDiagnostics.
 */
export function computeChannelMetrics(
  main: Float32Array,
  ref: Float32Array,
  shape: number[],
  batch = 0,
): ChannelMetrics[] {
  const dims = getSpatialDims(shape);
  const C = dims.channels;
  const H = dims.height;
  const W = dims.width;
  if (C === 0 || H === 0 || W === 0) return [];

  const metrics: ChannelMetrics[] = [];
  for (let c = 0; c < C; c++) {
    const refSlice = extractSlice(ref, shape, batch, c).data;
    const mainSlice = extractSlice(main, shape, batch, c).data;
    const n = refSlice.length;

    let dot = 0, normR = 0, normM = 0, sumAbsDiff = 0, maxAbs = 0;
    for (let i = 0; i < n; i++) {
      const rv = refSlice[i], mv = mainSlice[i];
      dot += rv * mv;
      normR += rv * rv;
      normM += mv * mv;
      const ad = Math.abs(mv - rv);
      sumAbsDiff += ad;
      if (ad > maxAbs) maxAbs = ad;
    }
    const denom = Math.sqrt(normR) * Math.sqrt(normM);
    metrics.push({
      ch: c,
      cosSim: denom > 0 ? dot / denom : 0,
      meanAbsDiff: n > 0 ? sumAbsDiff / n : 0,
      maxAbsDiff: maxAbs,
    });
  }
  return metrics;
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
  options: DiagOptions = {},
): DiagLayout | null {
  const {
    colormaps,
    signedDensity = true,
    channelOrder: orderParam,
    batch = 0,
    highlightWorst = true,
  } = options;

  const dims = getSpatialDims(shape);
  const C = dims.channels;
  const H = dims.height;
  const W = dims.width;
  if (C === 0 || H === 0 || W === 0) return null;

  // Build LUTs
  const grayLut = colormaps?.gray ? buildU32LUT(COLORMAPS[colormaps.gray]) : grayU32;
  const diffLut = colormaps?.diff ? buildU32LUT(COLORMAPS[colormaps.diff]) : coolwarmU32;
  const densityLut = colormaps?.density ? buildU32LUT(COLORMAPS[colormaps.density]) : viridisU32;

  const gap = 2;
  const blockGap = 8;
  const labelH = 26;
  const pad = 4; // padding around the entire grid to prevent label clipping

  // Responsive column count — subtract padding from available width
  const availW = containerWidth - pad * 2;
  const minBlockPx = 200;
  const blocksPerRow = Math.max(2, Math.min(8, Math.floor(availW / (minBlockPx + blockGap))));

  // panelW: account for inner gap between left/right panels, and blockGap between blocks
  const panelW = Math.max(16, ((availW - blocksPerRow * gap - (blocksPerRow - 1) * blockGap) / (blocksPerRow * 2)) | 0);
  const panelH = Math.max(16, (panelW * H / W + 0.5) | 0);
  const blockW = panelW * 2 + gap;
  const blockH = panelH * 2 + gap + labelH;

  // Channel ordering
  const order = orderParam ?? Array.from({ length: C }, (_, i) => i);
  const displayCount = order.length;
  const rows = Math.ceil(displayCount / blocksPerRow);
  const totalW = blocksPerRow * blockW + (blocksPerRow - 1) * blockGap + pad * 2;
  const totalH = rows * blockH + (rows - 1) * blockGap + pad * 2;

  canvas.width = totalW;
  canvas.height = totalH;
  canvas.style.width = `${totalW}px`;
  canvas.style.height = `${totalH}px`;

  // --- Global stats (single pass over selected batch) ---
  const batchStride = C * H * W;
  const batchOffset = batch * batchStride;
  const batchEnd = Math.min(batchOffset + batchStride, main.length);

  let globalDiffMax = 0;
  let grayMin = Infinity, grayMax = -Infinity;
  let mainMin = Infinity, mainMax = -Infinity;

  for (let i = batchOffset; i < batchEnd; i++) {
    const rv = ref[i], mv = main[i];
    const d = mv - rv;
    const ad = d < 0 ? -d : d;
    if (ad > globalDiffMax) globalDiffMax = ad;
    if (rv < grayMin) grayMin = rv;
    if (rv > grayMax) grayMax = rv;
    if (mv < grayMin) grayMin = mv;
    if (mv > grayMax) grayMax = mv;
    if (mv < mainMin) mainMin = mv;
    if (mv > mainMax) mainMax = mv;
  }
  if (globalDiffMax === 0) globalDiffMax = 1;
  const graySpan = grayMax - grayMin || 1;
  const mainSpan = mainMax - mainMin || 1;
  const invDiffMax = 1 / globalDiffMax;
  const invGraySpan = 255 / graySpan;
  const invMainSpan = (DENSITY_BINS - 1) / mainSpan;
  const invDiffBins = (DENSITY_BINS - 1) / globalDiffMax;
  const halfBins = DENSITY_BINS / 2;
  const invDiffBinsSigned = (DENSITY_BINS / 2 - 1) / globalDiffMax;

  // Precompute sample maps
  const xMap = new Int32Array(panelW);
  const yMap = new Int32Array(panelH);
  for (let x = 0; x < panelW; x++) xMap[x] = (x * W / panelW) | 0;
  for (let y = 0; y < panelH; y++) yMap[y] = (y * H / panelH) | 0;

  // --- Single full-canvas ImageData ---
  const imgData = new ImageData(totalW, totalH);
  const px32 = new Uint32Array(imgData.data.buffer);
  px32.fill(BG_U32);

  // Density upscale maps
  const dxMap = new Int32Array(panelW);
  const dyMap = new Int32Array(panelH);
  for (let x = 0; x < panelW; x++) dxMap[x] = (x * DENSITY_BINS / panelW) | 0;
  for (let y = 0; y < panelH; y++) dyMap[y] = (y * DENSITY_BINS / panelH) | 0;

  // Reusable density buffer
  const densityBuf = new Float32Array(DB2);

  // Label collection for batched text rendering
  const chLabels: { text: string; x: number; y: number }[] = [];
  const statsLabels: { text: string; x: number; y: number; color: string }[] = [];
  const panelLabels: { text: string; x: number; y: number }[] = [];

  const chSize = H * W;
  const stride = chSize > 4096 ? ((chSize / 4096) | 0) || 1 : 1;

  // Per-channel metrics (computed during render)
  const channelMetrics: ChannelMetrics[] = [];

  for (let displayIdx = 0; displayIdx < displayCount; displayIdx++) {
    const c = order[displayIdx];
    const row = (displayIdx / blocksPerRow) | 0;
    const col = displayIdx % blocksPerRow;
    const bx = pad + col * (blockW + blockGap);
    const by = pad + row * (blockH + blockGap);
    const panelY = by + labelH;

    const refSlice = extractSlice(ref, shape, batch, c).data;
    const mainSlice = extractSlice(main, shape, batch, c).data;

    // --- Merged loop: ref gray + main gray + diff coolwarm + stats ---
    const oxRef = bx;
    const oxMain = bx + panelW + gap;
    const oyTop = panelY;
    const oxDiff = bx;
    const oyBottom = panelY + panelH + gap;

    let dot = 0, normR = 0, normM = 0, sumAbsDiff = 0, maxAbs = 0;

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
        px32[canvasRowTop + oxRef + px] = grayLut[g];

        // Main grayscale
        g = ((mv - grayMin) * invGraySpan) | 0;
        if (g < 0) g = 0; else if (g > 255) g = 255;
        px32[canvasRowTop + oxMain + px] = grayLut[g];

        // Diff coolwarm
        let norm = (mv - rv) * invDiffMax * 127.5 + 127.5;
        let ci = norm | 0;
        if (ci < 0) ci = 0; else if (ci > 255) ci = 255;
        px32[canvasRowBot + oxDiff + px] = diffLut[ci];
      }
    }

    // Stats: full-resolution pass (not just sampled panel pixels)
    for (let i = 0; i < chSize; i++) {
      const rv = refSlice[i], mv = mainSlice[i];
      dot += rv * mv;
      normR += rv * rv;
      normM += mv * mv;
      const ad = Math.abs(mv - rv);
      sumAbsDiff += ad;
      if (ad > maxAbs) maxAbs = ad;
    }
    const denom = Math.sqrt(normR) * Math.sqrt(normM);
    const cosSim = denom > 0 ? dot / denom : 0;
    channelMetrics.push({
      ch: c,
      cosSim,
      meanAbsDiff: chSize > 0 ? sumAbsDiff / chSize : 0,
      maxAbsDiff: maxAbs,
    });

    // --- Density histogram (inline) ---
    densityBuf.fill(0);
    for (let i = 0; i < chSize; i += stride) {
      const mv = mainSlice[i];
      const d = mv - refSlice[i];
      let xi = ((mv - mainMin) * invMainSpan) | 0;
      let yi: number;
      if (signedDensity) {
        yi = ((d * invDiffBinsSigned) + halfBins) | 0;
      } else {
        yi = (((d < 0 ? -d : d) * invDiffBins) | 0);
        yi = DENSITY_BINS - 1 - yi; // flip so higher diffs at top
      }
      if (xi < 0) xi = 0; else if (xi >= DENSITY_BINS) xi = DENSITY_BINS - 1;
      if (yi < 0) yi = 0; else if (yi >= DENSITY_BINS) yi = DENSITY_BINS - 1;
      densityBuf[yi * DENSITY_BINS + xi]++;
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
      densityColored[i] = densityLut[POW04_LUT[lin > 255 ? 255 : lin]];
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
    const cosColor = cosSim > 0.999 ? '#4ade80' : cosSim > 0.99 ? '#facc15' : '#f87171';
    const maxStr = maxAbs < 0.0001 ? maxAbs.toExponential(1) : maxAbs.toFixed(4);
    statsLabels.push({ text: `cos=${cosSim.toFixed(4)}  max=${maxStr}`, x: bx + blockW / 2, y: by + 13, color: cosColor });
    const densLabel = signedDensity ? 'density \u00b1' : 'density |·|';
    panelLabels.push(
      { text: refLabel, x: bx + 2, y: panelY + 2 },
      { text: mainLabel, x: oxMain + 2, y: panelY + 2 },
      { text: 'diff', x: bx + 2, y: oyBottom + 2 },
      { text: densLabel, x: oxDensity + 2, y: oyDensity + 2 },
    );
  }

  // --- Single putImageData ---
  ctx.putImageData(imgData, 0, 0);

  // --- Worst-channel highlighting borders ---
  if (highlightWorst && channelMetrics.length > 1) {
    const sorted = [...channelMetrics].sort((a, b) => b.maxAbsDiff - a.maxAbsDiff);
    const top10 = sorted[Math.floor(sorted.length * 0.1)]?.maxAbsDiff ?? 0;
    const top25 = sorted[Math.floor(sorted.length * 0.25)]?.maxAbsDiff ?? 0;

    for (let displayIdx = 0; displayIdx < displayCount; displayIdx++) {
      const m = channelMetrics[displayIdx];
      let color: string | null = null;
      if (m.maxAbsDiff >= top10) color = '#f87171';
      else if (m.maxAbsDiff >= top25) color = '#facc15';

      if (color) {
        const row = (displayIdx / blocksPerRow) | 0;
        const col = displayIdx % blocksPerRow;
        const bx = pad + col * (blockW + blockGap);
        const by = pad + row * (blockH + blockGap);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(bx + 1, by + 1, blockW - 2, blockH - 2);
      }
    }
  }

  // --- Batched labels ---
  ctx.fillStyle = '#9ca3af';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const l of chLabels) {
    ctx.fillText(l.text, l.x, l.y);
  }

  // Stats labels (colored per channel)
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  for (const l of statsLabels) {
    ctx.fillStyle = l.color;
    ctx.fillText(l.text, l.x, l.y);
  }

  // Panel labels
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  for (const l of panelLabels) {
    ctx.fillText(l.text, l.x, l.y);
  }

  return {
    panelW, panelH, blocksPerRow, channelCount: C, blockW, blockH, blockGap, labelH, gap, pad,
    channelMetrics, channelOrder: order, xMap, yMap,
    grayMin, graySpan, globalDiffMax, mainMin, mainSpan: mainSpan,
    H, W,
  };
}
