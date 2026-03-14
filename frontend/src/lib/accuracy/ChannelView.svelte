<script lang="ts">
  import { getSpatialDims, extractSlice, computeStats, computeHistogram, cosineSimilarity, formatValue } from './tensorUtils';
  import { rangeScroll } from './rangeScroll';

  let {
    main,
    ref,
    shape,
    mainLabel = 'Main',
    refLabel = 'Reference',
  }: {
    main: Float32Array;
    ref: Float32Array;
    shape: number[];
    mainLabel?: string;
    refLabel?: string;
  } = $props();

  let dims = $derived(getSpatialDims(shape));

  function getChannelData(tensor: Float32Array, ch: number): Float32Array {
    return extractSlice(tensor, shape, 0, ch).data;
  }

  let channelIndex = $state(0);

  let mainChannel = $derived(getChannelData(main, channelIndex));
  let refChannel = $derived(getChannelData(ref, channelIndex));

  let mainStats = $derived(computeStats(mainChannel));
  let refStats = $derived(computeStats(refChannel));

  let diffChannel = $derived.by(() => {
    const diff = new Float32Array(mainChannel.length);
    for (let i = 0; i < mainChannel.length; i++) {
      diff[i] = Math.abs(mainChannel[i] - refChannel[i]);
    }
    return diff;
  });
  let diffStats = $derived(computeStats(diffChannel));

  let cosSim = $derived(cosineSimilarity(mainChannel, refChannel));

  // Tri-histogram canvases for the stat cards
  let histMain: HTMLCanvasElement;
  let histRef: HTMLCanvasElement;
  let histDiff: HTMLCanvasElement;

  // Compute shared range across ref+main+diff for the selected channel
  let histRange = $derived.by((): [number, number] => {
    let lo = Infinity, hi = -Infinity;
    for (const arr of [mainChannel, refChannel, diffChannel]) {
      for (let i = 0; i < arr.length; i++) {
        if (arr[i] < lo) lo = arr[i];
        if (arr[i] > hi) hi = arr[i];
      }
    }
    if (lo === Infinity) { lo = 0; hi = 1; }
    if (lo === hi) { lo -= 0.5; hi += 0.5; }
    return [lo, hi];
  });

  let mainHist = $derived(computeHistogram(mainChannel, 48, histRange));
  let refHist = $derived(computeHistogram(refChannel, 48, histRange));
  let diffHist = $derived(computeHistogram(diffChannel, 48, histRange));

  function drawTriHistogram(
    canvas: HTMLCanvasElement | undefined,
    emphasis: 'main' | 'ref' | 'diff',
  ) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const pad = { top: 4, right: 4, bottom: 18, left: 4 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Find max count across all three
    let maxCount = 1;
    for (const hist of [refHist, mainHist, diffHist]) {
      for (let i = 0; i < hist.counts.length; i++) {
        if (hist.counts[i] > maxCount) maxCount = hist.counts[i];
      }
    }

    const series: { hist: typeof mainHist; color: string; key: 'main' | 'ref' | 'diff' }[] = [
      { hist: refHist, color: 'rgba(59,130,246,0.35)', key: 'ref' },
      { hist: mainHist, color: 'rgba(34,197,94,0.35)', key: 'main' },
      { hist: diffHist, color: 'rgba(239,68,68,0.3)', key: 'diff' },
    ];

    // Draw emphasized series last (on top) with higher opacity
    const emphasisColors: Record<string, string> = {
      ref: 'rgba(59,130,246,0.7)',
      main: 'rgba(34,197,94,0.7)',
      diff: 'rgba(239,68,68,0.6)',
    };

    // Sort: non-emphasis first, emphasis last
    const sorted = [...series].sort((a, b) => {
      if (a.key === emphasis) return 1;
      if (b.key === emphasis) return -1;
      return 0;
    });

    const bins = mainHist.counts.length;
    const barW = plotW / bins;

    for (const s of sorted) {
      const isEmphasis = s.key === emphasis;
      ctx.fillStyle = isEmphasis ? emphasisColors[s.key] : s.color;
      for (let i = 0; i < s.hist.counts.length; i++) {
        const count = s.hist.counts[i];
        if (count === 0) continue;
        const barH = (count / maxCount) * plotH;
        const x = pad.left + (i / bins) * plotW;
        ctx.fillRect(x, pad.top + plotH - barH, barW, barH);
      }
    }

    // Mini x-axis
    ctx.strokeStyle = '#4b5563';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // Tick labels
    const [lo, hi] = histRange;
    ctx.fillStyle = '#6b7280';
    ctx.font = '9px monospace';
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    ctx.fillText(formatValue(lo), pad.left, pad.top + plotH + 3);
    ctx.textAlign = 'right';
    ctx.fillText(formatValue(hi), pad.left + plotW, pad.top + plotH + 3);
  }

  $effect(() => {
    mainChannel; refChannel; diffChannel;
    mainHist; refHist; diffHist;
    drawTriHistogram(histMain, 'main');
    drawTriHistogram(histRef, 'ref');
    drawTriHistogram(histDiff, 'diff');
  });

  // Sort state
  type SortKey = 'ch' | 'meanDiff' | 'maxDiff' | 'cosSim';
  let sortKey = $state<SortKey>('ch');
  let sortAsc = $state(true);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      sortAsc = !sortAsc;
    } else {
      sortKey = key;
      sortAsc = key === 'ch' || key === 'cosSim'; // default asc for ch/cosSim, but we set true here and let user toggle
    }
  }

  // Per-channel summary table — all channels, no cap
  let channelSummary = $derived.by(() => {
    const summary: { ch: number; meanDiff: number; maxDiff: number; cosSim: number }[] = [];
    for (let c = 0; c < dims.channels; c++) {
      const m = getChannelData(main, c);
      const r = getChannelData(ref, c);
      let sumDiff = 0, maxD = 0;
      for (let i = 0; i < m.length; i++) {
        const d = Math.abs(m[i] - r[i]);
        sumDiff += d;
        if (d > maxD) maxD = d;
      }
      summary.push({
        ch: c,
        meanDiff: sumDiff / m.length,
        maxDiff: maxD,
        cosSim: cosineSimilarity(m, r),
      });
    }
    return summary;
  });

  let sortedSummary = $derived.by(() => {
    const sorted = [...channelSummary];
    const dir = sortAsc ? 1 : -1;
    sorted.sort((a, b) => (a[sortKey] - b[sortKey]) * dir);
    return sorted;
  });

  // Compute column-wise min/max for color scaling
  let columnRanges = $derived.by(() => {
    if (channelSummary.length === 0) return { meanDiff: [0, 1], maxDiff: [0, 1], cosSim: [0, 1] };
    let meanMin = Infinity, meanMax = -Infinity;
    let maxMin = Infinity, maxMax = -Infinity;
    let cosMin = Infinity, cosMax = -Infinity;
    for (const row of channelSummary) {
      if (row.meanDiff < meanMin) meanMin = row.meanDiff;
      if (row.meanDiff > meanMax) meanMax = row.meanDiff;
      if (row.maxDiff < maxMin) maxMin = row.maxDiff;
      if (row.maxDiff > maxMax) maxMax = row.maxDiff;
      if (row.cosSim < cosMin) cosMin = row.cosSim;
      if (row.cosSim > cosMax) cosMax = row.cosSim;
    }
    return {
      meanDiff: [meanMin, meanMax] as [number, number],
      maxDiff: [maxMin, maxMax] as [number, number],
      cosSim: [cosMin, cosMax] as [number, number],
    };
  });

  function errorColor(value: number, range: [number, number]): string {
    const [lo, hi] = range;
    const span = hi - lo;
    if (span === 0) return 'rgb(156, 163, 175)';
    const t = (value - lo) / span;
    const r = Math.round(55 + t * 200);
    const g = Math.round(200 - t * 160);
    const b = Math.round(50);
    return `rgb(${r}, ${g}, ${b})`;
  }

  function cosSimColor(value: number, range: [number, number]): string {
    const [lo, hi] = range;
    const span = hi - lo;
    if (span === 0) return 'rgb(156, 163, 175)';
    const t = (value - lo) / span;
    const r = Math.round(220 - t * 170);
    const g = Math.round(60 + t * 150);
    const b = Math.round(50);
    return `rgb(${r}, ${g}, ${b})`;
  }
</script>

<div class="h-full flex flex-col gap-4">
  <!-- Channel selector -->
  {#if dims.channels > 1}
    <div class="flex items-center gap-2 text-xs shrink-0 w-full">
      <span class="text-gray-400 shrink-0">Channel:</span>
      <input
        use:rangeScroll
        type="range"
        min="0"
        max={dims.channels - 1}
        bind:value={channelIndex}
        class="flex-1"
      />
      <span class="text-gray-300 shrink-0">{channelIndex} / {dims.channels}</span>
    </div>
  {/if}

  <!-- Stats comparison with tri-histograms -->
  <div class="grid grid-cols-3 gap-4 shrink-0">
    <!-- Main Device -->
    <div class="bg-surface-base/50 rounded-lg p-3">
      <h4 class="text-xs font-medium text-blue-400 mb-2">{mainLabel}</h4>
      <div class="space-y-1 text-xs">
        <div class="flex justify-between"><span class="text-gray-500">Min</span><span class="font-mono text-gray-300">{formatValue(mainStats.min)}</span></div>
        <div class="flex justify-between"><span class="text-gray-500">Max</span><span class="font-mono text-gray-300">{formatValue(mainStats.max)}</span></div>
        <div class="flex justify-between"><span class="text-gray-500">Mean</span><span class="font-mono text-gray-300">{formatValue(mainStats.mean)}</span></div>
        <div class="flex justify-between"><span class="text-gray-500">Std</span><span class="font-mono text-gray-300">{formatValue(mainStats.std)}</span></div>
      </div>
      <canvas bind:this={histMain} class="w-full h-[120px] mt-2 rounded"></canvas>
    </div>

    <!-- Ref Device -->
    <div class="bg-surface-base/50 rounded-lg p-3">
      <h4 class="text-xs font-medium text-green-400 mb-2">{refLabel}</h4>
      <div class="space-y-1 text-xs">
        <div class="flex justify-between"><span class="text-gray-500">Min</span><span class="font-mono text-gray-300">{formatValue(refStats.min)}</span></div>
        <div class="flex justify-between"><span class="text-gray-500">Max</span><span class="font-mono text-gray-300">{formatValue(refStats.max)}</span></div>
        <div class="flex justify-between"><span class="text-gray-500">Mean</span><span class="font-mono text-gray-300">{formatValue(refStats.mean)}</span></div>
        <div class="flex justify-between"><span class="text-gray-500">Std</span><span class="font-mono text-gray-300">{formatValue(refStats.std)}</span></div>
      </div>
      <canvas bind:this={histRef} class="w-full h-[120px] mt-2 rounded"></canvas>
    </div>

    <!-- Abs Diff -->
    <div class="bg-surface-base/50 rounded-lg p-3">
      <h4 class="text-xs font-medium text-red-400 mb-2">Abs Diff</h4>
      <div class="space-y-1 text-xs">
        <div class="flex justify-between"><span class="text-gray-500">Min</span><span class="font-mono text-gray-300">{formatValue(diffStats.min)}</span></div>
        <div class="flex justify-between"><span class="text-gray-500">Max</span><span class="font-mono text-gray-300">{formatValue(diffStats.max)}</span></div>
        <div class="flex justify-between"><span class="text-gray-500">Mean</span><span class="font-mono text-gray-300">{formatValue(diffStats.mean)}</span></div>
        <div class="flex justify-between"><span class="text-gray-500">Std</span><span class="font-mono text-gray-300">{formatValue(diffStats.std)}</span></div>
      </div>
      <canvas bind:this={histDiff} class="w-full h-[120px] mt-2 rounded"></canvas>
    </div>
  </div>

  <!-- Cosine similarity -->
  <div class="text-xs text-gray-400 shrink-0">
    Cosine similarity (ch {channelIndex}): <span class="font-mono text-gray-200">{cosSim.toFixed(6)}</span>
    | Spatial size: {dims.height * dims.width} elements
  </div>

  <!-- Per-channel summary table — fills remaining space -->
  {#if dims.channels > 1}
    <div class="flex-1 min-h-0 flex flex-col">
      <h4 class="text-xs font-medium text-gray-400 mb-1 shrink-0">Per-Channel Summary ({dims.channels} channels)</h4>
      <div class="flex-1 min-h-0 overflow-y-auto rounded border border-edge">
        <table class="w-full text-xs">
          <thead class="sticky top-0 bg-surface-panel">
            <tr class="text-gray-500 border-b border-edge">
              <th class="px-2 py-1 text-left font-medium cursor-pointer hover:text-gray-300 select-none" onclick={() => toggleSort('ch')}>Ch{sortKey === 'ch' ? (sortAsc ? ' ▲' : ' ▼') : ''}</th>
              <th class="px-2 py-1 text-right font-medium cursor-pointer hover:text-gray-300 select-none" onclick={() => toggleSort('meanDiff')}>Mean Diff{sortKey === 'meanDiff' ? (sortAsc ? ' ▲' : ' ▼') : ''}</th>
              <th class="px-2 py-1 text-right font-medium cursor-pointer hover:text-gray-300 select-none" onclick={() => toggleSort('maxDiff')}>Max Diff{sortKey === 'maxDiff' ? (sortAsc ? ' ▲' : ' ▼') : ''}</th>
              <th class="px-2 py-1 text-right font-medium cursor-pointer hover:text-gray-300 select-none" onclick={() => toggleSort('cosSim')}>Cos Sim{sortKey === 'cosSim' ? (sortAsc ? ' ▲' : ' ▼') : ''}</th>
            </tr>
          </thead>
          <tbody>
            {#each sortedSummary as row (row.ch)}
              <tr
                class="border-b border-edge/50 cursor-pointer hover:bg-white/5 {row.ch === channelIndex ? 'bg-white/10' : ''}"
                onclick={() => channelIndex = row.ch}
              >
                <td class="px-2 py-1 font-mono text-gray-300">{row.ch}</td>
                <td class="px-2 py-1 text-right font-mono" style="color: {errorColor(row.meanDiff, columnRanges.meanDiff)}">{formatValue(row.meanDiff)}</td>
                <td class="px-2 py-1 text-right font-mono" style="color: {errorColor(row.maxDiff, columnRanges.maxDiff)}">{formatValue(row.maxDiff)}</td>
                <td class="px-2 py-1 text-right font-mono" style="color: {cosSimColor(row.cosSim, columnRanges.cosSim)}">{row.cosSim.toFixed(4)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </div>
  {/if}
</div>
