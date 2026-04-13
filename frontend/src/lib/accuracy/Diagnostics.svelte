<script lang="ts">
  import DensityPlot from './DensityPlot.svelte';
  import { computeChannelMetrics, type ChannelMetrics } from './diagRenderer';
  import { getSpatialDims, formatValue, ALL_COLORMAP_OPTIONS, type ColormapName } from './tensorUtils';
  import { rangeScroll } from './rangeScroll';

  let { main, ref, shape, mainLabel = 'Main', refLabel = 'Reference' }: {
    main: Float32Array;
    ref: Float32Array;
    shape: number[];
    mainLabel?: string;
    refLabel?: string;
  } = $props();

  let cols = $state(4);
  let signedDensity = $state(true);
  let cmapDensity: ColormapName = $state('viridis');
  let densityBins = $state(64);
  let batch = $state(0);
  let sortBy = $state<'index' | 'cosSim' | 'meanDiff' | 'maxDiff'>('index');

  const BIN_OPTIONS = [32, 64, 128, 256] as const;

  let dims = $derived(getSpatialDims(shape));

  function extractChannel(tensor: Float32Array, ch: number): Float32Array {
    const C = dims.channels, H = dims.height, W = dims.width;
    const chSize = H * W;
    const batchStride = C * H * W;
    const offset = batch * batchStride + ch * chSize;
    return tensor.subarray(offset, offset + chSize);
  }

  let allMetrics = $derived(computeChannelMetrics(main, ref, shape, batch));

  let channelOrder = $derived.by((): number[] => {
    if (sortBy === 'index') return Array.from({ length: dims.channels }, (_, i) => i);
    const sorted = [...allMetrics].sort((a, b) => {
      switch (sortBy) {
        case 'cosSim': return a.cosSim - b.cosSim;
        case 'meanDiff': return b.meanAbsDiff - a.meanAbsDiff;
        case 'maxDiff': return b.maxAbsDiff - a.maxAbsDiff;
        default: return a.ch - b.ch;
      }
    });
    return sorted.map(m => m.ch);
  });

  let metricsMap = $derived.by((): Map<number, ChannelMetrics> => {
    const map = new Map<number, ChannelMetrics>();
    for (const m of allMetrics) map.set(m.ch, m);
    return map;
  });

  function cosColor(cos: number): string {
    if (cos > 0.999) return 'text-green-400';
    if (cos > 0.99) return 'text-yellow-400';
    return 'text-red-400';
  }
</script>

<div class="w-full h-full flex flex-col gap-2">
  <!-- Controls -->
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
      Columns
      <input use:rangeScroll type="range" min="1" max="8" bind:value={cols} class="w-20" />
      <span class="w-4 text-center">{cols}</span>
    </label>

    <span class="border-l border-gray-600 h-4"></span>

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
      Colormap
      <select bind:value={cmapDensity} class="bg-gray-800 border border-gray-600 rounded px-1.5 py-0.5 text-xs text-gray-200">
        {#each ALL_COLORMAP_OPTIONS as opt}
          <option value={opt.value}>{opt.label}</option>
        {/each}
      </select>
    </label>

    <label class="flex items-center gap-1">
      Bins
      <select bind:value={densityBins} class="bg-gray-800 border border-gray-600 rounded px-1.5 py-0.5 text-xs text-gray-200">
        {#each BIN_OPTIONS as b}
          <option value={b}>{b}</option>
        {/each}
      </select>
    </label>

    <span class="border-l border-gray-600 h-4"></span>

    <label class="flex items-center gap-1.5 text-gray-400">
      <input type="checkbox" bind:checked={signedDensity} /> Signed
    </label>

    <span class="text-gray-500 ml-auto">{dims.channels} channels | {dims.height}&times;{dims.width}</span>
  </div>

  <!-- Grid -->
  <div class="flex-1 min-h-0 overflow-y-auto">
    <div class="grid gap-3" style="grid-template-columns: repeat({cols}, minmax(0, 1fr));">
      {#each channelOrder as ch (ch)}
        {@const m = metricsMap.get(ch)}
        <div class="bg-surface-base/50 rounded-lg p-2 flex flex-col gap-1">
          <div class="flex items-center justify-between text-[10px] shrink-0">
            <span class="font-medium text-gray-300">Ch {ch}</span>
            {#if m}
              <span class={cosColor(m.cosSim)}>cos={m.cosSim.toFixed(4)}</span>
            {/if}
          </div>
          {#if m}
            <div class="flex gap-2 text-[9px] text-gray-500 shrink-0">
              <span>MSE={formatValue(m.meanAbsDiff)}</span>
              <span>max={formatValue(m.maxAbsDiff)}</span>
            </div>
          {/if}
          <div class="flex-1 min-h-[180px]">
            <DensityPlot
              main={extractChannel(main, ch)}
              ref={extractChannel(ref, ch)}
              signed={signedDensity}
              colormap={cmapDensity}
              bins={densityBins}
            />
          </div>
        </div>
      {/each}
    </div>
  </div>
</div>
