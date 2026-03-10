<script lang="ts">
  let {
    main,
    ref,
    shape,
  }: {
    main: Float32Array;
    ref: Float32Array;
    shape: number[];
  } = $props();

  let channelIndex = $state(0);
  let numChannels = $derived(shape.length >= 4 ? shape[1] : shape.length >= 3 ? shape[0] : 1);
  let spatialSize = $derived(() => {
    if (shape.length >= 4) return shape[2] * shape[3];
    if (shape.length >= 3) return shape[1] * shape[2];
    if (shape.length >= 2) return shape[0] * shape[1];
    return main.length;
  });

  function getChannelData(tensor: Float32Array, ch: number): Float32Array {
    const size = spatialSize();
    const offset = ch * size;
    return tensor.slice(offset, offset + size);
  }

  function formatVal(v: number): string {
    if (Math.abs(v) < 0.0001 && v !== 0) return v.toExponential(3);
    return v.toFixed(4);
  }

  function channelStats(data: Float32Array): { min: number; max: number; mean: number } {
    let min = Infinity, max = -Infinity, sum = 0;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < min) min = data[i];
      if (data[i] > max) max = data[i];
      sum += data[i];
    }
    return { min, max, mean: sum / data.length };
  }

  let mainChannel = $derived(getChannelData(main, channelIndex));
  let refChannel = $derived(getChannelData(ref, channelIndex));
  let mainStats = $derived(channelStats(mainChannel));
  let refStats = $derived(channelStats(refChannel));

  let diffChannel = $derived.by(() => {
    const diff = new Float32Array(mainChannel.length);
    for (let i = 0; i < mainChannel.length; i++) {
      diff[i] = Math.abs(mainChannel[i] - refChannel[i]);
    }
    return diff;
  });
  let diffStats = $derived(channelStats(diffChannel));
</script>

<div class="flex flex-col gap-4">
  <!-- Channel selector -->
  {#if numChannels > 1}
    <div class="flex items-center gap-2 text-xs">
      <span class="text-gray-400">Channel:</span>
      <input
        type="range"
        min="0"
        max={numChannels - 1}
        bind:value={channelIndex}
        class="w-48"
      />
      <span class="text-gray-300">{channelIndex} / {numChannels}</span>
    </div>
  {/if}

  <!-- Stats comparison -->
  <div class="grid grid-cols-3 gap-4">
    {#each [
      { label: 'Main Device', stats: mainStats, color: 'text-blue-400' },
      { label: 'Ref Device', stats: refStats, color: 'text-green-400' },
      { label: 'Abs Diff', stats: diffStats, color: 'text-red-400' },
    ] as col (col.label)}
      <div class="bg-gray-900/50 rounded-lg p-3">
        <h4 class="text-xs font-medium {col.color} mb-2">{col.label}</h4>
        <div class="space-y-1 text-xs">
          <div class="flex justify-between">
            <span class="text-gray-500">Min</span>
            <span class="font-mono text-gray-300">{formatVal(col.stats.min)}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-500">Max</span>
            <span class="font-mono text-gray-300">{formatVal(col.stats.max)}</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-500">Mean</span>
            <span class="font-mono text-gray-300">{formatVal(col.stats.mean)}</span>
          </div>
        </div>
      </div>
    {/each}
  </div>

  <p class="text-xs text-gray-500">
    Spatial size: {spatialSize()} elements per channel
  </p>
</div>
