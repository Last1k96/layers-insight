<script lang="ts">
  import { onMount } from 'svelte';

  let {
    diff,
    shape,
  }: {
    diff: Float32Array | null;
    shape: number[];
  } = $props();

  let canvas: HTMLCanvasElement;
  let sliceIndex = $state(0);
  let channelIndex = $state(0);

  // For 4D tensors: [N, C, H, W]
  let numChannels = $derived(shape.length >= 4 ? shape[1] : shape.length >= 3 ? shape[0] : 1);
  let height = $derived(shape.length >= 4 ? shape[2] : shape.length >= 3 ? shape[1] : shape.length >= 2 ? shape[0] : 1);
  let width = $derived(shape.length >= 4 ? shape[3] : shape.length >= 3 ? shape[2] : shape.length >= 2 ? shape[1] : diff?.length || 1);

  function drawHeatmap() {
    if (!canvas || !diff) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const h = height;
    const w = width;
    canvas.width = w;
    canvas.height = h;

    // Extract 2D slice
    const offset = (sliceIndex * numChannels + channelIndex) * h * w;
    const imageData = ctx.createImageData(w, h);

    // Find max for normalization
    let maxVal = 0;
    for (let i = 0; i < h * w; i++) {
      const idx = offset + i;
      if (idx < diff.length) {
        maxVal = Math.max(maxVal, diff[idx]);
      }
    }
    if (maxVal === 0) maxVal = 1;

    // Draw pixels with blue-to-red colormap
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = offset + y * w + x;
        const val = idx < diff.length ? diff[idx] / maxVal : 0;
        const pixIdx = (y * w + x) * 4;

        // Blue (0) -> Green (0.5) -> Red (1)
        if (val < 0.5) {
          const t = val * 2;
          imageData.data[pixIdx] = Math.floor(t * 255);
          imageData.data[pixIdx + 1] = Math.floor(t * 255);
          imageData.data[pixIdx + 2] = Math.floor((1 - t) * 255);
        } else {
          const t = (val - 0.5) * 2;
          imageData.data[pixIdx] = 255;
          imageData.data[pixIdx + 1] = Math.floor((1 - t) * 255);
          imageData.data[pixIdx + 2] = 0;
        }
        imageData.data[pixIdx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }

  $effect(() => {
    drawHeatmap();
  });
</script>

<div class="flex flex-col gap-4">
  <!-- Controls -->
  <div class="flex gap-4 items-center text-xs">
    {#if shape.length >= 4}
      <label class="flex items-center gap-2">
        <span class="text-gray-400">Batch:</span>
        <input
          type="range"
          min="0"
          max={Math.max(0, shape[0] - 1)}
          bind:value={sliceIndex}
          class="w-24"
        />
        <span class="text-gray-300 w-6">{sliceIndex}</span>
      </label>
    {/if}
    {#if numChannels > 1}
      <label class="flex items-center gap-2">
        <span class="text-gray-400">Channel:</span>
        <input
          type="range"
          min="0"
          max={numChannels - 1}
          bind:value={channelIndex}
          class="w-32"
        />
        <span class="text-gray-300 w-8">{channelIndex}/{numChannels}</span>
      </label>
    {/if}
    <span class="text-gray-500">
      Shape: [{shape.join(', ')}] | Slice: {height}x{width}
    </span>
  </div>

  <!-- Canvas -->
  <div class="flex justify-center bg-gray-900 rounded-lg p-4 overflow-auto">
    <canvas
      bind:this={canvas}
      class="max-w-full"
      style="image-rendering: pixelated; min-width: {Math.min(width * 2, 800)}px;"
    ></canvas>
  </div>

  <!-- Color legend -->
  <div class="flex items-center gap-2 text-xs text-gray-400">
    <span>Low diff</span>
    <div class="h-3 w-48 rounded" style="background: linear-gradient(to right, #0000ff, #00ff00, #ff0000);"></div>
    <span>High diff</span>
  </div>
</div>
