<script lang="ts">
  import { renderDiagnostics } from './diagRenderer';

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
    renderDiagnostics(ctx, canvas, main, ref, shape, containerWidth, mainLabel, refLabel);
  });
</script>

<div bind:this={container} class="w-full">
  <canvas bind:this={canvas} class="block mx-auto"></canvas>
</div>
