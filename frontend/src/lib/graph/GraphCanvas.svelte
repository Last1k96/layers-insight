<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { initRenderer, destroyRenderer } from './renderer';
  import { setupInteractions } from './interactions';

  let container: HTMLDivElement;
  let error = $state<string | null>(null);

  onMount(() => {
    const unwatch = $effect.root(() => {
      $effect(() => {
        if (graphStore.graphData && container) {
          error = null;
          initRenderer(container, graphStore.graphData)
            .then(() => {
              setupInteractions();
            })
            .catch((err) => {
              console.error('[GraphCanvas] Failed to init renderer:', err);
              error = err?.message ?? String(err);
            });
        }
      });
    });

    return () => {
      unwatch();
    };
  });

  onDestroy(() => {
    destroyRenderer();
  });
</script>

{#if error}
  <div class="absolute inset-0 flex items-center justify-center" style="background: var(--bg-primary);">
    <div class="max-w-lg p-8 rounded-lg text-center" style="background: var(--bg-secondary); color: var(--text-primary);">
      <h2 class="text-xl font-semibold mb-4">WebGPU Required</h2>
      <p class="mb-4 opacity-80">
        This application requires WebGPU to render the graph, but your browser does not have it enabled.
      </p>
      <div class="text-left text-sm space-y-3 opacity-90">
        <p class="font-medium">To enable GPU acceleration in Chrome-based browsers:</p>
        <ol class="list-decimal list-inside space-y-1 ml-2">
          <li>Open <code class="px-1.5 py-0.5 rounded" style="background: var(--bg-primary);">chrome://settings/system</code> in your address bar</li>
          <li>Enable <strong>"Use graphics acceleration when available"</strong></li>
          <li>Relaunch the browser</li>
        </ol>
        <p class="mt-3 opacity-70 text-xs">
          If the issue persists, check <code class="px-1 py-0.5 rounded" style="background: var(--bg-primary);">chrome://gpu</code> for WebGPU status
          or try <code class="px-1 py-0.5 rounded" style="background: var(--bg-primary);">chrome://flags/#enable-unsafe-webgpu</code>.
        </p>
      </div>
    </div>
  </div>
{:else}
  <div bind:this={container} class="absolute inset-0" style="background: var(--bg-primary);"></div>
{/if}
