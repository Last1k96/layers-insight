<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { initRenderer, destroyRenderer } from './renderer';
  import { setupInteractions } from './interactions';

  let container: HTMLDivElement;

  onMount(() => {
    const unwatch = $effect.root(() => {
      $effect(() => {
        if (graphStore.graphData && container) {
          initRenderer(container, graphStore.graphData)
            .then(() => {
              setupInteractions();
            })
            .catch((err) => {
              console.error('[GraphCanvas] Failed to init renderer:', err);
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

<div bind:this={container} class="absolute inset-0" style="background: #1a1a2e;"></div>
