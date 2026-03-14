<script lang="ts">
  import { graphStore } from '../stores/graph.svelte';
  import { queueStore } from '../stores/queue.svelte';
  import { sessionStore } from '../stores/session.svelte';
  import { getGraph } from '../graph/renderer';

  let {
    nodeId,
    nodeName,
    onclose,
  }: {
    nodeId: string;
    nodeName: string;
    onclose: () => void;
  } = $props();

  let direction = $state<'forward' | 'backward'>('forward');
  let stride = $state(1);
  let maxCount = $state(10);
  let previewNodes = $state<{ id: string; name: string; type: string }[]>([]);
  let submitting = $state(false);

  function computePreview() {
    const graph = getGraph();
    if (!graph || !graph.hasNode(nodeId)) {
      previewNodes = [];
      return;
    }

    const visited = new Set<string>();
    const result: { id: string; name: string; type: string }[] = [];
    let queue = [nodeId];
    let stepCount = 0;

    while (queue.length > 0 && result.length < maxCount) {
      const nextQueue: string[] = [];
      for (const current of queue) {
        if (visited.has(current)) continue;
        visited.add(current);

        stepCount++;
        if (stepCount > 1 && (stepCount - 1) % stride === 0) {
          const attrs = graph.getNodeAttributes(current);
          result.push({
            id: current,
            name: attrs.nodeName as string || current,
            type: attrs.opType as string || '',
          });
        }

        const neighbors = direction === 'forward'
          ? graph.outNeighbors(current)
          : graph.inNeighbors(current);
        nextQueue.push(...neighbors);
      }
      queue = nextQueue;
    }

    previewNodes = result;
  }

  $effect(() => {
    // Re-compute when direction/stride/maxCount changes
    direction; stride; maxCount;
    computePreview();
  });

  async function queueAll() {
    const session = sessionStore.currentSession;
    if (!session || previewNodes.length === 0) return;

    submitting = true;
    try {
      const res = await fetch('/api/inference/enqueue-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: session.id,
          nodes: previewNodes.map(n => ({
            node_id: n.id,
            node_name: n.name,
            node_type: n.type,
          })),
        }),
      });
      if (res.ok) {
        const tasks = await res.json();
        for (const task of tasks) {
          queueStore.addTask(task);
        }
        onclose();
      }
    } catch (e) {
      console.error('Batch enqueue failed:', e);
    } finally {
      submitting = false;
    }
  }
</script>

<div class="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-[60] w-96">
  <div class="bg-[--bg-panel] backdrop-blur border border-[--border-color] rounded-xl shadow-2xl">
    <!-- Header -->
    <div class="flex items-center justify-between px-4 py-3 border-b border-[--border-color]">
      <h3 class="text-sm font-medium text-gray-200">Batch Queue</h3>
      <button class="text-gray-400 hover:text-gray-200 text-xs" onclick={onclose}>Close</button>
    </div>

    <!-- Config -->
    <div class="p-4 space-y-3 border-b border-[--border-color]">
      <div class="text-xs text-gray-400">
        Starting from: <span class="text-gray-200 font-mono">{nodeName}</span>
      </div>

      <div class="flex gap-4">
        <label class="flex items-center gap-2 text-xs cursor-pointer">
          <input type="radio" bind:group={direction} value="forward" />
          <span>Forward</span>
        </label>
        <label class="flex items-center gap-2 text-xs cursor-pointer">
          <input type="radio" bind:group={direction} value="backward" />
          <span>Backward</span>
        </label>
      </div>

      <div class="grid grid-cols-2 gap-3">
        <label class="text-xs">
          <span class="text-gray-400">Every N nodes:</span>
          <input
            type="number"
            min="1"
            max="100"
            bind:value={stride}
            class="w-full mt-1 px-2 py-1 bg-[--bg-panel] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none"
          />
        </label>
        <label class="text-xs">
          <span class="text-gray-400">Max count:</span>
          <input
            type="number"
            min="1"
            max="1000"
            bind:value={maxCount}
            class="w-full mt-1 px-2 py-1 bg-[--bg-panel] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none"
          />
        </label>
      </div>
    </div>

    <!-- Preview -->
    <div class="max-h-48 overflow-y-auto">
      {#if previewNodes.length === 0}
        <div class="p-4 text-center text-gray-500 text-xs">No nodes found</div>
      {:else}
        {#each previewNodes as node, i (node.id)}
          <div class="px-4 py-1.5 text-xs flex justify-between border-b border-[--border-color]">
            <span class="font-mono text-gray-300 truncate">{node.name}</span>
            <span class="text-gray-500">{node.type}</span>
          </div>
        {/each}
      {/if}
    </div>

    <!-- Actions -->
    <div class="p-4">
      <button
        class="w-full py-2 bg-accent hover:bg-accent-hover disabled:bg-[--bg-panel] disabled:text-content-secondary rounded text-sm font-medium transition-colors"
        disabled={previewNodes.length === 0 || submitting}
        onclick={queueAll}
      >
        {submitting ? 'Queuing...' : `Queue ${previewNodes.length} Nodes`}
      </button>
    </div>
  </div>
</div>
