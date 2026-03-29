<script lang="ts">
  import { graphStore } from '../stores/graph.svelte';
  import { queueStore } from '../stores/queue.svelte';
  import { sessionStore } from '../stores/session.svelte';
  import { getGraph } from '../graph/renderer';
  import { OP_CATEGORIES } from '../graph/opColors';

  type BatchMode = 'all' | 'by-type' | 'uninferred' | 'from-selection';

  let {
    nodeId = null,
    nodeName = null,
    initialMode = 'all' as BatchMode,
    onclose,
  }: {
    nodeId?: string | null;
    nodeName?: string | null;
    initialMode?: BatchMode;
    onclose: () => void;
  } = $props();

  let _mode = $state<BatchMode | null>(null);
  let mode = $derived<BatchMode>(_mode ?? initialMode);
  let stride = $state(1);
  let maxCount = $state(100);
  let direction = $state<'forward' | 'backward'>('forward');
  let previewNodes = $state<{ id: string; name: string; type: string }[]>([]);
  let submitting = $state(false);

  // By-type mode state
  let availableTypes = $state<{ type: string; count: number }[]>([]);
  let selectedTypes = $state<Set<string>>(new Set());

  // Compute available op types from the graph
  function computeAvailableTypes() {
    const graph = getGraph();
    if (!graph) return;

    const typeCounts = new Map<string, number>();
    const grayedNodes = graphStore.grayedNodes;
    for (const id of graph.nodes()) {
      if (grayedNodes.has(id)) continue;
      const attrs = graph.getNodeAttributes(id);
      const opType = attrs.opType as string;
      if (!opType || opType === 'Parameter' || opType === 'Result' || opType === 'Constant') continue;
      typeCounts.set(opType, (typeCounts.get(opType) ?? 0) + 1);
    }

    // Sort by count descending
    availableTypes = Array.from(typeCounts.entries())
      .map(([type, count]) => ({ type, count }))
      .sort((a, b) => b.count - a.count);
  }

  // Topological sort of non-grayed graph nodes
  function getTopologicalOrder(): string[] {
    const graph = getGraph();
    if (!graph) return [];

    const grayedNodes = graphStore.grayedNodes;
    const allNodes = graph.nodes().filter(id => !grayedNodes.has(id));

    // Compute in-degree (only from non-grayed predecessors)
    const inDegree = new Map<string, number>();
    for (const id of allNodes) {
      let deg = 0;
      for (const pred of graph.inNeighbors(id)) {
        if (!grayedNodes.has(pred)) deg++;
      }
      inDegree.set(id, deg);
    }

    // Kahn's algorithm
    const queue: string[] = [];
    for (const id of allNodes) {
      if ((inDegree.get(id) ?? 0) === 0) queue.push(id);
    }
    const sorted: string[] = [];
    while (queue.length > 0) {
      const node = queue.shift()!;
      sorted.push(node);
      for (const succ of graph.outNeighbors(node)) {
        if (grayedNodes.has(succ)) continue;
        const d = (inDegree.get(succ) ?? 1) - 1;
        inDegree.set(succ, d);
        if (d === 0) queue.push(succ);
      }
    }

    return sorted;
  }

  function computePreview() {
    const graph = getGraph();
    if (!graph) {
      previewNodes = [];
      return;
    }

    const grayedNodes = graphStore.grayedNodes;

    if (mode === 'all') {
      // All non-structural nodes in topological order with stride
      const sorted = getTopologicalOrder();
      const result: { id: string; name: string; type: string }[] = [];
      let stepCount = 0;
      for (const id of sorted) {
        const attrs = graph.getNodeAttributes(id);
        const opType = attrs.opType as string;
        if (opType === 'Parameter' || opType === 'Result' || opType === 'Constant') continue;
        stepCount++;
        if ((stepCount - 1) % stride === 0 && result.length < maxCount) {
          result.push({
            id,
            name: attrs.nodeName as string || id,
            type: opType || '',
          });
        }
      }
      previewNodes = result;

    } else if (mode === 'by-type') {
      // Nodes matching selected types in topological order
      if (selectedTypes.size === 0) {
        previewNodes = [];
        return;
      }
      const sorted = getTopologicalOrder();
      const result: { id: string; name: string; type: string }[] = [];
      for (const id of sorted) {
        const attrs = graph.getNodeAttributes(id);
        const opType = attrs.opType as string;
        if (!selectedTypes.has(opType)) continue;
        if (result.length < maxCount) {
          result.push({
            id,
            name: attrs.nodeName as string || id,
            type: opType || '',
          });
        }
      }
      previewNodes = result;

    } else if (mode === 'uninferred') {
      // Nodes without successful results
      const nodeStatusMap = graphStore.nodeStatusMap;
      const sorted = getTopologicalOrder();
      const result: { id: string; name: string; type: string }[] = [];
      let stepCount = 0;
      for (const id of sorted) {
        const attrs = graph.getNodeAttributes(id);
        const opType = attrs.opType as string;
        if (opType === 'Parameter' || opType === 'Result' || opType === 'Constant') continue;
        const status = nodeStatusMap.get(id);
        if (status?.status === 'success') continue;
        stepCount++;
        if ((stepCount - 1) % stride === 0 && result.length < maxCount) {
          result.push({
            id,
            name: attrs.nodeName as string || id,
            type: opType || '',
          });
        }
      }
      previewNodes = result;

    } else if (mode === 'from-selection') {
      // BFS from selected node (original behavior)
      if (!nodeId || !graph.hasNode(nodeId)) {
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
          if (grayedNodes.has(current)) continue;

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
  }

  // Compute types once on mount
  computeAvailableTypes();

  $effect(() => {
    // Re-compute when mode/direction/stride/maxCount/selectedTypes changes
    mode; direction; stride; maxCount; selectedTypes;
    computePreview();
  });

  function toggleType(type: string) {
    const newSet = new Set(selectedTypes);
    if (newSet.has(type)) {
      newSet.delete(type);
    } else {
      newSet.add(type);
    }
    selectedTypes = newSet;
  }

  function selectAllTypes() {
    selectedTypes = new Set(availableTypes.map(t => t.type));
  }

  function deselectAllTypes() {
    selectedTypes = new Set();
  }

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

  const MODE_LABELS: Record<BatchMode, string> = {
    'all': 'All Nodes',
    'by-type': 'By Type',
    'uninferred': 'Un-inferred',
    'from-selection': 'From Selection',
  };
</script>

<div class="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-[60] w-[28rem]">
  <div class="bg-[--bg-panel] backdrop-blur border border-[--border-color] rounded-xl shadow-2xl">
    <!-- Header -->
    <div class="flex items-center justify-between px-4 py-3 border-b border-[--border-color]">
      <h3 class="text-sm font-medium text-gray-200">Batch Queue</h3>
      <button class="text-gray-400 hover:text-gray-200 text-xs" onclick={onclose}>Close</button>
    </div>

    <!-- Mode Tabs -->
    <div class="flex border-b border-[--border-color]">
      {#each (['all', 'by-type', 'uninferred', 'from-selection'] as const) as m (m)}
        <button
          class="flex-1 px-2 py-2 text-xs text-center transition-colors {mode === m ? 'text-accent border-b-2 border-accent' : 'text-gray-400 hover:text-gray-200'}"
          disabled={m === 'from-selection' && !nodeId}
          onclick={() => { _mode = m; }}
        >
          {MODE_LABELS[m]}
        </button>
      {/each}
    </div>

    <!-- Mode Config -->
    <div class="p-4 space-y-3 border-b border-[--border-color]">
      {#if mode === 'from-selection'}
        <div class="text-xs text-gray-400">
          Starting from: <span class="text-gray-200 font-mono">{nodeName ?? '(none)'}</span>
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
      {/if}

      {#if mode === 'by-type'}
        <!-- Op type checklist -->
        <div class="flex justify-between items-center">
          <span class="text-xs text-gray-400">Select op types:</span>
          <div class="flex gap-2">
            <button class="text-[10px] text-accent hover:text-accent-hover" onclick={selectAllTypes}>All</button>
            <button class="text-[10px] text-gray-400 hover:text-gray-200" onclick={deselectAllTypes}>None</button>
          </div>
        </div>
        <div class="max-h-32 overflow-y-auto space-y-0.5 border border-[--border-color] rounded p-2">
          {#each availableTypes as item (item.type)}
            <label class="flex items-center gap-2 text-xs cursor-pointer hover:bg-[--bg-primary] px-1 py-0.5 rounded">
              <input
                type="checkbox"
                checked={selectedTypes.has(item.type)}
                onchange={() => toggleType(item.type)}
              />
              <span class="flex-1 font-mono">{item.type}</span>
              <span class="text-gray-500">{item.count}</span>
            </label>
          {/each}
        </div>
      {/if}

      {#if mode !== 'by-type'}
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
              max="5000"
              bind:value={maxCount}
              class="w-full mt-1 px-2 py-1 bg-[--bg-panel] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none"
            />
          </label>
        </div>
      {:else}
        <div class="grid grid-cols-1 gap-3">
          <label class="text-xs">
            <span class="text-gray-400">Max count:</span>
            <input
              type="number"
              min="1"
              max="5000"
              bind:value={maxCount}
              class="w-full mt-1 px-2 py-1 bg-[--bg-panel] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none"
            />
          </label>
        </div>
      {/if}

      <!-- Preview count -->
      <div class="text-xs text-gray-400">
        Will enqueue <span class="text-gray-200 font-medium">{previewNodes.length}</span> nodes
      </div>
    </div>

    <!-- Preview List -->
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
