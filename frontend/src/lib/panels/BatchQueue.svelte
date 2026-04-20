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

<div class="bq-modal">
  <div class="bq-shell">
    <!-- Header -->
    <div class="bq-header">
      <div class="bq-header__title">
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">
          <path d="M2 4h12M2 8h12M2 12h12" />
        </svg>
        <h3>Batch Queue</h3>
      </div>
      <button class="ll-icon-btn ll-icon-btn--sm" onclick={onclose} aria-label="Close">
        <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="4" y1="4" x2="12" y2="12"/><line x1="12" y1="4" x2="4" y2="12"/></svg>
      </button>
    </div>

    <!-- Mode Tabs -->
    <div class="bq-tabs">
      {#each (['all', 'by-type', 'uninferred', 'from-selection'] as const) as m (m)}
        <button
          class="bq-tab"
          class:bq-tab--active={mode === m}
          disabled={m === 'from-selection' && !nodeId}
          onclick={() => { _mode = m; }}
        >
          {MODE_LABELS[m]}
        </button>
      {/each}
    </div>

    <!-- Mode Config -->
    <div class="bq-body">
      {#if mode === 'from-selection'}
        <div class="bq-hint">
          Starting from: <span class="bq-node">{nodeName ?? '(none)'}</span>
        </div>
        <div class="bq-radio-row">
          <label class="bq-radio">
            <input type="radio" bind:group={direction} value="forward" />
            <span>Forward</span>
          </label>
          <label class="bq-radio">
            <input type="radio" bind:group={direction} value="backward" />
            <span>Backward</span>
          </label>
        </div>
      {/if}

      {#if mode === 'by-type'}
        <div class="bq-types-head">
          <span class="bq-label">Op types</span>
          <div class="bq-types-controls">
            <button class="ll-btn ll-btn--xs" onclick={selectAllTypes}>All</button>
            <button class="ll-btn ll-btn--xs ll-btn--ghost" onclick={deselectAllTypes}>None</button>
          </div>
        </div>
        <div class="bq-types">
          {#each availableTypes as item (item.type)}
            <label class="bq-type">
              <input
                type="checkbox"
                checked={selectedTypes.has(item.type)}
                onchange={() => toggleType(item.type)}
              />
              <span class="bq-type__name">{item.type}</span>
              <span class="bq-type__count">{item.count}</span>
            </label>
          {/each}
        </div>
      {/if}

      {#if mode !== 'by-type'}
        <div class="bq-grid">
          <label class="bq-field">
            <span class="bq-label">Every N nodes</span>
            <input
              type="number"
              min="1"
              max="100"
              bind:value={stride}
              onwheel={(e) => { e.preventDefault(); const d = e.deltaY < 0 ? 1 : -1; const s = e.shiftKey ? 10 : 1; stride = Math.max(1, Math.min(100, stride + d * s)); }}
              class="ll-field ll-field--mono"
            />
          </label>
          <label class="bq-field">
            <span class="bq-label">Max count</span>
            <input
              type="number"
              min="1"
              max="5000"
              bind:value={maxCount}
              onwheel={(e) => { e.preventDefault(); const d = e.deltaY < 0 ? 1 : -1; const s = e.shiftKey ? 100 : 10; maxCount = Math.max(1, Math.min(5000, maxCount + d * s)); }}
              class="ll-field ll-field--mono"
            />
          </label>
        </div>
      {:else}
        <label class="bq-field">
          <span class="bq-label">Max count</span>
          <input
            type="number"
            min="1"
            max="5000"
            bind:value={maxCount}
            onwheel={(e) => { e.preventDefault(); const d = e.deltaY < 0 ? 1 : -1; const s = e.shiftKey ? 100 : 10; maxCount = Math.max(1, Math.min(5000, maxCount + d * s)); }}
            class="ll-field ll-field--mono"
          />
        </label>
      {/if}

      <!-- Preview count -->
      <div class="bq-summary">
        <span class="ll-chip ll-chip--accent ll-chip--tiny">Preview</span>
        <span>Will enqueue <strong>{previewNodes.length}</strong> nodes</span>
      </div>
    </div>

    <!-- Preview List -->
    <div class="bq-preview">
      {#if previewNodes.length === 0}
        <div class="bq-preview__empty">No nodes found</div>
      {:else}
        {#each previewNodes as node (node.id)}
          <div class="bq-preview__row">
            <span class="bq-preview__name">{node.name}</span>
            <span class="bq-preview__type">{node.type}</span>
          </div>
        {/each}
      {/if}
    </div>

    <!-- Actions -->
    <div class="bq-actions">
      <button
        class="ll-btn ll-btn--primary ll-btn--block"
        disabled={previewNodes.length === 0 || submitting}
        onclick={queueAll}
      >
        {#if submitting}
          <svg class="ll-spin" width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-dasharray="31.4 31.4" stroke-linecap="round"/>
          </svg>
          Queuing…
        {:else}
          Queue {previewNodes.length} {previewNodes.length === 1 ? 'node' : 'nodes'}
        {/if}
      </button>
    </div>
  </div>
</div>

<style>
  .bq-modal {
    position: fixed;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    z-index: 60;
    width: 460px;
    max-width: 92vw;
  }
  .bq-shell {
    background: rgba(35, 38, 54, 0.97);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-modal);
    overflow: hidden;
  }

  .bq-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 14px;
    border-bottom: 1px solid var(--border-soft);
  }
  .bq-header__title {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--accent-soft);
  }
  .bq-header__title h3 {
    margin: 0;
    font-family: var(--font-display);
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    letter-spacing: 0.01em;
  }

  .bq-tabs {
    display: flex;
    background: var(--bg-primary);
  }
  .bq-tab {
    flex: 1;
    padding: 10px 8px;
    font-size: 11.5px;
    font-weight: 500;
    color: var(--text-muted);
    background: transparent;
    border: 0;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    transition: color var(--dur-fast) ease, background var(--dur-fast) ease, border-color var(--dur-fast) ease;
  }
  .bq-tab:hover:not(:disabled) { color: var(--text-primary); background: var(--accent-bg-soft); }
  .bq-tab--active {
    color: var(--accent);
    border-bottom-color: var(--accent);
    background: var(--accent-bg-soft);
  }
  .bq-tab:disabled { opacity: 0.4; cursor: default; }

  .bq-body {
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    border-bottom: 1px solid var(--border-soft);
  }

  .bq-hint {
    font-size: 12px;
    color: var(--text-muted-strong);
    line-height: 1.5;
  }
  .bq-node {
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 11.5px;
  }

  .bq-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
  }

  .bq-field { display: flex; flex-direction: column; gap: 5px; }

  .bq-radio-row { display: flex; gap: 14px; }
  .bq-radio {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-primary);
    cursor: pointer;
  }

  .bq-types-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }
  .bq-types-controls { display: inline-flex; gap: 4px; }

  .bq-types {
    max-height: 140px;
    overflow-y: auto;
    padding: 6px;
    background: var(--bg-primary);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-md);
    display: flex;
    flex-direction: column;
    gap: 1px;
  }
  .bq-type {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 7px;
    font-size: 12px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background var(--dur-fast) ease;
  }
  .bq-type:hover { background: var(--accent-bg-soft); }
  .bq-type__name { flex: 1; font-family: var(--font-mono); color: var(--text-primary); }
  .bq-type__count {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 10.5px;
    color: var(--text-muted);
    padding: 0 6px;
    border-radius: var(--radius-pill);
    background: rgba(155,161,181,0.08);
  }

  .bq-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }

  .bq-summary {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11.5px;
    color: var(--text-muted-strong);
  }
  .bq-summary strong { color: var(--text-primary); font-weight: 600; }

  .bq-preview {
    max-height: 200px;
    overflow-y: auto;
  }
  .bq-preview__row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding: 6px 16px;
    font-size: 11.5px;
    transition: background var(--dur-fast) ease;
  }
  .bq-preview__row:hover { background: var(--accent-bg-soft); }
  .bq-preview__name {
    flex: 1;
    font-family: var(--font-mono);
    color: var(--text-muted-strong);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bq-preview__type {
    color: var(--text-muted-soft);
    font-size: 10.5px;
    flex-shrink: 0;
  }
  .bq-preview__empty {
    padding: 28px 16px;
    text-align: center;
    font-size: 11.5px;
    color: var(--text-muted-soft);
  }

  .bq-actions {
    padding: 12px 16px 14px;
    border-top: 1px solid var(--border-soft);
  }
</style>
