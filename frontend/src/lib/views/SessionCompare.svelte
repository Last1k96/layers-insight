<script lang="ts">
  import { sessionStore } from '../stores/session.svelte';
  import { onMount } from 'svelte';
  import type { CompareResponse, CompareNodeResult, SessionInfo } from '../stores/types';

  let {
    sessionAId,
    sessionBId,
    onback,
    onnodeselected,
  }: {
    sessionAId: string;
    sessionBId: string;
    onback: () => void;
    onnodeselected?: (sessionId: string, nodeName: string) => void;
  } = $props();

  let compareData = $state<CompareResponse | null>(null);
  let loading = $state(true);
  let error = $state<string | null>(null);

  // Session info for labels
  let sessionAInfo = $state<SessionInfo | null>(null);
  let sessionBInfo = $state<SessionInfo | null>(null);

  // Sorting
  type SortColumn = 'node_name' | 'node_type' | 'cosine_a' | 'cosine_b' | 'delta_cosine' | 'mse_a' | 'mse_b' | 'delta_mse';
  let sortColumn = $state<SortColumn>('delta_cosine');
  let sortAsc = $state(true);

  // Filter
  let filterText = $state('');
  let filterCategory = $state<'all' | 'improved' | 'regressed' | 'unchanged' | 'only_a' | 'only_b'>('all');

  let filteredNodes = $derived.by(() => {
    if (!compareData) return [];
    let nodes = [...compareData.nodes];

    // Text filter
    if (filterText) {
      const lower = filterText.toLowerCase();
      nodes = nodes.filter(n =>
        n.node_name.toLowerCase().includes(lower) ||
        n.node_type.toLowerCase().includes(lower)
      );
    }

    // Category filter
    const TOLERANCE = 0.0001;
    if (filterCategory !== 'all') {
      nodes = nodes.filter(n => {
        if (filterCategory === 'only_a') return n.metrics_a && !n.metrics_b;
        if (filterCategory === 'only_b') return !n.metrics_a && n.metrics_b;
        if (n.delta_cosine === null || n.delta_cosine === undefined) return filterCategory === 'unchanged';
        if (filterCategory === 'improved') return n.delta_cosine > TOLERANCE;
        if (filterCategory === 'regressed') return n.delta_cosine < -TOLERANCE;
        if (filterCategory === 'unchanged') return Math.abs(n.delta_cosine) <= TOLERANCE;
        return true;
      });
    }

    // Sort
    nodes.sort((a, b) => {
      let va: any, vb: any;
      switch (sortColumn) {
        case 'node_name': va = a.node_name; vb = b.node_name; break;
        case 'node_type': va = a.node_type; vb = b.node_type; break;
        case 'cosine_a': va = a.metrics_a?.cosine_similarity ?? -1; vb = b.metrics_a?.cosine_similarity ?? -1; break;
        case 'cosine_b': va = a.metrics_b?.cosine_similarity ?? -1; vb = b.metrics_b?.cosine_similarity ?? -1; break;
        case 'delta_cosine': va = a.delta_cosine ?? 0; vb = b.delta_cosine ?? 0; break;
        case 'mse_a': va = a.metrics_a?.mse ?? 0; vb = b.metrics_a?.mse ?? 0; break;
        case 'mse_b': va = a.metrics_b?.mse ?? 0; vb = b.metrics_b?.mse ?? 0; break;
        case 'delta_mse': va = a.delta_mse ?? 0; vb = b.delta_mse ?? 0; break;
      }
      if (typeof va === 'string') {
        const cmp = va.localeCompare(vb);
        return sortAsc ? cmp : -cmp;
      }
      return sortAsc ? va - vb : vb - va;
    });

    return nodes;
  });

  function toggleSort(col: SortColumn) {
    if (sortColumn === col) {
      sortAsc = !sortAsc;
    } else {
      sortColumn = col;
      sortAsc = col === 'delta_cosine'; // default ascending for delta (worst first)
    }
  }

  function sortIndicator(col: SortColumn): string {
    if (sortColumn !== col) return '';
    return sortAsc ? ' ^' : ' v';
  }

  function formatMetric(val: number | null | undefined, digits = 6): string {
    if (val === null || val === undefined) return '-';
    return val.toFixed(digits);
  }

  function deltaClass(delta: number | null | undefined): string {
    if (delta === null || delta === undefined) return 'text-content-secondary';
    const TOLERANCE = 0.0001;
    if (delta > TOLERANCE) return 'text-green-400';
    if (delta < -TOLERANCE) return 'text-red-400';
    return 'text-content-secondary';
  }

  function deltaMseClass(delta: number | null | undefined): string {
    if (delta === null || delta === undefined) return 'text-content-secondary';
    const TOLERANCE = 0.00001;
    // For MSE, negative delta = improved (lower is better)
    if (delta < -TOLERANCE) return 'text-green-400';
    if (delta > TOLERANCE) return 'text-red-400';
    return 'text-content-secondary';
  }

  function formatDelta(val: number | null | undefined): string {
    if (val === null || val === undefined) return '-';
    const prefix = val > 0 ? '+' : '';
    return prefix + val.toFixed(6);
  }

  onMount(async () => {
    loading = true;
    error = null;

    // Fetch session info for labels
    const sessions = sessionStore.sessions;
    sessionAInfo = sessions.find(s => s.id === sessionAId) ?? null;
    sessionBInfo = sessions.find(s => s.id === sessionBId) ?? null;

    // If sessions aren't loaded, fetch them
    if (!sessionAInfo || !sessionBInfo) {
      await sessionStore.fetchSessions();
      sessionAInfo = sessionStore.sessions.find(s => s.id === sessionAId) ?? null;
      sessionBInfo = sessionStore.sessions.find(s => s.id === sessionBId) ?? null;
    }

    const result = await sessionStore.compareSessions(sessionAId, sessionBId);
    if (result) {
      compareData = result;
    } else {
      error = sessionStore.error ?? 'Failed to compare sessions';
    }
    loading = false;
  });
</script>

<div class="flex-1 flex flex-col bg-[--bg-primary] overflow-hidden">
  <!-- Header -->
  <div class="flex-shrink-0 p-4 border-b border-edge">
    <div class="flex items-center gap-3 mb-3">
      <button class="text-content-secondary hover:text-content-primary" onclick={onback}>&larr; Back</button>
      <h2 class="text-xl font-bold">Session Comparison</h2>
    </div>

    <!-- Session pills -->
    <div class="flex items-center gap-2 text-sm">
      <div class="px-3 py-1.5 bg-blue-900/30 border border-blue-700/50 rounded-lg">
        <span class="text-blue-400 font-medium">A:</span>
        <span class="text-content-primary ml-1">{sessionAInfo?.model_name ?? sessionAId}</span>
        <span class="text-content-secondary ml-1">({sessionAInfo?.main_device ?? '?'} vs {sessionAInfo?.ref_device ?? '?'})</span>
      </div>
      <span class="text-content-secondary">vs</span>
      <div class="px-3 py-1.5 bg-emerald-900/30 border border-emerald-700/50 rounded-lg">
        <span class="text-emerald-400 font-medium">B:</span>
        <span class="text-content-primary ml-1">{sessionBInfo?.model_name ?? sessionBId}</span>
        <span class="text-content-secondary ml-1">({sessionBInfo?.main_device ?? '?'} vs {sessionBInfo?.ref_device ?? '?'})</span>
      </div>
    </div>

    <!-- Summary bar -->
    {#if compareData}
      <div class="flex items-center gap-4 mt-3 text-sm">
        <button
          class="px-2 py-0.5 rounded transition-colors {filterCategory === 'all' ? 'bg-surface-elevated text-content-primary' : 'text-content-secondary hover:text-content-primary'}"
          onclick={() => filterCategory = 'all'}
        >
          {compareData.summary.total_compared + compareData.summary.only_in_a + compareData.summary.only_in_b} total
        </button>
        {#if compareData.summary.improved > 0}
          <button
            class="px-2 py-0.5 rounded transition-colors {filterCategory === 'improved' ? 'bg-green-900/50 text-green-400' : 'text-green-400/70 hover:text-green-400'}"
            onclick={() => filterCategory = filterCategory === 'improved' ? 'all' : 'improved'}
          >
            {compareData.summary.improved} improved
          </button>
        {/if}
        {#if compareData.summary.regressed > 0}
          <button
            class="px-2 py-0.5 rounded transition-colors {filterCategory === 'regressed' ? 'bg-red-900/50 text-red-400' : 'text-red-400/70 hover:text-red-400'}"
            onclick={() => filterCategory = filterCategory === 'regressed' ? 'all' : 'regressed'}
          >
            {compareData.summary.regressed} regressed
          </button>
        {/if}
        {#if compareData.summary.unchanged > 0}
          <button
            class="px-2 py-0.5 rounded transition-colors {filterCategory === 'unchanged' ? 'bg-surface-elevated text-content-primary' : 'text-content-secondary hover:text-content-primary'}"
            onclick={() => filterCategory = filterCategory === 'unchanged' ? 'all' : 'unchanged'}
          >
            {compareData.summary.unchanged} unchanged
          </button>
        {/if}
        {#if compareData.summary.only_in_a > 0}
          <button
            class="px-2 py-0.5 rounded transition-colors {filterCategory === 'only_a' ? 'bg-blue-900/50 text-blue-400' : 'text-blue-400/70 hover:text-blue-400'}"
            onclick={() => filterCategory = filterCategory === 'only_a' ? 'all' : 'only_a'}
          >
            {compareData.summary.only_in_a} only in A
          </button>
        {/if}
        {#if compareData.summary.only_in_b > 0}
          <button
            class="px-2 py-0.5 rounded transition-colors {filterCategory === 'only_b' ? 'bg-emerald-900/50 text-emerald-400' : 'text-emerald-400/70 hover:text-emerald-400'}"
            onclick={() => filterCategory = filterCategory === 'only_b' ? 'all' : 'only_b'}
          >
            {compareData.summary.only_in_b} only in B
          </button>
        {/if}
      </div>

      <!-- Search filter -->
      <div class="mt-2">
        <input
          type="text"
          bind:value={filterText}
          placeholder="Filter by node name or type..."
          class="w-full px-3 py-1.5 bg-[--bg-input] border border-[--border-color] rounded text-sm focus:border-accent focus:outline-none"
        />
      </div>
    {/if}
  </div>

  <!-- Content -->
  <div class="flex-1 overflow-auto">
    {#if loading}
      <div class="flex items-center justify-center h-full text-content-secondary">
        Loading comparison...
      </div>
    {:else if error}
      <div class="p-4">
        <div class="p-3 bg-red-900/50 border border-red-700 rounded text-red-300 text-sm">
          {error}
        </div>
      </div>
    {:else if compareData && filteredNodes.length === 0}
      <div class="flex items-center justify-center h-full text-content-secondary">
        No matching nodes to compare.
      </div>
    {:else if compareData}
      <table class="w-full text-sm">
        <thead class="sticky top-0 bg-[--bg-primary] z-10">
          <tr class="border-b border-edge text-left">
            <th class="px-3 py-2 font-medium text-content-secondary cursor-pointer hover:text-content-primary" onclick={() => toggleSort('node_name')}>
              Node{sortIndicator('node_name')}
            </th>
            <th class="px-3 py-2 font-medium text-content-secondary cursor-pointer hover:text-content-primary" onclick={() => toggleSort('node_type')}>
              Type{sortIndicator('node_type')}
            </th>
            <th class="px-3 py-2 font-medium text-blue-400 cursor-pointer hover:text-blue-300 text-right" onclick={() => toggleSort('cosine_a')}>
              Cos A{sortIndicator('cosine_a')}
            </th>
            <th class="px-3 py-2 font-medium text-emerald-400 cursor-pointer hover:text-emerald-300 text-right" onclick={() => toggleSort('cosine_b')}>
              Cos B{sortIndicator('cosine_b')}
            </th>
            <th class="px-3 py-2 font-medium text-content-secondary cursor-pointer hover:text-content-primary text-right" onclick={() => toggleSort('delta_cosine')}>
              Delta Cos{sortIndicator('delta_cosine')}
            </th>
            <th class="px-3 py-2 font-medium text-blue-400 cursor-pointer hover:text-blue-300 text-right" onclick={() => toggleSort('mse_a')}>
              MSE A{sortIndicator('mse_a')}
            </th>
            <th class="px-3 py-2 font-medium text-emerald-400 cursor-pointer hover:text-emerald-300 text-right" onclick={() => toggleSort('mse_b')}>
              MSE B{sortIndicator('mse_b')}
            </th>
            <th class="px-3 py-2 font-medium text-content-secondary cursor-pointer hover:text-content-primary text-right" onclick={() => toggleSort('delta_mse')}>
              Delta MSE{sortIndicator('delta_mse')}
            </th>
          </tr>
        </thead>
        <tbody>
          {#each filteredNodes as node (node.node_name)}
            <tr
              class="border-b border-edge/50 hover:bg-surface-elevated cursor-pointer transition-colors"
              onclick={() => onnodeselected?.(sessionAId, node.node_name)}
            >
              <td class="px-3 py-1.5 font-mono text-xs truncate max-w-[200px]" title={node.node_name}>
                {node.node_name}
              </td>
              <td class="px-3 py-1.5 text-content-secondary">
                {node.node_type}
              </td>
              <td class="px-3 py-1.5 text-right font-mono text-xs">
                {formatMetric(node.metrics_a?.cosine_similarity)}
              </td>
              <td class="px-3 py-1.5 text-right font-mono text-xs">
                {formatMetric(node.metrics_b?.cosine_similarity)}
              </td>
              <td class="px-3 py-1.5 text-right font-mono text-xs {deltaClass(node.delta_cosine)}">
                {formatDelta(node.delta_cosine)}
              </td>
              <td class="px-3 py-1.5 text-right font-mono text-xs">
                {formatMetric(node.metrics_a?.mse)}
              </td>
              <td class="px-3 py-1.5 text-right font-mono text-xs">
                {formatMetric(node.metrics_b?.mse)}
              </td>
              <td class="px-3 py-1.5 text-right font-mono text-xs {deltaMseClass(node.delta_mse)}">
                {formatDelta(node.delta_mse)}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>
</div>
