<script lang="ts">
  import { untrack } from 'svelte';
  import { bisectStore } from '../stores/bisect.svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { sessionStore } from '../stores/session.svelte';

  let {
    onclose,
    endNodeId = null,
    endNodeName = null,
    initialSearchFor = null,
  }: {
    onclose: () => void;
    endNodeId?: string | null;
    endNodeName?: string | null;
    initialSearchFor?: 'accuracy_drop' | 'compilation_failure' | null;
  } = $props();

  type BisectTab = 'full-model' | 'from-node';
  const TAB_LABELS: Record<BisectTab, string> = {
    'full-model': 'Full Model',
    'from-node': 'From Node',
  };

  let activeTab = $state<BisectTab>('full-model');

  // Set initial tab based on how the panel was opened
  $effect.pre(() => {
    if (endNodeId) {
      activeTab = 'from-node';
    }
  });

  // Auto-detect search_for when opened from a node
  $effect.pre(() => {
    if (initialSearchFor) {
      bisectStore.searchFor = initialSearchFor;
      bisectStore.threshold = defaultThreshold(untrack(() => bisectStore.metric), initialSearchFor);
    }
  });

  const metricLabels: Record<string, string> = {
    cosine_similarity: 'Cosine similarity',
    mse: 'MSE',
    max_abs_diff: 'Max abs diff',
  };

  const searchForLabels: Record<string, string> = {
    accuracy_drop: 'Accuracy drop',
    compilation_failure: 'Compilation failure',
  };

  function defaultThreshold(metric: string, searchFor: string): number {
    if (searchFor === 'compilation_failure') return 0;
    if (metric === 'cosine_similarity') return 0.999;
    if (metric === 'mse') return 0.001;
    return 0.01;
  }

  function onSearchForChange(e: Event) {
    const val = (e.target as HTMLSelectElement).value;
    bisectStore.searchFor = val as any;
    bisectStore.threshold = defaultThreshold(bisectStore.metric, val);
  }

  function onMetricChange(e: Event) {
    const val = (e.target as HTMLSelectElement).value;
    bisectStore.metric = val as any;
    bisectStore.threshold = defaultThreshold(val, bisectStore.searchFor);
  }

  async function startBisect() {
    const session = sessionStore.currentSession;
    if (!session) return;
    bisectStore.error = null;
    const endNode = activeTab === 'from-node' ? endNodeId : null;
    const ok = await bisectStore.start(session.id, graphStore.activeSubSessionId, endNode);
    if (ok) {
      onclose();
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      onclose();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="fixed inset-0 z-[59] bg-black/30" onclick={onclose} role="presentation"></div>
<div class="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-[60] w-[420px]">
  <div class="bg-[--bg-panel] backdrop-blur border border-[--border-color] rounded-xl shadow-2xl">
    <!-- Header -->
    <div class="flex items-center justify-between px-4 py-3 border-b border-[--border-color]">
      <h3 class="text-sm font-medium text-gray-200">Bisect</h3>
      <button class="text-gray-400 hover:text-gray-200 text-xs" onclick={onclose}>Close</button>
    </div>

    <!-- Tabs -->
    <div class="flex bg-surface-base/50">
      {#each (['full-model', 'from-node'] as const) as tab (tab)}
        <button
          class="flex-1 px-2 py-2.5 text-xs text-center transition-all duration-100 {activeTab === tab ? 'text-accent border-b-2 border-accent' : 'text-content-secondary/40 hover:text-content-secondary'}"
          disabled={tab === 'from-node' && !endNodeId}
          onclick={() => { activeTab = tab; }}
        >
          {TAB_LABELS[tab]}
        </button>
      {/each}
    </div>

      <div class="p-4 space-y-3">
        {#if activeTab === 'from-node' && endNodeName}
          <div class="text-xs text-gray-400">
            End node: <span class="text-gray-200 font-mono">{endNodeName}</span>
          </div>
        {/if}

        <label class="block text-xs">
          <span class="text-gray-400">Search for:</span>
          <select
            value={bisectStore.searchFor}
            onchange={onSearchForChange}
            class="w-full mt-1 px-2 py-1.5 bg-[--bg-panel] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none"
          >
            {#each Object.entries(searchForLabels) as [value, label]}
              <option {value}>{label}</option>
            {/each}
          </select>
        </label>

        {#if bisectStore.searchFor === 'accuracy_drop'}
          <label class="block text-xs">
            <span class="text-gray-400">Metric:</span>
            <select
              value={bisectStore.metric}
              onchange={onMetricChange}
              class="w-full mt-1 px-2 py-1.5 bg-[--bg-panel] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none"
            >
              {#each Object.entries(metricLabels) as [value, label]}
                <option {value}>{label}</option>
              {/each}
            </select>
          </label>

          <label class="block text-xs">
            <span class="text-gray-400">Threshold:</span>
            <input
              type="number"
              step="0.01"
              bind:value={bisectStore.threshold}
              onwheel={(e) => {
                e.preventDefault();
                const step = e.shiftKey ? 0.1 : 0.01;
                const delta = e.deltaY < 0 ? step : -step;
                bisectStore.threshold = Math.round((bisectStore.threshold + delta) * 1000) / 1000;
              }}
              class="w-full mt-1 px-2 py-1.5 bg-[--bg-panel] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none font-mono"
            />
          </label>
        {/if}

        <div class="text-xs text-gray-500">
          {#if activeTab === 'from-node' && endNodeName}
            Range: start &rarr; {endNodeName}
          {:else}
            Range: {graphStore.activeSubSessionId ? 'sub-model' : 'full model'} (auto)
          {/if}
        </div>

        <button
          class="w-full py-2 bg-accent hover:bg-accent-hover disabled:bg-[--bg-panel] disabled:text-content-secondary rounded text-sm font-medium transition-colors"
          disabled={bisectStore.busy}
          onclick={startBisect}
        >
          {bisectStore.busy ? 'Starting...' : 'Start Bisection'}
        </button>

        {#if bisectStore.error}
          <div class="text-xs text-red-400 mt-1">{bisectStore.error}</div>
        {/if}
      </div>
  </div>
</div>
