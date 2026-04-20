<script lang="ts">
  import { untrack } from 'svelte';
  import { bisectStore } from '../stores/bisect.svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { sessionStore } from '../stores/session.svelte';
  import { rangeScroll } from '../accuracy/rangeScroll';

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

  type BisectTab = 'all-outputs' | 'from-node';
  const TAB_LABELS: Record<BisectTab, string> = {
    'all-outputs': 'All Outputs',
    'from-node': 'From Node',
  };

  let activeTab = $state<BisectTab>('all-outputs');

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

    let ok = false;
    if (activeTab === 'all-outputs') {
      ok = await bisectStore.startAllOutputs(session.id, graphStore.activeSubSessionId);
    } else {
      ok = await bisectStore.start(session.id, graphStore.activeSubSessionId, endNodeId);
    }
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

<div class="bp-backdrop" onclick={onclose} role="presentation"></div>
<div class="bp-modal">
  <div class="bp-shell">
    <div class="bp-header">
      <div class="bp-header__title">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true">
          <path d="M8 2v12M4 6l4-4 4 4" />
        </svg>
        <h3>Bisect</h3>
      </div>
      <button class="ll-icon-btn ll-icon-btn--sm" onclick={onclose} aria-label="Close">
        <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="4" y1="4" x2="12" y2="12"/><line x1="12" y1="4" x2="4" y2="12"/></svg>
      </button>
    </div>

    <div class="bp-tabs">
      {#each (['all-outputs', 'from-node'] as const) as tab (tab)}
        <button
          class="bp-tab"
          class:bp-tab--active={activeTab === tab}
          disabled={tab === 'from-node' && !endNodeId}
          onclick={() => { activeTab = tab; }}
        >
          {TAB_LABELS[tab]}
        </button>
      {/each}
    </div>

    <div class="bp-body">
      {#if activeTab === 'all-outputs'}
        <p class="bp-hint">
          Start one bisect job per model output. Correct outputs finish in 1 inference. Graph-aware search follows actual data paths.
        </p>
      {:else if activeTab === 'from-node' && endNodeName}
        <div class="bp-hint">
          End node: <span class="bp-node">{endNodeName}</span>
        </div>
      {/if}

      <label class="bp-field">
        <span class="bp-label">Search for</span>
        <select
          use:rangeScroll
          value={bisectStore.searchFor}
          onchange={onSearchForChange}
          class="ll-field"
        >
          {#each Object.entries(searchForLabels) as [value, label]}
            <option {value}>{label}</option>
          {/each}
        </select>
      </label>

      {#if bisectStore.searchFor === 'accuracy_drop'}
        <label class="bp-field">
          <span class="bp-label">Metric</span>
          <select
            use:rangeScroll
            value={bisectStore.metric}
            onchange={onMetricChange}
            class="ll-field"
          >
            {#each Object.entries(metricLabels) as [value, label]}
              <option {value}>{label}</option>
            {/each}
          </select>
        </label>

        <label class="bp-field">
          <span class="bp-label">Threshold</span>
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
            class="ll-field ll-field--mono"
          />
        </label>
      {/if}

      <div class="bp-summary">
        {#if activeTab === 'all-outputs'}
          <span class="ll-chip ll-chip--accent ll-chip--tiny">Graph-aware</span>
          <span>per-output bisect</span>
        {:else if activeTab === 'from-node' && endNodeName}
          <span class="ll-chip ll-chip--accent ll-chip--tiny">Range</span>
          <span>start → <span class="bp-node">{endNodeName}</span></span>
        {/if}
      </div>

      <button
        class="ll-btn ll-btn--primary ll-btn--block"
        disabled={bisectStore.busy}
        onclick={startBisect}
      >
        {#if bisectStore.busy}
          <svg class="ll-spin" width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-dasharray="31.4 31.4" stroke-linecap="round"/>
          </svg>
          Starting…
        {:else}
          {activeTab === 'all-outputs' ? 'Bisect All Outputs' : 'Start Bisection'}
        {/if}
      </button>

      {#if bisectStore.error}
        <div class="bp-error">{bisectStore.error}</div>
      {/if}
    </div>
  </div>
</div>

<style>
  .bp-backdrop {
    position: fixed;
    inset: 0;
    z-index: 59;
    background: rgba(10, 12, 20, 0.55);
    backdrop-filter: blur(2px);
    -webkit-backdrop-filter: blur(2px);
  }
  .bp-modal {
    position: fixed;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    z-index: 60;
    width: 440px;
    max-width: 92vw;
  }
  .bp-shell {
    background: rgba(35, 38, 54, 0.97);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-modal);
    overflow: hidden;
  }

  .bp-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 14px;
    border-bottom: 1px solid var(--border-soft);
  }
  .bp-header__title {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: var(--accent-soft);
  }
  .bp-header__title h3 {
    margin: 0;
    font-family: var(--font-display);
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    letter-spacing: 0.01em;
  }

  .bp-tabs {
    display: flex;
    background: var(--bg-primary);
  }
  .bp-tab {
    flex: 1;
    padding: 10px 12px;
    font-size: 11.5px;
    font-weight: 500;
    color: var(--text-muted);
    border: 0;
    background: transparent;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    transition: color var(--dur-fast) ease, background var(--dur-fast) ease, border-color var(--dur-fast) ease;
  }
  .bp-tab:hover:not(:disabled) { color: var(--text-primary); background: var(--accent-bg-soft); }
  .bp-tab--active {
    color: var(--accent);
    border-bottom-color: var(--accent);
    background: var(--accent-bg-soft);
  }
  .bp-tab:disabled { opacity: 0.4; cursor: default; }

  .bp-body {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .bp-hint {
    margin: 0;
    font-size: 12px;
    color: var(--text-muted-strong);
    line-height: 1.5;
  }
  .bp-node {
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 11.5px;
  }

  .bp-field { display: flex; flex-direction: column; gap: 5px; }
  .bp-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
  }

  .bp-summary {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    color: var(--text-muted-strong);
  }

  .bp-error {
    margin-top: 2px;
    font-size: 11.5px;
    color: var(--status-err);
    background: var(--status-err-bg);
    border: 1px solid var(--status-err-border);
    padding: 7px 10px;
    border-radius: var(--radius-md);
  }
</style>
