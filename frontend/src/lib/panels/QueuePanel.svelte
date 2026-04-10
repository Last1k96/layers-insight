<script lang="ts">
  import { queueStore, type SortColumn } from '../stores/queue.svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { bisectStore } from '../stores/bisect.svelte';
  import { configStore } from '../stores/config.svelte';
  import { centerOnNode, refreshRenderer, setHoveredNode } from '../graph/renderer';
  import { getStatusColor } from '../graph/opColors';
  import { getAccuracyColor } from '../utils/accuracyColors';
  import { logStore } from '../stores/log.svelte';
  import { advancedFilterStore } from '../stores/advancedFilter.svelte';
  import AdvancedFilter from '../components/AdvancedFilter.svelte';
  import type { InferenceTask } from '../stores/types';
  import { tick } from 'svelte';

  let listEl = $state<HTMLDivElement | null>(null);
  let atBottom = $state(true);

  function onListScroll() {
    if (!listEl) return;
    atBottom = listEl.scrollTop + listEl.clientHeight >= listEl.scrollHeight - 8;
  }

  $effect(() => {
    // Track task list length to trigger auto-scroll
    queueStore.filteredTasks.length;
    bisectStore.hasJobs;
    if (atBottom && listEl) {
      tick().then(() => {
        if (listEl) listEl.scrollTop = listEl.scrollHeight;
      });
    }
  });

  let {
    onbatchinfer = () => {},
    onbisect = () => {},
    ontogglelog = () => {},
  }: {
    onbatchinfer?: () => void;
    onbisect?: () => void;
    ontogglelog?: () => void;
  } = $props();

  function selectTask(task: InferenceTask, e?: MouseEvent) {
    graphStore.selectNode(task.node_id);
    if (e && (e.ctrlKey || e.metaKey)) {
      centerOnNode(task.node_id);
    }
    refreshRenderer();
  }


  function formatMse(mse: number): string {
    if (mse < 0.0001) return mse.toExponential(1);
    if (mse < 0.01) return mse.toExponential(1);
    return mse.toFixed(4);
  }

  function formatCosine(cos: number): string {
    return cos.toFixed(4);
  }

  function metricColor(value: number, metric: 'cosine_similarity' | 'mse' | 'max_abs_diff'): string {
    return getAccuracyColor(metric, value, configStore.accuracyRanges[metric]);
  }

  function sortArrow(column: SortColumn): string {
    if (queueStore.sortColumn !== column) return '';
    return queueStore.sortDirection === 'asc' ? '\u25B2' : '\u25BC';
  }

  function handleSort(column: SortColumn) {
    queueStore.toggleSort(column);
  }

  function handlePauseResume() {
    if (queueStore.paused) {
      queueStore.resumeQueue();
    } else {
      queueStore.pauseQueue();
    }
  }

  let confirmingCancelAll = $state(false);

  function handleCancelAll() {
    confirmingCancelAll = true;
  }

  function confirmCancelAll() {
    queueStore.cancelAll();
    confirmingCancelAll = false;
  }

  function dismissCancelAll() {
    confirmingCancelAll = false;
  }

  let bisectCollapsed = $state<Record<string, boolean>>({});

  function isBisectCollapsed(jobId: string): boolean {
    return bisectCollapsed[jobId] ?? false;
  }

  function toggleBisectCollapsed(jobId: string): void {
    bisectCollapsed = { ...bisectCollapsed, [jobId]: !isBisectCollapsed(jobId) };
  }

  function bisectProgressColor(status: string): string {
    if (status === 'running') return 'rgba(59, 130, 246, 0.1)';
    if (status === 'paused') return 'rgba(234, 179, 8, 0.1)';
    if (status === 'done') return 'rgba(34, 197, 94, 0.1)';
    if (status === 'error') return 'rgba(239, 68, 68, 0.1)';
    return 'transparent';
  }

  function bisectIconColor(status: string): string {
    if (status === 'running') return '#60a5fa';
    if (status === 'paused') return '#facc15';
    if (status === 'done') return '#4ade80';
    if (status === 'error') return '#f87171';
    return '#9ca3af';
  }

  function bisectTextColor(status: string): string {
    if (status === 'running') return '#93c5fd';
    if (status === 'paused') return '#fde047';
    if (status === 'done') return '#86efac';
    if (status === 'error') return '#fca5a5';
    return '#d1d5db';
  }

  function statusLabel(status: string): string {
    if (status === 'executing') return 'RUN';
    if (status === 'waiting') return 'WAIT';
    if (status === 'success') return 'OK';
    if (status === 'failed') return 'ERR';
    return '';
  }

  function handleBisectClick(e: MouseEvent, bj: import('../stores/types').BisectQueueItem) {
    const nodeName = bj.found_node || bj.current_node;
    if (!nodeName) return;

    const node = graphStore.graphData?.nodes.find(n => n.name === nodeName);
    if (!node) return;

    graphStore.selectNode(node.id);
    if (e.ctrlKey || e.metaKey) {
      centerOnNode(node.id);
    }
    refreshRenderer();
  }
</script>

<div class="q-root" role="listbox" tabindex="-1">
  <!-- Toolbar -->
  <div class="q-toolbar">
    <div class="q-toolbar-left">
      <button
        class="q-icon-btn"
        class:q-icon-btn--disabled={queueStore.pauseTransitioning}
        title={queueStore.paused ? 'Resume queue' : 'Pause queue'}
        disabled={queueStore.pauseTransitioning}
        onclick={handlePauseResume}
      >
        {#if queueStore.pauseTransitioning}
          <svg class="animate-spin" width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="8" cy="8" r="6" stroke-dasharray="28" stroke-dashoffset="7" />
          </svg>
        {:else if queueStore.paused}
          <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><polygon points="4,2 14,8 4,14" /></svg>
        {:else}
          <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
            <rect x="3" y="2" width="4" height="12" rx="1" />
            <rect x="9" y="2" width="4" height="12" rx="1" />
          </svg>
        {/if}
      </button>

      {#if confirmingCancelAll}
        <span class="q-cancel-prompt">Cancel all?</span>
        <button class="q-cancel-yes" onclick={confirmCancelAll}>Yes</button>
        <button class="q-cancel-no" onclick={dismissCancelAll}>No</button>
      {:else}
        <button
          class="q-icon-btn"
          class:q-icon-btn--disabled={queueStore.waitingCount === 0}
          title="Cancel all waiting tasks"
          onclick={handleCancelAll}
        >
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
            <line x1="4" y1="4" x2="12" y2="12" /><line x1="12" y1="4" x2="4" y2="12" />
          </svg>
        </button>
      {/if}

      {#if queueStore.pauseTransitioning}
        <span class="q-badge q-badge--blue animate-pulse">
          {queueStore.paused ? 'Resuming' : 'Pausing'}
        </span>
      {:else if queueStore.paused}
        <span class="q-badge q-badge--yellow">Paused</span>
      {/if}
    </div>

    <span class="q-task-count">{queueStore.filteredTasks.length}</span>
  </div>

  <!-- Actions -->
  <div class="q-actions">
    <button
      class="q-action-btn {bisectStore.isActive ? 'q-action-btn--active' : ''}"
      onclick={onbisect}
    >
      <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
        <path d="M8 2v12M4 6l4-4 4 4" />
      </svg>
      Bisect{bisectStore.isActive ? ' \u25CF' : ''}
    </button>
    <button class="q-action-btn" onclick={onbatchinfer}>
      <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
        <path d="M2 4h12M2 8h12M2 12h12" />
      </svg>
      Batch
    </button>
    <button
      class="q-action-btn {logStore.visible ? 'q-action-btn--active' : ''}"
      onclick={ontogglelog}
    >
      <svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
        <path d="M2 3h12M2 7h8M2 11h10" />
      </svg>
      Logs{logStore.visible ? ' \u25CF' : ''}
    </button>
  </div>

  <!-- Filter -->
  <div class="q-filter">
    {#if advancedFilterStore.active}
      <AdvancedFilter ontoggle={() => advancedFilterStore.active = false} />
    {:else}
      <div class="q-filter-row">
        <div class="q-search-wrap">
          <svg class="q-search-icon" width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8">
            <circle cx="6.5" cy="6.5" r="4.5" /><line x1="10" y1="10" x2="14" y2="14" stroke-linecap="round" />
          </svg>
          <input
            type="text"
            bind:value={queueStore.filterText}
            placeholder="Filter..."
            class="q-search-input"
          />
        </div>
        <button
          class="q-icon-btn q-icon-btn--subtle"
          title="Advanced filter"
          onclick={() => advancedFilterStore.active = true}
        >
          <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
            <path d="M1 3h14M3 8h10M5 13h6" />
          </svg>
        </button>
      </div>
      <div class="q-status-tabs">
        {#each ['all', 'success', 'failed'] as status (status)}
          <button
            class="q-status-tab {queueStore.filterStatus === status ? 'q-status-tab--active' : ''}"
            onclick={() => queueStore.filterStatus = status as any}
          >
            {#if status === 'success'}
              <span class="q-status-dot" style:background="#34C77B"></span>
            {:else if status === 'failed'}
              <span class="q-status-dot" style:background="#E54D4D"></span>
            {/if}
            {status === 'all' ? 'All' : status.charAt(0).toUpperCase() + status.slice(1)}
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Column headers -->
  <div class="q-colheaders">
    <div class="q-col-status"></div>
    <button
      class="q-colheader q-col-node"
      class:q-colheader--active={queueStore.sortColumn === 'topo'}
      onclick={() => handleSort('topo')}
    >
      Node <span class="q-sort-arrow">{sortArrow('topo')}</span>
    </button>
    <button
      class="q-colheader q-col-type"
      class:q-colheader--active={queueStore.sortColumn === 'type'}
      onclick={() => handleSort('type')}
    >
      Type <span class="q-sort-arrow">{sortArrow('type')}</span>
    </button>
    <button
      class="q-colheader q-col-cos"
      class:q-colheader--active={queueStore.sortColumn === 'cosine'}
      onclick={() => handleSort('cosine')}
    >
      Cos <span class="q-sort-arrow">{sortArrow('cosine')}</span>
    </button>
    <button
      class="q-colheader q-col-mse"
      class:q-colheader--active={queueStore.sortColumn === 'mse'}
      onclick={() => handleSort('mse')}
    >
      MSE <span class="q-sort-arrow">{sortArrow('mse')}</span>
    </button>
    <div class="q-col-del"></div>
  </div>

  <!-- Task list -->
  <div class="q-list" bind:this={listEl} onscroll={onListScroll}>
    {#if queueStore.filteredTasks.length === 0 && !bisectStore.hasJobs && queueStore.bisectTasks.length === 0}
      <div class="q-empty">
        <div class="q-empty-rings">
          <div class="q-empty-ring q-empty-ring--outer"></div>
          <div class="q-empty-ring q-empty-ring--inner"></div>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="q-empty-icon">
            <path d="M12 5v14M5 12h14" stroke-linecap="round" />
          </svg>
        </div>
        <span class="q-empty-text">Click a node to infer</span>
      </div>
    {:else}
      <!-- Regular tasks -->
      {#each queueStore.filteredTasks as task, i (task.task_id)}
        {#if i > 0 && (task.status === 'executing' || task.status === 'waiting') && (queueStore.filteredTasks[i - 1].status === 'success' || queueStore.filteredTasks[i - 1].status === 'failed')}
          <div class="q-separator">
            <div class="q-separator-line"></div>
            <span class="q-separator-label">Queue</span>
            <div class="q-separator-line"></div>
          </div>
        {/if}
        <div
          class="q-task"
          class:q-task--selected={i === queueStore.selectedIndex}
          class:q-task--hovered={graphStore.hoveredNodeId === task.node_id && i !== queueStore.selectedIndex}
          class:q-task--executing={task.status === 'executing'}
          role="button"
          tabindex="-1"
          onclick={(e) => { queueStore.selectedIndex = i; selectTask(task, e); }}
          onkeydown={(e) => { if (e.key === 'Enter') { queueStore.selectedIndex = i; selectTask(task); }}}
          onmouseenter={() => { setHoveredNode(task.node_id); }}
          onmouseleave={() => { setHoveredNode(null); }}
        >
          <div class="q-task-status">
            <div
              class="q-task-dot"
              class:pulse-ring={task.status === 'executing'}
              class:status-glow={task.status === 'executing'}
              style:background-color={getStatusColor(task.status)}
            ></div>
          </div>
          <span class="q-task-name">{task.node_name}</span>
          <span class="q-task-type">{task.node_type}</span>
          <span class="q-task-metric q-col-cos">
            {#if task.status === 'success' && task.metrics}
              <span style:color={configStore.accuracyMetric === 'cosine_similarity' ? metricColor(task.metrics.cosine_similarity, 'cosine_similarity') : ''}>{formatCosine(task.metrics.cosine_similarity)}</span>
            {/if}
          </span>
          <span class="q-task-metric q-col-mse">
            {#if task.status === 'success' && task.metrics}
              <span style:color={configStore.accuracyMetric === 'mse' ? metricColor(task.metrics.mse, 'mse') : ''}>{formatMse(task.metrics.mse)}</span>
            {/if}
          </span>
          <button
            class="q-task-delete"
            title="Delete"
            onclick={(e) => { e.stopPropagation(); queueStore.deleteTask(task.task_id); }}
          >
            &#x2715;
          </button>
        </div>
      {/each}

      <!-- Active bisect jobs -->
      {#each bisectStore.activeJobs as bj (bj.job_id)}
        {@const jobTasks = queueStore.bisectTasksForJob(bj.job_id)}
        {@const isActive = bj.status === 'running' || bj.status === 'paused'}
        <div
          class="q-bisect-header"
          role="button" tabindex="-1"
          onclick={(e) => handleBisectClick(e, bj)}
          onkeydown={(e) => { if (e.key === 'Enter') handleBisectClick(e as unknown as MouseEvent, bj); }}
        >
          {#if bj.total_steps > 0}
            <div
              class="q-bisect-progress"
              style:background-color={bisectProgressColor(bj.status)}
              style:width={bj.status === 'done' || bj.status === 'error' ? '100%' : `${Math.min(100, (bj.step / bj.total_steps) * 100)}%`}
            ></div>
          {/if}
          <div class="q-bisect-content">
            <button class="q-bisect-chevron"
              aria-label="Toggle collapse"
              onclick={(e) => { e.stopPropagation(); toggleBisectCollapsed(bj.job_id); }}
            >
              <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor"
                class="transition-transform duration-150 {isBisectCollapsed(bj.job_id) ? '' : 'rotate-90'}"
              ><path d="M3 1l5 4-5 4z" /></svg>
            </button>
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" class="shrink-0" style:color={bisectIconColor(bj.status)}>
              <path d="M8 2v12M4 6l4-4 4 4" />
            </svg>
            <span class="q-bisect-label" style:color={bisectTextColor(bj.status)}>
              {#if bj.output_node}<span class="q-bisect-output">[{bj.output_node}]</span>{/if}
              {#if bj.status === 'running'}
                Bisecting... Step {bj.step}/{bj.total_steps}
                {#if bj.current_node}<span class="q-bisect-node">{bj.current_node}</span>{/if}
              {:else if bj.status === 'paused'}
                Bisect paused at step {bj.step}/{bj.total_steps}
              {/if}
            </span>
            {#if bisectStore.busy}
              <span class="q-bisect-tag q-bisect-tag--busy animate-pulse">Working...</span>
            {:else}
              <button class="q-bisect-tag q-bisect-tag--merge"
                onclick={(e) => { e.stopPropagation(); bisectStore.stopAndMerge(bj.job_id); }}
              >Stop &amp; Merge</button>
              <button class="q-bisect-tag q-bisect-tag--discard"
                onclick={(e) => { e.stopPropagation(); bisectStore.stopAndDiscard(bj.job_id); }}
              >Stop &amp; Discard</button>
            {/if}
          </div>
        </div>
        {#if !isBisectCollapsed(bj.job_id)}
          {#each jobTasks as task (task.task_id)}
            <div class="q-task q-task--nested"
              class:q-task--selected={queueStore.selectedTaskId === task.task_id}
              class:q-task--hovered={graphStore.hoveredNodeId === task.node_id && queueStore.selectedTaskId !== task.task_id}
              role="button" tabindex="-1"
              onclick={(e) => { queueStore.selectedTaskId = task.task_id; selectTask(task, e); }}
              onkeydown={(e) => { if (e.key === 'Enter') { queueStore.selectedTaskId = task.task_id; selectTask(task); }}}
              onmouseenter={() => { setHoveredNode(task.node_id); }}
              onmouseleave={() => { setHoveredNode(null); }}
            >
              <div class="q-task-status">
                <div class="q-task-dot" class:pulse-ring={task.status === 'executing'} class:status-glow={task.status === 'executing'}
                  style:background-color={getStatusColor(task.status)}></div>
              </div>
              <span class="q-task-name">{task.node_name}</span>
              <span class="q-task-type">{task.node_type}</span>
              <span class="q-task-metric q-col-cos">
                {#if task.status === 'success' && task.metrics}<span style:color={configStore.accuracyMetric === 'cosine_similarity' ? metricColor(task.metrics.cosine_similarity, 'cosine_similarity') : ''}>{formatCosine(task.metrics.cosine_similarity)}</span>{/if}
              </span>
              <span class="q-task-metric q-col-mse">
                {#if task.status === 'success' && task.metrics}<span style:color={configStore.accuracyMetric === 'mse' ? metricColor(task.metrics.mse, 'mse') : ''}>{formatMse(task.metrics.mse)}</span>{/if}
              </span>
              <div class="q-col-del"></div>
            </div>
          {/each}
        {/if}
      {/each}

      <!-- Finished bisect jobs -->
      {#each bisectStore.finishedJobs as bj (bj.job_id)}
        {@const jobTasks = queueStore.bisectTasksForJob(bj.job_id)}
        <div
          class="q-bisect-header"
          role="button" tabindex="-1"
          onclick={(e) => handleBisectClick(e, bj)}
          onkeydown={(e) => { if (e.key === 'Enter') handleBisectClick(e as unknown as MouseEvent, bj); }}
        >
          {#if bj.total_steps > 0}
            <div class="q-bisect-progress"
              style:background-color={bisectProgressColor(bj.status)}
              style:width="100%"
            ></div>
          {/if}
          <div class="q-bisect-content">
            <button class="q-bisect-chevron"
              aria-label="Toggle collapse"
              onclick={(e) => { e.stopPropagation(); toggleBisectCollapsed(bj.job_id); }}
            >
              <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor"
                class="transition-transform duration-150 {isBisectCollapsed(bj.job_id) ? '' : 'rotate-90'}"
              ><path d="M3 1l5 4-5 4z" /></svg>
            </button>
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" class="shrink-0" style:color={bisectIconColor(bj.status)}>
              <path d="M8 2v12M4 6l4-4 4 4" />
            </svg>
            <span class="q-bisect-label" style:color={bisectTextColor(bj.status)}>
              {#if bj.output_node}<span class="q-bisect-output">[{bj.output_node}]</span>{/if}
              {#if bj.status === 'done' && bj.found_node}
                Found: {bj.found_node}
              {:else if bj.status === 'done' && !bj.found_node}
                Output OK
              {:else if bj.status === 'error'}
                Bisect error{bj.error ? `: ${bj.error}` : ''}
              {:else}
                Bisection ({jobTasks.length})
              {/if}
            </span>
            {#if bisectStore.busy}
              <span class="q-bisect-tag q-bisect-tag--busy animate-pulse">Working...</span>
            {:else}
              <button class="q-bisect-tag q-bisect-tag--merge"
                onclick={(e) => { e.stopPropagation(); bisectStore.merge(bj.job_id); }}
              >Merge</button>
              <button class="q-bisect-tag q-bisect-tag--discard"
                onclick={(e) => { e.stopPropagation(); bisectStore.discard(bj.job_id); }}
              >Discard</button>
            {/if}
          </div>
        </div>
        {#if !isBisectCollapsed(bj.job_id)}
          {#each jobTasks as task (task.task_id)}
            <div class="q-task q-task--nested"
              class:q-task--selected={queueStore.selectedTaskId === task.task_id}
              class:q-task--hovered={graphStore.hoveredNodeId === task.node_id && queueStore.selectedTaskId !== task.task_id}
              role="button" tabindex="-1"
              onclick={(e) => { queueStore.selectedTaskId = task.task_id; selectTask(task, e); }}
              onkeydown={(e) => { if (e.key === 'Enter') { queueStore.selectedTaskId = task.task_id; selectTask(task); }}}
              onmouseenter={() => { setHoveredNode(task.node_id); }}
              onmouseleave={() => { setHoveredNode(null); }}
            >
              <div class="q-task-status">
                <div class="q-task-dot" class:pulse-ring={task.status === 'executing'} class:status-glow={task.status === 'executing'}
                  style:background-color={getStatusColor(task.status)}></div>
              </div>
              <span class="q-task-name">{task.node_name}</span>
              <span class="q-task-type">{task.node_type}</span>
              <span class="q-task-metric q-col-cos">
                {#if task.status === 'success' && task.metrics}<span style:color={configStore.accuracyMetric === 'cosine_similarity' ? metricColor(task.metrics.cosine_similarity, 'cosine_similarity') : ''}>{formatCosine(task.metrics.cosine_similarity)}</span>{/if}
              </span>
              <span class="q-task-metric q-col-mse">
                {#if task.status === 'success' && task.metrics}<span style:color={configStore.accuracyMetric === 'mse' ? metricColor(task.metrics.mse, 'mse') : ''}>{formatMse(task.metrics.mse)}</span>{/if}
              </span>
              <div class="q-col-del"></div>
            </div>
          {/each}
        {/if}
      {/each}
    {/if}
  </div>
</div>

<style>
  /* ── Root ── */
  .q-root {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: linear-gradient(180deg, rgba(35, 38, 54, 0.3) 0%, transparent 120px);
  }

  /* ── Toolbar ── */
  .q-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 12px;
    flex-shrink: 0;
  }
  .q-toolbar-left {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .q-icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    color: var(--text-secondary);
    transition: all 0.12s ease;
    border: 1px solid transparent;
  }
  .q-icon-btn:hover {
    background: rgba(76, 141, 255, 0.06);
    border-color: rgba(76, 141, 255, 0.12);
    color: var(--text-primary);
  }
  .q-icon-btn:active {
    transform: scale(0.93);
  }
  .q-icon-btn--disabled {
    opacity: 0.35;
    pointer-events: none;
  }
  .q-icon-btn--subtle {
    width: 30px;
    height: 30px;
    color: rgba(155, 161, 181, 0.5);
  }
  .q-icon-btn--subtle:hover {
    color: var(--text-secondary);
  }

  .q-cancel-prompt {
    font-size: 11px;
    color: #f87171;
    font-weight: 500;
  }
  .q-cancel-yes {
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
    border-radius: 5px;
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    transition: background 0.12s;
  }
  .q-cancel-yes:hover { background: rgba(239, 68, 68, 0.25); }
  .q-cancel-no {
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
    border-radius: 5px;
    background: rgba(155, 161, 181, 0.08);
    color: var(--text-secondary);
    transition: all 0.12s;
  }
  .q-cancel-no:hover { color: var(--text-primary); }

  .q-badge {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 99px;
  }
  .q-badge--blue {
    background: rgba(76, 141, 255, 0.12);
    color: #60a5fa;
  }
  .q-badge--yellow {
    background: rgba(234, 179, 8, 0.12);
    color: #facc15;
  }

  .q-task-count {
    font-family: var(--font-mono);
    font-size: 11px;
    color: rgba(155, 161, 181, 0.4);
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
  }

  /* ── Actions ── */
  .q-actions {
    display: flex;
    gap: 4px;
    padding: 4px 12px 8px;
    flex-shrink: 0;
  }
  .q-action-btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    padding: 6px 0;
    font-size: 11px;
    font-weight: 500;
    color: rgba(155, 161, 181, 0.7);
    border-radius: 7px;
    border: 1px solid rgba(47, 51, 71, 0.6);
    background: rgba(27, 30, 43, 0.4);
    transition: all 0.12s ease;
  }
  .q-action-btn:hover {
    color: var(--text-primary);
    background: rgba(76, 141, 255, 0.04);
    border-color: rgba(76, 141, 255, 0.15);
  }
  .q-action-btn:active {
    transform: scale(0.97);
  }
  .q-action-btn--active {
    color: #60a5fa;
    border-color: rgba(96, 165, 250, 0.2);
    background: rgba(76, 141, 255, 0.06);
  }

  /* ── Filter ── */
  .q-filter {
    padding: 8px 12px;
    flex-shrink: 0;
    border-bottom: 1px solid rgba(47, 51, 71, 0.5);
  }
  .q-filter-row {
    display: flex;
    gap: 6px;
  }
  .q-search-wrap {
    flex: 1;
    position: relative;
    display: flex;
    align-items: center;
  }
  .q-search-icon {
    position: absolute;
    left: 9px;
    color: rgba(155, 161, 181, 0.3);
    pointer-events: none;
  }
  .q-search-input {
    width: 100%;
    padding: 7px 10px 7px 28px;
    background: var(--bg-input);
    border: 1px solid rgba(47, 51, 71, 0.5);
    border-radius: 8px;
    font-size: 12px;
    color: var(--text-primary);
    outline: none;
    transition: all 0.15s ease;
  }
  .q-search-input::placeholder {
    color: rgba(155, 161, 181, 0.25);
  }
  .q-search-input:focus {
    border-color: rgba(76, 141, 255, 0.3);
    box-shadow: 0 0 0 3px rgba(76, 141, 255, 0.06);
  }

  .q-status-tabs {
    display: flex;
    gap: 2px;
    margin-top: 6px;
  }
  .q-status-tab {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 500;
    color: rgba(155, 161, 181, 0.5);
    border-radius: 6px;
    transition: all 0.12s;
  }
  .q-status-tab:hover {
    color: var(--text-secondary);
    background: rgba(155, 161, 181, 0.04);
  }
  .q-status-tab--active {
    color: #4C8DFF;
    background: rgba(76, 141, 255, 0.08);
  }
  .q-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 99px;
    flex-shrink: 0;
  }

  /* ── Column Headers ── */
  .q-colheaders {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    flex-shrink: 0;
    border-bottom: 1px solid rgba(47, 51, 71, 0.3);
    background: rgba(27, 30, 43, 0.3);
  }
  .q-col-status { width: 12px; flex-shrink: 0; }
  .q-colheader {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(155, 161, 181, 0.35);
    cursor: pointer;
    transition: color 0.12s;
    display: flex;
    align-items: center;
    gap: 2px;
    white-space: nowrap;
  }
  .q-colheader:hover { color: rgba(155, 161, 181, 0.6); }
  .q-colheader--active { color: rgba(155, 161, 181, 0.6); }
  .q-col-node { flex: 1; text-align: left; }
  .q-col-type { flex-shrink: 0; text-align: right; justify-content: flex-end; }
  .q-col-cos { width: 56px; flex-shrink: 0; text-align: right; justify-content: flex-end; }
  .q-col-mse { width: 64px; flex-shrink: 0; text-align: right; justify-content: flex-end; }
  .q-col-del { width: 20px; flex-shrink: 0; }
  .q-sort-arrow { font-size: 8px; opacity: 0.7; }

  /* ── Task List ── */
  .q-list {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
  }

  /* ── Empty state ── */
  .q-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px 16px;
    gap: 14px;
  }
  .q-empty-rings {
    position: relative;
    width: 52px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .q-empty-ring {
    position: absolute;
    border-radius: 99px;
    border: 1px solid rgba(76, 141, 255, 0.1);
  }
  .q-empty-ring--outer {
    inset: 0;
    animation: ring-breathe 3s ease-in-out infinite;
  }
  .q-empty-ring--inner {
    inset: 8px;
    border-color: rgba(76, 141, 255, 0.15);
    animation: ring-breathe 3s ease-in-out infinite 0.5s;
  }
  .q-empty-icon {
    color: rgba(155, 161, 181, 0.2);
  }
  .q-empty-text {
    font-size: 12px;
    color: rgba(155, 161, 181, 0.3);
    font-weight: 500;
  }

  @keyframes ring-breathe {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.15); opacity: 0.4; }
  }

  /* ── Separator ── */
  .q-separator {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 12px;
  }
  .q-separator-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(76, 141, 255, 0.12), transparent);
  }
  .q-separator-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(76, 141, 255, 0.3);
    flex-shrink: 0;
  }

  /* ── Task Row ── */
  .q-task {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    cursor: pointer;
    outline: none;
    position: relative;
    transition: background-color 0.1s ease;
    border-left: 2px solid transparent;
  }
  .q-task:hover {
    background: rgba(76, 141, 255, 0.03);
  }
  .q-task--selected {
    background: rgba(76, 141, 255, 0.06);
    border-left-color: #4C8DFF;
  }
  .q-task--hovered {
    background: rgba(76, 141, 255, 0.03);
  }
  .q-task--executing {
    background: rgba(76, 141, 255, 0.02);
  }
  .q-task--nested {
    padding-left: 32px;
  }

  .q-task-status {
    width: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }
  .q-task-dot {
    width: 8px;
    height: 8px;
    border-radius: 99px;
    flex-shrink: 0;
    box-shadow: 0 0 4px rgba(0,0,0,0.2);
  }
  .q-task-name {
    flex: 1;
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .q-task-type {
    font-size: 11px;
    color: rgba(155, 161, 181, 0.35);
    flex-shrink: 0;
    text-align: right;
    white-space: nowrap;
  }
  .q-task-metric {
    font-family: var(--font-mono);
    font-size: 11px;
    text-align: right;
    flex-shrink: 0;
    font-variant-numeric: tabular-nums;
  }
  .q-task-delete {
    width: 20px;
    flex-shrink: 0;
    font-size: 11px;
    color: rgba(155, 161, 181, 0.15);
    text-align: center;
    transition: color 0.12s;
    border-radius: 4px;
    padding: 2px 0;
  }
  .q-task-delete:hover {
    color: #f87171;
  }

  /* ── Bisect Job ── */
  .q-bisect-header {
    position: relative;
    padding: 8px 12px;
    cursor: pointer;
    overflow: hidden;
    transition: background-color 0.1s ease;
  }
  .q-bisect-header:hover {
    background: rgba(76, 141, 255, 0.03);
  }
  .q-bisect-progress {
    position: absolute;
    inset: 0;
    right: auto;
    transition: all 0.5s ease-out;
  }
  .q-bisect-content {
    position: relative;
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
  }
  .q-bisect-chevron {
    color: rgba(155, 161, 181, 0.4);
    transition: color 0.12s;
    flex-shrink: 0;
  }
  .q-bisect-chevron:hover { color: var(--text-secondary); }
  .q-bisect-label {
    flex: 1;
    font-size: 12px;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }
  .q-bisect-node {
    color: rgba(156, 163, 175, 0.5);
    font-family: var(--font-mono);
    font-size: 11px;
    margin-left: 4px;
  }
  .q-bisect-output {
    color: rgba(147, 197, 253, 0.6);
    font-family: var(--font-mono);
    font-size: 10px;
    margin-right: 4px;
  }
  .q-bisect-tag {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 5px;
    flex-shrink: 0;
    transition: all 0.12s;
  }
  .q-bisect-tag--busy {
    background: rgba(155, 161, 181, 0.1);
    color: rgba(155, 161, 181, 0.5);
  }
  .q-bisect-tag--merge {
    background: rgba(74, 222, 128, 0.1);
    color: #4ade80;
  }
  .q-bisect-tag--merge:hover { background: rgba(74, 222, 128, 0.18); }
  .q-bisect-tag--discard {
    background: rgba(248, 113, 113, 0.1);
    color: #f87171;
  }
  .q-bisect-tag--discard:hover { background: rgba(248, 113, 113, 0.18); }
</style>
