<script lang="ts">
  import { queueStore, type SortColumn } from '../stores/queue.svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { bisectStore } from '../stores/bisect.svelte';
  import { centerOnNode, refreshRenderer } from '../graph/renderer';
  import { getStatusColor } from '../graph/opColors';
  import { advancedFilterStore } from '../stores/advancedFilter.svelte';
  import AdvancedFilter from '../components/AdvancedFilter.svelte';
  import type { InferenceTask } from '../stores/types';

  function selectTask(task: InferenceTask) {
    graphStore.selectNode(task.node_id);
    centerOnNode(task.node_id);
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

  function mseColor(mse: number): string {
    if (mse < 0.001) return '#34C77B';   // green
    if (mse <= 0.01) return '#E5A820';   // yellow
    return '#E54D4D';                     // red
  }

  function cosineColor(cos: number): string {
    if (cos > 0.999) return '#34C77B';   // green
    if (cos >= 0.99) return '#E5A820';   // yellow
    return '#E54D4D';                     // red
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

  function handleCancelAll() {
    if (confirm('Cancel all waiting tasks?')) {
      queueStore.cancelAll();
    }
  }

  function handleBisectStop() {
    bisectStore.stop();
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

  function handleBisectClick(e: MouseEvent) {
    const bj = bisectStore.job;
    if (!bj) return;

    // Determine which node to target: found_node when done, else current_node
    const nodeName = bj.status === 'done' ? bj.found_node : bj.current_node;
    if (!nodeName) return;

    const node = graphStore.graphData?.nodes.find(n => n.name === nodeName);
    if (!node) return;

    graphStore.selectNode(node.id);
    // Ctrl+click (or Cmd on Mac) centers the view on that node
    if (e.ctrlKey || e.metaKey) {
      centerOnNode(node.id);
    }
    refreshRenderer();
  }
</script>

{#snippet bisectRow(bj: import('../stores/types').BisectQueueItem)}
  <div
    class="w-full relative px-3 py-2 text-sm flex items-center gap-2 overflow-hidden cursor-pointer row-hover"
    role="button"
    tabindex="-1"
    onclick={handleBisectClick}
    onkeydown={(e) => { if (e.key === 'Enter') handleBisectClick(e as unknown as MouseEvent); }}
  >
    {#if bj.total_steps > 0}
      <div
        class="absolute inset-0 transition-all duration-500 ease-out"
        style:background-color={bisectProgressColor(bj.status)}
        style:width="{Math.min(100, (bj.step / bj.total_steps) * 100)}%"
      ></div>
    {/if}
    <div class="relative flex items-center gap-2 w-full">
      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" class="shrink-0" style:color={bisectIconColor(bj.status)}>
        <path d="M8 2v12M4 6l4-4 4 4" />
      </svg>
      <span class="flex-1 text-xs font-medium truncate" style:color={bisectTextColor(bj.status)}>
        {#if bj.status === 'running'}
          Bisecting... Step {bj.step}/{bj.total_steps}
          {#if bj.current_node}
            <span class="text-gray-500 font-mono ml-1">{bj.current_node}</span>
          {/if}
        {:else if bj.status === 'paused'}
          Bisect paused at step {bj.step}/{bj.total_steps}
        {:else if bj.status === 'done' && bj.found_node}
          Found: {bj.found_node}
        {:else if bj.status === 'error'}
          Bisect error{bj.error ? `: ${bj.error}` : ''}
        {/if}
      </span>
      <button
        class="text-gray-600 hover:text-red-400 text-xs px-1 shrink-0 transition-colors relative"
        title="Stop bisect"
        onclick={(e) => { e.stopPropagation(); handleBisectStop(); }}
      >&#x2715;</button>
    </div>
  </div>
{/snippet}

<div class="flex flex-col h-full" role="listbox" tabindex="-1">
  <!-- Header controls -->
  <div class="flex items-center gap-1.5 px-3 py-2 shrink-0">
    <button
      class="flex items-center justify-center w-7 h-7 rounded-lg hover:bg-surface-elevated transition-all duration-100 active:scale-95"
      title={queueStore.paused ? 'Resume queue' : 'Pause queue'}
      onclick={handlePauseResume}
    >
      {#if queueStore.paused}
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
          <polygon points="4,2 14,8 4,14" />
        </svg>
      {:else}
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
          <rect x="3" y="2" width="4" height="12" />
          <rect x="9" y="2" width="4" height="12" />
        </svg>
      {/if}
    </button>
    <button
      class="flex items-center justify-center w-7 h-7 rounded-lg transition-all duration-100 hover:bg-surface-elevated active:scale-95"
      class:text-gray-600={queueStore.waitingCount === 0}
      class:pointer-events-none={queueStore.waitingCount === 0}
      title="Cancel all waiting tasks"
      onclick={handleCancelAll}
    >
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="4" y1="4" x2="12" y2="12" />
        <line x1="12" y1="4" x2="4" y2="12" />
      </svg>
    </button>
    {#if queueStore.paused}
      <span class="ml-1 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-yellow-500/15 text-yellow-400 rounded-full">Paused</span>
    {/if}
    <div class="flex-1"></div>
    <span class="text-[11px] text-content-secondary/60 tabular-nums">{queueStore.filteredTasks.length} tasks</span>
  </div>

  <!-- Column headers -->
  <div class="flex items-center gap-2 px-3 py-1.5 text-[10px] text-content-secondary/50 uppercase tracking-wider shrink-0 bg-surface-base/50">
    <div class="w-2.5 shrink-0"></div>
    <button
      class="flex-1 text-left flex items-center gap-0.5 hover:text-content-secondary transition-colors cursor-pointer whitespace-nowrap"
      class:text-content-secondary={queueStore.sortColumn === 'topo'}
      onclick={() => handleSort('topo')}
    >
      Node {sortArrow('topo')}
    </button>
    <button
      class="text-right shrink-0 flex items-center justify-end gap-0.5 hover:text-content-secondary transition-colors cursor-pointer whitespace-nowrap"
      class:text-content-secondary={queueStore.sortColumn === 'type'}
      onclick={() => handleSort('type')}
    >
      Type {sortArrow('type')}
    </button>
    <button
      class="w-14 text-right shrink-0 flex items-center justify-end gap-0.5 hover:text-content-secondary transition-colors cursor-pointer"
      class:text-content-secondary={queueStore.sortColumn === 'cosine'}
      onclick={() => handleSort('cosine')}
    >
      Cos {sortArrow('cosine')}
    </button>
    <button
      class="w-16 text-right shrink-0 flex items-center justify-end gap-0.5 hover:text-content-secondary transition-colors cursor-pointer"
      class:text-content-secondary={queueStore.sortColumn === 'mse'}
      onclick={() => handleSort('mse')}
    >
      MSE {sortArrow('mse')}
    </button>
    <div class="w-5 shrink-0"></div>
  </div>

  <!-- Task list -->
  <div class="flex-1 overflow-y-auto">
    {#if queueStore.filteredTasks.length === 0 && !bisectStore.job}
      <div class="flex flex-col items-center justify-center py-12 px-4">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="text-content-secondary/20 mb-3">
          <circle cx="12" cy="12" r="10" />
          <path d="M8 12h8M12 8v8" stroke-linecap="round" />
        </svg>
        <span class="text-content-secondary/40 text-xs">Click a node to start inference</span>
      </div>
    {:else}
      {#each queueStore.filteredTasks as task, i (task.task_id)}
        {#if i > 0 && (task.status === 'executing' || task.status === 'waiting') && (queueStore.filteredTasks[i - 1].status === 'success' || queueStore.filteredTasks[i - 1].status === 'failed')}
          <div class="flex items-center gap-2 px-3 py-1">
            <div class="flex-1 h-px bg-content-secondary/10"></div>
            <span class="text-[10px] text-content-secondary/30 uppercase tracking-wider shrink-0">Queue</span>
            <div class="flex-1 h-px bg-content-secondary/10"></div>
          </div>
          <!-- Bisect row: right below the Queue divider -->
          {#if bisectStore.job && bisectStore.job.status !== 'stopped'}
            {@const bj = bisectStore.job}
            {@render bisectRow(bj)}
          {/if}
        {/if}
        <div
          class="w-full text-left px-3 py-2 text-sm flex items-center gap-2 cursor-pointer outline-none row-hover"
          style:background-color={i === queueStore.selectedIndex ? 'rgba(76, 141, 255, 0.06)' : undefined}
          role="button"
          tabindex="-1"
          onclick={() => { queueStore.selectedIndex = i; selectTask(task); }}
          onkeydown={(e) => { if (e.key === 'Enter') { queueStore.selectedIndex = i; selectTask(task); }}}
        >
          <!-- Status dot -->
          <div
            class="w-2.5 h-2.5 rounded-full shrink-0"
            class:pulse-ring={task.status === 'executing'}
            class:status-glow={task.status === 'executing'}
            style:background-color={getStatusColor(task.status)}
          ></div>

          <!-- Node name -->
          <span class="flex-1 truncate font-mono text-xs">{task.node_name}</span>

          <!-- Op type -->
          <span class="text-content-secondary/40 text-xs shrink-0 text-right whitespace-nowrap">{task.node_type}</span>

          <!-- Cosine for completed -->
          <span class="text-xs font-mono w-14 text-right shrink-0 tabular-nums">
            {#if task.status === 'success' && task.metrics}
              <span style:color={cosineColor(task.metrics.cosine_similarity)}>{formatCosine(task.metrics.cosine_similarity)}</span>
            {/if}
          </span>

          <!-- MSE for completed -->
          <span class="text-xs font-mono w-16 text-right shrink-0 tabular-nums">
            {#if task.status === 'success' && task.metrics}
              <span style:color={mseColor(task.metrics.mse)}>{formatMse(task.metrics.mse)}</span>
            {/if}
          </span>

          <!-- Delete button -->
          <button
            class="text-content-secondary/30 hover:text-red-400 text-xs px-1 shrink-0 transition-colors"
            title="Delete"
            onclick={(e) => { e.stopPropagation(); queueStore.deleteTask(task.task_id); }}
          >
            &#x2715;
          </button>
        </div>
      {/each}

      <!-- Bisect row fallback: when the Queue divider didn't render (no done+pending mix) -->
      {#if bisectStore.job && bisectStore.job.status !== 'stopped'}
        {@const hasDivider = queueStore.filteredTasks.some((t, i) =>
          i > 0 && (t.status === 'executing' || t.status === 'waiting') &&
          (queueStore.filteredTasks[i - 1].status === 'success' || queueStore.filteredTasks[i - 1].status === 'failed')
        )}
        {#if !hasDivider}
          {@const bj = bisectStore.job}
          {@render bisectRow(bj)}
        {/if}
      {/if}
    {/if}
  </div>

  <!-- Filter -->
  <div class="p-2.5 shrink-0 bg-[--bg-panel]">
    {#if advancedFilterStore.active}
      <AdvancedFilter ontoggle={() => advancedFilterStore.active = false} />
    {:else}
      <div class="flex gap-1.5">
        <input
          type="text"
          bind:value={queueStore.filterText}
          placeholder="Filter by name or type..."
          class="flex-1 px-2.5 py-1.5 bg-[--bg-input] rounded-lg text-xs placeholder:text-content-secondary/30 focus:outline-none focus:ring-2 focus:ring-accent/30 transition-shadow"
        />
        <button
          class="px-2 py-1.5 rounded-lg hover:bg-surface-elevated transition-all duration-100 text-content-secondary/50 hover:text-content-secondary active:scale-95 shrink-0"
          title="Advanced filter"
          onclick={() => advancedFilterStore.active = true}
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M1 3h14M3 8h10M5 13h6" />
          </svg>
        </button>
      </div>
      <div class="flex gap-1 mt-1.5">
        {#each ['all', 'success', 'failed'] as status (status)}
          <button
            class="px-2.5 py-1 text-xs rounded-md transition-all duration-100 {queueStore.filterStatus === status ? 'text-accent' : 'text-content-secondary hover:text-content-primary'}"
            style:background-color={queueStore.filterStatus === status ? 'rgba(76, 141, 255, 0.1)' : undefined}
            onclick={() => queueStore.filterStatus = status as any}
          >
            {status === 'all' ? 'All' : status.charAt(0).toUpperCase() + status.slice(1)}
          </button>
        {/each}
      </div>
    {/if}
  </div>
</div>
