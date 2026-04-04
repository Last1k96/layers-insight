<script lang="ts">
  import { queueStore, type SortColumn } from '../stores/queue.svelte';
  import { graphStore } from '../stores/graph.svelte';
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
</script>

<div class="flex flex-col h-full" role="listbox" tabindex="-1">
  <!-- Header controls -->
  <div class="flex items-center gap-1 px-2 py-1.5 border-b border-[--border-color] shrink-0">
    <button
      class="flex items-center justify-center w-6 h-6 rounded hover:bg-[--bg-menu] transition-colors"
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
      class="flex items-center justify-center w-6 h-6 rounded transition-colors hover:bg-[--bg-menu]"
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
      <span class="ml-1 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-yellow-900/40 text-yellow-400 rounded">Paused</span>
    {/if}
    <div class="flex-1"></div>
    <span class="text-[10px] text-gray-500">{queueStore.filteredTasks.length} tasks</span>
  </div>

  <!-- Column headers -->
  <div class="flex items-center gap-2 px-3 py-1 border-b border-[--border-color] text-[10px] text-gray-500 uppercase tracking-wider shrink-0">
    <div class="w-2 shrink-0"></div>
    <button
      class="flex-1 text-left flex items-center gap-0.5 hover:text-gray-300 transition-colors cursor-pointer"
      class:text-gray-300={queueStore.sortColumn === 'topo'}
      onclick={() => handleSort('topo')}
    >
      Node {sortArrow('topo')}
    </button>
    <button
      class="w-16 text-right shrink-0 flex items-center justify-end gap-0.5 hover:text-gray-300 transition-colors cursor-pointer"
      class:text-gray-300={queueStore.sortColumn === 'type'}
      onclick={() => handleSort('type')}
    >
      Type {sortArrow('type')}
    </button>
    <button
      class="w-14 text-right shrink-0 flex items-center justify-end gap-0.5 hover:text-gray-300 transition-colors cursor-pointer"
      class:text-gray-300={queueStore.sortColumn === 'cosine'}
      onclick={() => handleSort('cosine')}
    >
      Cos {sortArrow('cosine')}
    </button>
    <button
      class="w-16 text-right shrink-0 flex items-center justify-end gap-0.5 hover:text-gray-300 transition-colors cursor-pointer"
      class:text-gray-300={queueStore.sortColumn === 'mse'}
      onclick={() => handleSort('mse')}
    >
      MSE {sortArrow('mse')}
    </button>
    <div class="w-5 shrink-0"></div>
  </div>

  <!-- Task list -->
  <div class="flex-1 overflow-y-auto">
    {#if queueStore.filteredTasks.length === 0}
      <div class="p-4 text-center text-gray-500 text-sm">
        No tasks yet. Click a node to start inference.
      </div>
    {:else}
      {#each queueStore.filteredTasks as task, i (task.task_id)}
        {#if i > 0 && (task.status === 'executing' || task.status === 'waiting') && (queueStore.filteredTasks[i - 1].status === 'success' || queueStore.filteredTasks[i - 1].status === 'failed')}
          <div class="flex items-center gap-2 px-3 py-1 bg-[--bg-menu]">
            <div class="flex-1 h-px bg-gray-600"></div>
            <span class="text-[10px] text-gray-500 uppercase tracking-wider shrink-0">Pending</span>
            <div class="flex-1 h-px bg-gray-600"></div>
          </div>
        {/if}
        <div
          class="w-full text-left px-3 py-2 text-sm border-b border-[--border-color] hover:bg-[--bg-menu] flex items-center gap-2 transition-colors cursor-pointer outline-none"
          class:bg-[--bg-menu]={i === queueStore.selectedIndex}
          role="button"
          tabindex="-1"
          onclick={() => { queueStore.selectedIndex = i; selectTask(task); }}
          onkeydown={(e) => { if (e.key === 'Enter') { queueStore.selectedIndex = i; selectTask(task); }}}
        >
          <!-- Status dot -->
          <div
            class="w-2 h-2 rounded-full shrink-0"
            class:pulse-ring={task.status === 'executing'}
            style:background-color={getStatusColor(task.status)}
          ></div>

          <!-- Node name -->
          <span class="flex-1 truncate font-mono text-xs">{task.node_name}</span>

          <!-- Op type -->
          <span class="text-gray-500 text-xs shrink-0 w-16 text-right truncate">{task.node_type}</span>

          <!-- Cosine for completed -->
          <span class="text-xs font-mono w-14 text-right shrink-0">
            {#if task.status === 'success' && task.metrics}
              <span style:color={cosineColor(task.metrics.cosine_similarity)}>{formatCosine(task.metrics.cosine_similarity)}</span>
            {/if}
          </span>

          <!-- MSE for completed -->
          <span class="text-xs font-mono w-16 text-right shrink-0">
            {#if task.status === 'success' && task.metrics}
              <span style:color={mseColor(task.metrics.mse)}>{formatMse(task.metrics.mse)}</span>
            {/if}
          </span>

          <!-- Delete button -->
          <button
            class="text-gray-500 hover:text-red-400 text-xs px-1 shrink-0"
            title="Delete"
            onclick={(e) => { e.stopPropagation(); queueStore.deleteTask(task.task_id); }}
          >
            &#x2715;
          </button>
        </div>
      {/each}
    {/if}
  </div>

  <!-- Filter -->
  <div class="border-t border-[--border-color] p-2 shrink-0">
    {#if advancedFilterStore.active}
      <AdvancedFilter ontoggle={() => advancedFilterStore.active = false} />
    {:else}
      <div class="flex gap-1">
        <input
          type="text"
          bind:value={queueStore.filterText}
          placeholder="Filter by name or type..."
          class="flex-1 px-2 py-1 bg-[--bg-panel] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none"
        />
        <button
          class="px-1.5 py-1 rounded border border-[--border-color] hover:bg-[--bg-menu] transition-colors text-gray-400 hover:text-gray-200 shrink-0"
          title="Advanced filter"
          onclick={() => advancedFilterStore.active = true}
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M1 3h14M3 8h10M5 13h6" />
          </svg>
        </button>
      </div>
      <div class="flex gap-1 mt-1">
        {#each ['all', 'success', 'failed'] as status (status)}
          <button
            class="px-2 py-0.5 text-xs rounded transition-colors"
            class:bg-surface-elevated={queueStore.filterStatus === status}
            class:text-content-primary={queueStore.filterStatus === status}
            class:text-content-secondary={queueStore.filterStatus !== status}
            onclick={() => queueStore.filterStatus = status as any}
          >
            {status === 'all' ? 'All' : status.charAt(0).toUpperCase() + status.slice(1)}
          </button>
        {/each}
      </div>
    {/if}
  </div>
</div>
