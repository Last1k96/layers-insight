<script lang="ts">
  import { queueStore } from '../stores/queue.svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { centerOnNode, refreshRenderer } from '../graph/renderer';
  import { getStatusColor } from '../graph/opColors';
  import type { InferenceTask } from '../stores/types';

  function selectTask(task: InferenceTask) {
    graphStore.selectNode(task.node_id);
    centerOnNode(task.node_id);
    refreshRenderer();
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      const task = queueStore.moveSelection(e.key === 'ArrowDown' ? 1 : -1);
      if (task) {
        selectTask(task);
      }
    }
  }

  function formatMse(mse: number): string {
    if (mse < 0.0001) return mse.toExponential(2);
    return mse.toFixed(6);
  }
</script>

<div class="flex flex-col h-full" onkeydown={handleKeydown} tabindex="-1">
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
          class="w-full text-left px-3 py-2 text-sm border-b border-[--border-color] hover:bg-[--bg-menu] flex items-center gap-2 transition-colors cursor-pointer"
          class:bg-[--bg-menu]={i === queueStore.selectedIndex}
          role="button"
          tabindex="0"
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
          <span class="text-gray-500 text-xs shrink-0">{task.node_type}</span>

          <!-- MSE for completed -->
          {#if task.status === 'success' && task.metrics}
            <span class="text-xs text-gray-400 font-mono">{formatMse(task.metrics.mse)}</span>
          {/if}

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

  <!-- Simple filter -->
  <div class="border-t border-[--border-color] p-2 shrink-0">
    <input
      type="text"
      bind:value={queueStore.filterText}
      placeholder="Filter by name or type..."
      class="w-full px-2 py-1 bg-[--bg-panel] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none"
    />
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
  </div>
</div>
