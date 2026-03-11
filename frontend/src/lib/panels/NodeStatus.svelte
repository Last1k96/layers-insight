<script lang="ts">
  import { graphStore } from '../stores/graph.svelte';
  import { queueStore } from '../stores/queue.svelte';
  import { sessionStore } from '../stores/session.svelte';
  import AccuracyView from '../accuracy/AccuracyView.svelte';
  import BatchQueue from './BatchQueue.svelte';

  let selectedNode = $derived(graphStore.selectedNode);
  let nodeStatus = $derived(graphStore.selectedNodeStatus);

  let showAccuracyView = $state(false);
  let showBatchQueue = $state(false);
  let cutting = $state(false);

  function handleRerun() {
    if (nodeStatus?.taskId) {
      queueStore.rerun(nodeStatus.taskId);
    }
  }

  function handleCancel() {
    if (nodeStatus?.taskId) {
      queueStore.cancel(nodeStatus.taskId);
    }
  }

  function handleDeepAccuracy() {
    showAccuracyView = true;
  }

  async function handleCut(cutType: 'output' | 'input') {
    const session = sessionStore.currentSession;
    if (!session || !selectedNode) return;

    cutting = true;
    try {
      const res = await fetch(`/api/sessions/${session.id}/cut`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          node_name: selectedNode.name,
          cut_type: cutType,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(err.detail || 'Cut failed');
      }
    } catch (e) {
      console.error('Cut failed:', e);
    } finally {
      cutting = false;
    }
  }

  function formatValue(v: number | undefined | null): string {
    if (v === undefined || v === null) return '-';
    if (Math.abs(v) < 0.0001 && v !== 0) return v.toExponential(4);
    return v.toFixed(6);
  }
</script>

<div class="p-3 overflow-y-auto h-full text-sm">
  {#if !selectedNode}
    <div class="text-gray-500 text-center py-8">
      Select a node to view details
    </div>
  {:else}
    <!-- Node Info -->
    <div class="mb-4">
      <div class="font-mono font-medium text-gray-200 break-all">{selectedNode.name}</div>
      <div class="text-gray-400 mt-1">{selectedNode.type}</div>
      {#if selectedNode.shape}
        <div class="text-gray-500 text-xs mt-1">
          Shape: [{selectedNode.shape.join(', ')}]
        </div>
      {/if}
      {#if selectedNode.element_type}
        <div class="text-gray-500 text-xs">
          Type: {selectedNode.element_type}
        </div>
      {/if}
    </div>

    {#if !nodeStatus}
      <div class="text-gray-500 text-xs">Not yet inferred</div>

    {:else if nodeStatus.status === 'waiting'}
      <div class="flex items-center gap-2 text-amber-400">
        <div class="w-2 h-2 rounded-full bg-amber-400"></div>
        Waiting in queue
      </div>
      <button
        class="mt-3 w-full py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
        onclick={handleCancel}
      >
        Cancel
      </button>

    {:else if nodeStatus.status === 'executing'}
      <div class="flex items-center gap-2 text-blue-400">
        <div class="w-2 h-2 rounded-full bg-blue-400 pulse-ring"></div>
        Executing
      </div>
      {#if nodeStatus.stage}
        <div class="text-gray-400 text-xs mt-1">{nodeStatus.stage}</div>
      {/if}

    {:else if nodeStatus.status === 'success'}
      <div class="flex items-center gap-2 text-green-400 mb-3">
        <div class="w-2 h-2 rounded-full bg-green-400"></div>
        Success
      </div>

      {#if nodeStatus.metrics}
        <div class="space-y-2">
          <h4 class="text-xs font-medium text-gray-400 uppercase tracking-wider">Accuracy</h4>
          <table class="w-full text-xs">
            <tbody>
              <tr class="border-b border-gray-700/50">
                <td class="py-1 text-gray-400">MSE</td>
                <td class="py-1 text-right font-mono">{formatValue(nodeStatus.metrics.mse)}</td>
              </tr>
              <tr class="border-b border-gray-700/50">
                <td class="py-1 text-gray-400">Max Abs Diff</td>
                <td class="py-1 text-right font-mono">{formatValue(nodeStatus.metrics.max_abs_diff)}</td>
              </tr>
              <tr>
                <td class="py-1 text-gray-400">Cosine Sim</td>
                <td class="py-1 text-right font-mono">{formatValue(nodeStatus.metrics.cosine_similarity)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      {/if}

      <!-- Per-device results -->
      {#if nodeStatus.mainResult || nodeStatus.refResult}
        <div class="mt-3 space-y-2">
          <h4 class="text-xs font-medium text-gray-400 uppercase tracking-wider">Device Outputs</h4>
          {#each [nodeStatus.mainResult, nodeStatus.refResult].filter(Boolean) as result}
            {#if result}
              <div class="bg-gray-900/50 rounded p-2">
                <div class="font-medium text-xs text-gray-300 mb-1">{result.device}</div>
                <div class="grid grid-cols-2 gap-x-3 text-xs">
                  <span class="text-gray-500">Min</span>
                  <span class="font-mono text-right">{formatValue(result.min_val)}</span>
                  <span class="text-gray-500">Max</span>
                  <span class="font-mono text-right">{formatValue(result.max_val)}</span>
                  <span class="text-gray-500">Mean</span>
                  <span class="font-mono text-right">{formatValue(result.mean_val)}</span>
                  <span class="text-gray-500">Std</span>
                  <span class="font-mono text-right">{formatValue(result.std_val)}</span>
                </div>
              </div>
            {/if}
          {/each}
        </div>
      {/if}

      <!-- Phase 2 actions -->
      <div class="mt-3 space-y-2">
        <button
          class="w-full py-1.5 bg-indigo-700 hover:bg-indigo-600 rounded text-xs transition-colors"
          onclick={handleDeepAccuracy}
        >
          Deep Accuracy View
        </button>
        <button
          class="w-full py-1.5 bg-amber-700 hover:bg-amber-600 rounded text-xs transition-colors"
          onclick={() => handleCut('output')}
        >
          Make Output Node
        </button>
        <button
          class="w-full py-1.5 bg-purple-700 hover:bg-purple-600 rounded text-xs transition-colors"
          onclick={() => handleCut('input')}
        >
          Make Input Node
        </button>
        <button
          class="w-full py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
          onclick={() => showBatchQueue = true}
        >
          Batch Queue
        </button>
      </div>

      <button
        class="mt-3 w-full py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
        onclick={handleRerun}
      >
        Re-run
      </button>

    {:else if nodeStatus.status === 'failed'}
      <div class="flex items-center gap-2 text-red-400 mb-3">
        <div class="w-2 h-2 rounded-full bg-red-400"></div>
        Failed
      </div>
      {#if nodeStatus.stage}
        <div class="text-gray-400 text-xs mb-2">Stage: {nodeStatus.stage}</div>
      {/if}
      {#if nodeStatus.errorDetail}
        <pre class="bg-gray-900 rounded p-2 text-xs text-red-300 overflow-x-auto max-h-48 whitespace-pre-wrap font-mono">{nodeStatus.errorDetail}</pre>
      {/if}
      <button
        class="mt-3 w-full py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
        onclick={handleRerun}
      >
        Retry
      </button>
    {/if}

    <!-- Inputs -->
    {#if selectedNode.inputs && selectedNode.inputs.length > 0}
      <details class="mt-4" open>
        <summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
          Inputs ({selectedNode.inputs.length})
        </summary>
        <div class="mt-1 space-y-1">
          {#each selectedNode.inputs as inp, idx}
            <div class="bg-gray-900/50 rounded p-2 text-xs">
              <div class="flex items-center gap-1.5">
                <span class="text-gray-600 font-mono w-4 shrink-0">{idx}</span>
                {#if inp.is_const}
                  <span class="px-1 py-0.5 bg-amber-900/40 text-amber-400 rounded text-[10px] leading-none">const</span>
                {/if}
                <span class="text-gray-300 font-mono truncate" title={inp.name}>{inp.name}</span>
              </div>
              {#if inp.shape}
                <div class="text-gray-500 ml-5 mt-0.5">
                  [{inp.shape.join(', ')}]{#if inp.element_type} <span class="text-gray-600">{inp.element_type}</span>{/if}
                </div>
              {/if}
            </div>
          {/each}
        </div>
      </details>
    {/if}

    <!-- Attributes -->
    {#if selectedNode.attributes && Object.keys(selectedNode.attributes).length > 0}
      <details class="mt-4">
        <summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
          Attributes ({Object.keys(selectedNode.attributes).length})
        </summary>
        <div class="mt-1 bg-gray-900/50 rounded p-2 text-xs font-mono max-h-32 overflow-y-auto">
          {#each Object.entries(selectedNode.attributes) as [key, value]}
            <div class="flex justify-between gap-2">
              <span class="text-gray-500">{key}</span>
              <span class="text-gray-300 truncate">{String(value)}</span>
            </div>
          {/each}
        </div>
      </details>
    {/if}
  {/if}
</div>

<!-- Overlays rendered with fixed positioning -->
{#if showAccuracyView && nodeStatus?.taskId && selectedNode}
  <div class="fixed inset-0 z-50">
    <AccuracyView
      taskId={nodeStatus.taskId}
      nodeId={selectedNode.name}
      onclose={() => showAccuracyView = false}
    />
  </div>
{/if}

{#if showBatchQueue && selectedNode}
  <div class="fixed inset-0 z-50 bg-black/30">
    <BatchQueue
      nodeId={selectedNode.id}
      nodeName={selectedNode.name}
      onclose={() => showBatchQueue = false}
    />
  </div>
{/if}
