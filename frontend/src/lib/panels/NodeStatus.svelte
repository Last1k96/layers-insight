<script lang="ts">
  import { graphStore } from '../stores/graph.svelte';
  import { queueStore } from '../stores/queue.svelte';
  import { sessionStore } from '../stores/session.svelte';
  import { refreshRenderer, centerOnNode, setHoveredNode } from '../graph/renderer';
  import type { ConstantData } from '../stores/types';

  let selectedNode = $derived(graphStore.selectedNode);
  let nodeStatus = $derived(graphStore.selectedNodeStatus);
  let nodeOverride = $derived(selectedNode ? graphStore.nodeOverrides.get(selectedNode.name) : undefined);

  /** Propagated (concrete) shape for the selected node, if available */
  let propagatedShape = $derived(
    selectedNode && graphStore.graphData?.propagated_shapes
      ? graphStore.graphData.propagated_shapes[selectedNode.name] ?? null
      : null
  );

  let outputs = $derived.by(() => {
    if (!selectedNode || !graphStore.graphData) return [];
    const edges = graphStore.graphData.edges.filter(e => e.source === selectedNode.id);
    const nodeMap = new Map(graphStore.graphData.nodes.map(n => [n.id, n]));
    return edges.map(e => ({
      source_port: e.source_port,
      target_port: e.target_port,
      targetNode: nodeMap.get(e.target),
      targetId: e.target,
    }));
  });

  // Constant data cache: const_node_name -> data
  let constDataCache = $state(new Map<string, ConstantData>());
  let constDataLoading = $state(new Set<string>());
  let constDataExpanded = $state(new Set<string>());

  async function toggleConstData(constNodeName: string) {
    const key = constNodeName;
    if (constDataExpanded.has(key)) {
      constDataExpanded = new Set([...constDataExpanded].filter(k => k !== key));
      return;
    }
    constDataExpanded = new Set([...constDataExpanded, key]);

    if (constDataCache.has(key)) return;

    const session = sessionStore.currentSession;
    if (!session) return;

    constDataLoading = new Set([...constDataLoading, key]);
    try {
      const res = await fetch(`/api/sessions/${session.id}/graph/constant/${encodeURIComponent(key)}`);
      if (res.ok) {
        const data: ConstantData = await res.json();
        constDataCache = new Map([...constDataCache, [key, data]]);
      }
    } catch (e) {
      console.error('Failed to fetch constant data:', e);
    } finally {
      constDataLoading = new Set([...constDataLoading].filter(k => k !== key));
    }
  }

  function formatFloat(v: number): string {
    if (v === 0) return '0';
    if (Math.abs(v) < 0.0001 || Math.abs(v) >= 1e6) return v.toExponential(3);
    return v.toPrecision(4);
  }

  let cutting = $state(false);

  function handleDelete() {
    if (nodeStatus?.taskId) {
      queueStore.deleteTask(nodeStatus.taskId);
    }
  }

  let {
    onshowaccuracy = () => {},
    onshowbatchqueue = () => {},
  }: {
    onshowaccuracy?: () => void;
    onshowbatchqueue?: () => void;
  } = $props();

  function handleDeepAccuracy() {
    onshowaccuracy();
  }

  async function handleCut(cutType: 'output' | 'input' | 'input_random') {
    const session = sessionStore.currentSession;
    if (!session || !selectedNode) return;

    cutting = true;
    try {
      const body: Record<string, unknown> = {
        node_name: selectedNode.name,
        cut_type: cutType,
      };
      if (graphStore.activeSubSessionId) {
        body.parent_sub_session_id = graphStore.activeSubSessionId;
      }
      const res = await fetch(`/api/sessions/${session.id}/cut`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
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

  function fmt4(v: number | undefined | null): string {
    if (v === undefined || v === null) return '-';
    if (Math.abs(v) < 0.0001 && v !== 0) return v.toExponential(4);
    return v.toFixed(4);
  }

  function diff(a: number | undefined | null, b: number | undefined | null): number | null {
    if (a == null || b == null) return null;
    return a - b;
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
      {#if (selectedNode.inputs && selectedNode.inputs.length > 0) || selectedNode.shape}
        <div class="mt-1.5 grid grid-cols-[auto_1fr] gap-x-1.5 gap-y-0.5 text-xs items-baseline">
          {#if selectedNode.inputs}
            {#each selectedNode.inputs as inp}
              {@const inpPropagated = graphStore.graphData?.propagated_shapes?.[inp.name] ?? null}
              {@const inpOrigShape = inp.shape}
              <span class="text-gray-600 shrink-0">{inp.is_const ? 'const' : 'input'}</span>
              <span>
                {#if inpPropagated && inpOrigShape}
                  <span class="font-mono text-gray-400">[{#each inpPropagated as dim, idx}{#if idx > 0}, {/if}{#if inpOrigShape[idx] !== undefined && typeof inpOrigShape[idx] === 'string'}<span class="text-yellow-400">{dim}</span>{:else}{dim}{/if}{/each}]</span>
                {:else if inpOrigShape}
                  <span class="font-mono text-gray-400">[{#each inpOrigShape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-yellow-400">?</span>{:else}{dim}{/if}{/each}]</span>
                {/if}
                {#if inp.element_type}<span class="text-gray-600 ml-1">{inp.element_type}</span>{/if}
              </span>
            {/each}
          {/if}
          {#if selectedNode.shape}
            <span class="text-gray-600 shrink-0">output</span>
            <span>
              {#if propagatedShape}
                <span class="font-mono text-gray-400">[{#each propagatedShape as dim, idx}{#if idx > 0}, {/if}{#if selectedNode.shape[idx] !== undefined && typeof selectedNode.shape[idx] === 'string'}<span class="text-yellow-400">{dim}</span>{:else}{dim}{/if}{/each}]</span>
              {:else}
                <span class="font-mono text-gray-400">[{#each selectedNode.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-yellow-400">?</span>{:else}{dim}{/if}{/each}]</span>
              {/if}
              {#if selectedNode.element_type}<span class="text-gray-600 ml-1">{selectedNode.element_type}</span>{/if}
            </span>
          {/if}
        </div>
      {/if}
    </div>

    {#if !nodeStatus}
      <div class="space-y-2">
        <button
          class="w-full py-1.5 bg-accent hover:bg-accent-hover rounded text-xs transition-colors"
          onclick={() => {
            const session = sessionStore.currentSession;
            if (session && selectedNode) {
              queueStore.enqueue(session.id, selectedNode.id, selectedNode.name, selectedNode.type, graphStore.activeSubSessionId);
            }
          }}
        >
          Infer
        </button>
        <button
          class="w-full py-1.5 bg-surface-elevated hover:bg-edge rounded text-xs transition-colors"
          onclick={onshowbatchqueue}
        >
          Batch Queue
        </button>
        <div class="flex gap-2">
          <button
            class="flex-1 py-1.5 bg-surface-elevated hover:bg-edge rounded text-xs transition-colors"
            onclick={() => handleCut('input_random')}
          >
            Convert into Parameter
          </button>
          <button
            class="flex-1 py-1.5 bg-surface-elevated hover:bg-edge rounded text-xs transition-colors"
            onclick={() => handleCut('output')}
          >
            Convert into Output
          </button>
        </div>
      </div>

    {:else if nodeStatus.status === 'waiting'}
      <div class="flex items-center gap-2 text-amber-400">
        <div class="w-2 h-2 rounded-full bg-amber-400"></div>
        Waiting in queue
      </div>
      <button
        class="mt-3 w-full py-1.5 bg-[--bg-menu] hover:bg-[--bg-primary] rounded text-xs transition-colors text-red-400"
        onclick={handleDelete}
      >
        Delete
      </button>

    {:else if nodeStatus.status === 'executing'}
      <div class="flex items-center gap-2 text-blue-400">
        <div class="w-2 h-2 rounded-full bg-blue-400 pulse-ring"></div>
        Executing
      </div>
      {#if nodeStatus.stage}
        <div class="text-gray-400 text-xs mt-1">{nodeStatus.stage}</div>
      {/if}
      <button
        class="mt-3 w-full py-1.5 bg-[--bg-menu] hover:bg-[--bg-primary] rounded text-xs transition-colors text-red-400"
        onclick={handleDelete}
      >
        Delete
      </button>

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
              <tr class="border-b border-[--border-color]/50">
                <td class="py-1 text-gray-400">MSE</td>
                <td class="py-1 text-right font-mono">{formatValue(nodeStatus.metrics.mse)}</td>
              </tr>
              <tr class="border-b border-[--border-color]/50">
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
        {@const main = nodeStatus.mainResult}
        {@const ref = nodeStatus.refResult}
        <div class="mt-3">
          <h4 class="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1">Device Outputs</h4>
          <table class="w-full text-xs table-fixed">
            <colgroup>
              <col class="w-[20%]" />
              <col class="w-[26.6%]" />
              <col class="w-[26.6%]" />
              {#if main && ref}<col class="w-[26.6%]" />{/if}
            </colgroup>
            <thead>
              <tr class="text-gray-500">
                <th class="py-1 text-left font-normal"></th>
                {#if main}<th class="py-1 text-right font-normal">{main.device}</th>{/if}
                {#if ref}<th class="py-1 text-right font-normal">{ref.device}</th>{/if}
                {#if main && ref}<th class="py-1 text-right font-normal">Diff</th>{/if}
              </tr>
            </thead>
            <tbody>
              {#each [
                { label: 'Min', key: 'min_val' as const },
                { label: 'Max', key: 'max_val' as const },
                { label: 'Mean', key: 'mean_val' as const },
                { label: 'Std', key: 'std_val' as const },
              ] as row}
                <tr class="border-t border-[--border-color]/50">
                  <td class="py-1 text-gray-400">{row.label}</td>
                  {#if main}<td class="py-1 text-right font-mono truncate">{fmt4(main[row.key])}</td>{/if}
                  {#if ref}<td class="py-1 text-right font-mono truncate">{fmt4(ref[row.key])}</td>{/if}
                  {#if main && ref}
                    <td class="py-1 text-right font-mono text-yellow-400 truncate">{fmt4(diff(main[row.key], ref[row.key]))}</td>
                  {/if}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}

      <!-- Phase 2 actions -->
      <div class="mt-3 space-y-2">
        <button
          class="w-full py-1.5 bg-accent hover:bg-accent-hover rounded text-xs transition-colors"
          onclick={handleDeepAccuracy}
        >
          Deep Accuracy View
        </button>
        <button
          class="w-full py-1.5 bg-surface-elevated hover:bg-edge rounded text-xs transition-colors"
          onclick={onshowbatchqueue}
        >
          Batch Queue
        </button>
        <div class="flex gap-2">
          <button
            class="flex-1 py-1.5 bg-surface-elevated hover:bg-edge rounded text-xs transition-colors"
            onclick={() => handleCut('input')}
          >
            Convert into Parameter
          </button>
          <button
            class="flex-1 py-1.5 bg-surface-elevated hover:bg-edge rounded text-xs transition-colors"
            onclick={() => handleCut('output')}
          >
            Convert into Output
          </button>
        </div>
      </div>

      <button
        class="mt-3 w-full py-1.5 bg-[--bg-menu] hover:bg-[--bg-primary] rounded text-xs transition-colors text-red-400"
        onclick={handleDelete}
      >
        Delete
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
        <pre class="bg-[--bg-panel] rounded p-2 text-xs text-red-300 overflow-x-auto max-h-48 whitespace-pre-wrap font-mono">{nodeStatus.errorDetail}</pre>
      {/if}
      <button
        class="mt-3 w-full py-1.5 bg-[--bg-menu] hover:bg-[--bg-primary] rounded text-xs transition-colors text-red-400"
        onclick={handleDelete}
      >
        Delete
      </button>
    {/if}

    <!-- Attributes -->
    {#if selectedNode.attributes && Object.keys(selectedNode.attributes).length > 0}
      <details class="mt-4">
        <summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
          Attributes ({Object.keys(selectedNode.attributes).length})
        </summary>
        <div class="mt-1 bg-[--bg-panel] rounded p-2 text-xs font-mono max-h-32 overflow-y-auto">
          {#each Object.entries(selectedNode.attributes) as [key, value]}
            <div class="flex justify-between gap-2">
              <span class="text-gray-500">{key}</span>
              <span class="text-gray-300 truncate">{String(value)}</span>
            </div>
          {/each}
        </div>
      </details>
    {/if}

    <!-- Inputs -->
    {#if selectedNode.inputs && selectedNode.inputs.length > 0}
      <details class="mt-4" open>
        <summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
          Inputs ({selectedNode.inputs.length})
        </summary>
        <div class="mt-1 space-y-1">
          {#each selectedNode.inputs as inp, idx}
            {@const sourceNode = graphStore.graphData?.nodes.find(n => n.name === inp.name)}
            <div class="bg-[--bg-panel] rounded p-2 text-xs">
              <div class="flex items-center gap-1.5">
                <span class="text-gray-600 font-mono w-4 shrink-0">{idx}</span>
                {#if !inp.is_const}
                  <button
                    class="text-blue-400 hover:text-blue-300 font-mono truncate transition-colors text-left"
                    title={inp.name}
                    onmouseenter={() => {
                      if (sourceNode) setHoveredNode(sourceNode.id);
                    }}
                    onmouseleave={() => setHoveredNode(null)}
                    onclick={(e) => {
                      if (sourceNode) {
                        graphStore.selectNode(sourceNode.id);
                        refreshRenderer();
                        if (e.ctrlKey || e.metaKey) centerOnNode(sourceNode.id);
                      }
                    }}
                  >{inp.name}</button>
                {:else}
                  <span class="text-gray-300 font-mono truncate" title={inp.name}>{inp.name}</span>
                {/if}
              </div>
              {#if sourceNode}
                <div class="text-gray-500 ml-5 mt-0.5">{sourceNode.type}</div>
              {:else if inp.is_const}
                <div class="ml-5 mt-0.5"><span class="px-1 py-0.5 bg-amber-900/40 text-amber-400 rounded text-[10px] leading-none">const</span></div>
              {/if}
              {#if inp.shape}
                <div class="text-gray-500 ml-5 mt-0.5">
                  [{#each inp.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-yellow-400">{dim}</span>{:else}{dim}{/if}{/each}] {#if inp.element_type}<span class="text-gray-600">{inp.element_type}</span>{/if}
                </div>
              {/if}
              {#if inp.is_const && inp.const_node_name}
                <button
                  class="ml-5 mt-1 text-[10px] text-blue-400 hover:text-blue-300 transition-colors"
                  onclick={() => toggleConstData(inp.const_node_name!)}
                >
                  {constDataExpanded.has(inp.const_node_name) ? 'Hide data' : 'View data'}
                </button>
                {#if constDataExpanded.has(inp.const_node_name)}
                  {#if constDataLoading.has(inp.const_node_name)}
                    <div class="ml-5 mt-1 text-gray-500 text-[10px]">Loading...</div>
                  {:else if constDataCache.has(inp.const_node_name)}
                    {@const cd = constDataCache.get(inp.const_node_name)!}
                    <div class="ml-5 mt-1 space-y-1">
                      <div class="grid grid-cols-2 gap-x-3 text-[10px]">
                        <span class="text-gray-600">dtype</span>
                        <span class="text-gray-400 font-mono text-right">{cd.dtype}</span>
                        <span class="text-gray-600">min</span>
                        <span class="text-gray-400 font-mono text-right">{formatFloat(cd.stats.min)}</span>
                        <span class="text-gray-600">max</span>
                        <span class="text-gray-400 font-mono text-right">{formatFloat(cd.stats.max)}</span>
                        <span class="text-gray-600">mean</span>
                        <span class="text-gray-400 font-mono text-right">{formatFloat(cd.stats.mean)}</span>
                        <span class="text-gray-600">std</span>
                        <span class="text-gray-400 font-mono text-right">{formatFloat(cd.stats.std)}</span>
                      </div>
                      <details>
                        <summary class="text-[10px] text-gray-600 cursor-pointer hover:text-gray-500">
                          Values ({cd.total_elements}{cd.truncated ? ', truncated' : ''})
                        </summary>
                        <pre class="mt-1 bg-[--bg-panel] rounded p-1.5 text-[9px] text-gray-400 font-mono overflow-x-auto max-h-40 whitespace-pre-wrap leading-tight">{cd.data.map(v => formatFloat(v)).join(', ')}</pre>
                      </details>
                    </div>
                  {/if}
                {/if}
              {/if}
            </div>
          {/each}
        </div>
      </details>
    {/if}

    <!-- Outputs -->
    {#if outputs.length > 0}
      <details class="mt-4" open>
        <summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
          Outputs ({outputs.length})
        </summary>
        <div class="mt-1 space-y-1">
          {#each outputs as out}
            <div class="bg-[--bg-panel] rounded p-2 text-xs">
              <div class="flex items-center gap-1.5">
                <span class="text-gray-600 font-mono w-4 shrink-0">{out.source_port}</span>
                <button
                  class="text-blue-400 hover:text-blue-300 font-mono truncate transition-colors text-left"
                  title={out.targetNode?.name ?? out.targetId}
                  onmouseenter={() => setHoveredNode(out.targetId)}
                  onmouseleave={() => setHoveredNode(null)}
                  onclick={(e) => {
                    graphStore.selectNode(out.targetId);
                    refreshRenderer();
                    if (e.ctrlKey || e.metaKey) centerOnNode(out.targetId);
                  }}
                >
                  {out.targetNode?.name ?? out.targetId}
                </button>
              </div>
              {#if out.targetNode}
                <div class="text-gray-500 ml-5 mt-0.5">{out.targetNode.type}</div>
                {#if out.targetNode.shape}
                  <div class="text-gray-500 ml-5 mt-0.5">
                    [{#each out.targetNode.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-yellow-400">{dim}</span>{:else}{dim}{/if}{/each}] {#if out.targetNode.element_type}<span class="text-gray-600">{out.targetNode.element_type}</span>{/if}
                  </div>
                {/if}
              {/if}
            </div>
          {/each}
        </div>
      </details>
    {/if}
  {/if}
</div>

