<script lang="ts">
  import { graphStore } from '../stores/graph.svelte';
  import { queueStore } from '../stores/queue.svelte';
  import { sessionStore } from '../stores/session.svelte';
  import { refreshRenderer, centerOnNode, setHoveredNode, setHoveredEdge } from '../graph/renderer';
  import type { ConstantData } from '../stores/types';

  let selectedNode = $derived(graphStore.selectedNode);
  let nodeStatus = $derived(graphStore.selectedNodeStatus);
  let selectedEdge = $derived(graphStore.selectedEdge);
  let nodeOverride = $derived(selectedNode ? graphStore.nodeOverrides.get(selectedNode.name) : undefined);

  /** Propagated (concrete) shape for the selected node, if available */
  let propagatedShape = $derived(
    selectedNode && graphStore.graphData?.propagated_shapes
      ? graphStore.graphData.propagated_shapes[selectedNode.name] ?? null
      : null
  );

  let outputs = $derived.by(() => {
    if (!selectedNode || !graphStore.graphData) return [];
    const allEdges = graphStore.graphData.edges;
    const nodeMap = new Map(graphStore.graphData.nodes.map(n => [n.id, n]));
    const result: Array<{ source_port: number; target_port: number; targetNode: typeof graphStore.graphData.nodes[0] | undefined; targetId: string; edgeIndex: number }> = [];
    for (let i = 0; i < allEdges.length; i++) {
      const e = allEdges[i];
      if (e.source === selectedNode.id) {
        result.push({ source_port: e.source_port, target_port: e.target_port, targetNode: nodeMap.get(e.target), targetId: e.target, edgeIndex: i });
      }
    }
    return result;
  });

  /** Find the edge index for an input connection (source -> selectedNode). */
  function findInputEdgeIndex(sourceNodeId: string, targetPort: number): number | null {
    if (!graphStore.graphData) return null;
    const edges = graphStore.graphData.edges;
    for (let i = 0; i < edges.length; i++) {
      if (edges[i].source === sourceNodeId && edges[i].target === selectedNode?.id && edges[i].target_port === targetPort) return i;
    }
    return null;
  }

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
    if (Number.isInteger(v)) return v.toString();
    if (Math.abs(v) < 0.0001 || Math.abs(v) >= 1e6) return v.toExponential(3);
    return v.toPrecision(4);
  }

  let cutting = $state(false);
  let exporting = $state(false);
  let minimalModel = $state(true);

  async function handleExport() {
    const session = sessionStore.currentSession;
    if (!session || !nodeStatus?.taskId) return;

    exporting = true;
    try {
      let url = `/api/tensors/${session.id}/${nodeStatus.taskId}/export`;
      if (minimalModel) url += '?minimal_model=true';
      const a = document.createElement('a');
      a.href = url;
      a.download = '';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } finally {
      // Brief delay so the user sees the loading state
      setTimeout(() => { exporting = false; }, 1000);
    }
  }

  function handleDelete() {
    if (nodeStatus?.taskId) {
      queueStore.deleteTask(nodeStatus.taskId);
    }
  }

  let {
    onshowaccuracy = (_outputIdx?: number) => {},
    onshowbatchqueue = () => {},
  }: {
    onshowaccuracy?: (outputIdx?: number) => void;
    onshowbatchqueue?: () => void;
  } = $props();

  function handleDeepAccuracy(outputIdx?: number) {
    onshowaccuracy(outputIdx);
  }

  /** Number of outputs for the current task (1 for single-output nodes). */
  let outputCount = $derived(
    nodeStatus?.perOutputMetrics ? nodeStatus.perOutputMetrics.length : 1
  );

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
  {#if selectedEdge}
    <!-- Edge Info -->
    <div class="mb-4">
      <h4 class="text-[10px] font-medium text-content-secondary/40 uppercase tracking-wider mb-2">Edge Connection</h4>

      <!-- Source Node -->
      <div class="bg-surface-base rounded-lg p-2.5 text-xs">
        <div class="text-[10px] text-content-secondary/30 uppercase tracking-wider mb-1">Source</div>
        <div class="flex items-center gap-1.5">
          <span class="text-content-secondary/25 font-mono w-4 shrink-0">:{selectedEdge.edge.source_port}</span>
          {#if selectedEdge.sourceNode}
            <button
              class="text-accent hover:text-accent-hover font-mono truncate transition-colors text-left"
              onmouseenter={() => setHoveredNode(selectedEdge!.sourceNode!.id)}
              onmouseleave={() => setHoveredNode(null)}
              onclick={() => centerOnNode(selectedEdge!.sourceNode!.id)}
            >{selectedEdge.sourceNode.name}</button>
          {:else}
            <span class="text-content-secondary/50 font-mono truncate">{selectedEdge.edge.source}</span>
          {/if}
        </div>
        {#if selectedEdge.sourceNode}
          <div class="text-content-secondary/40 ml-5 mt-0.5">{selectedEdge.sourceNode.type}</div>
          {#if selectedEdge.sourceNode.shape}
            <div class="text-content-secondary/40 ml-5 mt-0.5">
              [{#each selectedEdge.sourceNode.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-yellow-400">{dim}</span>{:else}{dim}{/if}{/each}]
              {#if selectedEdge.sourceNode.element_type}<span class="text-content-secondary/25"> {selectedEdge.sourceNode.element_type}</span>{/if}
            </div>
          {/if}
        {/if}
      </div>

      <!-- Arrow -->
      <div class="flex justify-center py-1">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" class="text-content-secondary/20">
          <path d="M8 3v10M4 9l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>

      <!-- Target Node -->
      <div class="bg-surface-base rounded-lg p-2.5 text-xs">
        <div class="text-[10px] text-content-secondary/30 uppercase tracking-wider mb-1">Target</div>
        <div class="flex items-center gap-1.5">
          <span class="text-content-secondary/25 font-mono w-4 shrink-0">:{selectedEdge.edge.target_port}</span>
          {#if selectedEdge.targetNode}
            <button
              class="text-accent hover:text-accent-hover font-mono truncate transition-colors text-left"
              onmouseenter={() => setHoveredNode(selectedEdge!.targetNode!.id)}
              onmouseleave={() => setHoveredNode(null)}
              onclick={() => centerOnNode(selectedEdge!.targetNode!.id)}
            >{selectedEdge.targetNode.name}</button>
          {:else}
            <span class="text-content-secondary/50 font-mono truncate">{selectedEdge.edge.target}</span>
          {/if}
        </div>
        {#if selectedEdge.targetNode}
          <div class="text-content-secondary/40 ml-5 mt-0.5">{selectedEdge.targetNode.type}</div>
          {#if selectedEdge.targetNode.shape}
            <div class="text-content-secondary/40 ml-5 mt-0.5">
              [{#each selectedEdge.targetNode.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-yellow-400">{dim}</span>{:else}{dim}{/if}{/each}]
              {#if selectedEdge.targetNode.element_type}<span class="text-content-secondary/25"> {selectedEdge.targetNode.element_type}</span>{/if}
            </div>
          {/if}
        {/if}
      </div>
    </div>
  {:else if selectedNode}
    <!-- Node Info -->
    <div class="mb-4">
      <button
        class="font-mono font-medium text-[13px] text-content-primary break-all leading-snug text-left"
        onclick={() => centerOnNode(selectedNode!.id)}
      >{selectedNode.name}</button>
      <div class="text-content-secondary/50 text-xs mt-1">{selectedNode.type}</div>
      {#if (selectedNode.inputs && selectedNode.inputs.length > 0) || selectedNode.shape}
        <div class="mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 text-xs items-baseline">
          {#if selectedNode.inputs}
            {#each selectedNode.inputs as inp}
              {@const inpPropagated = graphStore.graphData?.propagated_shapes?.[inp.name] ?? null}
              {@const inpOrigShape = inp.shape}
              <span class="text-content-secondary/30 shrink-0">{inp.is_const ? 'const' : 'input'}</span>
              <span>
                {#if inpPropagated && inpOrigShape}
                  <span class="font-mono text-content-secondary/60">[{#each inpPropagated as dim, idx}{#if idx > 0}, {/if}{#if inpOrigShape[idx] !== undefined && typeof inpOrigShape[idx] === 'string'}<span class="text-yellow-400">{dim}</span>{:else}{dim}{/if}{/each}]</span>
                {:else if inpOrigShape}
                  <span class="font-mono text-content-secondary/60">[{#each inpOrigShape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-yellow-400">?</span>{:else}{dim}{/if}{/each}]</span>
                {/if}
                {#if inp.element_type}<span class="text-content-secondary/30 ml-1">{inp.element_type}</span>{/if}
              </span>
            {/each}
          {/if}
          {#if selectedNode.shape}
            <span class="text-content-secondary/30 shrink-0">output</span>
            <span>
              {#if propagatedShape}
                <span class="font-mono text-content-secondary/60">[{#each propagatedShape as dim, idx}{#if idx > 0}, {/if}{#if selectedNode.shape[idx] !== undefined && typeof selectedNode.shape[idx] === 'string'}<span class="text-yellow-400">{dim}</span>{:else}{dim}{/if}{/each}]</span>
              {:else}
                <span class="font-mono text-content-secondary/60">[{#each selectedNode.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-yellow-400">?</span>{:else}{dim}{/if}{/each}]</span>
              {/if}
              {#if selectedNode.element_type}<span class="text-content-secondary/30 ml-1">{selectedNode.element_type}</span>{/if}
            </span>
          {/if}
        </div>
      {/if}
    </div>

    {#if !nodeStatus}
      <div class="space-y-2">
        <button
          class="w-full py-2 bg-accent hover:bg-accent-hover rounded-lg text-xs font-medium transition-all duration-100 active:scale-[0.98]"
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
          class="w-full py-2 bg-surface-elevated hover:bg-edge rounded-lg text-xs transition-all duration-100 active:scale-[0.98]"
          onclick={onshowbatchqueue}
        >
          Batch Queue
        </button>
        <div class="flex gap-2">
          <button
            class="flex-1 py-2 bg-surface-elevated hover:bg-edge rounded-lg text-xs transition-all duration-100 active:scale-[0.98]"
            onclick={() => handleCut('input_random')}
          >
            Make Parameter
          </button>
          <button
            class="flex-1 py-2 bg-surface-elevated hover:bg-edge rounded-lg text-xs transition-all duration-100 active:scale-[0.98]"
            onclick={() => handleCut('output')}
          >
            Make Output
          </button>
        </div>
      </div>

    {:else if nodeStatus.status === 'waiting'}
      <div class="flex items-center gap-2.5 text-amber-400">
        <div class="w-2.5 h-2.5 rounded-full bg-amber-400"></div>
        <span class="text-xs">Waiting in queue</span>
      </div>
      <button
        class="mt-3 w-full py-2 rounded-lg text-xs transition-all duration-100 text-red-400/70 hover:text-red-400 hover:bg-red-500/10 active:scale-[0.98]"
        onclick={handleDelete}
      >
        Delete
      </button>

    {:else if nodeStatus.status === 'executing'}
      <div class="flex items-center gap-2.5 text-blue-400">
        <div class="w-2.5 h-2.5 rounded-full bg-blue-400 pulse-ring status-glow"></div>
        <span class="text-xs">Executing</span>
      </div>
      {#if nodeStatus.stage}
        <div class="text-content-secondary/40 text-xs mt-1">{nodeStatus.stage}</div>
      {/if}
      <button
        class="mt-3 w-full py-2 rounded-lg text-xs transition-all duration-100 text-red-400/70 hover:text-red-400 hover:bg-red-500/10 active:scale-[0.98]"
        onclick={handleDelete}
      >
        Delete
      </button>

    {:else if nodeStatus.status === 'success'}
      <div class="flex items-center gap-2.5 text-green-400 mb-4">
        <div class="w-2.5 h-2.5 rounded-full bg-green-400"></div>
        <span class="text-xs font-medium">Success</span>
      </div>

      {#if outputCount <= 1}
        <!-- Single output -->
        {#if nodeStatus.metrics}
          <div class="space-y-2">
            <h4 class="text-[10px] font-medium text-content-secondary/40 uppercase tracking-wider">Accuracy</h4>
            <table class="w-full text-xs">
              <tbody>
                <tr>
                  <td class="py-1.5 text-content-secondary/50">MSE</td>
                  <td class="py-1.5 text-right font-mono tabular-nums">{formatValue(nodeStatus.metrics.mse)}</td>
                </tr>
                <tr>
                  <td class="py-1.5 text-content-secondary/50">Max Abs Diff</td>
                  <td class="py-1.5 text-right font-mono tabular-nums">{formatValue(nodeStatus.metrics.max_abs_diff)}</td>
                </tr>
                <tr>
                  <td class="py-1.5 text-content-secondary/50">Cosine Sim</td>
                  <td class="py-1.5 text-right font-mono tabular-nums">{formatValue(nodeStatus.metrics.cosine_similarity)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        {/if}

        <!-- Per-device results -->
        {#if nodeStatus.mainResult || nodeStatus.refResult}
          {@const main = nodeStatus.mainResult}
          {@const ref = nodeStatus.refResult}
          <div class="mt-4">
            <h4 class="text-[10px] font-medium text-content-secondary/40 uppercase tracking-wider mb-2">Device Outputs</h4>
            <table class="w-full text-xs table-fixed">
              <colgroup>
                <col class="w-[20%]" />
                <col class="w-[26.6%]" />
                <col class="w-[26.6%]" />
                {#if main && ref}<col class="w-[26.6%]" />{/if}
              </colgroup>
              <thead>
                <tr class="text-content-secondary/30">
                  <th class="py-1.5 text-left font-normal"></th>
                  {#if main}<th class="py-1.5 text-right font-normal">{main.device}</th>{/if}
                  {#if ref}<th class="py-1.5 text-right font-normal">{ref.device}</th>{/if}
                  {#if main && ref}<th class="py-1.5 text-right font-normal">Diff</th>{/if}
                </tr>
              </thead>
              <tbody>
                {#each [
                  { label: 'Min', key: 'min_val' as const },
                  { label: 'Max', key: 'max_val' as const },
                  { label: 'Mean', key: 'mean_val' as const },
                  { label: 'Std', key: 'std_val' as const },
                ] as row}
                  <tr class="border-t border-content-secondary/5">
                    <td class="py-1.5 text-content-secondary/50">{row.label}</td>
                    {#if main}<td class="py-1.5 text-right font-mono truncate tabular-nums">{fmt4(main[row.key])}</td>{/if}
                    {#if ref}<td class="py-1.5 text-right font-mono truncate tabular-nums">{fmt4(ref[row.key])}</td>{/if}
                    {#if main && ref}
                      <td class="py-1.5 text-right font-mono text-yellow-400/80 truncate tabular-nums">{fmt4(diff(main[row.key], ref[row.key]))}</td>
                    {/if}
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}

        <!-- Phase 2 actions -->
        <div class="mt-4 space-y-2">
          <button
            class="w-full py-2 bg-surface-elevated hover:bg-edge rounded-lg text-xs transition-all duration-100 active:scale-[0.98]"
            onclick={() => handleDeepAccuracy()}
          >
            Deep Accuracy View
          </button>
          <button
            class="w-full py-2 bg-surface-elevated hover:bg-edge rounded-lg text-xs transition-all duration-100 active:scale-[0.98]"
            onclick={onshowbatchqueue}
          >
            Batch Queue
          </button>
          <div class="flex gap-2">
            <button
              class="flex-1 py-2 bg-surface-elevated hover:bg-edge rounded-lg text-xs transition-all duration-100 active:scale-[0.98]"
              onclick={() => handleCut('input')}
            >
              Make Parameter
            </button>
            <button
              class="flex-1 py-2 bg-surface-elevated hover:bg-edge rounded-lg text-xs transition-all duration-100 active:scale-[0.98]"
              onclick={() => handleCut('output')}
            >
              Make Output
            </button>
          </div>
        </div>

      {:else}
        <!-- Multi-output: aggregate worst-case header -->
        {#if nodeStatus.metrics}
          <div class="space-y-2 mb-4">
            <h4 class="text-[10px] font-medium text-content-secondary/40 uppercase tracking-wider">Worst-Case Accuracy ({outputCount} outputs)</h4>
            <table class="w-full text-xs">
              <tbody>
                <tr>
                  <td class="py-1.5 text-content-secondary/50">MSE</td>
                  <td class="py-1.5 text-right font-mono tabular-nums">{formatValue(nodeStatus.metrics.mse)}</td>
                </tr>
                <tr>
                  <td class="py-1.5 text-content-secondary/50">Max Abs Diff</td>
                  <td class="py-1.5 text-right font-mono tabular-nums">{formatValue(nodeStatus.metrics.max_abs_diff)}</td>
                </tr>
                <tr>
                  <td class="py-1.5 text-content-secondary/50">Cosine Sim</td>
                  <td class="py-1.5 text-right font-mono tabular-nums">{formatValue(nodeStatus.metrics.cosine_similarity)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        {/if}

        <!-- Per-output breakdown stacked vertically -->
        {#each nodeStatus.perOutputMetrics! as outMetrics, outIdx}
          {@const outMain = nodeStatus.perOutputMainResults?.[outIdx]}
          {@const outRef = nodeStatus.perOutputRefResults?.[outIdx]}
          <div class="border-t border-content-secondary/8 pt-3 mt-3">
            <h4 class="text-xs font-medium text-content-primary/80 mb-1.5">
              Output {outIdx}
              {#if outMain?.dtype}
                <span class="text-content-secondary/30 font-normal ml-1">[{outMain.dtype}, {outMain.output_shapes?.[0]?.join('x') ?? '?'}]</span>
              {/if}
            </h4>

            <table class="w-full text-xs mb-1.5">
              <tbody>
                <tr>
                  <td class="py-1 text-content-secondary/50">MSE</td>
                  <td class="py-1 text-right font-mono tabular-nums">{formatValue(outMetrics.mse)}</td>
                </tr>
                <tr>
                  <td class="py-1 text-content-secondary/50">Max Abs Diff</td>
                  <td class="py-1 text-right font-mono tabular-nums">{formatValue(outMetrics.max_abs_diff)}</td>
                </tr>
                <tr>
                  <td class="py-1 text-content-secondary/50">Cosine Sim</td>
                  <td class="py-1 text-right font-mono tabular-nums">{formatValue(outMetrics.cosine_similarity)}</td>
                </tr>
              </tbody>
            </table>

            {#if outMain || outRef}
              <table class="w-full text-xs table-fixed mb-1.5">
                <colgroup>
                  <col class="w-[20%]" />
                  <col class="w-[26.6%]" />
                  <col class="w-[26.6%]" />
                  {#if outMain && outRef}<col class="w-[26.6%]" />{/if}
                </colgroup>
                <thead>
                  <tr class="text-content-secondary/30">
                    <th class="py-1 text-left font-normal"></th>
                    {#if outMain}<th class="py-1 text-right font-normal">{outMain.device}</th>{/if}
                    {#if outRef}<th class="py-1 text-right font-normal">{outRef.device}</th>{/if}
                    {#if outMain && outRef}<th class="py-1 text-right font-normal">Diff</th>{/if}
                  </tr>
                </thead>
                <tbody>
                  {#each [
                    { label: 'Min', key: 'min_val' as const },
                    { label: 'Max', key: 'max_val' as const },
                    { label: 'Mean', key: 'mean_val' as const },
                    { label: 'Std', key: 'std_val' as const },
                  ] as row}
                    <tr class="border-t border-content-secondary/5">
                      <td class="py-1 text-content-secondary/50">{row.label}</td>
                      {#if outMain}<td class="py-1 text-right font-mono truncate tabular-nums">{fmt4(outMain[row.key])}</td>{/if}
                      {#if outRef}<td class="py-1 text-right font-mono truncate tabular-nums">{fmt4(outRef[row.key])}</td>{/if}
                      {#if outMain && outRef}
                        <td class="py-1 text-right font-mono text-yellow-400/80 truncate tabular-nums">{fmt4(diff(outMain[row.key], outRef[row.key]))}</td>
                      {/if}
                    </tr>
                  {/each}
                </tbody>
              </table>
            {/if}

            <button
              class="w-full py-1.5 bg-surface-elevated hover:bg-edge rounded-lg text-xs transition-all duration-100 active:scale-[0.98]"
              onclick={() => handleDeepAccuracy(outIdx)}
            >
              Deep Accuracy View
            </button>
          </div>
        {/each}

        <!-- Shared actions below all outputs -->
        <div class="mt-4 space-y-2">
          <button
            class="w-full py-2 bg-surface-elevated hover:bg-edge rounded-lg text-xs transition-all duration-100 active:scale-[0.98]"
            onclick={onshowbatchqueue}
          >
            Batch Queue
          </button>
          <div class="flex gap-2">
            <button
              class="flex-1 py-2 bg-surface-elevated hover:bg-edge rounded-lg text-xs transition-all duration-100 active:scale-[0.98]"
              onclick={() => handleCut('input')}
            >
              Make Parameter
            </button>
            <button
              class="flex-1 py-2 bg-surface-elevated hover:bg-edge rounded-lg text-xs transition-all duration-100 active:scale-[0.98]"
              onclick={() => handleCut('output')}
            >
              Make Output
            </button>
          </div>
        </div>
      {/if}

      <!-- Export reproducer -->
      <button
        class="mt-2 w-full py-2 bg-surface-elevated hover:bg-edge rounded-lg text-xs transition-all duration-100 flex items-center justify-center gap-1.5 active:scale-[0.98]"
        onclick={handleExport}
        disabled={exporting}
      >
        {#if exporting}
          <svg class="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-dasharray="31.4 31.4" stroke-linecap="round"/>
          </svg>
          Exporting...
        {:else}
          <svg class="w-3 h-3" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 1a.5.5 0 0 1 .5.5v8.793l2.146-2.147a.5.5 0 0 1 .708.708l-3 3a.5.5 0 0 1-.708 0l-3-3a.5.5 0 1 1 .708-.708L7.5 10.293V1.5A.5.5 0 0 1 8 1z"/>
            <path d="M2 13.5a.5.5 0 0 1 .5-.5h11a.5.5 0 0 1 0 1h-11a.5.5 0 0 1-.5-.5z"/>
          </svg>
          Export Reproducer
        {/if}
      </button>
      <label class="mt-1 flex items-center gap-1.5 text-xs text-muted cursor-pointer select-none">
        <input type="checkbox" bind:checked={minimalModel} class="accent-accent w-3 h-3" />
        Minimal reproducer (cut model weights)
      </label>
      <button
        class="mt-2 w-full py-2 rounded-lg text-xs transition-all duration-100 text-red-400/70 hover:text-red-400 hover:bg-red-500/10 active:scale-[0.98]"
        onclick={handleDelete}
      >
        Delete
      </button>

    {:else if nodeStatus.status === 'failed'}
      <div class="flex items-center gap-2.5 text-red-400 mb-4">
        <div class="w-2.5 h-2.5 rounded-full bg-red-400"></div>
        <span class="text-xs font-medium">Failed</span>
      </div>
      {#if nodeStatus.stage}
        <div class="text-content-secondary/40 text-xs mb-2">Stage: {nodeStatus.stage}</div>
      {/if}
      {#if nodeStatus.errorDetail}
        <pre class="bg-surface-base rounded-lg p-3 text-xs text-red-300/80 overflow-x-auto max-h-48 whitespace-pre-wrap font-mono">{nodeStatus.errorDetail}</pre>
      {/if}
      <button
        class="mt-3 w-full py-2 rounded-lg text-xs transition-all duration-100 text-red-400/70 hover:text-red-400 hover:bg-red-500/10 active:scale-[0.98]"
        onclick={handleDelete}
      >
        Delete
      </button>
    {/if}

    <!-- Attributes -->
    {#if selectedNode.attributes && Object.keys(selectedNode.attributes).length > 0}
      <details class="mt-4">
        <summary class="text-xs text-content-secondary/40 cursor-pointer hover:text-content-secondary/60 transition-colors">
          Attributes ({Object.keys(selectedNode.attributes).length})
        </summary>
        <div class="mt-1.5 bg-surface-base rounded-lg p-2.5 text-xs font-mono max-h-32 overflow-y-auto">
          {#each Object.entries(selectedNode.attributes) as [key, value]}
            <div class="flex justify-between gap-2 py-0.5">
              <span class="text-content-secondary/40">{key}</span>
              <span class="text-content-secondary/70 truncate">{String(value)}</span>
            </div>
          {/each}
        </div>
      </details>
    {/if}

    <!-- Inputs -->
    {#if selectedNode.inputs && selectedNode.inputs.length > 0}
      <details class="mt-4" open>
        <summary class="text-xs text-content-secondary/40 cursor-pointer hover:text-content-secondary/60 transition-colors">
          Inputs ({selectedNode.inputs.length})
        </summary>
        <div class="mt-1.5 space-y-1">
          {#each selectedNode.inputs as inp, idx}
            {@const sourceNode = graphStore.graphData?.nodes.find(n => n.name === inp.name)}
            <div class="bg-surface-base rounded-lg p-2.5 text-xs">
              <div class="flex items-center gap-1.5">
                <span class="text-content-secondary/25 font-mono w-4 shrink-0">{idx}</span>
                {#if !inp.is_const}
                  <button
                    class="text-accent hover:text-accent-hover font-mono truncate transition-colors text-left"
                    onmouseenter={() => {
                      if (sourceNode) {
                        setHoveredNode(sourceNode.id);
                        setHoveredEdge(findInputEdgeIndex(sourceNode.id, idx));
                      }
                    }}
                    onmouseleave={() => { setHoveredNode(null); setHoveredEdge(null); }}
                    onclick={(e) => {
                      if (sourceNode) {
                        graphStore.selectNode(sourceNode.id);
                        refreshRenderer();
                        if (e.ctrlKey || e.metaKey) centerOnNode(sourceNode.id);
                      }
                    }}
                  >{inp.name}</button>
                {:else}
                  <span class="text-content-secondary/70 font-mono truncate" title={inp.name}>{inp.name}</span>
                {/if}
              </div>
              {#if sourceNode}
                <div class="text-content-secondary/40 ml-5 mt-0.5">{sourceNode.type}</div>
              {:else if inp.is_const}
                <div class="ml-5 mt-0.5"><span class="px-1.5 py-0.5 bg-amber-500/10 text-amber-400/70 rounded-full text-[10px] leading-none">const</span></div>
              {/if}
              {#if inp.shape}
                <div class="text-content-secondary/40 ml-5 mt-0.5">
                  [{#each inp.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-yellow-400">{dim}</span>{:else}{dim}{/if}{/each}] {#if inp.element_type}<span class="text-content-secondary/25">{inp.element_type}</span>{/if}
                </div>
              {/if}
              {#if inp.is_const && inp.const_node_name}
                <button
                  class="ml-5 mt-1 text-[10px] text-accent hover:text-accent-hover transition-colors"
                  onclick={() => toggleConstData(inp.const_node_name!)}
                >
                  {constDataExpanded.has(inp.const_node_name) ? 'Hide data' : 'View data'}
                </button>
                {#if constDataExpanded.has(inp.const_node_name)}
                  {#if constDataLoading.has(inp.const_node_name)}
                    <div class="ml-5 mt-1 text-content-secondary/30 text-[10px]">Loading...</div>
                  {:else if constDataCache.has(inp.const_node_name)}
                    {@const cd = constDataCache.get(inp.const_node_name)!}
                    <div class="ml-5 mt-1 space-y-1">
                      <div class="grid grid-cols-2 gap-x-3 text-[10px]">
                        <span class="text-content-secondary/30">dtype</span>
                        <span class="text-content-secondary/60 font-mono text-right">{cd.dtype}</span>
                        <span class="text-content-secondary/30">min</span>
                        <span class="text-content-secondary/60 font-mono text-right">{formatFloat(cd.stats.min)}</span>
                        <span class="text-content-secondary/30">max</span>
                        <span class="text-content-secondary/60 font-mono text-right">{formatFloat(cd.stats.max)}</span>
                        <span class="text-content-secondary/30">mean</span>
                        <span class="text-content-secondary/60 font-mono text-right">{formatFloat(cd.stats.mean)}</span>
                        <span class="text-content-secondary/30">std</span>
                        <span class="text-content-secondary/60 font-mono text-right">{formatFloat(cd.stats.std)}</span>
                      </div>
                      <details>
                        <summary class="text-[10px] text-content-secondary/30 cursor-pointer hover:text-content-secondary/50 transition-colors">
                          Values ({cd.total_elements}{cd.truncated ? ', truncated' : ''})
                        </summary>
                        <pre class="mt-1 bg-surface-base rounded-lg p-2 text-[9px] text-content-secondary/50 font-mono overflow-x-auto max-h-40 whitespace-pre-wrap leading-tight">{cd.data.map(v => formatFloat(v)).join(', ')}</pre>
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
        <summary class="text-xs text-content-secondary/40 cursor-pointer hover:text-content-secondary/60 transition-colors">
          Outputs ({outputs.length})
        </summary>
        <div class="mt-1.5 space-y-1">
          {#each outputs as out}
            <div class="bg-surface-base rounded-lg p-2.5 text-xs">
              <div class="flex items-center gap-1.5">
                <span class="text-content-secondary/25 font-mono w-4 shrink-0">{out.source_port}</span>
                <button
                  class="text-accent hover:text-accent-hover font-mono truncate transition-colors text-left"
                  onmouseenter={() => { setHoveredNode(out.targetId); setHoveredEdge(out.edgeIndex); }}
                  onmouseleave={() => { setHoveredNode(null); setHoveredEdge(null); }}
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
                <div class="text-content-secondary/40 ml-5 mt-0.5">{out.targetNode.type}</div>
                {#if out.targetNode.shape}
                  <div class="text-content-secondary/40 ml-5 mt-0.5">
                    [{#each out.targetNode.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-yellow-400">{dim}</span>{:else}{dim}{/if}{/each}] {#if out.targetNode.element_type}<span class="text-content-secondary/25">{out.targetNode.element_type}</span>{/if}
                  </div>
                {/if}
              {/if}
            </div>
          {/each}
        </div>
      </details>
    {/if}
  {:else}
    <div class="flex flex-col items-center justify-center py-16 px-4">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="text-content-secondary/15 mb-3">
        <rect x="3" y="3" width="18" height="18" rx="3" />
        <path d="M9 12h6M12 9v6" stroke-linecap="round" />
      </svg>
      <span class="text-content-secondary/30 text-xs">Select a node or edge to view details</span>
    </div>
  {/if}
</div>
