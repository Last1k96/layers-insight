<script lang="ts">
  import { graphStore } from '../stores/graph.svelte';
  import { queueStore } from '../stores/queue.svelte';
  import { sessionStore } from '../stores/session.svelte';
  import { configStore } from '../stores/config.svelte';
  import { refreshRenderer, centerOnNode, setHoveredNode, setHoveredEdge } from '../graph/renderer';
  import { getTextOnNodeColor } from '../graph/opColors';
  import { getAccuracyColor, getAccuracyGoodness } from '../utils/accuracyColors';
  import type { AccuracyMetrics } from '../stores/types';
  import type { ConstantData } from '../stores/types';

  /** Svelte action: if the cursor is already over the element on mount, fire the callback. */
  function hoverOnMount(node: HTMLElement, onHover: () => void) {
    requestAnimationFrame(() => {
      if (node.matches(':hover')) onHover();
    });
  }

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

  async function handleExport() {
    const session = sessionStore.currentSession;
    if (!session || !nodeStatus?.taskId) return;

    exporting = true;
    try {
      const url = `/api/tensors/${session.id}/${nodeStatus.taskId}/export?minimal_model=true`;
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
    onbisect = () => {},
  }: {
    onshowaccuracy?: (outputIdx?: number) => void;
    onshowbatchqueue?: () => void;
    onbisect?: () => void;
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

  /** Get the hex color for the currently-selected accuracy metric. */
  function selectedMetricColor(metrics: AccuracyMetrics): string {
    const metric = configStore.accuracyMetric;
    return getAccuracyColor(metric, metrics[metric], configStore.activeRange);
  }

  /** Get 0-1 goodness score for the currently-selected accuracy metric. */
  function selectedMetricGoodness(metrics: AccuracyMetrics): number {
    const metric = configStore.accuracyMetric;
    return getAccuracyGoodness(metric, metrics[metric], configStore.activeRange);
  }
</script>

<div class="p-3 overflow-y-auto h-full text-sm">
  {#if selectedEdge}
    <!-- Edge Info -->
    <div class="mb-4">
      <h4 class="text-[10px] font-medium text-muted-soft uppercase tracking-wider mb-2">Source — port {selectedEdge.edge.source_port}</h4>

      <!-- Source Node Card -->
      {#if selectedEdge.sourceNode}
        {@const srcNode = selectedEdge.sourceNode}
        {@const srcPropagated = graphStore.graphData?.propagated_shapes?.[srcNode.name] ?? null}
        <div
          class="ns-card ns-card--interactive mb-2"
          role="button"
          tabindex={0}
          use:hoverOnMount={() => setHoveredNode(srcNode.id)}
          onmouseenter={() => setHoveredNode(srcNode.id)}
          onmouseleave={() => setHoveredNode(null)}
          onmouseup={() => { if (!window.getSelection()?.toString()) centerOnNode(srcNode.id); }}
        >
          <span class="font-mono font-medium text-[13px] text-content-primary break-all leading-snug">{srcNode.name}</span>
          <div
            class="block w-fit text-xs font-medium mt-1 px-2 py-0.5 rounded-md"
            style="background-color: {srcNode.color}; color: {getTextOnNodeColor(srcNode.color)};"
          >{srcNode.type}</div>
          {#if (srcNode.inputs && srcNode.inputs.length > 0) || srcNode.shape}
            <div class="mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 text-xs items-baseline">
              {#if srcNode.inputs}
                {#each srcNode.inputs as inp}
                  {@const inpPropagated = graphStore.graphData?.propagated_shapes?.[inp.name] ?? null}
                  {@const inpOrigShape = inp.shape}
                  <span class="text-muted-soft shrink-0">{inp.is_const ? 'const' : 'input'}</span>
                  <span>
                    {#if inpPropagated && inpOrigShape}
                      <span class="font-mono text-muted">[{#each inpPropagated as dim, idx}{#if idx > 0}, {/if}{#if inpOrigShape[idx] !== undefined && typeof inpOrigShape[idx] === 'string'}<span class="text-status-hint">{dim}</span>{:else}{dim}{/if}{/each}]</span>
                    {:else if inpOrigShape}
                      <span class="font-mono text-muted">[{#each inpOrigShape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-status-hint">?</span>{:else}{dim}{/if}{/each}]</span>
                    {/if}
                    {#if inp.element_type}<span class="text-muted-soft ml-1">{inp.element_type}</span>{/if}
                  </span>
                {/each}
              {/if}
              {#if srcNode.shape}
                <span class="text-muted-soft shrink-0">output</span>
                <span>
                  {#if srcPropagated}
                    <span class="font-mono text-muted">[{#each srcPropagated as dim, idx}{#if idx > 0}, {/if}{#if srcNode.shape[idx] !== undefined && typeof srcNode.shape[idx] === 'string'}<span class="text-status-hint">{dim}</span>{:else}{dim}{/if}{/each}]</span>
                  {:else}
                    <span class="font-mono text-muted">[{#each srcNode.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-status-hint">?</span>{:else}{dim}{/if}{/each}]</span>
                  {/if}
                  {#if srcNode.element_type}<span class="text-muted-soft ml-1">{srcNode.element_type}</span>{/if}
                </span>
              {/if}
            </div>
          {/if}
        </div>
      {:else}
        <div class="ns-card mb-2 text-xs">
          <span class="text-muted font-mono truncate">{selectedEdge.edge.source}</span>
        </div>
      {/if}

      <!-- Arrow -->
      <div class="flex justify-center py-1">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" class="text-muted-faint">
          <path d="M8 3v10M4 9l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>

      <h4 class="text-[10px] font-medium text-muted-soft uppercase tracking-wider mb-2">Target — port {selectedEdge.edge.target_port}</h4>

      <!-- Target Node Card -->
      {#if selectedEdge.targetNode}
        {@const tgtNode = selectedEdge.targetNode}
        {@const tgtPropagated = graphStore.graphData?.propagated_shapes?.[tgtNode.name] ?? null}
        <div
          class="ns-card ns-card--interactive mb-2"
          role="button"
          tabindex={0}
          use:hoverOnMount={() => setHoveredNode(tgtNode.id)}
          onmouseenter={() => setHoveredNode(tgtNode.id)}
          onmouseleave={() => setHoveredNode(null)}
          onmouseup={() => { if (!window.getSelection()?.toString()) centerOnNode(tgtNode.id); }}
        >
          <span class="font-mono font-medium text-[13px] text-content-primary break-all leading-snug">{tgtNode.name}</span>
          <div
            class="block w-fit text-xs font-medium mt-1 px-2 py-0.5 rounded-md"
            style="background-color: {tgtNode.color}; color: {getTextOnNodeColor(tgtNode.color)};"
          >{tgtNode.type}</div>
          {#if (tgtNode.inputs && tgtNode.inputs.length > 0) || tgtNode.shape}
            <div class="mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 text-xs items-baseline">
              {#if tgtNode.inputs}
                {#each tgtNode.inputs as inp}
                  {@const inpPropagated = graphStore.graphData?.propagated_shapes?.[inp.name] ?? null}
                  {@const inpOrigShape = inp.shape}
                  <span class="text-muted-soft shrink-0">{inp.is_const ? 'const' : 'input'}</span>
                  <span>
                    {#if inpPropagated && inpOrigShape}
                      <span class="font-mono text-muted">[{#each inpPropagated as dim, idx}{#if idx > 0}, {/if}{#if inpOrigShape[idx] !== undefined && typeof inpOrigShape[idx] === 'string'}<span class="text-status-hint">{dim}</span>{:else}{dim}{/if}{/each}]</span>
                    {:else if inpOrigShape}
                      <span class="font-mono text-muted">[{#each inpOrigShape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-status-hint">?</span>{:else}{dim}{/if}{/each}]</span>
                    {/if}
                    {#if inp.element_type}<span class="text-muted-soft ml-1">{inp.element_type}</span>{/if}
                  </span>
                {/each}
              {/if}
              {#if tgtNode.shape}
                <span class="text-muted-soft shrink-0">output</span>
                <span>
                  {#if tgtPropagated}
                    <span class="font-mono text-muted">[{#each tgtPropagated as dim, idx}{#if idx > 0}, {/if}{#if tgtNode.shape[idx] !== undefined && typeof tgtNode.shape[idx] === 'string'}<span class="text-status-hint">{dim}</span>{:else}{dim}{/if}{/each}]</span>
                  {:else}
                    <span class="font-mono text-muted">[{#each tgtNode.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-status-hint">?</span>{:else}{dim}{/if}{/each}]</span>
                  {/if}
                  {#if tgtNode.element_type}<span class="text-muted-soft ml-1">{tgtNode.element_type}</span>{/if}
                </span>
              {/if}
            </div>
          {/if}
        </div>
      {:else}
        <div class="mb-2 bg-surface-base rounded-lg p-2.5 text-xs">
          <span class="text-muted font-mono truncate">{selectedEdge.edge.target}</span>
        </div>
      {/if}
    </div>
  {:else if selectedNode}
    <!-- Node Info -->
    <div
      class="ns-card ns-card--interactive mb-4"
      role="button"
      tabindex={0}
      use:hoverOnMount={() => setHoveredNode(selectedNode.id)}
      onmouseenter={() => { setHoveredNode(selectedNode.id); }}
      onmouseleave={() => { setHoveredNode(null); }}
      onmouseup={() => {
        if (!window.getSelection()?.toString()) {
          centerOnNode(selectedNode.id);
        }
      }}
    >
      <span
        class="font-mono font-medium text-[13px] text-content-primary break-all leading-snug"
      >{selectedNode.name}</span>
      <div
        class="block w-fit text-xs font-medium mt-1 px-2 py-0.5 rounded-md"
        style="background-color: {selectedNode.color}; color: {getTextOnNodeColor(selectedNode.color)};"
      >{selectedNode.type}</div>
      {#if (selectedNode.inputs && selectedNode.inputs.length > 0) || selectedNode.shape}
        <div class="mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 text-xs items-baseline">
          {#if selectedNode.inputs}
            {#each selectedNode.inputs as inp}
              {@const inpPropagated = graphStore.graphData?.propagated_shapes?.[inp.name] ?? null}
              {@const inpOrigShape = inp.shape}
              <span class="text-muted-soft shrink-0">{inp.is_const ? 'const' : 'input'}</span>
              <span>
                {#if inpPropagated && inpOrigShape}
                  <span class="font-mono text-muted">[{#each inpPropagated as dim, idx}{#if idx > 0}, {/if}{#if inpOrigShape[idx] !== undefined && typeof inpOrigShape[idx] === 'string'}<span class="text-status-hint">{dim}</span>{:else}{dim}{/if}{/each}]</span>
                {:else if inpOrigShape}
                  <span class="font-mono text-muted">[{#each inpOrigShape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-status-hint">?</span>{:else}{dim}{/if}{/each}]</span>
                {/if}
                {#if inp.element_type}<span class="text-muted-soft ml-1">{inp.element_type}</span>{/if}
              </span>
            {/each}
          {/if}
          {#if selectedNode.shape}
            <span class="text-muted-soft shrink-0">output</span>
            <span>
              {#if propagatedShape}
                <span class="font-mono text-muted">[{#each propagatedShape as dim, idx}{#if idx > 0}, {/if}{#if selectedNode.shape[idx] !== undefined && typeof selectedNode.shape[idx] === 'string'}<span class="text-status-hint">{dim}</span>{:else}{dim}{/if}{/each}]</span>
              {:else}
                <span class="font-mono text-muted">[{#each selectedNode.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-status-hint">?</span>{:else}{dim}{/if}{/each}]</span>
              {/if}
              {#if selectedNode.element_type}<span class="text-muted-soft ml-1">{selectedNode.element_type}</span>{/if}
            </span>
          {/if}
        </div>
      {/if}
    </div>

    {#if !nodeStatus}
      <div class="space-y-2">
        <button
          class="ll-btn ll-btn--primary ll-btn--block"
          onclick={() => {
            const session = sessionStore.currentSession;
            if (session && selectedNode) {
              queueStore.enqueue(session.id, selectedNode.id, selectedNode.name, selectedNode.type, graphStore.activeSubSessionId);
            }
          }}
        >
          <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
            <path d="M4 2.5a.5.5 0 0 1 .77-.42l9 5.5a.5.5 0 0 1 0 .84l-9 5.5A.5.5 0 0 1 4 13.5v-11z"/>
          </svg>
          Infer
        </button>
        <button class="ll-btn ll-btn--block" onclick={onshowbatchqueue}>Batch Queue</button>
        <button class="ll-btn ll-btn--block" onclick={() => handleCut('input_random')}>
          Cut as Input<span class="ns-btn-hint">(random)</span>
        </button>
        <button class="ll-btn ll-btn--block" onclick={() => handleCut('output')}>Cut as Output</button>
      </div>

    {:else if nodeStatus.status === 'waiting'}
      <div class="ns-status ns-status--warn">
        <div class="ns-status__dot status-glow"></div>
        <span>Waiting in queue</span>
      </div>
      <button class="ll-btn ll-btn--block ll-btn--danger mt-3" onclick={handleDelete}>Delete</button>

    {:else if nodeStatus.status === 'executing'}
      <div class="ns-status ns-status--info">
        <div class="ns-status__dot pulse-ring status-glow"></div>
        <span>Executing</span>
        {#if nodeStatus.stage}
          <span class="ns-status__stage">{nodeStatus.stage}</span>
        {/if}
      </div>
      <button class="ll-btn ll-btn--block ll-btn--danger mt-3" onclick={handleDelete}>Delete</button>

    {:else if nodeStatus.status === 'success'}
      <div class="ns-status ns-status--ok mb-4">
        <svg class="ns-status__check" width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path d="M3.5 8.5l3 3 6-7" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <span>Success</span>
      </div>

      {#if outputCount <= 1}
        <!-- Single output -->
        {#if nodeStatus.metrics}
          {@const color = selectedMetricColor(nodeStatus.metrics)}
          {@const goodness = selectedMetricGoodness(nodeStatus.metrics)}
          {@const activeMetric = configStore.accuracyMetric}
          <div class="space-y-2">
            <h4 class="text-[10px] font-medium text-muted-soft uppercase tracking-wider">Accuracy</h4>
            <table class="w-full text-xs">
              <tbody>
                <tr>
                  <td class="py-1.5 text-muted">MSE</td>
                  <td class="py-1.5 text-right font-mono tabular-nums" style={activeMetric === 'mse' ? `color: ${color}` : ''}>{formatValue(nodeStatus.metrics.mse)}</td>
                </tr>
                <tr>
                  <td class="py-1.5 text-muted">Max Abs Diff</td>
                  <td class="py-1.5 text-right font-mono tabular-nums" style={activeMetric === 'max_abs_diff' ? `color: ${color}` : ''}>{formatValue(nodeStatus.metrics.max_abs_diff)}</td>
                </tr>
                <tr>
                  <td class="py-1.5 text-muted">Cosine Sim</td>
                  <td class="py-1.5 text-right font-mono tabular-nums" style={activeMetric === 'cosine_similarity' ? `color: ${color}` : ''}>{formatValue(nodeStatus.metrics.cosine_similarity)}</td>
                </tr>
              </tbody>
            </table>
            <!-- Accuracy bar -->
            <div class="pt-1">
              <div class="h-1.5 w-full bg-surface-base rounded-full overflow-hidden">
                <div
                  class="h-full rounded-full transition-all duration-500"
                  style="width: {Math.max(2, goodness * 100)}%; background-color: {color}; box-shadow: 0 0 8px {color}40;"
                ></div>
              </div>
            </div>
          </div>
        {/if}

        <!-- Per-device results -->
        {#if nodeStatus.mainResult || nodeStatus.refResult}
          {@const main = nodeStatus.mainResult}
          {@const ref = nodeStatus.refResult}
          <div class="mt-4">
            <h4 class="text-[10px] font-medium text-muted-soft uppercase tracking-wider mb-2">Device Outputs</h4>
            <table class="w-full text-xs table-fixed">
              <colgroup>
                <col class="w-[20%]" />
                <col class="w-[26.6%]" />
                <col class="w-[26.6%]" />
                {#if main && ref}<col class="w-[26.6%]" />{/if}
              </colgroup>
              <thead>
                <tr class="text-muted-soft">
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
                    <td class="py-1.5 text-muted">{row.label}</td>
                    {#if main}<td class="py-1.5 text-right font-mono truncate tabular-nums">{fmt4(main[row.key])}</td>{/if}
                    {#if ref}<td class="py-1.5 text-right font-mono truncate tabular-nums">{fmt4(ref[row.key])}</td>{/if}
                    {#if main && ref}
                      <td class="py-1.5 text-right font-mono text-status-hint truncate tabular-nums">{fmt4(diff(main[row.key], ref[row.key]))}</td>
                    {/if}
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}

        <!-- Phase 2 actions -->
        <div class="mt-4 space-y-2">
          <button class="ll-btn ll-btn--block" onclick={() => handleDeepAccuracy()}>Deep Accuracy View</button>
          <button class="ll-btn ll-btn--block" onclick={onbisect}>Bisect from Node</button>
          <button class="ll-btn ll-btn--block" onclick={onshowbatchqueue}>Batch Infer</button>
          <div class="flex gap-2">
            <button class="ll-btn ll-btn--block" onclick={() => handleCut('input')}>Cut as Input</button>
            <button class="ll-btn ll-btn--block" onclick={() => handleCut('output')}>Cut as Output</button>
          </div>
        </div>

      {:else}
        <!-- Multi-output: aggregate worst-case header -->
        {#if nodeStatus.metrics}
          {@const color = selectedMetricColor(nodeStatus.metrics)}
          {@const goodness = selectedMetricGoodness(nodeStatus.metrics)}
          {@const activeMetric = configStore.accuracyMetric}
          <div class="space-y-2 mb-4">
            <h4 class="text-[10px] font-medium text-muted-soft uppercase tracking-wider">Worst-Case Accuracy ({outputCount} outputs)</h4>
            <table class="w-full text-xs">
              <tbody>
                <tr>
                  <td class="py-1.5 text-muted">MSE</td>
                  <td class="py-1.5 text-right font-mono tabular-nums" style={activeMetric === 'mse' ? `color: ${color}` : ''}>{formatValue(nodeStatus.metrics.mse)}</td>
                </tr>
                <tr>
                  <td class="py-1.5 text-muted">Max Abs Diff</td>
                  <td class="py-1.5 text-right font-mono tabular-nums" style={activeMetric === 'max_abs_diff' ? `color: ${color}` : ''}>{formatValue(nodeStatus.metrics.max_abs_diff)}</td>
                </tr>
                <tr>
                  <td class="py-1.5 text-muted">Cosine Sim</td>
                  <td class="py-1.5 text-right font-mono tabular-nums" style={activeMetric === 'cosine_similarity' ? `color: ${color}` : ''}>{formatValue(nodeStatus.metrics.cosine_similarity)}</td>
                </tr>
              </tbody>
            </table>
            <div class="pt-1">
              <div class="h-1.5 w-full bg-surface-base rounded-full overflow-hidden">
                <div
                  class="h-full rounded-full transition-all duration-500"
                  style="width: {Math.max(2, goodness * 100)}%; background-color: {color}; box-shadow: 0 0 8px {color}40;"
                ></div>
              </div>
            </div>
          </div>
        {/if}

        <!-- Per-output breakdown stacked vertically -->
        {#each nodeStatus.perOutputMetrics! as outMetrics, outIdx}
          {@const outMain = nodeStatus.perOutputMainResults?.[outIdx]}
          {@const outRef = nodeStatus.perOutputRefResults?.[outIdx]}
          {@const outColor = selectedMetricColor(outMetrics)}
          {@const outActiveMetric = configStore.accuracyMetric}
          <div class="border-t border-content-secondary/8 pt-3 mt-3">
            <div class="flex items-center gap-2 mb-1.5">
              <div class="w-1.5 h-1.5 rounded-full" style="background-color: {outColor}"></div>
              <h4 class="text-xs font-medium text-content-primary/80">
                Output {outIdx}
                {#if outMain?.dtype}
                  <span class="text-muted-soft font-normal ml-1">[{outMain.dtype}, {outMain.output_shapes?.[0]?.join('x') ?? '?'}]</span>
                {/if}
              </h4>
            </div>

            <table class="w-full text-xs mb-1.5">
              <tbody>
                <tr>
                  <td class="py-1 text-muted">MSE</td>
                  <td class="py-1 text-right font-mono tabular-nums" style={outActiveMetric === 'mse' ? `color: ${outColor}` : ''}>{formatValue(outMetrics.mse)}</td>
                </tr>
                <tr>
                  <td class="py-1 text-muted">Max Abs Diff</td>
                  <td class="py-1 text-right font-mono tabular-nums" style={outActiveMetric === 'max_abs_diff' ? `color: ${outColor}` : ''}>{formatValue(outMetrics.max_abs_diff)}</td>
                </tr>
                <tr>
                  <td class="py-1 text-muted">Cosine Sim</td>
                  <td class="py-1 text-right font-mono tabular-nums" style={outActiveMetric === 'cosine_similarity' ? `color: ${outColor}` : ''}>{formatValue(outMetrics.cosine_similarity)}</td>
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
                  <tr class="text-muted-soft">
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
                      <td class="py-1 text-muted">{row.label}</td>
                      {#if outMain}<td class="py-1 text-right font-mono truncate tabular-nums">{fmt4(outMain[row.key])}</td>{/if}
                      {#if outRef}<td class="py-1 text-right font-mono truncate tabular-nums">{fmt4(outRef[row.key])}</td>{/if}
                      {#if outMain && outRef}
                        <td class="py-1 text-right font-mono text-status-hint truncate tabular-nums">{fmt4(diff(outMain[row.key], outRef[row.key]))}</td>
                      {/if}
                    </tr>
                  {/each}
                </tbody>
              </table>
            {/if}

            <button class="ll-btn ll-btn--block ll-btn--sm" onclick={() => handleDeepAccuracy(outIdx)}>Deep Accuracy View</button>
          </div>
        {/each}

        <!-- Shared actions below all outputs -->
        <div class="mt-4 space-y-2">
          <button class="ll-btn ll-btn--block" onclick={onbisect}>Bisect from Node</button>
          <button class="ll-btn ll-btn--block" onclick={onshowbatchqueue}>Batch Infer</button>
          <div class="flex gap-2">
            <button class="ll-btn ll-btn--block" onclick={() => handleCut('input')}>Cut as Input</button>
            <button class="ll-btn ll-btn--block" onclick={() => handleCut('output')}>Cut as Output</button>
          </div>
        </div>
      {/if}

      <!-- Export reproducer -->
      <button class="ll-btn ll-btn--block mt-2" onclick={handleExport} disabled={exporting}>
        {#if exporting}
          <svg class="ll-spin" width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" stroke-dasharray="31.4 31.4" stroke-linecap="round"/>
          </svg>
          Exporting…
        {:else}
          <svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
            <path d="M8 1a.5.5 0 0 1 .5.5v8.793l2.146-2.147a.5.5 0 0 1 .708.708l-3 3a.5.5 0 0 1-.708 0l-3-3a.5.5 0 1 1 .708-.708L7.5 10.293V1.5A.5.5 0 0 1 8 1z"/>
            <path d="M2 13.5a.5.5 0 0 1 .5-.5h11a.5.5 0 0 1 0 1h-11a.5.5 0 0 1-.5-.5z"/>
          </svg>
          Export Reproducer
        {/if}
      </button>
      <button class="ll-btn ll-btn--block ll-btn--danger mt-2" onclick={handleDelete}>Delete</button>

    {:else if nodeStatus.status === 'failed'}
      <div class="ns-status ns-status--err mb-4">
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path d="M5 5l6 6M11 5l-6 6" stroke-linecap="round"/>
        </svg>
        <span>Failed</span>
        {#if nodeStatus.stage}
          <span class="ns-status__stage">Stage: {nodeStatus.stage}</span>
        {/if}
      </div>
      {#if nodeStatus.errorDetail}
        <pre class="ns-error-pre">{nodeStatus.errorDetail}</pre>
      {/if}
      <button class="ll-btn ll-btn--block mt-3" onclick={onbisect}>Bisect from Node</button>
      <button class="ll-btn ll-btn--block ll-btn--danger mt-2" onclick={handleDelete}>Delete</button>
    {/if}

    <!-- Attributes -->
    {#if selectedNode.attributes && Object.keys(selectedNode.attributes).length > 0}
      <details class="mt-4">
        <summary class="text-xs text-muted-soft cursor-pointer hover:text-muted transition-colors">
          Attributes ({Object.keys(selectedNode.attributes).length})
        </summary>
        <div class="mt-1.5 bg-surface-base rounded-lg p-2.5 text-xs font-mono max-h-32 overflow-y-auto">
          {#each Object.entries(selectedNode.attributes) as [key, value]}
            <div class="flex justify-between gap-2 py-0.5">
              <span class="text-muted-soft">{key}</span>
              <span class="text-muted truncate">{String(value)}</span>
            </div>
          {/each}
        </div>
      </details>
    {/if}

    <!-- Inputs -->
    {#if selectedNode.inputs && selectedNode.inputs.length > 0}
      <details class="mt-4" open>
        <summary class="text-xs text-muted-soft cursor-pointer hover:text-muted transition-colors">
          Inputs ({selectedNode.inputs.length})
        </summary>
        <div class="mt-1.5 space-y-1">
          {#each selectedNode.inputs as inp, idx}
            {@const sourceNode = graphStore.graphData?.nodes.find(n => n.name === inp.name)}
            <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
            <div
              class="ns-card text-xs {sourceNode ? 'ns-card--interactive' : ''}"
              role={sourceNode ? 'button' : undefined}
              tabindex={sourceNode ? 0 : undefined}
              onmouseenter={() => {
                if (sourceNode) {
                  setHoveredNode(sourceNode.id);
                  setHoveredEdge(findInputEdgeIndex(sourceNode.id, idx));
                }
              }}
              onmouseleave={() => { setHoveredNode(null); setHoveredEdge(null); }}
              onmouseup={(e) => {
                if (sourceNode && !window.getSelection()?.toString()) {
                  graphStore.selectNode(sourceNode.id);
                  refreshRenderer();
                  if (e.ctrlKey || e.metaKey) centerOnNode(sourceNode.id);
                }
              }}
            >
              <div class="flex items-center gap-1.5">
                <span class="text-muted-soft font-mono w-4 shrink-0">{idx}</span>
                {#if !inp.is_const}
                  <span class="text-accent font-mono truncate text-left">{inp.name}</span>
                {:else}
                  <span class="text-muted font-mono truncate" title={inp.name}>{inp.name}</span>
                {/if}
              </div>
              {#if sourceNode}
                <div class="w-fit text-xs font-medium mt-0.5 ml-5 px-2 py-0.5 rounded-md" style="background-color: {sourceNode.color}; color: {getTextOnNodeColor(sourceNode.color)};">{sourceNode.type}</div>
              {:else if inp.is_const}
                <div class="ml-5 mt-0.5"><span class="ll-chip ll-chip--warn ll-chip--tiny ll-chip--pill">const</span></div>
              {/if}
              {#if inp.shape}
                <div class="text-muted-soft ml-5 mt-0.5">
                  [{#each inp.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-status-hint">{dim}</span>{:else}{dim}{/if}{/each}] {#if inp.element_type}<span class="text-muted-soft">{inp.element_type}</span>{/if}
                </div>
              {/if}
              {#if inp.is_const && inp.const_node_name}
                <button
                  class="ml-5 mt-1 text-[10px] text-accent hover:text-accent-hover transition-colors"
                  onclick={(e) => { e.stopPropagation(); toggleConstData(inp.const_node_name!); }}
                >
                  {constDataExpanded.has(inp.const_node_name) ? 'Hide data' : 'View data'}
                </button>
                {#if constDataExpanded.has(inp.const_node_name)}
                  {#if constDataLoading.has(inp.const_node_name)}
                    <div class="ml-5 mt-1 text-muted-soft text-[10px]">Loading...</div>
                  {:else if constDataCache.has(inp.const_node_name)}
                    {@const cd = constDataCache.get(inp.const_node_name)!}
                    <div class="ml-5 mt-1 space-y-1">
                      <div class="grid grid-cols-2 gap-x-3 text-[10px]">
                        <span class="text-muted-soft">dtype</span>
                        <span class="text-muted font-mono text-right">{cd.dtype}</span>
                        <span class="text-muted-soft">min</span>
                        <span class="text-muted font-mono text-right">{formatFloat(cd.stats.min)}</span>
                        <span class="text-muted-soft">max</span>
                        <span class="text-muted font-mono text-right">{formatFloat(cd.stats.max)}</span>
                        <span class="text-muted-soft">mean</span>
                        <span class="text-muted font-mono text-right">{formatFloat(cd.stats.mean)}</span>
                        <span class="text-muted-soft">std</span>
                        <span class="text-muted font-mono text-right">{formatFloat(cd.stats.std)}</span>
                      </div>
                      <details>
                        <summary class="text-[10px] text-muted-soft cursor-pointer hover:text-muted transition-colors">
                          Values ({cd.total_elements}{cd.truncated ? ', truncated' : ''})
                        </summary>
                        <pre class="mt-1 bg-surface-base rounded-lg p-2 text-[9px] text-muted font-mono overflow-x-auto max-h-40 whitespace-pre-wrap leading-tight">{cd.data.map(v => formatFloat(v)).join(', ')}</pre>
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
        <summary class="text-xs text-muted-soft cursor-pointer hover:text-muted transition-colors">
          Outputs ({outputs.length})
        </summary>
        <div class="mt-1.5 space-y-1">
          {#each outputs as out, idx}
            <div
              class="ns-card ns-card--interactive text-xs"
              role="button"
              tabindex={0}
              onmouseenter={() => { setHoveredNode(out.targetId); setHoveredEdge(out.edgeIndex); }}
              onmouseleave={() => { setHoveredNode(null); setHoveredEdge(null); }}
              onmouseup={(e) => {
                if (!window.getSelection()?.toString()) {
                  graphStore.selectNode(out.targetId);
                  refreshRenderer();
                  if (e.ctrlKey || e.metaKey) centerOnNode(out.targetId);
                }
              }}
            >
              <div class="flex items-center gap-1.5">
                <span class="text-muted-soft font-mono w-4 shrink-0">{idx}</span>
                <span class="text-accent font-mono truncate text-left">{out.targetNode?.name ?? out.targetId}</span>
              </div>
              {#if out.targetNode}
                <div class="w-fit text-xs font-medium mt-0.5 ml-5 px-2 py-0.5 rounded-md" style="background-color: {out.targetNode.color}; color: {getTextOnNodeColor(out.targetNode.color)};">{out.targetNode.type}</div>
                {#if out.targetNode.shape}
                  <div class="text-muted-soft ml-5 mt-0.5">
                    [{#each out.targetNode.shape as dim, idx}{#if idx > 0}, {/if}{#if typeof dim === 'string'}<span class="text-status-hint">{dim}</span>{:else}{dim}{/if}{/each}] {#if out.targetNode.element_type}<span class="text-muted-soft">{out.targetNode.element_type}</span>{/if}
                  </div>
                {/if}
              {/if}
            </div>
          {/each}
        </div>
      </details>
    {/if}
  {:else}
    <div class="ns-empty">
      <div class="ns-empty__icon">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" aria-hidden="true">
          <rect x="3" y="3" width="18" height="18" rx="3" />
          <path d="M9 12h6M12 9v6" stroke-linecap="round" />
        </svg>
        <div class="ns-empty__halo"></div>
      </div>
      <span class="ns-empty__label">Select a node or edge</span>
    </div>
  {/if}
</div>

<style>
  /* ── Interactive cards inside the right panel ──────────────────────── */
  .ns-card {
    padding: 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius-md);
    transition:
      background var(--dur-fast) ease,
      border-color var(--dur-fast) ease,
      box-shadow var(--dur-fast) ease;
  }
  .ns-card--interactive { cursor: pointer; }
  .ns-card--interactive:hover {
    background: var(--bg-input);
    border-color: var(--accent-border);
    box-shadow: 0 0 0 1px var(--accent-bg-strong) inset;
  }

  /* ── Status banner (waiting / executing / success / failed) ───────── */
  .ns-status {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: var(--radius-md);
    font-size: 12px;
    font-weight: 500;
    border: 1px solid transparent;
  }
  .ns-status__dot {
    width: 8px;
    height: 8px;
    border-radius: var(--radius-pill);
    background: currentColor;
    flex-shrink: 0;
  }
  .ns-status__check { flex-shrink: 0; }
  .ns-status__stage {
    margin-left: auto;
    font-size: 10.5px;
    font-weight: 400;
    opacity: 0.7;
    font-family: var(--font-mono);
  }
  .ns-status--info { color: var(--status-info); background: var(--status-info-bg); }
  .ns-status--warn { color: var(--status-warn); background: var(--status-warn-bg); border-color: var(--status-warn-border); }
  .ns-status--ok   { color: var(--status-ok);   background: var(--status-ok-bg);   border-color: var(--status-ok-border); }
  .ns-status--err  { color: var(--status-err);  background: var(--status-err-bg);  border-color: var(--status-err-border); }

  /* Secondary hint inline inside a .ll-btn (e.g. "Cut as Input (random)") */
  .ns-btn-hint {
    margin-left: 5px;
    font-size: 10px;
    font-weight: 400;
    color: var(--text-muted-soft);
  }

  /* ── Failure error block ──────────────────────────────────────────── */
  .ns-error-pre {
    margin-top: 0;
    padding: 10px 12px;
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 1.45;
    color: var(--status-err);
    background: rgba(248, 113, 113, 0.05);
    border: 1px solid var(--status-err-border);
    border-radius: var(--radius-md);
    max-height: 200px;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }

  /* ── Empty state ──────────────────────────────────────────────────── */
  .ns-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 56px 16px;
    gap: 14px;
  }
  .ns-empty__icon {
    position: relative;
    width: 40px;
    height: 40px;
    color: var(--text-muted-faint);
  }
  .ns-empty__halo {
    position: absolute;
    inset: 0;
    border-radius: var(--radius-pill);
    background: radial-gradient(circle, var(--accent-bg-strong) 0%, transparent 70%);
    animation: node-breathe 3s ease-in-out infinite;
  }
  .ns-empty__label {
    font-size: 11.5px;
    color: var(--text-muted-strong);
  }
</style>
