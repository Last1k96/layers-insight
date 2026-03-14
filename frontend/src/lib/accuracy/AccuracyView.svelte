<script lang="ts">
  import Heatmap from './Heatmap.svelte';
  import ChannelView from './ChannelView.svelte';
  import SideBySide from './SideBySide.svelte';
  import ErrorTreemap from './ErrorTreemap.svelte';
  import Volume3D from './Volume3D.svelte';
  import Diagnostics from './Diagnostics.svelte';
  import { getSpatialDims } from './tensorUtils';
  import { tensorStore } from '../stores/tensors.svelte';
  import { sessionStore } from '../stores/session.svelte';

  let {
    taskId,
    nodeId,
    onclose,
  }: {
    taskId: string;
    nodeId: string;
    onclose: () => void;
  } = $props();

  type TabKey = 'heatmap' | 'sidebyside' | 'channel' | 'treemap' | 'volume3d' | 'diagnostics';

  let activeTab = $state<TabKey>('heatmap');
  let mainTensor = $state<Float32Array | null>(null);
  let refTensor = $state<Float32Array | null>(null);
  let tensorShape = $state<number[]>([]);
  let loadingTensors = $state(true);

  $effect(() => {
    loadTensors();
  });

  async function loadTensors() {
    const session = sessionStore.currentSession;
    if (!session) return;
    loadingTensors = true;

    const [main, ref] = await Promise.all([
      tensorStore.fetchTensor(session.id, taskId, 'main_output'),
      tensorStore.fetchTensor(session.id, taskId, 'ref_output'),
    ]);

    mainTensor = main;
    refTensor = ref;

    // Get shape from meta
    const meta = await tensorStore.fetchMeta(session.id, taskId, 'main_output');
    if (meta) tensorShape = meta.shape;

    loadingTensors = false;
  }

  let diffTensor = $derived.by(() => {
    if (!mainTensor || !refTensor || mainTensor.length !== refTensor.length) return null;
    const diff = new Float32Array(mainTensor.length);
    for (let i = 0; i < mainTensor.length; i++) {
      diff[i] = Math.abs(mainTensor[i] - refTensor[i]);
    }
    return diff;
  });

  // Volume 3D is only available for 3D tensors [C,H,W] or 4D with batch=1 [1,C,H,W]
  let canShow3D = $derived.by(() => {
    if (tensorShape.length === 3) return true;
    if (tensorShape.length === 4 && tensorShape[0] === 1) return true;
    return false;
  });

  let tabs = $derived.by(() => {
    const base: { key: TabKey; label: string }[] = [
      { key: 'heatmap', label: 'Heatmap' },
      { key: 'sidebyside', label: 'Side-by-Side' },
      { key: 'channel', label: 'Per-Channel' },
      { key: 'treemap', label: 'Error Map' },
      { key: 'diagnostics', label: 'Diagnostics' },
    ];
    if (canShow3D) {
      base.push({ key: 'volume3d', label: '3D Volume' });
    }
    return base;
  });

  // If active tab becomes unavailable, reset to heatmap
  $effect(() => {
    if (activeTab === 'volume3d' && !canShow3D) {
      activeTab = 'heatmap';
    }
  });

  let mainDeviceName = $derived(sessionStore.currentSession?.main_device ?? 'Main');
  let refDeviceName = $derived(sessionStore.currentSession?.ref_device ?? 'Reference');

  // Handle legacy tab values
  $effect(() => {
    if ((activeTab as string) === 'histogram') {
      activeTab = 'channel';
    }
  });
</script>

<div class="fixed inset-0 bg-surface-panel backdrop-blur z-[60] flex flex-col">
  <!-- Header -->
  <div class="flex items-center justify-between px-4 py-3 border-b border-edge shrink-0">
    <div>
      <h3 class="font-medium text-content-primary">Deep Accuracy View</h3>
      <p class="text-xs text-content-secondary mt-0.5">Node: {nodeId}</p>
    </div>
    <button
      class="text-content-secondary hover:text-content-primary px-2 py-1 text-sm"
      onclick={onclose}
    >
      Close
    </button>
  </div>

  <!-- Tabs -->
  <div class="flex border-b border-edge shrink-0 overflow-x-auto">
    {#each tabs as tab (tab.key)}
      <button
        class="px-4 py-2 text-sm transition-colors whitespace-nowrap"
        class:text-accent={activeTab === tab.key}
        class:border-b-2={activeTab === tab.key}
        class:border-accent={activeTab === tab.key}
        class:text-content-secondary={activeTab !== tab.key}
        onclick={() => activeTab = tab.key}
      >
        {tab.label}
      </button>
    {/each}
  </div>

  <!-- Content -->
  <div class="flex-1 overflow-auto p-4 min-h-0">
    {#if loadingTensors}
      <div class="flex items-center justify-center h-full text-content-secondary">
        Loading tensor data...
      </div>
    {:else if !mainTensor || !refTensor}
      <div class="flex items-center justify-center h-full text-content-secondary">
        Tensor data not available
      </div>
    {:else if activeTab === 'heatmap'}
      <Heatmap
        diff={diffTensor}
        main={mainTensor}
        ref={refTensor}
        shape={tensorShape}
      />
    {:else if activeTab === 'sidebyside'}
      <SideBySide
        main={mainTensor}
        ref={refTensor}
        shape={tensorShape}
      />
    {:else if activeTab === 'channel'}
      <ChannelView
        main={mainTensor}
        ref={refTensor}
        shape={tensorShape}
        mainLabel={mainDeviceName}
        refLabel={refDeviceName}
      />
    {:else if activeTab === 'treemap'}
      <ErrorTreemap
        main={mainTensor}
        ref={refTensor}
        shape={tensorShape}
      />
    {:else if activeTab === 'volume3d'}
      <Volume3D
        main={mainTensor}
        ref={refTensor}
        shape={tensorShape}
      />
    {:else if activeTab === 'diagnostics'}
      <Diagnostics
        main={mainTensor}
        ref={refTensor}
        shape={tensorShape}
      />
    {/if}
  </div>
</div>
