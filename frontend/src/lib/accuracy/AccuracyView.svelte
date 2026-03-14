<script lang="ts">
  import Heatmap from './Heatmap.svelte';
  import ChannelView from './ChannelView.svelte';
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

  let activeTab = $state<'heatmap' | 'channel'>('heatmap');
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
  <div class="flex border-b border-edge shrink-0">
    {#each [
      { key: 'heatmap', label: 'Heatmap' },
      { key: 'channel', label: 'Per-Channel' },
    ] as tab (tab.key)}
      <button
        class="px-4 py-2 text-sm transition-colors"
        class:text-accent={activeTab === tab.key}
        class:border-b-2={activeTab === tab.key}
        class:border-accent={activeTab === tab.key}
        class:text-content-secondary={activeTab !== tab.key}
        onclick={() => activeTab = tab.key as any}
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
        shape={tensorShape}
      />
    {:else if activeTab === 'channel'}
      <ChannelView
        main={mainTensor}
        ref={refTensor}
        shape={tensorShape}
      />
    {/if}
  </div>
</div>
