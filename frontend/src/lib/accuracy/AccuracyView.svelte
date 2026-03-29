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

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      e.preventDefault();
      onclose();
    }
  }

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

  // Volume 3D is available for 3D tensors [C,H,W] or 4D [B,C,H,W]
  let canShow3D = $derived.by(() => {
    if (tensorShape.length === 3) return true;
    if (tensorShape.length === 4) return true;
    return false;
  });

  let batchCount3D = $derived(tensorShape.length === 4 ? tensorShape[0] : 1);
  let selectedBatch3D = $state(0);

  // Reset batch selection when shape changes
  $effect(() => {
    if (selectedBatch3D >= batchCount3D) selectedBatch3D = 0;
  });

  // Slice tensors for the selected batch (pass [C,H,W] to Volume3D)
  let volume3DMain = $derived.by(() => {
    if (!mainTensor || batchCount3D <= 1) return mainTensor;
    const chw = tensorShape[1] * tensorShape[2] * tensorShape[3];
    const offset = selectedBatch3D * chw;
    return mainTensor.subarray(offset, offset + chw);
  });
  let volume3DRef = $derived.by(() => {
    if (!refTensor || batchCount3D <= 1) return refTensor;
    const chw = tensorShape[1] * tensorShape[2] * tensorShape[3];
    const offset = selectedBatch3D * chw;
    return refTensor.subarray(offset, offset + chw);
  });
  let volume3DShape = $derived(
    tensorShape.length === 4 ? tensorShape.slice(1) : tensorShape
  );

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

  let mainDeviceName = $derived(sessionStore.currentSession?.config.main_device ?? 'Main');
  let refDeviceName = $derived(sessionStore.currentSession?.config.ref_device ?? 'Reference');

  // Handle legacy tab values
  $effect(() => {
    if ((activeTab as string) === 'histogram') {
      activeTab = 'channel';
    }
  });
</script>

<svelte:window onkeydown={handleKeydown} />

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
  <div class="flex-1 p-4 min-h-0 relative">
    {#if loadingTensors}
      <div class="flex items-center justify-center h-full text-content-secondary">
        Loading tensor data...
      </div>
    {:else if !mainTensor || !refTensor}
      <div class="flex items-center justify-center h-full text-content-secondary">
        Tensor data not available
      </div>
    {:else}
      <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'heatmap' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'heatmap' ? 'auto' : 'none'}>
        <Heatmap
          diff={diffTensor}
          main={mainTensor}
          ref={refTensor}
          shape={tensorShape}
          mainLabel={mainDeviceName}
          refLabel={refDeviceName}
        />
      </div>
      <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'sidebyside' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'sidebyside' ? 'auto' : 'none'}>
        <SideBySide
          main={mainTensor}
          ref={refTensor}
          shape={tensorShape}
          mainLabel={mainDeviceName}
          refLabel={refDeviceName}
        />
      </div>
      <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'channel' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'channel' ? 'auto' : 'none'}>
        <ChannelView
          main={mainTensor}
          ref={refTensor}
          shape={tensorShape}
          mainLabel={mainDeviceName}
          refLabel={refDeviceName}
        />
      </div>
      <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'treemap' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'treemap' ? 'auto' : 'none'}>
        <ErrorTreemap
          main={mainTensor}
          ref={refTensor}
          shape={tensorShape}
          mainLabel={mainDeviceName}
          refLabel={refDeviceName}
        />
      </div>
      {#if canShow3D && volume3DMain && volume3DRef}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'volume3d' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'volume3d' ? 'auto' : 'none'}>
          {#if batchCount3D > 1}
            <div class="flex items-center gap-2 mb-2 text-sm text-content-secondary">
              <span>Batch:</span>
              <select
                class="bg-surface-panel border border-edge rounded px-2 py-1 text-content-primary text-sm"
                value={selectedBatch3D}
                onchange={(e) => selectedBatch3D = Number((e.target as HTMLSelectElement).value)}
              >
                {#each Array(batchCount3D) as _, i}
                  <option value={i}>{i}</option>
                {/each}
              </select>
              <span class="text-xs text-content-secondary">of {batchCount3D}</span>
            </div>
          {/if}
          <Volume3D
            main={volume3DMain}
            ref={volume3DRef}
            shape={volume3DShape}
            mainLabel={mainDeviceName}
            refLabel={refDeviceName}
          />
        </div>
      {/if}
      <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'diagnostics' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'diagnostics' ? 'auto' : 'none'}>
        <Diagnostics
          main={mainTensor}
          ref={refTensor}
          shape={tensorShape}
          mainLabel={mainDeviceName}
          refLabel={refDeviceName}
        />
      </div>
    {/if}
  </div>
</div>
