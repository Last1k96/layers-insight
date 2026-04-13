<script lang="ts">
  import ErrorMap from './ErrorMap.svelte';
  import SideBySide from './SideBySide.svelte';
  import ChannelStats from './ChannelStats.svelte';
  import Diagnostics from './Diagnostics.svelte';
  import Volume3D from './Volume3D.svelte';
  import CompareView from './CompareView.svelte';
  import SimilarityMap from './SimilarityMap.svelte';
  import ULPError from './ULPError.svelte';
  import DistributionView from './DistributionView.svelte';
  import FeatureGrid from './FeatureGrid.svelte';
  import ChannelCorrelation from './ChannelCorrelation.svelte';
  import SparsityMap from './SparsityMap.svelte';
  import NanInfInspector from './NanInfInspector.svelte';
  import FrequencyView from './FrequencyView.svelte';
  import GradientCompare from './GradientCompare.svelte';
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
    outputIndex = 0,
    onclose,
  }: {
    taskId: string;
    nodeId: string;
    outputIndex?: number;
    onclose: () => void;
  } = $props();

  type TabKey =
    | 'errorMap' | 'sidebyside' | 'channelStats' | 'diagnostics' | 'volume3d'
    | 'similarity' | 'ulpError'
    | 'compare'
    | 'featureGrid' | 'channelCorrelation' | 'sparsity'
    | 'distribution'
    | 'nanInf' | 'fft' | 'gradient';

  let activeTab = $state<TabKey>('errorMap');
  let visitedTabs = $state<Set<TabKey>>(new Set<TabKey>(['errorMap']));
  let mainTensor = $state<Float32Array | null>(null);
  let refTensor = $state<Float32Array | null>(null);
  let tensorShape = $state<number[]>([]);
  let loadingTensors = $state(true);

  // Collapsed group state
  let expandedGroups = $state<Set<string>>(new Set(['core', 'analysis', 'channels', 'special']));

  function toggleGroup(group: string) {
    const next = new Set(expandedGroups);
    if (next.has(group)) next.delete(group);
    else next.add(group);
    expandedGroups = next;
  }

  $effect(() => {
    loadTensors();
  });

  async function loadTensors() {
    const session = sessionStore.currentSession;
    if (!session) return;
    loadingTensors = true;

    const mainName = `main_output_${outputIndex}`;
    const refName = `ref_output_${outputIndex}`;

    const [main, ref] = await Promise.all([
      tensorStore.fetchTensor(session.id, taskId, mainName),
      tensorStore.fetchTensor(session.id, taskId, refName),
    ]);

    mainTensor = main;
    refTensor = ref;

    const meta = await tensorStore.fetchMeta(session.id, taskId, mainName);
    if (meta) tensorShape = meta.shape;

    loadingTensors = false;
  }

  // Volume 3D is available for 3D tensors [C,H,W] or 4D [B,C,H,W]
  let canShow3D = $derived.by(() => {
    if (tensorShape.length === 3) return true;
    if (tensorShape.length === 4) return true;
    return false;
  });

  let batchCount3D = $derived(tensorShape.length === 4 ? tensorShape[0] : 1);
  let selectedBatch3D = $state(0);

  $effect(() => {
    if (selectedBatch3D >= batchCount3D) selectedBatch3D = 0;
  });

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

  let dims = $derived(getSpatialDims(tensorShape));
  let hasSpatial = $derived(dims.height > 1 && dims.width > 1);
  let hasSimilarity = $derived(hasSpatial || dims.channels > 1);

  interface TabGroup {
    id: string;
    label: string;
    tabs: { key: TabKey; label: string }[];
  }

  let tabGroups = $derived.by((): TabGroup[] => {
    const groups: TabGroup[] = [
      {
        id: 'core',
        label: 'Core',
        tabs: [
          { key: 'errorMap', label: 'Error Map' },
          { key: 'sidebyside', label: 'Side-by-Side' },
          { key: 'channelStats', label: 'Channels' },
          { key: 'diagnostics', label: 'Density Grid' },
          ...(canShow3D ? [{ key: 'volume3d' as TabKey, label: '3D Volume' }] : []),
        ],
      },
      {
        id: 'analysis',
        label: 'Analysis',
        tabs: [
          { key: 'compare', label: 'Compare' },
          { key: 'distribution', label: 'Distribution' },
          ...(hasSimilarity ? [{ key: 'similarity' as TabKey, label: 'Similarity' }] : []),
          { key: 'ulpError', label: 'ULP Error' },
        ],
      },
      {
        id: 'channels',
        label: 'Channels',
        tabs: [
          ...(dims.channels > 1 ? [
            { key: 'featureGrid' as TabKey, label: 'Feature Grid' },
            { key: 'channelCorrelation' as TabKey, label: 'Correlation' },
          ] : []),
          { key: 'sparsity', label: 'Sparsity' },
        ],
      },
      {
        id: 'special',
        label: 'Special',
        tabs: [
          { key: 'nanInf', label: 'NaN/Inf' },
          ...(hasSpatial ? [{ key: 'fft' as TabKey, label: 'FFT' }] : []),
          ...(hasSpatial ? [{ key: 'gradient' as TabKey, label: 'Gradient' }] : []),
        ],
      },
    ];
    return groups.filter(g => g.tabs.length > 0);
  });

  // If active tab becomes unavailable, reset to errorMap
  $effect(() => {
    const allKeys = tabGroups.flatMap(g => g.tabs.map(t => t.key));
    if (!allKeys.includes(activeTab)) {
      activeTab = 'errorMap';
    }
  });

  let mainDeviceName = $derived(sessionStore.currentSession?.config.main_device ?? 'Main');
  let refDeviceName = $derived(sessionStore.currentSession?.config.ref_device ?? 'Reference');

  // Handle legacy tab values
  $effect(() => {
    const legacy = activeTab as string;
    if (legacy === 'histogram' || legacy === 'channel') activeTab = 'channelStats';
    if (legacy === 'heatmap' || legacy === 'relativeError' || legacy === 'logError' || legacy === 'signedDiff' || legacy === 'thresholdError') activeTab = 'errorMap';
    if (legacy === 'flicker' || legacy === 'swipe' || legacy === 'checkerboard') activeTab = 'compare';
    if (legacy === 'scatter' || legacy === 'qqPlot') activeTab = 'distribution';
    if (legacy === 'metricsDashboard') activeTab = 'channelStats';
    if (legacy === 'ssim' || legacy === 'spatialCosine') activeTab = 'similarity';
  });
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="fixed inset-0 bg-surface-panel backdrop-blur z-[60] flex flex-col">
  <!-- Header -->
  <div class="flex items-center justify-between px-4 py-3 border-b border-edge shrink-0">
    <div>
      <h3 class="font-medium text-content-primary">Deep Accuracy View{outputIndex > 0 ? ` (Output ${outputIndex})` : ''}</h3>
      <p class="text-xs text-content-secondary mt-0.5">Node: {nodeId}</p>
    </div>
    <button
      class="text-content-secondary hover:text-content-primary px-2 py-1 text-sm"
      onclick={onclose}
    >
      Close
    </button>
  </div>

  <!-- Grouped Tabs -->
  <div class="flex border-b border-edge shrink-0 overflow-x-auto px-2 gap-1 py-1">
    {#each tabGroups as group (group.id)}
      <div class="flex items-center">
        <button
          class="px-1.5 py-0.5 text-[10px] text-gray-500 hover:text-gray-300 font-medium uppercase tracking-wider"
          onclick={() => toggleGroup(group.id)}
        >
          {group.label}{expandedGroups.has(group.id) ? '' : '...'}
        </button>
        {#if expandedGroups.has(group.id)}
          {#each group.tabs as tab (tab.key)}
            <button
              class="px-2.5 py-1.5 text-xs transition-colors whitespace-nowrap rounded {activeTab === tab.key ? 'bg-accent text-white' : 'text-content-secondary hover:text-content-primary'}"
              onclick={() => { activeTab = tab.key; visitedTabs = new Set(visitedTabs).add(tab.key); }}
            >
              {tab.label}
            </button>
          {/each}
        {/if}
        {#if group !== tabGroups[tabGroups.length - 1]}
          <div class="w-px h-4 bg-edge/50 mx-1"></div>
        {/if}
      </div>
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
      <!-- Core tabs -->
      <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'errorMap' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'errorMap' ? 'auto' : 'none'}>
        <ErrorMap
          main={mainTensor}
          ref={refTensor}
          shape={tensorShape}
          mainLabel={mainDeviceName}
          refLabel={refDeviceName}
        />
      </div>
      {#if visitedTabs.has('sidebyside')}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'sidebyside' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'sidebyside' ? 'auto' : 'none'}>
          <SideBySide
            main={mainTensor}
            ref={refTensor}
            shape={tensorShape}
            mainLabel={mainDeviceName}
            refLabel={refDeviceName}
          />
        </div>
      {/if}
      {#if visitedTabs.has('channelStats')}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'channelStats' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'channelStats' ? 'auto' : 'none'}>
          <ChannelStats
            main={mainTensor}
            ref={refTensor}
            shape={tensorShape}
            mainLabel={mainDeviceName}
            refLabel={refDeviceName}
          />
        </div>
      {/if}
      {#if visitedTabs.has('diagnostics')}
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
      {#if canShow3D && volume3DMain && volume3DRef && visitedTabs.has('volume3d')}
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

      <!-- Analysis tabs -->
      {#if visitedTabs.has('compare')}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'compare' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'compare' ? 'auto' : 'none'}>
          <CompareView main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if visitedTabs.has('distribution')}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'distribution' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'distribution' ? 'auto' : 'none'}>
          <DistributionView main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if visitedTabs.has('similarity')}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'similarity' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'similarity' ? 'auto' : 'none'}>
          <SimilarityMap main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if visitedTabs.has('ulpError')}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'ulpError' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'ulpError' ? 'auto' : 'none'}>
          <ULPError main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}

      <!-- Channels tabs -->
      {#if visitedTabs.has('featureGrid')}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'featureGrid' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'featureGrid' ? 'auto' : 'none'}>
          <FeatureGrid main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if visitedTabs.has('channelCorrelation')}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'channelCorrelation' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'channelCorrelation' ? 'auto' : 'none'}>
          <ChannelCorrelation main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if visitedTabs.has('sparsity')}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'sparsity' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'sparsity' ? 'auto' : 'none'}>
          <SparsityMap main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}

      <!-- Special tabs -->
      {#if visitedTabs.has('nanInf')}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'nanInf' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'nanInf' ? 'auto' : 'none'}>
          <NanInfInspector main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if visitedTabs.has('fft')}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'fft' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'fft' ? 'auto' : 'none'}>
          <FrequencyView main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if visitedTabs.has('gradient')}
        <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'gradient' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'gradient' ? 'auto' : 'none'}>
          <GradientCompare main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
    {/if}
  </div>
</div>
