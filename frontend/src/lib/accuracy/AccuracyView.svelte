<script lang="ts">
  import Heatmap from './Heatmap.svelte';
  import ChannelView from './ChannelView.svelte';
  import SideBySide from './SideBySide.svelte';
  import ErrorTreemap from './ErrorTreemap.svelte';
  import Volume3D from './Volume3D.svelte';
  import Diagnostics from './Diagnostics.svelte';
  import RelativeError from './RelativeError.svelte';
  import LogError from './LogError.svelte';
  import SignedDiff from './SignedDiff.svelte';
  import ThresholdError from './ThresholdError.svelte';
  import FlickerCompare from './FlickerCompare.svelte';
  import NanInfInspector from './NanInfInspector.svelte';
  import SwipeCompare from './SwipeCompare.svelte';
  import FeatureGrid from './FeatureGrid.svelte';
  import ScatterPlot from './ScatterPlot.svelte';
  import CheckerboardCompare from './CheckerboardCompare.svelte';
  import SparsityMap from './SparsityMap.svelte';
  import MetricsDashboard from './MetricsDashboard.svelte';
  import SSIMMap from './SSIMMap.svelte';
  import SpatialCosine from './SpatialCosine.svelte';
  import QQPlot from './QQPlot.svelte';
  import ULPError from './ULPError.svelte';
  import ChannelCorrelation from './ChannelCorrelation.svelte';
  import GradientCompare from './GradientCompare.svelte';
  import FrequencyView from './FrequencyView.svelte';
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
    | 'heatmap' | 'sidebyside' | 'channel' | 'treemap' | 'volume3d' | 'diagnostics'
    | 'relativeError' | 'logError' | 'signedDiff' | 'thresholdError' | 'ssim' | 'spatialCosine' | 'ulpError'
    | 'flicker' | 'swipe' | 'checkerboard' | 'scatter'
    | 'featureGrid' | 'channelCorrelation' | 'sparsity'
    | 'qqPlot' | 'metricsDashboard'
    | 'nanInf' | 'fft' | 'gradient';

  let activeTab = $state<TabKey>('heatmap');
  let mainTensor = $state<Float32Array | null>(null);
  let refTensor = $state<Float32Array | null>(null);
  let tensorShape = $state<number[]>([]);
  let loadingTensors = $state(true);

  // Collapsed group state
  let expandedGroups = $state<Set<string>>(new Set(['current', 'error', 'compare', 'channels', 'stats', 'special']));

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

  // Spatial cosine needs >= 2 channels
  let dims = $derived(getSpatialDims(tensorShape));
  let hasSpatial = $derived(dims.height > 1 && dims.width > 1);

  interface TabGroup {
    id: string;
    label: string;
    tabs: { key: TabKey; label: string }[];
  }

  let tabGroups = $derived.by((): TabGroup[] => {
    const groups: TabGroup[] = [
      {
        id: 'current',
        label: 'Core',
        tabs: [
          { key: 'heatmap', label: 'Heatmap' },
          { key: 'sidebyside', label: 'Side-by-Side' },
          { key: 'channel', label: 'Per-Channel' },
          { key: 'treemap', label: 'Error Map' },
          { key: 'diagnostics', label: 'Diagnostics' },
          ...(canShow3D ? [{ key: 'volume3d' as TabKey, label: '3D Volume' }] : []),
        ],
      },
      {
        id: 'error',
        label: 'Error Analysis',
        tabs: [
          { key: 'relativeError', label: 'Relative Error' },
          { key: 'logError', label: 'Log Error' },
          { key: 'signedDiff', label: 'Signed Diff' },
          { key: 'thresholdError', label: 'Threshold' },
          ...(hasSpatial ? [{ key: 'ssim' as TabKey, label: 'SSIM' }] : []),
          ...(dims.channels > 1 ? [{ key: 'spatialCosine' as TabKey, label: 'Spatial Cosine' }] : []),
          { key: 'ulpError', label: 'ULP Error' },
        ],
      },
      {
        id: 'compare',
        label: 'Compare',
        tabs: [
          { key: 'flicker', label: 'Flicker' },
          { key: 'swipe', label: 'Swipe/Blend' },
          { key: 'checkerboard', label: 'Checkerboard' },
          { key: 'scatter', label: 'Scatter Plot' },
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
        id: 'stats',
        label: 'Statistics',
        tabs: [
          { key: 'qqPlot', label: 'Q-Q Plot' },
          { key: 'metricsDashboard', label: 'Metrics' },
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
    // Filter out empty groups
    return groups.filter(g => g.tabs.length > 0);
  });

  // If active tab becomes unavailable, reset to heatmap
  $effect(() => {
    const allKeys = tabGroups.flatMap(g => g.tabs.map(t => t.key));
    if (!allKeys.includes(activeTab)) {
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
              onclick={() => activeTab = tab.key}
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
      <div class="absolute inset-0 p-4 overflow-auto" style:visibility={activeTab === 'diagnostics' ? 'visible' : 'hidden'} style:pointer-events={activeTab === 'diagnostics' ? 'auto' : 'none'}>
        <Diagnostics
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

      <!-- Error Analysis tabs -->
      {#if activeTab === 'relativeError'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <RelativeError main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'logError'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <LogError main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'signedDiff'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <SignedDiff main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'thresholdError'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <ThresholdError main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'ssim'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <SSIMMap main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'spatialCosine'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <SpatialCosine main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'ulpError'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <ULPError main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}

      <!-- Compare tabs -->
      {#if activeTab === 'flicker'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <FlickerCompare main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'swipe'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <SwipeCompare main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'checkerboard'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <CheckerboardCompare main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'scatter'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <ScatterPlot main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}

      <!-- Channels tabs -->
      {#if activeTab === 'featureGrid'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <FeatureGrid main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'channelCorrelation'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <ChannelCorrelation main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'sparsity'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <SparsityMap main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}

      <!-- Statistics tabs -->
      {#if activeTab === 'qqPlot'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <QQPlot main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'metricsDashboard'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <MetricsDashboard main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}

      <!-- Special tabs -->
      {#if activeTab === 'nanInf'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <NanInfInspector main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'fft'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <FrequencyView main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
      {#if activeTab === 'gradient'}
        <div class="absolute inset-0 p-4 overflow-auto">
          <GradientCompare main={mainTensor} ref={refTensor} shape={tensorShape} mainLabel={mainDeviceName} refLabel={refDeviceName} />
        </div>
      {/if}
    {/if}
  </div>
</div>
