<script lang="ts">
  import { onMount } from 'svelte';
  import { configStore } from '../stores/config.svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { refreshRenderer } from './renderer';
  import { DEFAULT_RANGES, type AccuracyMetricKey } from '../utils/accuracyColors';
  import { toggleHelp } from '../shortcuts';

  let expanded = $state(false);

  // Sync persisted toggle state to graphStore on mount
  onMount(() => {
    if (configStore.accuracyEnabled) {
      graphStore.accuracyViewActive = true;
      // Defer refresh to after renderer is initialized
      requestAnimationFrame(() => refreshRenderer());
    }
  });

  const METRIC_LABELS: Record<AccuracyMetricKey, string> = {
    mse: 'MSE',
    max_abs_diff: 'Max Abs',
    cosine_similarity: 'Cosine',
  };

  function toggleAccuracy() {
    const next = !configStore.accuracyEnabled;
    configStore.setAccuracyEnabled(next);
    graphStore.accuracyViewActive = next;
    refreshRenderer();
  }

  function setMetric(m: AccuracyMetricKey) {
    configStore.setAccuracyMetric(m);
    refreshRenderer();
  }

  function handleRangeInput(field: 'min' | 'max', e: Event) {
    const input = e.target as HTMLInputElement;
    const val = parseFloat(input.value);
    if (isNaN(val)) return;
    const metric = configStore.accuracyMetric;
    const current = configStore.accuracyRanges[metric];
    configStore.setAccuracyRange(metric, { ...current, [field]: val });
    refreshRenderer();
  }

  function resetRange() {
    const metric = configStore.accuracyMetric;
    configStore.setAccuracyRange(metric, { ...DEFAULT_RANGES[metric] });
    refreshRenderer();
  }

  function handleRangeWheel(field: 'min' | 'max', e: WheelEvent) {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.01 : -0.01;
    const step = e.shiftKey ? 0.1 : 0.01;
    const actual = e.deltaY < 0 ? step : -step;
    const metric = configStore.accuracyMetric;
    const current = configStore.accuracyRanges[metric];
    const val = Math.round((current[field] + actual) * 1000) / 1000;
    configStore.setAccuracyRange(metric, { ...current, [field]: val });
    refreshRenderer();
  }
</script>

<div class="absolute top-3 right-3 z-30 flex items-start gap-2">
  <!-- Keyboard shortcuts hint -->
  <button
    onclick={() => toggleHelp()}
    class="px-1.5 py-1.5 rounded-md text-xs font-medium border transition-colors bg-[--bg-panel] border-[--border-color] text-gray-400 hover:text-gray-200 hover:bg-[--bg-menu]"
    title="Keyboard shortcuts (?)"
  >?</button>

  <!-- Toggle button -->
  <button
    onclick={toggleAccuracy}
    class="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors border"
    class:bg-emerald-600={configStore.accuracyEnabled}
    class:border-emerald-500={configStore.accuracyEnabled}
    class:text-white={configStore.accuracyEnabled}
    class:bg-[--bg-panel]={!configStore.accuracyEnabled}
    class:border-[--border-color]={!configStore.accuracyEnabled}
    class:text-gray-300={!configStore.accuracyEnabled}
    class:hover:bg-[--bg-menu]={!configStore.accuracyEnabled}
    title={configStore.accuracyEnabled ? 'Disable accuracy overlay' : 'Enable accuracy overlay (or hold Alt for quick preview)'}
  >
    <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
    Accuracy
  </button>

  <!-- Settings expand button -->
  <button
    onclick={() => expanded = !expanded}
    class="px-1.5 py-1.5 rounded-md text-xs border transition-colors"
    class:bg-[--bg-panel]={!expanded}
    class:border-[--border-color]={!expanded}
    class:text-gray-400={!expanded}
    class:bg-[--bg-menu]={expanded}
    class:border-gray-500={expanded}
    class:text-gray-200={expanded}
    title="Accuracy settings"
  >
    <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  </button>
</div>

<!-- Settings dropdown -->
{#if expanded}
  <div class="absolute top-12 right-3 z-30 w-56 bg-[--bg-panel] backdrop-blur border border-[--border-color] rounded-lg shadow-xl p-3 space-y-3">
    <!-- Metric selector -->
    <div>
      <span class="block text-[10px] uppercase tracking-wide text-gray-500 mb-1">Metric</span>
      <div class="flex gap-1">
        {#each Object.entries(METRIC_LABELS) as [key, label] (key)}
          <button
            onclick={() => setMetric(key as AccuracyMetricKey)}
            class="flex-1 px-2 py-1 rounded text-xs font-medium transition-colors border"
            class:bg-blue-600={configStore.accuracyMetric === key}
            class:border-blue-500={configStore.accuracyMetric === key}
            class:text-white={configStore.accuracyMetric === key}
            class:bg-transparent={configStore.accuracyMetric !== key}
            class:border-[--border-color]={configStore.accuracyMetric !== key}
            class:text-gray-400={configStore.accuracyMetric !== key}
            class:hover:text-gray-200={configStore.accuracyMetric !== key}
          >
            {label}
          </button>
        {/each}
      </div>
    </div>

    <!-- Range inputs -->
    <div>
      <div class="flex items-center justify-between mb-1">
        <span class="text-[10px] uppercase tracking-wide text-gray-500">Color range</span>
        <button
          onclick={resetRange}
          class="text-[10px] text-gray-500 hover:text-gray-300 transition-colors"
        >
          Reset
        </button>
      </div>
      <div class="flex gap-2 items-center">
        <div class="flex-1">
          <label for="acc-range-min" class="block text-[10px] text-gray-500 mb-0.5">Min</label>
          <input
            id="acc-range-min"
            type="number"
            step="0.01"
            value={configStore.activeRange.min}
            oninput={(e) => handleRangeInput('min', e)}
            onwheel={(e) => handleRangeWheel('min', e)}
            class="w-full bg-[--bg-primary] border border-[--border-color] rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-blue-500"
          />
        </div>
        <!-- Gradient preview bar -->
        <div class="w-6 h-4 rounded mt-3.5" style="background: linear-gradient(to right, #EF4444, #E5B010, #10B981);"></div>
        <div class="flex-1">
          <label for="acc-range-max" class="block text-[10px] text-gray-500 mb-0.5">Max</label>
          <input
            id="acc-range-max"
            type="number"
            step="0.01"
            value={configStore.activeRange.max}
            oninput={(e) => handleRangeInput('max', e)}
            onwheel={(e) => handleRangeWheel('max', e)}
            class="w-full bg-[--bg-primary] border border-[--border-color] rounded px-2 py-1 text-xs text-gray-200 focus:outline-none focus:border-blue-500"
          />
        </div>
      </div>
      <div class="flex justify-between text-[9px] text-gray-600 mt-0.5">
        <span>{configStore.accuracyMetric === 'cosine_similarity' ? 'Bad' : 'Good'}</span>
        <span>{configStore.accuracyMetric === 'cosine_similarity' ? 'Good' : 'Bad'}</span>
      </div>
    </div>
  </div>
{/if}
