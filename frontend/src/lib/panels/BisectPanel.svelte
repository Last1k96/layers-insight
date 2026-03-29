<script lang="ts">
  import { bisectStore } from '../stores/bisect.svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { sessionStore } from '../stores/session.svelte';
  import { centerOnNode } from '../graph/renderer';

  let {
    onclose,
  }: {
    onclose: () => void;
  } = $props();

  let starting = $state(false);

  const metricLabels: Record<string, string> = {
    cosine_similarity: 'Cosine similarity',
    mse: 'MSE',
    max_abs_diff: 'Max abs diff',
  };

  const searchForLabels: Record<string, string> = {
    accuracy_drop: 'Accuracy drop',
    compilation_failure: 'Compilation failure',
  };

  function defaultThreshold(metric: string, searchFor: string): number {
    if (searchFor === 'compilation_failure') return 0;
    if (metric === 'cosine_similarity') return 0.999;
    if (metric === 'mse') return 0.001;
    return 0.01;
  }

  function onSearchForChange(e: Event) {
    const val = (e.target as HTMLSelectElement).value;
    bisectStore.searchFor = val as any;
    bisectStore.threshold = defaultThreshold(bisectStore.metric, val);
  }

  function onMetricChange(e: Event) {
    const val = (e.target as HTMLSelectElement).value;
    bisectStore.metric = val as any;
    bisectStore.threshold = defaultThreshold(val, bisectStore.searchFor);
  }

  async function startBisect() {
    const session = sessionStore.currentSession;
    if (!session) return;
    starting = true;
    bisectStore.error = null;
    const ok = await bisectStore.start(session.id, graphStore.activeSubSessionId);
    starting = false;
  }

  async function stopBisect() {
    await bisectStore.stop();
  }

  function goToFoundNode() {
    if (!bisectStore.foundNode) return;
    const node = graphStore.graphData?.nodes.find(n => n.name === bisectStore.foundNode);
    if (node) {
      graphStore.selectNode(node.id);
      centerOnNode(node.id);
    }
  }

  function resetAndClose() {
    bisectStore.reset();
    onclose();
  }

  function formatMetricValue(val: number | undefined, metric: string): string {
    if (val === undefined || val === null) return '-';
    return val.toFixed(6);
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      onclose();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="fixed inset-0 z-[59] bg-black/30" onclick={onclose} role="presentation"></div>
<div class="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-[60] w-[420px]">
  <div class="bg-[--bg-panel] backdrop-blur border border-[--border-color] rounded-xl shadow-2xl">
    <!-- Header -->
    <div class="flex items-center justify-between px-4 py-3 border-b border-[--border-color]">
      <h3 class="text-sm font-medium text-gray-200">Bisect</h3>
      <button class="text-gray-400 hover:text-gray-200 text-xs" onclick={onclose}>Close</button>
    </div>

    <!-- Config (only when not running) -->
    {#if !bisectStore.isRunning}
      <div class="p-4 space-y-3 border-b border-[--border-color]">
        <label class="block text-xs">
          <span class="text-gray-400">Search for:</span>
          <select
            value={bisectStore.searchFor}
            onchange={onSearchForChange}
            class="w-full mt-1 px-2 py-1.5 bg-[--bg-panel] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none"
          >
            {#each Object.entries(searchForLabels) as [value, label]}
              <option {value}>{label}</option>
            {/each}
          </select>
        </label>

        {#if bisectStore.searchFor === 'accuracy_drop'}
          <label class="block text-xs">
            <span class="text-gray-400">Metric:</span>
            <select
              value={bisectStore.metric}
              onchange={onMetricChange}
              class="w-full mt-1 px-2 py-1.5 bg-[--bg-panel] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none"
            >
              {#each Object.entries(metricLabels) as [value, label]}
                <option {value}>{label}</option>
              {/each}
            </select>
          </label>

          <label class="block text-xs">
            <span class="text-gray-400">Threshold:</span>
            <input
              type="number"
              step="any"
              bind:value={bisectStore.threshold}
              class="w-full mt-1 px-2 py-1.5 bg-[--bg-panel] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none font-mono"
            />
          </label>
        {/if}

        <div class="text-xs text-gray-500">
          Range: full model (auto)
        </div>

        <button
          class="w-full py-2 bg-accent hover:bg-accent-hover disabled:bg-[--bg-panel] disabled:text-content-secondary rounded text-sm font-medium transition-colors"
          disabled={starting}
          onclick={startBisect}
        >
          {starting ? 'Starting...' : 'Start Bisection'}
        </button>

        {#if bisectStore.error}
          <div class="text-xs text-red-400 mt-1">{bisectStore.error}</div>
        {/if}
      </div>
    {/if}

    <!-- Progress -->
    {#if bisectStore.isRunning || bisectStore.isDone || bisectStore.status === 'stopped' || bisectStore.status === 'error'}
      <div class="p-4 space-y-2 border-b border-[--border-color]">
        <!-- Status badge -->
        <div class="flex items-center gap-2">
          <span class="text-xs font-medium" class:text-blue-400={bisectStore.isRunning} class:text-green-400={bisectStore.isDone} class:text-yellow-400={bisectStore.status === 'stopped'} class:text-red-400={bisectStore.status === 'error'}>
            {bisectStore.status === 'running' ? 'Running' : bisectStore.status === 'done' ? 'Complete' : bisectStore.status === 'stopped' ? 'Stopped' : 'Error'}
          </span>
          {#if bisectStore.totalSteps > 0}
            <span class="text-xs text-gray-500">Step {bisectStore.step}/{bisectStore.totalSteps}</span>
          {/if}
        </div>

        <!-- Progress bar -->
        {#if bisectStore.totalSteps > 0}
          <div class="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
            <div
              class="h-full rounded-full transition-all duration-300"
              class:bg-blue-500={bisectStore.isRunning}
              class:bg-green-500={bisectStore.isDone}
              class:bg-yellow-500={bisectStore.status === 'stopped'}
              class:bg-red-500={bisectStore.status === 'error'}
              style:width="{Math.min(100, (bisectStore.step / bisectStore.totalSteps) * 100)}%"
            ></div>
          </div>
        {/if}

        <!-- Range info -->
        {#if bisectStore.rangeStart && bisectStore.rangeEnd}
          <div class="text-xs text-gray-400">
            Range: <span class="font-mono text-gray-300">{bisectStore.rangeStart}</span>
            <span class="mx-1">--</span>
            <span class="font-mono text-gray-300">{bisectStore.rangeEnd}</span>
          </div>
        {/if}

        <!-- Current node -->
        {#if bisectStore.currentNode && bisectStore.isRunning}
          <div class="text-xs text-gray-400">
            Testing: <span class="font-mono text-gray-200">{bisectStore.currentNode}</span>
          </div>
        {/if}

        <!-- Result -->
        {#if bisectStore.foundNode}
          <div class="mt-2 p-2 bg-green-900/20 border border-green-800/30 rounded">
            <div class="text-xs text-green-400 font-medium">Found:</div>
            <button
              class="text-sm font-mono text-green-300 hover:text-green-100 underline cursor-pointer mt-0.5"
              onclick={goToFoundNode}
            >
              {bisectStore.foundNode}
            </button>
          </div>
        {/if}

        <!-- Error message -->
        {#if bisectStore.error}
          <div class="text-xs text-red-400">{bisectStore.error}</div>
        {/if}

        <!-- Actions -->
        <div class="flex gap-2 mt-2">
          {#if bisectStore.isRunning}
            <button
              class="flex-1 py-1.5 bg-red-600/80 hover:bg-red-600 rounded text-xs font-medium transition-colors"
              onclick={stopBisect}
            >
              Stop
            </button>
          {:else}
            <button
              class="flex-1 py-1.5 bg-[--bg-panel] border border-[--border-color] hover:bg-gray-700 rounded text-xs font-medium transition-colors"
              onclick={resetAndClose}
            >
              Close
            </button>
          {/if}
        </div>
      </div>
    {/if}

    <!-- Steps history -->
    {#if bisectStore.stepsHistory.length > 0}
      <div class="max-h-40 overflow-y-auto">
        {#each bisectStore.stepsHistory as step, i (i)}
          <div class="px-4 py-1.5 text-xs flex items-center justify-between border-b border-[--border-color]">
            <span class="font-mono text-gray-300 truncate flex-1">{step.node_name}</span>
            <span class="ml-2 shrink-0" class:text-green-400={step.passed} class:text-red-400={!step.passed}>
              {#if step.metric_value !== undefined && step.metric_value !== null}
                {formatMetricValue(step.metric_value, bisectStore.metric)}
              {:else if step.passed}
                OK
              {:else}
                FAIL
              {/if}
            </span>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>
