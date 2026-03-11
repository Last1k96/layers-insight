<script lang="ts">
  import { logStore } from '../stores/log.svelte';
  import { onMount } from 'svelte';

  let height = $state(200);
  let resizing = $state(false);
  let scrollContainer: HTMLDivElement;
  let autoScroll = $state(true);

  const LEVEL_COLORS: Record<string, string> = {
    info: 'text-blue-400',
    warning: 'text-yellow-400',
    error: 'text-red-400',
    debug: 'text-gray-500',
    ov: 'text-purple-400',
  };

  const LEVEL_BG: Record<string, string> = {
    info: 'bg-blue-900/50',
    warning: 'bg-yellow-900/50',
    error: 'bg-red-900/50',
    debug: 'bg-gray-700/50',
    ov: 'bg-purple-900/50',
  };

  onMount(() => {
    const saved = localStorage.getItem('log-panel-height');
    if (saved) height = parseInt(saved);
  });

  function startResize(e: MouseEvent) {
    e.preventDefault();
    resizing = true;
    const startY = e.clientY;
    const startHeight = height;

    function onMouseMove(e: MouseEvent) {
      const dy = startY - e.clientY;
      height = Math.max(100, Math.min(600, startHeight + dy));
    }

    function onMouseUp() {
      resizing = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      localStorage.setItem('log-panel-height', String(height));
    }

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }

  function handleScroll() {
    if (!scrollContainer) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
    autoScroll = scrollHeight - scrollTop - clientHeight < 30;
  }

  function formatTime(ts: string): string {
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        fractionalSecondDigits: 3,
      });
    } catch {
      return ts;
    }
  }

  $effect(() => {
    // Trigger on entries length change
    const _len = logStore.entries.length;
    if (autoScroll && scrollContainer) {
      requestAnimationFrame(() => {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      });
    }
  });
</script>

<!-- Toggle button — always visible, centered at bottom to avoid overlap with side panels -->
<button
  class="fixed left-1/2 -translate-x-1/2 z-50 px-4 py-1.5 text-xs font-medium rounded-md
    bg-gray-800 border border-gray-600 text-gray-200 shadow-lg
    hover:bg-gray-700 hover:text-white transition-colors"
  style:bottom={logStore.visible ? `${height + 4}px` : '0.5rem'}
  onclick={() => logStore.toggle()}
>
  {logStore.visible ? 'Hide' : 'Show'} Logs
  {#if logStore.entries.length > 0}
    <span class="ml-1 px-1.5 py-0.5 text-[10px] bg-blue-600 text-white rounded-full">
      {logStore.entries.length}
    </span>
  {/if}
</button>

{#if logStore.visible}
  <div
    class="fixed bottom-0 left-0 right-0 z-40 bg-gray-800/95 backdrop-blur border-t border-gray-700 flex flex-col"
    style:height={`${height}px`}
  >
    <!-- Resize handle -->
    <div
      class="h-1 cursor-row-resize hover:bg-blue-500/50 transition-colors shrink-0"
      onmousedown={startResize}
    ></div>

    <!-- Header -->
    <div class="flex items-center justify-between px-3 py-1 border-b border-gray-700 shrink-0">
      <span class="text-sm font-medium text-gray-300">Inference Logs</span>
      <div class="flex items-center gap-2">
        <button
          class="text-xs text-gray-500 hover:text-gray-300 transition-colors"
          onclick={() => logStore.clear()}
        >
          Clear
        </button>
      </div>
    </div>

    <!-- Log entries -->
    <div
      bind:this={scrollContainer}
      class="flex-1 overflow-y-auto font-mono text-xs p-1 min-h-0"
      onscroll={handleScroll}
    >
      {#each logStore.entries as entry}
        <div class="flex gap-2 px-2 py-0.5 hover:bg-gray-700/30">
          <span class="text-gray-600 shrink-0">{formatTime(entry.timestamp)}</span>
          <span class="shrink-0 px-1 rounded text-[10px] uppercase font-semibold {LEVEL_COLORS[entry.level] ?? 'text-gray-400'} {LEVEL_BG[entry.level] ?? 'bg-gray-700/50'}">
            {entry.level}
          </span>
          {#if entry.node_name}
            <span class="text-cyan-400/70 shrink-0 truncate max-w-[150px]" title={entry.node_name}>
              {entry.node_name}
            </span>
          {/if}
          <span class="text-gray-300 break-all">{entry.message}</span>
        </div>
      {/each}
      {#if logStore.entries.length === 0}
        <div class="text-gray-600 text-center py-4">No log entries yet. Click a node to trigger inference.</div>
      {/if}
    </div>
  </div>
{/if}
