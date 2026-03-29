<script lang="ts">
  import { logStore } from '../stores/log.svelte';
  import { onMount } from 'svelte';

  let height = $state(200);
  let resizing = $state(false);
  let scrollContainer: HTMLDivElement = $state()!;
  let autoScroll = $state(true);
  let scrollTop = $state(0);
  let containerHeight = $state(0);

  const ROW_HEIGHT = 22;
  const BUFFER = 10;

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

  const totalHeight = $derived(logStore.entries.length * ROW_HEIGHT);

  const startIndex = $derived(
    Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - BUFFER)
  );

  const endIndex = $derived(
    Math.min(
      logStore.entries.length,
      Math.ceil((scrollTop + containerHeight) / ROW_HEIGHT) + BUFFER
    )
  );

  const visibleEntries = $derived(logStore.entries.slice(startIndex, endIndex));

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

  let rafPending = false;

  function handleScroll() {
    if (!scrollContainer || rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      if (!scrollContainer) return;
      scrollTop = scrollContainer.scrollTop;
      const { scrollHeight, clientHeight } = scrollContainer;
      autoScroll = scrollHeight - scrollTop - clientHeight < 30;
    });
  }

  // Auto-scroll when new entries arrive
  $effect(() => {
    const _len = logStore.entries.length;
    if (autoScroll && scrollContainer) {
      requestAnimationFrame(() => {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      });
    }
  });

  // Track container height for virtual scroll calculations
  $effect(() => {
    if (scrollContainer) {
      containerHeight = scrollContainer.clientHeight;
    }
    // Re-measure when panel height changes
    const _h = height;
  });
</script>

<!-- Toggle button — always visible, centered at bottom to avoid overlap with side panels -->
<button
  class="fixed left-1/2 -translate-x-1/2 z-50 px-4 py-1.5 text-xs font-medium rounded-md
    bg-[--bg-panel] border border-[--border-color] text-gray-200 shadow-lg
    hover:bg-[--bg-menu] hover:text-white transition-colors"
  style:bottom={logStore.visible ? `${height + 4}px` : '0.5rem'}
  onclick={() => logStore.toggle()}
>
  {logStore.visible ? 'Hide' : 'Show'} Logs
</button>

{#if logStore.visible}
  <div
    class="fixed bottom-0 left-0 right-0 z-40 bg-[--bg-panel] backdrop-blur border-t border-[--border-color] flex flex-col"
    style:height={`${height}px`}
  >
    <!-- Resize handle -->
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      class="h-1 cursor-row-resize hover:bg-blue-500/50 transition-colors shrink-0"
      role="separator"
      aria-orientation="horizontal"
      onmousedown={startResize}
    ></div>

    <!-- Header -->
    <div class="flex items-center justify-between px-3 py-1 border-b border-[--border-color] shrink-0">
      <span class="text-sm font-medium text-content-primary">Inference Logs</span>
      <div class="flex items-center gap-2">
        <button
          class="text-xs text-gray-500 hover:text-gray-300 transition-colors"
          onclick={() => logStore.clear()}
        >
          Clear
        </button>
      </div>
    </div>

    <!-- Log entries (virtual scroll) -->
    <div
      bind:this={scrollContainer}
      class="flex-1 overflow-y-auto font-mono text-xs min-h-0"
      onscroll={handleScroll}
    >
      <div style:height="{totalHeight}px" style:position="relative">
        <div style:position="absolute" style:top="{startIndex * ROW_HEIGHT}px" style:left="0" style:right="0">
          {#each visibleEntries as entry (entry._id)}
            <div class="flex gap-2 px-2 hover:bg-[--bg-menu]" style:height="{ROW_HEIGHT}px" style:line-height="{ROW_HEIGHT}px">
              <span class="text-content-secondary/50 shrink-0">{entry.formattedTime}</span>
              <span class="shrink-0 px-1 rounded text-[10px] uppercase font-semibold {LEVEL_COLORS[entry.level] ?? 'text-gray-400'} {LEVEL_BG[entry.level] ?? 'bg-gray-700/50'}">
                {entry.level}
              </span>
              {#if entry.node_name}
                <span class="text-cyan-400/70 shrink-0 truncate max-w-[150px]" title={entry.node_name}>
                  {entry.node_name}
                </span>
              {/if}
              <span class="text-content-primary truncate">{entry.message}</span>
            </div>
          {/each}
        </div>
      </div>
      {#if logStore.entries.length === 0}
        <div class="text-gray-600 text-center py-4">No log entries yet. Click a node to trigger inference.</div>
      {/if}
    </div>
  </div>
{/if}
