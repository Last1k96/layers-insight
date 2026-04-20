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

  const LEVEL_CLASS: Record<string, string> = {
    info:    'log-lvl log-lvl--info',
    warning: 'log-lvl log-lvl--warn',
    error:   'log-lvl log-lvl--err',
    debug:   'log-lvl log-lvl--debug',
    ov:      'log-lvl log-lvl--ov',
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

{#if logStore.visible}
  <div class="log-panel" style:height={`${height}px`}>
    <!-- Resize handle -->
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      class="log-resize"
      role="separator"
      aria-orientation="horizontal"
      onmousedown={startResize}
      title="Drag to resize"
    ></div>

    <!-- Header -->
    <div class="log-header">
      <div class="log-header__title">
        <span class="log-header__dot"></span>
        Inference Logs
        <span class="log-header__count">{logStore.entries.length}</span>
      </div>
      <button
        class="ll-icon-btn ll-icon-btn--sm"
        onclick={() => logStore.toggle()}
        title="Close logs"
        aria-label="Close logs"
      >
        <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
          <line x1="4" y1="4" x2="12" y2="12" /><line x1="12" y1="4" x2="4" y2="12" />
        </svg>
      </button>
    </div>

    <!-- Log entries (virtual scroll) -->
    <div
      bind:this={scrollContainer}
      class="log-scroll"
      onscroll={handleScroll}
    >
      <div style:height="{totalHeight}px" style:position="relative">
        <div style:position="absolute" style:top="{startIndex * ROW_HEIGHT}px" style:left="0" style:right="0">
          {#each visibleEntries as entry (entry._id)}
            <div class="log-row" style:height="{ROW_HEIGHT}px" style:line-height="{ROW_HEIGHT}px">
              <span class="log-row__time">{entry.formattedTime}</span>
              <span class={LEVEL_CLASS[entry.level] ?? 'log-lvl log-lvl--debug'}>{entry.level}</span>
              {#if entry.node_name}
                <span class="log-row__node" title={entry.node_name}>{entry.node_name}</span>
              {/if}
              <span class="log-row__msg">{entry.message}</span>
            </div>
          {/each}
        </div>
      </div>
      {#if logStore.entries.length === 0}
        <div class="log-empty">No log entries yet</div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .log-panel {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 40;
    display: flex;
    flex-direction: column;
    background: rgba(35, 38, 54, 0.96);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-top: 1px solid var(--border-soft);
    box-shadow: 0 -6px 24px rgba(0, 0, 0, 0.25);
  }

  .log-resize {
    height: 5px;
    flex-shrink: 0;
    cursor: row-resize;
    background: transparent;
    transition: background var(--dur-fast) ease;
  }
  .log-resize:hover { background: var(--accent-bg); }

  .log-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 10px 6px 14px;
    flex-shrink: 0;
    border-bottom: 1px solid var(--border-soft);
  }
  .log-header__title {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 12.5px;
    font-weight: 500;
    color: var(--text-primary);
    letter-spacing: -0.01em;
  }
  .log-header__dot {
    width: 6px;
    height: 6px;
    border-radius: var(--radius-pill);
    background: var(--status-info);
    box-shadow: 0 0 6px var(--status-info);
  }
  .log-header__count {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 10px;
    color: var(--text-muted);
    background: var(--bg-primary);
    border: 1px solid var(--border-soft);
    padding: 1px 7px;
    border-radius: var(--radius-pill);
  }

  .log-scroll {
    flex: 1;
    overflow-y: auto;
    font-family: var(--font-mono);
    font-size: 11.5px;
    min-height: 0;
  }

  .log-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 14px;
    transition: background var(--dur-fast) ease;
  }
  .log-row:hover { background: var(--accent-bg-soft); }
  .log-row__time {
    font-variant-numeric: tabular-nums;
    color: var(--text-muted-soft);
    flex-shrink: 0;
  }
  .log-row__node {
    color: var(--accent-soft);
    opacity: 0.8;
    flex-shrink: 0;
    max-width: 160px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .log-row__msg {
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .log-lvl {
    flex-shrink: 0;
    display: inline-flex;
    align-items: center;
    padding: 0 7px;
    font-size: 9.5px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: var(--radius-pill);
    border: 1px solid transparent;
  }
  .log-lvl--info  { color: var(--status-info); background: var(--status-info-bg); }
  .log-lvl--warn  { color: var(--status-warn); background: var(--status-warn-bg); border-color: var(--status-warn-border); }
  .log-lvl--err   { color: var(--status-err);  background: var(--status-err-bg);  border-color: var(--status-err-border); }
  .log-lvl--debug { color: var(--text-muted);  background: rgba(155,161,181,0.08); }
  .log-lvl--ov    { color: var(--accent-soft); background: var(--accent-bg-soft); border-color: var(--accent-border); }

  .log-empty {
    text-align: center;
    padding: 32px 16px;
    color: var(--text-muted-soft);
    font-size: 11px;
  }
</style>
