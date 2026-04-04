<script lang="ts">
  import { advancedFilterStore } from '../stores/advancedFilter.svelte';
  import FilterRuleRow from './FilterRule.svelte';

  let { ontoggle }: { ontoggle: () => void } = $props();

  const CONN_COL_W = 'w-9';
  const GAP_PX = 6; // matches gap-1.5 = 0.375rem ≈ 6px

  // Drag state
  let draggingIndex = $state<number | null>(null);
  let dropIndex = $state<number | null>(null);
  let dragDeltaY = $state(0);
  let rowTops: number[] = [];
  let rowHeight = 0;
  let rowStride = 0;

  let rowElements: HTMLDivElement[] = [];
  let listEl: HTMLDivElement | undefined = $state();

  // Connector positions (measured from top of list container)
  let connectorTops = $state<number[]>([]);
  let measuredRowHeight = $state(0);

  function measureConnectors() {
    if (!listEl || rowElements.length < 2) {
      connectorTops = [];
      if (rowElements.length === 1) {
        measuredRowHeight = rowElements[0]?.getBoundingClientRect()?.height ?? 0;
      }
      return;
    }
    const listRect = listEl.getBoundingClientRect();
    const tops: number[] = [];
    for (let i = 1; i < rowElements.length; i++) {
      const prev = rowElements[i - 1]?.getBoundingClientRect();
      const curr = rowElements[i]?.getBoundingClientRect();
      if (prev && curr) {
        tops.push((prev.bottom + curr.top) / 2 - listRect.top);
      }
    }
    connectorTops = tops;
    measuredRowHeight = rowElements[0]?.getBoundingClientRect()?.height ?? 0;
  }

  // Re-measure when rules change (after DOM updates)
  $effect(() => {
    // Touch reactive deps
    advancedFilterStore.rules.length;
    // Wait a tick for DOM to update
    requestAnimationFrame(measureConnectors);
  });

  function startDrag(index: number, e: MouseEvent) {
    e.preventDefault();

    const rects = rowElements.map(el => el?.getBoundingClientRect());
    rowTops = rects.map(r => r?.top ?? 0);
    rowHeight = rects[0]?.height ?? 0;
    rowStride = rects.length > 1 ? (rects[1]!.top - rects[0]!.top) : rowHeight;

    const startY = e.clientY;
    draggingIndex = index;
    dropIndex = index;
    dragDeltaY = 0;

    function onMove(ev: MouseEvent) {
      if (draggingIndex === null) return;
      dragDeltaY = ev.clientY - startY;

      const draggedCenter = rowTops[draggingIndex!] + rowHeight / 2 + dragDeltaY;
      let newDrop = 0;
      for (let i = 0; i < rowTops.length; i++) {
        const mid = rowTops[i] + rowHeight / 2;
        if (draggedCenter > mid) {
          newDrop = i;
        }
      }
      dropIndex = Math.max(0, Math.min(rowTops.length - 1, newDrop));
    }

    function onUp() {
      if (draggingIndex !== null && dropIndex !== null && draggingIndex !== dropIndex) {
        advancedFilterStore.reorderRules(draggingIndex, dropIndex);
      }
      draggingIndex = null;
      dropIndex = null;
      dragDeltaY = 0;
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.userSelect = '';
    }

    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }

  function getRowStyle(index: number): string {
    if (draggingIndex === null || dropIndex === null) return '';

    if (index === draggingIndex) {
      return `transform: translateY(${dragDeltaY}px); z-index: 20; position: relative;`;
    }

    const di = draggingIndex;
    const dp = dropIndex;

    if (di < dp) {
      if (index > di && index <= dp) {
        return `transform: translateY(${-rowStride}px); transition: transform 150ms ease;`;
      }
    } else if (di > dp) {
      if (index >= dp && index < di) {
        return `transform: translateY(${rowStride}px); transition: transform 150ms ease;`;
      }
    }
    return 'transition: transform 150ms ease;';
  }
</script>

<div class="flex flex-col">
  <!-- Header -->
  <div class="flex items-center gap-1 mb-1.5">
    <span class="text-[10px] text-gray-500 uppercase tracking-wider">Advanced Filter</span>
    <div class="flex-1"></div>
    <button
      class="px-1.5 py-0.5 text-xs rounded transition-colors text-gray-400 hover:text-gray-200 hover:bg-[--bg-menu]"
      title="Switch to simple filter"
      onclick={ontoggle}
    >
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M1 3h14M3 8h10M5 13h6" />
      </svg>
    </button>
  </div>

  <!-- Content -->
  <div>
    {#if advancedFilterStore.rules.length === 0}
      <div class="flex">
        <div class="{CONN_COL_W} shrink-0"></div>
        <button
          class="flex-1 min-w-0 py-2 text-xs rounded border border-dashed border-[--border-color] text-gray-500 hover:text-blue-400 hover:border-blue-500/40 transition-colors"
          onclick={() => advancedFilterStore.addRule()}
        >+ Add Rule</button>
      </div>
    {:else}
      <div class="relative" bind:this={listEl}>
        <!-- AND/OR connectors: separate layer, not inside row divs -->
        {#each connectorTops as top, i (i)}
          {@const conn = advancedFilterStore.connectors[i] ?? 'AND'}
          <div
            class="absolute left-0 flex items-center justify-center z-10"
            style="top: {top}px; transform: translateY(-50%); width: 2.25rem;"
          >
            <button
              class="px-1.5 text-[10px] font-bold rounded border transition-colors flex items-center justify-center {conn === 'AND' ? 'text-blue-400 border-blue-500/40 bg-blue-900/20 hover:bg-blue-900/40' : 'text-amber-400 border-amber-500/40 bg-amber-900/20 hover:bg-amber-900/40'}"
              style="height: {measuredRowHeight * 0.75}px;"
              onclick={() => advancedFilterStore.toggleConnector(i)}
              title="Click to toggle AND/OR"
            >{conn}</button>
          </div>
        {/each}

        <!-- Rule rows -->
        <div class="flex flex-col gap-1.5">
          {#each advancedFilterStore.rules as rule, i (rule.id)}
            <div
              bind:this={rowElements[i]}
              class="flex"
              class:opacity-60={draggingIndex === i}
              style={getRowStyle(i)}
            >
              <div class="{CONN_COL_W} shrink-0"></div>
              <div class="flex-1 min-w-0">
                <FilterRuleRow
                  {rule}
                  onUpdate={(updates) => advancedFilterStore.updateRule(rule.id, updates)}
                  onDelete={() => advancedFilterStore.removeRule(rule.id)}
                  onDragStart={(e) => startDrag(i, e)}
                />
              </div>
            </div>
          {/each}
        </div>

        <!-- Add Rule button (max 10 rules) -->
        <div class="flex mt-1.5">
          <div class="{CONN_COL_W} shrink-0"></div>
          <button
            class="flex-1 min-w-0 py-2 text-xs rounded border border-dashed border-[--border-color] transition-colors {advancedFilterStore.rules.length >= 10 ? 'text-gray-600 cursor-not-allowed opacity-50' : 'text-gray-500 hover:text-blue-400 hover:border-blue-500/40'}"
            disabled={advancedFilterStore.rules.length >= 10}
            onclick={() => advancedFilterStore.addRule()}
          >+ Add Rule{advancedFilterStore.rules.length >= 10 ? ' (max 10)' : ''}</button>
        </div>
      </div>
    {/if}
  </div>
</div>
