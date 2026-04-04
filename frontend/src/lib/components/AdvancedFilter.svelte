<script lang="ts">
  import { advancedFilterStore } from '../stores/advancedFilter.svelte';
  import FilterRuleRow from './FilterRule.svelte';

  let { ontoggle }: { ontoggle: () => void } = $props();

  // Drag state
  let draggingIndex = $state<number | null>(null);
  let dropIndex = $state<number | null>(null);
  let ruleElements: HTMLDivElement[] = [];
  let containerEl: HTMLDivElement | undefined = $state();

  function startDrag(index: number, e: MouseEvent) {
    e.preventDefault();
    draggingIndex = index;
    dropIndex = index;
    const startY = e.clientY;

    function onMove(ev: MouseEvent) {
      if (draggingIndex === null || !containerEl) return;
      // Calculate drop position based on mouse Y relative to rule elements
      const rects = ruleElements.map(el => el?.getBoundingClientRect()).filter(Boolean);
      let newDrop = rects.length - 1;
      for (let i = 0; i < rects.length; i++) {
        const rect = rects[i];
        if (!rect) continue;
        const mid = rect.top + rect.height / 2;
        if (ev.clientY < mid) {
          newDrop = i;
          break;
        }
      }
      dropIndex = newDrop;
    }

    function onUp() {
      if (draggingIndex !== null && dropIndex !== null && draggingIndex !== dropIndex) {
        advancedFilterStore.reorderRules(draggingIndex, dropIndex);
      }
      draggingIndex = null;
      dropIndex = null;
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.userSelect = '';
    }

    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }

  function getRuleTransform(index: number): string {
    if (draggingIndex === null || dropIndex === null) return '';
    if (index === draggingIndex) return '';
    // Shift rules between drag source and drop target
    if (draggingIndex < dropIndex) {
      if (index > draggingIndex && index <= dropIndex) return 'translateY(-100%)';
    } else {
      if (index < draggingIndex && index >= dropIndex) return 'translateY(100%)';
    }
    return '';
  }
</script>

<div class="flex flex-col gap-0.5" bind:this={containerEl}>
  <!-- Header -->
  <div class="flex items-center gap-1 mb-0.5">
    <button
      class="px-2 py-0.5 text-xs rounded bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 transition-colors"
      onclick={() => advancedFilterStore.addRule()}
    >+ Add Rule</button>
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

  <!-- Rules list -->
  {#if advancedFilterStore.rules.length === 0}
    <div class="text-[10px] text-gray-500 text-center py-1">No rules. Click "+ Add Rule" to start.</div>
  {:else}
    <div class="max-h-40 overflow-y-auto flex flex-col">
      {#each advancedFilterStore.rules as rule, i (rule.id)}
        <!-- Connector between rules -->
        {#if i > 0}
          {@const conn = advancedFilterStore.connectors[i - 1] ?? 'AND'}
          <div class="flex items-center py-0.5 pl-1">
            <button
              class="px-1.5 py-0 text-[10px] font-semibold rounded transition-colors {conn === 'AND' ? 'text-blue-400 bg-blue-900/30' : 'text-amber-400 bg-amber-900/30'}"
              onclick={() => advancedFilterStore.toggleConnector(i - 1)}
              title="Click to toggle AND/OR"
            >{conn}</button>
          </div>
        {/if}

        <!-- Rule row -->
        <div
          bind:this={ruleElements[i]}
          class="transition-transform duration-150"
          class:opacity-50={draggingIndex === i}
          class:z-10={draggingIndex === i}
          style:transform={getRuleTransform(i)}
        >
          <FilterRuleRow
            {rule}
            onUpdate={(updates) => advancedFilterStore.updateRule(rule.id, updates)}
            onDelete={() => advancedFilterStore.removeRule(rule.id)}
            onDragStart={(e) => startDrag(i, e)}
          />
        </div>
      {/each}
    </div>
  {/if}
</div>
