<script lang="ts">
  import { FILTER_FIELD_META, type FilterField, type FilterOperator, type FilterRule } from '../stores/types';

  let {
    rule,
    onUpdate,
    onDelete,
    onDragStart,
  }: {
    rule: FilterRule;
    onUpdate: (updates: Partial<Omit<FilterRule, 'id'>>) => void;
    onDelete: () => void;
    onDragStart: (e: MouseEvent) => void;
  } = $props();

  let meta = $derived(FILTER_FIELD_META[rule.field]);
  let fields = Object.keys(FILTER_FIELD_META) as FilterField[];

  function handleFieldChange(e: Event) {
    const newField = (e.target as HTMLSelectElement).value as FilterField;
    const newMeta = FILTER_FIELD_META[newField];
    onUpdate({
      field: newField,
      operator: newMeta.operators[0],
      value: '',
    });
  }

  function handleOperatorChange(e: Event) {
    onUpdate({ operator: (e.target as HTMLSelectElement).value as FilterOperator });
  }

  function handleValueInput(e: Event) {
    onUpdate({ value: (e.target as HTMLInputElement).value });
  }

  function handleEnumChange(e: Event) {
    onUpdate({ value: (e.target as HTMLSelectElement).value });
  }

  function handleWheel(e: WheelEvent) {
    e.preventDefault();
    const step = meta.step ?? 0.01;
    const current = parseFloat(rule.value) || 0;
    const delta = e.deltaY < 0 ? step : -step;
    const newVal = +(current + delta).toFixed(6);
    onUpdate({ value: String(newVal) });
  }

  function handleNumberKeydown(e: KeyboardEvent) {
    const step = meta.step ?? 0.01;
    const current = parseFloat(rule.value) || 0;
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      onUpdate({ value: String(+(current + step).toFixed(6)) });
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      onUpdate({ value: String(+(current - step).toFixed(6)) });
    }
  }

  function operatorLabel(op: FilterOperator): string {
    switch (op) {
      case 'contains': return 'contains';
      case 'equals': return '=';
      case '!=': return '!=';
      case 'starts with': return 'starts';
      default: return op;
    }
  }
</script>

<div class="flex items-center gap-1 py-0.5">
  <!-- Drag handle -->
  <button
    class="cursor-grab text-gray-600 hover:text-gray-400 px-0.5 select-none shrink-0 text-xs"
    onmousedown={onDragStart}
    title="Drag to reorder"
  >&#x2801;&#x2802;</button>

  <!-- Field selector -->
  <select
    class="px-1 py-1 bg-[--bg-input] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none min-w-0"
    value={rule.field}
    onchange={handleFieldChange}
  >
    {#each fields as f (f)}
      <option value={f}>{FILTER_FIELD_META[f].label}</option>
    {/each}
  </select>

  <!-- Operator selector -->
  <select
    class="px-1 py-1 bg-[--bg-input] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none shrink-0"
    value={rule.operator}
    onchange={handleOperatorChange}
  >
    {#each meta.operators as op (op)}
      <option value={op}>{operatorLabel(op)}</option>
    {/each}
  </select>

  <!-- Value input -->
  {#if meta.type === 'enum'}
    <select
      class="flex-1 px-1 py-1 bg-[--bg-input] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none min-w-0"
      value={rule.value}
      onchange={handleEnumChange}
    >
      <option value="">—</option>
      {#each meta.enumValues ?? [] as v (v)}
        <option value={v}>{v}</option>
      {/each}
    </select>
  {:else if meta.type === 'number'}
    <input
      type="text"
      inputmode="decimal"
      class="flex-1 px-1 py-1 bg-[--bg-input] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none min-w-0 font-mono"
      value={rule.value}
      placeholder="0"
      oninput={handleValueInput}
      onwheel={handleWheel}
      onkeydown={handleNumberKeydown}
    />
  {:else}
    <input
      type="text"
      class="flex-1 px-1 py-1 bg-[--bg-input] border border-[--border-color] rounded text-xs focus:border-blue-500 focus:outline-none min-w-0"
      value={rule.value}
      placeholder="value"
      oninput={handleValueInput}
    />
  {/if}

  <!-- Delete button -->
  <button
    class="text-gray-500 hover:text-red-400 text-xs px-0.5 shrink-0 transition-colors"
    title="Remove rule"
    onclick={onDelete}
  >&#x2715;</button>
</div>
