<script lang="ts">
  interface Suggestion {
    name: string;
    path: string;
    is_dir: boolean;
  }

  let {
    value = $bindable(''),
    mode = 'file',
    placeholder = '',
    class: className = '',
    id,
    oninput,
    onnavigate,
  }: {
    value: string;
    mode?: 'directory' | 'file';
    placeholder?: string;
    class?: string;
    id?: string;
    oninput?: () => void;
    onnavigate?: (path: string) => void;
  } = $props();

  let suggestions = $state<Suggestion[]>([]);
  let ghostSuffix = $state('');
  let selectedIndex = $state(-1);
  let showDropdown = $state(false);
  let debounceTimer: ReturnType<typeof setTimeout> | undefined;
  let inputEl: HTMLInputElement | undefined = $state();
  let listEl: HTMLUListElement | undefined = $state();

  function updateGhost() {
    if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
      const s = suggestions[selectedIndex];
      const full = s.is_dir ? s.path + '/' : s.path;
      ghostSuffix = full.length > value.length ? full.slice(value.length) : '';
    } else if (suggestions.length > 0) {
      const target = commonPrefix(suggestions);
      ghostSuffix = target.startsWith(value) ? target.slice(value.length) : '';
    } else {
      ghostSuffix = '';
    }
  }

  function commonPrefix(items: Suggestion[]): string {
    if (items.length === 0) return '';
    if (items.length === 1) {
      const s = items[0];
      return s.is_dir ? s.path + '/' : s.path;
    }
    let prefix = items[0].path;
    for (let i = 1; i < items.length; i++) {
      const p = items[i].path;
      let j = 0;
      while (j < prefix.length && j < p.length && prefix[j] === p[j]) j++;
      prefix = prefix.slice(0, j);
    }
    return prefix;
  }

  async function fetchSuggestions(partial: string) {
    if (!partial.trim()) {
      suggestions = [];
      ghostSuffix = '';
      selectedIndex = -1;
      showDropdown = false;
      return;
    }
    try {
      const params = new URLSearchParams({ partial, mode });
      const res = await fetch(`/api/path-suggest?${params}`);
      if (!res.ok) { suggestions = []; ghostSuffix = ''; selectedIndex = -1; showDropdown = false; return; }
      const data = await res.json();
      if (value !== partial) return;
      suggestions = data.suggestions || [];
      selectedIndex = -1;
      showDropdown = suggestions.length > 0;
      updateGhost();
    } catch {
      suggestions = [];
      ghostSuffix = '';
      selectedIndex = -1;
      showDropdown = false;
    }
  }

  function onInputChange() {
    ghostSuffix = '';
    selectedIndex = -1;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      fetchSuggestions(value);
    }, 200);
    oninput?.();
  }

  function selectItem(index: number) {
    const s = suggestions[index];
    value = s.is_dir ? s.path + '/' : s.path;
    selectedIndex = -1;
    ghostSuffix = '';
    showDropdown = false;
    oninput?.();
    inputEl?.focus();
    if (s.is_dir) {
      onnavigate?.(value);
      fetchSuggestions(value);
    } else {
      suggestions = [];
    }
  }

  function scrollSelectedIntoView() {
    if (!listEl || selectedIndex < 0) return;
    const item = listEl.children[selectedIndex] as HTMLElement | undefined;
    item?.scrollIntoView({ block: 'nearest' });
  }

  /** Accept the ghost suffix into the value and continue navigating. */
  function acceptGhost() {
    if (!ghostSuffix) return false;
    value = value + ghostSuffix;
    ghostSuffix = '';
    selectedIndex = -1;
    oninput?.();
    if (value.endsWith('/')) {
      onnavigate?.(value);
      fetchSuggestions(value);
    } else {
      suggestions = [];
      showDropdown = false;
    }
    return true;
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Tab') {
      e.preventDefault();
      // 1) If there's a ghost suffix (common prefix or selected item), accept it first
      if (ghostSuffix) {
        acceptGhost();
        return;
      }
      // 2) No suggestions yet — fetch them
      if (suggestions.length === 0 && value.trim()) {
        fetchSuggestions(value);
        return;
      }
      if (suggestions.length === 0) return;
      // 3) Cycle through dropdown items
      showDropdown = true;
      if (e.shiftKey) {
        selectedIndex = selectedIndex <= 0 ? suggestions.length - 1 : selectedIndex - 1;
      } else {
        selectedIndex = selectedIndex < suggestions.length - 1 ? selectedIndex + 1 : 0;
      }
      updateGhost();
      scrollSelectedIntoView();
      return;
    }
    if ((e.key === 'ArrowDown' || e.key === 'ArrowUp') && suggestions.length > 0) {
      e.preventDefault();
      showDropdown = true;
      if (e.key === 'ArrowDown') {
        selectedIndex = selectedIndex < suggestions.length - 1 ? selectedIndex + 1 : 0;
      } else {
        selectedIndex = selectedIndex <= 0 ? suggestions.length - 1 : selectedIndex - 1;
      }
      updateGhost();
      scrollSelectedIntoView();
      return;
    }
    if (e.key === 'ArrowRight' && ghostSuffix) {
      if (inputEl && inputEl.selectionStart === value.length) {
        e.preventDefault();
        acceptGhost();
        return;
      }
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      // If there's a ghost suffix, accept it (complete the path)
      if (ghostSuffix) {
        acceptGhost();
        return;
      }
      // If an item is selected in dropdown, pick it
      if (selectedIndex >= 0) {
        selectItem(selectedIndex);
        return;
      }
      // Otherwise commit the current value
      ghostSuffix = '';
      suggestions = [];
      showDropdown = false;
      onnavigate?.(value);
      return;
    }
    if (e.key === 'Escape') {
      if (showDropdown || ghostSuffix) {
        e.preventDefault();
        ghostSuffix = '';
        suggestions = [];
        selectedIndex = -1;
        showDropdown = false;
      }
    }
  }

  function handleBlur(e: FocusEvent) {
    // Close dropdown when focus leaves the component entirely
    const related = e.relatedTarget as HTMLElement | null;
    if (related && (related.closest('.path-dropdown') || related === inputEl)) return;
    showDropdown = false;
    ghostSuffix = '';
    selectedIndex = -1;
  }
</script>

<div class="relative min-w-0 {className}">
  <div class="relative overflow-hidden w-full">
    {#if ghostSuffix}
      <div
        class="absolute inset-0 flex items-center pointer-events-none
               px-3 py-2 border border-transparent"
        aria-hidden="true"
      >
        <span class="invisible whitespace-pre">{value}</span><span
          class="text-[--text-secondary] opacity-50 whitespace-pre"
        >{ghostSuffix}</span>
      </div>
    {/if}
    <input
      bind:this={inputEl}
      type="text"
      {id}
      bind:value
      {placeholder}
      oninput={onInputChange}
      onkeydown={handleKeydown}
      onblur={handleBlur}
      autocomplete="off"
      class="w-full px-3 py-2 border border-[--border-color] rounded
             focus:border-accent focus:outline-none
             {ghostSuffix ? 'bg-transparent' : 'bg-[--bg-input]'}"
    />
  </div>

  {#if showDropdown && suggestions.length > 0}
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <ul
      bind:this={listEl}
      class="path-dropdown"
      onmousedown={(e) => e.preventDefault()}
    >
      {#each suggestions as s, i (s.path)}
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <li
          class="path-dropdown-item"
          class:selected={i === selectedIndex}
          onmouseenter={() => { selectedIndex = i; updateGhost(); }}
          onclick={() => selectItem(i)}
          role="option"
          aria-selected={i === selectedIndex}
        >
          <span class="item-icon">{s.is_dir ? '📁' : '📄'}</span>
          <span class="item-name" class:is-dir={s.is_dir}>{s.name}{#if s.is_dir}/{/if}</span>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .path-dropdown {
    position: absolute;
    z-index: 50;
    left: 0;
    right: 0;
    top: 100%;
    margin-top: 2px;
    max-height: 200px;
    overflow-y: auto;
    background: var(--bg-menu);
    border: 1px solid var(--border-color);
    border-radius: 0.375rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    list-style: none;
    padding: 0.25rem 0;
    margin-bottom: 0;
    margin-left: 0;
  }

  .path-dropdown::-webkit-scrollbar { width: 4px; }
  .path-dropdown::-webkit-scrollbar-track { background: transparent; }
  .path-dropdown::-webkit-scrollbar-thumb { background: #3A3F56; border-radius: 99px; }

  .path-dropdown-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.6rem;
    cursor: pointer;
    font-size: 0.8rem;
    font-family: var(--font-mono);
    color: var(--text-primary);
    transition: background 0.1s ease;
  }

  .path-dropdown-item:hover,
  .path-dropdown-item.selected {
    background: rgba(76, 141, 255, 0.12);
  }

  .path-dropdown-item.selected {
    background: rgba(76, 141, 255, 0.18);
  }

  .item-icon {
    font-size: 0.75rem;
    flex-shrink: 0;
    width: 1rem;
    text-align: center;
  }

  .item-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .item-name.is-dir {
    color: #4C8DFF;
  }
</style>
