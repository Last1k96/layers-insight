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
  let showDropdown = $state(false);
  let highlightIndex = $state(-1);
  let debounceTimer: ReturnType<typeof setTimeout> | undefined;
  let inputEl: HTMLInputElement | undefined = $state();
  let dropdownEl: HTMLDivElement | undefined = $state();

  async function fetchSuggestions(partial: string) {
    if (!partial.trim()) {
      suggestions = [];
      showDropdown = false;
      return;
    }
    try {
      const params = new URLSearchParams({ partial, mode });
      const res = await fetch(`/api/path-suggest?${params}`);
      if (!res.ok) { suggestions = []; showDropdown = false; return; }
      const data = await res.json();
      suggestions = data.suggestions || [];
      highlightIndex = -1;
      showDropdown = suggestions.length > 0;
    } catch {
      suggestions = [];
      showDropdown = false;
    }
  }

  function onInputChange() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      fetchSuggestions(value);
    }, 200);
    oninput?.();
  }

  function selectSuggestion(s: Suggestion) {
    value = s.is_dir ? s.path + '/' : s.path;
    showDropdown = false;
    suggestions = [];
    if (onnavigate && s.is_dir) {
      onnavigate(s.path);
    } else if (!s.is_dir && onnavigate) {
      onnavigate(s.path);
    }
    oninput?.();
    // If it's a directory, immediately fetch children
    if (s.is_dir) {
      fetchSuggestions(value);
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (!showDropdown || suggestions.length === 0) {
      if (e.key === 'Enter') {
        e.preventDefault();
        onnavigate?.(value);
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      highlightIndex = (highlightIndex + 1) % suggestions.length;
      scrollToHighlighted();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      highlightIndex = highlightIndex <= 0 ? suggestions.length - 1 : highlightIndex - 1;
      scrollToHighlighted();
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (highlightIndex >= 0 && highlightIndex < suggestions.length) {
        selectSuggestion(suggestions[highlightIndex]);
      } else {
        showDropdown = false;
        onnavigate?.(value);
      }
    } else if (e.key === 'Tab') {
      if (suggestions.length === 1) {
        e.preventDefault();
        selectSuggestion(suggestions[0]);
      } else if (highlightIndex >= 0 && highlightIndex < suggestions.length) {
        e.preventDefault();
        selectSuggestion(suggestions[highlightIndex]);
      }
    } else if (e.key === 'Escape') {
      showDropdown = false;
    }
  }

  function scrollToHighlighted() {
    if (!dropdownEl || highlightIndex < 0) return;
    const items = dropdownEl.querySelectorAll('[data-suggestion]');
    items[highlightIndex]?.scrollIntoView({ block: 'nearest' });
  }

  function handleBlur(e: FocusEvent) {
    // Delay to allow click on dropdown item
    setTimeout(() => {
      if (!dropdownEl?.contains(document.activeElement)) {
        showDropdown = false;
      }
    }, 150);
  }
</script>

<div class="relative {className}">
  <input
    bind:this={inputEl}
    type="text"
    {id}
    bind:value
    {placeholder}
    oninput={onInputChange}
    onkeydown={handleKeydown}
    onblur={handleBlur}
    onfocus={() => { if (suggestions.length > 0) showDropdown = true; }}
    class="w-full px-3 py-2 bg-[--bg-input] border border-[--border-color] rounded focus:border-accent focus:outline-none"
  />
  {#if showDropdown}
    <div
      bind:this={dropdownEl}
      class="absolute z-50 left-0 right-0 top-full mt-1 max-h-48 overflow-y-auto bg-[--bg-panel] border border-[--border-color] rounded shadow-lg"
    >
      {#each suggestions as s, i (s.path)}
        <button
          data-suggestion
          class="w-full px-3 py-1.5 text-left text-sm flex items-center gap-2 hover:bg-[--bg-input] transition-colors
            {i === highlightIndex ? 'bg-[--bg-input] text-accent' : ''}"
          onmousedown={(e) => { e.preventDefault(); selectSuggestion(s); }}
        >
          <span>{s.is_dir ? '\u{1F4C1}' : '\u{1F4C4}'}</span>
          <span class="truncate">{s.name}</span>
        </button>
      {/each}
    </div>
  {/if}
</div>
