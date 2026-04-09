<script lang="ts">
  interface BrowseEntry {
    name: string;
    path: string;
    is_dir: boolean;
  }

  interface BrowseResult {
    current: string;
    parent: string | null;
    entries: BrowseEntry[];
  }

  let {
    mode,
    initialPath = '',
    onselect,
    oncancel,
  }: {
    mode: 'directory' | 'file';
    initialPath?: string;
    onselect: (path: string) => void;
    oncancel: () => void;
  } = $props();

  let currentPath = $state('');
  let entries = $state<BrowseEntry[]>([]);
  let parentPath = $state<string | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);

  let searchQuery = $state('');
  let selectedIndex = $state(-1);
  let searchEl: HTMLInputElement | undefined = $state();
  let listEl: HTMLDivElement | undefined = $state();

  /** ".." entry prepended when a parent exists, so it's navigable via keyboard. */
  let parentEntry = $derived<BrowseEntry | null>(
    parentPath ? { name: '..', path: parentPath, is_dir: true } : null
  );

  /** Entries filtered by search query. Starts-with matches first, then contains. ".." always included. */
  let filteredEntries = $derived.by(() => {
    const q = searchQuery.toLowerCase().trim();
    const prefix: BrowseEntry[] = parentEntry ? [parentEntry] : [];
    if (!q) return [...prefix, ...entries];
    const startsWith: BrowseEntry[] = [];
    const contains: BrowseEntry[] = [];
    for (const e of entries) {
      const name = e.name.toLowerCase();
      if (name.startsWith(q)) startsWith.push(e);
      else if (name.includes(q)) contains.push(e);
    }
    return [...prefix, ...startsWith, ...contains];
  });

  // Auto-select first match when filtered entries change
  $effect(() => {
    if (searchQuery.trim() && filteredEntries.length > 0) {
      // Skip ".." when auto-selecting a search match
      selectedIndex = parentEntry ? Math.min(1, filteredEntries.length - 1) : 0;
    } else if (filteredEntries.length === 0) {
      selectedIndex = -1;
    }
  });

  async function browse(path: string) {
    loading = true;
    error = null;
    searchQuery = '';
    selectedIndex = -1;
    try {
      const params = new URLSearchParams({ path, mode });
      const res = await fetch(`/api/browse?${params}`);
      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: 'Request failed' }));
        error = detail.detail || 'Request failed';
        return;
      }
      const result: BrowseResult = await res.json();
      currentPath = result.current;
      parentPath = result.parent;
      entries = result.entries;
    } catch (e) {
      error = 'Failed to browse directory';
    } finally {
      loading = false;
      // Re-focus the search input after navigation
      requestAnimationFrame(() => searchEl?.focus());
    }
  }

  function handleEntryClick(entry: BrowseEntry) {
    if (entry.is_dir) {
      browse(entry.path);
    } else {
      // Find index in filtered list and select it
      const idx = filteredEntries.indexOf(entry);
      selectedIndex = idx;
    }
  }

  function handleEntryDblClick(entry: BrowseEntry) {
    if (entry.is_dir) {
      browse(entry.path);
    } else {
      onselect(entry.path);
    }
  }

  function handleSelect() {
    if (mode === 'directory') {
      onselect(currentPath);
    } else if (selectedIndex >= 0 && selectedIndex < filteredEntries.length) {
      const entry = filteredEntries[selectedIndex];
      if (!entry.is_dir) onselect(entry.path);
    }
  }

  function scrollSelectedIntoView() {
    if (!listEl || selectedIndex < 0) return;
    // listEl > div.divide-y > buttons
    const container = listEl.firstElementChild;
    if (!container) return;
    const item = container.children[selectedIndex] as HTMLElement | undefined;
    item?.scrollIntoView({ block: 'nearest' });
  }

  function handleSearchKeydown(e: KeyboardEvent) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (filteredEntries.length > 0) {
        selectedIndex = selectedIndex < filteredEntries.length - 1 ? selectedIndex + 1 : 0;
        scrollSelectedIntoView();
      }
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (filteredEntries.length > 0) {
        selectedIndex = selectedIndex <= 0 ? filteredEntries.length - 1 : selectedIndex - 1;
        scrollSelectedIntoView();
      }
      return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      if (selectedIndex >= 0 && selectedIndex < filteredEntries.length) {
        const entry = filteredEntries[selectedIndex];
        if (entry.is_dir) {
          browse(entry.path);
        } else {
          onselect(entry.path);
        }
      }
      return;
    }
    if (e.key === 'Escape') {
      oncancel();
    }
  }

  function handleGlobalKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') oncancel();
  }

  // Load initial path on mount
  $effect(() => {
    browse(initialPath || '');
  });
</script>

<svelte:window onkeydown={handleGlobalKeydown} />

<!-- backdrop -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
  onmousedown={(e) => { if (e.target === e.currentTarget) oncancel(); }}
>
  <div class="bg-[--bg-panel] border border-[--border-color] rounded-lg shadow-2xl w-[36rem] h-[70vh] flex flex-col">
    <!-- header: current path + search -->
    <div class="px-4 py-3 border-b border-[--border-color] flex flex-col gap-2">
      <div class="flex items-center justify-between gap-3">
        <div class="text-xs text-[--text-secondary] font-mono truncate flex-1" title={currentPath}>
          {currentPath || '...'}
        </div>
        <div class="text-xs text-[--text-secondary] whitespace-nowrap">
          {mode === 'directory' ? 'Select directory' : 'Select file'}
        </div>
      </div>
      <input
        bind:this={searchEl}
        type="text"
        bind:value={searchQuery}
        placeholder="Type to filter..."
        onkeydown={handleSearchKeydown}
        autocomplete="off"
        class="w-full px-3 py-1.5 text-sm font-mono bg-[--bg-input] border border-[--border-color] rounded
               focus:border-[#4C8DFF] focus:outline-none text-[--text-primary]
               placeholder:text-[--text-secondary] placeholder:opacity-50"
      />
    </div>

    <!-- entries -->
    <div bind:this={listEl} class="flex-1 overflow-y-auto min-h-0">
      {#if loading}
        <div class="p-4 text-center text-[--text-secondary] text-sm">Loading...</div>
      {:else if error}
        <div class="p-4 text-center text-red-400 text-sm">{error}</div>
      {:else}
        <div class="divide-y divide-[--border-color]">
          {#each filteredEntries as entry, i (entry.path)}
            <button
              class="w-full px-4 py-2 text-left text-sm hover:bg-[--bg-input] flex items-center gap-2 transition-colors
                {i === selectedIndex ? 'bg-[rgba(76,141,255,0.15)] text-[#4C8DFF]' : ''}"
              onclick={() => handleEntryClick(entry)}
              ondblclick={() => handleEntryDblClick(entry)}
            >
              <span>{entry.is_dir ? '📁' : '📄'}</span>
              <span class="truncate">{entry.name}</span>
            </button>
          {/each}
          {#if filteredEntries.length === 0 && searchQuery.trim()}
            <div class="p-4 text-center text-[--text-secondary] text-sm">No matches</div>
          {:else if entries.length === 0}
            <div class="p-4 text-center text-[--text-secondary] text-sm">Empty directory</div>
          {/if}
        </div>
      {/if}
    </div>

    <!-- footer -->
    <div class="px-4 py-3 border-t border-[--border-color] flex items-center justify-between gap-3">
      {#if mode === 'file' && selectedIndex >= 0 && filteredEntries[selectedIndex] && !filteredEntries[selectedIndex].is_dir}
        <div class="text-xs text-[--text-secondary] truncate flex-1 font-mono" title={filteredEntries[selectedIndex].path}>
          {filteredEntries[selectedIndex].name}
        </div>
      {:else}
        <div class="flex-1"></div>
      {/if}
      <div class="flex gap-2">
        <button
          class="px-4 py-1.5 text-sm border border-[--border-color] rounded hover:bg-[--bg-input] transition-colors"
          onclick={oncancel}
        >
          Cancel
        </button>
        <button
          class="px-4 py-1.5 text-sm bg-accent hover:bg-accent-hover rounded font-medium transition-colors disabled:opacity-40"
          disabled={mode === 'file' && (selectedIndex < 0 || !filteredEntries[selectedIndex] || filteredEntries[selectedIndex].is_dir)}
          onclick={handleSelect}
        >
          Select
        </button>
      </div>
    </div>
  </div>
</div>
