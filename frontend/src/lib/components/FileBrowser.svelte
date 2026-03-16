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
  let selectedFile = $state<string | null>(null);

  async function browse(path: string) {
    loading = true;
    error = null;
    selectedFile = null;
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
    }
  }

  function handleEntryClick(entry: BrowseEntry) {
    if (entry.is_dir) {
      browse(entry.path);
    } else {
      selectedFile = entry.path;
    }
  }

  function handleEntryDblClick(entry: BrowseEntry) {
    if (!entry.is_dir) {
      onselect(entry.path);
    }
  }

  function handleSelect() {
    if (mode === 'directory') {
      onselect(currentPath);
    } else if (selectedFile) {
      onselect(selectedFile);
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') oncancel();
  }

  // Load initial path on mount
  $effect(() => {
    browse(initialPath || '');
  });
</script>

<svelte:window onkeydown={handleKeydown} />

<!-- backdrop -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
  onmousedown={(e) => { if (e.target === e.currentTarget) oncancel(); }}
>
  <div class="bg-[--bg-panel] border border-[--border-color] rounded-lg shadow-2xl w-[36rem] max-h-[70vh] flex flex-col">
    <!-- header -->
    <div class="px-4 py-3 border-b border-[--border-color] flex items-center justify-between">
      <div class="text-sm font-medium text-content-secondary truncate flex-1 mr-4" title={currentPath}>
        {currentPath || '...'}
      </div>
      <div class="text-xs text-content-secondary">
        {mode === 'directory' ? 'Select directory' : 'Select file'}
      </div>
    </div>

    <!-- entries -->
    <div class="flex-1 overflow-y-auto min-h-0">
      {#if loading}
        <div class="p-4 text-center text-content-secondary text-sm">Loading...</div>
      {:else if error}
        <div class="p-4 text-center text-red-400 text-sm">{error}</div>
      {:else}
        <div class="divide-y divide-[--border-color]">
          {#if parentPath}
            <button
              class="w-full px-4 py-2 text-left text-sm hover:bg-[--bg-input] flex items-center gap-2 text-content-secondary"
              onclick={() => browse(parentPath!)}
            >
              <span>&#128193;</span>
              <span>..</span>
            </button>
          {/if}
          {#each entries as entry (entry.path)}
            <button
              class="w-full px-4 py-2 text-left text-sm hover:bg-[--bg-input] flex items-center gap-2 transition-colors
                {selectedFile === entry.path ? 'bg-accent/20 text-accent' : ''}"
              onclick={() => handleEntryClick(entry)}
              ondblclick={() => handleEntryDblClick(entry)}
            >
              <span>{entry.is_dir ? '\u{1F4C1}' : '\u{1F4C4}'}</span>
              <span class="truncate">{entry.name}</span>
            </button>
          {/each}
          {#if entries.length === 0}
            <div class="p-4 text-center text-content-secondary text-sm">Empty directory</div>
          {/if}
        </div>
      {/if}
    </div>

    <!-- footer -->
    <div class="px-4 py-3 border-t border-[--border-color] flex items-center justify-between gap-3">
      {#if mode === 'file' && selectedFile}
        <div class="text-xs text-content-secondary truncate flex-1" title={selectedFile}>
          {selectedFile.split('/').pop()}
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
          disabled={mode === 'file' && !selectedFile}
          onclick={handleSelect}
        >
          Select
        </button>
      </div>
    </div>
  </div>
</div>
