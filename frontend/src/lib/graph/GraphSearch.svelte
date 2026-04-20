<script lang="ts">
  import { graphStore } from '../stores/graph.svelte';
  import { sessionStore } from '../stores/session.svelte';
  import { centerOnNode, refreshRenderer } from './renderer';

  let inputEl: HTMLInputElement = $state()!;
  let listEl: HTMLDivElement = $state()!;
  let query = $state('');
  let debounceTimer: ReturnType<typeof setTimeout>;

  function handleInput() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      const sessionId = sessionStore.currentSession?.id;
      if (sessionId) {
        graphStore.searchNodes(sessionId, query);
        refreshRenderer();
      }
    }, 200);
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault();
      const current = graphStore.searchResults[graphStore.searchIndex];
      if (current) {
        graphStore.selectNode(current.id);
        centerOnNode(current.id);
        close();
      }
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      const node = graphStore.cycleSearchResult(e.key === 'ArrowDown' ? 1 : -1);
      if (node) {
        centerOnNode(node.id);
        refreshRenderer();
      }
    } else if (e.key === 'Escape') {
      close();
    }
  }

  function close() {
    graphStore.searchVisible = false;
    graphStore.searchResults = [];
    query = '';
    refreshRenderer();
  }

  $effect(() => {
    if (graphStore.searchVisible && inputEl) {
      inputEl.focus();
    }
  });

  $effect(() => {
    // Scroll the active result into view
    const idx = graphStore.searchIndex;
    if (idx >= 0 && listEl) {
      const item = listEl.children[idx] as HTMLElement | undefined;
      item?.scrollIntoView({ block: 'nearest' });
    }
  });
</script>

{#if graphStore.searchVisible}
  <div class="search-root">
    <div class="search-shell">
      <div class="search-header">
        <svg class="search-icon" width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          bind:this={inputEl}
          bind:value={query}
          oninput={handleInput}
          onkeydown={handleKeydown}
          placeholder="Search nodes by name or type…"
          class="search-input"
        />
        {#if graphStore.searchResults.length > 0}
          <span class="search-count">
            {graphStore.searchIndex + 1}<span class="search-count__sep">/</span>{graphStore.searchResults.length}
          </span>
        {/if}
        <button onclick={close} class="search-esc" title="Close (Esc)">ESC</button>
      </div>

      {#if graphStore.searchResults.length > 0}
        <div bind:this={listEl} class="search-list">
          {#each graphStore.searchResults as result, i (result.id)}
            <button
              class="search-row"
              class:search-row--active={i === graphStore.searchIndex}
              onclick={() => {
                graphStore.searchIndex = i;
                graphStore.selectNode(result.id);
                centerOnNode(result.id);
                refreshRenderer();
              }}
            >
              <span class="search-row__name">{result.name}</span>
              <span class="search-row__type">{result.type}</span>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .search-root {
    position: absolute;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 30;
    width: min(420px, 80vw);
  }
  .search-shell {
    background: rgba(35, 38, 54, 0.96);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-modal);
    overflow: hidden;
  }
  .search-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
  }
  .search-icon { color: var(--text-muted); flex-shrink: 0; }
  .search-input {
    flex: 1;
    background: transparent;
    border: 0;
    outline: none;
    font-size: 13px;
    color: var(--text-primary);
  }
  .search-input::placeholder { color: var(--text-muted-soft); }
  .search-count {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
    font-size: 11px;
    color: var(--text-muted);
    padding: 2px 7px;
    border-radius: var(--radius-pill);
    background: var(--accent-bg-soft);
  }
  .search-count__sep { opacity: 0.5; margin: 0 1px; }
  .search-esc {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 3px 7px;
    border-radius: var(--radius-sm);
    color: var(--text-muted);
    background: var(--bg-primary);
    border: 1px solid var(--border-soft);
    transition: color var(--dur-fast) ease, border-color var(--dur-fast) ease;
  }
  .search-esc:hover { color: var(--text-primary); border-color: var(--accent-border); }
  .search-list {
    border-top: 1px solid var(--border-soft);
    max-height: 240px;
    overflow-y: auto;
  }
  .search-row {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 7px 12px;
    font-size: 12.5px;
    color: var(--text-primary);
    background: transparent;
    border: 0;
    border-left: 2px solid transparent;
    text-align: left;
    cursor: pointer;
    transition: background var(--dur-fast) ease;
  }
  .search-row:hover { background: var(--accent-bg-soft); }
  .search-row--active {
    background: var(--accent-bg);
    border-left-color: var(--accent);
  }
  .search-row__name {
    font-family: var(--font-mono);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .search-row__type {
    font-size: 11px;
    color: var(--text-muted-soft);
    flex-shrink: 0;
  }
</style>
