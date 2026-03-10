<script lang="ts">
  import { graphStore } from '../stores/graph.svelte';
  import { sessionStore } from '../stores/session.svelte';
  import { centerOnNode, refreshRenderer } from './renderer';

  let inputEl: HTMLInputElement;
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
      const node = graphStore.cycleSearchResult(e.shiftKey ? -1 : 1);
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
</script>

{#if graphStore.searchVisible}
  <div class="absolute top-3 left-1/2 -translate-x-1/2 z-30 w-96">
    <div class="bg-gray-800/95 backdrop-blur border border-gray-600 rounded-lg shadow-xl">
      <div class="flex items-center px-3 py-2 gap-2">
        <svg class="w-4 h-4 text-gray-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          bind:this={inputEl}
          bind:value={query}
          oninput={handleInput}
          onkeydown={handleKeydown}
          placeholder="Search nodes by name or type..."
          class="flex-1 bg-transparent border-none focus:outline-none text-sm"
        />
        {#if graphStore.searchResults.length > 0}
          <span class="text-xs text-gray-400">
            {graphStore.searchIndex + 1}/{graphStore.searchResults.length}
          </span>
        {/if}
        <button onclick={close} class="text-gray-400 hover:text-gray-200 text-xs">ESC</button>
      </div>

      {#if graphStore.searchResults.length > 0}
        <div class="border-t border-gray-700 max-h-48 overflow-y-auto">
          {#each graphStore.searchResults.slice(0, 10) as result, i (result.id)}
            <button
              class="w-full text-left px-3 py-1.5 text-sm hover:bg-gray-700 flex justify-between"
              class:bg-gray-700={i === graphStore.searchIndex}
              onclick={() => {
                graphStore.searchIndex = i;
                graphStore.selectNode(result.id);
                centerOnNode(result.id);
                refreshRenderer();
              }}
            >
              <span>{result.name}</span>
              <span class="text-gray-500">{result.type}</span>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </div>
{/if}
