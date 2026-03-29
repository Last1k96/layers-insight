<script lang="ts">
  import { sessionStore } from '../stores/session.svelte';
  import { onMount } from 'svelte';

  let {
    onsessionselected,
    onnewsession,
    onclonesession,
    oncompare,
  }: {
    onsessionselected: (id: string) => void;
    onnewsession: () => void;
    onclonesession: (id: string) => void;
    oncompare: (a: string, b: string) => void;
  } = $props();

  let confirmingDelete: string | null = $state(null);
  let selectedIndex = $state(-1);
  let compareMode = $state(false);
  let compareSelection = $state<string[]>([]);

  function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  function handleKeydown(e: KeyboardEvent) {
    const len = sessionStore.sessions.length;
    if (len === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, len - 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, 0);
    } else if (e.key === 'Enter' && selectedIndex >= 0 && selectedIndex < len) {
      e.preventDefault();
      onsessionselected(sessionStore.sessions[selectedIndex].id);
    }
  }

  function handleDelete(e: MouseEvent, sessionId: string) {
    e.stopPropagation();
    if (confirmingDelete === sessionId) {
      sessionStore.deleteSession(sessionId);
      confirmingDelete = null;
    } else {
      confirmingDelete = sessionId;
    }
  }

  function cancelDelete(e: MouseEvent) {
    e.stopPropagation();
    confirmingDelete = null;
  }

  function handleClone(e: MouseEvent, sessionId: string) {
    e.stopPropagation();
    onclonesession(sessionId);
  }

  function toggleCompareSelect(e: MouseEvent, sessionId: string) {
    e.stopPropagation();
    if (compareSelection.includes(sessionId)) {
      compareSelection = compareSelection.filter(id => id !== sessionId);
    } else if (compareSelection.length < 2) {
      compareSelection = [...compareSelection, sessionId];
    }
  }

  function startCompare() {
    if (compareSelection.length === 2) {
      oncompare(compareSelection[0], compareSelection[1]);
    }
  }

  function toggleCompareMode() {
    compareMode = !compareMode;
    if (!compareMode) compareSelection = [];
  }

  onMount(() => {
    sessionStore.fetchSessions().then(() => {
      const sessions = sessionStore.sessions;
      if (sessions.length === 0) return;
      const lastId = sessionStore.lastSessionId;
      const idx = lastId ? sessions.findIndex(s => s.id === lastId) : -1;
      selectedIndex = idx >= 0 ? idx : 0;
    });
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });
</script>

<div class="flex-1 flex items-start justify-center p-8 pt-[15vh] bg-[--bg-primary]">
  <div class="max-w-2xl w-full">
    <h1 class="text-3xl font-bold mb-2">Layers Insight</h1>
    <p class="text-content-secondary mb-8">Neural Network Graph Debugger</p>

    {#if sessionStore.loading}
      <div class="text-content-secondary">Loading sessions...</div>
    {:else if sessionStore.sessions.length === 0}
      <div class="py-12">
        <p class="text-content-secondary mb-4 text-center">No sessions found</p>
        <button
          class="w-full py-3 border border-dashed border-edge hover:border-content-secondary/50 rounded-lg text-content-secondary hover:text-content-primary transition-colors"
          onclick={onnewsession}
        >
          + New Session
        </button>
      </div>
    {:else}
      <div class="space-y-3 mb-6 max-h-[60vh] overflow-y-auto pr-1 scrollbar-thin">
        {#each sessionStore.sessions as session, i (session.id)}
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          <div
            class="group w-full text-left p-4 bg-surface-panel hover:bg-surface-elevated rounded-lg border transition-colors cursor-pointer {i === selectedIndex ? 'border-accent bg-surface-elevated' : 'border-edge hover:border-content-secondary/30'} {compareMode && compareSelection.includes(session.id) ? 'ring-2 ring-accent' : ''}"
            role="button"
            tabindex="0"
            onclick={() => compareMode ? toggleCompareSelect(new MouseEvent('click'), session.id) : onsessionselected(session.id)}
            onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); compareMode ? toggleCompareSelect(new MouseEvent('click'), session.id) : onsessionselected(session.id); }}}
          >
            <div class="flex justify-between items-start">
              <div class="flex items-start gap-3 min-w-0 flex-1">
                {#if compareMode}
                  <div class="mt-1 flex-shrink-0">
                    <div class="w-4 h-4 border-2 rounded {compareSelection.includes(session.id) ? 'border-accent bg-accent' : 'border-content-secondary/50'} transition-colors flex items-center justify-center">
                      {#if compareSelection.includes(session.id)}
                        <svg xmlns="http://www.w3.org/2000/svg" class="w-3 h-3 text-white" viewBox="0 0 20 20" fill="currentColor">
                          <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                        </svg>
                      {/if}
                    </div>
                  </div>
                {/if}
                <div class="min-w-0">
                  <div class="font-medium">{session.model_name}</div>
                  <div class="text-sm text-content-secondary mt-1">
                    {session.main_device} vs {session.ref_device}
                  </div>
                </div>
              </div>
              <div class="text-right flex-shrink-0 flex items-start gap-2">
                <div>
                  <div class="text-sm text-content-secondary">
                    {session.success_count}/{session.task_count} tasks
                  </div>
                  {#if session.sub_sessions?.length > 0}
                    <div class="text-xs text-content-secondary/60">
                      {session.sub_sessions.length} sub-session{session.sub_sessions.length > 1 ? 's' : ''}
                    </div>
                  {/if}
                  <div class="text-xs text-content-secondary/60 mt-1">
                    {new Date(session.created_at).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}
                    {#if session.folder_size > 0}
                      <span class="ml-2">{formatSize(session.folder_size)}</span>
                    {/if}
                  </div>
                </div>
                {#if !compareMode}
                  {#if confirmingDelete === session.id}
                    <div class="flex gap-1">
                      <button
                        class="px-2 py-1 text-xs bg-surface-elevated hover:bg-edge text-content-primary rounded transition-colors"
                        onclick={cancelDelete}
                      >
                        Cancel
                      </button>
                      <button
                        class="px-2 py-1 text-xs bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
                        onclick={(e) => handleDelete(e, session.id)}
                      >
                        Confirm
                      </button>
                    </div>
                  {:else}
                    <div class="flex gap-1">
                      <button
                        class="opacity-0 group-hover:opacity-100 p-1 text-content-secondary hover:text-accent transition-all"
                        onclick={(e) => handleClone(e, session.id)}
                        title="Clone session"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M7 9a2 2 0 012-2h6a2 2 0 012 2v6a2 2 0 01-2 2H9a2 2 0 01-2-2V9z" />
                          <path d="M5 3a2 2 0 00-2 2v6a2 2 0 002 2V5h8a2 2 0 00-2-2H5z" />
                        </svg>
                      </button>
                      <button
                        class="opacity-0 group-hover:opacity-100 p-1 text-content-secondary hover:text-red-400 transition-all"
                        onclick={(e) => handleDelete(e, session.id)}
                        title="Delete session"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                          <path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd" />
                        </svg>
                      </button>
                    </div>
                  {/if}
                {/if}
              </div>
            </div>
          </div>
        {/each}
      </div>

      <div class="flex gap-2">
        <button
          class="flex-1 py-3 border border-dashed border-edge hover:border-content-secondary/50 rounded-lg text-content-secondary hover:text-content-primary transition-colors"
          onclick={onnewsession}
        >
          + New Session
        </button>
        {#if sessionStore.sessions.length >= 2}
          <button
            class="py-3 px-4 border rounded-lg transition-colors {compareMode ? 'border-accent text-accent bg-accent/10' : 'border-edge text-content-secondary hover:border-content-secondary/50 hover:text-content-primary'}"
            onclick={toggleCompareMode}
          >
            {compareMode ? 'Cancel' : 'Compare'}
          </button>
        {/if}
      </div>
      {#if compareMode && compareSelection.length === 2}
        <button
          class="w-full py-3 mt-2 bg-accent hover:bg-accent-hover rounded-lg font-medium transition-colors"
          onclick={startCompare}
        >
          Compare Selected Sessions
        </button>
      {:else if compareMode}
        <div class="text-center text-sm text-content-secondary mt-2">
          Select 2 sessions to compare ({compareSelection.length}/2)
        </div>
      {/if}
    {/if}

    {#if sessionStore.error}
      <div class="mt-4 p-3 bg-red-900/50 border border-red-700 rounded text-red-300 text-sm">
        {sessionStore.error}
      </div>
    {/if}
  </div>
</div>

<style>
  .scrollbar-thin::-webkit-scrollbar {
    width: 6px;
  }
  .scrollbar-thin::-webkit-scrollbar-track {
    background: transparent;
  }
  .scrollbar-thin::-webkit-scrollbar-thumb {
    background: #3A3F56;
    border-radius: 3px;
  }
  .scrollbar-thin::-webkit-scrollbar-thumb:hover {
    background: #4A5070;
  }
</style>
