<script lang="ts">
  import { sessionStore } from '../stores/session.svelte';
  import { onMount } from 'svelte';

  let {
    onsessionselected,
    onnewsession,
  }: {
    onsessionselected: (id: string) => void;
    onnewsession: () => void;
  } = $props();

  let confirmingDelete: string | null = $state(null);
  let selectedIndex = $state(-1);

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

<div class="flex-1 flex items-start justify-center p-8 pt-[15vh]">
  <div class="max-w-2xl w-full">
    <h1 class="text-3xl font-bold mb-2">Layers Insight</h1>
    <p class="text-gray-400 mb-8">Neural Network Graph Debugger</p>

    {#if sessionStore.loading}
      <div class="text-gray-400">Loading sessions...</div>
    {:else if sessionStore.sessions.length === 0}
      <div class="text-center py-12">
        <p class="text-gray-400 mb-4">No sessions found</p>
        <button
          class="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors"
          onclick={onnewsession}
        >
          New Session
        </button>
      </div>
    {:else}
      <div class="space-y-3 mb-6 max-h-[60vh] overflow-y-auto pr-1 scrollbar-thin">
        {#each sessionStore.sessions as session, i (session.id)}
          <button
            class="group w-full text-left p-4 bg-[--bg-panel] hover:bg-[--bg-menu] rounded-lg border transition-colors {i === selectedIndex ? 'border-blue-500 bg-[--bg-menu]' : 'border-[--border-color] hover:border-gray-600'}"
            onclick={() => onsessionselected(session.id)}
          >
            <div class="flex justify-between items-start">
              <div class="min-w-0 flex-1">
                <div class="font-medium">{session.model_name}</div>
                <div class="text-sm text-gray-400 mt-1">
                  {session.main_device} vs {session.ref_device}
                </div>
              </div>
              <div class="text-right flex-shrink-0 flex items-start gap-2">
                <div>
                  <div class="text-sm text-gray-400">
                    {session.success_count}/{session.task_count} tasks
                  </div>
                  <div class="text-xs text-gray-500 mt-1">
                    {new Date(session.created_at).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}
                  </div>
                </div>
                {#if confirmingDelete === session.id}
                  <div class="flex gap-1">
                    <button
                      class="px-2 py-1 text-xs bg-gray-600 hover:bg-gray-500 text-gray-200 rounded transition-colors"
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
                  <button
                    class="opacity-0 group-hover:opacity-100 p-1 text-gray-500 hover:text-red-400 transition-all"
                    onclick={(e) => handleDelete(e, session.id)}
                    title="Delete session"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                      <path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd" />
                    </svg>
                  </button>
                {/if}
              </div>
            </div>
          </button>
        {/each}
      </div>

      <button
        class="w-full py-3 border border-dashed border-gray-600 hover:border-gray-500 rounded-lg text-gray-400 hover:text-gray-300 transition-colors"
        onclick={onnewsession}
      >
        + New Session
      </button>
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
    background: #4b5563;
    border-radius: 3px;
  }
  .scrollbar-thin::-webkit-scrollbar-thumb:hover {
    background: #6b7280;
  }
</style>
