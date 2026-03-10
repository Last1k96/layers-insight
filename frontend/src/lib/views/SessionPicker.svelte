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

  onMount(() => {
    sessionStore.fetchSessions();
  });
</script>

<div class="flex-1 flex items-center justify-center p-8">
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
      <div class="space-y-3 mb-6">
        {#each sessionStore.sessions as session (session.id)}
          <button
            class="w-full text-left p-4 bg-gray-800 hover:bg-gray-750 rounded-lg border border-gray-700 hover:border-gray-600 transition-colors"
            onclick={() => onsessionselected(session.id)}
          >
            <div class="flex justify-between items-start">
              <div>
                <div class="font-medium">{session.model_name}</div>
                <div class="text-sm text-gray-400 mt-1">
                  {session.main_device} vs {session.ref_device}
                </div>
              </div>
              <div class="text-right">
                <div class="text-sm text-gray-400">
                  {session.success_count}/{session.task_count} tasks
                </div>
                <div class="text-xs text-gray-500 mt-1">
                  {new Date(session.created_at).toLocaleDateString()}
                </div>
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
