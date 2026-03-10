<script lang="ts">
  import type { SubSessionInfo } from '../stores/types';
  import { sessionStore } from '../stores/session.svelte';
  import { graphStore } from '../stores/graph.svelte';

  let subSessions = $state<SubSessionInfo[]>([]);
  let collapsed = $state(true);
  let activeSubSessionId = $state<string | null>(null);
  let loading = $state(false);

  async function fetchSubSessions() {
    const session = sessionStore.currentSession;
    if (!session) return;
    try {
      const res = await fetch(`/api/sessions/${session.id}/sub-sessions`);
      if (res.ok) {
        subSessions = await res.json();
      }
    } catch (e) {
      console.error('Failed to fetch sub-sessions:', e);
    }
  }

  function activateSubSession(sub: SubSessionInfo) {
    activeSubSessionId = sub.id;
    graphStore.setGrayedNodes(sub.grayed_nodes);
  }

  function activateRoot() {
    activeSubSessionId = null;
    graphStore.clearGrayedNodes();
  }

  export function addSubSession(sub: SubSessionInfo) {
    subSessions = [...subSessions, sub];
    activateSubSession(sub);
    collapsed = false;
  }

  $effect(() => {
    if (sessionStore.currentSession) {
      fetchSubSessions();
    }
  });
</script>

{#if subSessions.length > 0}
  <div class="absolute top-2 left-[340px] z-20">
    <div class="bg-gray-800/95 backdrop-blur border border-gray-700 rounded-lg shadow-xl">
      <button
        class="flex items-center gap-2 px-3 py-2 text-xs text-gray-300 hover:text-gray-100 w-full"
        onclick={() => collapsed = !collapsed}
      >
        <span class="text-gray-500">{collapsed ? '>' : 'v'}</span>
        <span>Sub-sessions ({subSessions.length})</span>
      </button>

      {#if !collapsed}
        <div class="border-t border-gray-700 max-h-64 overflow-y-auto">
          <!-- Root session -->
          <button
            class="w-full text-left px-3 py-1.5 text-xs hover:bg-gray-700 flex items-center gap-2"
            class:bg-gray-700={activeSubSessionId === null}
            onclick={activateRoot}
          >
            <span class="w-1.5 h-1.5 rounded-full bg-blue-400"></span>
            <span>Full Model (root)</span>
          </button>

          {#each subSessions as sub (sub.id)}
            <button
              class="w-full text-left px-3 py-1.5 text-xs hover:bg-gray-700 flex items-center gap-2"
              class:bg-gray-700={activeSubSessionId === sub.id}
              onclick={() => activateSubSession(sub)}
            >
              <span
                class="w-1.5 h-1.5 rounded-full"
                class:bg-amber-400={sub.cut_type === 'output'}
                class:bg-purple-400={sub.cut_type === 'input'}
              ></span>
              <span class="flex-1 truncate">
                {sub.cut_type === 'output' ? 'Output' : 'Input'} @ {sub.cut_node}
              </span>
              <span class="text-gray-500">{sub.task_count}</span>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </div>
{/if}
