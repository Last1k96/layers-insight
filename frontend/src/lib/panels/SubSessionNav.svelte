<script lang="ts">
  import type { SubSessionInfo } from '../stores/types';
  import { sessionStore } from '../stores/session.svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { queueStore } from '../stores/queue.svelte';
  import { refreshRenderer } from '../graph/renderer';

  interface TreeNode {
    sub: SubSessionInfo;
    children: TreeNode[];
    depth: number;
  }

  let subSessions = $state<SubSessionInfo[]>([]);
  let collapsed = $state(true);
  let activeSubSessionId = $state<string | null>(null);
  let loading = $state(false);

  let tree = $derived.by(() => buildTree(subSessions));

  function buildTree(subs: SubSessionInfo[]): TreeNode[] {
    const byId = new Map(subs.map(s => [s.id, s]));
    const childrenMap = new Map<string, TreeNode[]>();

    // Group subs by parent_id
    for (const sub of subs) {
      const parentKey = sub.parent_id;
      if (!childrenMap.has(parentKey)) childrenMap.set(parentKey, []);
      childrenMap.get(parentKey)!.push({ sub, children: [], depth: 0 });
    }

    // Build tree recursively from roots (parent_id = session id, i.e. not in byId)
    function attachChildren(node: TreeNode, depth: number): void {
      node.depth = depth;
      const kids = childrenMap.get(node.sub.id) || [];
      node.children = kids;
      for (const kid of kids) {
        attachChildren(kid, depth + 1);
      }
    }

    // Root nodes: parent_id is not a known sub-session id
    const roots: TreeNode[] = [];
    for (const sub of subs) {
      if (!byId.has(sub.parent_id)) {
        const node: TreeNode = { sub, children: [], depth: 0 };
        attachChildren(node, 0);
        roots.push(node);
      }
    }
    return roots;
  }

  function flattenTree(nodes: TreeNode[]): TreeNode[] {
    const result: TreeNode[] = [];
    function walk(nodes: TreeNode[]) {
      for (const node of nodes) {
        result.push(node);
        walk(node.children);
      }
    }
    walk(nodes);
    return result;
  }

  let flatTree = $derived(flattenTree(tree));

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
    graphStore.setGrayedNodes(sub.grayed_nodes, sub.cut_node, sub.cut_type, sub.ancestor_cuts);
    graphStore.setActiveSubSession(sub.id);
    queueStore.removeByNodeNames(new Set(sub.grayed_nodes));
    refreshRenderer();
  }

  function activateRoot() {
    activeSubSessionId = null;
    graphStore.clearGrayedNodes();
    graphStore.setActiveSubSession(null);
    refreshRenderer();
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
    <div class="bg-[--bg-panel] backdrop-blur border border-[--border-color] rounded-lg shadow-xl">
      <button
        class="flex items-center gap-2 px-3 py-2 text-xs text-gray-300 hover:text-gray-100 w-full"
        onclick={() => collapsed = !collapsed}
      >
        <span class="text-gray-500">{collapsed ? '>' : 'v'}</span>
        <span>Sub-sessions ({subSessions.length})</span>
      </button>

      {#if !collapsed}
        <div class="border-t border-[--border-color] max-h-64 overflow-y-auto">
          <!-- Root session -->
          <button
            class="w-full text-left px-3 py-1.5 text-xs hover:bg-[--bg-menu] flex items-center gap-2"
            class:bg-[--bg-menu]={activeSubSessionId === null}
            onclick={activateRoot}
          >
            <span class="w-1.5 h-1.5 rounded-full bg-accent"></span>
            <span>Full Model (root)</span>
          </button>

          {#each flatTree as node (node.sub.id)}
            <button
              class="w-full text-left py-1.5 text-xs hover:bg-[--bg-menu] flex items-center gap-2"
              class:bg-[--bg-menu]={activeSubSessionId === node.sub.id}
              style="padding-left: {12 + node.depth * 12}px; padding-right: 12px;"
              onclick={() => activateSubSession(node.sub)}
            >
              <span
                class="w-1.5 h-1.5 rounded-full shrink-0"
                class:bg-status-waiting={node.sub.cut_type === 'output'}
                class:bg-accent={node.sub.cut_type === 'input'}
              ></span>
              <span class="flex-1 truncate">
                {node.sub.cut_type === 'output' ? 'Output' : 'Input'} @ {node.sub.cut_node}
              </span>
              <span class="text-gray-500">{node.sub.task_count}</span>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </div>
{/if}
