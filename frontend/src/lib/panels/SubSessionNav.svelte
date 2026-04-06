<script lang="ts">
  import type { SubSessionInfo } from '../stores/types';
  import { sessionStore } from '../stores/session.svelte';
  import { graphStore } from '../stores/graph.svelte';

  import { refreshRenderer, fitToSubSession } from '../graph/renderer';

  interface TreeNode {
    sub: SubSessionInfo;
    children: TreeNode[];
    depth: number;
  }

  let subSessions = $state<SubSessionInfo[]>([]);
  let activeSubSessionId = $derived(graphStore.activeSubSessionId);
  let leftOffset = $state(330);
  let collapsed = $state(true);

  // Track left panel's right edge to position ourselves flush against it
  $effect(() => {
    const panel = document.querySelector('[data-panel-side="left"]') as HTMLElement | null;
    if (!panel) return;

    function sync() {
      leftOffset = panel!.offsetLeft + panel!.offsetWidth + 4;
    }
    sync();

    const ro = new ResizeObserver(sync);
    ro.observe(panel);
    return () => ro.disconnect();
  });

  let tree = $derived.by(() => buildTree(subSessions));

  function buildTree(subs: SubSessionInfo[]): TreeNode[] {
    const byId = new Map(subs.map(s => [s.id, s]));
    const childrenMap = new Map<string, TreeNode[]>();

    for (const sub of subs) {
      const parentKey = sub.parent_id;
      if (!childrenMap.has(parentKey)) childrenMap.set(parentKey, []);
      childrenMap.get(parentKey)!.push({ sub, children: [], depth: 0 });
    }

    function attachChildren(node: TreeNode, depth: number): void {
      node.depth = depth;
      const kids = childrenMap.get(node.sub.id) || [];
      node.children = kids;
      for (const kid of kids) attachChildren(kid, depth + 1);
    }

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
    graphStore.setGrayedNodes(sub.grayed_nodes, sub.cut_node, sub.cut_type, sub.ancestor_cuts);
    graphStore.setActiveSubSession(sub.id);
    refreshRenderer();
    // Animate camera to fit the non-grayed nodes
    requestAnimationFrame(() => fitToSubSession());
  }

  function activateRoot() {
    graphStore.clearGrayedNodes();
    graphStore.setActiveSubSession(null);
    refreshRenderer();
    // Animate camera to fit the entire graph
    requestAnimationFrame(() => fitToSubSession());
  }

  async function deleteSubSession(e: MouseEvent, subId: string) {
    e.stopPropagation();
    const session = sessionStore.currentSession;
    if (!session) return;

    try {
      const res = await fetch(`/api/sessions/${session.id}/sub-sessions/${subId}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        // If we deleted the active sub-session (or its ancestor), go to parent
        const deletedIds = getSubTreeIds(subId);
        if (activeSubSessionId && deletedIds.has(activeSubSessionId)) {
          const deleted = subSessions.find(s => s.id === subId);
          const parent = deleted ? subSessions.find(s => s.id === deleted.parent_id) : null;
          if (parent) {
            activateSubSession(parent);
          } else {
            activateRoot();
          }
        }
        subSessions = subSessions.filter(s => !deletedIds.has(s.id));
      }
    } catch (e) {
      console.error('Failed to delete sub-session:', e);
    }
  }

  function getSubTreeIds(rootId: string): Set<string> {
    const ids = new Set([rootId]);
    let changed = true;
    while (changed) {
      changed = false;
      for (const s of subSessions) {
        if (!ids.has(s.id) && ids.has(s.parent_id)) {
          ids.add(s.id);
          changed = true;
        }
      }
    }
    return ids;
  }

  export function addSubSession(sub: SubSessionInfo) {
    subSessions = [...subSessions, sub];
    activateSubSession(sub);
  }

  $effect(() => {
    // Re-fetch when session changes or when a sub-session is created/deleted
    const _session = sessionStore.currentSession;
    const _version = graphStore.subSessionVersion;
    if (_session) {
      fetchSubSessions();
    }
  });
</script>

{#if subSessions.length > 0}
  <div class="sub-session-panel" style="left: {leftOffset}px">
    <button class="panel-header" onclick={() => collapsed = !collapsed}>
      <svg
        width="10" height="10" viewBox="0 0 10 10" fill="none"
        class="chevron shrink-0 opacity-60"
        class:chevron-collapsed={collapsed}
      >
        <path d="M3 2l4 3-4 3" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" class="shrink-0 opacity-60">
        <path d="M2 3h12M2 8h8M2 13h10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
      <span class="panel-title">Sessions</span>
      <span class="count-badge">{subSessions.length}</span>
    </button>

    {#if !collapsed}
      <div class="tree-list">
        <!-- Root session -->
        <button
          class="tree-item"
          class:active={activeSubSessionId === null}
          onclick={activateRoot}
        >
          <span class="tree-indent" style="width: 0px"></span>
          <span class="dot dot-root"></span>
          <span class="tree-label">Full Model</span>
        </button>

        {#each flatTree as node (node.sub.id)}
          <div
            class="tree-item"
            class:active={activeSubSessionId === node.sub.id}
            onclick={() => activateSubSession(node.sub)}
            onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activateSubSession(node.sub); }}}
            role="button"
            tabindex="0"
          >
            <span class="tree-indent" style="width: {node.depth * 14}px">
              {#each Array(node.depth) as _, i}
                <span class="tree-guide" style="left: {i * 14 + 6}px"></span>
              {/each}
            </span>
            <span class="tree-connector"></span>
            <span
              class="dot"
              class:dot-output={node.sub.cut_type === 'output'}
              class:dot-input={node.sub.cut_type === 'input'}
            ></span>
            <span class="tree-label" title="{node.sub.cut_type} @ {node.sub.cut_node}">
              {node.sub.cut_node}
            </span>
            <span class="cut-type-tag" class:tag-output={node.sub.cut_type === 'output'} class:tag-input={node.sub.cut_type === 'input'}>
              {node.sub.cut_type === 'output' ? 'out' : 'in'}
            </span>
            <button
              class="delete-btn"
              onclick={(e) => deleteSubSession(e, node.sub.id)}
              title="Delete sub-session"
            >
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                <path d="M2.5 2.5l5 5M7.5 2.5l-5 5" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
              </svg>
            </button>
          </div>
        {/each}
      </div>
    {/if}
  </div>
{/if}

<style>
  .sub-session-panel {
    position: absolute;
    top: 8px;
    z-index: 20;
    background: var(--bg-panel);
    border-radius: 0 12px 12px 12px;
    box-shadow: var(--shadow-panel);
    min-width: 180px;
    max-width: 260px;
    backdrop-filter: blur(24px);
    overflow: hidden;
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 10px;
    color: var(--text-secondary);
    width: 100%;
    border: none;
    background: var(--bg-panel);
    cursor: pointer;
    transition: background 0.1s;
  }

  .panel-header:hover {
    background: var(--bg-menu);
  }

  .chevron {
    transition: transform 0.15s ease;
    transform: rotate(90deg);
  }

  .chevron-collapsed {
    transform: rotate(0deg);
  }

  .panel-title {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    flex: 1;
  }

  .count-badge {
    font-size: 10px;
    background: var(--bg-menu);
    color: var(--text-secondary);
    opacity: 0.6;
    padding: 1px 7px;
    border-radius: 99px;
    line-height: 1.4;
  }

  .tree-list {
    max-height: 280px;
    overflow-y: auto;
    padding: 4px 0;
  }

  .tree-item {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 4px 8px 4px 10px;
    border: none;
    background: none;
    color: var(--text-primary);
    font-size: 11px;
    text-align: left;
    cursor: pointer;
    position: relative;
    transition: background 0.1s;
  }

  .tree-item:hover {
    background: var(--bg-menu);
  }

  .tree-item.active {
    background: rgba(76, 141, 255, 0.1);
  }

  .tree-item.active .tree-label {
    color: #8bb4ff;
  }

  .tree-indent {
    position: relative;
    display: inline-block;
    flex-shrink: 0;
  }

  .tree-guide {
    position: absolute;
    top: -10px;
    bottom: -10px;
    width: 1px;
    background: var(--border-color);
    opacity: 0.4;
  }

  .tree-connector {
    width: 8px;
    height: 1px;
    background: var(--border-color);
    opacity: 0.4;
    flex-shrink: 0;
  }

  /* Root item has no connector */
  .tree-item:first-child .tree-connector {
    display: none;
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .dot-root {
    background: #4C8DFF;
  }

  .dot-output {
    background: #E5A820;
  }

  .dot-input {
    background: #34C77B;
  }

  .tree-label {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
  }

  .cut-type-tag {
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 99px;
    line-height: 1.4;
    flex-shrink: 0;
    opacity: 0.7;
  }

  .tag-output {
    background: rgba(229, 168, 32, 0.15);
    color: #E5A820;
  }

  .tag-input {
    background: rgba(52, 199, 123, 0.15);
    color: #34C77B;
  }

  .delete-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border: none;
    background: none;
    color: var(--text-secondary);
    border-radius: 6px;
    cursor: pointer;
    opacity: 0;
    transition: opacity 0.1s, background 0.1s, color 0.1s;
    flex-shrink: 0;
    padding: 0;
  }

  .tree-item:hover .delete-btn {
    opacity: 0.6;
  }

  .delete-btn:hover {
    opacity: 1 !important;
    background: rgba(229, 77, 77, 0.15);
    color: #E54D4D;
  }
</style>
