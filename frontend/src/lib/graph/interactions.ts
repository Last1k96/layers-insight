import { getSigma, getGraph, centerOnNode, refreshRenderer } from './renderer';
import { graphStore } from '../stores/graph.svelte';
import { queueStore } from '../stores/queue.svelte';
import { sessionStore } from '../stores/session.svelte';

let cleanupFns: (() => void)[] = [];

export function setupInteractions(): void {
  cleanupInteractions();

  const sigma = getSigma();
  const graph = getGraph();
  if (!sigma || !graph) return;

  // Click node
  sigma.on('clickNode', ({ node, event }) => {
    const sessionId = sessionStore.currentSession?.id;
    if (!sessionId) return;

    const attrs = graph.getNodeAttributes(node);
    const nodeStatus = graphStore.nodeStatusMap.get(node);

    graphStore.selectNode(node);
    centerOnNode(node);

    if (event.original.shiftKey) {
      // Shift+click: always enqueue
      queueStore.enqueue(sessionId, node, (attrs.nodeName || attrs.label) as string, attrs.opType as string);
    } else if (!nodeStatus) {
      // Click un-inferred: auto-queue
      queueStore.enqueue(sessionId, node, (attrs.nodeName || attrs.label) as string, attrs.opType as string);
    }
    // Click inferred/queued: just select (already done above)
  });

  // Click background: deselect
  sigma.on('clickStage', () => {
    graphStore.selectNode(null);
  });

  // Keyboard navigation
  function handleKeydown(e: KeyboardEvent) {
    // Ctrl+F: toggle search
    if (e.ctrlKey && e.key === 'f') {
      e.preventDefault();
      graphStore.searchVisible = !graphStore.searchVisible;
      return;
    }

    // Ctrl+Arrow: navigate graph edges
    if (e.ctrlKey && graph && graphStore.selectedNodeId) {
      const nodeId = graphStore.selectedNodeId;
      let targetId: string | null = null;

      if (e.key === 'ArrowUp') {
        // Move toward inputs (predecessors)
        e.preventDefault();
        const predecessors = graph.inNeighbors(nodeId);
        if (predecessors.length > 0) targetId = predecessors[0];
      } else if (e.key === 'ArrowDown') {
        // Move toward outputs (successors)
        e.preventDefault();
        const successors = graph.outNeighbors(nodeId);
        if (successors.length > 0) targetId = successors[0];
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        // Siblings: other children of the same parent
        e.preventDefault();
        const parents = graph.inNeighbors(nodeId);
        if (parents.length > 0) {
          const siblings = graph.outNeighbors(parents[0]).filter(n => n !== nodeId);
          if (siblings.length > 0) {
            const currentIdx = graph.outNeighbors(parents[0]).indexOf(nodeId);
            const dir = e.key === 'ArrowRight' ? 1 : -1;
            const allSiblings = graph.outNeighbors(parents[0]);
            const newIdx = (currentIdx + dir + allSiblings.length) % allSiblings.length;
            targetId = allSiblings[newIdx];
          }
        }
      }

      if (targetId) {
        graphStore.selectNode(targetId);
        centerOnNode(targetId);
        refreshRenderer();
      }
    }
  }

  document.addEventListener('keydown', handleKeydown);
  cleanupFns.push(() => document.removeEventListener('keydown', handleKeydown));
}

export function cleanupInteractions(): void {
  for (const fn of cleanupFns) fn();
  cleanupFns = [];
}
