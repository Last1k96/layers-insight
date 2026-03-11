/**
 * SVG DOM-based interactions — click, keyboard navigation.
 */
import { getGraph, getSVGState, centerOnNode, refreshRenderer } from './renderer';
import { graphStore } from '../stores/graph.svelte';
import { queueStore } from '../stores/queue.svelte';
import { sessionStore } from '../stores/session.svelte';

let cleanupFns: (() => void)[] = [];

export function setupInteractions(): void {
  cleanupInteractions();

  const state = getSVGState();
  const graph = getGraph();
  if (!state || !graph) return;

  // Click on nodes — delegated event on nodes group
  function handleNodeClick(e: MouseEvent) {
    const target = e.target as Element;
    const nodeEl = target.closest('.node') as SVGGElement | null;

    if (!nodeEl) {
      // Click on background → deselect
      graphStore.selectNode(null);
      refreshRenderer();
      return;
    }

    const nodeId = nodeEl.dataset.id;
    if (!nodeId || !graph!.hasNode(nodeId)) return;

    const sessionId = sessionStore.currentSession?.id;
    if (!sessionId) return;

    const attrs = graph!.getNodeAttributes(nodeId);
    const nodeStatus = graphStore.nodeStatusMap.get(nodeId);

    graphStore.selectNode(nodeId);
    centerOnNode(nodeId);
    refreshRenderer();

    if (e.shiftKey) {
      // Shift+click: always enqueue
      queueStore.enqueue(sessionId, nodeId, attrs.nodeName || attrs.label, attrs.opType);
    } else if (!nodeStatus) {
      // Click un-inferred: auto-queue
      queueStore.enqueue(sessionId, nodeId, attrs.nodeName || attrs.label, attrs.opType);
    }
  }

  state.svg.addEventListener('click', handleNodeClick);
  cleanupFns.push(() => state.svg.removeEventListener('click', handleNodeClick));

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
        e.preventDefault();
        const predecessors = graph.inNeighbors(nodeId);
        if (predecessors.length > 0) targetId = predecessors[0];
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        const successors = graph.outNeighbors(nodeId);
        if (successors.length > 0) targetId = successors[0];
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        const parents = graph.inNeighbors(nodeId);
        if (parents.length > 0) {
          const allSiblings = graph.outNeighbors(parents[0]);
          const currentIdx = allSiblings.indexOf(nodeId);
          const dir = e.key === 'ArrowRight' ? 1 : -1;
          const newIdx = (currentIdx + dir + allSiblings.length) % allSiblings.length;
          targetId = allSiblings[newIdx];
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
