/**
 * Graph interactions — click, hover, keyboard navigation.
 * Uses WebGPU hit testing for node detection.
 */
import { getGraph, getGPURenderer, getCamera, centerOnNode, refreshRenderer, setHoveredNode } from './renderer';
import { graphStore } from '../stores/graph.svelte';
import { queueStore } from '../stores/queue.svelte';
import { sessionStore } from '../stores/session.svelte';

let cleanupFns: (() => void)[] = [];

export function setupInteractions(): void {
  cleanupInteractions();

  const graph = getGraph();
  const camera = getCamera();
  const gpu = getGPURenderer();
  if (!graph || !camera || !gpu) return;

  const canvas = gpu.canvas;

  // Click handler
  function handleClick(e: MouseEvent) {
    if (camera!.didDrag) return;

    const rect = canvas.getBoundingClientRect();
    const gp = camera!.viewportToGraph(e.clientX - rect.left, e.clientY - rect.top);
    const nodeId = gpu!.hitGrid.query(gp.x, gp.y);

    if (!nodeId || !graph!.hasNode(nodeId)) {
      graphStore.selectNode(null);
      refreshRenderer();
      return;
    }

    const sessionId = sessionStore.currentSession?.id;
    if (!sessionId) return;

    const attrs = graph!.getNodeAttributes(nodeId);
    const nodeStatus = graphStore.nodeStatusMap.get(nodeId);

    graphStore.selectNode(nodeId);
    centerOnNode(nodeId);
    refreshRenderer();

    if (e.shiftKey) {
      queueStore.enqueue(sessionId, nodeId, attrs.nodeName || attrs.label, attrs.opType);
    } else if (!nodeStatus) {
      queueStore.enqueue(sessionId, nodeId, attrs.nodeName || attrs.label, attrs.opType);
    }
  }

  canvas.addEventListener('click', handleClick as EventListener);
  cleanupFns.push(() => canvas.removeEventListener('click', handleClick as EventListener));

  // Hover detection
  let hoverThrottleId: number | null = null;

  function handleMouseMove(e: MouseEvent) {
    if (hoverThrottleId !== null) return;
    hoverThrottleId = requestAnimationFrame(() => {
      hoverThrottleId = null;
      const rect = canvas.getBoundingClientRect();
      const gp = camera!.viewportToGraph(e.clientX - rect.left, e.clientY - rect.top);
      const hitId = gpu!.hitGrid.query(gp.x, gp.y);
      setHoveredNode(hitId);
      canvas.style.cursor = hitId ? 'pointer' : '';
    });
  }

  canvas.addEventListener('mousemove', handleMouseMove);
  cleanupFns.push(() => {
    canvas.removeEventListener('mousemove', handleMouseMove);
    if (hoverThrottleId !== null) cancelAnimationFrame(hoverThrottleId);
  });

  // Keyboard navigation
  function handleKeydown(e: KeyboardEvent) {
    if (e.ctrlKey && e.key === 'f') {
      e.preventDefault();
      graphStore.searchVisible = !graphStore.searchVisible;
      return;
    }

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
