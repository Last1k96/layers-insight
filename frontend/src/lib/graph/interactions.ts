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

  // Click/dblclick timer to distinguish single from double click
  let clickTimer: ReturnType<typeof setTimeout> | null = null;

  function hitTestNode(e: MouseEvent): string | null {
    const rect = canvas.getBoundingClientRect();
    const gp = camera!.viewportToGraph(e.clientX - rect.left, e.clientY - rect.top);
    const nodeId = gpu!.hitGrid.query(gp.x, gp.y);
    return (nodeId && graph!.hasNode(nodeId)) ? nodeId : null;
  }

  // Single click: select node, show info. Ctrl+click also centers.
  function handleClick(e: MouseEvent) {
    if (camera!.didDrag) return;

    const nodeId = hitTestNode(e);

    if (!nodeId) {
      // Clicked on empty space — deselect immediately (no timer needed)
      if (clickTimer) { clearTimeout(clickTimer); clickTimer = null; }
      graphStore.selectNode(null);
      refreshRenderer();
      return;
    }

    // Delay single-click action so dblclick can cancel it
    const ctrlKey = e.ctrlKey;
    if (clickTimer) { clearTimeout(clickTimer); clickTimer = null; }
    clickTimer = setTimeout(() => {
      clickTimer = null;
      graphStore.selectNode(nodeId);
      if (ctrlKey) centerOnNode(nodeId);
      refreshRenderer();
    }, 250);
  }

  // Double click: select node + enqueue inference. Shift+dblclick re-enqueues.
  function handleDblClick(e: MouseEvent) {
    if (clickTimer) { clearTimeout(clickTimer); clickTimer = null; }

    const nodeId = hitTestNode(e);
    if (!nodeId) return;

    const sessionId = sessionStore.currentSession?.id;
    if (!sessionId) return;

    const attrs = graph!.getNodeAttributes(nodeId);
    const nodeStatus = graphStore.nodeStatusMap.get(nodeId);

    graphStore.selectNode(nodeId);
    if (e.ctrlKey) centerOnNode(nodeId);
    refreshRenderer();

    if (!graphStore.grayedNodes.has(nodeId) && (e.shiftKey || !nodeStatus)) {
      queueStore.enqueue(sessionId, nodeId, attrs.nodeName || attrs.label, attrs.opType, graphStore.activeSubSessionId);
    }
  }

  canvas.addEventListener('click', handleClick as EventListener);
  canvas.addEventListener('dblclick', handleDblClick as EventListener);
  cleanupFns.push(() => {
    canvas.removeEventListener('click', handleClick as EventListener);
    canvas.removeEventListener('dblclick', handleDblClick as EventListener);
    if (clickTimer) { clearTimeout(clickTimer); clickTimer = null; }
  });

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

    // Plain arrow up/down: navigate queue list
    if (!e.ctrlKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
      e.preventDefault();
      const task = queueStore.moveSelection(e.key === 'ArrowDown' ? 1 : -1);
      if (task) {
        graphStore.selectNode(task.node_id);
        centerOnNode(task.node_id);
        refreshRenderer();
      }
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

  // Alt hold: accuracy view mode
  function handleKeydownAlt(e: KeyboardEvent) {
    if (e.key === 'Alt' && !graphStore.accuracyViewActive) {
      e.preventDefault();
      graphStore.accuracyViewActive = true;
      refreshRenderer();
    }
  }

  function handleKeyupAlt(e: KeyboardEvent) {
    if (e.key === 'Alt' && graphStore.accuracyViewActive) {
      graphStore.accuracyViewActive = false;
      refreshRenderer();
    }
  }

  function handleBlur() {
    if (graphStore.accuracyViewActive) {
      graphStore.accuracyViewActive = false;
      refreshRenderer();
    }
  }

  document.addEventListener('keydown', handleKeydownAlt);
  document.addEventListener('keyup', handleKeyupAlt);
  window.addEventListener('blur', handleBlur);
  cleanupFns.push(() => {
    document.removeEventListener('keydown', handleKeydownAlt);
    document.removeEventListener('keyup', handleKeyupAlt);
    window.removeEventListener('blur', handleBlur);
  });
}

export function cleanupInteractions(): void {
  for (const fn of cleanupFns) fn();
  cleanupFns = [];
}
