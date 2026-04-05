<script lang="ts">
  import { graphStore } from '../stores/graph.svelte';
  import { getCamera, getGPURenderer, getNodeSize, centerOnNode, refreshRenderer } from './renderer';

  interface GhostData {
    nodeId: string;
    label: string;
    x: number;
    y: number;
    angle: number;
  }

  const MARGIN = 40;

  let ghosts = $derived.by(() => {
    // Depend on cameraVersion so we recompute when the camera moves
    void graphStore.cameraVersion;

    const edgeIdx = graphStore.selectedEdgeIndex;
    if (edgeIdx === null || !graphStore.graphData) return [] as GhostData[];

    const edges = graphStore.graphData.edges;
    if (edgeIdx >= edges.length) return [] as GhostData[];

    const edge = edges[edgeIdx];
    const camera = getCamera();
    const gpu = getGPURenderer();
    if (!camera || !gpu) return [] as GhostData[];

    const canvas = gpu.canvas;
    if (!canvas) return [] as GhostData[];
    const viewW = canvas.clientWidth;
    const viewH = canvas.clientHeight;
    if (viewW === 0 || viewH === 0) return [] as GhostData[];

    const result: GhostData[] = [];
    for (const nodeId of [edge.source, edge.target]) {
      const node = graphStore.graphData.nodes.find(n => n.id === nodeId);
      if (!node) continue;

      const size = getNodeSize(nodeId);
      const cx = node.x + size.width / 2;
      const cy = node.y + size.height / 2;
      const vp = camera.graphToViewport(cx, cy);

      // Check if on-screen
      if (vp.x >= MARGIN && vp.x <= viewW - MARGIN &&
          vp.y >= MARGIN && vp.y <= viewH - MARGIN) continue;

      // Compute ghost position: ray from viewport center to node, clipped to inset rect
      const vcx = viewW / 2;
      const vcy = viewH / 2;
      const dx = vp.x - vcx;
      const dy = vp.y - vcy;

      if (Math.abs(dx) < 0.1 && Math.abs(dy) < 0.1) continue;

      const left = MARGIN, right = viewW - MARGIN;
      const top = MARGIN, bottom = viewH - MARGIN;

      let tMin = Infinity;
      if (dx !== 0) {
        for (const t of [(left - vcx) / dx, (right - vcx) / dx]) {
          if (t > 0) {
            const iy = vcy + dy * t;
            if (iy >= top && iy <= bottom && t < tMin) tMin = t;
          }
        }
      }
      if (dy !== 0) {
        for (const t of [(top - vcy) / dy, (bottom - vcy) / dy]) {
          if (t > 0) {
            const ix = vcx + dx * t;
            if (ix >= left && ix <= right && t < tMin) tMin = t;
          }
        }
      }

      if (!isFinite(tMin)) continue;

      result.push({
        nodeId,
        label: node.type,
        x: vcx + dx * tMin,
        y: vcy + dy * tMin,
        angle: Math.atan2(dy, dx),
      });
    }
    return result;
  });

  function handleGhostClick(nodeId: string, e: MouseEvent) {
    e.stopPropagation();
    graphStore.selectNode(nodeId);
    if (e.ctrlKey) centerOnNode(nodeId);
    refreshRenderer();
  }
</script>

{#each ghosts as ghost (ghost.nodeId)}
  <button
    class="ghost-node"
    style="left: {ghost.x}px; top: {ghost.y}px;"
    onclick={(e) => handleGhostClick(ghost.nodeId, e)}
    title="{ghost.label} — Click to select, Ctrl+Click to focus"
  >
    <svg
      class="ghost-arrow"
      style="transform: rotate({ghost.angle}rad);"
      width="12" height="12" viewBox="0 0 12 12"
    >
      <path d="M0 3 L12 6 L0 9 Z" fill="currentColor" />
    </svg>
    <span class="ghost-label">{ghost.label}</span>
  </button>
{/each}

<style>
  .ghost-node {
    position: absolute;
    transform: translate(-50%, -50%);
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: rgba(27, 30, 43, 0.9);
    border: 1.5px solid #4C8DFF;
    border-radius: 6px;
    color: #c8cce0;
    font-size: 11px;
    font-family: inherit;
    cursor: pointer;
    z-index: 20;
    white-space: nowrap;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
    transition: background 0.15s, border-color 0.15s;
    pointer-events: auto;
  }

  .ghost-node:hover {
    background: rgba(40, 44, 64, 0.95);
    border-color: #6aa3ff;
  }

  .ghost-arrow {
    flex-shrink: 0;
    color: #4C8DFF;
  }

  .ghost-label {
    max-width: 100px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
