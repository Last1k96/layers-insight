/**
 * Graph renderer facade — WebGPU-based graph rendering.
 */
import type { GraphData } from '../stores/types';
import { graphStore } from '../stores/graph.svelte';
import { GraphModel } from './graphModel';
import { PanZoom } from './panZoom';
import { WebGPURenderer } from './webgpu/WebGPURenderer';

let graphModel: GraphModel | null = null;
let panZoom: PanZoom | null = null;
let gpuRenderer: WebGPURenderer | null = null;
let refreshScheduled = false;
let hoveredNodeId: string | null = null;
let hoveredEdgeIndex: number | null = null;
let currentGraphData: GraphData | null = null;

/** Node dimensions cache (id -> {width, height}) */
const nodeSizes = new Map<string, { width: number; height: number }>();

export function getNodeSize(nodeId: string): { width: number; height: number } {
  return nodeSizes.get(nodeId) ?? { width: 100, height: 32 };
}

export function getGraph(): GraphModel | null {
  return graphModel;
}

export function getCamera(): PanZoom | null {
  return panZoom;
}

export function getGPURenderer(): WebGPURenderer | null {
  return gpuRenderer;
}

export function setHoveredNode(nodeId: string | null): void {
  if (hoveredNodeId !== nodeId) {
    hoveredNodeId = nodeId;
    scheduleRefresh();
  }
}

export function getHoveredNode(): string | null {
  return hoveredNodeId;
}

export function setHoveredEdge(index: number | null): void {
  if (hoveredEdgeIndex !== index) {
    hoveredEdgeIndex = index;
    scheduleRefresh();
  }
}

export function getHoveredEdge(): number | null {
  return hoveredEdgeIndex;
}

export async function initRenderer(container: HTMLElement, graphData: GraphData): Promise<void> {
  destroyRenderer();
  currentGraphData = graphData;

  // Build graph model and cache node sizes
  graphModel = new GraphModel();
  nodeSizes.clear();
  for (const node of graphData.nodes) {
    graphModel.addNode(node.id, {
      x: node.x,
      y: node.y,
      label: node.type,
      color: node.color,
      opType: node.type,
      nodeName: node.name,
      category: node.category,
      shape: node.shape,
      elementType: node.element_type,
      attributes: node.attributes,
    });
    nodeSizes.set(node.id, {
      width: node.width || 100,
      height: node.height || 32,
    });
  }
  for (const edge of graphData.edges) {
    if (graphModel.hasNode(edge.source) && graphModel.hasNode(edge.target)) {
      graphModel.addEdge(edge.source, edge.target);
    }
  }

  // Create canvas and init WebGPU renderer
  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  container.appendChild(canvas);

  const renderer = await WebGPURenderer.create(canvas);
  gpuRenderer = renderer;

  // Set up pan/zoom on canvas
  panZoom = new PanZoom(canvas, {
    isNodeHit: (cx, cy) => {
      const rect = canvas.getBoundingClientRect();
      const gp = panZoom!.viewportToGraph(cx - rect.left, cy - rect.top);
      return gpuRenderer!.hitGrid.query(gp.x, gp.y) !== null;
    },
  });

  renderer.setGraph(graphData, getNodeSize);

  // Register camera listener BEFORE fitToView so the setState triggers updateCamera
  panZoom.on('updated', () => {
    renderer.updateCamera(panZoom!.translateX, panZoom!.translateY, panZoom!.ratio);
    graphStore.cameraVersion++;
    scheduleRefresh();
  });

  fitToView(container);

  // Ensure initial camera + appearance are set (in case fitToView had no nodes)
  renderer.updateCamera(panZoom.translateX, panZoom.translateY, panZoom.ratio);
  doRefresh();
}

function fitToView(container: HTMLElement): void {
  if (!panZoom || !currentGraphData) return;

  const graphData = currentGraphData;
  if (graphData.nodes.length === 0) return;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const node of graphData.nodes) {
    const size = getNodeSize(node.id);
    minX = Math.min(minX, node.x);
    minY = Math.min(minY, node.y);
    maxX = Math.max(maxX, node.x + size.width);
    maxY = Math.max(maxY, node.y + size.height);
  }

  const graphWidth = maxX - minX;
  const graphHeight = maxY - minY;
  const containerWidth = container.clientWidth || 800;
  const containerHeight = container.clientHeight || 600;

  const scaleX = containerWidth / (graphWidth + 100);
  const scaleY = containerHeight / (graphHeight + 100);
  const scale = Math.min(scaleX, scaleY, 1.5);

  const tx = (containerWidth - graphWidth * scale) / 2 - minX * scale;
  const ty = (containerHeight - graphHeight * scale) / 2 - minY * scale;

  panZoom.setState({ tx, ty, scale });
}

export function destroyRenderer(): void {
  if (panZoom) {
    panZoom.destroy();
    panZoom = null;
  }
  if (gpuRenderer) {
    gpuRenderer.canvas.remove();
    gpuRenderer.destroy();
    gpuRenderer = null;
  }
  graphModel = null;
  currentGraphData = null;
  hoveredNodeId = null;
  hoveredEdgeIndex = null;
  nodeSizes.clear();
}

export function centerOnNode(nodeId: string, animate = true): void {
  if (!graphModel || !panZoom || !gpuRenderer || !graphModel.hasNode(nodeId)) return;

  const attrs = graphModel.getNodeAttributes(nodeId);
  const size = getNodeSize(nodeId);
  const containerWidth = gpuRenderer.canvas.clientWidth || 800;
  const containerHeight = gpuRenderer.canvas.clientHeight || 600;

  const targetScale = Math.max(panZoom.ratio, 0.8);
  const cx = attrs.x + size.width / 2;
  const cy = attrs.y + size.height / 2;
  const tx = containerWidth / 2 - cx * targetScale;
  const ty = containerHeight / 2 - cy * targetScale;

  if (animate) {
    panZoom.animate({ tx, ty, scale: targetScale }, 300);
  } else {
    panZoom.setState({ tx, ty, scale: targetScale });
  }
}

/** Animate camera to fit all non-grayed nodes. If no grayed nodes, fits the entire graph. */
export function fitToSubSession(): void {
  if (!panZoom || !currentGraphData) return;

  const grayedNodes = graphStore.grayedNodes;
  const nodes = currentGraphData.nodes;
  if (nodes.length === 0) return;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  let count = 0;

  for (const node of nodes) {
    if (grayedNodes.size > 0 && grayedNodes.has(node.id)) continue;
    const size = getNodeSize(node.id);
    minX = Math.min(minX, node.x);
    minY = Math.min(minY, node.y);
    maxX = Math.max(maxX, node.x + size.width);
    maxY = Math.max(maxY, node.y + size.height);
    count++;
  }

  if (count === 0) return;
  panZoom.fitToBounds({ minX, minY, maxX, maxY });
}

function scheduleRefresh(): void {
  if (refreshScheduled) return;
  refreshScheduled = true;
  requestAnimationFrame(() => {
    refreshScheduled = false;
    doRefresh();
  });
}

function doRefresh(): void {
  if (!panZoom || !gpuRenderer) return;

  gpuRenderer.updateAppearance(
    graphStore.nodeStatusMap,
    graphStore.selectedNodeId,
    hoveredNodeId,
    graphStore.searchResults,
    graphStore.searchVisible,
    graphStore.grayedNodes,
    panZoom.ratio,
    graphStore.nodeOverrides,
    graphStore.accuracyViewActive,
    hoveredEdgeIndex,
    graphStore.selectedEdgeIndex,
  );
}

export function refreshRenderer(): void {
  doRefresh();
}
