/**
 * Graph renderer facade — WebGPU-based graph rendering.
 */
import type { GraphData } from '../stores/types';
import { graphStore } from '../stores/graph.svelte';
import { GraphModel } from './graphModel';
import { PanZoom } from './panZoom';
import { WebGPURenderer } from './webgpu/WebGPURenderer';
import { MinimapTarget } from './webgpu/MinimapTarget';

let graphModel: GraphModel | null = null;
let panZoom: PanZoom | null = null;
let gpuRenderer: WebGPURenderer | null = null;
let minimapTarget: MinimapTarget | null = null;
let refreshScheduled = false;
let hoveredNodeId: string | null = null;
let hoveredEdgeIndex: number | null = null;
let currentGraphData: GraphData | null = null;
const rendererReadyListeners: Array<(r: WebGPURenderer) => void> = [];

/** Node dimensions cache (id -> {width, height}) */
const nodeSizes = new Map<string, { width: number; height: number }>();

export function getNodeSize(nodeId: string): { width: number; height: number } {
  return nodeSizes.get(nodeId) ?? { width: 100, height: 32 };
}

/**
 * Decide whether a zoom change requires rebuilding the text glyph buffer.
 * Mirrors the fade logic baked into WebGPURenderer.rebuildText():
 *  - zoom < 0.05  → glyphs not built at all
 *  - 0.05 ≤ zoom < 0.1  → glyphs built with alpha 0
 *  - 0.1  ≤ zoom < 0.3  → alpha varies linearly with zoom (must rebuild on every change)
 *  - zoom ≥ 0.3  → alpha = 1
 *
 * Returns true when prev and next zooms cross any of those boundaries, OR when
 * either of them sits in the linear-fade band (so the per-glyph alpha is changing).
 */
function textFadeChanged(prev: number, next: number): boolean {
  if ((prev < 0.05) !== (next < 0.05)) return true;
  if ((prev < 0.1) !== (next < 0.1)) return true;
  if ((prev < 0.3) !== (next < 0.3)) return true;
  // Both zooms inside [0.1, 0.3) → alpha is interpolating, rebuild every step
  if (prev >= 0.1 && prev < 0.3 && next >= 0.1 && next < 0.3) return true;
  return false;
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

/** Subscribe to be notified when the WebGPU renderer becomes available.
 *  Fires immediately if a renderer already exists. Returns an unsubscribe fn. */
export function onRendererReady(fn: (r: WebGPURenderer) => void): () => void {
  if (gpuRenderer) {
    fn(gpuRenderer);
  }
  rendererReadyListeners.push(fn);
  return () => {
    const idx = rendererReadyListeners.indexOf(fn);
    if (idx >= 0) rendererReadyListeners.splice(idx, 1);
  };
}

/** Attach a canvas as the minimap target. Idempotent — replaces any existing minimap canvas. */
export function attachMinimap(canvas: HTMLCanvasElement): MinimapTarget | null {
  if (!gpuRenderer) return null;
  if (minimapTarget) {
    minimapTarget.destroy();
    minimapTarget = null;
  }
  minimapTarget = MinimapTarget.create(canvas, gpuRenderer);
  gpuRenderer.attachMinimap(minimapTarget);
  return minimapTarget;
}

export function detachMinimap(): void {
  if (gpuRenderer) gpuRenderer.detachMinimap();
  if (minimapTarget) {
    minimapTarget.destroy();
    minimapTarget = null;
  }
}

export function getMinimapTarget(): MinimapTarget | null {
  return minimapTarget;
}

export function setHoveredNode(nodeId: string | null): void {
  if (hoveredNodeId !== nodeId) {
    hoveredNodeId = nodeId;
    graphStore.hoveredNodeId = nodeId;
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

function populateGraphModel(graphData: GraphData): GraphModel {
  const model = new GraphModel();
  for (const node of graphData.nodes) {
    model.addNode(node.id, {
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
    if (model.hasNode(edge.source) && model.hasNode(edge.target)) {
      model.addEdge(edge.source, edge.target);
    }
  }
  return model;
}

export function isRendererInitialized(): boolean {
  return gpuRenderer !== null;
}

/** Swap the rendered graph to a different ``GraphData`` object without
 *  tearing down the WebGPU pipeline. Used to switch between a sub-session's
 *  tight graph and the session's full graph. Preserves canvas/panZoom;
 *  rebuilds the graph model, node sizes, and GPU scene from scratch.
 *  Camera is preserved — callers may follow up with fitToSubSession() if
 *  a recenter is desired. */
export function setActiveGraph(graphData: GraphData): void {
  if (!gpuRenderer) return;
  currentGraphData = graphData;

  nodeSizes.clear();
  graphModel = populateGraphModel(graphData);

  hoveredNodeId = null;
  hoveredEdgeIndex = null;
  graphStore.hoveredNodeId = null;

  gpuRenderer.setGraph(graphData, getNodeSize);
  doRefresh();
}

export async function initRenderer(container: HTMLElement, graphData: GraphData): Promise<void> {
  destroyRenderer();
  currentGraphData = graphData;

  nodeSizes.clear();
  graphModel = populateGraphModel(graphData);

  // Create canvas and init WebGPU renderer
  const canvas = document.createElement('canvas');
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';
  container.appendChild(canvas);

  const renderer = await WebGPURenderer.create(canvas);
  gpuRenderer = renderer;
  for (const fn of rendererReadyListeners) fn(renderer);

  // Set up pan/zoom on canvas
  panZoom = new PanZoom(canvas, {
    isNodeHit: (cx, cy) => {
      const rect = canvas.getBoundingClientRect();
      const gp = panZoom!.viewportToGraph(cx - rect.left, cy - rect.top);
      return gpuRenderer!.hitGrid.query(gp.x, gp.y) !== null;
    },
  });

  renderer.setGraph(graphData, getNodeSize);

  // Register camera listener BEFORE fitToView so the setState triggers updateCamera.
  // Pure camera moves do NOT call scheduleRefresh() — node and edge instance data
  // don't depend on the camera matrix, so a full updateAppearance() rebuild would
  // be wasted work. We only kick the lighter rebuildCameraDependentParts() path
  // when the new zoom enters/leaves the text fade band or when an edge is selected
  // (in which case the ghost overlay needs reprojecting).
  let lastZoomRatio = panZoom.ratio;
  panZoom.on('updated', () => {
    const newZoom = panZoom!.ratio;
    const prevZoom = lastZoomRatio;
    lastZoomRatio = newZoom;

    renderer.updateCamera(panZoom!.translateX, panZoom!.translateY, newZoom);
    graphStore.cameraVersion++;

    const textNeedsRebuild = textFadeChanged(prevZoom, newZoom);
    const ghostsNeedRebuild = graphStore.selectedEdgeIndex !== null;
    if (!textNeedsRebuild && !ghostsNeedRebuild) return;

    renderer.rebuildCameraDependentParts(
      newZoom,
      graphStore.selectedEdgeIndex,
      hoveredEdgeIndex,
    );
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
  let scale = Math.min(scaleX, scaleY, 1.5);

  // For very tall/wide graphs, don't zoom out so far that nodes become
  // invisible. Instead, clamp to a minimum zoom where nodes are readable
  // and center on the top of the graph so the user sees something useful.
  const MIN_FIT_ZOOM = 0.15;
  if (scale < MIN_FIT_ZOOM) {
    scale = MIN_FIT_ZOOM;
    const tx = (containerWidth - graphWidth * scale) / 2 - minX * scale;
    const ty = 40 - minY * scale;
    panZoom.setState({ tx, ty, scale });
    return;
  }

  const tx = (containerWidth - graphWidth * scale) / 2 - minX * scale;
  const ty = (containerHeight - graphHeight * scale) / 2 - minY * scale;

  panZoom.setState({ tx, ty, scale });
}

export function destroyRenderer(): void {
  if (panZoom) {
    panZoom.destroy();
    panZoom = null;
  }
  if (minimapTarget) {
    // The renderer.destroy() will also tear it down, but null our handle first
    minimapTarget = null;
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
  graphStore.hoveredNodeId = null;
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
