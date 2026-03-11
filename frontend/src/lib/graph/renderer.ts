/**
 * Graph renderer facade — delegates to SVG renderer, GraphModel, and PanZoom.
 * Maintains the same public API as the previous Sigma.js-based renderer.
 */
import type { GraphData } from '../stores/types';
import { graphStore } from '../stores/graph.svelte';
import { GraphModel } from './graphModel';
import { PanZoom } from './panZoom';
import { createSVGStructure, renderGraph, updateNodeAppearance, getNodeSize } from './svgRenderer';
import type { SVGRendererState } from './svgRenderer';

let graphModel: GraphModel | null = null;
let panZoom: PanZoom | null = null;
let svgState: (SVGRendererState & { viewport: SVGGElement }) | null = null;
let refreshScheduled = false;

export function getGraph(): GraphModel | null {
  return graphModel;
}

export function getCamera(): PanZoom | null {
  return panZoom;
}

export function getSVGState(): SVGRendererState | null {
  return svgState;
}

export function initRenderer(container: HTMLElement, graphData: GraphData): void {
  destroyRenderer();

  // Build graph model
  graphModel = new GraphModel();
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
  }
  for (const edge of graphData.edges) {
    if (graphModel.hasNode(edge.source) && graphModel.hasNode(edge.target)) {
      graphModel.addEdge(edge.source, edge.target);
    }
  }

  // Create SVG structure and render
  svgState = createSVGStructure(container);
  renderGraph(svgState, graphData);

  // Set up pan/zoom
  panZoom = new PanZoom(svgState.svg, svgState.viewport);

  // Fit graph in view
  fitToView(container);

  // Listen for zoom changes to update LOD
  panZoom.on('updated', () => {
    scheduleRefresh();
  });
}

function fitToView(container: HTMLElement): void {
  if (!svgState || !panZoom) return;

  const graphData = svgState.graphData;
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
  if (svgState) {
    svgState.svg.remove();
    svgState = null;
  }
  graphModel = null;
}

export function centerOnNode(nodeId: string, animate = true): void {
  if (!graphModel || !panZoom || !svgState || !graphModel.hasNode(nodeId)) return;

  const attrs = graphModel.getNodeAttributes(nodeId);
  const size = getNodeSize(nodeId);
  const svg = svgState.svg;
  const containerWidth = svg.clientWidth || 800;
  const containerHeight = svg.clientHeight || 600;

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

function scheduleRefresh(): void {
  if (refreshScheduled) return;
  refreshScheduled = true;
  requestAnimationFrame(() => {
    refreshScheduled = false;
    doRefresh();
  });
}

function doRefresh(): void {
  if (!svgState || !panZoom) return;

  updateNodeAppearance(
    svgState,
    graphStore.nodeStatusMap,
    graphStore.selectedNodeId,
    graphStore.searchResults,
    graphStore.searchVisible,
    graphStore.grayedNodes,
    panZoom.ratio,
  );
}

export function refreshRenderer(): void {
  doRefresh();
}
