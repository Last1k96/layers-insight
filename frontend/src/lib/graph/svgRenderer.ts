/**
 * SVG renderer with Netron-style nodes and edges.
 * Nodes: colored rounded rectangles with op-type labels.
 * Edges: cubic Bézier B-splines through ELK waypoints.
 */
import type { GraphData, GraphEdge } from '../stores/types';
import type { NodeStatus } from '../stores/graph.svelte';
import { graphStore } from '../stores/graph.svelte';
import { configStore } from '../stores/config.svelte';
import { STATUS_COLORS, isLightNodeColor } from './opColors';

const NS = 'http://www.w3.org/2000/svg';
const NODE_HEIGHT = 32;
const NODE_MIN_WIDTH = 100;
const NODE_PADDING = 20;
const NODE_RADIUS = 5;
const FONT = '-apple-system, BlinkMacSystemFont, "Segoe WPC", "Segoe UI", Ubuntu, "Droid Sans", sans-serif';
const FONT_SIZE = 11;

// Offscreen canvas for text measurement
let measureCtx: CanvasRenderingContext2D | null = null;
function measureText(text: string): number {
  if (!measureCtx) {
    const canvas = document.createElement('canvas');
    measureCtx = canvas.getContext('2d')!;
    measureCtx.font = `${FONT_SIZE}px ${FONT}`;
  }
  return measureCtx.measureText(text).width;
}

function getAccuracyGradientColor(mse: number): string {
  if (configStore.gradientMode === 'threshold') {
    return mse <= configStore.globalThreshold ? '#10B981' : '#EF4444';
  }
  const logMse = mse > 0 ? Math.log10(mse) : -10;
  const t = Math.max(0, Math.min(1, (logMse + 8) / 6));
  const r = Math.floor(Math.min(255, t * 2 * 255));
  const g = Math.floor(Math.min(255, (1 - Math.max(0, t - 0.5) * 2) * 255));
  return `rgb(${r}, ${g}, 0)`;
}

/** Node dimensions cache (id -> {width, height}) */
const nodeSizes = new Map<string, { width: number; height: number }>();

export function getNodeSize(nodeId: string): { width: number; height: number } {
  return nodeSizes.get(nodeId) ?? { width: NODE_MIN_WIDTH, height: NODE_HEIGHT };
}

/**
 * Build the Netron-style cubic B-spline path through waypoints.
 * CP1 = (2*p0 + p1) / 3
 * CP2 = (p0 + 2*p1) / 3
 * End = (p0 + 4*p1 + p2) / 6
 */
function buildEdgePath(waypoints: { x: number; y: number }[]): string {
  if (waypoints.length < 2) return '';

  if (waypoints.length === 2) {
    // Simple straight line with gentle S-curve
    const [p0, p1] = waypoints;
    const midY = (p0.y + p1.y) / 2;
    return `M ${p0.x} ${p0.y} C ${p0.x} ${midY}, ${p1.x} ${midY}, ${p1.x} ${p1.y}`;
  }

  // B-spline through waypoints (Netron algorithm)
  const pts = waypoints;
  let d = `M ${pts[0].x} ${pts[0].y}`;

  if (pts.length === 3) {
    // Quadratic through middle point
    d += ` Q ${pts[1].x} ${pts[1].y}, ${pts[2].x} ${pts[2].y}`;
    return d;
  }

  // Cubic B-spline: process control points
  d += ` L ${(2 * pts[0].x + pts[1].x) / 3} ${(2 * pts[0].y + pts[1].y) / 3}`;
  for (let i = 1; i < pts.length - 2; i++) {
    const p0 = pts[i];
    const p1 = pts[i + 1];
    const cp1x = (2 * p0.x + p1.x) / 3;
    const cp1y = (2 * p0.y + p1.y) / 3;
    const cp2x = (p0.x + 2 * p1.x) / 3;
    const cp2y = (p0.y + 2 * p1.y) / 3;
    const endx = (p0.x + 4 * p1.x + pts[i + 2].x) / 6;
    const endy = (p0.y + 4 * p1.y + pts[i + 2].y) / 6;
    d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${endx} ${endy}`;
  }
  // Last segment to final point
  const n = pts.length;
  const pn2 = pts[n - 2];
  const pn1 = pts[n - 1];
  d += ` C ${(2 * pn2.x + pn1.x) / 3} ${(2 * pn2.y + pn1.y) / 3}, ${(pn2.x + 2 * pn1.x) / 3} ${(pn2.y + 2 * pn1.y) / 3}, ${pn1.x} ${pn1.y}`;

  return d;
}

/** Generate S-curve fallback when no waypoints are available */
function buildFallbackEdgePath(
  sx: number, sy: number, sw: number, sh: number,
  tx: number, ty: number, _tw: number, _th: number
): string {
  const startX = sx + sw / 2;
  const startY = sy + sh;
  const endX = tx + _tw / 2;
  const endY = ty;
  const midY = (startY + endY) / 2;
  return `M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`;
}

export interface SVGRendererState {
  svg: SVGSVGElement;
  edgesGroup: SVGGElement;
  nodesGroup: SVGGElement;
  nodeElements: Map<string, SVGGElement>;
  edgeElements: Map<number, SVGPathElement>;
  graphData: GraphData;
}

export function createSVGStructure(container: HTMLElement): SVGRendererState & { viewport: SVGGElement } {
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', '100%');
  svg.style.background = '#1a1a2e';
  svg.style.display = 'block';

  // Arrow marker definition
  const defs = document.createElementNS(NS, 'defs');
  const marker = document.createElementNS(NS, 'marker');
  marker.setAttribute('id', 'arrowhead');
  marker.setAttribute('viewBox', '0 0 8 6');
  marker.setAttribute('refX', '8');
  marker.setAttribute('refY', '3');
  marker.setAttribute('markerWidth', '8');
  marker.setAttribute('markerHeight', '6');
  marker.setAttribute('orient', 'auto');
  const arrowPath = document.createElementNS(NS, 'path');
  arrowPath.setAttribute('d', 'M 0 0 L 8 3 L 0 6 Z');
  arrowPath.setAttribute('fill', '#888');
  marker.appendChild(arrowPath);
  defs.appendChild(marker);
  svg.appendChild(defs);

  // Style for hover
  const style = document.createElementNS(NS, 'style');
  style.textContent = `
    .node:hover .node-rect { stroke: rgba(220,0,0,0.9); stroke-width: 2px; }
    .node { cursor: pointer; }
    .node-label { pointer-events: none; }
    .lod-hide-text .node-label { display: none; }
  `;
  svg.appendChild(style);

  const viewport = document.createElementNS(NS, 'g');
  viewport.id = 'viewport';
  const edgesGroup = document.createElementNS(NS, 'g');
  edgesGroup.id = 'edges';
  const nodesGroup = document.createElementNS(NS, 'g');
  nodesGroup.id = 'nodes';

  viewport.appendChild(edgesGroup);
  viewport.appendChild(nodesGroup);
  svg.appendChild(viewport);
  container.appendChild(svg);

  return {
    svg,
    viewport,
    edgesGroup,
    nodesGroup,
    nodeElements: new Map(),
    edgeElements: new Map(),
    graphData: { nodes: [], edges: [] },
  };
}

export function renderGraph(state: SVGRendererState, graphData: GraphData): void {
  state.graphData = graphData;
  state.nodesGroup.innerHTML = '';
  state.edgesGroup.innerHTML = '';
  state.nodeElements.clear();
  state.edgeElements.clear();
  nodeSizes.clear();

  // Create nodes
  for (const node of graphData.nodes) {
    const label = node.type;
    const textWidth = measureText(label);
    const nodeWidth = Math.max(NODE_MIN_WIDTH, textWidth + NODE_PADDING * 2);
    nodeSizes.set(node.id, { width: nodeWidth, height: NODE_HEIGHT });

    const g = document.createElementNS(NS, 'g');
    g.classList.add('node');
    g.dataset.id = node.id;
    g.setAttribute('transform', `translate(${node.x}, ${node.y})`);

    // Rounded rect
    const rect = document.createElementNS(NS, 'path');
    const r = NODE_RADIUS;
    const w = nodeWidth;
    const h = NODE_HEIGHT;
    rect.setAttribute('d', `M ${r} 0 H ${w - r} Q ${w} 0 ${w} ${r} V ${h - r} Q ${w} ${h} ${w - r} ${h} H ${r} Q 0 ${h} 0 ${h - r} V ${r} Q 0 0 ${r} 0 Z`);
    rect.setAttribute('fill', node.color);
    rect.setAttribute('stroke', '#333');
    rect.setAttribute('stroke-width', '1');
    rect.classList.add('node-rect');
    g.appendChild(rect);

    // Text label
    const text = document.createElementNS(NS, 'text');
    text.setAttribute('x', String(nodeWidth / 2));
    text.setAttribute('y', String(NODE_HEIGHT / 2));
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('dominant-baseline', 'central');
    text.setAttribute('font-family', FONT);
    text.setAttribute('font-size', String(FONT_SIZE));
    text.setAttribute('fill', isLightNodeColor(node.color) ? '#333' : '#fff');
    text.classList.add('node-label');
    text.textContent = label;
    g.appendChild(text);

    state.nodesGroup.appendChild(g);
    state.nodeElements.set(node.id, g);
  }

  // Create edges
  graphData.edges.forEach((edge, i) => {
    const path = document.createElementNS(NS, 'path');
    const d = computeEdgePath(edge, graphData);
    path.setAttribute('d', d);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', '#888');
    path.setAttribute('stroke-width', '1');
    path.setAttribute('marker-end', 'url(#arrowhead)');
    state.edgesGroup.appendChild(path);
    state.edgeElements.set(i, path);
  });
}

function computeEdgePath(edge: GraphEdge, graphData: GraphData): string {
  if (edge.waypoints && edge.waypoints.length >= 2) {
    return buildEdgePath(edge.waypoints);
  }

  // Fallback S-curve
  const sourceNode = graphData.nodes.find(n => n.id === edge.source);
  const targetNode = graphData.nodes.find(n => n.id === edge.target);
  if (!sourceNode || !targetNode) return '';

  const sSize = getNodeSize(edge.source);
  const tSize = getNodeSize(edge.target);
  return buildFallbackEdgePath(
    sourceNode.x, sourceNode.y, sSize.width, sSize.height,
    targetNode.x, targetNode.y, tSize.width, tSize.height,
  );
}

export function updateNodeAppearance(
  state: SVGRendererState,
  nodeStatusMap: Map<string, NodeStatus>,
  selectedNodeId: string | null,
  searchResults: { id: string }[] | null,
  searchVisible: boolean,
  grayedNodes: Set<string>,
  zoomRatio: number,
): void {
  const searchActive = searchVisible && searchResults && searchResults.length > 0;

  for (const node of state.graphData.nodes) {
    const g = state.nodeElements.get(node.id);
    if (!g) continue;

    const rect = g.querySelector('.node-rect') as SVGPathElement | null;
    const text = g.querySelector('.node-label') as SVGTextElement | null;
    if (!rect || !text) continue;

    const isGrayed = grayedNodes.has(node.id);
    const isSelected = selectedNodeId === node.id;
    const nodeStatus = nodeStatusMap.get(node.id);

    // Reset
    rect.setAttribute('fill', isGrayed ? '#1f2937' : node.color);
    rect.setAttribute('stroke', '#333');
    rect.setAttribute('stroke-width', '1');
    text.textContent = isGrayed ? '' : node.type;
    text.setAttribute('fill', isLightNodeColor(isGrayed ? '#1f2937' : node.color) ? '#333' : '#fff');
    g.style.opacity = '1';

    // Status border color
    if (nodeStatus && !isGrayed) {
      if (nodeStatus.status === 'success' && nodeStatus.metrics) {
        const color = getAccuracyGradientColor(nodeStatus.metrics.mse);
        rect.setAttribute('stroke', color);
        rect.setAttribute('stroke-width', isSelected ? '3' : '2');
      } else {
        const statusColor = STATUS_COLORS[nodeStatus.status];
        if (statusColor) {
          rect.setAttribute('stroke', statusColor);
          rect.setAttribute('stroke-width', isSelected ? '3' : '2');
        }
      }
      if (nodeStatus.status === 'executing') {
        rect.setAttribute('stroke-width', '3');
      }
    }

    // Selection highlight
    if (isSelected) {
      rect.setAttribute('stroke', 'rgba(220,0,0,0.9)');
      rect.setAttribute('stroke-width', '2');
    }

    // Search dimming
    if (searchActive) {
      const isMatch = searchResults!.some(r => r.id === node.id);
      if (!isMatch) {
        g.style.opacity = '0.15';
      }
    }

    // LOD: hide text when zoomed out
    if (zoomRatio < 0.4) {
      text.textContent = '';
    } else if (zoomRatio < 0.7) {
      text.textContent = '';
    }
  }

  // Dim edges during search
  for (const [, path] of state.edgeElements) {
    if (searchActive) {
      path.setAttribute('stroke', '#444');
      path.style.opacity = '0.3';
    } else {
      path.setAttribute('stroke', '#888');
      path.style.opacity = '1';
    }
  }
}
