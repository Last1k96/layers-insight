import Sigma from 'sigma';
import Graph from 'graphology';
import { EdgeCurvedArrowProgram } from '@sigma/edge-curve';
import type { GraphData, GraphNode } from '../stores/types';
import { graphStore, type NodeStatus } from '../stores/graph.svelte';
import { configStore } from '../stores/config.svelte';
import { STATUS_COLORS, isLightNodeColor } from './opColors';
import NodeRectProgram from './nodeRectProgram';
import { drawRectNodeLabel, drawRectNodeHover } from './drawLabel';

/**
 * Netron-style sizing:
 * Netron uses dynamic node sizes based on text content.
 * In Sigma.js we use a fixed size that looks proportionally similar.
 */
const NODE_SIZE = 6;
const NODE_SIZE_SELECTED = 7;
const NODE_SIZE_ZOOMED_OUT = 3;

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

let sigma: Sigma | null = null;
let graph: Graph | null = null;

export function getGraph(): Graph | null {
  return graph;
}

export function getSigma(): Sigma | null {
  return sigma;
}

export function getCamera() {
  return sigma?.getCamera() ?? null;
}

export function initRenderer(container: HTMLElement, graphData: GraphData): void {
  destroyRenderer();

  graph = new Graph();

  for (const node of graphData.nodes) {
    graph.addNode(node.id, {
      x: node.x,
      y: node.y,
      size: NODE_SIZE,
      label: node.type,       // Netron: show op type inside the node
      color: node.color,
      type: 'rect',
      opType: node.type,
      nodeName: node.name,
      category: node.category,
      shape: node.shape,
      elementType: node.element_type,
      attributes: node.attributes,
    });
  }

  for (const edge of graphData.edges) {
    try {
      graph.addEdge(edge.source, edge.target, {
        color: '#000000',       // Netron: black edges
        size: 1,                // Netron: 1px stroke
        type: 'curvedArrow',    // Netron-style curved edges
        sourcePort: edge.source_port,
        targetPort: edge.target_port,
      });
    } catch {
      // Skip duplicate edges
    }
  }

  sigma = new Sigma(graph, container, {
    renderLabels: true,
    renderEdgeLabels: false,
    // Netron system font stack (from grapher.css)
    labelFont: '-apple-system, BlinkMacSystemFont, "Segoe WPC", "Segoe UI", Ubuntu, "Droid Sans", sans-serif',
    labelSize: 11,
    labelWeight: 'normal',
    labelColor: { color: '#ffffff' },
    // Edges — Netron style: black, 1px
    defaultEdgeColor: '#000000',
    defaultEdgeType: 'curvedArrow',
    minEdgeThickness: 0.5,
    // Nodes
    defaultNodeType: 'rect',
    allowInvalidContainer: true,
    // Show labels generously
    labelRenderedSizeThreshold: 1,
    labelDensity: 3,
    labelGridCellSize: 60,
    // Custom programs & renderers
    nodeProgramClasses: {
      rect: NodeRectProgram,
    },
    edgeProgramClasses: {
      curvedArrow: EdgeCurvedArrowProgram,
    },
    defaultDrawNodeLabel: drawRectNodeLabel,
    defaultDrawNodeHover: drawRectNodeHover,

    nodeReducer: (node, data) => {
      const res = { ...data };
      const nodeStatus = graphStore.nodeStatusMap.get(node);
      const selected = graphStore.selectedNodeId === node;
      const searchActive = graphStore.searchVisible && graphStore.searchResults.length > 0;

      // Grayed nodes (model cutting)
      if (graphStore.grayedNodes.has(node)) {
        res.color = '#1f2937';
        res.label = '';
        res.size = NODE_SIZE_ZOOMED_OUT;
        return res;
      }

      // Label = op type (Netron style)
      res.label = (data.opType as string) || '';

      // Status border
      if (nodeStatus) {
        if (nodeStatus.status === 'success' && nodeStatus.metrics) {
          const gradientColor = getAccuracyGradientColor(nodeStatus.metrics.mse);
          if (gradientColor) {
            res.borderColor = gradientColor;
            res.borderSize = selected ? 3 : 2;
          }
        } else {
          const statusColor = STATUS_COLORS[nodeStatus.status];
          if (statusColor) {
            res.borderColor = statusColor;
            res.borderSize = selected ? 3 : 2;
          }
        }
        if (nodeStatus.status === 'executing') {
          res.borderSize = 3;
        }
      }

      // Selection
      if (selected) {
        res.highlighted = true;
        res.size = NODE_SIZE_SELECTED;
        res.zIndex = 10;
      }

      // Search dimming
      if (searchActive) {
        const isMatch = graphStore.searchResults.some(r => r.id === node);
        if (!isMatch) {
          res.color = '#222';
          res.label = '';
        }
      }

      // LOD: simplify at low zoom
      const camera = getCamera();
      if (camera) {
        const ratio = camera.ratio;
        if (ratio > 2.5) {
          res.label = '';
          res.size = NODE_SIZE_ZOOMED_OUT;
        } else if (ratio > 1.5) {
          res.label = '';
        }
      }

      return res;
    },
    edgeReducer: (edge, data) => {
      const res = { ...data };
      const searchActive = graphStore.searchVisible && graphStore.searchResults.length > 0;
      if (searchActive) {
        res.color = '#1a1a1a';
      }
      return res;
    },
  });

  sigma.getCamera().on('updated', () => {
    sigma?.refresh();
  });
}

export function destroyRenderer(): void {
  if (sigma) {
    sigma.kill();
    sigma = null;
  }
  graph = null;
}

export function centerOnNode(nodeId: string, animate = true): void {
  if (!sigma || !graph || !graph.hasNode(nodeId)) return;

  const attrs = graph.getNodeAttributes(nodeId);
  const camera = sigma.getCamera();

  if (animate) {
    camera.animate({ x: attrs.x, y: attrs.y, ratio: 0.3 }, { duration: 300 });
  } else {
    camera.setState({ x: attrs.x, y: attrs.y, ratio: 0.3 });
  }
}

export function refreshRenderer(): void {
  sigma?.refresh();
}
