import Sigma from 'sigma';
import Graph from 'graphology';
import type { GraphData, GraphNode } from '../stores/types';
import { graphStore, type NodeStatus } from '../stores/graph.svelte';
import { configStore } from '../stores/config.svelte';
import { STATUS_COLORS } from './opColors';

function getAccuracyGradientColor(mse: number): string {
  if (configStore.gradientMode === 'threshold') {
    return mse <= configStore.globalThreshold ? '#10B981' : '#EF4444';
  }
  // Auto-scale: use a log scale: green < 1e-6, yellow ~ 1e-4, red > 1e-2
  const logMse = mse > 0 ? Math.log10(mse) : -10;
  const t = Math.max(0, Math.min(1, (logMse + 8) / 6)); // -8 to -2 range
  // Green -> Yellow -> Red
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

  // Add nodes
  for (const node of graphData.nodes) {
    graph.addNode(node.id, {
      x: node.x,
      y: node.y,
      size: 8,
      label: node.name,
      color: node.color,
      type: 'circle',
      // Store extra data
      opType: node.type,
      category: node.category,
      shape: node.shape,
      elementType: node.element_type,
      attributes: node.attributes,
    });
  }

  // Add edges
  for (const edge of graphData.edges) {
    try {
      graph.addEdge(edge.source, edge.target, {
        color: '#4b5563',
        size: 1,
        type: 'arrow',
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
    labelColor: { color: '#e5e7eb' },
    labelSize: 12,
    labelFont: 'monospace',
    defaultEdgeColor: '#4b5563',
    defaultEdgeType: 'arrow',
    allowInvalidContainer: true,
    nodeReducer: (node, data) => {
      const res = { ...data };
      const nodeStatus = graphStore.nodeStatusMap.get(node);
      const selected = graphStore.selectedNodeId === node;
      const searchActive = graphStore.searchVisible && graphStore.searchResults.length > 0;

      // Grayed nodes (model cutting)
      if (graphStore.grayedNodes.has(node)) {
        res.color = '#1f2937';
        res.label = '';
        res.size = 4;
        res.borderColor = undefined;
        res.borderSize = 0;
        return res;  // Skip other rendering for grayed nodes
      }

      // Status ring via border color
      if (nodeStatus) {
        const statusColor = STATUS_COLORS[nodeStatus.status];
        // Phase 2: Accuracy gradient for completed nodes
        if (nodeStatus.status === 'success' && nodeStatus.metrics) {
          const gradientColor = getAccuracyGradientColor(nodeStatus.metrics.mse);
          if (gradientColor) {
            res.borderColor = gradientColor;
            res.borderSize = selected ? 3 : 2;
          }
        } else if (statusColor) {
          res.borderColor = statusColor;
          res.borderSize = selected ? 3 : 2;
        }
        // Pulsing effect for executing (handled via CSS)
        if (nodeStatus.status === 'executing') {
          res.borderSize = 3;
        }
      }

      // Selection highlight
      if (selected) {
        res.highlighted = true;
        res.size = 12;
        res.zIndex = 10;
      }

      // Search dimming
      if (searchActive) {
        const isMatch = graphStore.searchResults.some(r => r.id === node);
        if (!isMatch) {
          res.color = '#374151';
          res.label = '';
        }
      }

      // LOD: hide labels at low zoom
      const camera = getCamera();
      if (camera) {
        const ratio = camera.ratio;
        if (ratio > 2) {
          res.label = '';
          res.size = 4;
        } else if (ratio > 1) {
          res.label = data.opType as string || '';
        }
      }

      return res;
    },
    edgeReducer: (edge, data) => {
      const res = { ...data };
      const searchActive = graphStore.searchVisible && graphStore.searchResults.length > 0;

      if (searchActive) {
        res.color = '#1f2937';
      }

      return res;
    },
  });

  // Refresh on camera change for LOD
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
