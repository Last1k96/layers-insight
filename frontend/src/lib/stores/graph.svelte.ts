import type { GraphData, GraphNode, TaskStatus, AccuracyMetrics, DeviceResult } from './types';

export interface NodeStatus {
  status: TaskStatus;
  taskId?: string;
  stage?: string;
  metrics?: AccuracyMetrics;
  mainResult?: DeviceResult;
  refResult?: DeviceResult;
  errorDetail?: string;
}

class GraphStore {
  graphData = $state<GraphData | null>(null);
  selectedNodeId = $state<string | null>(null);
  nodeStatusMap = $state<Map<string, NodeStatus>>(new Map());
  searchResults = $state<GraphNode[]>([]);
  searchQuery = $state('');
  searchVisible = $state(false);
  searchIndex = $state(0);
  loading = $state(false);

  get selectedNode(): GraphNode | null {
    if (!this.graphData || !this.selectedNodeId) return null;
    return this.graphData.nodes.find(n => n.id === this.selectedNodeId) ?? null;
  }

  get selectedNodeStatus(): NodeStatus | null {
    if (!this.selectedNodeId) return null;
    return this.nodeStatusMap.get(this.selectedNodeId) ?? null;
  }

  async fetchGraph(sessionId: string): Promise<void> {
    this.loading = true;
    try {
      const res = await fetch(`/api/sessions/${sessionId}/graph`);
      if (!res.ok) throw new Error(`Failed to fetch graph: ${res.statusText}`);
      this.graphData = await res.json();
    } catch (e: any) {
      console.error('Failed to load graph:', e);
    } finally {
      this.loading = false;
    }
  }

  selectNode(nodeId: string | null): void {
    this.selectedNodeId = nodeId;
  }

  updateNodeStatus(nodeId: string, status: NodeStatus): void {
    const newMap = new Map(this.nodeStatusMap);
    newMap.set(nodeId, status);
    this.nodeStatusMap = newMap;
  }

  async searchNodes(sessionId: string, query: string): Promise<void> {
    this.searchQuery = query;
    if (!query.trim()) {
      this.searchResults = [];
      return;
    }
    try {
      const res = await fetch(`/api/sessions/${sessionId}/graph/search?q=${encodeURIComponent(query)}`);
      if (res.ok) {
        this.searchResults = await res.json();
        this.searchIndex = 0;
      }
    } catch (e) {
      console.error('Search failed:', e);
    }
  }

  cycleSearchResult(direction: 1 | -1 = 1): GraphNode | null {
    if (this.searchResults.length === 0) return null;
    this.searchIndex = (this.searchIndex + direction + this.searchResults.length) % this.searchResults.length;
    const node = this.searchResults[this.searchIndex];
    this.selectedNodeId = node.id;
    return node;
  }
}

export const graphStore = new GraphStore();
