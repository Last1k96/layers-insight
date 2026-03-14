import type { GraphData, GraphNode, TaskStatus, AccuracyMetrics, DeviceResult } from './types';
import { queueStore } from './queue.svelte';

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
  /** Per-sub-session node status maps. Key: sub_session_id or null for root. */
  private _nodeStatusMaps = $state<Map<string | null, Map<string, NodeStatus>>>(new Map([[null, new Map()]]));
  searchResults = $state<GraphNode[]>([]);
  searchQuery = $state('');
  searchVisible = $state(false);
  searchIndex = $state(0);
  loading = $state(false);
  grayedNodes = $state<Set<string>>(new Set());
  activeSubSessionId = $state<string | null>(null);
  /** Display overrides for nodes (e.g. cut node shown as Parameter) */
  nodeOverrides = $state<Map<string, { name: string; type: string; color: string }>>(new Map());

  /** Returns the nodeStatusMap for the active sub-session. */
  get nodeStatusMap(): Map<string, NodeStatus> {
    return this._nodeStatusMaps.get(this.activeSubSessionId) ?? new Map();
  }

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
    queueStore.selectByNodeId(nodeId);
  }

  updateNodeStatus(nodeId: string, status: NodeStatus, subSessionId?: string | null): void {
    const key = subSessionId ?? null;
    const newMaps = new Map(this._nodeStatusMaps);
    const existing = newMaps.get(key) ?? new Map();
    const newMap = new Map(existing);
    newMap.set(nodeId, status);
    newMaps.set(key, newMap);
    this._nodeStatusMaps = newMaps;
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

  setGrayedNodes(
    nodes: string[],
    cutNode?: string,
    cutType?: string,
    ancestorCuts?: { cut_node: string; cut_type: string }[],
  ): void {
    const s = new Set(nodes);
    const overrides = new Map<string, { name: string; type: string; color: string }>();

    // Build overrides for the current cut node
    if (cutType === 'input' && cutNode) {
      s.delete(cutNode);
      const origNode = this.graphData?.nodes.find(n => n.name === cutNode);
      const origType = origNode?.type ?? cutNode;
      overrides.set(cutNode, { name: `Parameter(${origType})`, type: `Parameter(${origType})`, color: '#eeeeee' });
    }

    // Build overrides for ancestor input-cut nodes that are still visible
    // (not grayed). If an ancestor cut node is in the grayed set, it means
    // it's fully upstream of the current cut and should stay grayed.
    if (ancestorCuts) {
      for (const ac of ancestorCuts) {
        if (ac.cut_type === 'input' && !s.has(ac.cut_node)) {
          const origNode = this.graphData?.nodes.find(n => n.name === ac.cut_node);
          const origType = origNode?.type ?? ac.cut_node;
          overrides.set(ac.cut_node, { name: `Parameter(${origType})`, type: `Parameter(${origType})`, color: '#eeeeee' });
        }
      }
    }

    this.nodeOverrides = overrides;
    this.grayedNodes = s;
  }

  clearGrayedNodes(): void {
    this.grayedNodes = new Set();
    this.nodeOverrides = new Map();
  }

  setActiveSubSession(id: string | null): void {
    this.activeSubSessionId = id;
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
