import type { GraphData, GraphNode, GraphEdge, TaskStatus, AccuracyMetrics, DeviceResult } from './types';
import { queueStore } from './queue.svelte';

export interface NodeStatus {
  status: TaskStatus;
  taskId?: string;
  stage?: string;
  metrics?: AccuracyMetrics;
  mainResult?: DeviceResult;
  refResult?: DeviceResult;
  perOutputMetrics?: AccuracyMetrics[];
  perOutputMainResults?: DeviceResult[];
  perOutputRefResults?: DeviceResult[];
  errorDetail?: string;
}

class GraphStore {
  /** The graph currently being rendered. Swaps between fullGraph and a
   *  sub-session's tight graph depending on view mode. */
  graphData = $state<GraphData | null>(null);
  /** The session's unmodified full graph — the source of truth for
   *  "full layout" mode, never mutated. */
  fullGraph = $state<GraphData | null>(null);
  selectedNodeId = $state<string | null>(null);
  /** Per-sub-session node status maps. Key: sub_session_id or null for root. */
  private _nodeStatusMaps = $state<Map<string | null, Map<string, NodeStatus>>>(new Map([[null, new Map()]]));
  searchResults = $state<GraphNode[]>([]);
  searchQuery = $state('');
  searchVisible = $state(false);
  searchIndex = $state(0);
  loading = $state(false);
  loadingStage = $state('');
  loadingDetail = $state('');
  /** Estimated layout duration in ms (from node count heuristic). */
  layoutEstimateMs = $state(0);
  /** Timestamp when layout phase started. */
  layoutStartedAt = $state(0);
  grayedNodes = $state<Set<string>>(new Set());
  activeSubSessionId = $state<string | null>(null);
  /** Display overrides for nodes (e.g. cut node shown as Parameter) */
  nodeOverrides = $state<Map<string, { name: string; type: string; color: string }>>(new Map());
  /** Bumped when a sub-session is created/deleted so watchers can re-fetch. */
  subSessionVersion = $state(0);
  /** Per-sub-session standalone tight graphs (complete GraphData objects). */
  subSessionTightGraphs = $state<Map<string, GraphData>>(new Map());
  /** True while Alt is held — switches to accuracy flow visualization. */
  accuracyViewActive = $state(false);
  /** Currently selected edge (index into graphData.edges[]). */
  selectedEdgeIndex = $state<number | null>(null);
  /** Bumped on camera changes so GhostNodes can recompute positions reactively. */
  cameraVersion = $state(0);
  /** Currently hovered node (from graph canvas or queue list). */
  hoveredNodeId = $state<string | null>(null);

  /** Returns the nodeStatusMap for the active sub-session. */
  get nodeStatusMap(): Map<string, NodeStatus> {
    return this._nodeStatusMaps.get(this.activeSubSessionId) ?? new Map();
  }

  get selectedNode(): GraphNode | null {
    if (!this.graphData || !this.selectedNodeId) return null;
    return this.graphData.nodes.find(n => n.id === this.selectedNodeId) ?? null;
  }

  get selectedEdge(): { edge: GraphEdge; sourceNode: GraphNode | null; targetNode: GraphNode | null } | null {
    if (!this.graphData || this.selectedEdgeIndex === null) return null;
    const edge = this.graphData.edges[this.selectedEdgeIndex];
    if (!edge) return null;
    const sourceNode = this.graphData.nodes.find(n => n.id === edge.source) ?? null;
    const targetNode = this.graphData.nodes.find(n => n.id === edge.target) ?? null;
    return { edge, sourceNode, targetNode };
  }

  get selectedNodeStatus(): NodeStatus | null {
    if (!this.selectedNodeId) return null;
    return this.nodeStatusMap.get(this.selectedNodeId) ?? null;
  }

  async fetchGraph(sessionId: string): Promise<void> {
    this.loading = true;
    this.selectedNodeId = null;
    this.selectedEdgeIndex = null;
    this.searchVisible = false;
    this.searchQuery = '';
    this.searchResults = [];
    try {
      const res = await fetch(`/api/sessions/${sessionId}/graph`);
      if (!res.ok) throw new Error(`Failed to fetch graph: ${res.statusText}`);
      const graph = await res.json() as GraphData;
      this.fullGraph = graph;
      this.graphData = graph;
    } catch (e: any) {
      console.error('Failed to load graph:', e);
    } finally {
      this.loading = false;
      this.loadingStage = '';
      this.loadingDetail = '';
    }
  }

  /** Swap rendered graph back to the session's full model. Callers are
   *  responsible for (re)applying grayed-node state if a sub-session is active. */
  activateFullGraph(): void {
    if (!this.fullGraph) return;
    this.graphData = this.fullGraph;
    this.selectedEdgeIndex = null;
    if (this.selectedNodeId && !this.fullGraph.nodes.find(n => n.id === this.selectedNodeId)) {
      this.selectedNodeId = null;
    }
  }

  /** Swap rendered graph to a sub-session's standalone tight graph. */
  activateTightGraph(subSessionId: string, graph: GraphData): void {
    this.setTightGraph(subSessionId, graph);
    this.graphData = graph;
    this.selectedEdgeIndex = null;
    if (this.selectedNodeId && !graph.nodes.find(n => n.id === this.selectedNodeId)) {
      this.selectedNodeId = null;
    }
    // Grayed-node dimming is meaningless in tight view: every node in the
    // scene is part of the sub-session's subgraph.
    this.grayedNodes = new Set();
  }

  selectNode(nodeId: string | null, syncQueue = true): void {
    this.selectedNodeId = nodeId;
    if (nodeId !== null) this.selectedEdgeIndex = null;
    if (syncQueue) queueStore.selectByNodeId(nodeId);
  }

  selectEdge(index: number | null): void {
    this.selectedEdgeIndex = index;
    if (index !== null) {
      this.selectedNodeId = null;
      queueStore.selectByNodeId(null);
    }
  }

  updateNodeStatus(nodeId: string, status: NodeStatus, subSessionId?: string | null): void {
    const key = subSessionId ?? null;
    const inner = new Map(this._nodeStatusMaps.get(key) ?? []);
    inner.set(nodeId, status);
    const outer = new Map(this._nodeStatusMaps);
    outer.set(key, inner);
    this._nodeStatusMaps = outer;
  }

  removeNodeStatus(nodeId: string, subSessionId?: string | null): void {
    const key = subSessionId ?? null;
    const old = this._nodeStatusMaps.get(key);
    if (!old || !old.has(nodeId)) return;
    const inner = new Map(old);
    inner.delete(nodeId);
    const outer = new Map(this._nodeStatusMaps);
    outer.set(key, inner);
    this._nodeStatusMaps = outer;
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
      overrides.set(cutNode, { name: 'Parameter', type: 'Parameter', color: '#eeeeee' });
    }

    // Ancestor input-cut nodes that are still reachable in the current sub-model
    // should be shown as Parameters. If they're in the grayed set from the backend,
    // it means they're upstream of the current cut and should stay grayed.
    if (ancestorCuts) {
      for (const ac of ancestorCuts) {
        if (ac.cut_type === 'input' && !s.has(ac.cut_node)) {
          overrides.set(ac.cut_node, { name: 'Parameter', type: 'Parameter', color: '#eeeeee' });
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

  setTightGraph(subSessionId: string, graph: GraphData): void {
    const next = new Map(this.subSessionTightGraphs);
    next.set(subSessionId, graph);
    this.subSessionTightGraphs = next;
  }

  removeTightGraph(subSessionId: string): void {
    if (!this.subSessionTightGraphs.has(subSessionId)) return;
    const next = new Map(this.subSessionTightGraphs);
    next.delete(subSessionId);
    this.subSessionTightGraphs = next;
  }

  getTightGraph(subSessionId: string | null): GraphData | null {
    if (!subSessionId) return null;
    return this.subSessionTightGraphs.get(subSessionId) ?? null;
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
