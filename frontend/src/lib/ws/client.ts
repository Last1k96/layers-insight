import { graphStore, type NodeStatus } from '../stores/graph.svelte';
import { queueStore } from '../stores/queue.svelte';
import { bisectStore } from '../stores/bisect.svelte';
import { cacheMetrics } from '../stores/metrics.svelte';
import { logStore } from '../stores/log.svelte';
import { refreshRenderer } from '../graph/renderer';
import type { TaskStatusMessage, TaskStatus } from '../stores/types';

let ws: WebSocket | null = null;
let sessionId: string | null = null;
let reconnectAttempts = 0;
let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
let onDisconnect: (() => void) | null = null;
let onReconnect: (() => void) | null = null;

export function setConnectionCallbacks(
  disconnectCb: () => void,
  reconnectCb: () => void
): void {
  onDisconnect = disconnectCb;
  onReconnect = reconnectCb;
}

export function connect(sid: string): void {
  sessionId = sid;
  reconnectAttempts = 0;
  _connect();
}

export function disconnect(): void {
  sessionId = null;
  if (reconnectTimeout) {
    clearTimeout(reconnectTimeout);
    reconnectTimeout = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
}

export function sendMessage(data: Record<string, unknown>): void {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  }
}

function _connect(): void {
  if (!sessionId) return;

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  ws = new WebSocket(`${protocol}//${host}/ws/${sessionId}`);

  ws.onopen = () => {
    reconnectAttempts = 0;
    if (onReconnect) onReconnect();
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    } catch (e) {
      console.error('WS message parse error:', e);
    }
  };

  ws.onclose = () => {
    if (sessionId) {
      if (onDisconnect) onDisconnect();
      _scheduleReconnect();
    }
  };

  ws.onerror = () => {
    // onclose will fire after this
  };
}

function _scheduleReconnect(): void {
  if (!sessionId) return;
  const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
  reconnectAttempts++;
  reconnectTimeout = setTimeout(_connect, delay);
}

function handleMessage(msg: any): void {
  if (msg.type === 'sub_session_created') {
    // Cancel any waiting tasks from the previous context
    for (const t of queueStore.tasks) {
      if (t.status === 'waiting') queueStore.cancel(t.task_id);
    }
    graphStore.setGrayedNodes(msg.grayed_nodes || [], msg.cut_node, msg.cut_type, msg.ancestor_cuts);
    graphStore.setActiveSubSession(msg.sub_session_id || null);
    graphStore.subSessionVersion++;
    refreshRenderer();
  }

  if (msg.type === 'task_status') {
    const tsMsg = msg as TaskStatusMessage;

    // Update queue store
    queueStore.updateTask(tsMsg.task_id, {
      status: tsMsg.status,
      stage: tsMsg.stage,
      error_detail: tsMsg.error_detail,
      metrics: tsMsg.metrics,
      main_result: tsMsg.main_result,
      ref_result: tsMsg.ref_result,
      per_output_metrics: tsMsg.per_output_metrics,
      per_output_main_results: tsMsg.per_output_main_results,
      per_output_ref_results: tsMsg.per_output_ref_results,
      sub_session_id: tsMsg.sub_session_id,
    });

    // Update graph node status
    const nodeStatus: NodeStatus = {
      status: tsMsg.status,
      taskId: tsMsg.task_id,
      stage: tsMsg.stage,
      metrics: tsMsg.metrics,
      mainResult: tsMsg.main_result,
      refResult: tsMsg.ref_result,
      perOutputMetrics: tsMsg.per_output_metrics,
      perOutputMainResults: tsMsg.per_output_main_results,
      perOutputRefResults: tsMsg.per_output_ref_results,
      errorDetail: tsMsg.error_detail,
    };
    graphStore.updateNodeStatus(tsMsg.node_id, nodeStatus, tsMsg.sub_session_id);
    refreshRenderer();

    // Cache metrics on success
    if (tsMsg.status === 'success' && tsMsg.metrics) {
      cacheMetrics(tsMsg.task_id, {
        metrics: tsMsg.metrics,
        main_result: tsMsg.main_result,
        ref_result: tsMsg.ref_result,
      });
    }
  }

  if (msg.type === 'task_deleted') {
    queueStore.removeTask(msg.task_id);
    // Find the node_id from graph data by node_name
    const node = graphStore.graphData?.nodes.find(n => n.name === msg.node_name);
    if (node) {
      graphStore.removeNodeStatus(node.id, graphStore.activeSubSessionId);
      refreshRenderer();
    }
  }

  if (msg.type === 'inference_log') {
    logStore.addEntry({
      task_id: msg.task_id,
      node_name: msg.node_name,
      level: msg.level,
      message: msg.message,
      timestamp: msg.timestamp,
    });
  }

  if (msg.type === 'bisect_progress') {
    bisectStore.handleWsMessage(msg);
    // When bisection completes, auto-select the found node
    if (msg.status === 'done' && msg.found_node) {
      const node = graphStore.graphData?.nodes.find(n => n.name === msg.found_node);
      if (node) {
        graphStore.selectNode(node.id);
      }
    }
  }
}
