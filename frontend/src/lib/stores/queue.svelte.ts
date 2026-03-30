import type { InferenceTask, TaskStatus } from './types';
import { graphStore } from './graph.svelte';

export type SortColumn = 'topo' | 'type' | 'cosine' | 'mse';
export type SortDirection = 'asc' | 'desc';

class QueueStore {
  /** All tasks across all sub-sessions. */
  tasks = $state<InferenceTask[]>([]);
  selectedTaskId = $state<string | null>(null);
  filterText = $state('');
  filterStatus = $state<TaskStatus | 'all'>('all');

  /** Sorting state */
  sortColumn = $state<SortColumn>('topo');
  sortDirection = $state<SortDirection>('asc');

  /** Queue pause state */
  paused = $state(false);

  /** Tasks visible for the current active sub-session. */
  get visibleTasks(): InferenceTask[] {
    const activeSubId = graphStore.activeSubSessionId;
    return this.tasks.filter(t => (t.sub_session_id ?? null) === activeSubId);
  }

  get selectedIndex(): number {
    if (!this.selectedTaskId) return -1;
    return this.filteredTasks.findIndex(t => t.task_id === this.selectedTaskId);
  }

  set selectedIndex(idx: number) {
    const tasks = this.filteredTasks;
    this.selectedTaskId = idx >= 0 && idx < tasks.length ? tasks[idx].task_id : null;
  }

  /** Topo index map: node_id -> index in graphStore.graphData.nodes */
  get topoIndexMap(): Map<string, number> {
    const nodes = graphStore.graphData?.nodes;
    const map = new Map<string, number>();
    if (nodes) {
      nodes.forEach((n, i) => map.set(n.id, i));
    }
    return map;
  }

  get filteredTasks(): InferenceTask[] {
    let result = this.sortedTasks;
    if (this.filterText) {
      const q = this.filterText.toLowerCase();
      result = result.filter(t =>
        t.node_name.toLowerCase().includes(q) || t.node_type.toLowerCase().includes(q)
      );
    }
    if (this.filterStatus !== 'all') {
      result = result.filter(t => t.status === this.filterStatus);
    }
    return result;
  }

  /** Returns the number of waiting tasks */
  get waitingCount(): number {
    return this.visibleTasks.filter(t => t.status === 'waiting').length;
  }

  get sortedTasks(): InferenceTask[] {
    const topoMap = this.topoIndexMap;
    const visible = this.visibleTasks;

    const done = visible.filter(t => t.status === 'success' || t.status === 'failed');
    const executing = visible.filter(t => t.status === 'executing');
    const waiting = visible.filter(t => t.status === 'waiting');

    const col = this.sortColumn;
    const dir = this.sortDirection;
    const mul = dir === 'asc' ? 1 : -1;

    done.sort((a, b) => {
      if (col === 'topo') {
        return ((topoMap.get(a.node_id) ?? 0) - (topoMap.get(b.node_id) ?? 0)) * mul;
      } else if (col === 'type') {
        const cmp = a.node_type.localeCompare(b.node_type) * mul;
        if (cmp !== 0) return cmp;
      } else if (col === 'cosine') {
        const av = a.status === 'success' && a.metrics ? a.metrics.cosine_similarity : (dir === 'asc' ? Infinity : -Infinity);
        const bv = b.status === 'success' && b.metrics ? b.metrics.cosine_similarity : (dir === 'asc' ? Infinity : -Infinity);
        const cmp = (av - bv) * mul;
        if (cmp !== 0) return cmp;
      } else if (col === 'mse') {
        const av = a.status === 'success' && a.metrics ? a.metrics.mse : (dir === 'asc' ? -Infinity : Infinity);
        const bv = b.status === 'success' && b.metrics ? b.metrics.mse : (dir === 'asc' ? -Infinity : Infinity);
        const cmp = (av - bv) * mul;
        if (cmp !== 0) return cmp;
      }
      // Default/fallback: topological order
      return (topoMap.get(a.node_id) ?? 0) - (topoMap.get(b.node_id) ?? 0);
    });

    return [...done, ...executing, ...waiting];
  }

  /** Toggle sort column. If same column, flip direction. Otherwise set column with its default direction. */
  toggleSort(column: SortColumn): void {
    if (this.sortColumn === column) {
      this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      this.sortColumn = column;
      // Default sort directions: cosine asc (worst first), mse desc (worst first), topo/type asc
      if (column === 'cosine') this.sortDirection = 'asc';
      else if (column === 'mse') this.sortDirection = 'desc';
      else this.sortDirection = 'asc';
    }
  }


  addTask(task: InferenceTask): void {
    const existing = this.tasks.findIndex(t => t.task_id === task.task_id);
    if (existing >= 0) {
      this.tasks = this.tasks.map((t, i) => i === existing ? task : t);
    } else {
      this.tasks = [...this.tasks, task];
    }
  }

  updateTask(taskId: string, updates: Partial<InferenceTask>): void {
    this.tasks = this.tasks.map(t =>
      t.task_id === taskId ? { ...t, ...updates } : t
    );
  }

  removeTask(taskId: string): void {
    this.tasks = this.tasks.filter(t => t.task_id !== taskId);
  }

  async enqueue(sessionId: string, nodeId: string, nodeName: string, nodeType: string, subSessionId?: string | null): Promise<InferenceTask | null> {
    try {
      const body: Record<string, string> = {
        session_id: sessionId,
        node_id: nodeId,
        node_name: nodeName,
        node_type: nodeType,
      };
      if (subSessionId) body.sub_session_id = subSessionId;
      const res = await fetch('/api/inference/enqueue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`Enqueue failed: ${res.statusText}`);
      const task: InferenceTask = await res.json();
      this.addTask(task);
      this.selectedTaskId = task.task_id;
      return task;
    } catch (e) {
      console.error('Enqueue failed:', e);
      return null;
    }
  }

  async rerun(taskId: string): Promise<void> {
    try {
      const res = await fetch(`/api/inference/${taskId}/rerun`, { method: 'POST' });
      if (res.ok) {
        const task: InferenceTask = await res.json();
        this.addTask(task);
      }
    } catch (e) {
      console.error('Rerun failed:', e);
    }
  }

  async cancel(taskId: string): Promise<void> {
    try {
      await fetch(`/api/inference/${taskId}`, { method: 'DELETE' });
    } catch (e) {
      console.error('Cancel failed:', e);
    }
  }

  async deleteTask(taskId: string): Promise<void> {
    try {
      const task = this.tasks.find(t => t.task_id === taskId);
      const sessionParam = task?.session_id ? `?session_id=${task.session_id}` : '';
      const res = await fetch(`/api/inference/${taskId}${sessionParam}`, { method: 'DELETE' });
      if (res.ok) {
        this.removeTask(taskId);
      }
    } catch (e) {
      console.error('Delete task failed:', e);
    }
  }

  loadTasks(tasks: InferenceTask[]): void {
    this.tasks = tasks;
    this.selectedTaskId = null;
  }

  removeByNodeNames(names: Set<string>): void {
    for (const t of this.tasks) {
      if (names.has(t.node_name) && t.status === 'waiting') {
        this.cancel(t.task_id);
      }
    }
    this.tasks = this.tasks.filter(t => !names.has(t.node_name));
  }

  async pauseQueue(): Promise<void> {
    try {
      const res = await fetch('/api/inference/pause', { method: 'POST' });
      if (res.ok) {
        this.paused = true;
      }
    } catch (e) {
      console.error('Pause failed:', e);
    }
  }

  async resumeQueue(): Promise<void> {
    try {
      const res = await fetch('/api/inference/resume', { method: 'POST' });
      if (res.ok) {
        this.paused = false;
      }
    } catch (e) {
      console.error('Resume failed:', e);
    }
  }

  async cancelAll(): Promise<void> {
    try {
      const res = await fetch('/api/inference/cancel-all', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        // Update local tasks that were cancelled
        this.tasks = this.tasks.map(t =>
          t.status === 'waiting' ? { ...t, status: 'failed' as TaskStatus, error_detail: 'Cancelled' } : t
        );
      }
    } catch (e) {
      console.error('Cancel all failed:', e);
    }
  }

  async fetchQueueState(): Promise<void> {
    try {
      const res = await fetch('/api/inference/queue-state');
      if (res.ok) {
        const data = await res.json();
        this.paused = data.paused;
      }
    } catch (e) {
      console.error('Fetch queue state failed:', e);
    }
  }

  clear(): void {
    this.tasks = [];
    this.selectedTaskId = null;
  }

  selectByNodeId(nodeId: string | null): void {
    if (!nodeId) { this.selectedTaskId = null; return; }
    const task = this.filteredTasks.find(t => t.node_id === nodeId);
    this.selectedTaskId = task?.task_id ?? null;
  }

  moveSelection(direction: 1 | -1): InferenceTask | null {
    const tasks = this.filteredTasks;
    if (tasks.length === 0) return null;
    const currentIdx = this.selectedTaskId
      ? tasks.findIndex(t => t.task_id === this.selectedTaskId)
      : -1;
    const newIdx = Math.max(0, Math.min(tasks.length - 1, currentIdx + direction));
    this.selectedTaskId = tasks[newIdx].task_id;
    return tasks[newIdx];
  }
}

export const queueStore = new QueueStore();
