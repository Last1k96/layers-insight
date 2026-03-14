import type { InferenceTask, TaskStatus } from './types';
import { graphStore } from './graph.svelte';

class QueueStore {
  tasks = $state<InferenceTask[]>([]);
  selectedTaskId = $state<string | null>(null);
  filterText = $state('');
  filterStatus = $state<TaskStatus | 'all'>('all');

  get selectedIndex(): number {
    if (!this.selectedTaskId) return -1;
    return this.filteredTasks.findIndex(t => t.task_id === this.selectedTaskId);
  }

  set selectedIndex(idx: number) {
    const tasks = this.filteredTasks;
    this.selectedTaskId = idx >= 0 && idx < tasks.length ? tasks[idx].task_id : null;
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

  get sortedTasks(): InferenceTask[] {
    const nodes = graphStore.graphData?.nodes;
    const orderMap = new Map<string, number>();
    if (nodes) {
      nodes.forEach((n, i) => orderMap.set(n.id, i));
    }

    const done = this.tasks
      .filter(t => t.status === 'success' || t.status === 'failed')
      .sort((a, b) => (orderMap.get(a.node_id) ?? 0) - (orderMap.get(b.node_id) ?? 0));

    const executing = this.tasks.filter(t => t.status === 'executing');
    const waiting = this.tasks.filter(t => t.status === 'waiting');

    return [...done, ...executing, ...waiting];
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
