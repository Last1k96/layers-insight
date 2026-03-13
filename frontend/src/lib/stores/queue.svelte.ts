import type { InferenceTask, TaskStatus } from './types';
import { graphStore } from './graph.svelte';

class QueueStore {
  tasks = $state<InferenceTask[]>([]);
  selectedIndex = $state(-1);
  filterText = $state('');
  filterStatus = $state<TaskStatus | 'all'>('all');

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

  async enqueue(sessionId: string, nodeId: string, nodeName: string, nodeType: string): Promise<InferenceTask | null> {
    try {
      const res = await fetch('/api/inference/enqueue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          node_id: nodeId,
          node_name: nodeName,
          node_type: nodeType,
        }),
      });
      if (!res.ok) throw new Error(`Enqueue failed: ${res.statusText}`);
      const task: InferenceTask = await res.json();
      this.addTask(task);
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
    this.selectedIndex = -1;
  }

  clear(): void {
    this.tasks = [];
    this.selectedIndex = -1;
  }

  selectByNodeId(nodeId: string | null): void {
    if (!nodeId) { this.selectedIndex = -1; return; }
    const idx = this.filteredTasks.findIndex(t => t.node_id === nodeId);
    this.selectedIndex = idx;
  }

  moveSelection(direction: 1 | -1): InferenceTask | null {
    const tasks = this.filteredTasks;
    if (tasks.length === 0) return null;
    this.selectedIndex = Math.max(0, Math.min(tasks.length - 1, this.selectedIndex + direction));
    return tasks[this.selectedIndex];
  }
}

export const queueStore = new QueueStore();
