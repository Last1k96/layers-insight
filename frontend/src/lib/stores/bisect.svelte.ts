import type { BisectQueueItem, BisectJobStatus } from './types';
import { queueStore } from './queue.svelte';

export type BisectSearchFor = 'accuracy_drop' | 'compilation_failure';
export type BisectMetric = 'cosine_similarity' | 'mse' | 'max_abs_diff';

class BisectStore {
  // Config state (for popup)
  searchFor = $state<BisectSearchFor>('accuracy_drop');
  metric = $state<BisectMetric>('cosine_similarity');
  threshold = $state(0.999);
  panelOpen = $state(false);
  error = $state<string | null>(null);

  // Active job (rendered in queue panel)
  job = $state<BisectQueueItem | null>(null);

  get isActive(): boolean {
    return this.job !== null && (this.job.status === 'running' || this.job.status === 'paused');
  }

  get isRunning(): boolean {
    return this.job !== null && this.job.status === 'running';
  }

  get isDone(): boolean {
    return this.job !== null && this.job.status === 'done';
  }

  async start(sessionId: string, subSessionId?: string | null): Promise<boolean> {
    try {
      const body: Record<string, any> = {
        session_id: sessionId,
        metric: this.metric,
        threshold: this.threshold,
        search_for: this.searchFor,
      };
      if (subSessionId) body.sub_session_id = subSessionId;

      const res = await fetch('/api/inference/bisect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        this.error = err.detail || 'Failed to start bisection';
        return false;
      }
      const data: BisectQueueItem = await res.json();
      this.job = data;
      this.error = null;
      return true;
    } catch (e: any) {
      this.error = e.message;
      return false;
    }
  }

  async stop(): Promise<void> {
    try {
      await fetch('/api/inference/bisect/stop', { method: 'POST' });
      // Keep job with 'stopped' status so UI can show merge/dismiss controls.
      // The WS handler will update the status.
    } catch (e) {
      console.error('Bisect stop failed:', e);
    }
  }

  /** Merge bisect child tasks into the main task list and dismiss. */
  merge(): void {
    queueStore.mergeBisectTasks();
    this.job = null;
    this.error = null;
  }

  /** Cancel any in-flight bisect tasks and remove them from the store. */
  private async cancelInFlightBisectTasks(): Promise<void> {
    const inFlight = queueStore.tasks.filter(
      t => t.batch_id === 'bisect' && (t.status === 'waiting' || t.status === 'executing')
    );
    await Promise.all(inFlight.map(t => queueStore.deleteTask(t.task_id)));
  }

  /** Stop bisect, cancel in-flight tasks, merge completed ones into the main list. */
  async stopAndMerge(): Promise<void> {
    await this.stop();
    await this.cancelInFlightBisectTasks();
    this.merge();
  }

  /** Stop bisect and delete all bisect tasks from frontend and backend. */
  async stopAndDiscard(): Promise<void> {
    await this.stop();
    await this.discard();
  }

  /** Discard bisect tasks from both frontend and backend. */
  async discard(): Promise<void> {
    const bisectTasks = queueStore.tasks.filter(t => t.batch_id === 'bisect');
    await Promise.all(bisectTasks.map(t => queueStore.deleteTask(t.task_id)));
    this.job = null;
    this.error = null;
  }

  handleWsMessage(msg: any): void {
    if (msg.type !== 'bisect_job_status') return;

    if (!this.job && msg.job_id) {
      // Job was started (e.g. on another tab) — create it
      this.job = {
        job_id: msg.job_id,
        session_id: msg.session_id || '',
        status: msg.status || 'running',
        search_for: msg.search_for || 'accuracy_drop',
        metric: msg.metric || 'cosine_similarity',
        threshold: msg.threshold ?? 0.999,
        step: msg.step ?? 0,
        total_steps: msg.total_steps ?? 0,
        current_node: msg.current_node,
        found_node: msg.found_node,
        error: msg.error,
        sub_session_id: msg.sub_session_id,
      };
      return;
    }

    if (this.job) {
      if (msg.status) this.job.status = msg.status as BisectJobStatus;
      if (msg.step !== undefined) this.job.step = msg.step;
      if (msg.total_steps !== undefined) this.job.total_steps = msg.total_steps;
      if (msg.current_node !== undefined) this.job.current_node = msg.current_node;
      if (msg.found_node) this.job.found_node = msg.found_node;
      if (msg.error) this.job.error = msg.error;

      // Keep job on 'stopped' so UI can show merge/dismiss controls
    }
  }

  reset(): void {
    this.job = null;
    this.error = null;
  }
}

export const bisectStore = new BisectStore();
