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

  /**
   * True while an async transition is in flight (start, stop, merge, discard).
   * All bisect action buttons should be disabled when this is true.
   */
  busy = $state(false);

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
    if (this.busy) return false;
    this.busy = true;
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
    } finally {
      this.busy = false;
    }
  }

  async stop(): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      const res = await fetch('/api/inference/bisect/stop', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        // Update job status synchronously from HTTP response — don't rely on WS.
        if (this.job) {
          this.job.status = (data.status as BisectJobStatus) || 'stopped';
        }
      }
    } catch (e) {
      console.error('Bisect stop failed:', e);
    } finally {
      this.busy = false;
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
    if (this.busy) return;
    this.busy = true;
    try {
      // 1. Stop the bisect on the backend (also cancels waiting bisect tasks server-side)
      const res = await fetch('/api/inference/bisect/stop', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (this.job) {
          this.job.status = (data.status as BisectJobStatus) || 'stopped';
        }
      }
      // 2. Cancel any in-flight tasks still visible on the frontend
      await this.cancelInFlightBisectTasks();
      // 3. Merge completed bisect tasks into the main list
      this.merge();
    } catch (e) {
      console.error('Bisect stopAndMerge failed:', e);
    } finally {
      this.busy = false;
    }
  }

  /** Stop bisect and delete all bisect tasks from frontend and backend. */
  async stopAndDiscard(): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      // 1. Stop the bisect on the backend (also cancels waiting bisect tasks server-side)
      const res = await fetch('/api/inference/bisect/stop', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (this.job) {
          this.job.status = (data.status as BisectJobStatus) || 'stopped';
        }
      }
      // 2. Discard all bisect tasks
      await this._discardTasks();
    } catch (e) {
      console.error('Bisect stopAndDiscard failed:', e);
    } finally {
      this.busy = false;
    }
  }

  /** Discard bisect tasks from both frontend and backend (no busy guard — called from guarded methods). */
  private async _discardTasks(): Promise<void> {
    const bisectTasks = queueStore.tasks.filter(t => t.batch_id === 'bisect');
    await Promise.all(bisectTasks.map(t => queueStore.deleteTask(t.task_id)));
    this.job = null;
    this.error = null;
  }

  /** Public discard — with busy guard. */
  async discard(): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      await this._discardTasks();
    } finally {
      this.busy = false;
    }
  }

  /** Fetch current bisect status from backend (for page reload recovery). */
  async fetchStatus(): Promise<void> {
    try {
      const res = await fetch('/api/inference/bisect/status');
      if (!res.ok) return;
      const data = await res.json();
      if (data.status && data.status !== 'idle' && data.job_id) {
        this.job = {
          job_id: data.job_id,
          session_id: data.session_id || '',
          status: data.status,
          search_for: data.search_for || 'accuracy_drop',
          metric: data.metric || 'cosine_similarity',
          threshold: data.threshold ?? 0.999,
          step: data.step ?? 0,
          total_steps: data.total_steps ?? 0,
          current_node: data.current_node,
          found_node: data.found_node,
          error: data.error,
          sub_session_id: data.sub_session_id,
        };
      }
    } catch {
      // Ignore — best effort
    }
  }

  handleWsMessage(msg: any): void {
    if (msg.type !== 'bisect_job_status') return;

    // Ignore WS updates while a transition is in flight — the HTTP response
    // already set the authoritative state and we don't want a stale WS
    // message to overwrite it.
    if (this.busy) return;

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
    this.busy = false;
  }
}

export const bisectStore = new BisectStore();
