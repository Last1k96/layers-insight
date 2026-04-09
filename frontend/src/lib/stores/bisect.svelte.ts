import type { BisectQueueItem, BisectJobStatus } from './types';
import { queueStore } from './queue.svelte';
import { graphStore } from './graph.svelte';

export type BisectSearchFor = 'accuracy_drop' | 'compilation_failure';
export type BisectMetric = 'cosine_similarity' | 'mse' | 'max_abs_diff';

class BisectStore {
  // Config state (for popup)
  searchFor = $state<BisectSearchFor>('accuracy_drop');
  metric = $state<BisectMetric>('cosine_similarity');
  threshold = $state(0.999);
  panelOpen = $state(false);
  error = $state<string | null>(null);

  // All tracked bisect jobs
  jobs = $state<BisectQueueItem[]>([]);

  /**
   * True while an async transition is in flight (start, stop, merge, discard).
   * All bisect action buttons should be disabled when this is true.
   */
  busy = $state(false);

  /** Jobs matching the currently active sub-session. */
  get visibleJobs(): BisectQueueItem[] {
    const activeSubId = graphStore.activeSubSessionId ?? null;
    return this.jobs.filter(j => (j.sub_session_id ?? null) === activeSubId);
  }

  get activeJobs(): BisectQueueItem[] {
    return this.visibleJobs.filter(j => j.status === 'running' || j.status === 'paused');
  }

  get finishedJobs(): BisectQueueItem[] {
    return this.visibleJobs.filter(j => j.status === 'done' || j.status === 'error' || j.status === 'stopped');
  }

  get isActive(): boolean {
    return this.activeJobs.length > 0;
  }

  get hasJobs(): boolean {
    return this.visibleJobs.length > 0;
  }

  async start(sessionId: string, subSessionId?: string | null, endNode?: string | null): Promise<boolean> {
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
      if (endNode) body.end_node = endNode;

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
      this.jobs = [...this.jobs, data];
      this.error = null;
      return true;
    } catch (e: any) {
      this.error = e.message;
      return false;
    } finally {
      this.busy = false;
    }
  }

  async startAllOutputs(sessionId: string, subSessionId?: string | null): Promise<boolean> {
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

      const res = await fetch('/api/inference/bisect/auto', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        this.error = err.detail || 'Failed to start per-output bisection';
        return false;
      }
      const data: BisectQueueItem[] = await res.json();
      this.jobs = [...this.jobs, ...data];
      this.error = null;
      return true;
    } catch (e: any) {
      this.error = e.message;
      return false;
    } finally {
      this.busy = false;
    }
  }

  async stop(jobId: string): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      const res = await fetch(`/api/inference/bisect/${jobId}/stop`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        this._updateJob(jobId, { status: (data.status as BisectJobStatus) || 'stopped' });
      }
    } catch (e) {
      console.error('Bisect stop failed:', e);
    } finally {
      this.busy = false;
    }
  }

  /** Merge a specific bisect job's child tasks into the main task list. */
  merge(jobId: string): void {
    const job = this.jobs.find(j => j.job_id === jobId);
    const sessionId = job?.session_id;
    queueStore.mergeBisectTasks(jobId);
    this.jobs = this.jobs.filter(j => j.job_id !== jobId);
    if (sessionId) this._mergePersistedJob(jobId, sessionId);
  }

  /** Stop bisect, cancel in-flight tasks, merge completed ones. */
  async stopAndMerge(jobId: string): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      const res = await fetch(`/api/inference/bisect/${jobId}/stop`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        this._updateJob(jobId, { status: (data.status as BisectJobStatus) || 'stopped' });
      }
      await this._cancelInFlightTasks(jobId);
      this.merge(jobId);
    } catch (e) {
      console.error('Bisect stopAndMerge failed:', e);
    } finally {
      this.busy = false;
    }
  }

  /** Stop bisect and delete all its tasks. */
  async stopAndDiscard(jobId: string): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      const res = await fetch(`/api/inference/bisect/${jobId}/stop`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        this._updateJob(jobId, { status: (data.status as BisectJobStatus) || 'stopped' });
      }
      await this._discardTasks(jobId);
    } catch (e) {
      console.error('Bisect stopAndDiscard failed:', e);
    } finally {
      this.busy = false;
    }
  }

  /** Discard a finished bisect job's tasks. */
  async discard(jobId: string): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      await this._discardTasks(jobId);
    } finally {
      this.busy = false;
    }
  }

  private async _cancelInFlightTasks(jobId: string): Promise<void> {
    const batchId = `bisect:${jobId}`;
    const inFlight = queueStore.tasks.filter(
      t => t.batch_id === batchId && (t.status === 'waiting' || t.status === 'executing')
    );
    await Promise.all(inFlight.map(t => queueStore.deleteTask(t.task_id)));
  }

  private async _discardTasks(jobId: string): Promise<void> {
    const job = this.jobs.find(j => j.job_id === jobId);
    const sessionId = job?.session_id;
    const batchId = `bisect:${jobId}`;
    const tasks = queueStore.tasks.filter(t => t.batch_id === batchId);
    await Promise.all(tasks.map(t => queueStore.deleteTask(t.task_id)));
    this.jobs = this.jobs.filter(j => j.job_id !== jobId);
    if (sessionId) this._clearPersistedJob(jobId, sessionId);
  }

  private _mergePersistedJob(jobId: string, sessionId: string): void {
    fetch(`/api/inference/bisect/${jobId}/merge?session_id=${sessionId}`, { method: 'POST' }).catch(() => {});
  }

  private _clearPersistedJob(jobId: string, sessionId: string): void {
    fetch(`/api/inference/bisect/${jobId}?session_id=${sessionId}`, { method: 'DELETE' }).catch(() => {});
  }

  /** Fetch all bisect jobs from backend (for page reload recovery). */
  async fetchStatus(sessionId?: string): Promise<void> {
    try {
      const url = sessionId
        ? `/api/inference/bisect/status?session_id=${sessionId}`
        : '/api/inference/bisect/status';
      const res = await fetch(url);
      if (!res.ok) return;
      const data = await res.json();
      const jobList: any[] = data.jobs || [];
      const restored: BisectQueueItem[] = [];
      for (const j of jobList) {
        if (j.status && j.status !== 'idle' && j.job_id) {
          restored.push({
            job_id: j.job_id,
            session_id: j.session_id || '',
            status: j.status,
            search_for: j.search_for || 'accuracy_drop',
            metric: j.metric || 'cosine_similarity',
            threshold: j.threshold ?? 0.999,
            step: j.step ?? 0,
            total_steps: j.total_steps ?? 0,
            current_node: j.current_node,
            found_node: j.found_node,
            error: j.error,
            sub_session_id: j.sub_session_id,
            output_node: j.output_node,
          });
        }
      }
      this.jobs = restored;
      // Migrate legacy batch_id="bisect" tasks to new "bisect:{jobId}" format
      if (restored.length > 0) {
        const legacyTasks = queueStore.tasks.filter(t => t.batch_id === 'bisect');
        if (legacyTasks.length > 0 && restored.length === 1) {
          const newBatchId = `bisect:${restored[0].job_id}`;
          queueStore.tasks = queueStore.tasks.map(t =>
            t.batch_id === 'bisect' ? { ...t, batch_id: newBatchId } : t
          );
        }
      }
    } catch {
      // Best effort
    }
  }

  handleWsMessage(msg: any): void {
    if (msg.type !== 'bisect_job_status') return;
    if (this.busy) return;

    const jobId = msg.job_id;
    if (!jobId) return;

    const existing = this.jobs.find(j => j.job_id === jobId);
    if (!existing) {
      // New job (e.g. started from another tab)
      this.jobs = [...this.jobs, {
        job_id: jobId,
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
        output_node: msg.output_node,
      }];
      return;
    }

    // Update existing job in-place
    if (msg.status) existing.status = msg.status as BisectJobStatus;
    if (msg.step !== undefined) existing.step = msg.step;
    if (msg.total_steps !== undefined) existing.total_steps = msg.total_steps;
    if (msg.current_node !== undefined) existing.current_node = msg.current_node;
    if (msg.found_node) existing.found_node = msg.found_node;
    if (msg.error) existing.error = msg.error;
  }

  private _updateJob(jobId: string, updates: Partial<BisectQueueItem>): void {
    const job = this.jobs.find(j => j.job_id === jobId);
    if (job) Object.assign(job, updates);
  }

  reset(): void {
    this.jobs = [];
    this.error = null;
    this.busy = false;
  }
}

export const bisectStore = new BisectStore();
