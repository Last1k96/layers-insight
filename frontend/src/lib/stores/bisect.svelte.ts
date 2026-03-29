export type BisectStatus = 'idle' | 'running' | 'done' | 'stopped' | 'error';
export type BisectSearchFor = 'accuracy_drop' | 'compilation_failure';
export type BisectMetric = 'cosine_similarity' | 'mse' | 'max_abs_diff';

export interface BisectStepInfo {
  node_name: string;
  node_id: string;
  task_id?: string;
  metric_value?: number;
  passed?: boolean;
  error?: string;
}

export interface BisectProgress {
  status: BisectStatus;
  session_id?: string;
  search_for?: BisectSearchFor;
  metric?: BisectMetric;
  threshold?: number;
  range_start?: string;
  range_end?: string;
  current_node?: string;
  step: number;
  total_steps: number;
  steps_history: BisectStepInfo[];
  found_node?: string;
  error?: string;
}

class BisectStore {
  status = $state<BisectStatus>('idle');
  searchFor = $state<BisectSearchFor>('accuracy_drop');
  metric = $state<BisectMetric>('cosine_similarity');
  threshold = $state(0.999);
  rangeStart = $state<string | null>(null);
  rangeEnd = $state<string | null>(null);
  currentNode = $state<string | null>(null);
  step = $state(0);
  totalSteps = $state(0);
  stepsHistory = $state<BisectStepInfo[]>([]);
  foundNode = $state<string | null>(null);
  error = $state<string | null>(null);
  panelOpen = $state(false);

  get isRunning(): boolean {
    return this.status === 'running';
  }

  get isDone(): boolean {
    return this.status === 'done';
  }

  async start(sessionId: string, subSessionId?: string | null): Promise<boolean> {
    try {
      const body: Record<string, any> = {
        session_id: sessionId,
        metric: this.metric,
        threshold: this.threshold,
        search_for: this.searchFor,
      };
      if (this.rangeStart) body.start_node = this.rangeStart;
      if (this.rangeEnd) body.end_node = this.rangeEnd;
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
      const data: BisectProgress = await res.json();
      this._applyProgress(data);
      return true;
    } catch (e: any) {
      this.error = e.message;
      return false;
    }
  }

  async stop(): Promise<void> {
    try {
      const res = await fetch('/api/inference/bisect/stop', { method: 'POST' });
      if (res.ok) {
        const data: BisectProgress = await res.json();
        this._applyProgress(data);
      }
    } catch (e) {
      console.error('Bisect stop failed:', e);
    }
  }

  async fetchStatus(): Promise<void> {
    try {
      const res = await fetch('/api/inference/bisect/status');
      if (res.ok) {
        const data: BisectProgress = await res.json();
        this._applyProgress(data);
      }
    } catch (e) {
      console.error('Bisect status fetch failed:', e);
    }
  }

  handleWsMessage(msg: any): void {
    if (msg.type !== 'bisect_progress') return;
    this.status = msg.status || this.status;
    if (msg.range_start) this.rangeStart = msg.range_start;
    if (msg.range_end) this.rangeEnd = msg.range_end;
    if (msg.current_node) this.currentNode = msg.current_node;
    if (msg.step !== undefined) this.step = msg.step;
    if (msg.total_steps !== undefined) this.totalSteps = msg.total_steps;
    if (msg.found_node) this.foundNode = msg.found_node;
    if (msg.error) this.error = msg.error;
  }

  reset(): void {
    this.status = 'idle';
    this.rangeStart = null;
    this.rangeEnd = null;
    this.currentNode = null;
    this.step = 0;
    this.totalSteps = 0;
    this.stepsHistory = [];
    this.foundNode = null;
    this.error = null;
  }

  private _applyProgress(data: BisectProgress): void {
    this.status = data.status;
    this.rangeStart = data.range_start ?? null;
    this.rangeEnd = data.range_end ?? null;
    this.currentNode = data.current_node ?? null;
    this.step = data.step;
    this.totalSteps = data.total_steps;
    this.stepsHistory = data.steps_history || [];
    this.foundNode = data.found_node ?? null;
    this.error = data.error ?? null;
  }
}

export const bisectStore = new BisectStore();
