import {
  FILTER_FIELD_META,
  type FilterConnector,
  type FilterField,
  type FilterOperator,
  type FilterRule,
  type GraphNode,
  type InferenceTask,
} from './types';

const STORAGE_KEY_PREFIX = 'layers-insight-advanced-filter:';
const storageKeyFor = (sessionId: string): string => `${STORAGE_KEY_PREFIX}${sessionId}`;

class AdvancedFilterStore {
  private _active = $state(false);
  rules = $state<FilterRule[]>([]);
  connectors = $state<FilterConnector[]>([]);
  /** The session whose filter state currently lives in this store. Null when
   *  no session is loaded — in that mode persist() is a no-op so the UI can
   *  still be inspected without leaking state to a global key. */
  private _sessionId: string | null = null;

  get active(): boolean { return this._active; }
  set active(v: boolean) { this._active = v; this.persist(); }

  private persist(): void {
    if (!this._sessionId) return;
    try {
      const snapshot = {
        active: this.active,
        rules: this.rules,
        connectors: this.connectors,
      };
      localStorage.setItem(storageKeyFor(this._sessionId), JSON.stringify(snapshot));
    } catch {
      // localStorage full or unavailable
    }
  }

  /** Switch the store to a different session: drops in-memory state and
   *  reloads from that session's localStorage slot (if any). Pass `null`
   *  when leaving the main view / no session is active. */
  loadForSession(sessionId: string | null): void {
    if (this._sessionId === sessionId) return;
    this._sessionId = sessionId;
    this._active = false;
    this.rules = [];
    this.connectors = [];
    if (!sessionId) return;
    try {
      const raw = localStorage.getItem(storageKeyFor(sessionId));
      if (!raw) return;
      const data = JSON.parse(raw);
      if (typeof data.active === 'boolean') this._active = data.active;
      if (Array.isArray(data.rules)) {
        this.rules = data.rules.filter(
          (r: any) => r.id && r.field && r.field in FILTER_FIELD_META
        );
      }
      if (Array.isArray(data.connectors)) {
        this.connectors = data.connectors
          .filter((c: any) => c === 'AND' || c === 'OR')
          .slice(0, Math.max(0, this.rules.length - 1));
      }
      while (this.connectors.length < Math.max(0, this.rules.length - 1)) {
        this.connectors.push('AND');
      }
    } catch {
      // Corrupted data, start fresh
    }
  }

  addRule(): void {
    if (this.rules.length >= 10) return;
    const rule: FilterRule = {
      id: crypto.randomUUID(),
      field: 'node_name',
      operator: 'contains',
      value: '',
    };
    if (this.rules.length > 0) {
      this.connectors = [...this.connectors, 'AND'];
    }
    this.rules = [...this.rules, rule];
    this.persist();
  }

  removeRule(id: string): void {
    const idx = this.rules.findIndex(r => r.id === id);
    if (idx < 0) return;
    const newRules = this.rules.filter(r => r.id !== id);
    const newConnectors = [...this.connectors];
    if (newConnectors.length > 0) {
      // Remove the connector adjacent to the removed rule
      // If removing first rule, remove first connector; otherwise remove connector before it
      const connIdx = idx === 0 ? 0 : idx - 1;
      newConnectors.splice(connIdx, 1);
    }
    this.rules = newRules;
    this.connectors = newConnectors;
    this.persist();
  }

  updateRule(id: string, updates: Partial<Omit<FilterRule, 'id'>>): void {
    this.rules = this.rules.map(r =>
      r.id === id ? { ...r, ...updates } : r
    );
    this.persist();
  }

  toggleConnector(index: number): void {
    if (index < 0 || index >= this.connectors.length) return;
    this.connectors = this.connectors.map((c, i) =>
      i === index ? (c === 'AND' ? 'OR' : 'AND') : c
    );
    this.persist();
  }

  reorderRules(fromIndex: number, toIndex: number): void {
    if (fromIndex === toIndex) return;
    const newRules = [...this.rules];
    const [moved] = newRules.splice(fromIndex, 1);
    newRules.splice(toIndex, 0, moved);
    this.rules = newRules;
    this.persist();
  }

  /** True when at least one rule has a non-empty value. */
  get hasActiveRules(): boolean {
    return this.rules.some(r => r.value !== '');
  }

  /** Split active rules into AND-groups separated by OR connectors.
   *  (AND binds tighter than OR.) Returns [] when no rules are active. */
  private buildGroups(): FilterRule[][] {
    const activeIndices = this.rules
      .map((r, i) => (r.value !== '' ? i : -1))
      .filter(i => i >= 0);
    if (activeIndices.length === 0) return [];

    const groups: FilterRule[][] = [[]];
    for (let ai = 0; ai < activeIndices.length; ai++) {
      const origIdx = activeIndices[ai];
      groups[groups.length - 1].push(this.rules[origIdx]);
      if (ai < activeIndices.length - 1) {
        const nextOrigIdx = activeIndices[ai + 1];
        let hasOr = false;
        for (let ci = origIdx; ci < nextOrigIdx; ci++) {
          if (ci < this.connectors.length && this.connectors[ci] === 'OR') {
            hasOr = true;
            break;
          }
        }
        if (hasOr) groups.push([]);
      }
    }
    return groups;
  }

  applyFilter(tasks: InferenceTask[]): InferenceTask[] {
    const groups = this.buildGroups();
    if (groups.length === 0) return tasks;
    return tasks.filter(task => groups.some(group => group.every(rule => this.evaluateRule(rule, task))));
  }

  /** Apply the active rules to graph nodes. Metric/status rules fall back to
   *  the corresponding task (if any) in `taskByNodeId`; nodes without a task
   *  evaluate those fields against `undefined`. */
  applyFilterToNodes(nodes: GraphNode[], taskByNodeId: Map<string, InferenceTask>): GraphNode[] {
    const groups = this.buildGroups();
    if (groups.length === 0) return nodes;
    return nodes.filter(node => {
      const task = taskByNodeId.get(node.id);
      const taskLike = {
        node_name: node.name,
        node_type: node.type,
        status: task?.status,
        metrics: task?.metrics,
      } as InferenceTask;
      return groups.some(group => group.every(rule => this.evaluateRule(rule, taskLike)));
    });
  }

  private evaluateRule(rule: FilterRule, task: InferenceTask): boolean {
    const meta = FILTER_FIELD_META[rule.field];
    const taskValue = this.getFieldValue(rule.field, task);

    if (meta.type === 'string') {
      const sv = String(taskValue ?? '').toLowerCase();
      const rv = rule.value.toLowerCase();
      switch (rule.operator) {
        case 'contains': return sv.includes(rv);
        case 'equals': return sv === rv;
        case '!=': return sv !== rv;
        default: return true;
      }
    }

    if (meta.type === 'enum') {
      const sv = String(taskValue ?? '');
      switch (rule.operator) {
        case '=': return sv === rule.value;
        case '!=': return sv !== rule.value;
        default: return true;
      }
    }

    if (meta.type === 'number') {
      if (taskValue === undefined || taskValue === null) return false;
      const nv = Number(taskValue);
      const rv = Number(rule.value);
      if (isNaN(nv) || isNaN(rv)) return false;
      switch (rule.operator) {
        case '>': return nv > rv;
        case '<': return nv < rv;
        case '=': return nv === rv;
        case '!=': return nv !== rv;
        case '>=': return nv >= rv;
        case '<=': return nv <= rv;
        default: return true;
      }
    }

    return true;
  }

  private getFieldValue(field: FilterField, task: InferenceTask): string | number | undefined {
    switch (field) {
      case 'node_name': return task.node_name;
      case 'node_type': return task.node_type;
      case 'status': return task.status;
      case 'cosine_similarity': return task.metrics?.cosine_similarity;
      case 'mse': return task.metrics?.mse;
      case 'max_abs_diff': return task.metrics?.max_abs_diff;
    }
  }
}

export const advancedFilterStore = new AdvancedFilterStore();
