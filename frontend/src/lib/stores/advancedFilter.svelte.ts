import {
  FILTER_FIELD_META,
  type FilterConnector,
  type FilterField,
  type FilterOperator,
  type FilterRule,
  type InferenceTask,
} from './types';

const STORAGE_KEY = 'layers-insight-advanced-filter';

class AdvancedFilterStore {
  private _active = $state(false);
  rules = $state<FilterRule[]>([]);
  connectors = $state<FilterConnector[]>([]);

  get active(): boolean { return this._active; }
  set active(v: boolean) { this._active = v; this.persist(); }

  constructor() {
    this.restore();
  }

  private persist(): void {
    try {
      const snapshot = {
        active: this.active,
        rules: this.rules,
        connectors: this.connectors,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
    } catch {
      // localStorage full or unavailable
    }
  }

  private restore(): void {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const data = JSON.parse(raw);
      if (typeof data.active === 'boolean') this._active = data.active;
      if (Array.isArray(data.rules)) {
        // Validate rules have valid fields
        this.rules = data.rules.filter(
          (r: any) => r.id && r.field && r.field in FILTER_FIELD_META
        );
      }
      if (Array.isArray(data.connectors)) {
        this.connectors = data.connectors
          .filter((c: any) => c === 'AND' || c === 'OR')
          .slice(0, Math.max(0, this.rules.length - 1));
      }
      // Ensure connector count matches
      while (this.connectors.length < Math.max(0, this.rules.length - 1)) {
        this.connectors.push('AND');
      }
    } catch {
      // Corrupted data, start fresh
    }
  }

  addRule(): void {
    const rule: FilterRule = {
      id: crypto.randomUUID(),
      field: 'node_name',
      operator: 'contains',
      value: '',
    };
    if (this.rules.length > 0) {
      this.connectors = ['AND', ...this.connectors];
    }
    this.rules = [rule, ...this.rules];
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

  applyFilter(tasks: InferenceTask[]): InferenceTask[] {
    const activeRules = this.rules.filter(r => r.value !== '');
    if (activeRules.length === 0) return tasks;

    // Build groups: split by OR connectors. AND binds tighter.
    // Map active rules back to their original indices to get connectors
    const activeIndices = this.rules
      .map((r, i) => (r.value !== '' ? i : -1))
      .filter(i => i >= 0);

    // Build groups from active rules, using connectors between original indices
    const groups: FilterRule[][] = [[]];
    for (let ai = 0; ai < activeIndices.length; ai++) {
      const origIdx = activeIndices[ai];
      groups[groups.length - 1].push(this.rules[origIdx]);

      if (ai < activeIndices.length - 1) {
        // Find the connector between this active rule and the next
        // Use the connector at the position of the current original index
        // (connector[i] is between rules[i] and rules[i+1])
        const nextOrigIdx = activeIndices[ai + 1];
        // Check if there's an OR connector between origIdx and nextOrigIdx
        let hasOr = false;
        for (let ci = origIdx; ci < nextOrigIdx; ci++) {
          if (ci < this.connectors.length && this.connectors[ci] === 'OR') {
            hasOr = true;
            break;
          }
        }
        if (hasOr) {
          groups.push([]);
        }
      }
    }

    return tasks.filter(task => groups.some(group => group.every(rule => this.evaluateRule(rule, task))));
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
        case 'starts with': return sv.startsWith(rv);
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
