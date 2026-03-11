export interface LogEntry {
  task_id: string;
  node_name: string;
  level: string;
  message: string;
  timestamp: string;
}

const MAX_ENTRIES = 2000;

class LogStore {
  entries = $state<LogEntry[]>([]);
  visible = $state(false);

  addEntry(entry: LogEntry) {
    this.entries.push(entry);
    if (this.entries.length > MAX_ENTRIES) {
      this.entries = this.entries.slice(-MAX_ENTRIES);
    }
    // Auto-show panel on first log entry
    if (!this.visible) {
      this.visible = true;
    }
  }

  clear() {
    this.entries = [];
  }

  toggle() {
    this.visible = !this.visible;
  }
}

export const logStore = new LogStore();
