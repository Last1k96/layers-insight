export interface LogEntry {
  _id: number;
  task_id: string;
  node_name: string;
  level: string;
  message: string;
  timestamp: string;
  formattedTime: string;
}

const MAX_ENTRIES = 2000;

const timeFmt = new Intl.DateTimeFormat('en-US', {
  hour12: false,
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  fractionalSecondDigits: 3,
});

let nextId = 0;

class LogStore {
  entries = $state<LogEntry[]>([]);
  visible = $state(false);

  private _buffer: LogEntry[] = [];
  private _flushScheduled = false;

  addEntry(entry: Omit<LogEntry, '_id' | 'formattedTime'>) {
    let formattedTime: string;
    try {
      formattedTime = timeFmt.format(new Date(entry.timestamp));
    } catch {
      formattedTime = entry.timestamp;
    }

    this._buffer.push({
      ...entry,
      _id: nextId++,
      formattedTime,
    });

    if (!this._flushScheduled) {
      this._flushScheduled = true;
      requestAnimationFrame(() => this._flush());
    }
  }

  private _flush() {
    this._flushScheduled = false;
    if (this._buffer.length === 0) return;

    const buf = this._buffer;
    this._buffer = [];

    for (const entry of buf) {
      this.entries.push(entry);
    }

    if (this.entries.length > MAX_ENTRIES) {
      this.entries = this.entries.slice(-MAX_ENTRIES);
    }
  }

  clear() {
    this.entries = [];
    this._buffer = [];
  }

  toggle() {
    this.visible = !this.visible;
  }
}

export const logStore = new LogStore();
