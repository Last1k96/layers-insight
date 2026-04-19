import type { SessionInfo, SessionDetail, SessionConfig, CloneRequest, CloneResponse, CompareResponse } from './types';
import { SESSION_STORAGE_KEY, VIEW_STORAGE_KEY } from '../initView';

async function extractError(res: Response): Promise<string> {
  try { return (await res.json()).detail || res.statusText; } catch { return res.statusText; }
}

class SessionStore {
  sessions = $state<SessionInfo[]>([]);
  currentSession = $state<SessionDetail | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  async fetchSessions(): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const res = await fetch('/api/sessions');
      if (!res.ok) throw new Error(await extractError(res));
      const data: SessionInfo[] = await res.json();
      data.sort((a, b) => b.created_at.localeCompare(a.created_at));
      this.sessions = data;
    } catch (e: any) {
      this.error = e.message;
    } finally {
      this.loading = false;
    }
  }

  async createSession(config: SessionConfig): Promise<SessionInfo | null> {
    this.loading = true;
    this.error = null;
    try {
      const res = await fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      });
      if (!res.ok) throw new Error(await extractError(res));
      const info: SessionInfo = await res.json();
      this.sessions = [info, ...this.sessions];
      return info;
    } catch (e: any) {
      this.error = e.message;
      return null;
    } finally {
      this.loading = false;
    }
  }

  async loadSession(sessionId: string): Promise<void> {
    this.loading = true;
    this.error = null;
    try {
      const res = await fetch(`/api/sessions/${sessionId}`);
      if (res.status === 404) {
        if (localStorage.getItem(SESSION_STORAGE_KEY) === sessionId) {
          localStorage.removeItem(SESSION_STORAGE_KEY);
          localStorage.removeItem(VIEW_STORAGE_KEY);
        }
        this.currentSession = null;
        this.error = `Session ${sessionId} no longer exists.`;
        return;
      }
      if (!res.ok) throw new Error(await extractError(res));
      this.currentSession = await res.json();
      localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    } catch (e: any) {
      this.error = e.message;
    } finally {
      this.loading = false;
    }
  }

  get lastSessionId(): string | null {
    return localStorage.getItem(SESSION_STORAGE_KEY);
  }

  async deleteSession(sessionId: string): Promise<void> {
    try {
      await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
      this.sessions = this.sessions.filter(s => s.id !== sessionId);
      if (this.currentSession?.id === sessionId) {
        this.currentSession = null;
      }
      if (localStorage.getItem(SESSION_STORAGE_KEY) === sessionId) {
        localStorage.removeItem(SESSION_STORAGE_KEY);
        localStorage.removeItem(VIEW_STORAGE_KEY);
      }
    } catch (e: any) {
      this.error = e.message;
    }
  }

  async renameSession(sessionId: string, name: string): Promise<{ id: string } | null> {
    try {
      const res = await fetch(`/api/sessions/${sessionId}/rename`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) throw new Error(await extractError(res));
      const data = await res.json() as { renamed: boolean; name: string; id: string };
      const newId = data.id;

      this.sessions = this.sessions.map(s =>
        s.id === sessionId ? { ...s, id: newId, model_name: name } : s
      );

      if (this.currentSession?.id === sessionId) {
        this.currentSession = {
          ...this.currentSession,
          id: newId,
          info: { ...this.currentSession.info, id: newId, model_name: name },
        };
      }

      if (localStorage.getItem(SESSION_STORAGE_KEY) === sessionId) {
        localStorage.setItem(SESSION_STORAGE_KEY, newId);
      }

      return { id: newId };
    } catch (e: any) {
      this.error = e.message;
      return null;
    }
  }

  async cloneSession(sourceSessionId: string, overrides: CloneRequest): Promise<CloneResponse | null> {
    this.loading = true;
    this.error = null;
    try {
      const res = await fetch(`/api/sessions/${sourceSessionId}/clone`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(overrides),
      });
      if (!res.ok) throw new Error(await extractError(res));
      const data: CloneResponse = await res.json();
      this.sessions = [data.session, ...this.sessions];
      return data;
    } catch (e: any) {
      this.error = e.message;
      return null;
    } finally {
      this.loading = false;
    }
  }

  async cloneEnqueue(sourceSessionId: string, targetSessionId: string, nodeNames: string[]): Promise<boolean> {
    try {
      const res = await fetch(`/api/sessions/${sourceSessionId}/clone-enqueue`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_session_id: targetSessionId, node_names: nodeNames }),
      });
      if (!res.ok) throw new Error(await extractError(res));
      return true;
    } catch (e: any) {
      this.error = e.message;
      return false;
    }
  }

  async compareSessions(sessionA: string, sessionB: string): Promise<CompareResponse | null> {
    try {
      const res = await fetch(`/api/sessions/compare?session_a=${sessionA}&session_b=${sessionB}`);
      if (!res.ok) throw new Error(await extractError(res));
      return await res.json();
    } catch (e: any) {
      this.error = e.message;
      return null;
    }
  }
}

export const sessionStore = new SessionStore();
