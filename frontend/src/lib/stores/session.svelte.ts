import type { SessionInfo, SessionDetail, SessionConfig, CloneRequest, CloneResponse, CompareResponse } from './types';

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
      if (!res.ok) throw new Error(`Failed to fetch sessions: ${res.statusText}`);
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
      if (!res.ok) throw new Error(`Failed to create session: ${res.statusText}`);
      const info: SessionInfo = await res.json();
      this.sessions = [...this.sessions, info];
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
      if (!res.ok) throw new Error(`Failed to load session: ${res.statusText}`);
      this.currentSession = await res.json();
      localStorage.setItem('lastSessionId', sessionId);
    } catch (e: any) {
      this.error = e.message;
    } finally {
      this.loading = false;
    }
  }

  get lastSessionId(): string | null {
    return localStorage.getItem('lastSessionId');
  }

  async deleteSession(sessionId: string): Promise<void> {
    try {
      await fetch(`/api/sessions/${sessionId}`, { method: 'DELETE' });
      this.sessions = this.sessions.filter(s => s.id !== sessionId);
      if (this.currentSession?.id === sessionId) {
        this.currentSession = null;
      }
    } catch (e: any) {
      this.error = e.message;
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
      if (!res.ok) throw new Error(`Failed to clone session: ${res.statusText}`);
      const data: CloneResponse = await res.json();
      this.sessions = [...this.sessions, data.session];
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
      if (!res.ok) throw new Error(`Failed to enqueue nodes: ${res.statusText}`);
      return true;
    } catch (e: any) {
      this.error = e.message;
      return false;
    }
  }

  async compareSessions(sessionA: string, sessionB: string): Promise<CompareResponse | null> {
    try {
      const res = await fetch(`/api/sessions/compare?session_a=${sessionA}&session_b=${sessionB}`);
      if (!res.ok) throw new Error(`Failed to compare sessions: ${res.statusText}`);
      return await res.json();
    } catch (e: any) {
      this.error = e.message;
      return null;
    }
  }
}

export const sessionStore = new SessionStore();
