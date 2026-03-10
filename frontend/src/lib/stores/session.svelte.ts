import type { SessionInfo, SessionDetail, SessionConfig } from './types';

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
      this.sessions = await res.json();
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
    } catch (e: any) {
      this.error = e.message;
    } finally {
      this.loading = false;
    }
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
}

export const sessionStore = new SessionStore();
