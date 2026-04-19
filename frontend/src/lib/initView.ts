import type { AppDefaults, SessionInfo } from './stores/types';
import type { Route } from './router';

export type View = 'picker' | 'new-session' | 'main' | 'compare';

export const VIEW_STORAGE_KEY = 'li.lastView';
export const SESSION_STORAGE_KEY = 'li.lastSessionId';
export const CLI_CONSUMED_KEY = 'li.cliConsumed';

export interface InitDecision {
  view: View;
  sessionId?: string;
  subSessionId?: string;
  compare?: { a: string; b: string };
  banner?: string;
}

export function cliFingerprint(d: AppDefaults | null): string {
  if (!d) return '';
  return JSON.stringify({
    m: d.model_path ?? '',
    o: d.ov_path ?? '',
    i: d.cli_inputs ?? [],
    md: d.main_device,
    rd: d.ref_device,
  });
}

function hasCliIntent(d: AppDefaults | null): boolean {
  if (!d) return false;
  return Boolean(d.model_path) || (d.cli_inputs?.length ?? 0) > 0;
}

export function pickInitialView(
  defaults: AppDefaults | null,
  sessions: SessionInfo[],
  route: Route | null,
  persisted: { view: string | null; sessionId: string | null; cliConsumed: string | null },
): InitDecision {
  // 1. Hash route wins
  if (route) {
    if (route.kind === 'session') {
      if (sessions.some(s => s.id === route.id)) {
        return { view: 'main', sessionId: route.id, subSessionId: route.subSessionId };
      }
      return { view: sessions.length > 0 ? 'picker' : 'new-session', banner: `Session ${route.id} no longer exists.` };
    }
    if (route.kind === 'compare') {
      const aOk = sessions.some(s => s.id === route.a);
      const bOk = sessions.some(s => s.id === route.b);
      if (aOk && bOk) return { view: 'compare', compare: { a: route.a, b: route.b } };
      return { view: 'picker', banner: 'One or both compared sessions no longer exist.' };
    }
    if (route.kind === 'new') return { view: 'new-session' };
    if (route.kind === 'picker') return { view: 'picker' };
  }

  // 2. Persisted active session
  if (persisted.view === 'main' && persisted.sessionId && sessions.some(s => s.id === persisted.sessionId)) {
    return { view: 'main', sessionId: persisted.sessionId };
  }

  // 3. CLI-launch intent (first visit only)
  if (hasCliIntent(defaults)) {
    const fp = cliFingerprint(defaults);
    if (fp && persisted.cliConsumed !== fp) {
      return { view: 'new-session' };
    }
  }

  // 4. Default
  if (sessions.length > 0) return { view: 'picker' };
  return { view: 'new-session' };
}
