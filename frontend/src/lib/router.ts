export type Route =
  | { kind: 'picker' }
  | { kind: 'new' }
  | { kind: 'session'; id: string; subSessionId?: string }
  | { kind: 'compare'; a: string; b: string };

const SESSION_SUB_RE = /^#\/session\/([A-Za-z0-9._-]+)\/sub\/([A-Za-z0-9._-]+)$/;
const SESSION_RE = /^#\/session\/([A-Za-z0-9._-]+)$/;
const COMPARE_RE = /^#\/compare\/([A-Za-z0-9._-]+)\/([A-Za-z0-9._-]+)$/;

export function parseHash(hash: string): Route | null {
  if (!hash || hash === '#' || hash === '#/') return null;
  if (hash === '#/picker') return { kind: 'picker' };
  if (hash === '#/new') return { kind: 'new' };
  const ssm = SESSION_SUB_RE.exec(hash);
  if (ssm) return { kind: 'session', id: ssm[1], subSessionId: ssm[2] };
  const sm = SESSION_RE.exec(hash);
  if (sm) return { kind: 'session', id: sm[1] };
  const cm = COMPARE_RE.exec(hash);
  if (cm) return { kind: 'compare', a: cm[1], b: cm[2] };
  return null;
}

export function formatHash(route: Route): string {
  switch (route.kind) {
    case 'picker': return '#/picker';
    case 'new': return '#/new';
    case 'session':
      return route.subSessionId
        ? `#/session/${route.id}/sub/${route.subSessionId}`
        : `#/session/${route.id}`;
    case 'compare': return `#/compare/${route.a}/${route.b}`;
  }
}

export function installHashListener(onChange: (route: Route | null) => void): () => void {
  const handler = () => onChange(parseHash(location.hash));
  window.addEventListener('hashchange', handler);
  return () => window.removeEventListener('hashchange', handler);
}
