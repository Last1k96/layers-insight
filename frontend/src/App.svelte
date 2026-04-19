<script lang="ts">
  import { onMount } from 'svelte';
  import SessionPicker from './lib/views/SessionPicker.svelte';
  import NewSession from './lib/views/NewSession.svelte';
  import MainView from './lib/views/MainView.svelte';
  import SessionCompare from './lib/views/SessionCompare.svelte';
  import ErrorBanner from './lib/panels/ErrorBanner.svelte';
  import PerfHud from './lib/perf/PerfHud.svelte';
  import { sessionStore } from './lib/stores/session.svelte';
  import { graphStore } from './lib/stores/graph.svelte';
  import { configStore } from './lib/stores/config.svelte';
  import type { SessionConfig } from './lib/stores/types';
  import {
    pickInitialView,
    VIEW_STORAGE_KEY,
    SESSION_STORAGE_KEY,
    CLI_CONSUMED_KEY,
    type View,
  } from './lib/initView';
  import { parseHash, formatHash, installHashListener, type Route } from './lib/router';

  let currentView = $state<View | null>(null);
  let initBanner = $state<string | null>(null);
  let wsError = $state(false);

  // Clone mode state
  let cloneSourceId = $state<string | undefined>(undefined);
  let cloneSourceConfig = $state<SessionConfig | undefined>(undefined);
  let cloneSourceName = $state<string | undefined>(undefined);

  // Compare mode state
  let compareSessionA = $state('');
  let compareSessionB = $state('');

  // Hash routing — set when we are programmatically updating the hash so the
  // hashchange listener does not loop back into our own setView.
  let suppressHashHandler = false;

  function syncHash(view: View): void {
    let route: Route;
    if (view === 'main' && sessionStore.currentSession) {
      route = {
        kind: 'session',
        id: sessionStore.currentSession.id,
        subSessionId: graphStore.activeSubSessionId ?? undefined,
      };
    } else if (view === 'compare' && compareSessionA && compareSessionB) {
      route = { kind: 'compare', a: compareSessionA, b: compareSessionB };
    } else if (view === 'new-session') {
      route = { kind: 'new' };
    } else {
      route = { kind: 'picker' };
    }
    const next = formatHash(route);
    if (location.hash !== next) {
      suppressHashHandler = true;
      try { history.replaceState(null, '', next); } finally { suppressHashHandler = false; }
    }
  }

  function setView(next: View): void {
    currentView = next;
    try { localStorage.setItem(VIEW_STORAGE_KEY, next); } catch { /* ignore */ }
    syncHash(next);
  }

  async function handleHashChange(route: Route | null): Promise<void> {
    if (suppressHashHandler) return;
    if (!route) {
      setView(sessionStore.sessions.length > 0 ? 'picker' : 'new-session');
      return;
    }
    if (route.kind === 'picker') {
      setView('picker');
    } else if (route.kind === 'new') {
      cloneSourceId = undefined;
      cloneSourceConfig = undefined;
      cloneSourceName = undefined;
      setView('new-session');
    } else if (route.kind === 'compare') {
      compareSessionA = route.a;
      compareSessionB = route.b;
      setView('compare');
    } else if (route.kind === 'session') {
      const desiredSub = route.subSessionId ?? null;
      if (sessionStore.currentSession?.id === route.id) {
        if (graphStore.activeSubSessionId !== desiredSub) {
          graphStore.setActiveSubSession(desiredSub);
        }
        setView('main');
        return;
      }
      graphStore.setActiveSubSession(desiredSub);
      await sessionStore.loadSession(route.id);
      if (sessionStore.currentSession) {
        setView('main');
      } else {
        initBanner = `Session ${route.id} no longer exists.`;
        setView(sessionStore.sessions.length > 0 ? 'picker' : 'new-session');
      }
    }
  }

  async function init(): Promise<void> {
    try {
      const [defaults] = await Promise.all([
        configStore.fetchDefaults(),
        sessionStore.fetchSessions(),
      ]);

      const persisted = {
        view: localStorage.getItem(VIEW_STORAGE_KEY),
        sessionId: localStorage.getItem(SESSION_STORAGE_KEY),
        cliConsumed: sessionStorage.getItem(CLI_CONSUMED_KEY),
      };

      const route = parseHash(location.hash);
      const decision = pickInitialView(defaults, sessionStore.sessions, route, persisted);

      if (decision.banner) initBanner = decision.banner;

      if (decision.view === 'main' && decision.sessionId) {
        graphStore.setActiveSubSession(decision.subSessionId ?? null);
        await sessionStore.loadSession(decision.sessionId);
        if (sessionStore.currentSession) {
          setView('main');
        } else {
          setView(sessionStore.sessions.length > 0 ? 'picker' : 'new-session');
        }
      } else if (decision.view === 'compare' && decision.compare) {
        compareSessionA = decision.compare.a;
        compareSessionB = decision.compare.b;
        setView('compare');
      } else {
        setView(decision.view);
      }
    } catch (e: any) {
      console.error('Init failed:', e);
      initBanner = String(e?.message ?? e);
      setView('picker');
    }
  }

  onMount(() => {
    init();
    return installHashListener((route) => {
      handleHashChange(route);
    });
  });

  $effect(() => {
    const _sub = graphStore.activeSubSessionId;
    if (currentView === 'main' && sessionStore.currentSession) {
      syncHash('main');
    }
  });

  function onSessionSelected(sessionId: string) {
    graphStore.setActiveSubSession(null);
    sessionStore.loadSession(sessionId).then(() => {
      if (sessionStore.currentSession) setView('main');
    });
  }

  function onNewSession() {
    cloneSourceId = undefined;
    cloneSourceConfig = undefined;
    cloneSourceName = undefined;
    setView('new-session');
  }

  function onCloneSession(sessionId: string) {
    sessionStore.loadSession(sessionId).then(() => {
      const detail = sessionStore.currentSession;
      if (detail) {
        cloneSourceId = sessionId;
        cloneSourceConfig = detail.config;
        cloneSourceName = detail.info.model_name;
      }
      setView('new-session');
    });
  }

  function onCompare(sessionA: string, sessionB: string) {
    compareSessionA = sessionA;
    compareSessionB = sessionB;
    setView('compare');
  }

  function onSessionCreated(sessionId: string) {
    cloneSourceId = undefined;
    cloneSourceConfig = undefined;
    cloneSourceName = undefined;
    graphStore.setActiveSubSession(null);
    sessionStore.loadSession(sessionId).then(() => {
      if (sessionStore.currentSession) setView('main');
    });
  }

  function onBackToPicker() {
    cloneSourceId = undefined;
    cloneSourceConfig = undefined;
    cloneSourceName = undefined;
    setView('picker');
  }
</script>

<div class="h-screen w-screen flex flex-col bg-[--bg-primary] text-gray-100">
  {#if wsError}
    <ErrorBanner message="Connection lost. Reconnecting..." onretry={() => wsError = false} />
  {/if}
  {#if initBanner}
    <ErrorBanner message={initBanner} onretry={() => initBanner = null} />
  {/if}

  {#if currentView === null}
    <div class="flex-1 flex items-center justify-center text-gray-400 text-sm">
      Loading…
    </div>
  {:else if currentView === 'picker'}
    <SessionPicker
      onsessionselected={onSessionSelected}
      onnewsession={onNewSession}
      onclonesession={onCloneSession}
      oncompare={onCompare}
    />
  {:else if currentView === 'new-session'}
    <NewSession
      onsessioncreated={onSessionCreated}
      onback={onBackToPicker}
      {cloneSourceId}
      {cloneSourceConfig}
      {cloneSourceName}
    />
  {:else if currentView === 'compare'}
    <SessionCompare
      sessionAId={compareSessionA}
      sessionBId={compareSessionB}
      onback={onBackToPicker}
      onnodeselected={(sessionId, _nodeName) => {
        graphStore.setActiveSubSession(null);
        sessionStore.loadSession(sessionId).then(() => {
          if (sessionStore.currentSession) setView('main');
        });
      }}
    />
  {:else if currentView === 'main'}
    <MainView onback={onBackToPicker} />
  {/if}

  <PerfHud />
</div>
