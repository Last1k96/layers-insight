<script lang="ts">
  import SessionPicker from './lib/views/SessionPicker.svelte';
  import NewSession from './lib/views/NewSession.svelte';
  import MainView from './lib/views/MainView.svelte';
  import SessionCompare from './lib/views/SessionCompare.svelte';
  import ErrorBanner from './lib/panels/ErrorBanner.svelte';
  import { sessionStore } from './lib/stores/session.svelte';
  import type { SessionConfig } from './lib/stores/types';

  type View = 'picker' | 'new-session' | 'main' | 'compare';

  let currentView = $state<View>('picker');
  let wsError = $state(false);

  // Clone mode state
  let cloneSourceId = $state<string | undefined>(undefined);
  let cloneSourceConfig = $state<SessionConfig | undefined>(undefined);
  let cloneSourceName = $state<string | undefined>(undefined);

  // Compare mode state
  let compareSessionA = $state('');
  let compareSessionB = $state('');

  function onSessionSelected(sessionId: string) {
    sessionStore.loadSession(sessionId).then(() => {
      currentView = 'main';
    });
  }

  function onNewSession() {
    cloneSourceId = undefined;
    cloneSourceConfig = undefined;
    cloneSourceName = undefined;
    currentView = 'new-session';
  }

  function onCloneSession(sessionId: string) {
    // Load source session detail for pre-filling
    sessionStore.loadSession(sessionId).then(() => {
      const detail = sessionStore.currentSession;
      if (detail) {
        cloneSourceId = sessionId;
        cloneSourceConfig = detail.config;
        cloneSourceName = detail.info.model_name;
      }
      currentView = 'new-session';
    });
  }

  function onCompare(sessionA: string, sessionB: string) {
    compareSessionA = sessionA;
    compareSessionB = sessionB;
    currentView = 'compare';
  }

  function onSessionCreated(sessionId: string) {
    cloneSourceId = undefined;
    cloneSourceConfig = undefined;
    cloneSourceName = undefined;
    sessionStore.loadSession(sessionId).then(() => {
      currentView = 'main';
    });
  }

  function onBackToPicker() {
    cloneSourceId = undefined;
    cloneSourceConfig = undefined;
    cloneSourceName = undefined;
    currentView = 'picker';
  }
</script>

<div class="h-screen w-screen flex flex-col bg-[--bg-primary] text-gray-100">
  {#if wsError}
    <ErrorBanner message="Connection lost. Reconnecting..." onretry={() => wsError = false} />
  {/if}

  {#if currentView === 'picker'}
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
      onnodeselected={(sessionId, nodeName) => {
        sessionStore.loadSession(sessionId).then(() => {
          currentView = 'main';
        });
      }}
    />
  {:else if currentView === 'main'}
    <MainView />
  {/if}
</div>
