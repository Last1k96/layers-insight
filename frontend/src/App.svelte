<script lang="ts">
  import SessionPicker from './lib/views/SessionPicker.svelte';
  import NewSession from './lib/views/NewSession.svelte';
  import MainView from './lib/views/MainView.svelte';
  import ErrorBanner from './lib/panels/ErrorBanner.svelte';
  import { sessionStore } from './lib/stores/session.svelte';

  type View = 'picker' | 'new-session' | 'main';

  let currentView = $state<View>('picker');
  let wsError = $state(false);

  function onSessionSelected(sessionId: string) {
    sessionStore.loadSession(sessionId).then(() => {
      currentView = 'main';
    });
  }

  function onNewSession() {
    currentView = 'new-session';
  }

  function onSessionCreated(sessionId: string) {
    sessionStore.loadSession(sessionId).then(() => {
      currentView = 'main';
    });
  }

  function onBackToPicker() {
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
    />
  {:else if currentView === 'new-session'}
    <NewSession
      onsessioncreated={onSessionCreated}
      onback={onBackToPicker}
    />
  {:else if currentView === 'main'}
    <MainView />
  {/if}
</div>
