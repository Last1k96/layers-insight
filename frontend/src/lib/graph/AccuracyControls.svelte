<script lang="ts">
  import { onMount } from 'svelte';
  import { configStore } from '../stores/config.svelte';
  import { graphStore } from '../stores/graph.svelte';
  import { refreshRenderer } from './renderer';
  import { toggleHelp } from '../shortcuts';

  let { settingsOpen = $bindable(false) }: { settingsOpen?: boolean } = $props();

  onMount(() => {
    if (configStore.accuracyEnabled) {
      graphStore.accuracyViewActive = true;
      requestAnimationFrame(() => refreshRenderer());
    }
  });

  function toggleAccuracy() {
    const next = !configStore.accuracyEnabled;
    configStore.setAccuracyEnabled(next);
    graphStore.accuracyViewActive = next;
    refreshRenderer();
  }
</script>

<div class="control-cluster">
  <button
    type="button"
    class="ctl-btn"
    onclick={() => toggleHelp()}
    title="Keyboard shortcuts (?)"
    aria-label="Keyboard shortcuts"
  >
    <span class="qmark">?</span>
  </button>

  <button
    type="button"
    class="ctl-btn"
    class:active={configStore.accuracyEnabled}
    onclick={toggleAccuracy}
    title={configStore.accuracyEnabled
      ? 'Disable accuracy overlay (A)'
      : 'Enable accuracy overlay (A) — hold Alt for quick preview'}
    aria-label="Toggle accuracy overlay"
    aria-pressed={configStore.accuracyEnabled}
  >
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  </button>

  <button
    type="button"
    class="ctl-btn"
    class:active={settingsOpen}
    onclick={() => (settingsOpen = !settingsOpen)}
    title="Accuracy settings"
    aria-label="Accuracy settings"
    aria-expanded={settingsOpen}
  >
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  </button>
</div>

<style>
  .control-cluster {
    display: inline-flex;
    align-items: center;
    gap: 2px;
  }

  .ctl-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    padding: 0;
    border: 1px solid transparent;
    background: transparent;
    color: var(--text-muted);
    border-radius: 6px;
    cursor: pointer;
    transition:
      background 140ms ease,
      color 140ms ease,
      border-color 140ms ease,
      transform 140ms ease;
  }
  .ctl-btn:hover {
    color: var(--text-primary);
    background: var(--bg-menu);
    border-color: var(--border-soft);
  }
  .ctl-btn:focus-visible {
    outline: none;
    color: var(--text-primary);
    border-color: var(--accent);
    background: var(--accent-bg);
  }
  .ctl-btn.active {
    color: var(--text-primary);
    background: var(--accent-bg-strong);
    border-color: rgba(76, 141, 255, 0.4);
  }

  .ctl-btn svg {
    width: 14px;
    height: 14px;
  }
  .qmark {
    font-family: var(--font-display);
    font-size: 13px;
    font-weight: 600;
    line-height: 1;
  }
</style>
