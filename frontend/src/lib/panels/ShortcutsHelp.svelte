<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { getShortcuts, onHelpVisibilityChange, setHelpVisible } from '../shortcuts';

  let visible = $state(false);

  let unsub: (() => void) | null = null;

  onMount(() => {
    unsub = onHelpVisibilityChange((v) => { visible = v; });
  });

  onDestroy(() => {
    unsub?.();
  });

  function close() {
    setHelpVisible(false);
  }

  function handleBackdropKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') close();
  }
</script>

{#if visible}
  <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
  <div
    class="fixed inset-0 z-[70] flex items-center justify-center bg-black/40"
    onclick={close}
    onkeydown={handleBackdropKeydown}
    role="dialog"
    tabindex="-1"
    aria-modal="true"
    aria-label="Keyboard shortcuts"
  >
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      class="shortcuts-modal"
      onclick={(e) => e.stopPropagation()}
      onkeydown={(e) => e.stopPropagation()}
      role="document"
    >
      <div class="modal-header">
        <h3>Keyboard Shortcuts</h3>
        <button class="close-btn" onclick={close} aria-label="Close">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </button>
      </div>
      <div class="modal-body">
        {#each getShortcuts() as shortcut (shortcut.key)}
          <div class="shortcut-row">
            <kbd class="shortcut-key">{shortcut.key}</kbd>
            <span class="shortcut-desc">{shortcut.description}</span>
          </div>
        {/each}
        <div class="shortcut-row">
          <kbd class="shortcut-key">Ctrl+F</kbd>
          <span class="shortcut-desc">Open search (also works in interactions)</span>
        </div>
        <div class="shortcut-row">
          <kbd class="shortcut-key">Alt (hold)</kbd>
          <span class="shortcut-desc">Accuracy view mode</span>
        </div>
        <div class="shortcut-row">
          <kbd class="shortcut-key">Arrow keys</kbd>
          <span class="shortcut-desc">Navigate queue / graph (with Ctrl)</span>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .shortcuts-modal {
    background: var(--bg-panel);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    min-width: 320px;
    max-width: 420px;
    backdrop-filter: blur(16px);
    overflow: hidden;
  }

  .modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 16px;
    border-bottom: 1px solid var(--border-color);
  }

  .modal-header h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border: none;
    background: none;
    color: var(--text-secondary);
    border-radius: 4px;
    cursor: pointer;
    transition: background 0.1s, color 0.1s;
    padding: 0;
  }

  .close-btn:hover {
    background: var(--bg-menu);
    color: var(--text-primary);
  }

  .modal-body {
    padding: 10px 16px 14px;
  }

  .shortcut-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0;
  }

  .shortcut-key {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 28px;
    padding: 2px 8px;
    font-size: 11px;
    font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, monospace;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    color: var(--text-primary);
    white-space: nowrap;
  }

  .shortcut-desc {
    font-size: 12px;
    color: var(--text-secondary);
  }
</style>
