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
          <kbd class="shortcut-key">Alt (hold)</kbd>
          <span class="shortcut-desc">Accuracy view mode</span>
        </div>

        <div class="section-label">Navigation</div>
        <div class="shortcut-row">
          <kbd class="shortcut-key">↑ / ↓</kbd>
          <span class="shortcut-desc">Navigate queue list</span>
        </div>
        <div class="shortcut-row">
          <kbd class="shortcut-key">Ctrl + ↑</kbd>
          <span class="shortcut-desc">Go to predecessor node</span>
        </div>
        <div class="shortcut-row">
          <kbd class="shortcut-key">Ctrl + ↓</kbd>
          <span class="shortcut-desc">Go to successor node</span>
        </div>
        <div class="shortcut-row">
          <kbd class="shortcut-key">Ctrl + ← / →</kbd>
          <span class="shortcut-desc">Go to sibling node</span>
        </div>

        <div class="section-label">Mouse</div>
        <div class="shortcut-row">
          <kbd class="shortcut-key">Ctrl + Click</kbd>
          <span class="shortcut-desc">Select and center on node</span>
        </div>
        <div class="shortcut-row">
          <kbd class="shortcut-key">Double-click</kbd>
          <span class="shortcut-desc">Enqueue inference</span>
        </div>
        <div class="shortcut-row">
          <kbd class="shortcut-key">Shift + Dbl-click</kbd>
          <span class="shortcut-desc">Re-enqueue inference (force)</span>
        </div>

        <div class="section-label">Search</div>
        <div class="shortcut-row">
          <kbd class="shortcut-key">↑ / ↓</kbd>
          <span class="shortcut-desc">Cycle through results</span>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .shortcuts-modal {
    background: var(--bg-panel);
    border-radius: 16px;
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4), 0 2px 8px rgba(0, 0, 0, 0.2);
    min-width: 320px;
    max-width: 420px;
    backdrop-filter: blur(24px);
    overflow: hidden;
  }

  .modal-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 16px;
    background: var(--bg-panel);
  }

  .modal-header h3 {
    font-size: 13px;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: var(--text-primary);
    margin: 0;
  }

  .close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border: none;
    background: none;
    color: var(--text-secondary);
    opacity: 0.4;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.1s;
    padding: 0;
  }

  .close-btn:hover {
    background: var(--bg-menu);
    color: var(--text-primary);
    opacity: 1;
  }

  .modal-body {
    padding: 10px 16px 16px;
  }

  .shortcut-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 7px 0;
  }

  .shortcut-key {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 28px;
    padding: 3px 8px;
    font-size: 11px;
    font-family: var(--font-mono);
    background: var(--bg-primary);
    border-radius: 6px;
    color: var(--text-primary);
    white-space: nowrap;
  }

  .shortcut-desc {
    font-size: 12px;
    color: var(--text-secondary);
    opacity: 0.6;
  }

  .section-label {
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    opacity: 0.4;
    margin-top: 10px;
    margin-bottom: 2px;
  }
</style>
