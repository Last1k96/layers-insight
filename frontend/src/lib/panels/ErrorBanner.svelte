<script lang="ts">
  let {
    message,
    onretry,
  }: {
    message: string;
    onretry?: () => void;
  } = $props();
</script>

<div class="err-banner" role="alert">
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.75" aria-hidden="true">
    <circle cx="8" cy="8" r="6.5" />
    <path d="M8 5v3.5" stroke-linecap="round" />
    <circle cx="8" cy="10.5" r="0.6" fill="currentColor" stroke="none" />
  </svg>
  <span class="err-banner__msg">{message}</span>
  {#if onretry}
    <button class="ll-btn ll-btn--xs ll-btn--danger" onclick={onretry}>Retry</button>
  {/if}
</div>

<style>
  .err-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: var(--status-err-bg);
    border-bottom: 1px solid var(--status-err-border);
    color: var(--status-err);
    font-size: 12.5px;
    font-weight: 500;
    z-index: 50;
  }
  .err-banner__msg {
    flex: 1;
    color: var(--text-primary);
  }
</style>
