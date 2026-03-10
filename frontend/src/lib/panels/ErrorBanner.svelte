<script lang="ts">
  let {
    message,
    onretry,
  }: {
    message: string;
    onretry?: () => void;
  } = $props();
</script>

<div class="bg-red-900/80 border-b border-red-700 px-4 py-2 flex items-center justify-between text-sm z-50">
  <span class="text-red-200">{message}</span>
  {#if onretry}
    <button
      class="px-3 py-1 bg-red-800 hover:bg-red-700 rounded text-red-200 text-xs transition-colors"
      onclick={onretry}
    >
      Retry
    </button>
  {/if}
</div>
