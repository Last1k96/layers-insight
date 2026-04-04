<script lang="ts">
  import type { Snippet } from 'svelte';

  let {
    side,
    title,
    initialWidth = 320,
    collapsed: initialCollapsed = false,
    children,
  }: {
    side: 'left' | 'right';
    title: string;
    initialWidth?: number;
    collapsed?: boolean;
    children: Snippet;
  } = $props();

  let width = $state(320);
  let collapsed = $state(false);
  let resizing = $state(false);

  function startResize(e: MouseEvent) {
    e.preventDefault();
    resizing = true;
    const startX = e.clientX;
    const startWidth = width;

    function onMouseMove(e: MouseEvent) {
      const dx = side === 'left' ? e.clientX - startX : startX - e.clientX;
      width = Math.max(200, Math.min(1200, startWidth + dx));
    }

    function onMouseUp() {
      resizing = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      localStorage.setItem(`panel-${side}-width`, String(width));
    }

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }

  function toggleCollapse() {
    collapsed = !collapsed;
    localStorage.setItem(`panel-${side}-collapsed`, String(collapsed));
  }

  // Restore from localStorage, falling back to props
  $effect(() => {
    const savedWidth = localStorage.getItem(`panel-${side}-width`);
    width = savedWidth ? parseInt(savedWidth) : initialWidth;
    const savedCollapsed = localStorage.getItem(`panel-${side}-collapsed`);
    collapsed = savedCollapsed ? savedCollapsed === 'true' : initialCollapsed;
  });
</script>

<div
  class="absolute top-2 bottom-2 flex flex-col bg-[--bg-panel]/95 backdrop-blur-xl rounded-xl z-10 overflow-hidden"
  class:transition-all={!resizing}
  class:left-2={side === 'left'}
  class:right-2={side === 'right'}
  style:width={collapsed ? '40px' : `${width}px`}
  style:box-shadow="var(--shadow-panel)"
>
  <!-- Header -->
  <div
    class="flex items-center justify-between px-3 py-2.5 cursor-pointer select-none shrink-0 bg-[--bg-panel]"
    role="button"
    tabindex="0"
    onclick={toggleCollapse}
    onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleCollapse(); }}}
  >
    {#if !collapsed}
      <span class="text-[13px] font-medium tracking-tight text-content-primary">{title}</span>
    {/if}
    <span class="text-content-secondary/60 text-xs">{collapsed ? (side === 'left' ? '>' : '<') : (side === 'left' ? '<' : '>')}</span>
  </div>

  <!-- Content -->
  {#if !collapsed}
    <div class="flex-1 overflow-hidden flex flex-col min-h-0">
      {@render children()}
    </div>

    <!-- Resize handle -->
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      class="absolute top-0 bottom-0 w-1.5 cursor-col-resize hover:bg-accent/30 transition-colors"
      class:right-0={side === 'left'}
      class:left-0={side === 'right'}
      role="separator"
      aria-orientation="vertical"
      onmousedown={startResize}
    ></div>
  {/if}
</div>
