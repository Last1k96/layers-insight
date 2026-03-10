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

  let width = $state(initialWidth);
  let collapsed = $state(initialCollapsed);
  let resizing = $state(false);

  function startResize(e: MouseEvent) {
    e.preventDefault();
    resizing = true;
    const startX = e.clientX;
    const startWidth = width;

    function onMouseMove(e: MouseEvent) {
      const dx = side === 'left' ? e.clientX - startX : startX - e.clientX;
      width = Math.max(200, Math.min(600, startWidth + dx));
    }

    function onMouseUp() {
      resizing = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      // Save to localStorage
      localStorage.setItem(`panel-${side}-width`, String(width));
    }

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }

  function toggleCollapse() {
    collapsed = !collapsed;
    localStorage.setItem(`panel-${side}-collapsed`, String(collapsed));
  }

  // Restore from localStorage
  $effect(() => {
    const savedWidth = localStorage.getItem(`panel-${side}-width`);
    if (savedWidth) width = parseInt(savedWidth);
    const savedCollapsed = localStorage.getItem(`panel-${side}-collapsed`);
    if (savedCollapsed) collapsed = savedCollapsed === 'true';
  });
</script>

<div
  class="absolute top-2 bottom-2 flex flex-col bg-gray-800/95 backdrop-blur border border-gray-700 rounded-lg shadow-xl z-10 transition-all"
  class:left-2={side === 'left'}
  class:right-2={side === 'right'}
  style:width={collapsed ? '40px' : `${width}px`}
>
  <!-- Header -->
  <div
    class="flex items-center justify-between px-3 py-2 border-b border-gray-700 cursor-pointer select-none shrink-0"
    onclick={toggleCollapse}
  >
    {#if !collapsed}
      <span class="text-sm font-medium text-gray-300">{title}</span>
    {/if}
    <span class="text-gray-500 text-xs">{collapsed ? (side === 'left' ? '>' : '<') : (side === 'left' ? '<' : '>')}</span>
  </div>

  <!-- Content -->
  {#if !collapsed}
    <div class="flex-1 overflow-hidden flex flex-col min-h-0">
      {@render children()}
    </div>

    <!-- Resize handle -->
    <div
      class="absolute top-0 bottom-0 w-1 cursor-col-resize hover:bg-blue-500/50 transition-colors"
      class:right-0={side === 'left'}
      class:left-0={side === 'right'}
      onmousedown={startResize}
    ></div>
  {/if}
</div>
