<script lang="ts">
  import type { Snippet } from 'svelte';
  import { onMount, onDestroy } from 'svelte';
  import { registerShortcut, unregisterShortcut } from '../shortcuts';

  let {
    side,
    title,
    initialWidth = 320,
    children,
    header,
  }: {
    side: 'left' | 'right';
    title: string;
    initialWidth?: number;
    children: Snippet;
    header?: Snippet;
  } = $props();

  const RAIL_W = 38;
  const COLLAPSE_AT = 140;
  const MIN_EXPANDED = 220;
  const MAX_EXPANDED = 900;

  let width = $state(320);
  let collapsed = $state(false);
  let resizing = $state(false);
  let willCollapse = $state(false);

  let widthKey = $derived(`panel-${side}-width-v2`);
  let collapsedKey = $derived(`panel-${side}-collapsed-v2`);

  function persist() {
    localStorage.setItem(collapsedKey, String(collapsed));
    if (!collapsed) localStorage.setItem(widthKey, String(Math.round(width)));
  }

  function startResize(e: MouseEvent) {
    if (e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    resizing = true;
    const startX = e.clientX;
    const startCollapsed = collapsed;
    const startWidth = collapsed ? RAIL_W : width;

    function onMove(ev: MouseEvent) {
      const dx = side === 'left' ? ev.clientX - startX : startX - ev.clientX;
      const raw = startWidth + dx;

      if (startCollapsed) {
        // Drag-out from the rail: width follows the cursor 1:1; no collapse-on-release.
        willCollapse = false;
        if (raw <= RAIL_W) {
          if (!collapsed) collapsed = true;
          width = startWidth;
        } else {
          if (collapsed) collapsed = false;
          width = Math.min(MAX_EXPANDED, raw);
        }
        return;
      }

      // Drag-in from an expanded panel: elastic past the collapse threshold.
      if (raw < COLLAPSE_AT) {
        willCollapse = true;
        const past = COLLAPSE_AT - raw;
        width = Math.max(RAIL_W, COLLAPSE_AT - past * 0.45);
      } else {
        willCollapse = false;
        width = Math.min(MAX_EXPANDED, Math.max(RAIL_W, raw));
      }
    }

    function onUp() {
      resizing = false;
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      if (willCollapse) {
        collapsed = true;
        const saved = parseInt(localStorage.getItem(widthKey) ?? '');
        width = !isNaN(saved) && saved >= MIN_EXPANDED ? saved : initialWidth;
      } else if (!collapsed && width < MIN_EXPANDED) {
        width = MIN_EXPANDED;
      }
      willCollapse = false;
      persist();
    }

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }

  function expand() {
    if (!collapsed) return;
    collapsed = false;
    persist();
  }

  function collapse() {
    if (collapsed) return;
    collapsed = true;
    persist();
  }

  function toggle() {
    collapsed ? expand() : collapse();
  }

  $effect(() => {
    const saved = parseInt(localStorage.getItem(widthKey) ?? '');
    width = !isNaN(saved) && saved >= MIN_EXPANDED ? saved : initialWidth;
    collapsed = localStorage.getItem(collapsedKey) === 'true';
  });

  let shortcutKey = $derived(side === 'left' ? '[' : ']');

  function handleExternalExpand(e: Event) {
    const detail = (e as CustomEvent).detail;
    if (detail?.side === side) expand();
  }

  onMount(() => {
    registerShortcut({
      key: shortcutKey,
      description: `Toggle ${title.toLowerCase()} panel`,
      handler(e) {
        e.preventDefault();
        toggle();
      },
    });
    window.addEventListener('floating-panel-expand', handleExternalExpand);
  });

  onDestroy(() => {
    unregisterShortcut(shortcutKey);
    window.removeEventListener('floating-panel-expand', handleExternalExpand);
  });

  let displayWidth = $derived(collapsed ? RAIL_W : width);
</script>

<div
  class="panel-shell"
  class:collapsed
  class:resizing
  class:dimming={willCollapse}
  class:left={side === 'left'}
  class:right={side === 'right'}
  data-panel-side={side}
  style:width="{displayWidth}px"
>
    <!-- Expanded contents — kept mounted so child stores never reset. -->
    <div class="content-wrap" aria-hidden={collapsed}>
      <div class="panel-header">
        <div class="panel-header-inner">
          {#if header}
            {@render header()}
          {:else}
            <span class="panel-title">{title}</span>
          {/if}
        </div>
        <button
          class="collapse-btn"
          onclick={collapse}
          title="Hide {title} panel ({shortcutKey})"
          aria-label="Hide {title} panel"
        >
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
            {#if side === 'left'}
              <path d="M9.5 4L5.5 8l4 4" />
              <path d="M3 4v8" />
            {:else}
              <path d="M6.5 4l4 4-4 4" />
              <path d="M13 4v8" />
            {/if}
          </svg>
        </button>
      </div>
      <div class="panel-content">
        {@render children()}
      </div>
    </div>

    <!-- Collapsed rail — full-height vertical slab. -->
    <button
      class="rail"
      class:left={side === 'left'}
      class:right={side === 'right'}
      onclick={expand}
      aria-hidden={!collapsed}
      tabindex={collapsed ? 0 : -1}
      title="Show {title} panel ({shortcutKey})"
      aria-label="Show {title} panel"
    >
      <span class="rail-stripe"></span>
      <span class="rail-chevron">
        <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
          {#if side === 'left'}<path d="M4 3l4 3-4 3" />{:else}<path d="M8 3L4 6l4 3" />{/if}
        </svg>
      </span>
      <span class="rail-label">{title}</span>
      <span class="rail-chevron">
        <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
          {#if side === 'left'}<path d="M4 3l4 3-4 3" />{:else}<path d="M8 3L4 6l4 3" />{/if}
        </svg>
      </span>
    </button>

    <!-- Resize handle (always present — drag from rail to expand, drag inward to collapse). -->
    <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
    <div
      class="resize-handle"
      role="separator"
      aria-orientation="vertical"
      onmousedown={startResize}
    >
      <span class="resize-grip"></span>
    </div>

  {#if willCollapse && resizing}
    <div class="release-hint" class:left={side === 'left'} class:right={side === 'right'}>
      release to hide
    </div>
  {/if}
</div>

<style>
  /* ───────── Shell ───────── */
  .panel-shell {
    position: absolute;
    z-index: 10;
    color: var(--text-primary);
    overflow: visible;
    will-change: width, top, bottom, border-radius;
    transition:
      width 360ms cubic-bezier(0.32, 0.72, 0, 1),
      top 360ms cubic-bezier(0.32, 0.72, 0, 1),
      bottom 360ms cubic-bezier(0.32, 0.72, 0, 1),
      left 360ms cubic-bezier(0.32, 0.72, 0, 1),
      right 360ms cubic-bezier(0.32, 0.72, 0, 1),
      border-radius 360ms cubic-bezier(0.32, 0.72, 0, 1),
      background 360ms cubic-bezier(0.32, 0.72, 0, 1),
      box-shadow 360ms cubic-bezier(0.32, 0.72, 0, 1),
      opacity 200ms ease;
  }
  .panel-shell.resizing { transition: none; }
  .panel-shell.dimming { opacity: 0.78; }

  /* Expanded — full-bleed slab anchored to the viewport edge. */
  .panel-shell:not(.collapsed) {
    top: 0;
    bottom: 0;
    border-radius: 0;
    background: rgba(35, 38, 54, 0.96);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
  }
  /* Directional inner-edge shadow only — outer edge sits flush against the viewport. */
  .panel-shell.left:not(.collapsed) {
    left: 0;
    box-shadow: 6px 0 22px -6px rgba(0, 0, 0, 0.45);
  }
  .panel-shell.right:not(.collapsed) {
    right: 0;
    box-shadow: -6px 0 22px -6px rgba(0, 0, 0, 0.45);
  }

  /* Collapsed — full-height edge slab. */
  .panel-shell.collapsed {
    top: 0;
    bottom: 0;
    border-radius: 0;
    background: linear-gradient(
      var(--rail-grad-dir, 90deg),
      rgba(35, 38, 54, 0.82) 0%,
      rgba(27, 30, 43, 0.95) 100%
    );
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    box-shadow: none;
  }
  .panel-shell.collapsed.left {
    left: 0;
    --rail-grad-dir: 90deg;
    border-right: 1px solid var(--border-soft);
  }
  .panel-shell.collapsed.right {
    right: 0;
    --rail-grad-dir: -90deg;
    border-left: 1px solid var(--border-soft);
  }

  /* ───────── Expanded content ───────── */
  .content-wrap {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
    border-radius: inherit;
    transition: opacity 180ms ease;
  }
  .panel-shell.collapsed .content-wrap {
    opacity: 0;
    pointer-events: none;
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 6px 8px 12px;
    flex-shrink: 0;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border-soft);
    min-width: 0;
    min-height: 40px;
  }
  .panel-header-inner {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: center;
    overflow: hidden;
  }
  .panel-title {
    font-family: var(--font-display);
    font-size: 12.5px;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: var(--text-primary);
  }
  .collapse-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    width: 24px;
    height: 24px;
    border: none;
    background: none;
    color: var(--text-muted-soft);
    cursor: pointer;
    border-radius: 6px;
    transition: color 140ms ease, background 140ms ease, transform 140ms ease;
  }
  .collapse-btn:hover {
    color: var(--accent);
    background: var(--accent-bg);
    transform: scale(1.06);
  }
  .panel-content {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ───────── Collapsed rail ───────── */
  .rail {
    position: absolute;
    inset: 0;
    width: 100%;
    border: none;
    background: transparent;
    color: var(--text-muted);
    cursor: pointer;
    padding: 18px 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    transition: color 220ms ease, background 220ms ease, opacity 200ms ease;
    border-radius: inherit;
  }
  .panel-shell:not(.collapsed) .rail {
    opacity: 0;
    pointer-events: none;
  }
  .rail:hover {
    color: var(--text-primary);
    background: rgba(76, 141, 255, 0.05);
  }
  .rail:focus-visible {
    outline: none;
    color: var(--text-primary);
    background: rgba(76, 141, 255, 0.07);
  }

  /* Inner accent stripe — wakes up on hover. */
  .rail-stripe {
    position: absolute;
    top: 12px;
    bottom: 12px;
    width: 2px;
    background: linear-gradient(
      to bottom,
      transparent 0%,
      var(--accent) 18%,
      var(--accent) 82%,
      transparent 100%
    );
    opacity: 0;
    transition: opacity 220ms ease, width 220ms ease;
    pointer-events: none;
  }
  .rail.left .rail-stripe { right: 0; }
  .rail.right .rail-stripe { left: 0; }
  .rail:hover .rail-stripe,
  .rail:focus-visible .rail-stripe {
    opacity: 0.85;
    width: 3px;
  }

  .rail-label {
    font-family: var(--font-display);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.32em;
    writing-mode: vertical-rl;
    color: inherit;
    transition: letter-spacing 220ms ease, color 220ms ease;
    user-select: none;
    white-space: nowrap;
  }
  /* On the left rail, rotate so it reads bottom-to-top — the conventional
     orientation for left-side vertical labels. Right rail reads top-to-bottom. */
  .rail.left .rail-label { transform: rotate(180deg); }
  .rail:hover .rail-label,
  .rail:focus-visible .rail-label {
    letter-spacing: 0.42em;
    color: var(--text-primary);
  }

  .rail-chevron {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted-faint);
    opacity: 0.7;
    transition: opacity 220ms ease, transform 220ms ease, color 220ms ease;
  }
  .rail:hover .rail-chevron,
  .rail:focus-visible .rail-chevron {
    opacity: 1;
    color: var(--accent);
  }
  .rail:hover .rail-chevron:first-of-type { transform: translateY(-2px); }
  .rail:hover .rail-chevron:last-of-type { transform: translateY(2px); }

  /* ───────── Resize handle ───────── */
  .resize-handle {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 8px;
    cursor: col-resize;
    z-index: 25;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .panel-shell.left .resize-handle { right: -4px; }
  .panel-shell.right .resize-handle { left: -4px; }
  .resize-grip {
    width: 2px;
    height: 38px;
    background: var(--accent);
    border-radius: 99px;
    opacity: 0;
    transition: opacity 180ms ease, height 180ms ease;
    pointer-events: none;
  }
  .resize-handle:hover .resize-grip,
  .panel-shell.resizing .resize-grip {
    opacity: 0.55;
    height: 60px;
  }

  /* ───────── Release-to-hide hint ───────── */
  .release-hint {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    font-family: var(--font-mono);
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--accent);
    background: rgba(76, 141, 255, 0.14);
    border: 1px solid rgba(76, 141, 255, 0.3);
    border-radius: 99px;
    padding: 5px 12px;
    pointer-events: none;
    white-space: nowrap;
    box-shadow: 0 4px 16px rgba(76, 141, 255, 0.18);
    animation: hint-pop 180ms cubic-bezier(0.32, 0.72, 0, 1);
  }
  .release-hint.left { right: 14px; }
  .release-hint.right { left: 14px; }

  @keyframes hint-pop {
    from { opacity: 0; transform: translateY(-50%) scale(0.85); }
    to { opacity: 1; transform: translateY(-50%) scale(1); }
  }
</style>
