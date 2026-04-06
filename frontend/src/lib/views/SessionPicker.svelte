<script lang="ts">
  import { sessionStore } from '../stores/session.svelte';
  import { onMount } from 'svelte';

  let {
    onsessionselected,
    onnewsession,
    onclonesession,
    oncompare,
  }: {
    onsessionselected: (id: string) => void;
    onnewsession: () => void;
    onclonesession: (id: string) => void;
    oncompare: (a: string, b: string) => void;
  } = $props();

  let confirmingDelete: string | null = $state(null);
  let selectedIndex = $state(-1);
  let compareMode = $state(false);
  let compareSelection = $state<string[]>([]);
  let renamingId: string | null = $state(null);
  let renameValue = $state('');

  function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  function completionColor(success: number, total: number): string {
    if (total === 0) return 'var(--edge)';
    const pct = success / total;
    if (pct >= 1) return '#34C77B';
    if (pct >= 0.5) return '#4C8DFF';
    if (pct > 0) return '#E8A849';
    return 'var(--edge)';
  }

  function handleKeydown(e: KeyboardEvent) {
    const len = sessionStore.sessions.length;
    if (len === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, len - 1);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, 0);
    } else if (e.key === 'Enter' && selectedIndex >= 0 && selectedIndex < len) {
      e.preventDefault();
      onsessionselected(sessionStore.sessions[selectedIndex].id);
    }
  }

  function handleDelete(e: MouseEvent, sessionId: string) {
    e.stopPropagation();
    if (confirmingDelete === sessionId) {
      sessionStore.deleteSession(sessionId);
      confirmingDelete = null;
    } else {
      confirmingDelete = sessionId;
    }
  }

  function cancelDelete(e: MouseEvent) {
    e.stopPropagation();
    confirmingDelete = null;
  }

  function handleClone(e: MouseEvent, sessionId: string) {
    e.stopPropagation();
    onclonesession(sessionId);
  }

  function startRename(e: MouseEvent, session: { id: string; model_name: string }) {
    e.stopPropagation();
    renamingId = session.id;
    renameValue = session.model_name;
  }

  async function commitRename() {
    if (!renamingId) return;
    const trimmed = renameValue.trim();
    if (trimmed) {
      await sessionStore.renameSession(renamingId, trimmed);
    }
    renamingId = null;
    renameValue = '';
  }

  function cancelRename() {
    renamingId = null;
    renameValue = '';
  }

  function handleRenameKeydown(e: KeyboardEvent) {
    e.stopPropagation();
    if (e.key === 'Enter') {
      e.preventDefault();
      commitRename();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      cancelRename();
    }
  }

  function toggleCompareSelect(e: MouseEvent, sessionId: string) {
    e.stopPropagation();
    if (compareSelection.includes(sessionId)) {
      compareSelection = compareSelection.filter(id => id !== sessionId);
    } else if (compareSelection.length < 2) {
      compareSelection = [...compareSelection, sessionId];
    }
  }

  function startCompare() {
    if (compareSelection.length === 2) {
      oncompare(compareSelection[0], compareSelection[1]);
    }
  }

  function toggleCompareMode() {
    compareMode = !compareMode;
    if (!compareMode) compareSelection = [];
  }

  onMount(() => {
    sessionStore.fetchSessions().then(() => {
      const sessions = sessionStore.sessions;
      if (sessions.length === 0) return;
      const lastId = sessionStore.lastSessionId;
      const idx = lastId ? sessions.findIndex(s => s.id === lastId) : -1;
      selectedIndex = idx >= 0 ? idx : 0;
    });
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  });
</script>

<div class="picker-root">
  <div class="picker-inner">
    <!-- Header -->
    <header class="picker-header">
      <h1 class="picker-title">Layers Insight</h1>
      <p class="picker-tagline">Neural Network Graph Debugger</p>
    </header>

    {#if sessionStore.loading}
      <div class="loading-state">
        <svg class="spin" width="16" height="16" viewBox="0 0 16 16" fill="none">
          <circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="1.5" stroke-dasharray="28" stroke-dashoffset="8" stroke-linecap="round" />
        </svg>
        <span>Loading sessions...</span>
      </div>

    {:else if sessionStore.sessions.length === 0}
      <div class="empty-state">
        <div class="empty-icon-wrap">
          <div class="empty-pulse"></div>
          <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round">
            <circle cx="12" cy="12" r="2.5" />
            <circle cx="4.5" cy="7" r="1.5" />
            <circle cx="19.5" cy="7" r="1.5" />
            <circle cx="4.5" cy="17" r="1.5" />
            <circle cx="19.5" cy="17" r="1.5" />
            <line x1="10" y1="10.5" x2="6" y2="8" />
            <line x1="14" y1="10.5" x2="18" y2="8" />
            <line x1="10" y1="13.5" x2="6" y2="16" />
            <line x1="14" y1="13.5" x2="18" y2="16" />
          </svg>
        </div>
        <p class="empty-title">No sessions yet</p>
        <p class="empty-hint">Create a session to start debugging your model</p>
        <button class="cta-btn" onclick={onnewsession}>
          Start New Session
        </button>
      </div>

    {:else}
      <!-- Session list -->
      <div class="session-list">
        {#each sessionStore.sessions as session, i (session.id)}
          {@const total = session.task_count}
          {@const success = session.success_count}
          {@const completion = total > 0 ? success / total : 0}
          {@const accent = completionColor(success, total)}
          <!-- svelte-ignore a11y_no_static_element_interactions -->
          {@const isLast = session.id === sessionStore.lastSessionId}
          <div
            class="session-card"
            class:is-selected={i === selectedIndex}
            class:is-last={isLast}
            class:is-compare-selected={compareMode && compareSelection.includes(session.id)}
            style="--accent-bar: {accent};"
            role="button"
            tabindex="0"
            onclick={() => compareMode ? toggleCompareSelect(new MouseEvent('click'), session.id) : onsessionselected(session.id)}
            onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); compareMode ? toggleCompareSelect(new MouseEvent('click'), session.id) : onsessionselected(session.id); }}}
          >
            <!-- Left accent bar -->
            <div class="card-accent-bar"></div>

            <div class="card-body">
              <!-- Top row: name + task count -->
              <div class="card-top-row">
                {#if compareMode}
                  <div class="compare-checkbox" class:checked={compareSelection.includes(session.id)}>
                    {#if compareSelection.includes(session.id)}
                      <svg width="10" height="10" viewBox="0 0 20 20" fill="currentColor">
                        <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                      </svg>
                    {/if}
                  </div>
                {/if}
                {#if renamingId === session.id}
                  <!-- svelte-ignore a11y_autofocus -->
                  <input
                    class="rename-input"
                    type="text"
                    bind:value={renameValue}
                    onkeydown={handleRenameKeydown}
                    onblur={commitRename}
                    onclick={(e) => e.stopPropagation()}
                    autofocus
                  />
                {:else}
                  <span class="model-name-wrap">
                    <span class="model-name-text">{session.model_name}</span><button
                      class="rename-btn"
                      onclick={(e) => startRename(e, session)}
                      title="Rename session"
                    ><svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor"><path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" /></svg></button>
                  </span>
                {/if}
                {#if isLast}
                  <span class="last-badge">recent</span>
                {/if}
                <span class="task-badge" style="color: {accent}">
                  {success}<span class="task-sep">/</span>{total}
                </span>
              </div>

              <!-- Middle row: devices + meta -->
              <div class="card-meta-row">
                <div class="device-group">
                  <span class="device-pill main">{session.main_device}</span>
                  <span class="vs-label">vs</span>
                  <span class="device-pill ref">{session.ref_device}</span>
                  {#if session.sub_sessions?.length > 0}
                    <span class="sub-count">{session.sub_sessions.length} sub</span>
                  {/if}
                </div>
                <div class="card-date-size">
                  {new Date(session.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                  {#if session.folder_size > 0}
                    <span class="size-dot">&middot;</span>
                    {formatSize(session.folder_size)}
                  {/if}
                </div>
              </div>

              <!-- Progress bar -->
              {#if total > 0}
                <div class="progress-track">
                  <div class="progress-fill" style="width: {completion * 100}%; background: {accent};"></div>
                </div>
              {/if}
            </div>

            <!-- Hover actions -->
            {#if !compareMode}
              <div class="card-actions">
                {#if confirmingDelete === session.id}
                  <button class="act-btn act-cancel" onclick={cancelDelete}>Cancel</button>
                  <button class="act-btn act-confirm" onclick={(e) => handleDelete(e, session.id)}>Delete</button>
                {:else}
                  <button
                    class="act-btn act-icon"
                    onclick={(e) => handleClone(e, session.id)}
                    title="Clone session"
                  >
                    <svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor">
                      <path d="M7 9a2 2 0 012-2h6a2 2 0 012 2v6a2 2 0 01-2 2H9a2 2 0 01-2-2V9z" />
                      <path d="M5 3a2 2 0 00-2 2v6a2 2 0 002 2V5h8a2 2 0 00-2-2H5z" />
                    </svg>
                  </button>
                  <button
                    class="act-btn act-icon act-delete"
                    onclick={(e) => handleDelete(e, session.id)}
                    title="Delete session"
                  >
                    <svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor">
                      <path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd" />
                    </svg>
                  </button>
                {/if}
              </div>
            {/if}
          </div>
        {/each}
      </div>

      <!-- Footer actions -->
      <div class="picker-footer">
        <button class="new-btn" onclick={onnewsession}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
            <line x1="7" y1="2" x2="7" y2="12" />
            <line x1="2" y1="7" x2="12" y2="7" />
          </svg>
          New Session
        </button>
        {#if sessionStore.sessions.length >= 2}
          <button
            class="compare-toggle"
            class:active={compareMode}
            onclick={toggleCompareMode}
          >
            {compareMode ? 'Cancel' : 'Compare'}
          </button>
        {/if}
      </div>

      {#if compareMode && compareSelection.length === 2}
        <button class="compare-submit" onclick={startCompare}>
          Compare Selected Sessions
        </button>
      {:else if compareMode}
        <div class="compare-hint">
          Select 2 sessions to compare ({compareSelection.length}/2)
        </div>
      {/if}
    {/if}

    {#if sessionStore.error}
      <div class="error-bar">
        <svg class="error-icon" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M8 1.5L1.5 13.5h13L8 1.5z" stroke-linejoin="round" />
          <line x1="8" y1="6.5" x2="8" y2="9.5" stroke-linecap="round" />
          <circle cx="8" cy="11.25" r="0.5" fill="currentColor" stroke="none" />
        </svg>
        <span class="error-text">{sessionStore.error}</span>
      </div>
    {/if}
  </div>
</div>

<style>
  /* ── Root & Background ── */
  .picker-root {
    flex: 1;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 2rem;
    padding-top: 14vh;
    background: var(--bg-primary);
    background-image:
      radial-gradient(ellipse 60% 50% at 50% 0%, rgba(76, 141, 255, 0.04) 0%, transparent 100%),
      radial-gradient(circle, rgba(76, 141, 255, 0.045) 1px, transparent 1px);
    background-size: 100% 100%, 28px 28px;
    min-height: 100%;
    overflow-y: auto;
  }

  .picker-inner {
    max-width: 40rem;
    width: 100%;
  }

  /* ── Header ── */
  .picker-header {
    margin-bottom: 2.5rem;
    animation: fade-down 0.5s ease-out both;
  }

  .picker-title {
    font-family: var(--font-display);
    font-size: 2.75rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: var(--text-primary);
    margin: 0;
  }

  .picker-tagline {
    margin: 0.5rem 0 0;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-secondary);
    opacity: 0.4;
  }

  /* ── Loading ── */
  .loading-state {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    justify-content: center;
    padding: 4rem 0;
    color: var(--text-secondary);
    font-size: 0.875rem;
  }

  .spin {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* ── Empty State ── */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 4rem 0 2rem;
    animation: fade-down 0.5s ease-out 0.15s both;
  }

  .empty-icon-wrap {
    position: relative;
    color: var(--text-secondary);
    opacity: 0.2;
    margin-bottom: 1.5rem;
  }

  .empty-pulse {
    position: absolute;
    inset: -8px;
    border-radius: 50%;
    background: rgba(76, 141, 255, 0.15);
    animation: node-breathe 3s ease-in-out infinite;
  }

  .empty-title {
    margin: 0;
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-secondary);
    opacity: 0.6;
  }

  .empty-hint {
    margin: 0.35rem 0 0;
    font-size: 0.8rem;
    color: var(--text-secondary);
    opacity: 0.35;
  }

  .cta-btn {
    margin-top: 2rem;
    padding: 0.75rem 2rem;
    background: rgba(76, 141, 255, 0.12);
    border: 1px solid rgba(76, 141, 255, 0.2);
    border-radius: 0.75rem;
    color: #4C8DFF;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .cta-btn:hover {
    background: rgba(76, 141, 255, 0.18);
    border-color: rgba(76, 141, 255, 0.35);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(76, 141, 255, 0.15);
  }

  /* ── Session List ── */
  .session-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    max-height: 60vh;
    overflow-y: auto;
    padding-right: 6px;
    margin-bottom: 1rem;
    scrollbar-width: thin;
    scrollbar-color: #4A5070 transparent;
  }

  .session-list::-webkit-scrollbar { width: 6px; }
  .session-list::-webkit-scrollbar-track { background: transparent; }
  .session-list::-webkit-scrollbar-thumb { background: #4A5070; border-radius: 99px; }
  .session-list::-webkit-scrollbar-thumb:hover { background: #5E6590; }

  /* ── Session Card ── */
  .session-card {
    display: flex;
    align-items: stretch;
    background: var(--bg-panel);
    border-radius: 0.75rem;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    flex-shrink: 0;
    transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
  }

  .session-card:hover {
    transform: translateY(-1px);
    background: var(--bg-menu);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(76, 141, 255, 0.06);
  }

  .session-card.is-selected {
    background: var(--bg-menu);
    box-shadow: 0 2px 16px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(76, 141, 255, 0.25);
  }

  .session-card.is-last {
    background: rgba(76, 141, 255, 0.04);
    box-shadow: 0 0 0 1px rgba(76, 141, 255, 0.15);
  }

  .session-card.is-last .card-accent-bar {
    background: #4C8DFF;
    opacity: 0.9;
  }

  .session-card.is-compare-selected {
    box-shadow: 0 0 0 2px #4C8DFF;
  }

  /* ── Accent Bar ── */
  .card-accent-bar {
    width: 3px;
    flex-shrink: 0;
    background: var(--border-color);
    border-radius: 3px 0 0 3px;
    transition: width 0.2s ease, opacity 0.2s ease;
    opacity: 0.6;
  }

  .session-card:hover .card-accent-bar {
    width: 4px;
    opacity: 1;
  }

  /* ── Card Body ── */
  .card-body {
    flex: 1;
    min-width: 0;
    padding: 0.875rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .card-top-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .model-name-wrap {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    white-space: nowrap;
  }

  .model-name-text {
    font-weight: 500;
    font-size: 0.95rem;
    color: var(--text-primary);
  }

  .rename-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    vertical-align: middle;
    border: none;
    background: none;
    cursor: pointer;
    padding: 0.35rem;
    margin-left: 0.25rem;
    border-radius: 0.25rem;
    color: var(--text-secondary);
    opacity: 0;
    transition: opacity 0.15s ease, color 0.15s ease, background 0.15s ease;
  }

  .session-card:hover .rename-btn {
    opacity: 0.6;
  }

  .rename-btn:hover {
    opacity: 1 !important;
    color: #4C8DFF;
    background: rgba(76, 141, 255, 0.1);
  }

  .rename-input {
    flex: 1;
    font-weight: 500;
    font-size: 0.95rem;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid #4C8DFF;
    border-radius: 0.25rem;
    padding: 0 0.3rem;
    outline: none;
    min-width: 0;
  }

  .last-badge {
    flex-shrink: 0;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    background: rgba(76, 141, 255, 0.12);
    color: rgba(76, 141, 255, 0.75);
  }

  .task-badge {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 500;
    flex-shrink: 0;
    letter-spacing: 0.02em;
  }

  .task-sep {
    opacity: 0.4;
    margin: 0 0.05em;
  }

  .card-meta-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }

  .device-group {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    flex-shrink: 0;
  }

  .device-pill {
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.1rem 0.45rem;
    border-radius: 4px;
    letter-spacing: 0.02em;
  }

  .device-pill.main {
    background: rgba(76, 141, 255, 0.1);
    color: rgba(76, 141, 255, 0.8);
  }

  .device-pill.ref {
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-secondary);
  }

  .vs-label {
    font-size: 0.65rem;
    color: var(--text-secondary);
    opacity: 0.35;
    font-style: italic;
  }

  .sub-count {
    font-size: 0.65rem;
    color: var(--text-secondary);
    opacity: 0.45;
    margin-left: 0.25rem;
  }

  .card-date-size {
    font-size: 0.7rem;
    color: var(--text-secondary);
    opacity: 0.5;
    white-space: nowrap;
  }

  .size-dot {
    margin: 0 0.25rem;
    opacity: 0.4;
  }

  /* ── Progress Bar ── */
  .progress-track {
    height: 2px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 1px;
    overflow: hidden;
    margin-top: 0.15rem;
  }

  .progress-fill {
    height: 100%;
    border-radius: 1px;
    transition: width 0.4s ease;
  }

  /* ── Hover Actions ── */
  .card-actions {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding-right: 0.5rem;
    flex-shrink: 0;
    opacity: 0;
    transition: opacity 0.15s ease;
  }

  .session-card:hover .card-actions {
    opacity: 1;
  }

  .act-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    border: none;
    background: none;
    cursor: pointer;
    transition: color 0.15s ease, background 0.15s ease;
    border-radius: 0.375rem;
  }

  .act-icon {
    padding: 0.35rem;
    color: var(--text-secondary);
  }

  .act-icon:hover {
    color: #4C8DFF;
    background: rgba(76, 141, 255, 0.1);
  }

  .act-delete:hover {
    color: #E54D4D;
    background: rgba(229, 77, 77, 0.1);
  }

  .act-cancel, .act-confirm {
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.25rem 0.5rem;
  }

  .act-cancel {
    color: var(--text-secondary);
  }

  .act-cancel:hover {
    color: var(--text-primary);
    background: rgba(255, 255, 255, 0.06);
  }

  .act-confirm {
    color: #E54D4D;
    background: rgba(229, 77, 77, 0.12);
  }

  .act-confirm:hover {
    background: rgba(229, 77, 77, 0.2);
  }

  /* ── Compare Checkbox ── */
  .compare-checkbox {
    width: 1rem;
    height: 1rem;
    border: 2px solid var(--text-secondary);
    border-radius: 0.25rem;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;
    color: white;
    opacity: 0.5;
  }

  .compare-checkbox.checked {
    background: #4C8DFF;
    border-color: #4C8DFF;
    opacity: 1;
  }

  /* ── Footer Actions ── */
  .picker-footer {
    display: flex;
    gap: 0.5rem;
  }

  .new-btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    background: rgba(76, 141, 255, 0.08);
    border: 1px solid rgba(76, 141, 255, 0.15);
    border-radius: 0.75rem;
    color: rgba(76, 141, 255, 0.85);
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.18s ease;
  }

  .new-btn:hover {
    background: rgba(76, 141, 255, 0.14);
    border-color: rgba(76, 141, 255, 0.3);
    color: #4C8DFF;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(76, 141, 255, 0.1);
  }

  .compare-toggle {
    padding: 0.75rem 1.25rem;
    border-radius: 0.75rem;
    border: none;
    background: none;
    color: var(--text-secondary);
    opacity: 0.5;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .compare-toggle:hover {
    opacity: 0.8;
    background: rgba(255, 255, 255, 0.04);
  }

  .compare-toggle.active {
    color: #4C8DFF;
    opacity: 1;
    background: rgba(76, 141, 255, 0.08);
  }

  .compare-submit {
    width: 100%;
    margin-top: 0.5rem;
    padding: 0.75rem;
    background: #4C8DFF;
    border: none;
    border-radius: 0.75rem;
    color: white;
    font-weight: 500;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.18s ease;
  }

  .compare-submit:hover {
    background: #6BA1FF;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(76, 141, 255, 0.25);
  }

  .compare-submit:active {
    transform: scale(0.99);
  }

  .compare-hint {
    text-align: center;
    font-size: 0.8rem;
    color: var(--text-secondary);
    opacity: 0.5;
    margin-top: 0.5rem;
  }

  /* ── Error ── */
  .error-bar {
    margin-top: 1rem;
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.85rem 1rem;
    background: rgba(229, 77, 77, 0.12);
    border: 1px solid rgba(229, 77, 77, 0.35);
    border-left: 3px solid #E54D4D;
    border-radius: 0.75rem;
    color: #F0A0A0;
    font-size: 0.85rem;
    line-height: 1.45;
    animation: error-slide-in 0.35s ease-out both;
    box-shadow: 0 2px 12px rgba(229, 77, 77, 0.1), inset 0 0 20px rgba(229, 77, 77, 0.03);
  }

  .error-icon {
    flex-shrink: 0;
    color: #E54D4D;
    margin-top: 1px;
    animation: error-pulse 2s ease-in-out infinite;
  }

  .error-text {
    flex: 1;
    word-break: break-word;
  }

  @keyframes error-slide-in {
    from {
      opacity: 0;
      transform: translateY(6px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes error-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
</style>
