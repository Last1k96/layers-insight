<script lang="ts">
  import GraphCanvas from '../graph/GraphCanvas.svelte';
  import Minimap from '../graph/Minimap.svelte';
  import GraphSearch from '../graph/GraphSearch.svelte';
  import AccuracyToggle from '../graph/AccuracyToggle.svelte';
  import SubSessionNav from '../panels/SubSessionNav.svelte';
  import FloatingPanel from '../panels/FloatingPanel.svelte';
  import QueuePanel from '../panels/QueuePanel.svelte';
  import NodeStatus from '../panels/NodeStatus.svelte';
  import ErrorBanner from '../panels/ErrorBanner.svelte';
  import BottomLogPanel from '../panels/BottomLogPanel.svelte';
  import AccuracyView from '../accuracy/AccuracyView.svelte';
  import BatchQueue from '../panels/BatchQueue.svelte';
  import ShortcutsHelp from '../panels/ShortcutsHelp.svelte';
  import BisectPanel from '../panels/BisectPanel.svelte';
  import { sessionStore } from '../stores/session.svelte';
  import { graphStore, type NodeStatus as NodeStatusData } from '../stores/graph.svelte';
  import { queueStore } from '../stores/queue.svelte';
  import { bisectStore } from '../stores/bisect.svelte';
  import { logStore } from '../stores/log.svelte';
  import { cacheMetrics } from '../stores/metrics.svelte';
  import { connect, disconnect, setConnectionCallbacks } from '../ws/client';
  import { refreshRenderer } from '../graph/renderer';
  import { installShortcuts, uninstallShortcuts, registerShortcut, toggleHelp, setHelpVisible } from '../shortcuts';
  import { onMount, onDestroy } from 'svelte';

  let { onback = () => {} }: { onback?: () => void } = $props();

  let wsDisconnected = $state(false);
  let renamingSession = $state(false);
  let renameValue = $state('');
  let showAccuracyView = $state(false);
  let accuracyOutputIndex = $state(0);
  let showBatchQueue = $state(false);
  let showBisectPanel = $state(false);
  let bisectEndNodeId = $state<string | null>(null);
  let bisectEndNodeName = $state<string | null>(null);
  let bisectInitialSearchFor = $state<'accuracy_drop' | 'compilation_failure' | null>(null);
  let batchQueueInitialMode = $state<'all' | 'by-type' | 'uninferred' | 'from-selection'>('all');
  let loadingPct = $state(0);

  // Animate progress bar — smooth interpolation during layout phase
  $effect(() => {
    const stage = graphStore.loadingStage;
    const basePct: Record<string, number> = { loading_model: 2, extracting: 4, layout: 5, shapes: 95 };
    const base = basePct[stage] ?? 0;

    if (stage === 'layout' && graphStore.layoutStartedAt > 0) {
      // Animate from 5% → 93% over the estimated duration (ease-out curve)
      const est = graphStore.layoutEstimateMs;
      const start = graphStore.layoutStartedAt;
      const interval = setInterval(() => {
        const elapsed = performance.now() - start;
        const t = Math.min(elapsed / est, 1);
        const eased = 1 - (1 - t) ** 2;
        loadingPct = 5 + eased * 88;
      }, 100);
      return () => clearInterval(interval);
    } else {
      loadingPct = base;
    }
  });

  let sessionName = $derived(sessionStore.currentSession?.info.model_name ?? 'Queue');

  function startRenameSession() {
    renamingSession = true;
    renameValue = sessionName;
  }

  async function commitRenameSession() {
    const trimmed = renameValue.trim();
    const sessionId = sessionStore.currentSession?.id;
    if (trimmed && sessionId) {
      await sessionStore.renameSession(sessionId, trimmed);
      if (sessionStore.currentSession) {
        sessionStore.currentSession.info.model_name = trimmed;
      }
    }
    renamingSession = false;
    renameValue = '';
  }

  function cancelRenameSession() {
    renamingSession = false;
    renameValue = '';
  }

  function handleRenameKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault();
      commitRenameSession();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      cancelRenameSession();
    }
  }

  function restoreSessionTasks(): void {
    const session = sessionStore.currentSession;
    if (!session || !session.tasks || session.tasks.length === 0) return;

    queueStore.loadTasks(session.tasks);

    for (const task of session.tasks) {
      const nodeStatus: NodeStatusData = {
        status: task.status,
        taskId: task.task_id,
        metrics: task.metrics,
        mainResult: task.main_result,
        refResult: task.ref_result,
        perOutputMetrics: task.per_output_metrics,
        perOutputMainResults: task.per_output_main_results,
        perOutputRefResults: task.per_output_ref_results,
        errorDetail: task.error_detail,
      };
      graphStore.updateNodeStatus(task.node_id, nodeStatus, task.sub_session_id);

      if (task.status === 'success' && task.metrics) {
        cacheMetrics(task.task_id, {
          metrics: task.metrics,
          main_result: task.main_result,
          ref_result: task.ref_result,
        });
      }
    }
  }

  onMount(async () => {
    const session = sessionStore.currentSession;
    if (session) {
      await connect(session.id);
      graphStore.fetchGraph(session.id);
      restoreSessionTasks();

      // Restore bisect state if a bisection was active before page refresh
      bisectStore.fetchStatus(session.id);

      setConnectionCallbacks(
        () => { wsDisconnected = true; },
        () => { wsDisconnected = false; },
      );
    }

    // Install global keyboard shortcuts
    installShortcuts();

    registerShortcut({
      key: 'A',
      description: 'Toggle accuracy view',
      handler(e) {
        e.preventDefault();
        graphStore.accuracyViewActive = !graphStore.accuracyViewActive;
        refreshRenderer();
      },
    });

    registerShortcut({
      key: 'Escape',
      description: 'Close overlay / deselect node / close search',
      allowInInput: true,
      handler(e) {
        // Priority: help overlay > accuracy view > search > deselect node
        // (AccuracyView handles its own Escape via svelte:window)
        if (graphStore.searchVisible) {
          graphStore.searchVisible = false;
          graphStore.searchResults = [];
          refreshRenderer();
          return;
        }
        if (showAccuracyView) {
          showAccuracyView = false;
          return;
        }
        if (showBatchQueue) {
          showBatchQueue = false;
          return;
        }
        if (graphStore.selectedEdgeIndex !== null) {
          graphStore.selectEdge(null);
          refreshRenderer();
          return;
        }
        if (graphStore.selectedNodeId) {
          graphStore.selectNode(null);
          refreshRenderer();
          return;
        }
      },
    });

    registerShortcut({
      key: '?',
      description: 'Show keyboard shortcuts help',
      handler(e) {
        e.preventDefault();
        toggleHelp();
      },
    });

    registerShortcut({
      key: 'Ctrl+F',
      description: 'Open search',
      allowInInput: true,
      handler(e) {
        e.preventDefault();
        graphStore.searchVisible = !graphStore.searchVisible;
      },
    });
  });

  onDestroy(() => {
    disconnect();
    queueStore.clear();
    uninstallShortcuts();
  });
</script>

<div class="relative w-full h-full">
  {#if wsDisconnected}
    <div class="absolute top-0 left-0 right-0 z-50">
      <ErrorBanner message="Connection lost. Reconnecting..." />
    </div>
  {/if}

  <!-- Graph fills entire background -->
  <GraphCanvas />

  <!-- Minimap -->
  <Minimap />


  <!-- Accuracy toggle (top-right of graph canvas) -->
  <AccuracyToggle />

  <!-- Search overlay -->
  <GraphSearch />

  <!-- Sub-session navigation (Phase 2) -->
  <SubSessionNav />

  <!-- Floating panels -->
  <FloatingPanel side="left" title="Queue">
    {#snippet header()}
      <button
        class="back-btn"
        onclick={onback}
        title="Back to sessions"
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M10 12L6 8l4-4" />
        </svg>
      </button>
      {#if renamingSession}
        <!-- svelte-ignore a11y_autofocus -->
        <input
          class="rename-input"
          type="text"
          bind:value={renameValue}
          onkeydown={handleRenameKeydown}
          onblur={commitRenameSession}
          onclick={(e) => e.stopPropagation()}
          autofocus
        />
      {:else}
        <span class="session-name-header">
          <span class="session-name-text">{sessionName}</span><button
            class="rename-btn-header"
            onclick={(e) => { e.stopPropagation(); startRenameSession(); }}
            title="Rename session"
          ><svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor"><path d="M13.586 3.586a2 2 0 112.828 2.828l-.793.793-2.828-2.828.793-.793zM11.379 5.793L3 14.172V17h2.828l8.38-8.379-2.83-2.828z" /></svg></button>
        </span>
      {/if}
    {/snippet}
    <QueuePanel
      onbatchinfer={() => { batchQueueInitialMode = 'all'; showBatchQueue = true; }}
      onbisect={() => { bisectEndNodeId = null; bisectEndNodeName = null; bisectInitialSearchFor = null; showBisectPanel = !showBisectPanel; }}
      ontogglelog={() => logStore.toggle()}
    />
  </FloatingPanel>

  <FloatingPanel side="right" title="Node Status">
    <NodeStatus
      onshowaccuracy={(outputIdx?: number) => { accuracyOutputIndex = outputIdx ?? 0; showAccuracyView = true; }}
      onshowbatchqueue={() => { batchQueueInitialMode = 'from-selection'; showBatchQueue = true; }}
      onbisect={() => {
        const node = graphStore.selectedNode;
        const status = graphStore.selectedNodeStatus;
        if (!node) return;
        bisectEndNodeId = node.id;
        bisectEndNodeName = node.name;
        bisectInitialSearchFor = status?.status === 'failed' ? 'compilation_failure' : 'accuracy_drop';
        showBisectPanel = true;
      }}
    />
  </FloatingPanel>

  {#if graphStore.loading}
    <div class="absolute inset-0 flex items-center justify-center bg-surface-base/80 z-20">
      <div class="flex flex-col items-center gap-3">
        <div class="w-48 h-1.5 bg-white/10 rounded-full overflow-hidden">
          <div class="h-full bg-blue-500 rounded-full transition-all duration-300" style="width: {loadingPct}%"></div>
        </div>
        <div class="text-content-secondary text-sm">{graphStore.loadingDetail || 'Connecting…'}</div>
      </div>
    </div>
  {/if}
</div>

<!-- Bottom log panel — outside relative container so z-index works against sigma canvases -->
<BottomLogPanel />

<!-- Fullscreen overlays — rendered outside all containers so fixed positioning works -->
{#if showAccuracyView && graphStore.selectedNodeStatus?.taskId && graphStore.selectedNode}
  <AccuracyView
    taskId={graphStore.selectedNodeStatus.taskId}
    nodeId={graphStore.selectedNode.name}
    outputIndex={accuracyOutputIndex}
    onclose={() => showAccuracyView = false}
  />
{/if}

{#if showBatchQueue}
  <div class="fixed inset-0 z-[59] bg-black/30" onclick={() => showBatchQueue = false} role="presentation"></div>
  <BatchQueue
    nodeId={graphStore.selectedNode?.id ?? null}
    nodeName={graphStore.selectedNode?.name ?? null}
    initialMode={batchQueueInitialMode}
    onclose={() => showBatchQueue = false}
  />
{/if}

<!-- Keyboard shortcuts help overlay -->
<ShortcutsHelp />

<style>
  .back-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: none;
    background: none;
    cursor: pointer;
    padding: 0.3rem;
    margin-right: 0.25rem;
    border-radius: 0.375rem;
    color: var(--text-secondary);
    opacity: 0.5;
    transition: opacity 0.15s ease, color 0.15s ease, background 0.15s ease;
  }

  .back-btn:hover {
    opacity: 1;
    color: var(--text-primary);
    background: rgba(76, 141, 255, 0.1);
  }

  .session-name-header {
    overflow: hidden;
    white-space: nowrap;
  }

  .session-name-text {
    font-size: 13px;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: var(--text-primary);
  }

  .rename-btn-header {
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

  .session-name-header:hover .rename-btn-header {
    opacity: 0.6;
  }

  .rename-btn-header:hover {
    opacity: 1 !important;
    color: #4C8DFF;
    background: rgba(76, 141, 255, 0.1);
  }

  .rename-input {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
    background: var(--bg-primary);
    border: 1px solid #4C8DFF;
    border-radius: 0.25rem;
    padding: 0 0.3rem;
    outline: none;
    width: 100%;
  }
</style>

{#if showBisectPanel}
  <BisectPanel
    onclose={() => { showBisectPanel = false; bisectEndNodeId = null; bisectEndNodeName = null; bisectInitialSearchFor = null; }}
    endNodeId={bisectEndNodeId}
    endNodeName={bisectEndNodeName}
    initialSearchFor={bisectInitialSearchFor}
  />
{/if}
