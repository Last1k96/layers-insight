<script lang="ts">
  import GraphCanvas from '../graph/GraphCanvas.svelte';
  import Minimap from '../graph/Minimap.svelte';
  import GraphSearch from '../graph/GraphSearch.svelte';
  import SubSessionNav from '../panels/SubSessionNav.svelte';
  import FloatingPanel from '../panels/FloatingPanel.svelte';
  import QueuePanel from '../panels/QueuePanel.svelte';
  import NodeStatus from '../panels/NodeStatus.svelte';
  import ErrorBanner from '../panels/ErrorBanner.svelte';
  import BottomLogPanel from '../panels/BottomLogPanel.svelte';
  import AccuracyView from '../accuracy/AccuracyView.svelte';
  import BatchQueue from '../panels/BatchQueue.svelte';
  import BisectPanel from '../panels/BisectPanel.svelte';
  import { sessionStore } from '../stores/session.svelte';
  import { graphStore, type NodeStatus as NodeStatusData } from '../stores/graph.svelte';
  import { queueStore } from '../stores/queue.svelte';
  import { bisectStore } from '../stores/bisect.svelte';
  import { cacheMetrics } from '../stores/metrics.svelte';
  import { connect, disconnect, setConnectionCallbacks } from '../ws/client';
  import { onMount, onDestroy } from 'svelte';

  let wsDisconnected = $state(false);
  let showAccuracyView = $state(false);
  let showBatchQueue = $state(false);
  let showBisectPanel = $state(false);

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

  onMount(() => {
    const session = sessionStore.currentSession;
    if (session) {
      graphStore.fetchGraph(session.id);
      connect(session.id);
      restoreSessionTasks();

      setConnectionCallbacks(
        () => { wsDisconnected = true; },
        () => { wsDisconnected = false; },
      );
    }
  });

  onDestroy(() => {
    disconnect();
    queueStore.clear();
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

  <!-- Search overlay -->
  <GraphSearch />

  <!-- Sub-session navigation (Phase 2) -->
  <SubSessionNav />

  <!-- Bisect toolbar button -->
  <button
    class="absolute top-2 left-1/2 -translate-x-1/2 z-10 px-3 py-1.5 bg-[--bg-panel] backdrop-blur border border-[--border-color] rounded-lg shadow text-xs font-medium text-gray-300 hover:text-gray-100 hover:border-blue-500/50 transition-colors"
    class:border-blue-500={bisectStore.isRunning}
    class:text-blue-400={bisectStore.isRunning}
    onclick={() => showBisectPanel = !showBisectPanel}
  >
    {bisectStore.isRunning ? `Bisect (${bisectStore.step}/${bisectStore.totalSteps})` : 'Bisect'}
  </button>

  <!-- Floating panels -->
  <FloatingPanel side="left" title="Queue">
    <QueuePanel />
  </FloatingPanel>

  <FloatingPanel side="right" title="Node Status">
    <NodeStatus
      onshowaccuracy={() => showAccuracyView = true}
      onshowbatchqueue={() => showBatchQueue = true}
    />
  </FloatingPanel>

  {#if graphStore.loading}
    <div class="absolute inset-0 flex items-center justify-center bg-surface-base/80 z-20">
      <div class="text-content-secondary">Loading graph...</div>
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
    onclose={() => showAccuracyView = false}
  />
{/if}

{#if showBatchQueue && graphStore.selectedNode}
  <div class="fixed inset-0 z-[59] bg-black/30" onclick={() => showBatchQueue = false} role="presentation"></div>
  <BatchQueue
    nodeId={graphStore.selectedNode.id}
    nodeName={graphStore.selectedNode.name}
    onclose={() => showBatchQueue = false}
  />
{/if}

{#if showBisectPanel}
  <BisectPanel onclose={() => showBisectPanel = false} />
{/if}
