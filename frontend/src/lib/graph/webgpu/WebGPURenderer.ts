/**
 * Main WebGPU renderer orchestrating nodes, edges, and text pipelines.
 * Manages device init, frame loop, and buffer updates.
 */
import type { GraphData, GraphNode, AccuracyMetrics } from '../../stores/types';
import type { NodeStatus } from '../../stores/graph.svelte';
import { configStore } from '../../stores/config.svelte';
import { isLightNodeColor, STATUS_COLORS } from '../opColors';
import { getAccuracyColorRgb, type AccuracyMetricKey, type AccuracyRange } from '../../utils/accuracyColors';
import { buildCameraMatrix, CLEAR_COLOR, EDGE_COLOR, NODE_FLOATS, NODE_RADIUS, GRAPH_FONT_SIZE, GHOST_VERTEX_FLOATS } from './types';
import { createNodesPipeline, updateNodeInstances, drawNodes } from './nodesPipeline';
import type { NodesPipelineState } from './nodesPipeline';
import {
  createEdgesPipeline,
  uploadEdgeData,
  uploadEdgeColors,
  useCachedEdgeColors,
  updateEdgeViewport,
  drawEdges,
  buildEdgeGeometry,
  buildEdgeColors,
  buildPathGeometry,
} from './edgesPipeline';
import type { AccuracyPath, EdgeGeometry, EdgesPipelineState } from './edgesPipeline';
import { createTextPipeline, updateGlyphInstances, updateTextAlpha, drawText } from './textPipeline';
import type { TextPipelineState } from './textPipeline';
import { createGhostPipeline, updateGhostVertices, updateGhostViewport, drawGhosts } from './ghostPipeline';
import type { GhostPipelineState } from './ghostPipeline';
import { createTextAtlas, measureText } from './textAtlas';
import type { TextAtlasData } from './textAtlas';
import { SpatialGrid } from './hitTest';
import { EdgeHitIndex } from './edgeHitTest';
import type { FrameStats } from '../../perf/instrumentation';
import type { MinimapTarget } from './MinimapTarget';

export class WebGPURenderer {
  readonly canvas: HTMLCanvasElement;
  readonly hitGrid = new SpatialGrid();
  readonly edgeHitIndex = new EdgeHitIndex();

  readonly device: GPUDevice;
  readonly format: GPUTextureFormat;
  private context: GPUCanvasContext;
  readonly cameraBuffer: GPUBuffer;
  private nodesPipeline: NodesPipelineState;
  private edgesPipeline: EdgesPipelineState;
  /** Overlay edges drawn on top of nodes (accuracy paths). Shares pipeline/bindGroup with edgesPipeline. */
  private overlayEdgesPipeline: EdgesPipelineState | null = null;
  private textPipeline: TextPipelineState;
  private ghostPipeline: GhostPipelineState;
  private atlas: TextAtlasData;
  private dirty = true;
  private animFrameId: number | null = null;
  private resizeObserver: ResizeObserver;
  private msaaTexture: GPUTexture | null = null;
  private msaaView: GPUTextureView | null = null;
  private currentZoom = 1;
  private graphData: GraphData | null = null;
  private nodeSizeFn: ((id: string) => { width: number; height: number }) | null = null;
  /** Cached edge tessellation. Built once per graph, color-only updates leave it untouched. */
  private edgeGeometry: EdgeGeometry | null = null;

  /** Reusable Float32Array for node instance data, grown geometrically as needed. */
  private nodeDataScratch: Float32Array = new Float32Array(256 * NODE_FLOATS);
  /** Reusable Float32Array for glyph instance data, grown geometrically as needed. */
  private glyphDataScratch: Float32Array = new Float32Array(4096 * 12);
  private _accuracyViewActive = false;
  private _lastEdgeMode: 'default' | 'search' | 'accuracy' = 'default';
  private _lastHoveredEdge: number | null = null;
  private _lastSelectedEdge: number | null = null;
  private _lastGrayedNodes: Set<string> = new Set();
  private _lastInferredCount = 0;
  private _lastTextGrayed: Set<string> | null = null;
  private _lastTextOverrides: Map<string, { name: string; type: string; color: string }> | undefined = undefined;
  private _lastAccuracyIds: Set<string> | undefined = undefined;

  // ---- Performance instrumentation ----
  private frameStats: FrameStats | null = null;

  // GPU timestamp-query state (optional, depends on adapter feature support)
  private hasTimestampQuery = false;
  private gpuQuerySet: GPUQuerySet | null = null;
  private gpuResolveBuffer: GPUBuffer | null = null;
  private gpuReadbackBuffer: GPUBuffer | null = null;
  private gpuQueryPending = false;
  /** Most recently resolved GPU pass time (ms), attached to the next frame to close. */
  private lastResolvedGpuMs: number | null = null;

  // ---- Minimap target ----
  private minimapTarget: MinimapTarget | null = null;

  // Store last camera params so we can re-apply on resize
  private lastCameraTx = 0;
  private lastCameraTy = 0;
  private lastCameraScale = 1;

  /** Ghost node bounding boxes for hit testing (screen-space pixels) */
  ghostBounds: Array<{ nodeId: string; x: number; y: number; w: number; h: number }> = [];


  private constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    cameraBuffer: GPUBuffer,
    nodesPipeline: NodesPipelineState,
    edgesPipeline: EdgesPipelineState,
    textPipeline: TextPipelineState,
    ghostPipeline: GhostPipelineState,
    atlas: TextAtlasData,
    hasTimestampQuery: boolean,
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.format = format;
    this.cameraBuffer = cameraBuffer;
    this.nodesPipeline = nodesPipeline;
    this.edgesPipeline = edgesPipeline;
    this.textPipeline = textPipeline;
    this.ghostPipeline = ghostPipeline;
    this.atlas = atlas;
    this.hasTimestampQuery = hasTimestampQuery;

    if (hasTimestampQuery) {
      this.gpuQuerySet = device.createQuerySet({ type: 'timestamp', count: 2 });
      this.gpuResolveBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.gpuReadbackBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }

    // Handle canvas resize
    this.resizeObserver = new ResizeObserver(() => {
      this.handleResize();
      // Re-apply camera with new dimensions
      this.applyCameraMatrix();
      this.markDirty();
    });
    this.resizeObserver.observe(canvas);
    this.handleResize();

    // Start frame loop
    this.frameLoop();
  }

  static async create(canvas: HTMLCanvasElement): Promise<WebGPURenderer> {
    if (!navigator.gpu) {
      throw new Error('[WebGPU] navigator.gpu not available. Check chrome://flags/#enable-unsafe-webgpu or chrome://gpu');
    }
    console.log('[WebGPU] navigator.gpu found');

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error('[WebGPU] requestAdapter() returned null. Check chrome://gpu for WebGPU status');
    }
    console.log('[WebGPU] adapter:', adapter.info);

    const requiredFeatures: GPUFeatureName[] = [];
    const hasTimestampQuery = adapter.features.has('timestamp-query');
    if (hasTimestampQuery) requiredFeatures.push('timestamp-query');

    const device = await adapter.requestDevice({ requiredFeatures });
    console.log('[WebGPU] device acquired (timestamp-query:', hasTimestampQuery, ')');

    const context = canvas.getContext('webgpu');
    if (!context) {
      throw new Error('[WebGPU] canvas.getContext("webgpu") returned null');
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    // Camera uniform buffer (4x4 matrix = 64 bytes)
    const cameraBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const atlas = createTextAtlas(device);
    const nodesPipeline = createNodesPipeline(device, format, cameraBuffer);
    const edgesPipeline = createEdgesPipeline(device, format, cameraBuffer);
    const textPipeline = createTextPipeline(device, format, cameraBuffer, atlas);
    const ghostPipeline = createGhostPipeline(device, format, atlas);

    return new WebGPURenderer(
      canvas, device, context, format, cameraBuffer,
      nodesPipeline, edgesPipeline, textPipeline, ghostPipeline, atlas,
      hasTimestampQuery,
    );
  }

  // ---- Performance instrumentation API ----

  setFrameStats(stats: FrameStats | null): void {
    this.frameStats = stats;
  }

  getFrameStats(): FrameStats | null {
    return this.frameStats;
  }

  /** True when GPU pass timing is available via the timestamp-query feature. */
  get supportsGpuTimestamps(): boolean {
    return this.hasTimestampQuery;
  }

  /**
   * Mark dirty, render once, and resolve when the GPU has finished the submit.
   * Used by scripted scenarios to render-and-wait between steps.
   */
  async forceRender(): Promise<void> {
    this.dirty = true;
    // Run one render synchronously, then await GPU completion.
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;
    if (this.frameStats) this.frameStats.beginFrame();
    this.render();
    if (this.frameStats) {
      this.frameStats.endFrame(this.lastResolvedGpuMs);
      this.lastResolvedGpuMs = null;
    }
    this.dirty = false;
    await this.device.queue.onSubmittedWorkDone();
  }

  // ---- Minimap target attach/detach ----

  attachMinimap(target: MinimapTarget): void {
    this.minimapTarget = target;
    target.bindToRenderer(this);
    this.markDirty();
  }

  detachMinimap(): void {
    this.minimapTarget = null;
  }

  getMinimapTarget(): MinimapTarget | null {
    return this.minimapTarget;
  }

  getCurrentCamera(): { tx: number; ty: number; scale: number } {
    return { tx: this.lastCameraTx, ty: this.lastCameraTy, scale: this.lastCameraScale };
  }

  getNodesPipeline(): NodesPipelineState { return this.nodesPipeline; }
  getEdgesPipeline(): EdgesPipelineState { return this.edgesPipeline; }
  getGraphData(): GraphData | null { return this.graphData; }
  getNodeSizeFn(): ((id: string) => { width: number; height: number }) | null { return this.nodeSizeFn; }

  /** Load graph data and build GPU buffers */
  setGraph(
    graphData: GraphData,
    nodeSize: (id: string) => { width: number; height: number },
  ): void {
    this.graphData = graphData;
    this.nodeSizeFn = nodeSize;

    // Build spatial grid for hit testing
    this.hitGrid.build(graphData.nodes.map(n => ({
      id: n.id,
      x: n.x,
      y: n.y,
      width: nodeSize(n.id).width,
      height: nodeSize(n.id).height,
    })));

    // Build edge hit index for proximity queries
    this.edgeHitIndex.build(graphData.edges, graphData.nodes, nodeSize);

    // Tessellate every edge once and cache the geometry. Subsequent color
    // changes (highlight, search dim, accuracy view) reuse this geometry and
    // only rewrite the color buffer — see rebuildEdges() and the highlight
    // branches in updateAppearance().
    const geometry = buildEdgeGeometry(graphData.edges, graphData.nodes, nodeSize);
    this.edgeGeometry = geometry;
    const defaultColors = buildEdgeColors(geometry.edgeRanges, graphData.edges, geometry.vertexCount);
    this.edgesPipeline = uploadEdgeData(
      this.edgesPipeline, this.device,
      geometry.positions, defaultColors, geometry.vertexCount,
    );
    // Pre-populate the 'default' cached color buffer so the first toggle back
    // from accuracy view doesn't pay the build cost. The closure reuses the
    // colors array we already built — no duplicate work.
    useCachedEdgeColors(this.edgesPipeline, this.device, 'default', () => defaultColors);
    updateEdgeViewport(this.edgesPipeline, this.device, this.canvas.width, this.canvas.height);

    if (this.minimapTarget) this.minimapTarget.invalidate();
    this.markDirty();
  }

  /** Update camera from PanZoom state */
  updateCamera(tx: number, ty: number, scale: number): void {
    this.lastCameraTx = tx;
    this.lastCameraTy = ty;
    this.lastCameraScale = scale;
    this.currentZoom = scale;
    this.applyCameraMatrix();
    if (this.minimapTarget) this.minimapTarget.notifyMainCameraChanged();
    this.markDirty();
  }

  /**
   * Rebuild only the appearance pieces that depend on the camera (text glyphs
   * for fade animation, ghost overlay positions). Used by the pan/zoom path
   * to avoid running a full updateAppearance() — node and edge data don't
   * depend on the camera, so they shouldn't be touched on a camera move.
   *
   * Caller is responsible for deciding whether the camera change actually
   * needs this work (see textFadeChanged() in renderer.ts).
   */
  rebuildCameraDependentParts(
    zoomRatio: number,
    selectedEdgeIndex: number | null,
    hoveredEdgeIndex: number | null = null,
  ): void {
    if (!this.graphData) return;
    const fs = this.frameStats;
    fs?.beginFrame();
    fs?.beginPhase('appearance.total');
    this.currentZoom = zoomRatio;

    this.updateTextAlphaUniform(zoomRatio);

    const ghostEdge = selectedEdgeIndex ?? hoveredEdgeIndex;
    if (ghostEdge !== null) {
      fs?.beginPhase('appearance.ghostRebuild');
      this.updateGhosts(ghostEdge);
      fs?.endPhase('appearance.ghostRebuild');
    }

    fs?.endPhase('appearance.total');
    this.markDirty();
  }

  private applyCameraMatrix(): void {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;
    const mat = buildCameraMatrix(w, h, this.lastCameraTx, this.lastCameraTy, this.lastCameraScale);
    this.device.queue.writeBuffer(this.cameraBuffer, 0, mat as Float32Array<ArrayBuffer>);
  }

  /** Rebuild node + text instance data based on current appearance state */
  updateAppearance(
    nodeStatusMap: Map<string, NodeStatus>,
    selectedNodeId: string | null,
    hoveredNodeId: string | null,
    searchResults: { id: string }[] | null,
    searchVisible: boolean,
    grayedNodes: Set<string>,
    zoomRatio: number,
    nodeOverrides?: Map<string, { name: string; type: string; color: string }>,
    accuracyViewActive = false,
    hoveredEdgeIndex: number | null = null,
    selectedEdgeIndex: number | null = null,
  ): void {
    if (!this.graphData) return;
    const fs = this.frameStats;
    // Open a frame window if one isn't already open. updateAppearance often
    // runs from a separate RAF/microtask than render(), so its phases would
    // otherwise land outside any frame bracket. beginFrame is idempotent.
    fs?.beginFrame();
    fs?.beginPhase('appearance.total');
    this.currentZoom = zoomRatio;

    // Rebuild edge geometry when accuracy view toggles or new inferred nodes arrive
    if (accuracyViewActive !== this._accuracyViewActive) {
      this._accuracyViewActive = accuracyViewActive;
      fs?.beginPhase('appearance.edgeRebuild');
      this.rebuildEdges(accuracyViewActive, nodeStatusMap);
      fs?.endPhase('appearance.edgeRebuild');
      if (accuracyViewActive) {
        let count = 0;
        for (const [, status] of nodeStatusMap) {
          if (status.status === 'success' && status.metrics) count++;
        }
        this._lastInferredCount = count;
      } else {
        this._lastInferredCount = 0;
      }
      // rebuildEdges() repainted with the current highlight already, so the
      // cached "last seen" highlight is up to date — record the actual values
      // so the next updateAppearance doesn't see a phantom change.
      this._lastHoveredEdge = hoveredEdgeIndex;
      this._lastSelectedEdge = selectedEdgeIndex;
    } else if (accuracyViewActive) {
      // Accuracy view already active — rebuild edges only when new metrics arrive
      let count = 0;
      for (const [, status] of nodeStatusMap) {
        if (status.status === 'success' && status.metrics) count++;
      }
      if (count !== this._lastInferredCount) {
        this._lastInferredCount = count;
        fs?.beginPhase('appearance.edgeRebuild');
        this.rebuildEdges(true, nodeStatusMap);
        fs?.endPhase('appearance.edgeRebuild');
        this._lastHoveredEdge = hoveredEdgeIndex;
        this._lastSelectedEdge = selectedEdgeIndex;
      }
    }

    // Track which edge is highlighted (hovered or selected)
    const highlightEdge = selectedEdgeIndex ?? hoveredEdgeIndex;
    const prevHighlight = this._lastSelectedEdge ?? this._lastHoveredEdge;


    const nodes = this.graphData.nodes;
    const edges = this.graphData.edges;
    const searchActive = searchVisible && searchResults && searchResults.length > 0;
    const searchSet = searchActive ? new Set(searchResults!.map(r => r.id)) : null;

    // Edge-connected node highlighting
    const edgeEndpoints = new Set<string>();
    if (selectedEdgeIndex !== null && selectedEdgeIndex < edges.length) {
      edgeEndpoints.add(edges[selectedEdgeIndex].source);
      edgeEndpoints.add(edges[selectedEdgeIndex].target);
    }

    // Build node instances. Scratch buffer is reused across frames; grown
    // geometrically when the node count exceeds capacity. Slice handed to
    // updateNodeInstances has the exact length the GPU upload needs.
    fs?.beginPhase('appearance.nodeBuild');
    const nodeFloatsNeeded = nodes.length * NODE_FLOATS;
    if (nodeFloatsNeeded > this.nodeDataScratch.length) {
      const newCap = Math.max(nodeFloatsNeeded, this.nodeDataScratch.length * 2);
      this.nodeDataScratch = new Float32Array(newCap);
    }
    const nodeData = this.nodeDataScratch;
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const off = i * NODE_FLOATS;
      let size = this.nodeSizeFn!(node.id);
      const override = nodeOverrides?.get(node.id);
      const isGrayed = grayedNodes.has(node.id);
      const isSelected = selectedNodeId === node.id;
      const isHovered = hoveredNodeId === node.id;
      const isEdgeEndpoint = edgeEndpoints.has(node.id);
      const nodeStatus = nodeStatusMap.get(node.id);

      // Widen node if override label needs more space
      let dx = 0;
      if (override) {
        const pad = 16;
        const labelWidth = measureText(this.atlas, override.type, GRAPH_FONT_SIZE);
        if (labelWidth + pad > size.width) {
          const newWidth = labelWidth + pad;
          dx = (newWidth - size.width) / 2;
          size = { width: newWidth, height: size.height };
        }
      }

      const fillColor = override ? override.color : isGrayed ? '#232636' : node.color;
      // Local fill RGB scalars instead of mutating an object — keeps the
      // hexToRgb cache from being corrupted and avoids per-node allocations.
      let fillR = 0, fillG = 0, fillB = 0;

      let strokeR = 0.2, strokeG = 0.2, strokeB = 0.2;
      let strokeWidth = 1;

      if (accuracyViewActive) {
        // Accuracy view: inferred nodes filled with accuracy color, others gray
        if (nodeStatus?.status === 'success' && nodeStatus.metrics) {
          const c = getMetricColor(nodeStatus.metrics);
          fillR = c.r; fillG = c.g; fillB = c.b;
        } else {
          fillR = EDGE_COLOR.r; fillG = EDGE_COLOR.g; fillB = EDGE_COLOR.b;
        }
      } else {
        const fc = hexToRgb(fillColor);
        fillR = fc.r; fillG = fc.g; fillB = fc.b;

        // Status colors — neutral outline for inferred, status color for others
        if (nodeStatus && !isGrayed) {
          if (nodeStatus.status === 'success') {
            // Neutral light gray — distinct from accuracy colors and selection blue
            strokeR = 0.75; strokeG = 0.75; strokeB = 0.78;
            strokeWidth = isSelected ? 4 : 3;
          } else {
            const statusColor = STATUS_COLORS[nodeStatus.status];
            if (statusColor) {
              const c = hexToRgb(statusColor);
              strokeR = c.r; strokeG = c.g; strokeB = c.b;
              strokeWidth = isSelected ? 3 : 2;
            }
          }
          if (nodeStatus.status === 'executing') strokeWidth = 3;
        }
      }

      // Selection / hover / edge-endpoint tint — mutate locals only
      if (isSelected) {
        fillR = fillR + (1.0 - fillR) * 0.4;
        fillG = fillG + (1.0 - fillG) * 0.4;
        fillB = fillB + (1.0 - fillB) * 0.4;
        strokeR = 0.298; strokeG = 0.553; strokeB = 1.0; // #4C8DFF
        strokeWidth = 3;
      } else if (isHovered) {
        fillR = fillR + (1.0 - fillR) * 0.12;
        fillG = fillG + (1.0 - fillG) * 0.12;
        fillB = fillB + (1.0 - fillB) * 0.12;
        strokeR = 0.298; strokeG = 0.553; strokeB = 1.0;
        strokeWidth = 3;
      } else if (isEdgeEndpoint) {
        fillR = fillR + (1.0 - fillR) * 0.25;
        fillG = fillG + (1.0 - fillG) * 0.25;
        fillB = fillB + (1.0 - fillB) * 0.25;
        strokeR = 0.298; strokeG = 0.553; strokeB = 1.0;
        strokeWidth = 3;
      }

      // Opacity: grayed nodes semi-transparent, search dims non-matches
      let opacity = 1;
      if (isGrayed) {
        opacity = 0.35;
      } else if (searchActive && searchSet && !searchSet.has(node.id)) {
        opacity = 0.15;
      }

      nodeData[off + 0] = node.x - dx;
      nodeData[off + 1] = node.y;
      nodeData[off + 2] = size.width;
      nodeData[off + 3] = size.height;
      nodeData[off + 4] = fillR;
      nodeData[off + 5] = fillG;
      nodeData[off + 6] = fillB;
      nodeData[off + 7] = 1.0;
      nodeData[off + 8] = strokeR;
      nodeData[off + 9] = strokeG;
      nodeData[off + 10] = strokeB;
      nodeData[off + 11] = 1.0;
      nodeData[off + 12] = strokeWidth;
      nodeData[off + 13] = NODE_RADIUS;
      nodeData[off + 14] = opacity;
      nodeData[off + 15] = 0; // padding
    }

    fs?.endPhase('appearance.nodeBuild');
    fs?.beginPhase('appearance.nodeUpload');
    this.nodesPipeline = updateNodeInstances(
      this.nodesPipeline, this.device, this.cameraBuffer,
      nodeData, nodes.length,
    );
    fs?.endPhase('appearance.nodeUpload');

    // Rebuild edges for search dimming / edge highlighting / grayed nodes (skip if accuracy view handles it)
    const edgeHighlightChanged = highlightEdge !== prevHighlight;
    const grayedChanged = grayedNodes !== this._lastGrayedNodes;
    if (!accuracyViewActive) {
      // Only rebuild when edge coloring actually changes — not every frame
      const hasGrayed = grayedNodes.size > 0;
      const hasOverrides = nodeOverrides !== undefined && nodeOverrides.size > 0;
      const isColored = searchActive || highlightEdge !== null || hasGrayed || hasOverrides;
      const wasColored = this._lastEdgeMode !== 'default';
      const needsRebuild = edgeHighlightChanged || grayedChanged || (isColored !== wasColored);

      if (needsRebuild && this.edgeGeometry) {
        fs?.beginPhase('appearance.edgeColorRebuild');
        const hl = { r: 0.298, g: 0.553, b: 1.0, a: 0.9 }; // #4C8DFF
        const dim = { r: 0.184, g: 0.200, b: 0.255, a: 0.3 };
        const grayDim = { r: 0.353, g: 0.376, b: 0.502, a: 0.35 };
        const colors = buildEdgeColors(
          this.edgeGeometry.edgeRanges,
          this.graphData.edges,
          this.edgeGeometry.vertexCount,
          (edge, idx) => {
            if (idx === highlightEdge) return { start: hl, end: hl };
            if (grayedNodes.has(edge.source) || grayedNodes.has(edge.target)) return { start: grayDim, end: grayDim };
            // Gray incoming edges to Parameter nodes (cut-as-input overrides)
            if (hasOverrides && nodeOverrides!.has(edge.target)) return { start: grayDim, end: grayDim };
            if (searchActive) return { start: dim, end: dim };
            return undefined;
          },
        );
        uploadEdgeColors(this.edgesPipeline, this.device, colors, this.edgeGeometry.vertexCount);
        fs?.endPhase('appearance.edgeColorRebuild');
      }
      this._lastEdgeMode = isColored ? 'search' : 'default';
    } else if (edgeHighlightChanged && highlightEdge !== null && this.edgeGeometry) {
      // In accuracy view, rewrite base edge colors so the highlighted edge
      // pops over the dimmed background. Geometry stays cached.
      fs?.beginPhase('appearance.edgeColorRebuild');
      const dimColor = { r: EDGE_COLOR.r, g: EDGE_COLOR.g, b: EDGE_COLOR.b, a: 0.15 };
      const hl = { r: 0.298, g: 0.553, b: 1.0, a: 0.9 };
      const colors = buildEdgeColors(
        this.edgeGeometry.edgeRanges,
        this.graphData.edges,
        this.edgeGeometry.vertexCount,
        (_edge, idx) => {
          if (idx === highlightEdge) return { start: hl, end: hl };
          return { start: dimColor, end: dimColor };
        },
      );
      uploadEdgeColors(this.edgesPipeline, this.device, colors, this.edgeGeometry.vertexCount);
      fs?.endPhase('appearance.edgeColorRebuild');
    }
    this._lastHoveredEdge = hoveredEdgeIndex;
    this._lastSelectedEdge = selectedEdgeIndex;
    this._lastGrayedNodes = grayedNodes;

    // Build text glyph instances only when text-affecting state changed.
    // Hover/selection don't affect text — only grayed, overrides, accuracy mode do.
    let accuracyInferredIds: Set<string> | undefined;
    if (accuracyViewActive) {
      accuracyInferredIds = new Set<string>();
      for (const [nodeId, status] of nodeStatusMap) {
        if (status.status === 'success' && status.metrics) accuracyInferredIds.add(nodeId);
      }
    }
    const textDirty = grayedNodes !== this._lastTextGrayed
      || nodeOverrides !== this._lastTextOverrides
      || accuracyInferredIds !== this._lastAccuracyIds;
    fs?.beginPhase('appearance.textRebuild');
    if (textDirty) {
      this.rebuildText(grayedNodes, nodeOverrides, accuracyInferredIds);
      this._lastTextGrayed = grayedNodes;
      this._lastTextOverrides = nodeOverrides;
      this._lastAccuracyIds = accuracyInferredIds;
    }
    this.updateTextAlphaUniform(zoomRatio);
    fs?.endPhase('appearance.textRebuild');

    fs?.beginPhase('appearance.ghostRebuild');
    this.updateGhosts(selectedEdgeIndex ?? hoveredEdgeIndex);
    fs?.endPhase('appearance.ghostRebuild');

    fs?.endPhase('appearance.total');

    if (this.minimapTarget) this.minimapTarget.invalidate();
    this.markDirty();
  }

  private updateTextAlphaUniform(zoomRatio: number): void {
    const alpha = zoomRatio >= 0.3 ? 1.0
      : zoomRatio <= 0.1 ? 0.0
      : (zoomRatio - 0.1) / 0.2;
    updateTextAlpha(this.textPipeline, this.device, alpha);
  }

  private rebuildText(grayedNodes: Set<string>, nodeOverrides?: Map<string, { name: string; type: string; color: string }>, accuracyInferredIds?: Set<string>): void {
    if (!this.graphData) {
      this.textPipeline = updateGlyphInstances(
        this.textPipeline, this.device, this.cameraBuffer, this.atlas,
        new Float32Array(0), 0,
      );
      return;
    }

    const nodes = this.graphData.nodes;
    const atlas = this.atlas;
    const atlasW = atlas.atlasWidth;
    const atlasH = atlas.atlasHeight;
    const FIRST_CHAR = 32;
    const CHAR_COUNT = 95;

    // In accuracy mode, we emit halo glyphs (4 shadow copies per glyph),
    // so multiply estimate by 5.
    const haloMode = !!accuracyInferredIds;
    const glyphsPerChar = haloMode ? 5 : 1;

    // Estimate glyph count
    let totalGlyphs = 0;
    for (const node of nodes) {
      const ov = nodeOverrides?.get(node.id);
      const charCount = ov ? ov.type.length : node.type.length;
      const perChar = (haloMode && accuracyInferredIds.has(node.id)) ? 5 : 1;
      totalGlyphs += charCount * perChar;
    }

    // Reuse the glyph scratch buffer; grow geometrically when label counts spike.
    const glyphFloatsNeeded = totalGlyphs * 12;
    if (glyphFloatsNeeded > this.glyphDataScratch.length) {
      const newCap = Math.max(glyphFloatsNeeded, this.glyphDataScratch.length * 2);
      this.glyphDataScratch = new Float32Array(newCap);
    }
    const glyphData = this.glyphDataScratch;
    let glyphCount = 0;

    // Halo offsets in graph-space pixels (small offset for shadow copies)
    const HALO_OFFSET = 0.8;
    const haloOffsets: [number, number][] = [
      [-HALO_OFFSET, 0], [HALO_OFFSET, 0],
      [0, -HALO_OFFSET], [0, HALO_OFFSET],
    ];

    for (const node of nodes) {

      const override = nodeOverrides?.get(node.id);
      const isGrayed = grayedNodes.has(node.id);

      const label = override ? override.type : node.type;
      let size = this.nodeSizeFn!(node.id);
      let dx = 0;
      if (override) {
        const oPad = 16;
        const labelW = measureText(atlas, label, GRAPH_FONT_SIZE);
        if (labelW + oPad > size.width) {
          const newWidth = labelW + oPad;
          dx = (newWidth - size.width) / 2;
          size = { width: newWidth, height: size.height };
        }
      }
      const fillColor = override ? override.color : isGrayed ? '#232636' : node.color;
      const glyphAlpha = isGrayed ? 0.35 : 1.0;
      // In accuracy view, use white text with dark halo for legibility
      const isAccuracyNode = accuracyInferredIds?.has(node.id) ?? false;
      const textColor = isAccuracyNode
        ? { r: 1, g: 1, b: 1 }
        : (isLightNodeColor(fillColor) ? { r: 0.1, g: 0.1, b: 0.1 } : { r: 1, g: 1, b: 1 });
      const haloColor = { r: 0.05, g: 0.05, b: 0.08 };

      // Measure text width for centering; shrink font if it overflows the node
      const pad = 6; // horizontal padding
      let fontSize = GRAPH_FONT_SIZE;
      let textWidth = measureText(atlas, label, fontSize);
      if (textWidth > size.width - pad * 2) {
        fontSize = fontSize * (size.width - pad * 2) / textWidth;
        textWidth = size.width - pad * 2;
      }
      const labelScale = fontSize / atlas.fontSize;
      const startX = (node.x - dx) + (size.width - textWidth) / 2;
      const startY = node.y + (size.height - fontSize) / 2;

      // Helper to emit one glyph at given position with given color
      const emitGlyph = (ci: number, baseX: number, baseY: number, offX: number, offY: number, cr: number, cg: number, cb: number, alpha: number): boolean => {
        const code = label.charCodeAt(ci) - FIRST_CHAR;
        if (code < 0 || code >= CHAR_COUNT) return false;
        const g = atlas.glyphs[code];
        const off = glyphCount * 12;

        glyphData[off + 0] = baseX + offX;
        glyphData[off + 1] = baseY + offY;
        glyphData[off + 2] = g.w * labelScale;
        glyphData[off + 3] = g.h * labelScale;
        glyphData[off + 4] = g.x / atlasW;
        glyphData[off + 5] = g.y / atlasH;
        glyphData[off + 6] = (g.x + g.w) / atlasW;
        glyphData[off + 7] = (g.y + g.h) / atlasH;
        glyphData[off + 8] = cr;
        glyphData[off + 9] = cg;
        glyphData[off + 10] = cb;
        glyphData[off + 11] = alpha;
        glyphCount++;
        return true;
      };

      // In halo mode: emit shadow copies first, then foreground on top
      if (haloMode && isAccuracyNode) {
        // Pass 1: halo shadows
        let curX = startX;
        for (let ci = 0; ci < label.length; ci++) {
          const code = label.charCodeAt(ci) - FIRST_CHAR;
          if (code < 0 || code >= CHAR_COUNT) {
            curX += 6 * labelScale;
            continue;
          }
          for (const [ox, oy] of haloOffsets) {
            emitGlyph(ci, curX, startY, ox, oy, haloColor.r, haloColor.g, haloColor.b, glyphAlpha * 0.9);
          }
          curX += atlas.glyphs[code].advance * labelScale;
        }

        // Pass 2: foreground
        curX = startX;
        for (let ci = 0; ci < label.length; ci++) {
          const code = label.charCodeAt(ci) - FIRST_CHAR;
          if (code < 0 || code >= CHAR_COUNT) {
            curX += 6 * labelScale;
            continue;
          }
          emitGlyph(ci, curX, startY, 0, 0, textColor.r, textColor.g, textColor.b, glyphAlpha);
          curX += atlas.glyphs[code].advance * labelScale;
        }
      } else {
        // Normal mode: single pass
        let curX = startX;
        for (let ci = 0; ci < label.length; ci++) {
          const code = label.charCodeAt(ci) - FIRST_CHAR;
          if (code < 0 || code >= CHAR_COUNT) {
            curX += 6 * labelScale;
            continue;
          }
          emitGlyph(ci, curX, startY, 0, 0, textColor.r, textColor.g, textColor.b, glyphAlpha);
          curX += atlas.glyphs[code].advance * labelScale;
        }
      }
    }

    this.textPipeline = updateGlyphInstances(
      this.textPipeline, this.device, this.cameraBuffer, this.atlas,
      glyphData, glyphCount,
    );
  }

  /** Build ghost indicator vertices for off-screen edge endpoints (screen-space). */
  private updateGhosts(selectedEdgeIndex: number | null): void {
    if (selectedEdgeIndex === null || !this.graphData) {
      this.ghostBounds = [];
      this.ghostPipeline = updateGhostVertices(this.ghostPipeline, this.device, this.atlas, new Float32Array(0), 0);
      return;
    }

    const edges = this.graphData.edges;
    if (selectedEdgeIndex >= edges.length) {
      this.ghostBounds = [];
      this.ghostPipeline = updateGhostVertices(this.ghostPipeline, this.device, this.atlas, new Float32Array(0), 0);
      return;
    }

    const edge = edges[selectedEdgeIndex];
    const nodes = this.graphData.nodes;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;

    const MARGIN = 40;
    const FONT_SIZE = 14;
    const PAD_X = 12;
    const PAD_Y = 8;
    const ARROW_SIZE = 16;
    const GAP = 6;

    // Colors matching the old ghost node style
    const bgColor = { r: 0.106, g: 0.118, b: 0.169, a: 0.9 };          // rgba(27,30,43,0.9)
    const borderColor = { r: 0.298, g: 0.553, b: 1.0, a: 1.0 };        // #4C8DFF
    const textColor = { r: 0.784, g: 0.800, b: 0.878, a: 1.0 };        // #c8cce0
    const arrowColor = { r: 0.298, g: 0.553, b: 1.0, a: 1.0 };         // #4C8DFF

    const bounds: typeof this.ghostBounds = [];
    // Estimate max vertices per ghost: 6 bg + 24 border + 3 arrow + 6*labelLen text
    const srcNode = nodes.find(n => n.id === edge.source);
    const tgtNode = nodes.find(n => n.id === edge.target);
    const maxLabelLen = Math.max(srcNode?.type.length ?? 0, tgtNode?.type.length ?? 0);
    const maxVertsPerGhost = 33 + 6 * maxLabelLen;
    const verts = new Float32Array(2 * maxVertsPerGhost * GHOST_VERTEX_FLOATS);
    let vc = 0;

    const pushVert = (x: number, y: number, u: number, v: number, r: number, g: number, b: number, a: number, isText: number) => {
      const off = vc * GHOST_VERTEX_FLOATS;
      verts[off] = x; verts[off + 1] = y;
      verts[off + 2] = u; verts[off + 3] = v;
      verts[off + 4] = r; verts[off + 5] = g; verts[off + 6] = b; verts[off + 7] = a;
      verts[off + 8] = isText;
      vc++;
    };

    const pushRect = (x: number, y: number, rw: number, rh: number, cr: number, cg: number, cb: number, ca: number) => {
      pushVert(x, y, 0, 0, cr, cg, cb, ca, 0);
      pushVert(x + rw, y, 0, 0, cr, cg, cb, ca, 0);
      pushVert(x + rw, y + rh, 0, 0, cr, cg, cb, ca, 0);
      pushVert(x, y, 0, 0, cr, cg, cb, ca, 0);
      pushVert(x + rw, y + rh, 0, 0, cr, cg, cb, ca, 0);
      pushVert(x, y + rh, 0, 0, cr, cg, cb, ca, 0);
    };

    // Border: draw outer rect then inner rect (simple border via 4 edge rects)
    const pushBorder = (x: number, y: number, bw: number, bh: number, thickness: number, cr: number, cg: number, cb: number, ca: number) => {
      pushRect(x, y, bw, thickness, cr, cg, cb, ca);                     // top
      pushRect(x, y + bh - thickness, bw, thickness, cr, cg, cb, ca);    // bottom
      pushRect(x, y + thickness, thickness, bh - 2 * thickness, cr, cg, cb, ca); // left
      pushRect(x + bw - thickness, y + thickness, thickness, bh - 2 * thickness, cr, cg, cb, ca); // right
    };

    const atlas = this.atlas;
    const atlasW = atlas.atlasWidth;
    const atlasH = atlas.atlasHeight;
    const FIRST_CHAR = 32;
    const CHAR_COUNT = 95;
    const labelScale = FONT_SIZE / atlas.fontSize;

    for (const nodeId of [edge.source, edge.target]) {
      const node = nodes.find(n => n.id === nodeId);
      if (!node) continue;

      const size = this.nodeSizeFn!(nodeId);
      const cx = node.x + size.width / 2;
      const cy = node.y + size.height / 2;
      const sx = cx * this.lastCameraScale + this.lastCameraTx;
      const sy = cy * this.lastCameraScale + this.lastCameraTy;

      // Check if on-screen
      if (sx >= MARGIN && sx <= w - MARGIN && sy >= MARGIN && sy <= h - MARGIN) continue;

      // Ray-cast from viewport center to node position, clipped to inset viewport rect
      const vcx = w / 2, vcy = h / 2;
      const dx = sx - vcx, dy = sy - vcy;
      if (Math.abs(dx) < 0.1 && Math.abs(dy) < 0.1) continue;

      const left = MARGIN, right = w - MARGIN, top = MARGIN, bottom = h - MARGIN;
      let tMin = Infinity;
      if (dx !== 0) {
        for (const t of [(left - vcx) / dx, (right - vcx) / dx]) {
          if (t > 0) { const iy = vcy + dy * t; if (iy >= top && iy <= bottom && t < tMin) tMin = t; }
        }
      }
      if (dy !== 0) {
        for (const t of [(top - vcy) / dy, (bottom - vcy) / dy]) {
          if (t > 0) { const ix = vcx + dx * t; if (ix >= left && ix <= right && t < tMin) tMin = t; }
        }
      }
      if (!isFinite(tMin)) continue;

      const ghostX = vcx + dx * tMin;
      const ghostY = vcy + dy * tMin;
      const angle = Math.atan2(dy, dx);

      // Measure label text width
      const label = node.type;
      let textWidth = 0;
      for (let i = 0; i < label.length; i++) {
        const code = label.charCodeAt(i) - FIRST_CHAR;
        if (code >= 0 && code < CHAR_COUNT) textWidth += atlas.glyphs[code].advance * labelScale;
      }

      // Total badge dimensions
      const badgeW = ARROW_SIZE + GAP + textWidth + PAD_X * 2;
      const badgeH = FONT_SIZE + PAD_Y * 2;
      const bx = ghostX - badgeW / 2;
      const by = ghostY - badgeH / 2;

      // Store bounding box for hit testing
      bounds.push({ nodeId, x: bx, y: by, w: badgeW, h: badgeH });

      // Background rect
      pushRect(bx, by, badgeW, badgeH, bgColor.r, bgColor.g, bgColor.b, bgColor.a);

      // Border (1.5px)
      pushBorder(bx, by, badgeW, badgeH, 1.5, borderColor.r, borderColor.g, borderColor.b, borderColor.a);

      // Arrow triangle (rotated by angle)
      const arrowCX = bx + PAD_X + ARROW_SIZE / 2;
      const arrowCY = by + badgeH / 2;
      const cos = Math.cos(angle), sin = Math.sin(angle);
      const arrowR = ARROW_SIZE / 2;
      // Triangle vertices: tip at (r,0), base at (-r/2, ±r*0.5)
      const tipX = arrowR, tipY = 0;
      const baseX = -arrowR * 0.5, baseY1 = -arrowR * 0.5, baseY2 = arrowR * 0.5;
      pushVert(
        arrowCX + tipX * cos - tipY * sin, arrowCY + tipX * sin + tipY * cos,
        0, 0, arrowColor.r, arrowColor.g, arrowColor.b, arrowColor.a, 0,
      );
      pushVert(
        arrowCX + baseX * cos - baseY1 * sin, arrowCY + baseX * sin + baseY1 * cos,
        0, 0, arrowColor.r, arrowColor.g, arrowColor.b, arrowColor.a, 0,
      );
      pushVert(
        arrowCX + baseX * cos - baseY2 * sin, arrowCY + baseX * sin + baseY2 * cos,
        0, 0, arrowColor.r, arrowColor.g, arrowColor.b, arrowColor.a, 0,
      );

      // Text glyphs
      let curX = bx + PAD_X + ARROW_SIZE + GAP;
      const textY = by + PAD_Y;
      for (let ci = 0; ci < label.length; ci++) {
        const code = label.charCodeAt(ci) - FIRST_CHAR;
        if (code < 0 || code >= CHAR_COUNT) { curX += 6 * labelScale; continue; }
        const g = atlas.glyphs[code];
        const gw = g.w * labelScale;
        const gh = g.h * labelScale;
        const u0 = g.x / atlasW, v0 = g.y / atlasH;
        const u1 = (g.x + g.w) / atlasW, v1 = (g.y + g.h) / atlasH;

        pushVert(curX, textY, u0, v0, textColor.r, textColor.g, textColor.b, textColor.a, 1);
        pushVert(curX + gw, textY, u1, v0, textColor.r, textColor.g, textColor.b, textColor.a, 1);
        pushVert(curX + gw, textY + gh, u1, v1, textColor.r, textColor.g, textColor.b, textColor.a, 1);
        pushVert(curX, textY, u0, v0, textColor.r, textColor.g, textColor.b, textColor.a, 1);
        pushVert(curX + gw, textY + gh, u1, v1, textColor.r, textColor.g, textColor.b, textColor.a, 1);
        pushVert(curX, textY + gh, u0, v1, textColor.r, textColor.g, textColor.b, textColor.a, 1);

        curX += g.advance * labelScale;
      }
    }

    this.ghostBounds = bounds;
    this.ghostPipeline = updateGhostVertices(this.ghostPipeline, this.device, this.atlas, verts, vc);
  }

  /** Check if a viewport point hits a ghost indicator. Returns the nodeId or null. */
  ghostHitTest(viewportX: number, viewportY: number): string | null {
    for (const b of this.ghostBounds) {
      if (viewportX >= b.x && viewportX <= b.x + b.w && viewportY >= b.y && viewportY <= b.y + b.h) {
        return b.nodeId;
      }
    }
    return null;
  }

  private rebuildEdges(accuracyView: boolean, nodeStatusMap: Map<string, NodeStatus>): void {
    if (!this.graphData || !this.nodeSizeFn) return;

    if (accuracyView) {
      // Collect inferred nodes with metrics
      const inferredMetrics = new Map<string, AccuracyMetrics>();
      for (const [nodeId, status] of nodeStatusMap) {
        if (status.status === 'success' && status.metrics) {
          inferredMetrics.set(nodeId, status.metrics);
        }
      }

      // Build adjacency list for BFS (source -> edges)
      const adjList = new Map<string, typeof this.graphData.edges>();
      for (const edge of this.graphData.edges) {
        let list = adjList.get(edge.source);
        if (!list) { list = []; adjList.set(edge.source, list); }
        list.push(edge);
      }

      // BFS from each inferred node downward to find paths to other inferred nodes
      const paths: AccuracyPath[] = [];
      const inferredIds = [...inferredMetrics.keys()];

      for (const startId of inferredIds) {
        const c = getMetricColor(inferredMetrics.get(startId)!);
        const pathColor = { r: c.r, g: c.g, b: c.b, a: 1.0 };

        const queue: { nodeId: string; edgePath: GraphData['edges'] }[] =
          [{ nodeId: startId, edgePath: [] }];
        const visited = new Set<string>([startId]);

        while (queue.length > 0) {
          const { nodeId, edgePath } = queue.shift()!;
          const outEdges = adjList.get(nodeId);
          if (!outEdges) continue;

          for (const edge of outEdges) {
            if (visited.has(edge.target)) continue;
            visited.add(edge.target);

            const newPath = [...edgePath, edge];

            if (inferredMetrics.has(edge.target)) {
              // Found another inferred node -- record as a path
              paths.push({ edges: newPath, color: pathColor });
            } else {
              queue.push({ nodeId: edge.target, edgePath: newPath });
            }
          }
        }
      }

      // Dimmed base edges — uniform color, served from a cached buffer so
      // the second-and-onward toggle is just a buffer reference swap.
      if (this.edgeGeometry) {
        const geom = this.edgeGeometry;
        const edges = this.graphData.edges;
        useCachedEdgeColors(this.edgesPipeline, this.device, 'dim', () => {
          const dimColor = { r: EDGE_COLOR.r, g: EDGE_COLOR.g, b: EDGE_COLOR.b, a: 0.15 };
          return buildEdgeColors(
            geom.edgeRanges,
            edges,
            geom.vertexCount,
            () => ({ start: dimColor, end: dimColor }),
          );
        });
      }

      // Colored overlay paths (drawn on top of nodes). The set of paths
      // changes whenever new metrics arrive, so the overlay still rebuilds
      // its tessellation — but only for the BFS-discovered paths, not the
      // full edge list.
      const pathGeom = buildPathGeometry(paths, this.graphData.nodes, this.nodeSizeFn!);
      if (!this.overlayEdgesPipeline) {
        this.overlayEdgesPipeline = createEdgesPipeline(this.device, this.format, this.cameraBuffer);
        updateEdgeViewport(this.overlayEdgesPipeline, this.device, this.canvas.width, this.canvas.height);
      }
      this.overlayEdgesPipeline = uploadEdgeData(
        this.overlayEdgesPipeline, this.device,
        pathGeom.positions, pathGeom.colors, pathGeom.vertexCount,
      );
      this._lastEdgeMode = 'accuracy';
    } else {
      // Restore default colors — uniform color, served from the cached
      // "default" buffer that was built once at setGraph() time.
      if (this.edgeGeometry) {
        const geom = this.edgeGeometry;
        const edges = this.graphData.edges;
        useCachedEdgeColors(this.edgesPipeline, this.device, 'default', () => {
          return buildEdgeColors(geom.edgeRanges, edges, geom.vertexCount);
        });
      }
      // Clear overlay paths
      if (this.overlayEdgesPipeline) {
        this.overlayEdgesPipeline = uploadEdgeData(
          this.overlayEdgesPipeline, this.device,
          new Float32Array(0), new Float32Array(0), 0,
        );
      }
      this._lastEdgeMode = 'default';
    }
  }

  markDirty(): void {
    this.dirty = true;
  }

  private frameLoop(): void {
    this.animFrameId = requestAnimationFrame(() => this.frameLoop());
    if (!this.dirty) return;
    this.dirty = false;

    // Guard against rendering to a zero-sized canvas
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;

    if (this.frameStats) this.frameStats.beginFrame();
    this.render();
    if (this.frameStats) {
      this.frameStats.endFrame(this.lastResolvedGpuMs);
      this.lastResolvedGpuMs = null;
    }
  }

  private render(): void {
    const fs = this.frameStats;
    fs?.beginPhase('render.total');

    let textureView: GPUTextureView;
    try {
      textureView = this.context.getCurrentTexture().createView();
    } catch {
      // Context lost or canvas not ready
      fs?.endPhase('render.total');
      return;
    }

    fs?.beginPhase('render.encode');
    const encoder = this.device.createCommandEncoder();

    // Attach GPU timestamps to the main pass when available and not already pending
    const useTimestamps = this.hasTimestampQuery && !this.gpuQueryPending && this.gpuQuerySet !== null;
    const passDesc: GPURenderPassDescriptor = {
      colorAttachments: [{
        view: this.msaaView!,
        resolveTarget: textureView,
        clearValue: CLEAR_COLOR,
        loadOp: 'clear',
        storeOp: 'discard',
      }],
    };
    if (useTimestamps) {
      passDesc.timestampWrites = {
        querySet: this.gpuQuerySet!,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      };
    }
    const pass = encoder.beginRenderPass(passDesc);

    // Draw order: edges → nodes → overlay paths (on top of nodes) → text → ghosts
    drawEdges(pass, this.edgesPipeline);
    drawNodes(pass, this.nodesPipeline);
    if (this.overlayEdgesPipeline) {
      drawEdges(pass, this.overlayEdgesPipeline);
    }
    if (this.currentZoom >= 0.05) {
      drawText(pass, this.textPipeline);
    }
    drawGhosts(pass, this.ghostPipeline);

    pass.end();

    // Resolve GPU timestamps + start async readback
    if (useTimestamps) {
      encoder.resolveQuerySet(this.gpuQuerySet!, 0, 2, this.gpuResolveBuffer!, 0);
      encoder.copyBufferToBuffer(this.gpuResolveBuffer!, 0, this.gpuReadbackBuffer!, 0, 16);
    }
    fs?.endPhase('render.encode');

    // Minimap pass (after main, in same encoder for one submit)
    if (this.minimapTarget) {
      fs?.beginPhase('render.minimap');
      this.minimapTarget.draw(encoder);
      fs?.endPhase('render.minimap');
    }

    fs?.beginPhase('render.submit');
    this.device.queue.submit([encoder.finish()]);
    fs?.endPhase('render.submit');

    if (useTimestamps) {
      this.gpuQueryPending = true;
      const readback = this.gpuReadbackBuffer!;
      readback.mapAsync(GPUMapMode.READ).then(() => {
        try {
          const arr = new BigInt64Array(readback.getMappedRange().slice(0));
          const ns = Number(arr[1] - arr[0]);
          if (ns >= 0) this.lastResolvedGpuMs = ns / 1_000_000;
          readback.unmap();
        } catch {
          // Ignore — the device may have been destroyed mid-frame
        } finally {
          this.gpuQueryPending = false;
        }
      }).catch(() => {
        this.gpuQueryPending = false;
      });
    }

    fs?.endPhase('render.total');
  }

  private handleResize(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;

    const physW = Math.floor(w * dpr);
    const physH = Math.floor(h * dpr);

    // Only reconfigure if size actually changed
    if (this.canvas.width === physW && this.canvas.height === physH) return;

    this.canvas.width = physW;
    this.canvas.height = physH;

    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'opaque',
    });

    // Recreate MSAA texture at new size
    if (this.msaaTexture) this.msaaTexture.destroy();
    this.msaaTexture = this.device.createTexture({
      size: { width: physW, height: physH },
      format: this.format,
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.msaaView = this.msaaTexture.createView();

    updateEdgeViewport(this.edgesPipeline, this.device, physW, physH);
    if (this.overlayEdgesPipeline) updateEdgeViewport(this.overlayEdgesPipeline, this.device, physW, physH);
    updateGhostViewport(this.ghostPipeline, this.device, w, h);
  }

  destroy(): void {
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    this.resizeObserver.disconnect();
    if (this.minimapTarget) {
      this.minimapTarget.destroy();
      this.minimapTarget = null;
    }
    this.nodesPipeline.storageBuffer.destroy();
    this.edgesPipeline.geometryBuffer.destroy();
    this.edgesPipeline.liveColorBuffer.destroy();
    for (const buf of this.edgesPipeline.cachedColorBuffers.values()) buf.destroy();
    this.edgesPipeline.cachedColorBuffers.clear();
    this.edgesPipeline.zoomBuffer.destroy();
    if (this.overlayEdgesPipeline) {
      this.overlayEdgesPipeline.geometryBuffer.destroy();
      this.overlayEdgesPipeline.liveColorBuffer.destroy();
      for (const buf of this.overlayEdgesPipeline.cachedColorBuffers.values()) buf.destroy();
      this.overlayEdgesPipeline.cachedColorBuffers.clear();
      this.overlayEdgesPipeline.zoomBuffer.destroy();
    }
    this.textPipeline.storageBuffer.destroy();
    this.textPipeline.textAlphaBuffer.destroy();
    this.ghostPipeline.vertexBuffer.destroy();
    this.ghostPipeline.viewportBuffer.destroy();
    this.atlas.texture.destroy();
    this.cameraBuffer.destroy();
    if (this.msaaTexture) this.msaaTexture.destroy();
    if (this.gpuQuerySet) this.gpuQuerySet.destroy();
    if (this.gpuResolveBuffer) this.gpuResolveBuffer.destroy();
    if (this.gpuReadbackBuffer) this.gpuReadbackBuffer.destroy();
    this.device.destroy();
  }
}

// ---- Helpers ----

/**
 * Memoized hex → rgb conversion. Real graphs only use ~20 unique color
 * strings (one per op category) but updateAppearance calls this once per
 * node per frame. Caching turns 10k parseInt-pairs into 10k Map.get hits.
 *
 * The cache lives at module scope and is shared across renderer instances —
 * harmless because the input space is small and pure (string → rgb).
 */
const HEX_RGB_CACHE = new Map<string, { r: number; g: number; b: number }>();
const HEX_RGB_FALLBACK = { r: 0.2, g: 0.2, b: 0.2 };

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const cached = HEX_RGB_CACHE.get(hex);
  if (cached) return cached;
  const h = hex.charCodeAt(0) === 35 /* '#' */ ? hex.slice(1) : hex;
  if (h.length !== 6) return HEX_RGB_FALLBACK;
  const result = {
    r: parseInt(h.slice(0, 2), 16) / 255,
    g: parseInt(h.slice(2, 4), 16) / 255,
    b: parseInt(h.slice(4, 6), 16) / 255,
  };
  HEX_RGB_CACHE.set(hex, result);
  return result;
}

/**
 * Get accuracy gradient color for a node using the current metric/range
 * from configStore (unified accuracy system).
 */
function getMetricColor(metrics: AccuracyMetrics): { r: number; g: number; b: number } {
  const metric = configStore.accuracyMetric;
  const range = configStore.activeRange;
  const value = metrics[metric];
  return getAccuracyColorRgb(metric, value, range);
}
