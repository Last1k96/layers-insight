/**
 * Main WebGPU renderer orchestrating nodes, edges, and text pipelines.
 * Manages device init, frame loop, and buffer updates.
 */
import type { GraphData, GraphNode } from '../../stores/types';
import type { NodeStatus } from '../../stores/graph.svelte';
import { configStore } from '../../stores/config.svelte';
import { isLightNodeColor, STATUS_COLORS } from '../opColors';
import { buildCameraMatrix, CLEAR_COLOR, NODE_FLOATS, NODE_RADIUS, GRAPH_FONT_SIZE } from './types';
import { createNodesPipeline, updateNodeInstances, drawNodes } from './nodesPipeline';
import type { NodesPipelineState } from './nodesPipeline';
import { createEdgesPipeline, updateEdgeVertices, setEdgeColor, updateEdgeViewport, drawEdges, buildEdgeGeometry } from './edgesPipeline';
import type { EdgesPipelineState } from './edgesPipeline';
import { createTextPipeline, updateGlyphInstances, drawText } from './textPipeline';
import type { TextPipelineState } from './textPipeline';
import { createTextAtlas, measureText } from './textAtlas';
import type { TextAtlasData } from './textAtlas';
import { SpatialGrid } from './hitTest';

export class WebGPURenderer {
  readonly canvas: HTMLCanvasElement;
  readonly hitGrid = new SpatialGrid();

  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;
  private cameraBuffer: GPUBuffer;
  private nodesPipeline: NodesPipelineState;
  private edgesPipeline: EdgesPipelineState;
  private textPipeline: TextPipelineState;
  private atlas: TextAtlasData;
  private dirty = true;
  private animFrameId: number | null = null;
  private resizeObserver: ResizeObserver;
  private msaaTexture: GPUTexture | null = null;
  private msaaView: GPUTextureView | null = null;
  private currentZoom = 1;
  private graphData: GraphData | null = null;
  private nodeSizeFn: ((id: string) => { width: number; height: number }) | null = null;

  // Store last camera params so we can re-apply on resize
  private lastCameraTx = 0;
  private lastCameraTy = 0;
  private lastCameraScale = 1;

  private constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    cameraBuffer: GPUBuffer,
    nodesPipeline: NodesPipelineState,
    edgesPipeline: EdgesPipelineState,
    textPipeline: TextPipelineState,
    atlas: TextAtlasData,
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.format = format;
    this.cameraBuffer = cameraBuffer;
    this.nodesPipeline = nodesPipeline;
    this.edgesPipeline = edgesPipeline;
    this.textPipeline = textPipeline;
    this.atlas = atlas;

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

  static async create(canvas: HTMLCanvasElement): Promise<WebGPURenderer | null> {
    if (!navigator.gpu) {
      console.error('[WebGPU] navigator.gpu not available. Check chrome://flags/#enable-unsafe-webgpu or chrome://gpu');
      return null;
    }
    console.log('[WebGPU] navigator.gpu found');

    let adapter: GPUAdapter | null;
    try {
      adapter = await navigator.gpu.requestAdapter();
    } catch (e) {
      console.error('[WebGPU] requestAdapter() threw:', e);
      return null;
    }
    if (!adapter) {
      console.error('[WebGPU] requestAdapter() returned null. Check chrome://gpu for WebGPU status');
      return null;
    }
    console.log('[WebGPU] adapter:', adapter.info);

    let device: GPUDevice;
    try {
      device = await adapter.requestDevice();
    } catch (e) {
      console.error('[WebGPU] requestDevice() threw:', e);
      return null;
    }
    console.log('[WebGPU] device acquired');

    const context = canvas.getContext('webgpu');
    if (!context) {
      console.error('[WebGPU] canvas.getContext("webgpu") returned null');
      return null;
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

    return new WebGPURenderer(
      canvas, device, context, format, cameraBuffer,
      nodesPipeline, edgesPipeline, textPipeline, atlas,
    );
  }

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

    // Build edge geometry (4 floats per vertex: centerX, centerY, normalX, normalY)
    const edgeVerts = buildEdgeGeometry(graphData.edges, graphData.nodes, nodeSize);
    this.edgesPipeline = updateEdgeVertices(
      this.edgesPipeline, this.device, this.cameraBuffer,
      edgeVerts, edgeVerts.length / 4,
    );
    updateEdgeViewport(this.edgesPipeline, this.device, this.canvas.width, this.canvas.height);

    this.markDirty();
  }

  /** Update camera from PanZoom state */
  updateCamera(tx: number, ty: number, scale: number): void {
    this.lastCameraTx = tx;
    this.lastCameraTy = ty;
    this.lastCameraScale = scale;
    this.currentZoom = scale;
    this.applyCameraMatrix();
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
  ): void {
    if (!this.graphData) return;
    this.currentZoom = zoomRatio;

    const nodes = this.graphData.nodes;
    const searchActive = searchVisible && searchResults && searchResults.length > 0;
    const searchSet = searchActive ? new Set(searchResults!.map(r => r.id)) : null;

    // Build node instances
    const nodeData = new Float32Array(nodes.length * NODE_FLOATS);
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const off = i * NODE_FLOATS;
      let size = this.nodeSizeFn!(node.id);
      const override = nodeOverrides?.get(node.id);
      const isGrayed = grayedNodes.has(node.id);
      const isSelected = selectedNodeId === node.id;
      const isHovered = hoveredNodeId === node.id;
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

      const fillColor = override ? override.color : isGrayed ? '#303030' : node.color;
      const fill = hexToRgb(fillColor);

      let strokeR = 0.2, strokeG = 0.2, strokeB = 0.2;
      let strokeWidth = 1;

      // Status colors
      if (nodeStatus && !isGrayed) {
        if (nodeStatus.status === 'success' && nodeStatus.metrics) {
          const c = getAccuracyGradientRgb(nodeStatus.metrics.mse);
          strokeR = c.r; strokeG = c.g; strokeB = c.b;
          strokeWidth = isSelected ? 3 : 2;
        } else {
          const statusColor = STATUS_COLORS[nodeStatus.status];
          if (statusColor) {
            const c = hexToRgb(statusColor);
            strokeR = c.r; strokeG = c.g; strokeB = c.b;
            strokeWidth = isSelected ? 3 : 2;
          }
          if (nodeStatus.status === 'executing') strokeWidth = 3;
        }
      }

      // Selection highlight
      if (isSelected) {
        strokeR = 0.863; strokeG = 0; strokeB = 0;
        strokeWidth = 2;
      }

      // Hover highlight
      if (isHovered && !isSelected) {
        strokeR = 0.863; strokeG = 0; strokeB = 0;
        strokeWidth = 2;
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
      nodeData[off + 4] = fill.r;
      nodeData[off + 5] = fill.g;
      nodeData[off + 6] = fill.b;
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

    this.nodesPipeline = updateNodeInstances(
      this.nodesPipeline, this.device, this.cameraBuffer,
      nodeData, nodes.length,
    );

    // Update edge color for search dimming
    if (searchActive) {
      setEdgeColor(this.edgesPipeline, this.device, 0.267, 0.267, 0.267, 0.3);
    } else {
      setEdgeColor(this.edgesPipeline, this.device, 0.667, 0.667, 0.667, 1.0);
    }

    // Build text glyph instances (skip if zoomed out)
    this.rebuildText(grayedNodes, zoomRatio, nodeOverrides);

    this.markDirty();
  }

  private rebuildText(grayedNodes: Set<string>, zoomRatio: number, nodeOverrides?: Map<string, { name: string; type: string; color: string }>): void {
    if (!this.graphData || zoomRatio < 0.05) {
      this.textPipeline = updateGlyphInstances(
        this.textPipeline, this.device, this.cameraBuffer, this.atlas,
        new Float32Array(0), 0,
      );
      return;
    }

    // Alpha fade: fully opaque above 0.3, linear fade to 0 between 0.3 and 0.1
    const textAlpha = zoomRatio >= 0.3 ? 1.0
      : zoomRatio <= 0.1 ? 0.0
      : (zoomRatio - 0.1) / 0.2;

    const nodes = this.graphData.nodes;
    const atlas = this.atlas;
    const scale = GRAPH_FONT_SIZE / atlas.fontSize;
    const atlasW = atlas.atlasWidth;
    const atlasH = atlas.atlasHeight;
    const FIRST_CHAR = 32;
    const CHAR_COUNT = 95;

    // Estimate glyph count
    let totalGlyphs = 0;
    for (const node of nodes) {
      const ov = nodeOverrides?.get(node.id);
      totalGlyphs += ov ? ov.type.length : node.type.length;
    }

    const glyphData = new Float32Array(totalGlyphs * 12);
    let glyphCount = 0;

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
      const fillColor = override ? override.color : isGrayed ? '#303030' : node.color;
      const glyphAlpha = isGrayed ? textAlpha * 0.35 : textAlpha;
      const textColor = isLightNodeColor(fillColor) ? { r: 0.2, g: 0.2, b: 0.2 } : { r: 1, g: 1, b: 1 };

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

      let curX = startX;
      for (let ci = 0; ci < label.length; ci++) {
        const code = label.charCodeAt(ci) - FIRST_CHAR;
        if (code < 0 || code >= CHAR_COUNT) {
          curX += 6 * labelScale; // space for unknown chars
          continue;
        }

        const g = atlas.glyphs[code];
        const off = glyphCount * 12;

        // World position & size
        glyphData[off + 0] = curX;
        glyphData[off + 1] = startY;
        glyphData[off + 2] = g.w * labelScale;
        glyphData[off + 3] = g.h * labelScale;

        // UV rect (normalized)
        glyphData[off + 4] = g.x / atlasW;
        glyphData[off + 5] = g.y / atlasH;
        glyphData[off + 6] = (g.x + g.w) / atlasW;
        glyphData[off + 7] = (g.y + g.h) / atlasH;

        // Color
        glyphData[off + 8] = textColor.r;
        glyphData[off + 9] = textColor.g;
        glyphData[off + 10] = textColor.b;
        glyphData[off + 11] = glyphAlpha;

        curX += g.advance * labelScale;
        glyphCount++;
      }
    }

    this.textPipeline = updateGlyphInstances(
      this.textPipeline, this.device, this.cameraBuffer, this.atlas,
      glyphData, glyphCount,
    );
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

    this.render();
  }

  private render(): void {
    let textureView: GPUTextureView;
    try {
      textureView = this.context.getCurrentTexture().createView();
    } catch {
      // Context lost or canvas not ready
      return;
    }

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.msaaView!,
        resolveTarget: textureView,
        clearValue: CLEAR_COLOR,
        loadOp: 'clear',
        storeOp: 'discard',
      }],
    });

    // Draw order: edges → nodes → text
    drawEdges(pass, this.edgesPipeline);
    drawNodes(pass, this.nodesPipeline);
    if (this.currentZoom >= 0.05) {
      drawText(pass, this.textPipeline);
    }

    pass.end();
    this.device.queue.submit([encoder.finish()]);
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
  }

  destroy(): void {
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    this.resizeObserver.disconnect();
    this.nodesPipeline.storageBuffer.destroy();
    this.edgesPipeline.vertexBuffer.destroy();
    this.edgesPipeline.colorBuffer.destroy();
    this.edgesPipeline.zoomBuffer.destroy();
    this.textPipeline.storageBuffer.destroy();
    this.atlas.texture.destroy();
    this.cameraBuffer.destroy();
    if (this.msaaTexture) this.msaaTexture.destroy();
    this.device.destroy();
  }
}

// ---- Helpers ----

function hexToRgb(hex: string): { r: number; g: number; b: number } {
  const h = hex.replace('#', '');
  if (h.length !== 6) return { r: 0.2, g: 0.2, b: 0.2 };
  return {
    r: parseInt(h.slice(0, 2), 16) / 255,
    g: parseInt(h.slice(2, 4), 16) / 255,
    b: parseInt(h.slice(4, 6), 16) / 255,
  };
}

function getAccuracyGradientRgb(mse: number): { r: number; g: number; b: number } {
  if (configStore.gradientMode === 'threshold') {
    return mse <= configStore.globalThreshold
      ? { r: 0.063, g: 0.725, b: 0.506 }  // #10B981
      : { r: 0.937, g: 0.267, b: 0.267 };  // #EF4444
  }
  const logMse = mse > 0 ? Math.log10(mse) : -10;
  const t = Math.max(0, Math.min(1, (logMse + 8) / 6));
  const r = Math.min(1, t * 2);
  const g = Math.min(1, (1 - Math.max(0, t - 0.5) * 2));
  return { r, g, b: 0 };
}
