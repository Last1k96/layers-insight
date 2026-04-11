/**
 * WebGPU minimap target.
 *
 * Shares the main renderer's GPUDevice, node storage buffer, and edge vertex
 * buffer. Owns its own canvas/context, MSAA texture, camera UBO, viewport UBO
 * (for edge thickness), and bind groups. Drawn back-to-back with the main pass
 * inside the main renderer's command encoder so a single submit covers both.
 *
 * Also draws a viewport rectangle overlay representing the main camera's
 * visible region in graph coordinates, using a tiny dedicated pipeline.
 */
import type { WebGPURenderer } from './WebGPURenderer';
import { ALPHA_BLEND, CLEAR_COLOR, buildCameraMatrix } from './types';

const VIEWPORT_SHADER = /* wgsl */ `
@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex
fn vertexMain(
  @location(0) pos: vec2<f32>,
  @location(1) color: vec4<f32>,
) -> VertexOutput {
  var out: VertexOutput;
  out.position = camera * vec4(pos, 0.0, 1.0);
  out.color = color;
  return out;
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4<f32> {
  return in.color;
}
`;

interface ViewportPipelineState {
  pipeline: GPURenderPipeline;
  vertexBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  vertexCount: number;
}

const VIEWPORT_VERTEX_FLOATS = 6; // x, y, r, g, b, a
const VIEWPORT_VERTEX_BYTES = VIEWPORT_VERTEX_FLOATS * 4;
const VIEWPORT_MAX_VERTS = 24; // outline = 4 quads × 6 verts

export class MinimapTarget {
  readonly canvas: HTMLCanvasElement;
  private renderer: WebGPURenderer | null = null;

  // Per-target GPU resources
  private context: GPUCanvasContext;
  private cameraBuffer: GPUBuffer;
  private edgeViewportBuffer: GPUBuffer;
  private msaaTexture: GPUTexture | null = null;
  private msaaView: GPUTextureView | null = null;

  // Shared-pipeline bind groups (recreated when shared buffers grow)
  private nodesBindGroup: GPUBindGroup | null = null;
  private edgesBindGroup: GPUBindGroup | null = null;
  private lastNodesStorageBuffer: GPUBuffer | null = null;
  private lastEdgesVertexBuffer: GPUBuffer | null = null;

  // Viewport rect overlay
  private viewportPipeline: ViewportPipelineState | null = null;

  // Geometry tracking
  private graphBounds: { minX: number; minY: number; maxX: number; maxY: number } | null = null;
  private cachedScale = 1;
  private cachedOffsetX = 0;
  private cachedOffsetY = 0;
  /** True when the bench wants to skip drawing this target (used by minimapCost scenario). */
  private suspended = false;

  private constructor(canvas: HTMLCanvasElement, context: GPUCanvasContext, cameraBuffer: GPUBuffer, edgeViewportBuffer: GPUBuffer) {
    this.canvas = canvas;
    this.context = context;
    this.cameraBuffer = cameraBuffer;
    this.edgeViewportBuffer = edgeViewportBuffer;
  }

  /** Create a MinimapTarget bound to a canvas. The renderer must already be initialized. */
  static create(canvas: HTMLCanvasElement, renderer: WebGPURenderer): MinimapTarget {
    const device = renderer.device;
    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('[MinimapTarget] canvas.getContext("webgpu") returned null');
    context.configure({ device, format: renderer.format, alphaMode: 'opaque' });

    const cameraBuffer = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const edgeViewportBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(edgeViewportBuffer, 0, new Float32Array([200, 150]) as Float32Array<ArrayBuffer>);

    const target = new MinimapTarget(canvas, context, cameraBuffer, edgeViewportBuffer);
    target.bindToRenderer(renderer);
    target.handleResize();
    return target;
  }

  bindToRenderer(renderer: WebGPURenderer): void {
    this.renderer = renderer;

    // Create the viewport rect overlay pipeline (uses minimap camera UBO).
    const device = renderer.device;
    const module = device.createShaderModule({ code: VIEWPORT_SHADER });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } }],
    });
    const pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module,
        entryPoint: 'vertexMain',
        buffers: [{
          arrayStride: VIEWPORT_VERTEX_BYTES,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x2' as GPUVertexFormat },
            { shaderLocation: 1, offset: 8, format: 'float32x4' as GPUVertexFormat },
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fragmentMain',
        targets: [{ format: renderer.format, blend: ALPHA_BLEND }],
      },
      primitive: { topology: 'triangle-list' },
      multisample: { count: 4 },
    });
    const vertexBuffer = device.createBuffer({
      size: VIEWPORT_MAX_VERTS * VIEWPORT_VERTEX_BYTES,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    const viewportBindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.cameraBuffer } }],
    });
    this.viewportPipeline = { pipeline, vertexBuffer, bindGroup: viewportBindGroup, vertexCount: 0 };

    // Build initial shared-pipeline bind groups
    this.refreshSharedBindGroups();
    this.recomputeBoundsAndCamera();
    this.notifyMainCameraChanged();
  }

  setSuspended(s: boolean): void {
    this.suspended = s;
  }

  isSuspended(): boolean { return this.suspended; }

  /** The graph data or its instance buffers may have changed; rebuild bind groups + camera. */
  invalidate(): void {
    this.refreshSharedBindGroups();
    this.recomputeBoundsAndCamera();
    this.notifyMainCameraChanged();
  }

  /** Called by the renderer whenever the main camera moves (so we can update the viewport rect). */
  notifyMainCameraChanged(): void {
    if (!this.renderer || !this.viewportPipeline) return;
    this.rebuildViewportRect();
  }

  /** Compute fit-all camera matrix from current graph bounds + minimap canvas size. */
  private recomputeBoundsAndCamera(): void {
    if (!this.renderer) return;
    const data = this.renderer.getGraphData();
    const sizeFn = this.renderer.getNodeSizeFn();
    if (!data || !sizeFn || data.nodes.length === 0) {
      this.graphBounds = null;
      return;
    }

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const n of data.nodes) {
      const s = sizeFn(n.id);
      if (n.x < minX) minX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.x + s.width > maxX) maxX = n.x + s.width;
      if (n.y + s.height > maxY) maxY = n.y + s.height;
    }
    this.graphBounds = { minX, minY, maxX, maxY };

    const w = this.canvas.clientWidth || this.canvas.width || 200;
    const h = this.canvas.clientHeight || this.canvas.height || 150;
    const padding = 0.08;
    const usableW = w * (1 - padding * 2);
    const usableH = h * (1 - padding * 2);
    const gw = maxX - minX;
    const gh = maxY - minY;
    if (gw <= 0 || gh <= 0) return;
    const scale = Math.min(usableW / gw, usableH / gh);
    const tx = (w - gw * scale) / 2 - minX * scale;
    const ty = (h - gh * scale) / 2 - minY * scale;
    this.cachedScale = scale;
    this.cachedOffsetX = tx;
    this.cachedOffsetY = ty;

    const mat = buildCameraMatrix(w, h, tx, ty, scale);
    this.renderer.device.queue.writeBuffer(this.cameraBuffer, 0, mat as Float32Array<ArrayBuffer>);
  }

  /** Convert minimap-canvas pixel coords to graph coords (used for click-to-pan). */
  minimapToGraph(px: number, py: number): { x: number; y: number } | null {
    if (this.cachedScale <= 0) return null;
    return {
      x: (px - this.cachedOffsetX) / this.cachedScale,
      y: (py - this.cachedOffsetY) / this.cachedScale,
    };
  }

  /** Recreate bind groups for the shared pipelines if their underlying buffers have been replaced. */
  private refreshSharedBindGroups(): void {
    if (!this.renderer) return;
    const device = this.renderer.device;
    const nodes = this.renderer.getNodesPipeline();
    const edges = this.renderer.getEdgesPipeline();

    if (nodes.storageBuffer !== this.lastNodesStorageBuffer || this.nodesBindGroup === null) {
      this.lastNodesStorageBuffer = nodes.storageBuffer;
      this.nodesBindGroup = device.createBindGroup({
        layout: nodes.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.cameraBuffer } },
          { binding: 1, resource: { buffer: nodes.storageBuffer } },
        ],
      });
    }

    if (edges.vertexBuffer !== this.lastEdgesVertexBuffer || this.edgesBindGroup === null) {
      this.lastEdgesVertexBuffer = edges.vertexBuffer;
      this.edgesBindGroup = device.createBindGroup({
        layout: edges.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.cameraBuffer } },
          { binding: 1, resource: { buffer: this.edgeViewportBuffer } },
        ],
      });
    }
  }

  /** Update the geometry of the viewport rect from the main camera state. */
  private rebuildViewportRect(): void {
    if (!this.renderer || !this.viewportPipeline) return;
    if (!this.graphBounds) {
      this.viewportPipeline.vertexCount = 0;
      return;
    }
    const device = this.renderer.device;
    const cam = this.renderer.getCurrentCamera();
    const mainW = this.renderer.canvas.clientWidth || 800;
    const mainH = this.renderer.canvas.clientHeight || 600;
    if (cam.scale === 0) return;

    // Visible-region rect in graph coordinates
    const x0 = -cam.tx / cam.scale;
    const y0 = -cam.ty / cam.scale;
    const x1 = (mainW - cam.tx) / cam.scale;
    const y1 = (mainH - cam.ty) / cam.scale;

    // Outline thickness in graph units (so it stays visible regardless of zoom)
    const thickness = Math.max(2 / this.cachedScale, (this.graphBounds.maxX - this.graphBounds.minX) * 0.0015);

    const r = 0.298, g = 0.553, b = 1.0, a = 1.0;
    const data = new Float32Array(VIEWPORT_MAX_VERTS * VIEWPORT_VERTEX_FLOATS);
    let off = 0;
    const pushQuad = (qx0: number, qy0: number, qx1: number, qy1: number) => {
      // Two triangles for a filled rect
      const verts: [number, number][] = [
        [qx0, qy0], [qx1, qy0], [qx1, qy1],
        [qx0, qy0], [qx1, qy1], [qx0, qy1],
      ];
      for (const [vx, vy] of verts) {
        data[off++] = vx; data[off++] = vy;
        data[off++] = r; data[off++] = g; data[off++] = b; data[off++] = a;
      }
    };
    // Top, bottom, left, right edges
    pushQuad(x0, y0, x1, y0 + thickness);
    pushQuad(x0, y1 - thickness, x1, y1);
    pushQuad(x0, y0 + thickness, x0 + thickness, y1 - thickness);
    pushQuad(x1 - thickness, y0 + thickness, x1, y1 - thickness);

    device.queue.writeBuffer(this.viewportPipeline.vertexBuffer, 0, data as Float32Array<ArrayBuffer>);
    this.viewportPipeline.vertexCount = VIEWPORT_MAX_VERTS;
  }

  /** Resize the canvas drawing buffer + recreate MSAA texture. */
  handleResize(): void {
    if (!this.renderer) return;
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth || this.canvas.width || 200;
    const h = this.canvas.clientHeight || this.canvas.height || 150;
    const physW = Math.max(1, Math.floor(w * dpr));
    const physH = Math.max(1, Math.floor(h * dpr));

    if (this.canvas.width !== physW || this.canvas.height !== physH) {
      this.canvas.width = physW;
      this.canvas.height = physH;
    }

    const device = this.renderer.device;
    if (this.msaaTexture) this.msaaTexture.destroy();
    this.msaaTexture = device.createTexture({
      size: { width: physW, height: physH },
      format: this.renderer.format,
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.msaaView = this.msaaTexture.createView();

    // Edge thickness uniform uses minimap viewport size
    device.queue.writeBuffer(this.edgeViewportBuffer, 0, new Float32Array([physW, physH]) as Float32Array<ArrayBuffer>);

    this.recomputeBoundsAndCamera();
    this.notifyMainCameraChanged();
  }

  /** Draw the minimap into the same command encoder as the main pass. */
  draw(encoder: GPUCommandEncoder): void {
    if (this.suspended) return;
    if (!this.renderer || !this.msaaView) return;
    if (this.canvas.width === 0 || this.canvas.height === 0) return;

    // Re-bind shared buffers if they were reallocated since last frame
    this.refreshSharedBindGroups();
    if (!this.nodesBindGroup || !this.edgesBindGroup) return;

    let textureView: GPUTextureView;
    try {
      textureView = this.context.getCurrentTexture().createView();
    } catch {
      return;
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.msaaView,
        resolveTarget: textureView,
        clearValue: CLEAR_COLOR,
        loadOp: 'clear',
        storeOp: 'discard',
      }],
    });

    const nodes = this.renderer.getNodesPipeline();
    const edges = this.renderer.getEdgesPipeline();

    if (edges.vertexCount > 0) {
      pass.setPipeline(edges.pipeline);
      pass.setBindGroup(0, this.edgesBindGroup);
      pass.setVertexBuffer(0, edges.vertexBuffer);
      pass.draw(edges.vertexCount);
    }

    if (nodes.instanceCount > 0) {
      pass.setPipeline(nodes.pipeline);
      pass.setBindGroup(0, this.nodesBindGroup);
      pass.draw(6, nodes.instanceCount);
    }

    if (this.viewportPipeline && this.viewportPipeline.vertexCount > 0) {
      pass.setPipeline(this.viewportPipeline.pipeline);
      pass.setBindGroup(0, this.viewportPipeline.bindGroup);
      pass.setVertexBuffer(0, this.viewportPipeline.vertexBuffer);
      pass.draw(this.viewportPipeline.vertexCount);
    }

    pass.end();
  }

  destroy(): void {
    this.cameraBuffer.destroy();
    this.edgeViewportBuffer.destroy();
    if (this.msaaTexture) this.msaaTexture.destroy();
    if (this.viewportPipeline) this.viewportPipeline.vertexBuffer.destroy();
    this.renderer = null;
    this.viewportPipeline = null;
    this.nodesBindGroup = null;
    this.edgesBindGroup = null;
  }
}
