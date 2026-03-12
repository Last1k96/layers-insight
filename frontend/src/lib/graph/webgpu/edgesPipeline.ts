/**
 * WebGPU pipeline for rendering edges as tessellated thick line strips + arrowheads.
 * Reuses the B-spline math from svgRenderer.ts.
 *
 * Vertex format: 4 floats per vertex (centerX, centerY, normalX, normalY).
 * The vertex shader expands the normal to guarantee a minimum screen-space
 * edge width, keeping edges visible when zoomed out.
 */
import type { GraphEdge, GraphNode } from '../../stores/types';
import { ALPHA_BLEND } from './types';

const SHADER = /* wgsl */ `
@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> edgeColor: vec4<f32>;
@group(0) @binding(2) var<uniform> viewportSize: vec2<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
};

@vertex
fn vertexMain(@location(0) center: vec2<f32>, @location(1) normal: vec2<f32>) -> VertexOutput {
  var out: VertexOutput;
  // Transform center to clip space
  let clipPos = camera * vec4(center, 0.0, 1.0);
  // Transform normal through camera (linear part only, w=0)
  var clipNormal = vec2(camera[0][0] * normal.x, camera[1][1] * normal.y);
  // Measure screen-pixel length of the normal
  let pixelNormal = clipNormal * viewportSize * 0.5;
  let pixelLen = length(pixelNormal);
  // Ensure minimum ~0.5 pixel half-width (1px total edge width)
  if (pixelLen > 0.0001 && pixelLen < 0.5) {
    clipNormal = clipNormal * (0.5 / pixelLen);
  }
  out.position = vec4(clipPos.xy + clipNormal, clipPos.zw);
  return out;
}

@fragment
fn fragmentMain() -> @location(0) vec4<f32> {
  return edgeColor;
}
`;

export interface EdgesPipelineState {
  pipeline: GPURenderPipeline;
  vertexBuffer: GPUBuffer;
  colorBuffer: GPUBuffer;
  zoomBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  bindGroupLayout: GPUBindGroupLayout;
  vertexCount: number;
  capacity: number;
}

export function createEdgesPipeline(
  device: GPUDevice,
  format: GPUTextureFormat,
  cameraBuffer: GPUBuffer,
): EdgesPipelineState {
  const module = device.createShaderModule({ code: SHADER });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module,
      entryPoint: 'vertexMain',
      buffers: [{
        arrayStride: 16,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x2' as GPUVertexFormat },
          { shaderLocation: 1, offset: 8, format: 'float32x2' as GPUVertexFormat },
        ],
      }],
    },
    fragment: {
      module,
      entryPoint: 'fragmentMain',
      targets: [{ format, blend: ALPHA_BLEND }],
    },
    primitive: { topology: 'triangle-list' },
    multisample: { count: 4 },
  });

  const initialCapacity = 4096;
  const vertexBuffer = device.createBuffer({
    size: initialCapacity * 16,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  const colorBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // Default edge color: #aaa
  device.queue.writeBuffer(colorBuffer, 0, new Float32Array([0.667, 0.667, 0.667, 1.0]) as Float32Array<ArrayBuffer>);

  const zoomBuffer = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // Default viewport size; updated on resize
  device.queue.writeBuffer(zoomBuffer, 0, new Float32Array([1920, 1080]) as Float32Array<ArrayBuffer>);

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: { buffer: colorBuffer } },
      { binding: 2, resource: { buffer: zoomBuffer } },
    ],
  });

  return { pipeline, vertexBuffer, colorBuffer, zoomBuffer, bindGroup, bindGroupLayout, vertexCount: 0, capacity: initialCapacity };
}

export function setEdgeColor(state: EdgesPipelineState, device: GPUDevice, r: number, g: number, b: number, a: number): void {
  device.queue.writeBuffer(state.colorBuffer, 0, new Float32Array([r, g, b, a]) as Float32Array<ArrayBuffer>);
}

export function updateEdgeViewport(state: EdgesPipelineState, device: GPUDevice, width: number, height: number): void {
  device.queue.writeBuffer(state.zoomBuffer, 0, new Float32Array([width, height]) as Float32Array<ArrayBuffer>);
}

export function updateEdgeVertices(
  state: EdgesPipelineState,
  device: GPUDevice,
  cameraBuffer: GPUBuffer,
  data: Float32Array,
  vertexCount: number,
): EdgesPipelineState {
  if (vertexCount > state.capacity) {
    state.vertexBuffer.destroy();
    const newCapacity = Math.max(vertexCount, state.capacity * 2);
    const vertexBuffer = device.createBuffer({
      size: newCapacity * 16,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    // Rebuild bind group since we store layout ref (vertex buffer not in bind group, but keep consistent)
    state = { ...state, vertexBuffer, capacity: newCapacity };
  }

  device.queue.writeBuffer(state.vertexBuffer, 0, data as Float32Array<ArrayBuffer>, 0, vertexCount * 4);
  state.vertexCount = vertexCount;
  return state;
}

export function drawEdges(pass: GPURenderPassEncoder, state: EdgesPipelineState): void {
  if (state.vertexCount === 0) return;
  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, state.bindGroup);
  pass.setVertexBuffer(0, state.vertexBuffer);
  pass.draw(state.vertexCount);
}

// ---- Edge tessellation (B-spline → triangles) ----

interface Point { x: number; y: number }

const LINE_HALF_WIDTH = 0.6;
const MIN_SEGMENTS = 8;
const MAX_SEGMENTS = 64;
const PIXELS_PER_SEGMENT = 8; // one segment per ~8 pixels of arc length
const ARROW_LENGTH = 8;
const ARROW_HALF_WIDTH = 3;

/** Tessellate all edges into a flat Float32Array of triangle vertices.
 *  Each vertex is 4 floats: (centerX, centerY, normalX, normalY).
 *  The normal encodes the perpendicular offset direction * LINE_HALF_WIDTH.
 *  Arrowhead vertices use normal = (0,0) so they render at exact position. */
export function buildEdgeGeometry(
  edges: GraphEdge[],
  nodes: GraphNode[],
  nodeSize: (id: string) => { width: number; height: number },
): Float32Array {
  const nodeMap = new Map<string, GraphNode>();
  for (const n of nodes) nodeMap.set(n.id, n);

  // Estimate vertex count.  Each B-spline span produces up to MAX_SEGMENTS line
  // segments (each = quad = 6 verts), + 3 for arrowhead.
  let estimatedVertices = 0;
  for (const edge of edges) {
    const wp = edge.waypoints?.length ?? 0;
    const spans = wp >= 2 ? (wp - 1) : 1;
    estimatedVertices += spans * MAX_SEGMENTS * 6 + 3;
  }
  const buf = new Float32Array(estimatedVertices * 4);
  let offset = 0;

  function pushVertex(cx: number, cy: number, nx: number, ny: number): void {
    if (offset + 4 > buf.length) return; // safety
    buf[offset++] = cx;
    buf[offset++] = cy;
    buf[offset++] = nx;
    buf[offset++] = ny;
  }

  function pushQuad(
    p0x: number, p0y: number, p1x: number, p1y: number,
    n0x: number, n0y: number,
    n1x?: number, n1y?: number,
  ): void {
    // Two triangles forming a quad: vertices store center + signed normal.
    // Supports per-endpoint normals for miter joins.
    const nx1 = n1x ?? n0x;
    const ny1 = n1y ?? n0y;
    // Triangle 1: (p0,+n0), (p1,+n1), (p1,-n1)
    pushVertex(p0x, p0y, +n0x, +n0y);
    pushVertex(p1x, p1y, +nx1, +ny1);
    pushVertex(p1x, p1y, -nx1, -ny1);
    // Triangle 2: (p0,+n0), (p1,-n1), (p0,-n0)
    pushVertex(p0x, p0y, +n0x, +n0y);
    pushVertex(p1x, p1y, -nx1, -ny1);
    pushVertex(p0x, p0y, -n0x, -n0y);
  }

  for (const edge of edges) {
    const points = evaluateEdgeCurve(edge, nodeMap, nodeSize);
    if (points.length < 2) continue;

    // Generate thick line strip with miter joins for seamless bends.
    // Precompute per-segment normals.
    const segNormals: { nx: number; ny: number }[] = [];
    for (let i = 0; i < points.length - 1; i++) {
      const dx = points[i + 1].x - points[i].x;
      const dy = points[i + 1].y - points[i].y;
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len < 0.001) {
        segNormals.push({ nx: 0, ny: 0 });
      } else {
        segNormals.push({ nx: -dy / len * LINE_HALF_WIDTH, ny: dx / len * LINE_HALF_WIDTH });
      }
    }

    // Compute miter normal at each point (average of adjacent segment normals).
    const ptNormals: { nx: number; ny: number }[] = [];
    for (let i = 0; i < points.length; i++) {
      if (i === 0) {
        ptNormals.push(segNormals[0]);
      } else if (i === points.length - 1) {
        ptNormals.push(segNormals[segNormals.length - 1]);
      } else {
        const n0 = segNormals[i - 1];
        const n1 = segNormals[i];
        let mx = (n0.nx + n1.nx) * 0.5;
        let my = (n0.ny + n1.ny) * 0.5;
        const mLen = Math.sqrt(mx * mx + my * my);
        if (mLen > 0.001) {
          // Scale miter to preserve LINE_HALF_WIDTH perpendicular thickness
          const scale = LINE_HALF_WIDTH / mLen;
          mx *= scale;
          my *= scale;
        }
        ptNormals.push({ nx: mx, ny: my });
      }
    }

    for (let i = 0; i < points.length - 1; i++) {
      const sn = segNormals[i];
      if (sn.nx === 0 && sn.ny === 0) continue;
      pushQuad(
        points[i].x, points[i].y, points[i + 1].x, points[i + 1].y,
        ptNormals[i].nx, ptNormals[i].ny,
        ptNormals[i + 1].nx, ptNormals[i + 1].ny,
      );
    }

    // Arrowhead at the endpoint — use normal=(0,0) so vertices stay fixed
    const last = points[points.length - 1];
    const prev = points[points.length - 2];
    const adx = last.x - prev.x;
    const ady = last.y - prev.y;
    const alen = Math.sqrt(adx * adx + ady * ady);
    if (alen > 0.001) {
      const dirX = adx / alen;
      const dirY = ady / alen;
      const perpX = -dirY;
      const perpY = dirX;

      const tip = last;
      const base1 = { x: tip.x - dirX * ARROW_LENGTH + perpX * ARROW_HALF_WIDTH, y: tip.y - dirY * ARROW_LENGTH + perpY * ARROW_HALF_WIDTH };
      const base2 = { x: tip.x - dirX * ARROW_LENGTH - perpX * ARROW_HALF_WIDTH, y: tip.y - dirY * ARROW_LENGTH - perpY * ARROW_HALF_WIDTH };

      pushVertex(tip.x, tip.y, 0, 0);
      pushVertex(base1.x, base1.y, 0, 0);
      pushVertex(base2.x, base2.y, 0, 0);
    }
  }

  return buf.subarray(0, offset);
}

function evaluateEdgeCurve(
  edge: GraphEdge,
  nodeMap: Map<string, GraphNode>,
  nodeSize: (id: string) => { width: number; height: number },
): Point[] {
  if (edge.waypoints && edge.waypoints.length >= 2) {
    return evaluateBSpline(edge.waypoints);
  }

  // Fallback S-curve
  const src = nodeMap.get(edge.source);
  const tgt = nodeMap.get(edge.target);
  if (!src || !tgt) return [];

  const ss = nodeSize(edge.source);
  const ts = nodeSize(edge.target);
  const startX = src.x + ss.width / 2;
  const startY = src.y + ss.height;
  const endX = tgt.x + ts.width / 2;
  const endY = tgt.y;

  // S-curve via cubic bezier
  const midY = (startY + endY) / 2;
  const sp = { x: startX, y: startY };
  const c1 = { x: startX, y: midY };
  const c2 = { x: endX, y: midY };
  const ep = { x: endX, y: endY };
  return evaluateCubicBezier(sp, c1, c2, ep, adaptiveSegments(sp, c1, c2, ep));
}

function evaluateBSpline(waypoints: Point[]): Point[] {
  const pts = waypoints;
  if (pts.length < 2) return [];

  if (pts.length === 2) {
    const midY = (pts[0].y + pts[1].y) / 2;
    const cp1 = { x: pts[0].x, y: midY };
    const cp2 = { x: pts[1].x, y: midY };
    return evaluateCubicBezier(pts[0], cp1, cp2, pts[1], adaptiveSegments(pts[0], cp1, cp2, pts[1]));
  }

  if (pts.length === 3) {
    return evaluateQuadBezier(pts[0], pts[1], pts[2], adaptiveSegments(pts[0], pts[1], pts[2]));
  }

  // Cubic B-spline
  const result: Point[] = [pts[0]];

  // First segment: proper cubic Bezier from pts[0] to first interior knot
  {
    const cp1 = { x: (2 * pts[0].x + pts[1].x) / 3, y: (2 * pts[0].y + pts[1].y) / 3 };
    const cp2 = { x: (pts[0].x + 2 * pts[1].x) / 3, y: (pts[0].y + 2 * pts[1].y) / 3 };
    const firstEnd = { x: (pts[0].x + 4 * pts[1].x + pts[2].x) / 6, y: (pts[0].y + 4 * pts[1].y + pts[2].y) / 6 };
    const firstBezier = evaluateCubicBezier(pts[0], cp1, cp2, firstEnd, adaptiveSegments(pts[0], cp1, cp2, firstEnd));
    for (let j = 1; j < firstBezier.length; j++) result.push(firstBezier[j]);
  }

  for (let i = 1; i < pts.length - 2; i++) {
    const p0 = pts[i];
    const p1 = pts[i + 1];
    const cp1 = { x: (2 * p0.x + p1.x) / 3, y: (2 * p0.y + p1.y) / 3 };
    const cp2 = { x: (p0.x + 2 * p1.x) / 3, y: (p0.y + 2 * p1.y) / 3 };
    const end = { x: (p0.x + 4 * p1.x + pts[i + 2].x) / 6, y: (p0.y + 4 * p1.y + pts[i + 2].y) / 6 };
    const start = result[result.length - 1];
    const bezierPts = evaluateCubicBezier(start, cp1, cp2, end, adaptiveSegments(start, cp1, cp2, end));
    for (let j = 1; j < bezierPts.length; j++) result.push(bezierPts[j]);
  }

  const n = pts.length;
  const pn2 = pts[n - 2];
  const pn1 = pts[n - 1];
  const lastStart = result[result.length - 1];
  const lastCp1 = { x: (2 * pn2.x + pn1.x) / 3, y: (2 * pn2.y + pn1.y) / 3 };
  const lastCp2 = { x: (pn2.x + 2 * pn1.x) / 3, y: (pn2.y + 2 * pn1.y) / 3 };
  const lastBezier = evaluateCubicBezier(lastStart, lastCp1, lastCp2, pn1, adaptiveSegments(lastStart, lastCp1, lastCp2, pn1));
  for (let j = 1; j < lastBezier.length; j++) result.push(lastBezier[j]);

  return result;
}

function adaptiveSegments(...pts: Point[]): number {
  let dist = 0;
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i].x - pts[i - 1].x;
    const dy = pts[i].y - pts[i - 1].y;
    dist += Math.sqrt(dx * dx + dy * dy);
  }
  return Math.max(MIN_SEGMENTS, Math.min(MAX_SEGMENTS, Math.round(dist / PIXELS_PER_SEGMENT)));
}

function evaluateQuadBezier(p0: Point, p1: Point, p2: Point, segments: number): Point[] {
  const points: Point[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const mt = 1 - t;
    points.push({
      x: mt * mt * p0.x + 2 * mt * t * p1.x + t * t * p2.x,
      y: mt * mt * p0.y + 2 * mt * t * p1.y + t * t * p2.y,
    });
  }
  return points;
}

function evaluateCubicBezier(p0: Point, p1: Point, p2: Point, p3: Point, segments: number): Point[] {
  const points: Point[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const mt = 1 - t;
    points.push({
      x: mt * mt * mt * p0.x + 3 * mt * mt * t * p1.x + 3 * mt * t * t * p2.x + t * t * t * p3.x,
      y: mt * mt * mt * p0.y + 3 * mt * mt * t * p1.y + 3 * mt * t * t * p2.y + t * t * t * p3.y,
    });
  }
  return points;
}
