/**
 * WebGPU pipeline for rendering edges as tessellated thick line strips + arrowheads.
 * Reuses the B-spline math from svgRenderer.ts.
 */
import type { GraphEdge, GraphNode } from '../../stores/types';
import { ALPHA_BLEND } from './types';

const SHADER = /* wgsl */ `
@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> edgeColor: vec4<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
};

@vertex
fn vertexMain(@location(0) pos: vec2<f32>) -> VertexOutput {
  var out: VertexOutput;
  out.position = camera * vec4(pos, 0.0, 1.0);
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
  bindGroup: GPUBindGroup;
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
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module,
      entryPoint: 'vertexMain',
      buffers: [{
        arrayStride: 8,
        attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' as GPUVertexFormat }],
      }],
    },
    fragment: {
      module,
      entryPoint: 'fragmentMain',
      targets: [{ format, blend: ALPHA_BLEND }],
    },
    primitive: { topology: 'triangle-list' },
  });

  const initialCapacity = 4096;
  const vertexBuffer = device.createBuffer({
    size: initialCapacity * 8,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  const colorBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // Default edge color: #888
  device.queue.writeBuffer(colorBuffer, 0, new Float32Array([0.533, 0.533, 0.533, 1.0]) as Float32Array<ArrayBuffer>);

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: { buffer: colorBuffer } },
    ],
  });

  return { pipeline, vertexBuffer, colorBuffer, bindGroup, vertexCount: 0, capacity: initialCapacity };
}

export function setEdgeColor(state: EdgesPipelineState, device: GPUDevice, r: number, g: number, b: number, a: number): void {
  device.queue.writeBuffer(state.colorBuffer, 0, new Float32Array([r, g, b, a]) as Float32Array<ArrayBuffer>);
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
      size: newCapacity * 8,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    state = { ...state, vertexBuffer, capacity: newCapacity };
  }

  device.queue.writeBuffer(state.vertexBuffer, 0, data as Float32Array<ArrayBuffer>, 0, vertexCount * 2);
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
const SEGMENTS_PER_CURVE = 12;
const ARROW_LENGTH = 8;
const ARROW_HALF_WIDTH = 3;

/** Tessellate all edges into a flat Float32Array of triangle vertices */
export function buildEdgeGeometry(
  edges: GraphEdge[],
  nodes: GraphNode[],
  nodeSize: (id: string) => { width: number; height: number },
): Float32Array {
  const nodeMap = new Map<string, GraphNode>();
  for (const n of nodes) nodeMap.set(n.id, n);

  // Estimate vertex count.  Each B-spline span produces SEGMENTS_PER_CURVE line
  // segments (each = quad = 6 verts), + 3 for arrowhead.
  // Dense waypoints (>20) are simplified first, but use raw count as upper bound.
  let estimatedVertices = 0;
  for (const edge of edges) {
    const wp = edge.waypoints?.length ?? 0;
    // B-spline with N control points → (N-1) spans × SEGMENTS_PER_CURVE segments
    const spans = wp >= 2 ? (wp - 1) : 1;
    estimatedVertices += spans * SEGMENTS_PER_CURVE * 6 + 3;
  }
  const buf = new Float32Array(estimatedVertices * 2);
  let offset = 0;

  function pushVertex(x: number, y: number): void {
    if (offset + 2 > buf.length) return; // safety
    buf[offset++] = x;
    buf[offset++] = y;
  }

  function pushQuad(a: Point, b: Point, c: Point, d: Point): void {
    pushVertex(a.x, a.y); pushVertex(b.x, b.y); pushVertex(c.x, c.y);
    pushVertex(a.x, a.y); pushVertex(c.x, c.y); pushVertex(d.x, d.y);
  }

  for (const edge of edges) {
    const points = evaluateEdgeCurve(edge, nodeMap, nodeSize);
    if (points.length < 2) continue;

    // Generate thick line segments
    for (let i = 0; i < points.length - 1; i++) {
      const p0 = points[i];
      const p1 = points[i + 1];
      const dx = p1.x - p0.x;
      const dy = p1.y - p0.y;
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len < 0.001) continue;

      const nx = -dy / len * LINE_HALF_WIDTH;
      const ny = dx / len * LINE_HALF_WIDTH;

      pushQuad(
        { x: p0.x + nx, y: p0.y + ny },
        { x: p1.x + nx, y: p1.y + ny },
        { x: p1.x - nx, y: p1.y - ny },
        { x: p0.x - nx, y: p0.y - ny },
      );
    }

    // Arrowhead at the endpoint
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

      pushVertex(tip.x, tip.y);
      pushVertex(base1.x, base1.y);
      pushVertex(base2.x, base2.y);
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
    // ELK SPLINE routing can produce hundreds of near-collinear waypoints for
    // long-distance edges.  Simplify first, then evaluate as B-spline.
    const pts = edge.waypoints.length > 20
      ? simplifyPolyline(edge.waypoints, 2.0)
      : edge.waypoints;
    return evaluateBSpline(pts);
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
  return evaluateCubicBezier(
    { x: startX, y: startY },
    { x: startX, y: midY },
    { x: endX, y: midY },
    { x: endX, y: endY },
    SEGMENTS_PER_CURVE,
  );
}

function evaluateBSpline(waypoints: Point[]): Point[] {
  const pts = waypoints;
  if (pts.length < 2) return [];

  if (pts.length === 2) {
    const midY = (pts[0].y + pts[1].y) / 2;
    return evaluateCubicBezier(
      pts[0],
      { x: pts[0].x, y: midY },
      { x: pts[1].x, y: midY },
      pts[1],
      SEGMENTS_PER_CURVE,
    );
  }

  if (pts.length === 3) {
    return evaluateQuadBezier(pts[0], pts[1], pts[2], SEGMENTS_PER_CURVE);
  }

  // Cubic B-spline
  const result: Point[] = [pts[0]];

  result.push({
    x: (2 * pts[0].x + pts[1].x) / 3,
    y: (2 * pts[0].y + pts[1].y) / 3,
  });

  for (let i = 1; i < pts.length - 2; i++) {
    const p0 = pts[i];
    const p1 = pts[i + 1];
    const cp1 = { x: (2 * p0.x + p1.x) / 3, y: (2 * p0.y + p1.y) / 3 };
    const cp2 = { x: (p0.x + 2 * p1.x) / 3, y: (p0.y + 2 * p1.y) / 3 };
    const end = { x: (p0.x + 4 * p1.x + pts[i + 2].x) / 6, y: (p0.y + 4 * p1.y + pts[i + 2].y) / 6 };
    const start = result[result.length - 1];
    const bezierPts = evaluateCubicBezier(start, cp1, cp2, end, SEGMENTS_PER_CURVE);
    for (let j = 1; j < bezierPts.length; j++) result.push(bezierPts[j]);
  }

  const n = pts.length;
  const pn2 = pts[n - 2];
  const pn1 = pts[n - 1];
  const lastStart = result[result.length - 1];
  const lastBezier = evaluateCubicBezier(
    lastStart,
    { x: (2 * pn2.x + pn1.x) / 3, y: (2 * pn2.y + pn1.y) / 3 },
    { x: (pn2.x + 2 * pn1.x) / 3, y: (pn2.y + 2 * pn1.y) / 3 },
    pn1,
    SEGMENTS_PER_CURVE,
  );
  for (let j = 1; j < lastBezier.length; j++) result.push(lastBezier[j]);

  return result;
}

/** Douglas-Peucker polyline simplification. */
function simplifyPolyline(pts: Point[], epsilon: number): Point[] {
  if (pts.length <= 2) return pts.slice();

  let maxDist = 0;
  let maxIdx = 0;
  const first = pts[0];
  const last = pts[pts.length - 1];
  const dx = last.x - first.x;
  const dy = last.y - first.y;
  const lenSq = dx * dx + dy * dy;

  for (let i = 1; i < pts.length - 1; i++) {
    let dist: number;
    if (lenSq < 0.001) {
      const ex = pts[i].x - first.x;
      const ey = pts[i].y - first.y;
      dist = Math.sqrt(ex * ex + ey * ey);
    } else {
      const cross = Math.abs(dx * (pts[i].y - first.y) - dy * (pts[i].x - first.x));
      dist = cross / Math.sqrt(lenSq);
    }
    if (dist > maxDist) {
      maxDist = dist;
      maxIdx = i;
    }
  }

  if (maxDist > epsilon) {
    const left = simplifyPolyline(pts.slice(0, maxIdx + 1), epsilon);
    const right = simplifyPolyline(pts.slice(maxIdx), epsilon);
    return left.slice(0, -1).concat(right);
  }

  return [first, last];
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
