/**
 * WebGPU pipeline for rendering edges as tessellated thick line strips +
 * arrowheads.
 *
 * Edge data lives in TWO vertex buffers:
 *
 *   geometryBuffer: 4 floats per vertex (centerX, centerY, normalX, normalY)
 *                   — built once per graph in setGraph(), then immutable
 *                   until the graph data changes.
 *   colorBuffer:    4 floats per vertex (r, g, b, a)
 *                   — rewritten on color-only updates (highlight, search dim,
 *                   accuracy view, etc.) without re-running the spline math.
 *
 * The vertex shader expands the normal to guarantee a minimum screen-space
 * edge width, keeping edges visible when zoomed out. Per-vertex color in the
 * second buffer enables highlight / dim / accuracy color states with a tiny
 * writeBuffer call instead of a full re-tessellation.
 */
import type { GraphEdge, GraphNode } from '../../stores/types';
import { ALPHA_BLEND, EDGE_COLOR } from './types';

/** Floats per vertex in the geometry buffer */
const GEOMETRY_FLOATS = 4;
const GEOMETRY_BYTES = GEOMETRY_FLOATS * 4;
/** Floats per vertex in the color buffer */
const COLOR_FLOATS = 4;
const COLOR_BYTES = COLOR_FLOATS * 4;

const SHADER = /* wgsl */ `
@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<uniform> viewportSize: vec2<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex
fn vertexMain(
  @location(0) center: vec2<f32>,
  @location(1) normal: vec2<f32>,
  @location(2) color: vec4<f32>,
) -> VertexOutput {
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
  out.color = color;
  return out;
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4<f32> {
  return in.color;
}
`;

export interface EdgesPipelineState {
  pipeline: GPURenderPipeline;
  geometryBuffer: GPUBuffer;
  /**
   * Currently-active color buffer. drawEdges() binds this. Points at either
   * `liveColorBuffer` (for transient mixed-color states like hover/select)
   * or one of the cached uniform-color buffers in `cachedColorBuffers`.
   */
  colorBuffer: GPUBuffer;
  /**
   * Writable color buffer for transient mixed states. Always rebuilt fresh
   * when the highlight/search/grayed sets change.
   */
  liveColorBuffer: GPUBuffer;
  /**
   * Lazily-allocated cache of uniform-color buffers, keyed by a state name
   * (e.g. "default", "dim"). Built once per (graph, key) pair and reused on
   * every subsequent activation — accuracy view toggling becomes a buffer
   * reference swap instead of a 38 MB upload.
   *
   * Cleared whenever the geometry is reuploaded with a new vertex capacity.
   */
  cachedColorBuffers: Map<string, GPUBuffer>;
  zoomBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  bindGroupLayout: GPUBindGroupLayout;
  vertexCount: number;
  vertexCapacity: number;
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
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module,
      entryPoint: 'vertexMain',
      buffers: [
        {
          arrayStride: GEOMETRY_BYTES,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x2' as GPUVertexFormat },
            { shaderLocation: 1, offset: 8, format: 'float32x2' as GPUVertexFormat },
          ],
        },
        {
          arrayStride: COLOR_BYTES,
          attributes: [
            { shaderLocation: 2, offset: 0, format: 'float32x4' as GPUVertexFormat },
          ],
        },
      ],
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
  const geometryBuffer = device.createBuffer({
    size: initialCapacity * GEOMETRY_BYTES,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  const liveColorBuffer = device.createBuffer({
    size: initialCapacity * COLOR_BYTES,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  const zoomBuffer = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(zoomBuffer, 0, new Float32Array([1920, 1080]) as Float32Array<ArrayBuffer>);

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: { buffer: zoomBuffer } },
    ],
  });

  return {
    pipeline,
    geometryBuffer,
    colorBuffer: liveColorBuffer,
    liveColorBuffer,
    cachedColorBuffers: new Map(),
    zoomBuffer,
    bindGroup,
    bindGroupLayout,
    vertexCount: 0,
    vertexCapacity: initialCapacity,
  };
}

export function updateEdgeViewport(state: EdgesPipelineState, device: GPUDevice, width: number, height: number): void {
  device.queue.writeBuffer(state.zoomBuffer, 0, new Float32Array([width, height]) as Float32Array<ArrayBuffer>);
}

/**
 * Upload both geometry and colors. Use this when the graph data has changed
 * (or for the accuracy overlay pipeline whose path geometry is rebuilt each
 * time the set of inferred nodes changes).
 *
 * Grows the geometry + live color buffers if vertexCount exceeds the current
 * capacity. Also invalidates any cached uniform-color buffers since their
 * vertex counts no longer match the new geometry.
 */
export function uploadEdgeData(
  state: EdgesPipelineState,
  device: GPUDevice,
  positions: Float32Array,
  colors: Float32Array,
  vertexCount: number,
): EdgesPipelineState {
  if (vertexCount > state.vertexCapacity) {
    state.geometryBuffer.destroy();
    state.liveColorBuffer.destroy();
    for (const buf of state.cachedColorBuffers.values()) buf.destroy();
    state.cachedColorBuffers.clear();

    const newCapacity = Math.max(vertexCount, state.vertexCapacity * 2);
    const geometryBuffer = device.createBuffer({
      size: newCapacity * GEOMETRY_BYTES,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    const liveColorBuffer = device.createBuffer({
      size: newCapacity * COLOR_BYTES,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    state = {
      ...state,
      geometryBuffer,
      liveColorBuffer,
      colorBuffer: liveColorBuffer,
      vertexCapacity: newCapacity,
    };
  } else {
    // Geometry buffer reused. Cached uniform buffers still match the vertex
    // count, but their *content* is stale if the new geometry is laid out
    // differently — invalidate them so the next useCachedEdgeColors() call
    // rebuilds with the current edge layout.
    for (const buf of state.cachedColorBuffers.values()) buf.destroy();
    state.cachedColorBuffers.clear();
  }

  if (vertexCount > 0) {
    device.queue.writeBuffer(state.geometryBuffer, 0, positions as Float32Array<ArrayBuffer>, 0, vertexCount * GEOMETRY_FLOATS);
    device.queue.writeBuffer(state.liveColorBuffer, 0, colors as Float32Array<ArrayBuffer>, 0, vertexCount * COLOR_FLOATS);
  }
  state.vertexCount = vertexCount;
  state.colorBuffer = state.liveColorBuffer;
  return state;
}

/**
 * Color-only update for the live (mixed) color buffer. Used by the highlight
 * / search / grayed paths whose color pattern is different on every call.
 * Always re-binds the live buffer as the active draw target.
 */
export function uploadEdgeColors(
  state: EdgesPipelineState,
  device: GPUDevice,
  colors: Float32Array,
  vertexCount: number,
): void {
  if (vertexCount === 0) return;
  device.queue.writeBuffer(state.liveColorBuffer, 0, colors as Float32Array<ArrayBuffer>, 0, vertexCount * COLOR_FLOATS);
  state.colorBuffer = state.liveColorBuffer;
}

/**
 * If the active draw target is a cached buffer, upload `colors` into the live
 * buffer and switch to it. Call this before patchEdgeColor() when the current
 * base is a cached buffer (e.g. accuracy-view 'dim') — otherwise patchEdgeColor
 * would flip the draw target to a liveColorBuffer whose other vertices are
 * stale from a previous mode.
 */
export function ensureLiveColorBuffer(
  state: EdgesPipelineState,
  device: GPUDevice,
  colors: Float32Array,
): void {
  if (state.colorBuffer === state.liveColorBuffer) return;
  if (state.vertexCount === 0) {
    state.colorBuffer = state.liveColorBuffer;
    return;
  }
  device.queue.writeBuffer(state.liveColorBuffer, 0, colors as Float32Array<ArrayBuffer>, 0, state.vertexCount * COLOR_FLOATS);
  state.colorBuffer = state.liveColorBuffer;
}

/**
 * Patch a single edge's color in the live color buffer. Used for incremental
 * highlight changes — writes only the vertices belonging to one edge instead
 * of rebuilding the entire color array.
 */
export function patchEdgeColor(
  state: EdgesPipelineState,
  device: GPUDevice,
  edgeRanges: Uint32Array,
  edgeIndex: number,
  color: { r: number; g: number; b: number; a: number },
): void {
  const start = edgeRanges[edgeIndex * 2];
  const end = edgeRanges[edgeIndex * 2 + 1];
  if (start >= end) return;
  const count = end - start;
  const data = new Float32Array(count * COLOR_FLOATS);
  for (let i = 0; i < count; i++) {
    const off = i * COLOR_FLOATS;
    data[off] = color.r;
    data[off + 1] = color.g;
    data[off + 2] = color.b;
    data[off + 3] = color.a;
  }
  device.queue.writeBuffer(state.liveColorBuffer, start * COLOR_BYTES, data as Float32Array<ArrayBuffer>);
  state.colorBuffer = state.liveColorBuffer;
}

/**
 * Activate a cached uniform-color buffer by name (e.g. "default", "dim").
 * Builds + uploads the buffer once on first use; subsequent calls are O(1)
 * — they just swap which buffer drawEdges() reads from.
 *
 * The fillFn is invoked only on cache miss. It should return a Float32Array
 * of length vertexCount * 4 holding the per-vertex colors.
 */
export function useCachedEdgeColors(
  state: EdgesPipelineState,
  device: GPUDevice,
  key: string,
  fillFn: () => Float32Array,
): void {
  let cached = state.cachedColorBuffers.get(key);
  if (!cached) {
    cached = device.createBuffer({
      size: state.vertexCapacity * COLOR_BYTES,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    const colors = fillFn();
    if (state.vertexCount > 0) {
      device.queue.writeBuffer(cached, 0, colors as Float32Array<ArrayBuffer>, 0, state.vertexCount * COLOR_FLOATS);
    }
    state.cachedColorBuffers.set(key, cached);
  }
  state.colorBuffer = cached;
}

export function drawEdges(pass: GPURenderPassEncoder, state: EdgesPipelineState): void {
  if (state.vertexCount === 0) return;
  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, state.bindGroup);
  pass.setVertexBuffer(0, state.geometryBuffer);
  pass.setVertexBuffer(1, state.colorBuffer);
  pass.draw(state.vertexCount);
}

// ---- Edge tessellation (B-spline → triangles) ----

interface Point { x: number; y: number }
interface RGBA { r: number; g: number; b: number; a: number }
export interface EdgeColor { start: RGBA; end: RGBA }

const LINE_HALF_WIDTH = 0.6;
const MIN_SEGMENTS = 8;
const MAX_SEGMENTS = 64;
const PIXELS_PER_SEGMENT = 8;
const ARROW_LENGTH = 8;
const ARROW_HALF_WIDTH = 3;

const DEFAULT_RGBA: RGBA = { r: EDGE_COLOR.r, g: EDGE_COLOR.g, b: EDGE_COLOR.b, a: EDGE_COLOR.a };

/**
 * Cached geometry for an edge list. Built once per graph and reused for the
 * lifetime of the renderer (until the graph data changes). The `edgeRanges`
 * array lets the color-only build path locate each edge's vertex slice
 * without re-running the spline math.
 */
export interface EdgeGeometry {
  /** 4 floats per vertex: centerX, centerY, normalX, normalY */
  positions: Float32Array;
  /**
   * Two entries per edge: [startVertex, endVertex). Half-open range covers
   * both the line strip and the arrowhead. Empty edges (no points) get an
   * empty range (start === end).
   */
  edgeRanges: Uint32Array;
  vertexCount: number;
}

/**
 * Tessellate every edge into triangles. Returns positions + per-edge vertex
 * ranges. Color is NOT emitted here — call buildEdgeColors() separately.
 *
 * This is the expensive call (B-spline evaluation, miter normals, etc.) and
 * should only run when the graph data changes — once per setGraph().
 */
export function buildEdgeGeometry(
  edges: GraphEdge[],
  nodes: GraphNode[],
  nodeSize: (id: string) => { width: number; height: number },
): EdgeGeometry {
  const nodeMap = new Map<string, GraphNode>();
  for (const n of nodes) nodeMap.set(n.id, n);

  // Estimate vertex count: per edge, up to MAX_SEGMENTS line segments × 6
  // verts per segment + 3 arrow verts.
  let estimatedVertices = 0;
  for (const edge of edges) {
    const wp = edge.waypoints?.length ?? 0;
    const spans = wp >= 2 ? (wp - 1) : 1;
    estimatedVertices += spans * MAX_SEGMENTS * 6 + 3;
  }
  const positions = new Float32Array(estimatedVertices * GEOMETRY_FLOATS);
  const edgeRanges = new Uint32Array(edges.length * 2);
  let vertexCount = 0;

  function pushVertex(cx: number, cy: number, nx: number, ny: number): void {
    if ((vertexCount + 1) * GEOMETRY_FLOATS > positions.length) return; // safety
    const off = vertexCount * GEOMETRY_FLOATS;
    positions[off] = cx;
    positions[off + 1] = cy;
    positions[off + 2] = nx;
    positions[off + 3] = ny;
    vertexCount++;
  }

  function pushQuad(
    p0x: number, p0y: number, p1x: number, p1y: number,
    n0x: number, n0y: number, n1x: number, n1y: number,
  ): void {
    pushVertex(p0x, p0y, +n0x, +n0y);
    pushVertex(p1x, p1y, +n1x, +n1y);
    pushVertex(p1x, p1y, -n1x, -n1y);
    pushVertex(p0x, p0y, +n0x, +n0y);
    pushVertex(p1x, p1y, -n1x, -n1y);
    pushVertex(p0x, p0y, -n0x, -n0y);
  }

  for (let ei = 0; ei < edges.length; ei++) {
    const edge = edges[ei];
    const startVert = vertexCount;

    const points = evaluateEdgeCurve(edge, nodeMap, nodeSize);
    if (points.length < 2) {
      edgeRanges[ei * 2] = startVert;
      edgeRanges[ei * 2 + 1] = startVert;
      continue;
    }

    // Per-segment normals
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

    // Miter normals (averaged neighbors, scaled to preserve LINE_HALF_WIDTH)
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

    // Arrowhead at the endpoint — base walks back along the curve
    const tip = points[points.length - 1];
    let baseX = tip.x, baseY = tip.y;
    let remaining = ARROW_LENGTH;
    for (let i = points.length - 1; i > 0 && remaining > 0; i--) {
      const dx = points[i].x - points[i - 1].x;
      const dy = points[i].y - points[i - 1].y;
      const segLen = Math.sqrt(dx * dx + dy * dy);
      if (segLen >= remaining) {
        const t = remaining / segLen;
        baseX = points[i].x - dx * t;
        baseY = points[i].y - dy * t;
        remaining = 0;
      } else {
        remaining -= segLen;
        baseX = points[i - 1].x;
        baseY = points[i - 1].y;
      }
    }
    const adx = tip.x - baseX;
    const ady = tip.y - baseY;
    const alen = Math.sqrt(adx * adx + ady * ady);
    if (alen > 0.001) {
      const dirX = adx / alen;
      const dirY = ady / alen;
      const perpX = -dirY;
      const perpY = dirX;
      const base1 = { x: baseX + perpX * ARROW_HALF_WIDTH, y: baseY + perpY * ARROW_HALF_WIDTH };
      const base2 = { x: baseX - perpX * ARROW_HALF_WIDTH, y: baseY - perpY * ARROW_HALF_WIDTH };
      pushVertex(tip.x, tip.y, 0, 0);
      pushVertex(base1.x, base1.y, 0, 0);
      pushVertex(base2.x, base2.y, 0, 0);
    }

    edgeRanges[ei * 2] = startVert;
    edgeRanges[ei * 2 + 1] = vertexCount;
  }

  return {
    positions: positions.subarray(0, vertexCount * GEOMETRY_FLOATS),
    edgeRanges,
    vertexCount,
  };
}

/**
 * Build the colors buffer for a previously-tessellated edge list. Walks the
 * `edgeRanges` from a cached EdgeGeometry and writes the resolved color into
 * each vertex slot. No spline math, no allocation beyond the output array.
 *
 * Pure function — caller is responsible for sizing/uploading. Costs roughly
 * O(vertexCount × 4 float writes), typically <2 ms even on a midrange laptop.
 */
export function buildEdgeColors(
  edgeRanges: Uint32Array,
  edges: GraphEdge[],
  vertexCount: number,
  edgeColorFn?: (edge: GraphEdge, index: number) => EdgeColor | undefined,
): Float32Array {
  const colors = new Float32Array(vertexCount * COLOR_FLOATS);
  for (let ei = 0; ei < edges.length; ei++) {
    const start = edgeRanges[ei * 2];
    const end = edgeRanges[ei * 2 + 1];
    if (start === end) continue;

    // All current callers return uniform start/end colors. We honor only the
    // start color; if a future caller wants gradients, the EdgeGeometry would
    // need to store per-vertex t values.
    const c = (edgeColorFn ? edgeColorFn(edges[ei], ei) : undefined)?.start ?? DEFAULT_RGBA;
    const r = c.r, g = c.g, b = c.b, a = c.a;
    for (let v = start; v < end; v++) {
      const off = v * COLOR_FLOATS;
      colors[off] = r;
      colors[off + 1] = g;
      colors[off + 2] = b;
      colors[off + 3] = a;
    }
  }
  return colors;
}

// ---- Accuracy overlay paths ----

const PATH_HALF_WIDTH = 3.0;
const PATH_ARROW_LENGTH = 14;
const PATH_ARROW_HALF_WIDTH = 7;

/** A path between two inferred nodes: a sequence of edges to stitch into one continuous arrow. */
export interface AccuracyPath {
  edges: GraphEdge[];
  color: { r: number; g: number; b: number; a: number };
}

export interface PathGeometry {
  /** 4 floats per vertex: centerX, centerY, normalX, normalY */
  positions: Float32Array;
  /** 4 floats per vertex: rgba */
  colors: Float32Array;
  vertexCount: number;
}

/**
 * Tessellate accuracy paths as thick continuous arrows. Each path has its
 * own uniform color (one per AccuracyPath, copied to every vertex it owns).
 * The result is uploaded via uploadEdgeData to the overlay edges pipeline.
 */
export function buildPathGeometry(
  paths: AccuracyPath[],
  nodes: GraphNode[],
  nodeSize: (id: string) => { width: number; height: number },
): PathGeometry {
  const nodeMap = new Map<string, GraphNode>();
  for (const n of nodes) nodeMap.set(n.id, n);

  let estimatedVertices = 0;
  for (const path of paths) {
    for (const edge of path.edges) {
      const wp = edge.waypoints?.length ?? 0;
      const spans = wp >= 2 ? (wp - 1) : 1;
      estimatedVertices += spans * MAX_SEGMENTS * 6;
    }
    estimatedVertices += 3;
  }

  const positions = new Float32Array(estimatedVertices * GEOMETRY_FLOATS);
  const colors = new Float32Array(estimatedVertices * COLOR_FLOATS);
  let vertexCount = 0;
  let cr = 0, cg = 0, cb = 0, ca = 1;

  function pushVertex(cx: number, cy: number, nx: number, ny: number): void {
    if ((vertexCount + 1) * GEOMETRY_FLOATS > positions.length) return;
    const goff = vertexCount * GEOMETRY_FLOATS;
    positions[goff] = cx;
    positions[goff + 1] = cy;
    positions[goff + 2] = nx;
    positions[goff + 3] = ny;
    const coff = vertexCount * COLOR_FLOATS;
    colors[coff] = cr;
    colors[coff + 1] = cg;
    colors[coff + 2] = cb;
    colors[coff + 3] = ca;
    vertexCount++;
  }

  function pushQuad(
    p0x: number, p0y: number, p1x: number, p1y: number,
    n0x: number, n0y: number, n1x: number, n1y: number,
  ): void {
    pushVertex(p0x, p0y, +n0x, +n0y);
    pushVertex(p1x, p1y, +n1x, +n1y);
    pushVertex(p1x, p1y, -n1x, -n1y);
    pushVertex(p0x, p0y, +n0x, +n0y);
    pushVertex(p1x, p1y, -n1x, -n1y);
    pushVertex(p0x, p0y, -n0x, -n0y);
  }

  for (const path of paths) {
    cr = path.color.r; cg = path.color.g; cb = path.color.b; ca = path.color.a;

    // Stitch all edge curves into one continuous point list
    const allPoints: Point[] = [];
    for (let ei = 0; ei < path.edges.length; ei++) {
      const edge = path.edges[ei];
      const pts = evaluateEdgeCurve(edge, nodeMap, nodeSize);
      if (pts.length === 0) continue;

      if (allPoints.length > 0) {
        const midNode = nodeMap.get(edge.source);
        if (midNode) {
          const ns = nodeSize(edge.source);
          const cx = midNode.x + ns.width / 2;
          allPoints.push({ x: cx, y: midNode.y + ns.height * 0.5 });
          allPoints.push({ x: cx, y: midNode.y + ns.height });
        }
        for (let i = 1; i < pts.length; i++) allPoints.push(pts[i]);
      } else {
        for (const p of pts) allPoints.push(p);
      }
    }

    if (allPoints.length < 2) continue;

    const segNormals: { nx: number; ny: number }[] = [];
    for (let i = 0; i < allPoints.length - 1; i++) {
      const dx = allPoints[i + 1].x - allPoints[i].x;
      const dy = allPoints[i + 1].y - allPoints[i].y;
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len < 0.001) {
        segNormals.push({ nx: 0, ny: 0 });
      } else {
        segNormals.push({ nx: -dy / len * PATH_HALF_WIDTH, ny: dx / len * PATH_HALF_WIDTH });
      }
    }

    const ptNormals: { nx: number; ny: number }[] = [];
    for (let i = 0; i < allPoints.length; i++) {
      if (i === 0) {
        ptNormals.push(segNormals[0]);
      } else if (i === allPoints.length - 1) {
        ptNormals.push(segNormals[segNormals.length - 1]);
      } else {
        const n0 = segNormals[i - 1];
        const n1 = segNormals[i];
        let mx = (n0.nx + n1.nx) * 0.5;
        let my = (n0.ny + n1.ny) * 0.5;
        const mLen = Math.sqrt(mx * mx + my * my);
        if (mLen > 0.001) {
          const scale = PATH_HALF_WIDTH / mLen;
          mx *= scale;
          my *= scale;
        }
        ptNormals.push({ nx: mx, ny: my });
      }
    }

    for (let i = 0; i < allPoints.length - 1; i++) {
      const sn = segNormals[i];
      if (sn.nx === 0 && sn.ny === 0) continue;
      pushQuad(
        allPoints[i].x, allPoints[i].y, allPoints[i + 1].x, allPoints[i + 1].y,
        ptNormals[i].nx, ptNormals[i].ny,
        ptNormals[i + 1].nx, ptNormals[i + 1].ny,
      );
    }

    // Arrowhead at the end
    const pathTip = allPoints[allPoints.length - 1];
    let pathBaseX = pathTip.x, pathBaseY = pathTip.y;
    let pathRemaining = PATH_ARROW_LENGTH;
    for (let i = allPoints.length - 1; i > 0 && pathRemaining > 0; i--) {
      const dx = allPoints[i].x - allPoints[i - 1].x;
      const dy = allPoints[i].y - allPoints[i - 1].y;
      const segLen = Math.sqrt(dx * dx + dy * dy);
      if (segLen >= pathRemaining) {
        const t = pathRemaining / segLen;
        pathBaseX = allPoints[i].x - dx * t;
        pathBaseY = allPoints[i].y - dy * t;
        pathRemaining = 0;
      } else {
        pathRemaining -= segLen;
        pathBaseX = allPoints[i - 1].x;
        pathBaseY = allPoints[i - 1].y;
      }
    }
    const padx = pathTip.x - pathBaseX;
    const pady = pathTip.y - pathBaseY;
    const palen = Math.sqrt(padx * padx + pady * pady);
    if (palen > 0.001) {
      const dirX = padx / palen;
      const dirY = pady / palen;
      const perpX = -dirY;
      const perpY = dirX;
      const b1 = { x: pathBaseX + perpX * PATH_ARROW_HALF_WIDTH, y: pathBaseY + perpY * PATH_ARROW_HALF_WIDTH };
      const b2 = { x: pathBaseX - perpX * PATH_ARROW_HALF_WIDTH, y: pathBaseY - perpY * PATH_ARROW_HALF_WIDTH };
      pushVertex(pathTip.x, pathTip.y, 0, 0);
      pushVertex(b1.x, b1.y, 0, 0);
      pushVertex(b2.x, b2.y, 0, 0);
    }
  }

  return {
    positions: positions.subarray(0, vertexCount * GEOMETRY_FLOATS),
    colors: colors.subarray(0, vertexCount * COLOR_FLOATS),
    vertexCount,
  };
}

export function evaluateEdgeCurve(
  edge: GraphEdge,
  nodeMap: Map<string, GraphNode>,
  nodeSize: (id: string) => { width: number; height: number },
): Point[] {
  if (edge.waypoints && edge.waypoints.length >= 2) {
    return evaluateBSpline(edge.waypoints);
  }

  const src = nodeMap.get(edge.source);
  const tgt = nodeMap.get(edge.target);
  if (!src || !tgt) return [];

  const ss = nodeSize(edge.source);
  const ts = nodeSize(edge.target);
  const startX = src.x + ss.width / 2;
  const startY = src.y + ss.height;
  const endX = tgt.x + ts.width / 2;
  const endY = tgt.y;

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

  const result: Point[] = [pts[0]];

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
