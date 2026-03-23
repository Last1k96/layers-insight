/**
 * Canvas 2D fallback renderer — used when WebGPU is not available.
 * Mirrors the public API of WebGPURenderer so the facade can swap transparently.
 */
import type { GraphData, GraphEdge, GraphNode } from '../../stores/types';
import type { NodeStatus } from '../../stores/graph.svelte';
import { configStore } from '../../stores/config.svelte';
import { isLightNodeColor, STATUS_COLORS } from '../opColors';
import { SpatialGrid } from '../webgpu/hitTest';

const NODE_RADIUS = 5;
const GRAPH_FONT_SIZE = 16;
const CLEAR_COLOR = '#1B1E2B';
const EDGE_COLOR = '#5A6080';
const EDGE_COLOR_DIMMED = 'rgba(47,51,65,0.3)';
const LINE_HALF_WIDTH = 0.6;
const ARROW_LENGTH = 8;
const ARROW_HALF_WIDTH = 3;
const MIN_SEGMENTS = 8;
const MAX_SEGMENTS = 64;
const PIXELS_PER_SEGMENT = 8;

interface Point { x: number; y: number }

export class Canvas2DRenderer {
  readonly canvas: HTMLCanvasElement;
  readonly hitGrid = new SpatialGrid();

  private ctx: CanvasRenderingContext2D;
  private dirty = true;
  private animFrameId: number | null = null;
  private resizeObserver: ResizeObserver;

  private cameraTx = 0;
  private cameraTy = 0;
  private cameraScale = 1;
  private currentZoom = 1;

  private graphData: GraphData | null = null;
  private nodeSizeFn: ((id: string) => { width: number; height: number }) | null = null;

  // Cached appearance state
  private nodeStatusMap: Map<string, NodeStatus> = new Map();
  private selectedNodeId: string | null = null;
  private hoveredNodeId: string | null = null;
  private searchResults: { id: string }[] | null = null;
  private searchVisible = false;
  private grayedNodes: Set<string> = new Set();
  private nodeOverrides?: Map<string, { name: string; type: string; color: string }>;

  // Pre-evaluated edge curves (graph-space point arrays)
  private edgeCurves: Point[][] = [];

  private constructor(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D) {
    this.canvas = canvas;
    this.ctx = ctx;

    this.resizeObserver = new ResizeObserver(() => {
      this.handleResize();
      this.markDirty();
    });
    this.resizeObserver.observe(canvas);
    this.handleResize();
    this.frameLoop();
  }

  static create(canvas: HTMLCanvasElement): Canvas2DRenderer {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('[Canvas2D] getContext("2d") returned null');
    console.log('[Canvas2D] fallback renderer initialized');
    return new Canvas2DRenderer(canvas, ctx);
  }

  setGraph(
    graphData: GraphData,
    nodeSize: (id: string) => { width: number; height: number },
  ): void {
    this.graphData = graphData;
    this.nodeSizeFn = nodeSize;

    this.hitGrid.build(graphData.nodes.map(n => ({
      id: n.id,
      x: n.x,
      y: n.y,
      width: nodeSize(n.id).width,
      height: nodeSize(n.id).height,
    })));

    // Pre-evaluate edge curves
    const nodeMap = new Map<string, GraphNode>();
    for (const n of graphData.nodes) nodeMap.set(n.id, n);
    this.edgeCurves = graphData.edges.map(e => evaluateEdgeCurve(e, nodeMap, nodeSize));

    this.markDirty();
  }

  updateCamera(tx: number, ty: number, scale: number): void {
    this.cameraTx = tx;
    this.cameraTy = ty;
    this.cameraScale = scale;
    this.currentZoom = scale;
    this.markDirty();
  }

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
    this.nodeStatusMap = nodeStatusMap;
    this.selectedNodeId = selectedNodeId;
    this.hoveredNodeId = hoveredNodeId;
    this.searchResults = searchResults;
    this.searchVisible = searchVisible;
    this.grayedNodes = grayedNodes;
    this.currentZoom = zoomRatio;
    this.nodeOverrides = nodeOverrides;
    this.markDirty();
  }

  markDirty(): void {
    this.dirty = true;
  }

  destroy(): void {
    if (this.animFrameId !== null) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    this.resizeObserver.disconnect();
  }

  // ---- Private ----

  private handleResize(): void {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;
    const physW = Math.floor(w * dpr);
    const physH = Math.floor(h * dpr);
    if (this.canvas.width === physW && this.canvas.height === physH) return;
    this.canvas.width = physW;
    this.canvas.height = physH;
  }

  private frameLoop(): void {
    this.animFrameId = requestAnimationFrame(() => this.frameLoop());
    if (!this.dirty) return;
    this.dirty = false;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;
    this.render();
  }

  private render(): void {
    const ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Clear
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = CLEAR_COLOR;
    ctx.fillRect(0, 0, w, h);

    // Apply camera: scale by dpr, then pan+zoom
    ctx.setTransform(
      this.cameraScale * dpr, 0,
      0, this.cameraScale * dpr,
      this.cameraTx * dpr, this.cameraTy * dpr,
    );

    if (!this.graphData) return;

    const searchActive = this.searchVisible && this.searchResults && this.searchResults.length > 0;
    const searchSet = searchActive ? new Set(this.searchResults!.map(r => r.id)) : null;

    this.drawEdges(ctx, searchActive);
    this.drawNodes(ctx, searchActive, searchSet);
    if (this.currentZoom >= 0.05) {
      this.drawText(ctx, searchActive, searchSet);
    }
  }

  private drawEdges(ctx: CanvasRenderingContext2D, searchActive: boolean): void {
    ctx.strokeStyle = searchActive ? EDGE_COLOR_DIMMED : EDGE_COLOR;
    ctx.lineWidth = LINE_HALF_WIDTH * 2 / this.cameraScale;
    // Keep edges at least 1 CSS pixel wide
    const minWidth = 1 / this.cameraScale;
    if (ctx.lineWidth < minWidth) ctx.lineWidth = minWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const fillStyle = ctx.strokeStyle;

    for (const points of this.edgeCurves) {
      if (points.length < 2) continue;

      // Draw curve
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      ctx.stroke();

      // Arrowhead
      const last = points[points.length - 1];
      const prev = points[points.length - 2];
      const dx = last.x - prev.x;
      const dy = last.y - prev.y;
      const len = Math.sqrt(dx * dx + dy * dy);
      if (len > 0.001) {
        const dirX = dx / len;
        const dirY = dy / len;
        const perpX = -dirY;
        const perpY = dirX;

        ctx.fillStyle = fillStyle;
        ctx.beginPath();
        ctx.moveTo(last.x, last.y);
        ctx.lineTo(
          last.x - dirX * ARROW_LENGTH + perpX * ARROW_HALF_WIDTH,
          last.y - dirY * ARROW_LENGTH + perpY * ARROW_HALF_WIDTH,
        );
        ctx.lineTo(
          last.x - dirX * ARROW_LENGTH - perpX * ARROW_HALF_WIDTH,
          last.y - dirY * ARROW_LENGTH - perpY * ARROW_HALF_WIDTH,
        );
        ctx.closePath();
        ctx.fill();
      }
    }
  }

  private drawNodes(
    ctx: CanvasRenderingContext2D,
    searchActive: boolean,
    searchSet: Set<string> | null,
  ): void {
    if (!this.graphData || !this.nodeSizeFn) return;

    for (const node of this.graphData.nodes) {
      const override = this.nodeOverrides?.get(node.id);
      const isGrayed = this.grayedNodes.has(node.id);
      const isSelected = this.selectedNodeId === node.id;
      const isHovered = this.hoveredNodeId === node.id;
      const nodeStatus = this.nodeStatusMap.get(node.id);

      let size = this.nodeSizeFn(node.id);
      let dx = 0;
      if (override) {
        const pad = 16;
        const labelWidth = ctx.measureText(override.type).width;
        if (labelWidth + pad > size.width) {
          const newWidth = labelWidth + pad;
          dx = (newWidth - size.width) / 2;
          size = { width: newWidth, height: size.height };
        }
      }

      // Opacity
      let opacity = 1;
      if (isGrayed) opacity = 0.35;
      else if (searchActive && searchSet && !searchSet.has(node.id)) opacity = 0.15;
      ctx.globalAlpha = opacity;

      // Fill color
      const fillColor = override ? override.color : isGrayed ? '#232636' : node.color;

      // Stroke
      let strokeColor = '#333333';
      let strokeWidth = 1;

      if (nodeStatus && !isGrayed) {
        if (nodeStatus.status === 'success' && nodeStatus.metrics) {
          strokeColor = getAccuracyGradientHex(nodeStatus.metrics.mse);
          strokeWidth = isSelected ? 3 : 2;
        } else {
          const sc = STATUS_COLORS[nodeStatus.status];
          if (sc) {
            strokeColor = sc;
            strokeWidth = isSelected ? 3 : 2;
          }
          if (nodeStatus.status === 'executing') strokeWidth = 3;
        }
      }

      if (isSelected) { strokeColor = '#DC0000'; strokeWidth = 2; }
      if (isHovered && !isSelected) { strokeColor = '#DC0000'; strokeWidth = 2; }

      const x = node.x - dx;
      const y = node.y;

      // Draw rounded rect
      ctx.fillStyle = fillColor;
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = strokeWidth;
      roundRect(ctx, x, y, size.width, size.height, NODE_RADIUS);
      ctx.fill();
      ctx.stroke();

      ctx.globalAlpha = 1;
    }
  }

  private drawText(
    ctx: CanvasRenderingContext2D,
    searchActive: boolean,
    searchSet: Set<string> | null,
  ): void {
    if (!this.graphData || !this.nodeSizeFn) return;

    const textAlpha = this.currentZoom >= 0.3 ? 1.0
      : this.currentZoom <= 0.1 ? 0.0
      : (this.currentZoom - 0.1) / 0.2;
    if (textAlpha <= 0) return;

    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';

    for (const node of this.graphData.nodes) {
      const override = this.nodeOverrides?.get(node.id);
      const isGrayed = this.grayedNodes.has(node.id);

      const label = override ? override.type : node.type;
      let size = this.nodeSizeFn(node.id);
      let dx = 0;
      if (override) {
        const pad = 16;
        ctx.font = `${GRAPH_FONT_SIZE}px monospace`;
        const labelW = ctx.measureText(label).width;
        if (labelW + pad > size.width) {
          const newWidth = labelW + pad;
          dx = (newWidth - size.width) / 2;
          size = { width: newWidth, height: size.height };
        }
      }

      const fillColor = override ? override.color : isGrayed ? '#232636' : node.color;
      const glyphAlpha = isGrayed ? textAlpha * 0.35 : textAlpha;
      const isLight = isLightNodeColor(fillColor);

      // Shrink font to fit
      let fontSize = GRAPH_FONT_SIZE;
      ctx.font = `${fontSize}px monospace`;
      const pad = 6;
      let tw = ctx.measureText(label).width;
      if (tw > size.width - pad * 2) {
        fontSize = fontSize * (size.width - pad * 2) / tw;
        ctx.font = `${fontSize}px monospace`;
      }

      ctx.globalAlpha = glyphAlpha;
      ctx.fillStyle = isLight ? '#333333' : '#ffffff';
      ctx.fillText(label, (node.x - dx) + size.width / 2, node.y + size.height / 2);
    }
    ctx.globalAlpha = 1;
  }
}

// ---- Helpers ----

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}

function getAccuracyGradientHex(mse: number): string {
  if (configStore.gradientMode === 'threshold') {
    return mse <= configStore.globalThreshold ? '#10B981' : '#EF4444';
  }
  const logMse = mse > 0 ? Math.log10(mse) : -10;
  const t = Math.max(0, Math.min(1, (logMse + 8) / 6));
  const r = Math.min(1, t * 2);
  const g = Math.min(1, (1 - Math.max(0, t - 0.5) * 2));
  return `rgb(${Math.round(r * 255)},${Math.round(g * 255)},0)`;
}

// ---- Edge curve evaluation (same math as WebGPU edgesPipeline) ----

function evaluateEdgeCurve(
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
  return evaluateCubicBezier(
    { x: startX, y: startY }, { x: startX, y: midY },
    { x: endX, y: midY }, { x: endX, y: endY },
    adaptiveSegments({ x: startX, y: startY }, { x: startX, y: midY }, { x: endX, y: midY }, { x: endX, y: endY }),
  );
}

function evaluateBSpline(waypoints: Point[]): Point[] {
  const pts = waypoints;
  if (pts.length < 2) return [];
  if (pts.length === 2) {
    const midY = (pts[0].y + pts[1].y) / 2;
    return evaluateCubicBezier(pts[0], { x: pts[0].x, y: midY }, { x: pts[1].x, y: midY }, pts[1],
      adaptiveSegments(pts[0], { x: pts[0].x, y: midY }, { x: pts[1].x, y: midY }, pts[1]));
  }
  if (pts.length === 3) {
    return evaluateQuadBezier(pts[0], pts[1], pts[2], adaptiveSegments(pts[0], pts[1], pts[2]));
  }
  const result: Point[] = [pts[0]];
  {
    const cp1 = { x: (2 * pts[0].x + pts[1].x) / 3, y: (2 * pts[0].y + pts[1].y) / 3 };
    const cp2 = { x: (pts[0].x + 2 * pts[1].x) / 3, y: (pts[0].y + 2 * pts[1].y) / 3 };
    const firstEnd = { x: (pts[0].x + 4 * pts[1].x + pts[2].x) / 6, y: (pts[0].y + 4 * pts[1].y + pts[2].y) / 6 };
    const bezier = evaluateCubicBezier(pts[0], cp1, cp2, firstEnd, adaptiveSegments(pts[0], cp1, cp2, firstEnd));
    for (let j = 1; j < bezier.length; j++) result.push(bezier[j]);
  }
  for (let i = 1; i < pts.length - 2; i++) {
    const p0 = pts[i], p1 = pts[i + 1];
    const cp1 = { x: (2 * p0.x + p1.x) / 3, y: (2 * p0.y + p1.y) / 3 };
    const cp2 = { x: (p0.x + 2 * p1.x) / 3, y: (p0.y + 2 * p1.y) / 3 };
    const end = { x: (p0.x + 4 * p1.x + pts[i + 2].x) / 6, y: (p0.y + 4 * p1.y + pts[i + 2].y) / 6 };
    const start = result[result.length - 1];
    const bezier = evaluateCubicBezier(start, cp1, cp2, end, adaptiveSegments(start, cp1, cp2, end));
    for (let j = 1; j < bezier.length; j++) result.push(bezier[j]);
  }
  const n = pts.length;
  const pn2 = pts[n - 2], pn1 = pts[n - 1];
  const lastStart = result[result.length - 1];
  const lastCp1 = { x: (2 * pn2.x + pn1.x) / 3, y: (2 * pn2.y + pn1.y) / 3 };
  const lastCp2 = { x: (pn2.x + 2 * pn1.x) / 3, y: (pn2.y + 2 * pn1.y) / 3 };
  const bezier = evaluateCubicBezier(lastStart, lastCp1, lastCp2, pn1, adaptiveSegments(lastStart, lastCp1, lastCp2, pn1));
  for (let j = 1; j < bezier.length; j++) result.push(bezier[j]);
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
