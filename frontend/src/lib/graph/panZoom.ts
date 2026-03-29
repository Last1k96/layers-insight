/**
 * Pan/zoom handler for the WebGPU canvas.
 */

export type PanZoomListener = () => void;

export interface PanZoomOptions {
  /** Returns true if the click point hits a node (prevents drag) */
  isNodeHit?: (clientX: number, clientY: number) => boolean;
}

export class PanZoom {
  private tx = 0;
  private ty = 0;
  private scale = 1;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private dragStartX = 0;
  private dragStartY = 0;
  private _didDrag = false;
  private canvas: HTMLCanvasElement;
  private listeners: PanZoomListener[] = [];
  private animationId: number | null = null;
  private isNodeHit: ((clientX: number, clientY: number) => boolean) | null;

  // Bound handlers for cleanup
  private onWheel: (e: WheelEvent) => void;
  private onMouseDown: (e: MouseEvent) => void;
  private onMouseMove: (e: MouseEvent) => void;
  private onMouseUp: (e: MouseEvent) => void;

  constructor(canvas: HTMLCanvasElement, options?: PanZoomOptions) {
    this.canvas = canvas;
    this.isNodeHit = options?.isNodeHit ?? null;

    this.onWheel = this._handleWheel.bind(this);
    this.onMouseDown = this._handleMouseDown.bind(this);
    this.onMouseMove = this._handleMouseMove.bind(this);
    this.onMouseUp = this._handleMouseUp.bind(this);

    canvas.addEventListener('wheel', this.onWheel as EventListener, { passive: false });
    canvas.addEventListener('mousedown', this.onMouseDown as EventListener);
    window.addEventListener('mousemove', this.onMouseMove);
    window.addEventListener('mouseup', this.onMouseUp);
  }

  /** Current zoom level */
  get ratio(): number {
    return this.scale;
  }

  get translateX(): number { return this.tx; }
  get translateY(): number { return this.ty; }

  /** True if the last mousedown->mouseup sequence involved dragging */
  get didDrag(): boolean { return this._didDrag; }

  on(_event: string, fn: PanZoomListener): void {
    this.listeners.push(fn);
  }

  off(_event: string, fn: PanZoomListener): void {
    this.listeners = this.listeners.filter(l => l !== fn);
  }

  private emit(): void {
    for (const fn of this.listeners) fn();
  }

  private _handleWheel(e: WheelEvent): void {
    e.preventDefault();
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const factor = e.deltaY < 0 ? 1.25 : 1 / 1.25;
    const newScale = Math.max(0.01, Math.min(50, this.scale * factor));

    // Zoom centered on cursor
    this.tx = mx - (mx - this.tx) * (newScale / this.scale);
    this.ty = my - (my - this.ty) * (newScale / this.scale);
    this.scale = newScale;
    this.emit();
  }

  private _handleMouseDown(e: MouseEvent): void {
    if (e.button !== 0) return;

    this._didDrag = false;

    // Don't start drag if clicking a node
    if (this.isNodeHit && this.isNodeHit(e.clientX, e.clientY)) return;

    this.dragging = true;
    this.dragStartX = e.clientX;
    this.dragStartY = e.clientY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.canvas.style.cursor = 'grabbing';
  }

  private _handleMouseMove(e: MouseEvent): void {
    if (!this.dragging) return;

    // Drag threshold to distinguish clicks from drags
    if (!this._didDrag) {
      const dx = Math.abs(e.clientX - this.dragStartX);
      const dy = Math.abs(e.clientY - this.dragStartY);
      if (dx + dy < 3) return;
      this._didDrag = true;
    }

    this.tx += e.clientX - this.lastX;
    this.ty += e.clientY - this.lastY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.emit();
  }

  private _handleMouseUp(_e: MouseEvent): void {
    if (this.dragging) {
      this.dragging = false;
      this.canvas.style.cursor = '';
    }
  }

  /** Animate to a target state */
  animate(target: { tx?: number; ty?: number; scale?: number }, duration = 300): void {
    if (this.animationId !== null) cancelAnimationFrame(this.animationId);

    const startTx = this.tx;
    const startTy = this.ty;
    const startScale = this.scale;
    const endTx = target.tx ?? this.tx;
    const endTy = target.ty ?? this.ty;
    const endScale = target.scale ?? this.scale;
    const startTime = performance.now();

    const step = (now: number) => {
      const elapsed = now - startTime;
      const t = Math.min(1, elapsed / duration);
      const ease = 1 - Math.pow(1 - t, 3);

      this.tx = startTx + (endTx - startTx) * ease;
      this.ty = startTy + (endTy - startTy) * ease;
      this.scale = startScale + (endScale - startScale) * ease;
      this.emit();

      if (t < 1) {
        this.animationId = requestAnimationFrame(step);
      } else {
        this.animationId = null;
      }
    };
    this.animationId = requestAnimationFrame(step);
  }

  /** Convert graph coordinates to viewport (screen) coordinates */
  graphToViewport(gx: number, gy: number): { x: number; y: number } {
    return {
      x: gx * this.scale + this.tx,
      y: gy * this.scale + this.ty,
    };
  }

  /** Convert viewport (screen) coordinates to graph coordinates */
  viewportToGraph(vx: number, vy: number): { x: number; y: number } {
    return {
      x: (vx - this.tx) / this.scale,
      y: (vy - this.ty) / this.scale,
    };
  }

  /** Animate the camera so the given bounding box fits the viewport with padding. */
  fitToBounds(bounds: { minX: number; minY: number; maxX: number; maxY: number }, animate = true): void {
    const w = this.canvas.clientWidth || 800;
    const h = this.canvas.clientHeight || 600;
    const margin = 0.1; // 10% padding on each side

    const graphW = bounds.maxX - bounds.minX;
    const graphH = bounds.maxY - bounds.minY;
    if (graphW <= 0 || graphH <= 0) return;

    const usableW = w * (1 - margin * 2);
    const usableH = h * (1 - margin * 2);
    const scaleX = usableW / graphW;
    const scaleY = usableH / graphH;
    const newScale = Math.min(scaleX, scaleY, 1.5);

    const tx = (w - graphW * newScale) / 2 - bounds.minX * newScale;
    const ty = (h - graphH * newScale) / 2 - bounds.minY * newScale;

    if (animate) {
      this.animate({ tx, ty, scale: newScale }, 400);
    } else {
      this.setState({ tx, ty, scale: newScale });
    }
  }

  /** Set state directly without animation */
  setState(state: { tx: number; ty: number; scale: number }): void {
    this.tx = state.tx;
    this.ty = state.ty;
    this.scale = state.scale;
    this.emit();
  }

  destroy(): void {
    if (this.animationId !== null) cancelAnimationFrame(this.animationId);
    this.canvas.removeEventListener('wheel', this.onWheel as EventListener);
    this.canvas.removeEventListener('mousedown', this.onMouseDown as EventListener);
    window.removeEventListener('mousemove', this.onMouseMove);
    window.removeEventListener('mouseup', this.onMouseUp);
    this.listeners = [];
  }
}
