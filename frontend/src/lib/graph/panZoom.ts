/**
 * Pan/zoom handler for SVG viewport — same approach as Netron.
 * Applies CSS transform on a <g> element.
 */

export type PanZoomListener = () => void;

export class PanZoom {
  private tx = 0;
  private ty = 0;
  private scale = 1;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private viewport: SVGGElement;
  private svg: SVGSVGElement;
  private listeners: PanZoomListener[] = [];
  private animationId: number | null = null;

  // Bound handlers for cleanup
  private onWheel: (e: WheelEvent) => void;
  private onMouseDown: (e: MouseEvent) => void;
  private onMouseMove: (e: MouseEvent) => void;
  private onMouseUp: (e: MouseEvent) => void;

  constructor(svg: SVGSVGElement, viewport: SVGGElement) {
    this.svg = svg;
    this.viewport = viewport;

    this.onWheel = this._handleWheel.bind(this);
    this.onMouseDown = this._handleMouseDown.bind(this);
    this.onMouseMove = this._handleMouseMove.bind(this);
    this.onMouseUp = this._handleMouseUp.bind(this);

    svg.addEventListener('wheel', this.onWheel, { passive: false });
    svg.addEventListener('mousedown', this.onMouseDown);
    window.addEventListener('mousemove', this.onMouseMove);
    window.addEventListener('mouseup', this.onMouseUp);
  }

  /** Current zoom level (higher = more zoomed in) */
  get ratio(): number {
    return this.scale;
  }

  get translateX(): number { return this.tx; }
  get translateY(): number { return this.ty; }

  on(_event: string, fn: PanZoomListener): void {
    this.listeners.push(fn);
  }

  off(_event: string, fn: PanZoomListener): void {
    this.listeners = this.listeners.filter(l => l !== fn);
  }

  private emit(): void {
    for (const fn of this.listeners) fn();
  }

  private applyTransform(): void {
    this.viewport.setAttribute('transform', `translate(${this.tx},${this.ty}) scale(${this.scale})`);
    this.emit();
  }

  private _handleWheel(e: WheelEvent): void {
    e.preventDefault();
    const rect = this.svg.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    const newScale = Math.max(0.01, Math.min(50, this.scale * factor));

    // Zoom centered on cursor
    this.tx = mx - (mx - this.tx) * (newScale / this.scale);
    this.ty = my - (my - this.ty) * (newScale / this.scale);
    this.scale = newScale;
    this.applyTransform();
  }

  private _handleMouseDown(e: MouseEvent): void {
    if (e.button !== 0) return;
    // Only drag on background (svg itself or viewport group)
    const target = e.target as Element;
    if (target !== this.svg && !target.closest('#viewport')) return;
    // Don't drag if clicking a node
    if (target.closest('.node')) return;

    this.dragging = true;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.svg.style.cursor = 'grabbing';
  }

  private _handleMouseMove(e: MouseEvent): void {
    if (!this.dragging) return;
    this.tx += e.clientX - this.lastX;
    this.ty += e.clientY - this.lastY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    this.applyTransform();
  }

  private _handleMouseUp(_e: MouseEvent): void {
    if (this.dragging) {
      this.dragging = false;
      this.svg.style.cursor = '';
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
      // ease-out cubic
      const ease = 1 - Math.pow(1 - t, 3);

      this.tx = startTx + (endTx - startTx) * ease;
      this.ty = startTy + (endTy - startTy) * ease;
      this.scale = startScale + (endScale - startScale) * ease;
      this.applyTransform();

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

  /** Set state directly without animation */
  setState(state: { tx: number; ty: number; scale: number }): void {
    this.tx = state.tx;
    this.ty = state.ty;
    this.scale = state.scale;
    this.applyTransform();
  }

  destroy(): void {
    if (this.animationId !== null) cancelAnimationFrame(this.animationId);
    this.svg.removeEventListener('wheel', this.onWheel);
    this.svg.removeEventListener('mousedown', this.onMouseDown);
    window.removeEventListener('mousemove', this.onMouseMove);
    window.removeEventListener('mouseup', this.onMouseUp);
    this.listeners = [];
  }
}
