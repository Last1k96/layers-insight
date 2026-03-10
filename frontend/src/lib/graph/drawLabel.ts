/**
 * Netron-style node label and hover renderers for Sigma.js.
 *
 * Labels are drawn centered inside the rounded rectangle nodes.
 * Matches Netron's exact styling: 11px system font, white text on dark nodes,
 * black text on light nodes (Constant, Parameter, Result).
 */
import type { Settings } from 'sigma/settings';
import { NODE_RECT_ASPECT } from './nodeRectProgram';
import { isLightNodeColor } from './opColors';

/** Netron system font stack */
const NETRON_FONT = '-apple-system, BlinkMacSystemFont, "Segoe WPC", "Segoe UI", Ubuntu, "Droid Sans", sans-serif';

type LabelData = {
  x: number;
  y: number;
  size: number;
  label: string | null;
  color: string;
  [key: string]: unknown;
};

/**
 * Draw the op-type label centered inside the rectangle node.
 * Uses white text on dark backgrounds, black text on light backgrounds (Netron style).
 */
export function drawRectNodeLabel(
  context: CanvasRenderingContext2D,
  data: LabelData,
  settings: Settings,
): void {
  if (!data.label) return;

  const size = settings.labelSize;
  const halfW = data.size * NODE_RECT_ASPECT;

  context.font = `${size}px ${NETRON_FONT}`;
  context.textAlign = 'center';
  context.textBaseline = 'middle';

  // Truncate label if too wide for the node
  let label = data.label;
  const maxWidth = halfW * 2 - 10;
  if (context.measureText(label).width > maxWidth) {
    while (label.length > 3 && context.measureText(label + '\u2026').width > maxWidth) {
      label = label.slice(0, -1);
    }
    label += '\u2026';
  }

  // Netron: white text on dark nodes, black text on light nodes
  context.fillStyle = isLightNodeColor(data.color) ? '#000' : '#fff';
  context.fillText(label, data.x, data.y + 1);
}

/**
 * Hover effect: Netron-style red selection border.
 * Netron selection: stroke rgba(220, 0, 0, 0.9), stroke-width 2px.
 */
export function drawRectNodeHover(
  context: CanvasRenderingContext2D,
  data: LabelData,
  settings: Settings,
): void {
  const halfH = data.size;
  const halfW = halfH * NODE_RECT_ASPECT;
  const r = Math.min(5, halfH);

  // Subtle shadow glow
  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;
  context.shadowBlur = 8;
  context.shadowColor = 'rgba(255,255,255,0.15)';

  // Netron red highlight border (selection stroke)
  context.beginPath();
  roundedRect(context, data.x - halfW - 1, data.y - halfH - 1, (halfW + 1) * 2, (halfH + 1) * 2, r + 1);
  context.strokeStyle = 'rgba(220, 0, 0, 0.9)';
  context.lineWidth = 2;
  context.stroke();

  // Reset shadow
  context.shadowBlur = 0;

  // Redraw label
  drawRectNodeLabel(context, data, settings);
}

function roundedRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number, w: number, h: number, r: number,
): void {
  r = Math.min(r, w / 2, h / 2);
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
