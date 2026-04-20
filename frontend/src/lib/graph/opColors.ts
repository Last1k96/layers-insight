/**
 * Operation type to category mapping with exact Netron colors.
 * Colors match Netron's grapher.css .node-item-type-{category} classes.
 *
 * Netron categories and their RGB values:
 *   layer:          rgb(51, 85, 136)  → #335588
 *   activation:     rgb(112, 41, 33)  → #702921
 *   pool:           rgb(51, 85, 51)   → #335533
 *   normalization:  rgb(51, 85, 68)   → #335544
 *   dropout:        rgb(69, 71, 112)  → #454770
 *   shape:          rgb(108, 79, 71)  → #6C4F47
 *   tensor:         rgb(89, 66, 59)   → #59423B
 *   transform:      rgb(51, 85, 68)   → #335544
 *   data:           rgb(85, 85, 85)   → #555555
 *   quantization:   rgb(80, 40, 0)    → #502800
 *   attention:      rgb(120, 60, 0)   → #783C00
 *   constant:       #eeeeee (light bg, dark text)
 *   control:        #eeeeee (light bg, dark text)
 */

/** Light-colored categories that need dark text */
export const LIGHT_NODE_CATEGORIES = new Set(['Constant', 'Parameter', 'Result']);

export const OP_CATEGORIES: Record<string, { category: string; color: string }> = {
  // Layer / Convolution — deep blue (Netron "layer")
  Convolution: { category: 'Convolution', color: '#335588' },
  GroupConvolution: { category: 'Convolution', color: '#335588' },
  DeformableConvolution: { category: 'Convolution', color: '#335588' },

  // Normalization — teal green (Netron "normalization")
  BatchNormInference: { category: 'Normalization', color: '#335544' },
  MVN: { category: 'Normalization', color: '#335544' },
  NormalizeL2: { category: 'Normalization', color: '#335544' },
  LRN: { category: 'Normalization', color: '#335544' },
  LayerNormalization: { category: 'Normalization', color: '#335544' },
  GroupNormalization: { category: 'Normalization', color: '#335544' },

  // Activation — dark red/maroon (Netron "activation")
  Relu: { category: 'Activation', color: '#702921' },
  Sigmoid: { category: 'Activation', color: '#702921' },
  Tanh: { category: 'Activation', color: '#702921' },
  Clamp: { category: 'Activation', color: '#702921' },
  Elu: { category: 'Activation', color: '#702921' },
  Swish: { category: 'Activation', color: '#702921' },
  PRelu: { category: 'Activation', color: '#702921' },
  Mish: { category: 'Activation', color: '#702921' },
  SoftMax: { category: 'Activation', color: '#702921' },
  Gelu: { category: 'Activation', color: '#702921' },
  HSigmoid: { category: 'Activation', color: '#702921' },
  HSwish: { category: 'Activation', color: '#702921' },

  // Pooling — dark green (Netron "pool")
  MaxPool: { category: 'Pooling', color: '#335533' },
  AvgPool: { category: 'Pooling', color: '#335533' },
  AdaptiveAvgPool: { category: 'Pooling', color: '#335533' },

  // Elementwise — teal green (Netron "transform")
  Add: { category: 'Elementwise', color: '#335544' },
  Multiply: { category: 'Elementwise', color: '#335544' },
  Subtract: { category: 'Elementwise', color: '#335544' },
  Divide: { category: 'Elementwise', color: '#335544' },
  Maximum: { category: 'Elementwise', color: '#335544' },
  Minimum: { category: 'Elementwise', color: '#335544' },
  Power: { category: 'Elementwise', color: '#335544' },

  // MatMul — deep blue (Netron "layer")
  MatMul: { category: 'MatMul', color: '#335588' },
  FullyConnected: { category: 'MatMul', color: '#335588' },

  // DataMovement / Shape — brown (Netron "shape")
  Reshape: { category: 'DataMovement', color: '#6C4F47' },
  Transpose: { category: 'DataMovement', color: '#6C4F47' },
  Concat: { category: 'DataMovement', color: '#6C4F47' },
  Split: { category: 'DataMovement', color: '#6C4F47' },
  StridedSlice: { category: 'DataMovement', color: '#6C4F47' },
  Gather: { category: 'DataMovement', color: '#6C4F47' },
  Squeeze: { category: 'DataMovement', color: '#6C4F47' },
  Unsqueeze: { category: 'DataMovement', color: '#6C4F47' },
  ShapeOf: { category: 'DataMovement', color: '#6C4F47' },
  Convert: { category: 'DataMovement', color: '#6C4F47' },
  Broadcast: { category: 'DataMovement', color: '#6C4F47' },
  Tile: { category: 'DataMovement', color: '#6C4F47' },
  Pad: { category: 'DataMovement', color: '#6C4F47' },
  Interpolate: { category: 'DataMovement', color: '#6C4F47' },

  // Quantization — dark orange (Netron "quantization")
  FakeQuantize: { category: 'Quantization', color: '#502800' },
  Quantize: { category: 'Quantization', color: '#502800' },
  Dequantize: { category: 'Quantization', color: '#502800' },

  // Reduce — muted purple (Netron "dropout")
  ReduceMean: { category: 'Reduce', color: '#454770' },
  ReduceSum: { category: 'Reduce', color: '#454770' },
  ReduceMax: { category: 'Reduce', color: '#454770' },
  ReduceMin: { category: 'Reduce', color: '#454770' },
  ReduceProd: { category: 'Reduce', color: '#454770' },

  // Constant / Parameter / Result — light gray (Netron "constant")
  // These use light background with dark text (like Netron)
  Parameter: { category: 'Parameter', color: '#eeeeee' },
  Result: { category: 'Result', color: '#eeeeee' },
  Constant: { category: 'Constant', color: '#eeeeee' },

  // Attention — orange-brown (Netron "attention")
  ScaledDotProductAttention: { category: 'Attention', color: '#783C00' },
  MultiHeadAttention: { category: 'Attention', color: '#783C00' },

  // Tensor ops — dark brown (Netron "tensor")
  TopK: { category: 'Tensor', color: '#59423B' },
  NonMaxSuppression: { category: 'Tensor', color: '#59423B' },
  ROIPooling: { category: 'Tensor', color: '#59423B' },
};

export const STATUS_COLORS: Record<string, string> = {
  waiting: '#E5A820',
  executing: '#4C8DFF',
  success: '#34C77B',
  failed: '#E54D4D',
};

/** Default color for unknown op types — Netron's default dark charcoal */
const DEFAULT_COLOR = '#333333';

export function getOpColor(opType: string): string {
  return OP_CATEGORIES[opType]?.color ?? DEFAULT_COLOR;
}

export function getOpCategory(opType: string): string {
  return OP_CATEGORIES[opType]?.category ?? 'Other';
}

export function getStatusColor(status: string): string {
  return STATUS_COLORS[status] ?? 'transparent';
}

/**
 * Check if a node color is light (needs dark text).
 *
 * Hot path — this is called per row in QueuePanel / NodeStatus lists.
 * Results are memoized in a module-level Map so parsing happens once
 * per distinct color string across the whole app lifetime.
 */
const _lightCache = new Map<string, boolean>();
export function isLightNodeColor(color: string): boolean {
  const cached = _lightCache.get(color);
  if (cached !== undefined) return cached;
  const hex = color.replace('#', '');
  let result = false;
  if (hex.length === 6) {
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    result = (r * 299 + g * 587 + b * 114) / 1000 > 180;
  }
  _lightCache.set(color, result);
  return result;
}

/**
 * Return the themed foreground hex to render on top of a node-category color.
 * Resolves to either `--text-on-light` (dark) or `--text-primary` (light).
 * Cached alongside `isLightNodeColor`.
 */
const _fgCache = new Map<string, string>();
export const TEXT_ON_LIGHT = '#1B1E2B';
export const TEXT_ON_DARK = '#E1E4ED';
export function getTextOnNodeColor(color: string): string {
  const cached = _fgCache.get(color);
  if (cached !== undefined) return cached;
  const fg = isLightNodeColor(color) ? TEXT_ON_LIGHT : TEXT_ON_DARK;
  _fgCache.set(color, fg);
  return fg;
}
