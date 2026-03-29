/**
 * Shared accuracy color utilities.
 *
 * Provides a single function that maps a metric value + range to a CSS color
 * string. Used by the graph renderer (node fill / stroke), queue panel, and
 * any other component that needs accuracy-driven coloring.
 */

export type AccuracyMetricKey = 'cosine_similarity' | 'mse' | 'max_abs_diff';

export interface AccuracyRange {
  min: number;
  max: number;
}

/** Default ranges per metric */
export const DEFAULT_RANGES: Record<AccuracyMetricKey, AccuracyRange> = {
  cosine_similarity: { min: 0.99, max: 1.0 },
  mse: { min: 0, max: 0.01 },
  max_abs_diff: { min: 0, max: 0.01 },
};

/**
 * Compute an accuracy color for a given metric value within the configured range.
 *
 * For cosine_similarity: low values are bad (red), high values are good (green).
 * For mse / max_abs_diff: low values are good (green), high values are bad (red).
 *
 * Values outside the range are clamped.
 *
 * @returns CSS hex color string (e.g. "#10B981")
 */
export function getAccuracyColor(
  metric: AccuracyMetricKey,
  value: number,
  range: AccuracyRange,
): string {
  const { r, g, b } = getAccuracyColorRgb(metric, value, range);
  const ri = Math.round(r * 255);
  const gi = Math.round(g * 255);
  const bi = Math.round(b * 255);
  return '#' + ((1 << 24) | (ri << 16) | (gi << 8) | bi).toString(16).slice(1);
}

/**
 * Same as getAccuracyColor but returns normalized 0-1 RGB.
 * Used by the WebGPU renderer which works in float color space.
 */
export function getAccuracyColorRgb(
  metric: AccuracyMetricKey,
  value: number,
  range: AccuracyRange,
): { r: number; g: number; b: number } {
  const span = range.max - range.min;
  if (span === 0) return { r: 0.5, g: 0.5, b: 0.0 };

  // t = 0 means "bad", t = 1 means "good"
  let t: number;
  if (metric === 'cosine_similarity') {
    // Higher is better
    t = Math.max(0, Math.min(1, (value - range.min) / span));
  } else {
    // Lower is better (mse, max_abs_diff)
    t = Math.max(0, Math.min(1, 1 - (value - range.min) / span));
  }

  // Smooth red-to-green gradient via yellow midpoint
  //   t=0 (bad)  -> red   (0.937, 0.267, 0.267) ~ #EF4444
  //   t=0.5      -> amber (0.900, 0.700, 0.100)
  //   t=1 (good) -> green (0.063, 0.725, 0.506) ~ #10B981
  const r = t < 0.5
    ? 0.937 - (0.937 - 0.900) * (t / 0.5)
    : 0.900 - (0.900 - 0.063) * ((t - 0.5) / 0.5);
  const g = t < 0.5
    ? 0.267 + (0.700 - 0.267) * (t / 0.5)
    : 0.700 + (0.725 - 0.700) * ((t - 0.5) / 0.5);
  const b = t < 0.5
    ? 0.267 - (0.267 - 0.100) * (t / 0.5)
    : 0.100 + (0.506 - 0.100) * ((t - 0.5) / 0.5);

  return { r, g, b };
}
