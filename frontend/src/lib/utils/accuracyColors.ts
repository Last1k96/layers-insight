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
/**
 * Memoization cache for {@link getAccuracyColor}. Keys are stringified
 * `metric|value|range.min|range.max` tuples. The cache is bounded — if it
 * grows past {@link _COLOR_CACHE_MAX} entries we clear it rather than LRU,
 * because bounded clear is O(1) and the cache re-fills within one frame.
 */
const _COLOR_CACHE_MAX = 4096;
const _colorCache = new Map<string, string>();

export function getAccuracyColor(
  metric: AccuracyMetricKey,
  value: number,
  range: AccuracyRange,
): string {
  const key = `${metric}|${value}|${range.min}|${range.max}`;
  const hit = _colorCache.get(key);
  if (hit !== undefined) return hit;

  const { r, g, b } = getAccuracyColorRgb(metric, value, range);
  const ri = Math.round(r * 255);
  const gi = Math.round(g * 255);
  const bi = Math.round(b * 255);
  const out = '#' + ((1 << 24) | (ri << 16) | (gi << 8) | bi).toString(16).slice(1);

  if (_colorCache.size >= _COLOR_CACHE_MAX) _colorCache.clear();
  _colorCache.set(key, out);
  return out;
}

/**
 * Compute a 0-1 "goodness" score for a metric value within the configured range.
 * 0 means "bad", 1 means "good", regardless of metric directionality.
 * Used for progress bar widths and as input to color computation.
 */
export function getAccuracyGoodness(
  metric: AccuracyMetricKey,
  value: number,
  range: AccuracyRange,
): number {
  const span = range.max - range.min;
  if (span === 0) return 0.5;
  if (metric === 'cosine_similarity') {
    return Math.max(0, Math.min(1, (value - range.min) / span));
  } else {
    return Math.max(0, Math.min(1, 1 - (value - range.min) / span));
  }
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
  const t = getAccuracyGoodness(metric, value, range);

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
