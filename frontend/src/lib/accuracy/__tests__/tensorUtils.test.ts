import { describe, it, expect } from 'vitest';
import {
  getSpatialDims,
  extractSlice,
  computeStats,
  computeHistogram,
  colormapRGB,
  cosineSimilarity,
  formatValue,
} from '../tensorUtils';

describe('getSpatialDims', () => {
  it('handles 1D shape', () => {
    expect(getSpatialDims([10])).toEqual({ batches: 1, channels: 1, height: 1, width: 10 });
  });

  it('handles 2D shape', () => {
    expect(getSpatialDims([3, 4])).toEqual({ batches: 1, channels: 1, height: 3, width: 4 });
  });

  it('handles 3D shape', () => {
    expect(getSpatialDims([3, 8, 8])).toEqual({ batches: 1, channels: 3, height: 8, width: 8 });
  });

  it('handles 4D shape', () => {
    expect(getSpatialDims([1, 3, 224, 224])).toEqual({ batches: 1, channels: 3, height: 224, width: 224 });
  });

  it('handles 5D shape by flattening depth into channels', () => {
    expect(getSpatialDims([2, 3, 4, 8, 8])).toEqual({ batches: 2, channels: 12, height: 8, width: 8 });
  });

  it('handles empty shape', () => {
    expect(getSpatialDims([])).toEqual({ batches: 1, channels: 1, height: 1, width: 1 });
  });

  it('handles 6D+ shape', () => {
    const dims = getSpatialDims([2, 3, 4, 5, 8, 8]);
    expect(dims.batches).toBe(2);
    expect(dims.width).toBe(8);
    expect(dims.height).toBe(8);
  });
});

describe('extractSlice', () => {
  it('returns correct subarray for 4D tensor', () => {
    // [1, 2, 3, 4] — 1 batch, 2 channels, 3 height, 4 width
    const data = new Float32Array(24);
    for (let i = 0; i < 24; i++) data[i] = i;

    const result = extractSlice(data, [1, 2, 3, 4], 0, 0);
    expect(result.h).toBe(3);
    expect(result.w).toBe(4);
    expect(result.data.length).toBe(12);
    expect(result.data[0]).toBe(0);

    const result2 = extractSlice(data, [1, 2, 3, 4], 0, 1);
    expect(result2.data[0]).toBe(12);
  });
});

describe('computeStats', () => {
  it('computes correct stats for normal array', () => {
    const data = new Float32Array([1, 2, 3, 4, 5]);
    const stats = computeStats(data);
    expect(stats.min).toBe(1);
    expect(stats.max).toBe(5);
    expect(stats.mean).toBe(3);
    expect(stats.std).toBeCloseTo(Math.sqrt(2), 4);
  });

  it('returns zeros for empty array', () => {
    const stats = computeStats(new Float32Array(0));
    expect(stats).toEqual({ min: 0, max: 0, mean: 0, std: 0 });
  });
});

describe('computeHistogram', () => {
  it('bins data correctly', () => {
    const data = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const hist = computeHistogram(data, 5);
    expect(hist.counts.length).toBe(5);
    expect(hist.edges.length).toBe(6);
    // Each bin should have 2 values
    for (let i = 0; i < 4; i++) {
      expect(hist.counts[i]).toBe(2);
    }
  });

  it('handles uniform values (degenerate case)', () => {
    const data = new Float32Array([5, 5, 5, 5]);
    const hist = computeHistogram(data, 4);
    // All values should be in one bin
    const total = Array.from(hist.counts).reduce((a, b) => a + b, 0);
    expect(total).toBe(4);
  });
});

describe('colormapRGB', () => {
  it('returns RGB for value 0', () => {
    const [r, g, b] = colormapRGB(0, 'viridis');
    expect(r).toBeGreaterThanOrEqual(0);
    expect(r).toBeLessThanOrEqual(255);
  });

  it('returns RGB for value 1', () => {
    const [r, g, b] = colormapRGB(1, 'viridis');
    expect(r).toBeGreaterThanOrEqual(0);
  });

  it('clamps values below 0', () => {
    const [r1] = colormapRGB(-1, 'viridis');
    const [r2] = colormapRGB(0, 'viridis');
    expect(r1).toBe(r2);
  });

  it('clamps values above 1', () => {
    const [r1] = colormapRGB(2, 'viridis');
    const [r2] = colormapRGB(1, 'viridis');
    expect(r1).toBe(r2);
  });
});

describe('cosineSimilarity', () => {
  it('returns 1 for identical arrays', () => {
    const a = new Float32Array([1, 2, 3]);
    expect(cosineSimilarity(a, a)).toBeCloseTo(1, 5);
  });

  it('returns 0 for orthogonal arrays', () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([0, 1]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0, 5);
  });

  it('returns 0 for zero vectors', () => {
    const a = new Float32Array([0, 0, 0]);
    const b = new Float32Array([1, 2, 3]);
    expect(cosineSimilarity(a, b)).toBe(0);
  });
});

describe('formatValue', () => {
  it('formats zero', () => {
    expect(formatValue(0)).toBe('0');
  });

  it('formats tiny values in exponential', () => {
    expect(formatValue(0.00001)).toMatch(/e/);
  });

  it('formats normal values with fixed(4)', () => {
    expect(formatValue(3.14159)).toBe('3.1416');
  });

  it('formats negative tiny values in exponential', () => {
    expect(formatValue(-0.00001)).toMatch(/e/);
  });
});
