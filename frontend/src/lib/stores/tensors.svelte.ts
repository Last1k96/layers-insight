import type { TensorMeta } from './types';

class TensorStore {
  cache = $state<Map<string, Float32Array>>(new Map());
  metaCache = $state<Map<string, TensorMeta>>(new Map());
  loading = $state<Set<string>>(new Set());

  private _cacheKey(sessionId: string, taskId: string, outputName: string): string {
    return `${sessionId}:${taskId}:${outputName}`;
  }

  async fetchTensor(sessionId: string, taskId: string, outputName: string): Promise<Float32Array | null> {
    const key = this._cacheKey(sessionId, taskId, outputName);

    // Return cached
    const cached = this.cache.get(key);
    if (cached) return cached;

    // Already loading
    if (this.loading.has(key)) return null;

    this.loading = new Set([...this.loading, key]);

    try {
      const res = await fetch(`/api/tensors/${sessionId}/${taskId}/${outputName}`);
      if (!res.ok) throw new Error(`Failed to fetch tensor: ${res.statusText}`);

      const shapeStr = res.headers.get('X-Tensor-Shape') || '';
      const shape = shapeStr.split(',').map(Number);
      const dtype = res.headers.get('X-Tensor-Dtype') || 'float32';

      const buffer = await res.arrayBuffer();

      // Convert fp16 binary to Float32Array
      const fp16 = new Uint16Array(buffer);
      const fp32 = new Float32Array(fp16.length);
      for (let i = 0; i < fp16.length; i++) {
        fp32[i] = fp16ToFp32(fp16[i]);
      }

      const newCache = new Map(this.cache);
      newCache.set(key, fp32);
      this.cache = newCache;

      // Store meta
      const newMeta = new Map(this.metaCache);
      newMeta.set(key, {
        shape,
        dtype,
        size_bytes: buffer.byteLength,
        min: Math.min(...fp32),
        max: Math.max(...fp32),
        mean: fp32.reduce((a, b) => a + b, 0) / fp32.length,
        std: 0, // computed on demand
      });
      this.metaCache = newMeta;

      return fp32;
    } catch (e) {
      console.error('Failed to fetch tensor:', e);
      return null;
    } finally {
      const newLoading = new Set(this.loading);
      newLoading.delete(key);
      this.loading = newLoading;
    }
  }

  async fetchMeta(sessionId: string, taskId: string, outputName: string): Promise<TensorMeta | null> {
    const key = this._cacheKey(sessionId, taskId, outputName);
    const cached = this.metaCache.get(key);
    if (cached) return cached;

    try {
      const res = await fetch(`/api/tensors/${sessionId}/${taskId}/${outputName}/meta`);
      if (!res.ok) return null;
      const meta: TensorMeta = await res.json();
      const newMeta = new Map(this.metaCache);
      newMeta.set(key, meta);
      this.metaCache = newMeta;
      return meta;
    } catch {
      return null;
    }
  }

  isLoading(sessionId: string, taskId: string, outputName: string): boolean {
    return this.loading.has(this._cacheKey(sessionId, taskId, outputName));
  }

  getTensor(sessionId: string, taskId: string, outputName: string): Float32Array | undefined {
    return this.cache.get(this._cacheKey(sessionId, taskId, outputName));
  }
}

/** Convert a single fp16 value (as uint16) to fp32. */
function fp16ToFp32(h: number): number {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // Subnormal
    const val = (frac / 1024) * Math.pow(2, -14);
    return sign ? -val : val;
  }
  if (exp === 0x1f) {
    return frac ? NaN : (sign ? -Infinity : Infinity);
  }

  const val = Math.pow(2, exp - 15) * (1 + frac / 1024);
  return sign ? -val : val;
}

export const tensorStore = new TensorStore();
