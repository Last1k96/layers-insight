import type { AppDefaults, DeviceProperty, ModelInputInfo, OvValidationResult } from './types';
import { DEFAULT_RANGES, type AccuracyMetricKey, type AccuracyRange } from '../utils/accuracyColors';

/** localStorage key for accuracy settings */
const ACC_STORAGE_KEY = 'layers-insight-accuracy';

interface AccuracySettings {
  enabled: boolean;
  metric: AccuracyMetricKey;
  ranges: Record<AccuracyMetricKey, AccuracyRange>;
}

function loadAccuracySettings(): AccuracySettings {
  try {
    const raw = localStorage.getItem(ACC_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      return {
        enabled: !!parsed.enabled,
        metric: parsed.metric ?? 'cosine_similarity',
        ranges: {
          cosine_similarity: parsed.ranges?.cosine_similarity ?? { ...DEFAULT_RANGES.cosine_similarity },
          mse: parsed.ranges?.mse ?? { ...DEFAULT_RANGES.mse },
          max_abs_diff: parsed.ranges?.max_abs_diff ?? { ...DEFAULT_RANGES.max_abs_diff },
        },
      };
    }
  } catch { /* ignore */ }
  return {
    enabled: false,
    metric: 'cosine_similarity',
    ranges: {
      cosine_similarity: { ...DEFAULT_RANGES.cosine_similarity },
      mse: { ...DEFAULT_RANGES.mse },
      max_abs_diff: { ...DEFAULT_RANGES.max_abs_diff },
    },
  };
}

class ConfigStore {
  devices = $state<string[]>([]);
  defaults = $state<AppDefaults | null>(null);
  loading = $state(false);

  // Accuracy gradient settings (legacy — kept for backward compat)
  gradientMode = $state<'auto' | 'threshold'>('auto');
  globalThreshold = $state(0.01); // MSE threshold
  categoryThresholds = $state<Record<string, number>>({});

  // Unified accuracy overlay settings (persisted to localStorage)
  private _accSettings = loadAccuracySettings();
  accuracyEnabled = $state(this._accSettings.enabled);
  accuracyMetric = $state<AccuracyMetricKey>(this._accSettings.metric);
  accuracyRanges = $state<Record<AccuracyMetricKey, AccuracyRange>>(this._accSettings.ranges);

  /** Convenience getter for the active range */
  get activeRange(): AccuracyRange {
    return this.accuracyRanges[this.accuracyMetric];
  }

  setAccuracyEnabled(enabled: boolean): void {
    this.accuracyEnabled = enabled;
    this.persistAccuracy();
  }

  setAccuracyMetric(metric: AccuracyMetricKey): void {
    this.accuracyMetric = metric;
    this.persistAccuracy();
  }

  setAccuracyRange(metric: AccuracyMetricKey, range: AccuracyRange): void {
    this.accuracyRanges = { ...this.accuracyRanges, [metric]: range };
    this.persistAccuracy();
  }

  private persistAccuracy(): void {
    try {
      localStorage.setItem(ACC_STORAGE_KEY, JSON.stringify({
        enabled: this.accuracyEnabled,
        metric: this.accuracyMetric,
        ranges: this.accuracyRanges,
      }));
    } catch { /* ignore quota errors */ }
  }

  setGradientMode(mode: 'auto' | 'threshold'): void {
    this.gradientMode = mode;
  }

  setGlobalThreshold(value: number): void {
    this.globalThreshold = value;
  }

  setCategoryThreshold(category: string, value: number): void {
    this.categoryThresholds = { ...this.categoryThresholds, [category]: value };
  }

  async fetchDevices(): Promise<void> {
    this.loading = true;
    try {
      const res = await fetch('/api/devices');
      if (res.ok) {
        this.devices = await res.json();
      }
    } catch (e) {
      console.error('Failed to fetch devices:', e);
      this.devices = ['CPU'];
    } finally {
      this.loading = false;
    }
  }

  async fetchDefaults(): Promise<AppDefaults | null> {
    try {
      const res = await fetch('/api/defaults');
      if (res.ok) {
        this.defaults = await res.json();
        return this.defaults;
      }
    } catch (e) {
      console.error('Failed to fetch defaults:', e);
    }
    return null;
  }

  async validateOvPath(ovPath: string): Promise<OvValidationResult> {
    try {
      const params = new URLSearchParams({ ov_path: ovPath });
      const res = await fetch(`/api/validate-ov-path?${params}`);
      if (res.ok) {
        const result: OvValidationResult = await res.json();
        if (result.valid) {
          this.devices = result.devices;
        }
        return result;
      }
    } catch (e) {
      console.error('Failed to validate OV path:', e);
    }
    return { valid: false, devices: ['CPU'], error: 'Request failed' };
  }

  async fetchModelInputs(modelPath: string, ovPath?: string): Promise<ModelInputInfo[]> {
    try {
      const params = new URLSearchParams({ model_path: modelPath });
      if (ovPath) params.set('ov_path', ovPath);
      const res = await fetch(`/api/model-inputs?${params}`);
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.error('Failed to fetch model inputs:', e);
    }
    return [];
  }

  async fetchDeviceConfig(deviceName: string): Promise<DeviceProperty[]> {
    try {
      const res = await fetch(`/api/device-config/${encodeURIComponent(deviceName)}`);
      if (res.ok) {
        return await res.json();
      }
    } catch (e) {
      console.error('Failed to fetch device config:', e);
    }
    return [];
  }
}

export const configStore = new ConfigStore();
