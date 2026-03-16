import type { AppDefaults, ModelInputInfo, OvValidationResult } from './types';

class ConfigStore {
  devices = $state<string[]>([]);
  defaults = $state<AppDefaults | null>(null);
  loading = $state(false);

  // Accuracy gradient settings
  gradientMode = $state<'auto' | 'threshold'>('auto');
  globalThreshold = $state(0.01); // MSE threshold
  categoryThresholds = $state<Record<string, number>>({});

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
}

export const configStore = new ConfigStore();
