class ConfigStore {
  devices = $state<string[]>([]);
  ovPath = $state('');
  loading = $state(false);

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
}

export const configStore = new ConfigStore();
