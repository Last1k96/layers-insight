<script lang="ts">
  import { sessionStore } from '../stores/session.svelte';
  import { configStore } from '../stores/config.svelte';
  import { onMount } from 'svelte';

  let {
    onsessioncreated,
    onback,
  }: {
    onsessioncreated: (id: string) => void;
    onback: () => void;
  } = $props();

  let ovPath = $state('');
  let modelPath = $state('');
  let inputPath = $state('');
  let inputMode = $state<'random' | 'file'>('random');
  let mainDevice = $state('CPU');
  let refDevice = $state('CPU');
  let inputPrecision = $state('fp32');
  let inputLayout = $state('NCHW');
  let submitting = $state(false);
  let error = $state<string | null>(null);

  onMount(() => {
    configStore.fetchDevices();
  });

  async function handleSubmit() {
    if (!modelPath.trim()) {
      error = 'Model path is required';
      return;
    }
    submitting = true;
    error = null;

    const info = await sessionStore.createSession({
      ov_path: ovPath || undefined,
      model_path: modelPath,
      input_path: inputMode === 'file' ? inputPath : undefined,
      main_device: mainDevice,
      ref_device: refDevice,
      input_precision: inputPrecision,
      input_layout: inputLayout,
    });

    submitting = false;
    if (info) {
      onsessioncreated(info.id);
    } else {
      error = sessionStore.error ?? 'Failed to create session';
    }
  }
</script>

<div class="flex-1 flex items-center justify-center p-8">
  <div class="max-w-lg w-full">
    <div class="flex items-center gap-4 mb-6">
      <button class="text-gray-400 hover:text-gray-200" onclick={onback}>&larr; Back</button>
      <h2 class="text-2xl font-bold">New Session</h2>
    </div>

    <form class="space-y-4" onsubmit={(e) => { e.preventDefault(); handleSubmit(); }}>
      <div>
        <label class="block text-sm text-gray-400 mb-1">OpenVINO Path (optional)</label>
        <input
          type="text"
          bind:value={ovPath}
          placeholder="/opt/intel/openvino"
          class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:border-blue-500 focus:outline-none"
        />
      </div>

      <div>
        <label class="block text-sm text-gray-400 mb-1">Model Path (.xml) *</label>
        <input
          type="text"
          bind:value={modelPath}
          placeholder="/path/to/model.xml"
          class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:border-blue-500 focus:outline-none"
          required
        />
      </div>

      <div>
        <label class="block text-sm text-gray-400 mb-1">Input Data</label>
        <div class="flex gap-4 mb-2">
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="radio" bind:group={inputMode} value="random" class="text-blue-500" />
            <span>Random</span>
          </label>
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="radio" bind:group={inputMode} value="file" class="text-blue-500" />
            <span>From File</span>
          </label>
        </div>
        {#if inputMode === 'file'}
          <input
            type="text"
            bind:value={inputPath}
            placeholder="/path/to/input.npy"
            class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:border-blue-500 focus:outline-none"
          />
        {/if}
      </div>

      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-sm text-gray-400 mb-1">Main Device</label>
          <select bind:value={mainDevice} class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:border-blue-500 focus:outline-none">
            {#each configStore.devices as device (device)}
              <option value={device}>{device}</option>
            {/each}
          </select>
        </div>
        <div>
          <label class="block text-sm text-gray-400 mb-1">Reference Device</label>
          <select bind:value={refDevice} class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:border-blue-500 focus:outline-none">
            {#each configStore.devices as device (device)}
              <option value={device}>{device}</option>
            {/each}
          </select>
        </div>
      </div>

      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-sm text-gray-400 mb-1">Input Precision</label>
          <select bind:value={inputPrecision} class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:border-blue-500 focus:outline-none">
            <option value="fp32">FP32</option>
            <option value="fp16">FP16</option>
            <option value="i32">INT32</option>
            <option value="u8">UINT8</option>
          </select>
        </div>
        <div>
          <label class="block text-sm text-gray-400 mb-1">Input Layout</label>
          <select bind:value={inputLayout} class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:border-blue-500 focus:outline-none">
            <option value="NCHW">NCHW</option>
            <option value="NHWC">NHWC</option>
          </select>
        </div>
      </div>

      {#if error}
        <div class="p-3 bg-red-900/50 border border-red-700 rounded text-red-300 text-sm">
          {error}
        </div>
      {/if}

      <button
        type="submit"
        disabled={submitting}
        class="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg font-medium transition-colors"
      >
        {submitting ? 'Creating...' : 'Start Session'}
      </button>
    </form>
  </div>
</div>
