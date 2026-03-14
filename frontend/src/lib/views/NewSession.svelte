<script lang="ts">
  import { sessionStore } from '../stores/session.svelte';
  import { configStore } from '../stores/config.svelte';
  import { onMount } from 'svelte';
  import type { InputConfig, ModelInputInfo } from '../stores/types';

  let {
    onsessioncreated,
    onback,
  }: {
    onsessioncreated: (id: string) => void;
    onback: () => void;
  } = $props();

  let ovPath = $state('');
  let modelPath = $state('');
  let mainDevice = $state('CPU');
  let refDevice = $state('CPU');
  let submitting = $state(false);
  let error = $state<string | null>(null);

  // Model inputs
  let modelInputs = $state<InputConfig[]>([]);
  let loadingInputs = $state(false);
  let inputsError = $state<string | null>(null);
  let inspectedModelPath = $state('');

  const PRECISIONS = ['fp32', 'fp16', 'i32', 'i64', 'u8', 'i8'];

  const FLOAT_TYPES = new Set(['f16', 'f32', 'f64', 'bf16', 'float16', 'float32', 'float64', 'bfloat16']);
  const FLOAT_PRECISIONS = ['fp32', 'fp16'];
  const INT_PRECISIONS = ['i32', 'i64', 'u8', 'i8'];

  function getAllowedPrecisions(elementType: string): string[] {
    const norm = normalizeElementType(elementType);
    if (FLOAT_TYPES.has(norm)) return FLOAT_PRECISIONS;
    if (norm === 'boolean') return ['u8'];
    if (norm) return INT_PRECISIONS;
    return PRECISIONS;
  }

  // OpenVINO layout options by tensor rank
  // Dimension letters: N=batch, C=channels, D=depth, H=height, W=width
  const LAYOUTS_BY_RANK: Record<number, string[]> = {
    1: ['N'],
    2: ['HW', 'WH'],
    3: ['CHW', 'HWC', 'CWH', 'WHC', 'HCW', 'WCH'],
    4: ['NCHW', 'NHWC', 'NCWH', 'NWHC', 'NHCW', 'NWCH'],
    5: ['NCDHW', 'NDHWC'],
  };

  function getLayoutOptions(shape: number[]): string[] {
    return LAYOUTS_BY_RANK[shape.length] || ['...'];
  }

  function getDefaultLayout(shape: number[]): string {
    const options = getLayoutOptions(shape);
    return options[0] || '...';
  }

  onMount(async () => {
    // Fetch devices and defaults in parallel
    const [defaults] = await Promise.all([
      configStore.fetchDefaults(),
      configStore.fetchDevices(),
    ]);

    // Pre-fill from CLI defaults
    if (defaults) {
      if (defaults.ov_path) ovPath = defaults.ov_path;
      if (defaults.model_path) modelPath = defaults.model_path;
      if (defaults.main_device) mainDevice = defaults.main_device;
      if (defaults.ref_device) refDevice = defaults.ref_device;

      // Auto-inspect model if provided via CLI
      if (defaults.model_path) {
        await inspectModel();
      }
    }
  });

  async function inspectModel() {
    const path = modelPath.trim();
    if (!path || path === inspectedModelPath) return;

    loadingInputs = true;
    inputsError = null;
    modelInputs = [];

    const infos: ModelInputInfo[] = await configStore.fetchModelInputs(path, ovPath || undefined);

    if (infos.length === 0) {
      inputsError = 'Could not read model inputs. Check the model path.';
    } else {
      modelInputs = infos.map((info) => ({
        name: info.name,
        shape: info.shape,
        element_type: info.element_type,
        data_type: elementTypeToDataType(info.element_type),
        source: 'random' as const,
        path: undefined,
        layout: getDefaultLayout(info.shape),
      }));
      inspectedModelPath = path;
    }
    loadingInputs = false;
  }

  /** Extract the inner type name from OV strings like "<Type: 'float32'>" or plain "f32" */
  function normalizeElementType(et: string): string {
    const m = et.match(/'([^']+)'/);
    return m ? m[1] : et;
  }

  function elementTypeToDataType(et: string): string {
    const norm = normalizeElementType(et);
    const map: Record<string, string> = {
      'f32': 'fp32', 'f16': 'fp16', 'f64': 'fp32',
      'float32': 'fp32', 'float16': 'fp16', 'float64': 'fp32', 'bfloat16': 'fp16',
      'i32': 'i32', 'i64': 'i64', 'i8': 'i8',
      'int32': 'i32', 'int64': 'i64', 'int8': 'i8',
      'u8': 'u8', 'uint8': 'u8',
      'boolean': 'u8',
    };
    return map[norm] || 'fp32';
  }

  function formatShape(shape: number[]): string {
    if (shape.length === 0) return 'dynamic';
    return `[${shape.join(', ')}]`;
  }

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
      main_device: mainDevice,
      ref_device: refDevice,
      input_precision: modelInputs.length > 0 ? modelInputs[0].data_type : 'fp32',
      input_layout: modelInputs.length > 0 ? modelInputs[0].layout : 'NCHW',
      inputs: modelInputs.length > 0 ? modelInputs : undefined,
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
          class="w-full px-3 py-2 bg-[--bg-menu] border border-[--border-color] rounded focus:border-blue-500 focus:outline-none"
        />
      </div>

      <div>
        <label class="block text-sm text-gray-400 mb-1">Model Path (.xml) *</label>
        <div class="flex gap-2">
          <input
            type="text"
            bind:value={modelPath}
            placeholder="/path/to/model.xml"
            class="flex-1 px-3 py-2 bg-[--bg-menu] border border-[--border-color] rounded focus:border-blue-500 focus:outline-none"
            required
          />
          <button
            type="button"
            onclick={inspectModel}
            disabled={loadingInputs || !modelPath.trim()}
            class="px-3 py-2 bg-[--bg-menu] hover:bg-[--bg-primary] disabled:bg-[--bg-panel] disabled:text-gray-600 rounded text-sm transition-colors"
          >
            {loadingInputs ? 'Reading...' : 'Inspect'}
          </button>
        </div>
      </div>

      <!-- Model Inputs Section -->
      {#if modelInputs.length > 0}
        <div class="border border-[--border-color] rounded p-3 space-y-3">
          <div class="text-sm text-gray-400 font-medium">Model Inputs ({modelInputs.length})</div>
          {#each modelInputs as input, i (input.name)}
            <div class="bg-[--bg-menu] rounded p-3 space-y-2">
              <div class="flex items-center justify-between">
                <div class="font-mono text-sm text-blue-400">{input.name}</div>
                <div class="text-xs text-gray-500">
                  {input.element_type} &middot; {formatShape(input.shape)}
                </div>
              </div>
              <div class="flex gap-3 items-center">
                <div class="flex-1">
                  <label class="block text-xs text-gray-500 mb-0.5">Source</label>
                  <select
                    bind:value={modelInputs[i].source}
                    class="w-full px-2 py-1.5 bg-[--bg-menu] border border-[--border-color] rounded text-sm focus:border-blue-500 focus:outline-none"
                  >
                    <option value="random">Random</option>
                    <option value="file">File</option>
                  </select>
                </div>
                <div class="flex-1">
                  <label class="block text-xs text-gray-500 mb-0.5">Data Type</label>
                  <select
                    bind:value={modelInputs[i].data_type}
                    class="w-full px-2 py-1.5 bg-[--bg-menu] border border-[--border-color] rounded text-sm focus:border-blue-500 focus:outline-none"
                  >
                    {#each getAllowedPrecisions(input.element_type) as p (p)}
                      <option value={p}>{p.toUpperCase()}</option>
                    {/each}
                  </select>
                </div>
                <div class="flex-1">
                  <label class="block text-xs text-gray-500 mb-0.5">Layout</label>
                  <select
                    bind:value={modelInputs[i].layout}
                    class="w-full px-2 py-1.5 bg-[--bg-menu] border border-[--border-color] rounded text-sm focus:border-blue-500 focus:outline-none"
                  >
                    {#each getLayoutOptions(input.shape) as l (l)}
                      <option value={l}>{l}</option>
                    {/each}
                  </select>
                </div>
              </div>
              {#if input.source === 'file'}
                <div>
                  <input
                    type="text"
                    bind:value={modelInputs[i].path}
                    placeholder="/path/to/input.npy"
                    class="w-full px-2 py-1.5 bg-[--bg-menu] border border-[--border-color] rounded text-sm focus:border-blue-500 focus:outline-none"
                  />
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {:else if inputsError}
        <div class="p-3 bg-yellow-900/30 border border-yellow-800 rounded text-yellow-400 text-sm">
          {inputsError}
        </div>
      {:else if !loadingInputs && modelPath.trim()}
        <div class="text-sm text-gray-500">
          Click "Inspect" to read model inputs.
        </div>
      {/if}

      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-sm text-gray-400 mb-1">Main Device</label>
          <select bind:value={mainDevice} class="w-full px-3 py-2 bg-[--bg-menu] border border-[--border-color] rounded focus:border-blue-500 focus:outline-none">
            {#each configStore.devices as device (device)}
              <option value={device}>{device}</option>
            {/each}
          </select>
        </div>
        <div>
          <label class="block text-sm text-gray-400 mb-1">Reference Device</label>
          <select bind:value={refDevice} class="w-full px-3 py-2 bg-[--bg-menu] border border-[--border-color] rounded focus:border-blue-500 focus:outline-none">
            {#each configStore.devices as device (device)}
              <option value={device}>{device}</option>
            {/each}
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
        class="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-[--bg-panel] disabled:text-gray-500 rounded-lg font-medium transition-colors"
      >
        {submitting ? 'Creating...' : 'Start Session'}
      </button>
    </form>
  </div>
</div>
