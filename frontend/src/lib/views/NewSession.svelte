<script lang="ts">
  import { sessionStore } from '../stores/session.svelte';
  import { configStore } from '../stores/config.svelte';
  import { onMount } from 'svelte';
  import type { InputConfig, ModelInputInfo } from '../stores/types';
  import FileBrowser from '../components/FileBrowser.svelte';
  import PathInput from '../components/PathInput.svelte';

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

  // OV path validation
  let ovValidating = $state(false);
  let ovError = $state<string | null>(null);

  // Model inputs
  let modelInputs = $state<InputConfig[]>([]);
  let loadingInputs = $state(false);
  let inputsError = $state<string | null>(null);
  let inspectedModelPath = $state('');

  // File browser state
  let showBrowser = $state(false);
  let browserMode = $state<'directory' | 'file'>('directory');
  let browserInitialPath = $state('');
  let browserTarget = $state<'ov' | 'model'>('ov');

  function openBrowser(target: 'ov' | 'model') {
    browserTarget = target;
    if (target === 'ov') {
      browserMode = 'directory';
      browserInitialPath = ovPath || '';
    } else {
      browserMode = 'file';
      browserInitialPath = modelPath ? modelPath.substring(0, modelPath.lastIndexOf('/')) : '';
    }
    showBrowser = true;
  }

  function onBrowserSelect(path: string) {
    showBrowser = false;
    if (browserTarget === 'ov') {
      ovPath = path;
      onOvPathInput();
    } else {
      modelPath = path;
      onModelPathInput();
    }
  }

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

  // --- Debounce utility ---
  function debounce<T extends (...args: any[]) => void>(fn: T, ms: number): T {
    let timer: ReturnType<typeof setTimeout>;
    return ((...args: any[]) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), ms);
    }) as unknown as T;
  }

  // --- Reactive OV path validation ---
  async function doValidateOvPath(path: string) {
    ovValidating = true;
    ovError = null;
    const result = await configStore.validateOvPath(path);
    ovValidating = false;
    if (!result.valid) {
      ovError = result.error || 'Invalid OpenVINO path';
    }
  }

  const debouncedValidateOv = debounce((path: string) => {
    doValidateOvPath(path);
  }, 500);

  function onOvPathInput() {
    const path = ovPath.trim();
    if (!path) {
      // Empty path is valid (system OV)
      ovError = null;
      configStore.devices = ['CPU'];
      return;
    }
    ovValidating = true;
    debouncedValidateOv(path);
  }

  // --- Reactive model inspection ---
  async function doInspectModel(path: string) {
    if (!path) return;
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

  const debouncedInspectModel = debounce((path: string) => {
    doInspectModel(path);
  }, 500);

  function onModelPathInput() {
    const path = modelPath.trim();
    if (!path) {
      modelInputs = [];
      inputsError = null;
      inspectedModelPath = '';
      return;
    }
    if (path === inspectedModelPath) return;
    loadingInputs = true;
    debouncedInspectModel(path);
  }

  onMount(async () => {
    // Fetch defaults to pre-fill form
    const defaults = await configStore.fetchDefaults();

    if (defaults) {
      if (defaults.ov_path) ovPath = defaults.ov_path;
      if (defaults.model_path) modelPath = defaults.model_path;
      if (defaults.main_device) mainDevice = defaults.main_device;
      if (defaults.ref_device) refDevice = defaults.ref_device;

      // Validate OV path to populate devices from the pre-filled path
      if (defaults.ov_path) {
        await doValidateOvPath(defaults.ov_path);
      } else {
        configStore.devices = ['CPU'];
      }

      // Auto-inspect model if provided via CLI
      if (defaults.model_path) {
        await doInspectModel(defaults.model_path);
      }
    }
  });

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

<div class="flex-1 flex items-start justify-center p-8 pt-[15vh] bg-[--bg-primary]">
  <div class="max-w-lg w-full">
    <div class="flex items-center gap-4 mb-6">
      <button class="text-content-secondary hover:text-content-primary" onclick={onback}>&larr; Back</button>
      <h2 class="text-2xl font-bold">New Session</h2>
    </div>

    <form class="space-y-4" onsubmit={(e) => { e.preventDefault(); handleSubmit(); }}>
      <div>
        <label for="ov-path" class="block text-sm text-content-secondary mb-1">OpenVINO Path (optional)</label>
        <div class="relative flex gap-1">
          <PathInput
            bind:value={ovPath}
            mode="directory"
            placeholder="/opt/intel/openvino"
            class="flex-1"
            id="ov-path"
            oninput={onOvPathInput}
          />
          <button
            type="button"
            class="px-2 py-2 bg-[--bg-input] border border-[--border-color] rounded hover:bg-[--bg-panel] transition-colors text-sm"
            title="Browse directories"
            onclick={() => openBrowser('ov')}
          >&#128194;</button>
          {#if ovValidating}
            <div class="absolute right-12 top-1/2 -translate-y-1/2 text-xs text-content-secondary">
              Checking...
            </div>
          {:else if ovError}
            <div class="absolute right-12 top-1/2 -translate-y-1/2 text-yellow-400 cursor-help" title={ovError}>
              &#9888;
            </div>
          {/if}
        </div>
      </div>

      <div>
        <label for="model-path" class="block text-sm text-content-secondary mb-1">Model Path (.xml) *</label>
        <div class="relative flex gap-1">
          <PathInput
            bind:value={modelPath}
            mode="file"
            placeholder="/path/to/model.xml"
            class="flex-1"
            id="model-path"
            oninput={onModelPathInput}
          />
          <button
            type="button"
            class="px-2 py-2 bg-[--bg-input] border border-[--border-color] rounded hover:bg-[--bg-panel] transition-colors text-sm"
            title="Browse files"
            onclick={() => openBrowser('model')}
          >&#128194;</button>
          {#if loadingInputs}
            <div class="absolute right-12 top-1/2 -translate-y-1/2 text-xs text-content-secondary">
              Reading...
            </div>
          {:else if inputsError}
            <div class="absolute right-12 top-1/2 -translate-y-1/2 text-yellow-400 cursor-help" title={inputsError}>
              &#9888;
            </div>
          {/if}
        </div>
      </div>

      <!-- Model Inputs Section -->
      {#if modelInputs.length > 0}
        <div class="border border-[--border-color] rounded p-3 space-y-3">
          <div class="text-sm text-content-secondary font-medium">Model Inputs ({modelInputs.length})</div>
          {#each modelInputs as input, i (input.name)}
            <div class="bg-[--bg-panel] rounded p-3 space-y-2">
              <div class="flex items-center justify-between">
                <div class="font-mono text-sm text-accent">{input.name}</div>
                <div class="text-xs text-content-secondary">
                  {input.element_type} &middot; {formatShape(input.shape)}
                </div>
              </div>
              <div class="flex gap-3 items-center">
                <div class="flex-1">
                  <label for="source-{i}" class="block text-xs text-content-secondary mb-0.5">Source</label>
                  <select
                    id="source-{i}"
                    bind:value={modelInputs[i].source}
                    class="w-full px-2 py-1.5 bg-[--bg-input] border border-[--border-color] rounded text-sm focus:border-accent focus:outline-none"
                  >
                    <option value="random">Random</option>
                    <option value="file">File</option>
                  </select>
                </div>
                <div class="flex-1">
                  <label for="dtype-{i}" class="block text-xs text-content-secondary mb-0.5">Data Type</label>
                  <select
                    id="dtype-{i}"
                    bind:value={modelInputs[i].data_type}
                    class="w-full px-2 py-1.5 bg-[--bg-input] border border-[--border-color] rounded text-sm focus:border-accent focus:outline-none"
                  >
                    {#each getAllowedPrecisions(input.element_type) as p (p)}
                      <option value={p}>{p.toUpperCase()}</option>
                    {/each}
                  </select>
                </div>
                <div class="flex-1">
                  <label for="layout-{i}" class="block text-xs text-content-secondary mb-0.5">Layout</label>
                  <select
                    id="layout-{i}"
                    bind:value={modelInputs[i].layout}
                    class="w-full px-2 py-1.5 bg-[--bg-input] border border-[--border-color] rounded text-sm focus:border-accent focus:outline-none"
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
                    class="w-full px-2 py-1.5 bg-[--bg-input] border border-[--border-color] rounded text-sm focus:border-accent focus:outline-none"
                  />
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/if}

      <div class="grid grid-cols-2 gap-4">
        <div>
          <label for="main-device" class="block text-sm text-content-secondary mb-1">Main Device</label>
          <select id="main-device" bind:value={mainDevice} class="w-full px-3 py-2 bg-[--bg-input] border border-[--border-color] rounded focus:border-accent focus:outline-none">
            {#each configStore.devices as device (device)}
              <option value={device}>{device}</option>
            {/each}
          </select>
        </div>
        <div>
          <label for="ref-device" class="block text-sm text-content-secondary mb-1">Reference Device</label>
          <select id="ref-device" bind:value={refDevice} class="w-full px-3 py-2 bg-[--bg-input] border border-[--border-color] rounded focus:border-accent focus:outline-none">
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
        class="w-full py-3 bg-accent hover:bg-accent-hover disabled:bg-[--bg-panel] disabled:text-content-secondary rounded-lg font-medium transition-colors"
      >
        {submitting ? 'Creating...' : 'Start Session'}
      </button>
    </form>
  </div>
</div>

{#if showBrowser}
  <FileBrowser
    mode={browserMode}
    initialPath={browserInitialPath}
    onselect={onBrowserSelect}
    oncancel={() => showBrowser = false}
  />
{/if}
