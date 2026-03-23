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
  let browserTarget = $state<'ov' | 'model' | 'input'>('ov');
  let browserInputIndex = $state(0);

  function openBrowser(target: 'ov' | 'model', inputIndex?: number) {
    if (target === 'ov') {
      browserTarget = 'ov';
      browserMode = 'directory';
      browserInitialPath = ovPath || '';
    } else {
      browserMode = 'file';
      if (inputIndex !== undefined) {
        browserTarget = 'input';
        browserInputIndex = inputIndex;
        const p = modelInputs[inputIndex]?.path || '';
        browserInitialPath = p ? p.substring(0, p.lastIndexOf('/')) : '';
      } else {
        browserTarget = 'model';
        browserInitialPath = modelPath ? modelPath.substring(0, modelPath.lastIndexOf('/')) : '';
      }
    }
    showBrowser = true;
  }

  function onBrowserSelect(path: string) {
    showBrowser = false;
    if (browserTarget === 'ov') {
      ovPath = path;
      onOvPathInput();
    } else if (browserTarget === 'input') {
      modelInputs[browserInputIndex] = { ...modelInputs[browserInputIndex], path };
    } else {
      modelPath = path;
      onModelPathInput();
    }
  }

  const PRECISIONS = ['fp32', 'fp16', 'i32', 'i64', 'u8', 'i8'];

  const FLOAT_TYPES = new Set([
    'f16', 'f32', 'f64', 'bf16', 'float16', 'float32', 'float64', 'bfloat16',
    'nfloat4', 'f4e2m1', 'f8e4m3', 'f8e5m2', 'f8e8m0',
  ]);
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

  function getLayoutOptions(shape: (number | string)[]): string[] {
    return LAYOUTS_BY_RANK[shape.length] || ['...'];
  }

  function getDefaultLayout(shape: (number | string)[]): string {
    const options = getLayoutOptions(shape);
    return options[0] || '...';
  }

  function hasDynamicDims(shape: (number | string)[]): boolean {
    return shape.some(d => typeof d === 'string');
  }

  function handleDimWheel(e: WheelEvent, getValue: () => number, setValue: (v: number) => void, min = 1) {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 1 : -1;
    const step = e.shiftKey ? 10 : 1;
    setValue(Math.max(min, getValue() + delta * step));
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
      modelInputs = infos.map((info) => {
        const dynDims = hasDynamicDims(info.shape);
        return {
          name: info.name,
          shape: info.shape,
          element_type: info.element_type,
          data_type: elementTypeToDataType(info.element_type),
          source: 'random' as const,
          path: undefined,
          layout: getDefaultLayout(info.shape),
          // Initialize dynamic shape fields
          resolved_shape: dynDims ? info.shape.map(d => typeof d === 'string' ? 1 : d as number) : undefined,
          lower_bounds: dynDims ? info.shape.map(d => typeof d === 'string' ? 1 : d as number) : undefined,
          upper_bounds: dynDims ? info.shape.map(d => typeof d === 'string' ? 1024 : d as number) : undefined,
        };
      });
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

        // Apply CLI --input flags to model inputs
        if (defaults.cli_inputs?.length && modelInputs.length > 0) {
          applyCliInputs(defaults.cli_inputs);
        }
      }
    }
  });

  function applyCliInputs(cliInputs: string[]) {
    // Build a map of named inputs (name=path) and a list of positional inputs
    const named = new Map<string, string>();
    const positional: string[] = [];
    for (const entry of cliInputs) {
      const eq = entry.indexOf('=');
      if (eq > 0) {
        named.set(entry.slice(0, eq), entry.slice(eq + 1));
      } else if (entry !== 'random') {
        positional.push(entry);
      }
    }

    // Apply named inputs first
    for (let i = 0; i < modelInputs.length; i++) {
      const path = named.get(modelInputs[i].name);
      if (path) {
        modelInputs[i] = { ...modelInputs[i], source: 'file', path };
      }
    }

    // Apply remaining positional inputs in order to unassigned slots
    let posIdx = 0;
    for (let i = 0; i < modelInputs.length && posIdx < positional.length; i++) {
      if (modelInputs[i].source === 'random') {
        modelInputs[i] = { ...modelInputs[i], source: 'file', path: positional[posIdx++] };
      }
    }
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
      'nfloat4': 'fp16', 'f4e2m1': 'fp16', 'f8e4m3': 'fp16', 'f8e5m2': 'fp16', 'f8e8m0': 'fp16',
      'i32': 'i32', 'i64': 'i64', 'i8': 'i8',
      'int32': 'i32', 'int64': 'i64', 'int8': 'i8',
      'u8': 'u8', 'uint8': 'u8', 'u4': 'u8', 'uint4_t': 'u8',
      'i4': 'i8', 'int4_t': 'i8', 'u1': 'u8', 'uint1_t': 'u8',
      'boolean': 'u8',
    };
    return map[norm] || 'fp32';
  }

  function formatShape(shape: (number | string)[]): string {
    if (shape.length === 0) return 'dynamic';
    return `[${shape.map(d => typeof d === 'string' ? '?' : d).join(', ')}]`;
  }

  async function handleSubmit() {
    if (!modelPath.trim()) {
      error = 'Model path is required';
      return;
    }

    // Validate dynamic shape inputs
    for (const input of modelInputs) {
      if (!hasDynamicDims(input.shape)) continue;

      const rs = input.resolved_shape ?? [];
      const lo = input.lower_bounds ?? [];
      const hi = input.upper_bounds ?? [];

      for (let d = 0; d < input.shape.length; d++) {
        if (typeof input.shape[d] !== 'string') continue; // static dim
        if (input.source === 'random' && (!rs[d] || rs[d] < 1)) {
          error = `Input "${input.name}": dimension ${d} requires a concrete value`;
          return;
        }
        if (lo[d] > hi[d]) {
          error = `Input "${input.name}": dimension ${d} min (${lo[d]}) > max (${hi[d]})`;
          return;
        }
        if (input.source === 'random' && (rs[d] < lo[d] || rs[d] > hi[d])) {
          error = `Input "${input.name}": dimension ${d} value (${rs[d]}) outside bounds [${lo[d]}, ${hi[d]}]`;
          return;
        }
      }
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

<div class="flex-1 flex items-start justify-center p-6 pt-8 bg-[--bg-primary] overflow-y-auto">
  <div class="max-w-lg w-full">
    <div class="flex items-center gap-3 mb-4">
      <button class="text-content-secondary hover:text-content-primary" onclick={onback}>&larr; Back</button>
      <h2 class="text-xl font-bold">New Session</h2>
    </div>

    <form class="space-y-3" onsubmit={(e) => { e.preventDefault(); handleSubmit(); }}>
      <div>
        <label for="ov-path" class="block text-xs text-content-secondary mb-0.5">OpenVINO Path (optional)</label>
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
        <label for="model-path" class="block text-xs text-content-secondary mb-0.5">Model Path (.xml) *</label>
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
        <div class="border border-[--border-color] rounded p-2 space-y-2 max-h-[45vh] overflow-y-auto">
          <div class="text-xs text-content-secondary font-medium sticky top-0 bg-[--bg-surface] z-10 pb-1">Model Inputs ({modelInputs.length})</div>
          {#each modelInputs as input, i (input.name)}
            <div class="bg-[--bg-panel] rounded px-2 py-1.5 space-y-1">
              <div class="flex items-center justify-between">
                <div class="font-mono text-xs text-accent truncate">{input.name}</div>
                <div class="text-[11px] text-content-secondary whitespace-nowrap ml-2">
                  {input.element_type} &middot; {formatShape(input.shape)}
                </div>
              </div>
              <div class="flex gap-2 items-center">
                <div class="flex-1">
                  <select
                    id="source-{i}"
                    bind:value={modelInputs[i].source}
                    title="Source"
                    class="w-full px-1.5 py-1 bg-[--bg-input] border border-[--border-color] rounded text-xs focus:border-accent focus:outline-none"
                  >
                    <option value="random">Random</option>
                    <option value="file">File</option>
                  </select>
                </div>
                <div class="flex-1">
                  <select
                    id="dtype-{i}"
                    bind:value={modelInputs[i].data_type}
                    title="Data Type"
                    class="w-full px-1.5 py-1 bg-[--bg-input] border border-[--border-color] rounded text-xs focus:border-accent focus:outline-none"
                  >
                    {#each getAllowedPrecisions(input.element_type) as p (p)}
                      <option value={p}>{p.toUpperCase()}</option>
                    {/each}
                  </select>
                </div>
                <div class="flex-1">
                  <select
                    id="layout-{i}"
                    bind:value={modelInputs[i].layout}
                    title="Layout"
                    class="w-full px-1.5 py-1 bg-[--bg-input] border border-[--border-color] rounded text-xs focus:border-accent focus:outline-none"
                  >
                    {#each getLayoutOptions(input.shape) as l (l)}
                      <option value={l}>{l}</option>
                    {/each}
                  </select>
                </div>
              </div>
              {#if hasDynamicDims(input.shape)}
                <div class="border border-yellow-700/50 rounded px-2 py-1 space-y-0.5">
                  <div class="text-[11px] text-yellow-400 font-medium">Dynamic Shape</div>
                  <div class="grid gap-1" style="grid-template-columns: auto 1fr 1fr 1fr;">
                    <div class="text-xs text-content-secondary font-medium px-1">Dim</div>
                    {#if input.source === 'random'}
                      <div class="text-xs text-content-secondary font-medium px-1">Value</div>
                    {:else}
                      <div></div>
                    {/if}
                    <div class="text-xs text-content-secondary font-medium px-1">Min</div>
                    <div class="text-xs text-content-secondary font-medium px-1">Max</div>
                    {#each input.shape as dim, d}
                      <div class="text-xs text-content-secondary px-1 py-1.5 font-mono">[{d}]</div>
                      {#if typeof dim === 'string'}
                        {#if input.source === 'random'}
                          <input
                            type="number"
                            min="1"
                            value={input.resolved_shape?.[d] ?? 1}
                            oninput={(e) => {
                              const rs = [...(modelInputs[i].resolved_shape ?? [])];
                              rs[d] = parseInt((e.target as HTMLInputElement).value) || 1;
                              modelInputs[i] = { ...modelInputs[i], resolved_shape: rs };
                            }}
                            onwheel={(e) => handleDimWheel(e,
                              () => modelInputs[i].resolved_shape?.[d] ?? 1,
                              (v) => { const rs = [...(modelInputs[i].resolved_shape ?? [])]; rs[d] = v; modelInputs[i] = { ...modelInputs[i], resolved_shape: rs }; }
                            )}
                            class="w-full px-1.5 py-1 bg-[--bg-input] border border-[--border-color] rounded text-sm focus:border-accent focus:outline-none"
                          />
                        {:else}
                          <div></div>
                        {/if}
                        <input
                          type="number"
                          min="1"
                          value={input.lower_bounds?.[d] ?? 1}
                          oninput={(e) => {
                            const lb = [...(modelInputs[i].lower_bounds ?? [])];
                            lb[d] = parseInt((e.target as HTMLInputElement).value) || 1;
                            modelInputs[i] = { ...modelInputs[i], lower_bounds: lb };
                          }}
                          onwheel={(e) => handleDimWheel(e,
                            () => modelInputs[i].lower_bounds?.[d] ?? 1,
                            (v) => { const lb = [...(modelInputs[i].lower_bounds ?? [])]; lb[d] = v; modelInputs[i] = { ...modelInputs[i], lower_bounds: lb }; }
                          )}
                          class="w-full px-1.5 py-1 bg-[--bg-input] border border-[--border-color] rounded text-sm focus:border-accent focus:outline-none"
                        />
                        <input
                          type="number"
                          min="1"
                          value={input.upper_bounds?.[d] ?? 1024}
                          oninput={(e) => {
                            const ub = [...(modelInputs[i].upper_bounds ?? [])];
                            ub[d] = parseInt((e.target as HTMLInputElement).value) || 1;
                            modelInputs[i] = { ...modelInputs[i], upper_bounds: ub };
                          }}
                          onwheel={(e) => handleDimWheel(e,
                            () => modelInputs[i].upper_bounds?.[d] ?? 1024,
                            (v) => { const ub = [...(modelInputs[i].upper_bounds ?? [])]; ub[d] = v; modelInputs[i] = { ...modelInputs[i], upper_bounds: ub }; }
                          )}
                          class="w-full px-1.5 py-1 bg-[--bg-input] border border-[--border-color] rounded text-sm focus:border-accent focus:outline-none"
                        />
                      {:else}
                        {#if input.source === 'random'}
                          <div class="text-xs text-content-secondary px-1.5 py-1.5 font-mono">{dim}</div>
                        {:else}
                          <div></div>
                        {/if}
                        <div class="text-xs text-content-secondary px-1.5 py-1.5 font-mono">{dim}</div>
                        <div class="text-xs text-content-secondary px-1.5 py-1.5 font-mono">{dim}</div>
                      {/if}
                    {/each}
                  </div>
                </div>
              {/if}
              {#if input.source === 'file'}
                <div class="flex gap-1">
                  <input
                    type="text"
                    bind:value={modelInputs[i].path}
                    placeholder="/path/to/input.npy"
                    class="flex-1 px-2 py-1.5 bg-[--bg-input] border border-[--border-color] rounded text-sm focus:border-accent focus:outline-none"
                  />
                  <button
                    type="button"
                    class="px-2 py-1.5 bg-[--bg-input] border border-[--border-color] rounded hover:bg-[--bg-panel] transition-colors text-sm"
                    title="Browse files"
                    onclick={() => openBrowser('model', i)}
                  >&#128194;</button>
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {/if}

      <div class="grid grid-cols-2 gap-3">
        <div>
          <label for="main-device" class="block text-xs text-content-secondary mb-0.5">Main Device</label>
          <select id="main-device" bind:value={mainDevice} class="w-full px-2 py-1.5 bg-[--bg-input] border border-[--border-color] rounded text-sm focus:border-accent focus:outline-none">
            {#each configStore.devices as device (device)}
              <option value={device}>{device}</option>
            {/each}
          </select>
        </div>
        <div>
          <label for="ref-device" class="block text-xs text-content-secondary mb-0.5">Reference Device</label>
          <select id="ref-device" bind:value={refDevice} class="w-full px-2 py-1.5 bg-[--bg-input] border border-[--border-color] rounded text-sm focus:border-accent focus:outline-none">
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
        class="w-full py-2 bg-accent hover:bg-accent-hover disabled:bg-[--bg-panel] disabled:text-content-secondary rounded-lg font-medium transition-colors text-sm"
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
