<script lang="ts">
  import { sessionStore } from '../stores/session.svelte';
  import { configStore } from '../stores/config.svelte';
  import { onMount } from 'svelte';
  import type { InputConfig, ModelInputInfo, DeviceProperty } from '../stores/types';
  import FileBrowser from '../components/FileBrowser.svelte';
  import PathInput from '../components/PathInput.svelte';
  import SourceField from '../components/SourceField.svelte';
  import { deleteUploadGroup, type StagedFile } from '../stores/upload.svelte';
  import { CLI_CONSUMED_KEY, cliFingerprint } from '../initView';

  type SourceMode = 'server' | 'upload';

  let {
    onsessioncreated,
    onback,
    cloneSourceId = undefined,
    cloneSourceConfig = undefined,
    cloneSourceName = undefined,
  }: {
    onsessioncreated: (id: string) => void;
    onback: () => void;
    cloneSourceId?: string;
    cloneSourceConfig?: import('../stores/types').SessionConfig;
    cloneSourceName?: string;
  } = $props();

  let ovPath = $state('');
  let modelPath = $state('');
  let sessionName = $state('');
  let mainDevice = $state('CPU');
  let refDevice = $state('CPU');
  let submitting = $state(false);
  let error = $state<string | null>(null);

  // Upload state — shared group for the whole NewSession draft.
  let uploadGroupId = $state<string | null>(null);
  let modelSource = $state<SourceMode>('server');
  let modelUploads = $state<StagedFile[]>([]);
  let inputSources = $state<Record<number, SourceMode>>({});
  let inputUploads = $state<Record<number, StagedFile[]>>({});

  /** Recognized model file extensions in priority order. */
  const MODEL_PRIMARY_EXT = ['.xml', '.onnx', '.pb', '.tflite', '.pt', '.pth'];

  /** Find the file in `files` that should drive `model_path`. */
  function pickPrimaryModel(files: StagedFile[]): StagedFile | null {
    if (files.length === 0) return null;
    for (const ext of MODEL_PRIMARY_EXT) {
      const hit = files.find(f => f.original_filename.toLowerCase().endsWith(ext));
      if (hit) return hit;
    }
    return files[files.length - 1];
  }

  // Clone mode tracking
  let isCloneMode = $derived(!!cloneSourceId);
  let originalMainDevice = $state('');
  let originalRefDevice = $state('');
  let originalOvPath = $state('');
  let changedFields = $derived.by(() => {
    if (!isCloneMode) return new Set<string>();
    const changed = new Set<string>();
    if (mainDevice !== originalMainDevice) changed.add('main_device');
    if (refDevice !== originalRefDevice) changed.add('ref_device');
    if (ovPath !== originalOvPath) changed.add('ov_path');
    return changed;
  });

  // Session name placeholder derived from model path
  let modelNameFromPath = $derived(modelPath ? modelPath.split('/').pop()?.replace(/\.[^.]+$/, '') || '' : '');

  // OV path validation
  let ovValidating = $state(false);
  let ovError = $state<string | null>(null);

  // Model inputs
  let modelInputs = $state<InputConfig[]>([]);
  let loadingInputs = $state(false);
  let inputsError = $state<string | null>(null);
  let inspectedModelPath = $state('');

  // Input file path validation
  let inputFileErrors = $state<Record<number, string | null>>({});

  /** Map bytes-per-element to the precision strings used in this form. */
  const BYTES_TO_PRECISION: Record<number, string[]> = {
    1: ['u8', 'i8'],
    2: ['fp16'],
    4: ['fp32', 'i32'],
    8: ['i64'],
  };

  /**
   * For a .bin file, infer precision from file_size / num_elements.
   * Returns the best matching precision from the input's allowed list, or null.
   */
  function inferPrecisionFromBin(input: typeof modelInputs[0], fileSize: number): string | null {
    const shape = input.resolved_shape ?? input.shape;
    const numElements = shape.reduce<number>((acc, d) => {
      const n = typeof d === 'number' ? d : 0;
      return n > 0 ? acc * n : 0;
    }, 1);
    if (numElements <= 0) return null;
    const bpe = fileSize / numElements;
    if (!Number.isInteger(bpe)) return null;
    const candidates = BYTES_TO_PRECISION[bpe];
    if (!candidates) return null;
    const allowed = getAllowedPrecisions(input.element_type);
    return candidates.find(c => allowed.includes(c)) ?? candidates[0];
  }

  function debounceInputFileCheck(index: number) {
    const path = modelInputs[index]?.path?.trim();
    if (!path) {
      inputFileErrors = { ...inputFileErrors, [index]: null };
      return;
    }
    configStore.checkPath(path).then(result => {
      if (modelInputs[index]?.path?.trim() !== path) return; // stale
      if (!result.exists) {
        inputFileErrors = { ...inputFileErrors, [index]: `File not found: ${path}` };
      } else if (!result.is_file) {
        inputFileErrors = { ...inputFileErrors, [index]: `Not a file: ${path}` };
      } else {
        inputFileErrors = { ...inputFileErrors, [index]: null };
        // Auto-detect precision for .bin files based on file size and shape
        if (result.file_size != null && path.toLowerCase().endsWith('.bin')) {
          const inferred = inferPrecisionFromBin(modelInputs[index], result.file_size);
          if (inferred) {
            modelInputs[index] = { ...modelInputs[index], data_type: inferred };
          }
        }
      }
    });
  }

  // Plugin configuration -- separate state for main and reference devices
  let pluginConfigExpanded = $state(false);
  let activePluginTab = $state<'main' | 'ref'>('main');

  let mainPluginProperties = $state<DeviceProperty[]>([]);
  let mainPluginConfigValues = $state<Record<string, string>>({});
  let loadingMainPluginConfig = $state(false);

  let refPluginProperties = $state<DeviceProperty[]>([]);
  let refPluginConfigValues = $state<Record<string, string>>({});
  let loadingRefPluginConfig = $state(false);

  async function fetchMainPluginConfig(device: string) {
    loadingMainPluginConfig = true;
    mainPluginProperties = await configStore.fetchDeviceConfig(device);
    mainPluginConfigValues = {};
    loadingMainPluginConfig = false;
  }

  async function fetchRefPluginConfig(device: string) {
    loadingRefPluginConfig = true;
    refPluginProperties = await configStore.fetchDeviceConfig(device);
    refPluginConfigValues = {};
    loadingRefPluginConfig = false;
  }

  function onMainDeviceChange() {
    if (pluginConfigExpanded) {
      fetchMainPluginConfig(mainDevice);
    }
  }

  function onRefDeviceChange() {
    if (pluginConfigExpanded) {
      fetchRefPluginConfig(refDevice);
    }
  }

  function getChangedValues(values: Record<string, string>, properties: DeviceProperty[]): Record<string, string> {
    const result: Record<string, string> = {};
    for (const [key, value] of Object.entries(values)) {
      const prop = properties.find(p => p.name === key);
      if (prop && value !== prop.value) {
        result[key] = value;
      }
    }
    return result;
  }

  function getPluginConfigPayload(): Record<string, string> {
    return getChangedValues(mainPluginConfigValues, mainPluginProperties);
  }

  function getRefPluginConfigPayload(): Record<string, string> {
    return getChangedValues(refPluginConfigValues, refPluginProperties);
  }

  let mainChangedCount = $derived(Object.keys(getPluginConfigPayload()).length);
  let refChangedCount = $derived(Object.keys(getRefPluginConfigPayload()).length);

  let mainPluginFilter = $state('');
  let refPluginFilter = $state('');

  // File browser state
  let showBrowser = $state(false);
  let browserMode = $state<'directory' | 'file'>('directory');
  let browserInitialPath = $state('');
  let browserTarget = $state<'ov' | 'model' | 'input'>('ov');
  let browserInputIndex = $state(0);
  let lastBrowsedFolder = $state('');

  /** Extract the parent directory from a file path, or return '' */
  function parentDir(p: string): string {
    // Handle both / and \ separators
    const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
    return i > 0 ? p.substring(0, i) : '';
  }

  function openBrowser(target: 'ov' | 'model', inputIndex?: number) {
    if (target === 'ov') {
      browserTarget = 'ov';
      browserMode = 'directory';
      browserInitialPath = ovPath || lastBrowsedFolder;
    } else {
      browserMode = 'file';
      if (inputIndex !== undefined) {
        browserTarget = 'input';
        browserInputIndex = inputIndex;
        const p = modelInputs[inputIndex]?.path || '';
        browserInitialPath = parentDir(p) || lastBrowsedFolder;
      } else {
        browserTarget = 'model';
        browserInitialPath = parentDir(modelPath) || lastBrowsedFolder;
      }
    }
    showBrowser = true;
  }

  function onBrowserSelect(path: string) {
    showBrowser = false;
    // Remember the folder so the next browser open starts here
    lastBrowsedFolder = browserMode === 'directory' ? path : (parentDir(path) || path);
    if (browserTarget === 'ov') {
      ovPath = path;
      onOvPathInput();
    } else if (browserTarget === 'input') {
      modelInputs[browserInputIndex] = { ...modelInputs[browserInputIndex], path };
      debounceInputFileCheck(browserInputIndex);
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

  function handlePluginWheel(e: WheelEvent, currentValue: string, propName: string, onValueChange: (k: string, v: string) => void) {
    e.preventDefault();
    const cur = parseInt(currentValue) || 0;
    const delta = e.deltaY < 0 ? 1 : -1;
    const step = e.shiftKey ? 10 : 1;
    onValueChange(propName, String(Math.max(0, cur + delta * step)));
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

    const result = await configStore.fetchModelInputs(path, ovPath || undefined);

    if (result.error) {
      inputsError = result.error;
    } else if (result.inputs.length === 0) {
      inputsError = 'Model has no inputs.';
    } else {
      // Reset per-input upload state when re-inspecting a different model.
      inputSources = {};
      inputUploads = {};
      modelInputs = result.inputs.map((info) => {
        const dynDims = hasDynamicDims(info.shape);
        return {
          name: info.name,
          shape: info.shape,
          element_type: info.element_type,
          port_names: info.port_names,
          data_type: elementTypeToDataType(info.element_type),
          source: 'random' as const,
          path: '',
          layout: getDefaultLayout(info.shape),
          // Initialize dynamic shape fields
          // For dynamic dims: batch (dim 0, 4D only) = 1, channels (dim 1, >= 3D) = 3, others = 1024
          // resolved_shape defaults to upper_bounds so random inputs use the max size
          lower_bounds: dynDims ? info.shape.map(d => typeof d === 'string' ? 1 : d as number) : undefined,
          upper_bounds: dynDims ? info.shape.map((d, idx) => {
            if (typeof d !== 'string') return d as number;
            const rank = info.shape.length;
            if (idx === 0 && rank === 4) return 1;
            if (idx === 0 && rank === 3) return 3;
            if (idx === 1 && rank === 4) return 3;
            return 1024;
          }) : undefined,
          resolved_shape: dynDims ? info.shape.map((d, idx) => {
            if (typeof d !== 'string') return d as number;
            const rank = info.shape.length;
            if (idx === 0 && rank === 4) return 1;
            if (idx === 0 && rank === 3) return 3;
            if (idx === 1 && rank === 4) return 3;
            return 1024;
          }) : undefined,
        };
      });
      // Initialize per-input upload state so SourceField bindings always have
      // a defined value to write to.
      const initSources: Record<number, SourceMode> = {};
      const initUploads: Record<number, StagedFile[]> = {};
      for (let i = 0; i < modelInputs.length; i++) {
        initSources[i] = 'server';
        initUploads[i] = [];
      }
      inputSources = initSources;
      inputUploads = initUploads;
      inspectedModelPath = path;
    }
    loadingInputs = false;
  }

  /** Toggle a per-input source between random and file. When switching to file
   *  with no path yet, default the SourceField to upload mode (most users
   *  picking File on a remote browser want to drop a local file). */
  function onInputSourceChange(i: number) {
    if (modelInputs[i].source === 'file' && !modelInputs[i].path) {
      inputSources[i] = 'upload';
    }
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
    // Clone mode: pre-fill from source session config
    if (isCloneMode && cloneSourceConfig) {
      if (cloneSourceConfig.ov_path) ovPath = cloneSourceConfig.ov_path;
      modelPath = cloneSourceConfig.model_path;
      mainDevice = cloneSourceConfig.main_device;
      refDevice = cloneSourceConfig.ref_device;

      // Save originals for change highlighting
      originalMainDevice = mainDevice;
      originalRefDevice = refDevice;
      originalOvPath = ovPath;

      // Validate OV path for device list
      if (cloneSourceConfig.ov_path) {
        await doValidateOvPath(cloneSourceConfig.ov_path);
      } else {
        configStore.devices = ['CPU'];
      }

      // Inspect model to load inputs
      if (modelPath) {
        await doInspectModel(modelPath);
        // Apply source session's input configs if present
        if (cloneSourceConfig.inputs?.length && modelInputs.length > 0) {
          for (let i = 0; i < modelInputs.length && i < cloneSourceConfig.inputs.length; i++) {
            const srcInp = cloneSourceConfig.inputs[i];
            modelInputs[i] = {
              ...modelInputs[i],
              data_type: srcInp.data_type || modelInputs[i].data_type,
              source: srcInp.source || modelInputs[i].source,
              path: srcInp.path || modelInputs[i].path,
              layout: srcInp.layout || modelInputs[i].layout,
            };
          }
        }
      }
      return;
    }

    // Read CLI defaults already fetched by App init.
    const defaults = configStore.defaults;

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

      // Mark this CLI fingerprint as consumed so an in-tab refresh
      // doesn't bounce the user back to this form.
      try {
        const fp = cliFingerprint(defaults);
        if (fp) sessionStorage.setItem(CLI_CONSUMED_KEY, fp);
      } catch { /* ignore */ }
    } else {
      configStore.devices = ['CPU'];
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
        debounceInputFileCheck(i);
      }
    }

    // Apply remaining positional inputs in order to unassigned slots
    let posIdx = 0;
    for (let i = 0; i < modelInputs.length && posIdx < positional.length; i++) {
      if (modelInputs[i].source === 'random') {
        modelInputs[i] = { ...modelInputs[i], source: 'file', path: positional[posIdx++] };
        debounceInputFileCheck(i);
      }
    }
  }

  function onModelUploadComplete() {
    // Re-pick the primary model file from the current upload list.
    // For IR pairs the .xml wins; for ONNX/PB/TFLite/PyTorch the file itself.
    const primary = pickPrimaryModel(modelUploads);
    if (!primary) return;
    if (primary.staged_path !== modelPath) {
      modelPath = primary.staged_path;
      onModelPathInput();
    }
  }

  $effect(() => {
    // Whenever the upload list changes (file added or removed), re-pick the
    // primary so modelPath stays in sync.
    if (modelSource !== 'upload') return;
    const primary = pickPrimaryModel(modelUploads);
    if (primary && primary.staged_path !== modelPath) {
      modelPath = primary.staged_path;
      onModelPathInput();
    } else if (!primary && modelPath) {
      modelPath = '';
      onModelPathInput();
    }
  });

  function handleBack() {
    if (uploadGroupId) {
      // Fire-and-forget; the TTL sweeper catches anything that fails.
      deleteUploadGroup(uploadGroupId);
      uploadGroupId = null;
    }
    onback();
  }

  function onInputUploadComplete(i: number, file: StagedFile) {
    if (!uploadGroupId) uploadGroupId = file.group_id;
    modelInputs[i] = { ...modelInputs[i], path: file.staged_path };
    // Auto-infer precision from .bin file size when possible.
    if (file.original_filename.toLowerCase().endsWith('.bin')) {
      const inferred = inferPrecisionFromBin(modelInputs[i], file.size);
      if (inferred) {
        modelInputs[i] = { ...modelInputs[i], data_type: inferred };
      }
    }
    debounceInputFileCheck(i);
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
      'int32_t': 'i32', 'int64_t': 'i64', 'int8_t': 'i8',
      'u8': 'u8', 'uint8': 'u8', 'uint8_t': 'u8', 'u4': 'u8', 'uint4_t': 'u8',
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

    const pluginCfg = getPluginConfigPayload();
    const refPluginCfg = getRefPluginConfigPayload();

    if (isCloneMode && cloneSourceId) {
      // Clone mode: use clone endpoint with overrides
      const overrides: Record<string, any> = {};
      if (changedFields.has('main_device')) overrides.main_device = mainDevice;
      if (changedFields.has('ref_device')) overrides.ref_device = refDevice;
      if (Object.keys(pluginCfg).length > 0) overrides.plugin_config = pluginCfg;
      if (Object.keys(refPluginCfg).length > 0) overrides.ref_plugin_config = refPluginCfg;
      // Always send inputs so user edits (source, precision, path, etc.) take effect
      if (modelInputs.length > 0) overrides.inputs = modelInputs;

      const result = await sessionStore.cloneSession(cloneSourceId, overrides);
      submitting = false;
      if (result) {
        // Enqueue source session's inferred nodes into the new session
        const nodeNames = result.inferred_nodes.map(n => n.node_name);
        if (nodeNames.length > 0) {
          await sessionStore.cloneEnqueue(cloneSourceId, result.session.id, nodeNames);
        }
        onsessioncreated(result.session.id);
      } else {
        error = sessionStore.error ?? 'Failed to clone session';
      }
    } else {
      const info = await sessionStore.createSession({
        ov_path: ovPath || undefined,
        model_path: modelPath,
        session_name: sessionName.trim() || undefined,
        main_device: mainDevice,
        ref_device: refDevice,
        input_precision: modelInputs.length > 0 ? modelInputs[0].data_type : 'fp32',
        input_layout: modelInputs.length > 0 ? modelInputs[0].layout : 'NCHW',
        inputs: modelInputs.length > 0 ? modelInputs : undefined,
        plugin_config: Object.keys(pluginCfg).length > 0 ? pluginCfg : undefined,
        ref_plugin_config: Object.keys(refPluginCfg).length > 0 ? refPluginCfg : undefined,
      });

      submitting = false;
      if (info) {
        // Session already self-copies model + inputs into its own dir,
        // so the staging group can be released immediately.
        if (uploadGroupId) {
          deleteUploadGroup(uploadGroupId);
          uploadGroupId = null;
        }
        onsessioncreated(info.id);
      } else {
        error = sessionStore.error ?? 'Failed to create session';
      }
    }
  }
</script>

<div class="form-root">
  <div class="form-inner">
    <!-- Header -->
    <div class="form-header">
      <button class="back-btn" onclick={handleBack}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="12" y1="8" x2="4" y2="8" />
          <polyline points="8,4 4,8 8,12" />
        </svg>
        Back
      </button>
      <h2 class="form-title">
        {#if isCloneMode}
          Clone of <span class="title-model">{cloneSourceName || 'session'}</span>
        {:else}
          New Session
        {/if}
      </h2>
    </div>

    <form class="form-body" onsubmit={(e) => { e.preventDefault(); handleSubmit(); }}>
      <!-- Section: Paths -->
      <div class="form-section">
        <div class="section-label">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round">
            <path d="M2 5.5V3a1 1 0 011-1h3l1.5 1.5H11a1 1 0 011 1V11a1 1 0 01-1 1H3a1 1 0 01-1-1V5.5z" />
          </svg>
          Paths
        </div>

        <div class="field-group">
          <label for="ov-path" class="field-label">OpenVINO Path <span class="field-opt">optional</span></label>
          <div class="path-row">
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
              class="browse-btn"
              title="Browse directories"
              onclick={() => openBrowser('ov')}
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round">
                <path d="M2 5.5V3a1 1 0 011-1h3l1.5 1.5H11a1 1 0 011 1V11a1 1 0 01-1 1H3a1 1 0 01-1-1V5.5z" />
              </svg>
            </button>
            <div class="field-status-slot">
              {#if ovValidating}
                <div class="field-status">
                  <svg class="status-spin" width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <circle cx="6" cy="6" r="4.5" stroke="currentColor" stroke-width="1.2" stroke-dasharray="20" stroke-dashoffset="6" stroke-linecap="round" />
                  </svg>
                </div>
              {:else if ovError}
                <div class="field-status field-warn" title={ovError}>
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.3">
                    <path d="M7 1.5L12.5 11.5H1.5L7 1.5z" stroke-linejoin="round" />
                    <line x1="7" y1="6" x2="7" y2="8.5" stroke-linecap="round" />
                    <circle cx="7" cy="10" r="0.5" fill="currentColor" />
                  </svg>
                </div>
              {/if}
            </div>
          </div>
        </div>

        <div class="field-group">
          <label for="model-path" class="field-label">Model <span class="field-req">.xml / .onnx / ...</span></label>
          <SourceField
            kind="model"
            multi={true}
            bind:mode={modelSource}
            bind:serverPath={modelPath}
            bind:uploadedFiles={modelUploads}
            bind:groupId={uploadGroupId}
            placeholder="/path/to/model.xml"
            inputId="model-path"
            onServerInput={onModelPathInput}
            onBrowse={() => openBrowser('model')}
            onUploadComplete={() => onModelUploadComplete()}
          />
          {#if loadingInputs}
            <div class="field-hint">Loading model inputs…</div>
          {:else if inputsError}
            <div class="field-hint field-warn">{inputsError}</div>
          {/if}
        </div>

        <div class="field-group">
          <label for="session-name" class="field-label">Session Name <span class="field-opt">optional</span></label>
          <input
            type="text"
            id="session-name"
            bind:value={sessionName}
            placeholder={modelNameFromPath || 'model name'}
            class="session-name-input"
          />
        </div>
      </div>

      <!-- Section: Model Inputs -->
      {#if modelInputs.length > 0}
        <div class="form-section">
          <div class="section-label">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round">
              <rect x="2" y="2" width="10" height="10" rx="2" />
              <line x1="5" y1="5" x2="9" y2="5" />
              <line x1="5" y1="7" x2="9" y2="7" />
              <line x1="5" y1="9" x2="7" y2="9" />
            </svg>
            Model Inputs
            <span class="section-count">{modelInputs.length}</span>
          </div>

          <div class="inputs-list">
            {#each modelInputs as input, i (input.name)}
              <div class="input-card">
                <div class="input-header">
                  <div class="font-mono text-xs text-accent truncate">
                    {input.name}{#if input.port_names?.length} <span class="text-content-secondary">({input.port_names.join(', ')})</span>{/if}
                  </div>
                  <div class="input-type">
                    {input.element_type} &middot; {formatShape(input.shape)}
                  </div>
                </div>
                <div class="input-controls">
                  <div class="input-ctrl">
                    <select
                      id="source-{i}"
                      bind:value={modelInputs[i].source}
                      onchange={() => onInputSourceChange(i)}
                      title="Source"
                      class="ctrl-select"
                    >
                      <option value="random">Random</option>
                      <option value="file">File</option>
                    </select>
                  </div>
                  <div class="input-ctrl">
                    <select
                      id="dtype-{i}"
                      bind:value={modelInputs[i].data_type}
                      title="Data Type"
                      class="ctrl-select"
                    >
                      {#each getAllowedPrecisions(input.element_type) as p (p)}
                        <option value={p}>{p.toUpperCase()}</option>
                      {/each}
                    </select>
                  </div>
                  <div class="input-ctrl">
                    <select
                      id="layout-{i}"
                      bind:value={modelInputs[i].layout}
                      title="Layout"
                      class="ctrl-select"
                    >
                      {#each getLayoutOptions(input.shape) as l (l)}
                        <option value={l}>{l}</option>
                      {/each}
                    </select>
                  </div>
                </div>
                {#if hasDynamicDims(input.shape)}
                  <div class="dynamic-shape-box">
                    <div class="dynamic-label">
                      <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor"><circle cx="5" cy="5" r="3" /></svg>
                      Dynamic Shape
                    </div>
                    <div class="dynamic-grid" style="grid-template-columns: auto 1fr 1fr 1fr;">
                      <div class="dg-header">Dim</div>
                      {#if input.source === 'random'}
                        <div class="dg-header">Value</div>
                      {:else}
                        <div></div>
                      {/if}
                      <div class="dg-header">Min</div>
                      <div class="dg-header">Max</div>
                      {#each input.shape as dim, d}
                        <div class="dg-dim">[{d}]</div>
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
                              class="dg-input"
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
                            class="dg-input"
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
                            class="dg-input"
                          />
                        {:else}
                          {#if input.source === 'random'}
                            <div class="dg-static">{dim}</div>
                          {:else}
                            <div></div>
                          {/if}
                          <div class="dg-static">{dim}</div>
                          <div class="dg-static">{dim}</div>
                        {/if}
                      {/each}
                    </div>
                  </div>
                {/if}
                {#if input.source === 'file'}
                  <div style="margin-top: 0.25rem;">
                    <SourceField
                      kind="input"
                      bind:mode={inputSources[i]}
                      bind:serverPath={modelInputs[i].path}
                      bind:uploadedFiles={inputUploads[i]}
                      bind:groupId={uploadGroupId}
                      placeholder="/path/to/input.npy"
                      onServerInput={() => debounceInputFileCheck(i)}
                      onBrowse={() => openBrowser('model', i)}
                      onUploadComplete={(f) => onInputUploadComplete(i, f)}
                    />
                  </div>
                  {#if inputFileErrors[i]}
                    <div class="input-file-error">{inputFileErrors[i]}</div>
                  {/if}
                {/if}
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Section: Device Selection -->
      <div class="form-section">
        <div class="section-label">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round">
            <rect x="1" y="4" width="5" height="6" rx="1" />
            <rect x="8" y="4" width="5" height="6" rx="1" />
            <line x1="6" y1="7" x2="8" y2="7" />
          </svg>
          Devices
        </div>

        <div class="device-row">
          <div class="device-col">
            <label for="main-device" class="field-label">
              Main Device
              {#if changedFields.has('main_device')}
                <span class="changed-badge">changed</span>
              {/if}
            </label>
            <select id="main-device" bind:value={mainDevice} onchange={onMainDeviceChange} class="device-select {changedFields.has('main_device') ? 'is-changed' : ''}">
              {#each configStore.devices as device (device)}
                <option value={device}>{device}</option>
              {/each}
            </select>
          </div>
          <div class="device-vs">vs</div>
          <div class="device-col">
            <label for="ref-device" class="field-label">
              Reference Device
              {#if changedFields.has('ref_device')}
                <span class="changed-badge">changed</span>
              {/if}
            </label>
            <select id="ref-device" bind:value={refDevice} onchange={onRefDeviceChange} class="device-select {changedFields.has('ref_device') ? 'is-changed' : ''}">
              {#each configStore.devices as device (device)}
                <option value={device}>{device}</option>
              {/each}
            </select>
          </div>
        </div>
      </div>

      <!-- Plugin Configuration -->
      {#snippet pluginConfigPanel(label: string, properties: DeviceProperty[], configValues: Record<string, string>, loading: boolean, prefix: string, onValueChange: (key: string, val: string) => void, filterText: string, onFilterChange: (v: string) => void)}
        <div class="plugin-panel">
          <div class="plugin-panel-label">{label}</div>
          {#if loading}
            <div class="plugin-loading">Loading...</div>
          {:else if properties.length === 0}
            <div class="plugin-loading">No configurable properties.</div>
          {:else}
            {#if properties.length > 8}
              <div class="plugin-filter-row">
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round">
                  <circle cx="5" cy="5" r="3.5" />
                  <line x1="7.5" y1="7.5" x2="10" y2="10" />
                </svg>
                <input
                  type="text"
                  class="plugin-filter-input"
                  placeholder="Filter properties..."
                  value={filterText}
                  oninput={(e) => onFilterChange((e.target as HTMLInputElement).value)}
                />
                {#if filterText}
                  <button type="button" class="plugin-filter-clear" aria-label="Clear filter" onclick={() => onFilterChange('')}>
                    <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round">
                      <line x1="2" y1="2" x2="8" y2="8" /><line x1="8" y1="2" x2="2" y2="8" />
                    </svg>
                  </button>
                {/if}
              </div>
            {/if}
            {@const filtered = filterText
              ? properties.filter(p => p.name.toLowerCase().includes(filterText.toLowerCase()))
              : properties}
            <div class="plugin-list">
              {#each filtered as prop (prop.name)}
                {@const currentVal = configValues[prop.name] ?? prop.value}
                {@const isModified = configValues[prop.name] != null && configValues[prop.name] !== prop.value}
                <div class="plugin-prop" class:modified={isModified}>
                  <label for="{prefix}-{prop.name}" class="plugin-prop-name" title={prop.name}>
                    {prop.name}
                  </label>
                  <div class="plugin-prop-field" class:has-reset={isModified}>
                    {#if prop.type === 'bool'}
                      {@const isCaps = prop.value === 'YES' || prop.value === 'NO'}
                      {@const isPython = prop.value === 'True' || prop.value === 'False'}
                      {@const trueVal = isCaps ? 'YES' : isPython ? 'True' : 'true'}
                      {@const falseVal = isCaps ? 'NO' : isPython ? 'False' : 'false'}
                      <select
                        id="{prefix}-{prop.name}"
                        value={currentVal}
                        onchange={(e) => onValueChange(prop.name, (e.target as HTMLSelectElement).value)}
                        class="plugin-prop-input"
                      >
                        <option value={trueVal}>{trueVal}</option>
                        <option value={falseVal}>{falseVal}</option>
                      </select>
                    {:else if prop.type === 'enum' && prop.options.length > 0}
                      <select
                        id="{prefix}-{prop.name}"
                        value={currentVal}
                        onchange={(e) => onValueChange(prop.name, (e.target as HTMLSelectElement).value)}
                        class="plugin-prop-input"
                      >
                        {#each prop.options as opt (opt)}
                          <option value={opt}>{opt || '(default)'}</option>
                        {/each}
                      </select>
                    {:else if prop.type === 'int'}
                      <input
                        id="{prefix}-{prop.name}"
                        type="number"
                        value={currentVal}
                        oninput={(e) => onValueChange(prop.name, (e.target as HTMLInputElement).value)}
                        onwheel={(e) => handlePluginWheel(e, currentVal, prop.name, onValueChange)}
                        class="plugin-prop-input"
                      />
                    {:else}
                      <input
                        id="{prefix}-{prop.name}"
                        type="text"
                        value={currentVal}
                        oninput={(e) => onValueChange(prop.name, (e.target as HTMLInputElement).value)}
                        class="plugin-prop-input"
                      />
                    {/if}
                    {#if isModified}
                      <button
                        type="button"
                        class="plugin-prop-reset"
                        title="Reset to default ({prop.value})"
                        onclick={() => onValueChange(prop.name, prop.value)}
                      >
                        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
                          <line x1="2" y1="2" x2="8" y2="8" /><line x1="8" y1="2" x2="2" y2="8" />
                        </svg>
                      </button>
                    {/if}
                  </div>
                </div>
              {/each}
              {#if filterText && filtered.length === 0}
                <div class="plugin-loading">No matching properties.</div>
              {/if}
            </div>
          {/if}
        </div>
      {/snippet}

      <details
        class="plugin-details"
        ontoggle={(e) => {
          const open = (e.currentTarget as HTMLDetailsElement).open;
          pluginConfigExpanded = open;
          if (open) {
            if (mainPluginProperties.length === 0) fetchMainPluginConfig(mainDevice);
            if (refPluginProperties.length === 0) fetchRefPluginConfig(refDevice);
          }
        }}
      >
        <summary class="plugin-summary">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round">
            <circle cx="6" cy="6" r="4.5" />
            <circle cx="4.5" cy="5.5" r="0.8" fill="currentColor" />
            <circle cx="7.5" cy="5.5" r="0.8" fill="currentColor" />
            <line x1="4.5" y1="7.5" x2="7.5" y2="7.5" />
          </svg>
          Plugin Configuration
          {#if mainChangedCount + refChangedCount > 0}
            <span class="plugin-changed-count">{mainChangedCount + refChangedCount} changed</span>
          {/if}
        </summary>
        <div class="plugin-body">
          <div class="plugin-hint">
            Only changed values will be sent to the inference engine.
          </div>

          <!-- Tab switcher -->
          <div class="plugin-tabs">
            <button
              type="button"
              class="plugin-tab"
              class:active={activePluginTab === 'main'}
              onclick={() => activePluginTab = 'main'}
            >
              <span class="plugin-tab-role">Main</span>
              <span class="plugin-tab-device">{mainDevice}</span>
              {#if mainChangedCount > 0}
                <span class="plugin-tab-badge">{mainChangedCount}</span>
              {/if}
            </button>
            <button
              type="button"
              class="plugin-tab"
              class:active={activePluginTab === 'ref'}
              onclick={() => activePluginTab = 'ref'}
            >
              <span class="plugin-tab-role">Reference</span>
              <span class="plugin-tab-device">{refDevice}</span>
              {#if refChangedCount > 0}
                <span class="plugin-tab-badge">{refChangedCount}</span>
              {/if}
            </button>
          </div>

          <!-- Tab content -->
          <div class="plugin-tab-content">
            {#if activePluginTab === 'main'}
              {@render pluginConfigPanel(
                `Main: ${mainDevice}`,
                mainPluginProperties,
                mainPluginConfigValues,
                loadingMainPluginConfig,
                'main-plugin',
                (key, val) => { mainPluginConfigValues = { ...mainPluginConfigValues, [key]: val }; },
                mainPluginFilter,
                (v) => { mainPluginFilter = v; }
              )}
            {:else}
              {@render pluginConfigPanel(
                `Reference: ${refDevice}`,
                refPluginProperties,
                refPluginConfigValues,
                loadingRefPluginConfig,
                'ref-plugin',
                (key, val) => { refPluginConfigValues = { ...refPluginConfigValues, [key]: val }; },
                refPluginFilter,
                (v) => { refPluginFilter = v; }
              )}
            {/if}
          </div>
        </div>
      </details>

      {#if error}
        <div class="form-error">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.3">
            <circle cx="7" cy="7" r="5.5" />
            <line x1="7" y1="4.5" x2="7" y2="7.5" stroke-linecap="round" />
            <circle cx="7" cy="9.5" r="0.5" fill="currentColor" />
          </svg>
          {error}
        </div>
      {/if}

      <button
        type="submit"
        disabled={submitting}
        class="submit-btn"
        class:is-submitting={submitting}
      >
        {#if submitting}
          <svg class="submit-spin" width="14" height="14" viewBox="0 0 14 14" fill="none">
            <circle cx="7" cy="7" r="5" stroke="currentColor" stroke-width="1.5" stroke-dasharray="24" stroke-dashoffset="6" stroke-linecap="round" />
          </svg>
        {/if}
        {submitting ? (isCloneMode ? 'Cloning...' : 'Creating...') : (isCloneMode ? 'Clone & Run' : 'Start Session')}
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

<style>
  /* ── Root ── */
  .form-root {
    flex: 1;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 1.5rem;
    padding-top: 2rem;
    background: var(--bg-primary);
    background-image:
      radial-gradient(ellipse 60% 40% at 50% 0%, rgba(76, 141, 255, 0.03) 0%, transparent 100%);
    overflow-y: auto;
  }

  .form-inner {
    max-width: 40rem;
    width: 100%;
    animation: fade-down 0.4s ease-out both;
  }

  /* ── Header ── */
  .form-header {
    margin-bottom: 1.5rem;
  }

  .back-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.6rem 0.3rem 0.4rem;
    border: none;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 0.5rem;
    color: var(--text-secondary);
    font-size: 0.8rem;
    cursor: pointer;
    transition: all 0.15s ease;
    margin-bottom: 0.75rem;
  }

  .back-btn:hover {
    background: rgba(255, 255, 255, 0.08);
    color: var(--text-primary);
  }

  .form-title {
    font-family: var(--font-display);
    font-size: 1.75rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0;
    line-height: 1.2;
  }

  .title-model {
    color: #4C8DFF;
  }

  /* ── Form Body ── */
  .form-body {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  /* ── Sections ── */
  .form-section {
    background: var(--bg-panel);
    border-radius: 0.75rem;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .section-label {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-secondary);
    opacity: 0.6;
  }

  .section-count {
    background: rgba(76, 141, 255, 0.12);
    color: rgba(76, 141, 255, 0.8);
    padding: 0.05rem 0.4rem;
    border-radius: 99px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0;
    text-transform: none;
  }

  /* ── Field Groups ── */
  .field-group {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }

  .field-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-secondary);
  }

  .field-opt {
    font-weight: 400;
    opacity: 0.5;
    font-size: 0.7rem;
    margin-left: 0.25rem;
  }

  .field-req {
    font-family: var(--font-mono);
    font-weight: 400;
    opacity: 0.5;
    font-size: 0.65rem;
    margin-left: 0.25rem;
  }

  .session-name-input {
    width: 100%;
    padding: 0.4rem 0.5rem;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 0.3rem;
    font-size: 0.8rem;
    color: var(--text-primary);
    transition: border-color 0.15s ease;
  }

  .session-name-input:focus {
    border-color: #4C8DFF;
    outline: none;
  }

  .session-name-input::placeholder {
    color: var(--text-secondary);
    opacity: 0.5;
  }

  .path-row {
    display: flex;
    gap: 0.35rem;
    align-items: center;
  }

  .browse-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.5rem;
    background: var(--bg-input);
    border: 1px solid var(--border-color);
    border-radius: 0.375rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
    flex-shrink: 0;
  }

  .browse-btn:hover {
    background: var(--bg-menu);
    color: var(--text-primary);
    border-color: rgba(76, 141, 255, 0.25);
  }

  .browse-btn.small {
    padding: 0.375rem;
  }

  .field-status-slot {
    width: 20px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .field-status {
    color: var(--text-secondary);
    display: flex;
    align-items: center;
  }

  .field-warn {
    color: #E8A849;
    cursor: help;
    background: rgba(232, 168, 73, 0.12);
    border-radius: 50%;
    padding: 3px;
    animation: field-warn-pulse 2s ease-in-out infinite;
  }

  @keyframes field-warn-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(232, 168, 73, 0.3); }
    50% { box-shadow: 0 0 0 4px rgba(232, 168, 73, 0); }
  }

  .status-spin {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* ── Model Inputs ── */
  .inputs-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-height: 45vh;
    overflow-y: auto;
    padding-right: 2px;
  }

  .inputs-list::-webkit-scrollbar { width: 4px; }
  .inputs-list::-webkit-scrollbar-track { background: transparent; }
  .inputs-list::-webkit-scrollbar-thumb { background: #3A3F56; border-radius: 99px; }

  .input-card {
    background: var(--bg-input);
    border-radius: 0.5rem;
    padding: 0.625rem 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    border: 1px solid transparent;
    transition: border-color 0.15s ease;
  }

  .input-card:hover {
    border-color: rgba(76, 141, 255, 0.1);
  }

  .input-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
  }

  .input-type {
    font-size: 0.65rem;
    color: var(--text-secondary);
    opacity: 0.6;
    white-space: nowrap;
    font-family: var(--font-mono);
  }

  .input-controls {
    display: flex;
    gap: 0.35rem;
    align-items: center;
  }

  .input-ctrl {
    flex: 1;
  }

  .ctrl-select {
    width: 100%;
    padding: 0.3rem 0.4rem;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 0.3rem;
    font-size: 0.7rem;
    color: var(--text-primary);
    transition: border-color 0.15s ease;
  }

  .ctrl-select:focus {
    border-color: #4C8DFF;
    outline: none;
  }

  /* ── Dynamic Shape ── */
  .dynamic-shape-box {
    border: 1px solid rgba(232, 168, 73, 0.25);
    border-radius: 0.375rem;
    padding: 0.5rem;
    background: rgba(232, 168, 73, 0.03);
  }

  .dynamic-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: #E8A849;
    display: flex;
    align-items: center;
    gap: 0.3rem;
    margin-bottom: 0.35rem;
  }

  .dynamic-grid {
    display: grid;
    gap: 0.25rem;
  }

  .dg-header {
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--text-secondary);
    padding: 0 0.25rem;
  }

  .dg-dim {
    font-size: 0.7rem;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    padding: 0.375rem 0.25rem;
  }

  .dg-input {
    width: 100%;
    padding: 0.3rem 0.4rem;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 0.3rem;
    font-size: 0.8rem;
    color: var(--text-primary);
    font-family: var(--font-mono);
  }

  .dg-input:focus {
    border-color: #4C8DFF;
    outline: none;
  }

  .dg-static {
    font-size: 0.7rem;
    color: var(--text-secondary);
    font-family: var(--font-mono);
    padding: 0.375rem 0.4rem;
    opacity: 0.6;
  }

  :global(.file-input) :global(input) {
    padding: 0.375rem 0.5rem;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 0.375rem;
    font-size: 0.8rem;
    color: var(--text-primary);
    font-family: var(--font-mono);
    width: 100%;
  }

  :global(.file-input) :global(input:focus) {
    border-color: #4C8DFF;
    outline: none;
  }

  :global(.file-input.input-error) :global(input) {
    border-color: rgba(232, 168, 73, 0.5);
  }

  :global(.file-input) :global([aria-hidden]) {
    padding: 0.375rem 0.5rem;
    font-size: 0.8rem;
    font-family: var(--font-mono);
  }

  .input-file-error {
    font-size: 0.7rem;
    color: #E8A849;
    margin-top: 0.2rem;
    padding-left: 0.1rem;
  }

  /* ── Device Selection ── */
  .device-row {
    display: flex;
    align-items: flex-end;
    gap: 0;
  }

  .device-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
  }

  .device-vs {
    flex-shrink: 0;
    padding: 0 0.75rem;
    padding-bottom: 0.4rem;
    font-size: 0.7rem;
    font-style: italic;
    color: var(--text-secondary);
    opacity: 0.35;
  }

  .device-select {
    width: 100%;
    padding: 0.5rem 0.6rem;
    background: var(--bg-input);
    border: 1px solid var(--border-color);
    border-radius: 0.5rem;
    font-size: 0.85rem;
    color: var(--text-primary);
    transition: border-color 0.15s ease;
  }

  .device-select:focus {
    border-color: #4C8DFF;
    outline: none;
  }

  .device-select.is-changed {
    border-color: rgba(232, 168, 73, 0.5);
  }

  .changed-badge {
    font-size: 0.6rem;
    font-weight: 500;
    color: #E8A849;
    margin-left: 0.35rem;
    letter-spacing: 0.02em;
  }

  /* ── Plugin Configuration ── */
  .plugin-details {
    border: 1px solid var(--border-color);
    border-radius: 0.75rem;
    overflow: hidden;
    transition: border-color 0.15s ease;
  }

  .plugin-details[open] {
    border-color: rgba(76, 141, 255, 0.15);
  }

  .plugin-summary {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.65rem 0.85rem;
    font-size: 0.75rem;
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
    transition: color 0.15s ease;
    list-style: none;
  }

  .plugin-summary::-webkit-details-marker { display: none; }

  .plugin-summary:hover {
    color: var(--text-primary);
  }

  .plugin-changed-count {
    margin-left: auto;
    font-size: 0.65rem;
    color: #4C8DFF;
    font-weight: 500;
  }

  .plugin-body {
    padding: 0 0.85rem 0.85rem;
  }

  .plugin-hint {
    font-size: 0.65rem;
    color: var(--text-secondary);
    opacity: 0.5;
    margin-bottom: 0.75rem;
  }

  /* ── Plugin Tabs ── */
  .plugin-tabs {
    display: flex;
    gap: 0.25rem;
    margin-bottom: 0.75rem;
    background: var(--bg-primary);
    border-radius: 0.5rem;
    padding: 0.2rem;
  }

  .plugin-tab {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.35rem;
    padding: 0.5rem 0.6rem;
    border: 1px solid transparent;
    border-radius: 0.375rem;
    background: transparent;
    color: var(--text-secondary);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
  }

  .plugin-tab:hover:not(.active) {
    color: var(--text-primary);
    background: rgba(255, 255, 255, 0.03);
  }

  .plugin-tab.active {
    background: var(--bg-panel);
    color: var(--text-primary);
    border-color: var(--border-color);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
  }

  .plugin-tab-role {
    font-weight: 500;
    opacity: 0.6;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .plugin-tab.active .plugin-tab-role {
    opacity: 0.8;
  }

  .plugin-tab-device {
    font-weight: 600;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    letter-spacing: -0.01em;
  }

  .plugin-tab.active .plugin-tab-device {
    color: #4C8DFF;
  }

  .plugin-tab-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.1rem;
    height: 1.1rem;
    padding: 0 0.3rem;
    border-radius: 99px;
    background: rgba(76, 141, 255, 0.15);
    color: #4C8DFF;
    font-size: 0.6rem;
    font-weight: 600;
    line-height: 1;
  }

  .plugin-tab-content {
    animation: plugin-fade-in 0.15s ease-out;
  }

  @keyframes plugin-fade-in {
    from { opacity: 0; transform: translateY(2px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .plugin-panel {
    min-width: 0;
  }

  .plugin-panel-label {
    display: none;
  }

  .plugin-loading {
    font-size: 0.75rem;
    color: var(--text-secondary);
    opacity: 0.5;
    padding: 0.75rem 0;
  }

  .plugin-list {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    max-height: 18rem;
    overflow-y: auto;
  }

  .plugin-list::-webkit-scrollbar { width: 4px; }
  .plugin-list::-webkit-scrollbar-track { background: transparent; }
  .plugin-list::-webkit-scrollbar-thumb { background: #3A3F56; border-radius: 99px; }

  .plugin-prop {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.25rem 0.4rem;
    border-radius: 0.3rem;
    transition: background 0.1s ease;
  }

  .plugin-prop:hover {
    background: rgba(255, 255, 255, 0.02);
  }

  .plugin-prop-name {
    flex: 1;
    font-size: 0.75rem;
    font-family: var(--font-mono);
    color: var(--text-secondary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .plugin-prop-input {
    width: 100%;
    padding: 0.3rem 0.5rem;
    background: var(--bg-input);
    border: 1px solid var(--border-color);
    border-radius: 0.3rem;
    font-size: 0.75rem;
    color: var(--text-primary);
    transition: border-color 0.15s ease;
  }

  .plugin-prop-input:focus {
    border-color: #4C8DFF;
    outline: none;
  }

  .plugin-prop.modified {
    background: rgba(76, 141, 255, 0.04);
    border-left: 2px solid rgba(76, 141, 255, 0.4);
    padding-left: calc(0.4rem - 2px);
  }

  .plugin-prop-field {
    position: relative;
    flex-shrink: 0;
    width: 10rem;
  }

  .plugin-prop-field .plugin-prop-input {
    width: 100%;
  }

  .plugin-prop-field.has-reset .plugin-prop-input {
    padding-right: 1.5rem;
  }

  .plugin-prop-reset {
    position: absolute;
    right: 0.25rem;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    align-items: center;
    justify-content: center;
    width: 1rem;
    height: 1rem;
    padding: 0;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    opacity: 0.35;
    cursor: pointer;
    border-radius: 0.2rem;
    transition: all 0.15s ease;
  }

  .plugin-prop-reset:hover {
    opacity: 1;
    color: #4C8DFF;
    background: rgba(76, 141, 255, 0.12);
  }

  .plugin-filter-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.5rem;
    padding: 0.35rem 0.5rem;
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 0.375rem;
    color: var(--text-secondary);
    opacity: 0.7;
    transition: opacity 0.15s ease, border-color 0.15s ease;
  }

  .plugin-filter-row:focus-within {
    opacity: 1;
    border-color: rgba(76, 141, 255, 0.3);
  }

  .plugin-filter-input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    font-size: 0.7rem;
    font-family: var(--font-mono);
    color: var(--text-primary);
  }

  .plugin-filter-input::placeholder {
    color: var(--text-secondary);
    opacity: 0.5;
  }

  .plugin-filter-clear {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.15rem;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    border-radius: 0.2rem;
    opacity: 0.5;
    transition: opacity 0.1s ease;
  }

  .plugin-filter-clear:hover {
    opacity: 1;
  }

  /* ── Error ── */
  .form-error {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.85rem 1rem;
    background: rgba(229, 77, 77, 0.12);
    border: 1px solid rgba(229, 77, 77, 0.35);
    border-left: 3px solid #E54D4D;
    border-radius: 0.75rem;
    color: #F0A0A0;
    font-size: 0.8rem;
    line-height: 1.45;
    animation: error-slide-in 0.35s ease-out both;
    box-shadow: 0 2px 12px rgba(229, 77, 77, 0.1), inset 0 0 20px rgba(229, 77, 77, 0.03);
  }

  .form-error svg {
    flex-shrink: 0;
    margin-top: 1px;
    color: #E54D4D;
    animation: error-icon-pulse 2s ease-in-out infinite;
  }

  @keyframes error-slide-in {
    from {
      opacity: 0;
      transform: translateY(6px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes error-icon-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  /* ── Submit ── */
  .submit-btn {
    width: 100%;
    padding: 0.75rem 1rem;
    background: #4C8DFF;
    border: none;
    border-radius: 0.75rem;
    color: white;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.18s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 0.25rem;
  }

  .submit-btn:hover:not(:disabled) {
    background: #6BA1FF;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(76, 141, 255, 0.25);
  }

  .submit-btn:active:not(:disabled) {
    transform: scale(0.99);
  }

  .submit-btn:disabled {
    background: var(--bg-panel);
    color: var(--text-secondary);
    cursor: not-allowed;
    opacity: 0.6;
  }

  .submit-spin {
    animation: spin 1s linear infinite;
  }
</style>
