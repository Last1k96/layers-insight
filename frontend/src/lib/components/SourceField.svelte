<script lang="ts">
  import PathInput from './PathInput.svelte';
  import { uploadFile, type StagedFile, type UploadKind } from '../stores/upload.svelte';

  type Mode = 'server' | 'upload';

  let {
    kind,
    multi = false,
    mode = $bindable('server'),
    serverPath = $bindable(''),
    uploadedFiles = $bindable<StagedFile[]>([]),
    groupId = $bindable<string | null>(null),
    onServerInput,
    onBrowse,
    onUploadComplete,
    placeholder = '',
    inputId,
    disabled = false,
  }: {
    kind: UploadKind;
    multi?: boolean;
    mode?: Mode;
    serverPath?: string;
    uploadedFiles?: StagedFile[];
    groupId?: string | null;
    onServerInput?: () => void;
    onBrowse?: () => void;
    onUploadComplete?: (file: StagedFile) => void;
    placeholder?: string;
    inputId?: string;
    disabled?: boolean;
  } = $props();

  const ACCEPT = $derived(
    kind === 'model'
      ? '.xml,.bin,.onnx,.pb,.tflite,.pt,.pth'
      : '.npy,.bin,.raw,.png,.jpg,.jpeg,.bmp'
  );

  const HINT = $derived(
    kind === 'model'
      ? 'Drop a model (.xml + .bin pair, .onnx, .pb, .tflite, .pt) or click to choose'
      : 'Drop an input (.npy / .bin / .raw) or image (.png / .jpg / .bmp) or click to choose'
  );

  let dragOver = $state(false);
  let uploading = $state(false);
  let progress = $state(0); // 0..1
  let error = $state<string | null>(null);
  let cancelHandle = $state<(() => void) | null>(null);
  let fileInput: HTMLInputElement | undefined = $state();

  async function startUpload(file: File): Promise<void> {
    error = null;
    progress = 0;
    uploading = true;
    const handle = uploadFile(file, {
      groupId,
      kind,
      onProgress: (loaded, total) => {
        progress = total > 0 ? loaded / total : 0;
      },
    });
    cancelHandle = handle.cancel;
    try {
      const result = await handle.promise;
      if (!groupId) groupId = result.group_id;
      if (multi) {
        // Replace any existing entry with the same filename, otherwise append.
        const existing = uploadedFiles.findIndex(
          f => f.original_filename === result.original_filename
        );
        if (existing >= 0) {
          const copy = uploadedFiles.slice();
          copy[existing] = result;
          uploadedFiles = copy;
        } else {
          uploadedFiles = [...uploadedFiles, result];
        }
      } else {
        uploadedFiles = [result];
      }
      onUploadComplete?.(result);
    } catch (e: any) {
      error = e?.message ?? String(e);
    } finally {
      uploading = false;
      cancelHandle = null;
      progress = 0;
    }
  }

  async function handleFiles(files: FileList | File[]): Promise<void> {
    const arr = Array.from(files);
    if (arr.length === 0) return;
    // Upload sequentially so the same group_id is reused for the whole batch.
    for (const f of arr) {
      await startUpload(f);
      if (error) break;
    }
  }

  function onDrop(e: DragEvent): void {
    e.preventDefault();
    dragOver = false;
    if (!e.dataTransfer?.files) return;
    handleFiles(e.dataTransfer.files);
  }

  function onDragOver(e: DragEvent): void {
    e.preventDefault();
    dragOver = true;
  }

  function onDragLeave(): void {
    dragOver = false;
  }

  function onPick(e: Event): void {
    const input = e.target as HTMLInputElement;
    if (input.files) handleFiles(input.files);
    input.value = '';
  }

  function cancelCurrent(): void {
    cancelHandle?.();
  }

  function removeStaged(idx: number): void {
    uploadedFiles = uploadedFiles.filter((_, i) => i !== idx);
  }

  function fmtSize(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }

  const allWarnings = $derived(uploadedFiles.flatMap(f => f.warnings ?? []));
</script>

<div class="source-field">
  <div class="source-toggle" role="tablist">
    <button
      type="button"
      class="toggle-btn"
      class:active={mode === 'server'}
      role="tab"
      aria-selected={mode === 'server'}
      onclick={() => (mode = 'server')}
      {disabled}
    >
      Server path
    </button>
    <button
      type="button"
      class="toggle-btn"
      class:active={mode === 'upload'}
      role="tab"
      aria-selected={mode === 'upload'}
      onclick={() => (mode = 'upload')}
      {disabled}
    >
      Upload from PC
    </button>
  </div>

  {#if mode === 'server'}
    <div class="path-row">
      <PathInput
        bind:value={serverPath}
        mode="file"
        {placeholder}
        class="flex-1"
        id={inputId}
        oninput={onServerInput}
      />
      {#if onBrowse}
        <button type="button" class="browse-btn" title="Browse files" onclick={onBrowse}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round">
            <path d="M2 5.5V3a1 1 0 011-1h3l1.5 1.5H11a1 1 0 011 1V11a1 1 0 01-1 1H3a1 1 0 01-1-1V5.5z" />
          </svg>
        </button>
      {/if}
    </div>
  {:else}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="drop-zone"
      class:dragover={dragOver}
      class:has-file={uploadedFiles.length > 0}
      ondrop={onDrop}
      ondragover={onDragOver}
      ondragleave={onDragLeave}
      onclick={() => fileInput?.click()}
      onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput?.click(); } }}
      role="button"
      tabindex="0"
    >
      <input
        type="file"
        accept={ACCEPT}
        bind:this={fileInput}
        onchange={onPick}
        style="display: none"
        multiple={multi}
      />

      {#if uploadedFiles.length > 0}
        <div class="staged-list">
          {#each uploadedFiles as f, idx (f.staged_path)}
            <div class="staged-row">
              <svg width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round">
                <path d="M3 1h6l3 3v9a1 1 0 01-1 1H3a1 1 0 01-1-1V2a1 1 0 011-1z" />
                <path d="M9 1v3h3" />
              </svg>
              <span class="staged-name">{f.original_filename}</span>
              <span class="staged-size">{fmtSize(f.size)}</span>
              <button type="button" class="staged-remove" title="Remove" onclick={(e) => { e.stopPropagation(); removeStaged(idx); }}>
                ×
              </button>
            </div>
          {/each}
        </div>
      {/if}

      {#if uploading}
        <div class="upload-status">
          <div class="upload-bar">
            <div class="upload-bar-fill" style="width: {Math.round(progress * 100)}%"></div>
          </div>
          <div class="upload-meta">
            <span>{Math.round(progress * 100)}%</span>
            <button type="button" class="link-btn" onclick={(e) => { e.stopPropagation(); cancelCurrent(); }}>
              Cancel
            </button>
          </div>
        </div>
      {/if}

      {#if !uploading}
        <div class="drop-hint">
          <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
            <path d="M10 14V3" />
            <path d="M5 8l5-5 5 5" />
            <path d="M3 17h14" />
          </svg>
          <span>
            {#if uploadedFiles.length === 0}
              {HINT}
            {:else if multi}
              Drop another file or click to add
            {:else}
              Drop a different file to replace
            {/if}
          </span>
        </div>
      {/if}

      {#if allWarnings.length > 0}
        <div class="upload-warning">
          {allWarnings.join('; ')}
        </div>
      {/if}

      {#if error}
        <div class="upload-error">
          <span>{error}</span>
          <button type="button" class="link-btn" onclick={(e) => { e.stopPropagation(); error = null; }}>
            Dismiss
          </button>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .source-field {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .source-toggle {
    display: inline-flex;
    border: 1px solid var(--edge);
    border-radius: 6px;
    overflow: hidden;
    align-self: flex-start;
    background: var(--bg-secondary);
  }

  .toggle-btn {
    padding: 0.25rem 0.65rem;
    font-size: 0.72rem;
    color: var(--content-secondary);
    background: transparent;
    border: none;
    cursor: pointer;
    transition: background 0.15s, color 0.15s;
  }

  .toggle-btn + .toggle-btn {
    border-left: 1px solid var(--edge);
  }

  .toggle-btn:hover:not(:disabled) {
    color: var(--content-primary);
  }

  .toggle-btn.active {
    background: var(--accent-bg, rgba(76, 141, 255, 0.18));
    color: var(--accent, #4C8DFF);
  }

  .toggle-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .path-row {
    display: flex;
    gap: 0.35rem;
    align-items: stretch;
  }

  .browse-btn {
    background: var(--bg-secondary);
    border: 1px solid var(--edge);
    color: var(--content-secondary);
    padding: 0 0.5rem;
    border-radius: 4px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .browse-btn:hover {
    color: var(--content-primary);
    border-color: var(--accent);
  }

  .drop-zone {
    border: 1px dashed var(--edge);
    border-radius: 6px;
    padding: 0.75rem;
    background: var(--bg-secondary);
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    min-height: 64px;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    justify-content: center;
  }

  .drop-zone:hover {
    border-color: var(--accent);
  }

  .drop-zone.dragover {
    border-color: var(--accent);
    background: var(--accent-bg, rgba(76, 141, 255, 0.08));
  }

  .drop-zone.has-file {
    border-style: solid;
  }

  .drop-hint {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--content-secondary);
    font-size: 0.78rem;
  }

  .staged-list {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .staged-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(76, 141, 255, 0.06);
    border: 1px solid var(--edge);
    border-radius: 4px;
    padding: 0.3rem 0.5rem;
    color: var(--content-primary);
    font-size: 0.76rem;
  }

  .staged-name {
    font-family: ui-monospace, monospace;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
    min-width: 0;
  }

  .staged-size {
    color: var(--content-secondary);
    font-size: 0.7rem;
  }

  .staged-remove {
    background: transparent;
    border: none;
    color: var(--content-secondary);
    font-size: 1rem;
    line-height: 1;
    cursor: pointer;
    padding: 0 0.25rem;
  }

  .staged-remove:hover {
    color: #E26666;
  }

  .upload-status {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
  }

  .upload-bar {
    height: 4px;
    background: var(--edge);
    border-radius: 2px;
    overflow: hidden;
  }

  .upload-bar-fill {
    height: 100%;
    background: var(--accent, #4C8DFF);
    transition: width 0.1s linear;
  }

  .upload-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.7rem;
    color: var(--content-secondary);
  }

  .upload-warning {
    color: #E8A849;
    font-size: 0.7rem;
  }

  .upload-error {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #E26666;
    font-size: 0.72rem;
    gap: 0.5rem;
  }

  .link-btn {
    background: transparent;
    border: none;
    color: var(--accent, #4C8DFF);
    cursor: pointer;
    font-size: 0.7rem;
    padding: 0;
  }

  .link-btn:hover {
    text-decoration: underline;
  }
</style>
