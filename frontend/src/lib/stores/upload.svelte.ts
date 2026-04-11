/**
 * XHR-based uploader for browser-side file staging.
 *
 * Uses XMLHttpRequest rather than fetch because fetch lacks upload progress
 * in most browsers. Each upload returns a StagedFile with the absolute path
 * the backend wrote to disk; that path is then used as a normal model_path
 * or InputConfig.path when the session is created.
 */

export interface StagedFile {
  staged_path: string;
  original_filename: string;
  size: number;
  group_id: string;
  warnings: string[];
}

export interface UploadHandle {
  promise: Promise<StagedFile>;
  cancel: () => void;
}

export type UploadKind = 'model' | 'input';

export function uploadFile(
  file: File,
  opts: {
    groupId?: string | null;
    kind?: UploadKind;
    onProgress?: (loaded: number, total: number) => void;
  } = {},
): UploadHandle {
  const xhr = new XMLHttpRequest();
  const form = new FormData();
  form.append('file', file, file.name);
  if (opts.groupId) form.append('group_id', opts.groupId);
  if (opts.kind) form.append('kind', opts.kind);

  const promise = new Promise<StagedFile>((resolve, reject) => {
    xhr.open('POST', '/api/uploads', true);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && opts.onProgress) {
        opts.onProgress(e.loaded, e.total);
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch (e) {
          reject(new Error(`Bad response: ${e}`));
        }
      } else {
        let detail = xhr.statusText;
        try {
          const body = JSON.parse(xhr.responseText);
          if (body?.detail) detail = body.detail;
        } catch { /* keep statusText */ }
        reject(new Error(detail));
      }
    };

    xhr.onerror = () => reject(new Error('Network error'));
    xhr.onabort = () => reject(new Error('Upload cancelled'));

    xhr.send(form);
  });

  return {
    promise,
    cancel: () => xhr.abort(),
  };
}

export async function deleteUploadGroup(groupId: string): Promise<void> {
  try {
    await fetch(`/api/uploads/${encodeURIComponent(groupId)}`, { method: 'DELETE' });
  } catch {
    // best-effort; the TTL sweeper will catch leaks
  }
}
