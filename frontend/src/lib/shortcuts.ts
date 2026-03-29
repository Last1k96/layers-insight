/**
 * Global keyboard shortcuts manager.
 * Registers shortcuts on the document level, skipping when the user is typing in an input/textarea.
 */

export interface Shortcut {
  key: string;
  description: string;
  handler: (e: KeyboardEvent) => void;
  /** If true, fires even when an input/textarea is focused. Default: false. */
  allowInInput?: boolean;
}

const shortcuts: Shortcut[] = [];
let installed = false;
let helpVisible = false;
let helpListeners: Array<(visible: boolean) => void> = [];

function isInputFocused(): boolean {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if ((el as HTMLElement).isContentEditable) return true;
  return false;
}

function handleKeydown(e: KeyboardEvent): void {
  for (const shortcut of shortcuts) {
    if (matchKey(e, shortcut.key)) {
      if (!shortcut.allowInInput && isInputFocused()) continue;
      shortcut.handler(e);
      return;
    }
  }
}

function matchKey(e: KeyboardEvent, pattern: string): boolean {
  // Patterns like "Ctrl+F", "Escape", "?", "A"
  const parts = pattern.split('+').map(p => p.trim().toLowerCase());
  const needCtrl = parts.includes('ctrl');
  const needShift = parts.includes('shift');
  const needAlt = parts.includes('alt');
  const keyPart = parts.filter(p => p !== 'ctrl' && p !== 'shift' && p !== 'alt')[0];

  if (needCtrl !== e.ctrlKey) return false;
  if (needShift !== e.shiftKey) return false;
  if (needAlt !== e.altKey) return false;

  // Match the key itself
  if (keyPart === 'escape') return e.key === 'Escape';
  if (keyPart === '?') return e.key === '?';
  return e.key.toLowerCase() === keyPart;
}

/** Register a shortcut. */
export function registerShortcut(shortcut: Shortcut): void {
  shortcuts.push(shortcut);
}

/** Remove a shortcut by key pattern. */
export function unregisterShortcut(key: string): void {
  const idx = shortcuts.findIndex(s => s.key === key);
  if (idx >= 0) shortcuts.splice(idx, 1);
}

/** Install the global keydown listener. Call once at app init. */
export function installShortcuts(): void {
  if (installed) return;
  installed = true;
  document.addEventListener('keydown', handleKeydown);
}

/** Remove the global keydown listener. */
export function uninstallShortcuts(): void {
  if (!installed) return;
  installed = false;
  document.removeEventListener('keydown', handleKeydown);
  shortcuts.length = 0;
}

/** Get all registered shortcuts (for the help overlay). */
export function getShortcuts(): ReadonlyArray<Shortcut> {
  return shortcuts;
}

/** Toggle the help overlay visibility. */
export function toggleHelp(): void {
  helpVisible = !helpVisible;
  for (const fn of helpListeners) fn(helpVisible);
}

/** Set the help overlay visibility. */
export function setHelpVisible(visible: boolean): void {
  helpVisible = visible;
  for (const fn of helpListeners) fn(helpVisible);
}

/** Subscribe to help visibility changes. Returns unsubscribe function. */
export function onHelpVisibilityChange(fn: (visible: boolean) => void): () => void {
  helpListeners.push(fn);
  return () => {
    helpListeners = helpListeners.filter(l => l !== fn);
  };
}
