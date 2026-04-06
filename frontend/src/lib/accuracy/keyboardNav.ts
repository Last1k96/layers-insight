/**
 * Svelte action for keyboard shortcuts in visualization components.
 * Attach to a focusable root element (tabindex="0").
 * Components opt-in to whichever shortcuts are relevant.
 */

export interface KeyboardNavConfig {
	onResetZoom?: () => void;
	onNextChannel?: () => void;
	onPrevChannel?: () => void;
	onNextBatch?: () => void;
	onPrevBatch?: () => void;
	onTogglePlay?: () => void;
}

export function keyboardNav(node: HTMLElement, config: KeyboardNavConfig) {
	function handler(e: KeyboardEvent) {
		// Ignore if typing in an input/select
		const tag = (e.target as HTMLElement).tagName;
		if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

		switch (e.key) {
			case 'r':
			case '0':
				if (config.onResetZoom) { e.preventDefault(); config.onResetZoom(); }
				break;
			case ']':
				if (config.onNextChannel) { e.preventDefault(); config.onNextChannel(); }
				break;
			case '[':
				if (config.onPrevChannel) { e.preventDefault(); config.onPrevChannel(); }
				break;
			case '}':
				if (config.onNextBatch) { e.preventDefault(); config.onNextBatch(); }
				break;
			case '{':
				if (config.onPrevBatch) { e.preventDefault(); config.onPrevBatch(); }
				break;
			case ' ':
				if (config.onTogglePlay) { e.preventDefault(); config.onTogglePlay(); }
				break;
		}
	}

	node.addEventListener('keydown', handler);
	return {
		update(newConfig: KeyboardNavConfig) {
			config = newConfig;
		},
		destroy() {
			node.removeEventListener('keydown', handler);
		},
	};
}
