/** Svelte action: scroll wheel adjusts a range input's value by its step. */
export function rangeScroll(node: HTMLInputElement) {
	function onWheel(e: WheelEvent) {
		e.preventDefault();
		const step = parseFloat(node.step) || 1;
		const min = parseFloat(node.min);
		const max = parseFloat(node.max);
		const cur = parseFloat(node.value);
		const next = e.deltaY < 0 ? cur + step : cur - step;
		const clamped = Math.max(min, Math.min(max, Math.round(next / step) * step));
		if (clamped !== cur) {
			node.value = String(clamped);
			node.dispatchEvent(new Event('input', { bubbles: true }));
		}
	}
	node.addEventListener('wheel', onWheel, { passive: false });
	return { destroy: () => node.removeEventListener('wheel', onWheel) };
}
