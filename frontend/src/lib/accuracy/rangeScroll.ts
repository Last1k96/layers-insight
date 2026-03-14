/** Svelte action: scroll wheel adjusts a range input's value by its step,
 *  or cycles through a select element's options. */
export function rangeScroll(node: HTMLInputElement | HTMLSelectElement) {
	function onWheel(e: WheelEvent) {
		e.preventDefault();
		if (node instanceof HTMLSelectElement) {
			const idx = node.selectedIndex;
			const next = e.deltaY > 0 ? idx + 1 : idx - 1;
			if (next >= 0 && next < node.options.length) {
				node.selectedIndex = next;
				node.dispatchEvent(new Event('change', { bubbles: true }));
			}
			return;
		}
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
	node.addEventListener('wheel', onWheel as EventListener, { passive: false });
	return { destroy: () => node.removeEventListener('wheel', onWheel as EventListener) };
}
