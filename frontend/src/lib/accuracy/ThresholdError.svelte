<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		valueToImageData,
		formatValue,
		computeStats,
		type ColormapName,
	} from './tensorUtils';
	import { rangeScroll } from './rangeScroll';

	let {
		main,
		ref,
		shape,
		mainLabel = 'Main',
		refLabel = 'Reference',
	}: {
		main: Float32Array;
		ref: Float32Array;
		shape: number[];
		mainLabel?: string;
		refLabel?: string;
	} = $props();

	let canvas: HTMLCanvasElement;
	let batch = $state(0);
	let channel = $state(0);
	let warningExp = $state(-4);
	let criticalExp = $state(-2);
	let showRef = $state(true);
	let refColormap: ColormapName = $state('gray');
	let warningThreshold = $derived(Math.pow(10, warningExp));
	let criticalThreshold = $derived(Math.pow(10, criticalExp));

	let hoverX = $state(-1);
	let hoverY = $state(-1);
	let showTooltip = $state(false);
	let tooltipScreenX = $state(0);
	let tooltipScreenY = $state(0);

	let zoom = $state(1);
	let panX = $state(0);
	let panY = $state(0);
	let dragging = $state(false);
	let dragStartX = 0;
	let dragStartY = 0;
	let panStartX = 0;
	let panStartY = 0;

	let dims = $derived(getSpatialDims(shape));

	// Abs diff
	let absDiff = $derived.by(() => {
		const out = new Float32Array(main.length);
		for (let i = 0; i < main.length; i++) out[i] = Math.abs(main[i] - ref[i]);
		return out;
	});

	let refSlice = $derived(extractSlice(ref, shape, batch, channel));
	let mainSlice = $derived(extractSlice(main, shape, batch, channel));
	let diffSlice = $derived(extractSlice(absDiff, shape, batch, channel));

	// Count exceeding thresholds
	let thresholdCounts = $derived.by(() => {
		let warning = 0, critical = 0;
		for (let i = 0; i < diffSlice.data.length; i++) {
			if (diffSlice.data[i] > criticalThreshold) critical++;
			else if (diffSlice.data[i] > warningThreshold) warning++;
		}
		return { warning, critical };
	});
	let warningCount = $derived(thresholdCounts.warning);
	let criticalCount = $derived(thresholdCounts.critical);
	let warningPct = $derived((warningCount / (diffSlice.data.length || 1) * 100));
	let criticalPct = $derived((criticalCount / (diffSlice.data.length || 1) * 100));

	let baseScale = $derived.by(() => {
		if (!refSlice || !canvas) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh || !refSlice.w || !refSlice.h) return 1;
		return Math.min(dw / refSlice.w, dh / refSlice.h);
	});

	function redraw() {
		if (!canvas || !refSlice) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const w = refSlice.w, h = refSlice.h;

		// Build composite image: reference in grayscale + red overlay where error > threshold
		const img = new ImageData(w, h);
		const pixels = img.data;

		const refImg = showRef ? valueToImageData(refSlice.data, w, h, refColormap) : null;

		for (let i = 0; i < w * h; i++) {
			const val = diffSlice.data[i];
			const px = i * 4;
			const isCritical = val > criticalThreshold;
			const isWarning = !isCritical && val > warningThreshold;
			if (showRef && refImg) {
				const r = refImg.data[px], g = refImg.data[px + 1], b = refImg.data[px + 2];
				if (isCritical) {
					const intensity = Math.min(1, Math.log10(val / criticalThreshold + 1));
					pixels[px] = Math.round(r * (1 - intensity) + 255 * intensity);
					pixels[px + 1] = Math.round(g * (1 - intensity) + 40 * intensity);
					pixels[px + 2] = Math.round(b * (1 - intensity) + 40 * intensity);
				} else if (isWarning) {
					const intensity = Math.min(1, Math.log10(val / warningThreshold + 1));
					pixels[px] = Math.round(r * (1 - intensity) + 200 * intensity);
					pixels[px + 1] = Math.round(g * (1 - intensity) + 180 * intensity);
					pixels[px + 2] = Math.round(b * (1 - intensity) + 30 * intensity);
				} else {
					pixels[px] = r;
					pixels[px + 1] = g;
					pixels[px + 2] = b;
				}
			} else {
				if (isCritical) {
					const intensity = Math.min(1, Math.log10(val / criticalThreshold + 1));
					pixels[px] = Math.round(55 + 200 * intensity);
					pixels[px + 1] = 30;
					pixels[px + 2] = 30;
				} else if (isWarning) {
					const intensity = Math.min(1, Math.log10(val / warningThreshold + 1));
					pixels[px] = Math.round(200 * intensity);
					pixels[px + 1] = Math.round(180 * intensity);
					pixels[px + 2] = Math.round(30 * intensity);
				} else {
					pixels[px] = 20;
					pixels[px + 1] = 20;
					pixels[px + 2] = 20;
				}
			}
			pixels[px + 3] = 255;
		}

		const offscreen = new OffscreenCanvas(w, h);
		offscreen.getContext('2d')!.putImageData(img, 0, 0);

		const es = baseScale * zoom;
		const ox = (dw - w * baseScale) / 2 + panX;
		const oy = (dh - h * baseScale) / 2 + panY;

		ctx.clearRect(0, 0, dw, dh);
		ctx.setTransform(es, 0, 0, es, ox, oy);
		ctx.imageSmoothingEnabled = false;
		ctx.drawImage(offscreen, 0, 0);
		ctx.resetTransform();

		if (showTooltip && hoverX >= 0 && hoverY >= 0) {
			const sx = hoverX * es + ox, sy = hoverY * es + oy;
			ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1;
			ctx.beginPath(); ctx.moveTo(sx + 0.5, 0); ctx.lineTo(sx + 0.5, dh); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(0, sy + 0.5); ctx.lineTo(dw, sy + 0.5); ctx.stroke();
		}
	}

	$effect(() => { refSlice; diffSlice; warningThreshold; criticalThreshold; showRef; refColormap; zoom; panX; panY; showTooltip; hoverX; hoverY; redraw(); });

	function resetView() { zoom = 1; panX = 0; panY = 0; }

	function screenToData(cx: number, cy: number): [number, number] {
		if (!canvas || !refSlice) return [-1, -1];
		const rect = canvas.getBoundingClientRect();
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const ox = (dw - refSlice.w * baseScale) / 2 + panX;
		const oy = (dh - refSlice.h * baseScale) / 2 + panY;
		const es = baseScale * zoom;
		return [(cx - rect.left - ox) / es, (cy - rect.top - oy) / es];
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
		const rect = canvas.getBoundingClientRect();
		const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const cxOff = (dw - (refSlice?.w ?? 1) * baseScale) / 2;
		const cyOff = (dh - (refSlice?.h ?? 1) * baseScale) / 2;
		panX = cx - (cx - cxOff - panX) * factor - cxOff;
		panY = cy - (cy - cyOff - panY) * factor - cyOff;
		zoom *= factor;
	}

	function handleMouseDown(e: MouseEvent) {
		if (e.button !== 0) return;
		dragging = true; dragStartX = e.clientX; dragStartY = e.clientY;
		panStartX = panX; panStartY = panY;
	}
	function handleMouseMove(e: MouseEvent) {
		if (dragging) { panX = panStartX + (e.clientX - dragStartX); panY = panStartY + (e.clientY - dragStartY); return; }
		const [dx, dy] = screenToData(e.clientX, e.clientY);
		const ix = Math.floor(dx), iy = Math.floor(dy);
		if (refSlice && ix >= 0 && ix < refSlice.w && iy >= 0 && iy < refSlice.h) {
			hoverX = ix; hoverY = iy; tooltipScreenX = e.clientX; tooltipScreenY = e.clientY; showTooltip = true;
		} else { showTooltip = false; }
	}
	function handleMouseUp() { dragging = false; }
	function handleMouseLeave() { dragging = false; showTooltip = false; }

	$effect(() => { shape; zoom = 1; panX = 0; panY = 0; });
</script>

<svelte:window onmouseup={handleMouseUp} />

<div class="flex flex-col gap-4 relative h-full">
	<div class="flex flex-wrap gap-4 items-center text-xs">
		{#if dims.batches > 1}
			<label class="flex items-center gap-2 flex-1 min-w-[10rem]">
				<span class="text-gray-400 shrink-0">Batch:</span>
				<input use:rangeScroll type="range" min="0" max={dims.batches - 1} bind:value={batch} class="flex-1" />
				<span class="text-gray-300 w-6 shrink-0">{batch}</span>
			</label>
		{/if}
		{#if dims.channels > 1}
			<label class="flex items-center gap-2 flex-1 min-w-[10rem]">
				<span class="text-gray-400 shrink-0">Channel:</span>
				<input use:rangeScroll type="range" min="0" max={dims.channels - 1} bind:value={channel} class="flex-1" />
				<span class="text-gray-300 w-8 shrink-0">{channel}/{dims.channels}</span>
			</label>
		{/if}
		<label class="flex items-center gap-2 flex-1 min-w-[14rem]">
			<span class="text-yellow-400 shrink-0">Warning: 10^</span>
			<input use:rangeScroll type="range" min="-8" max="2" step="0.5" bind:value={warningExp} class="flex-1" />
			<span class="text-gray-300 shrink-0 font-mono">{warningThreshold.toExponential(1)}</span>
		</label>
		<label class="flex items-center gap-2 flex-1 min-w-[14rem]">
			<span class="text-red-400 shrink-0">Critical: 10^</span>
			<input use:rangeScroll type="range" min="-8" max="2" step="0.5" bind:value={criticalExp} class="flex-1" />
			<span class="text-gray-300 shrink-0 font-mono">{criticalThreshold.toExponential(1)}</span>
		</label>
		<label class="flex items-center gap-1.5 text-gray-400">
			<input type="checkbox" bind:checked={showRef} /> Show reference
		</label>
		<button class="px-2 py-0.5 text-gray-400 hover:text-gray-200 border border-edge rounded text-xs" onclick={resetView}>Reset view</button>
	</div>

	<div class="flex gap-4 text-xs">
		<span class="text-yellow-400">{warningCount} warning ({warningPct.toFixed(1)}%)</span>
		<span class="text-red-400">{criticalCount} critical ({criticalPct.toFixed(1)}%)</span>
		<span class="text-gray-500">{diffSlice.data.length - warningCount - criticalCount} within tolerance</span>
	</div>

	<div class="flex-1 flex justify-center bg-surface-base rounded-lg p-4 overflow-hidden min-h-0">
		<canvas
			bind:this={canvas}
			class="w-full h-full cursor-crosshair"
			style="image-rendering: pixelated;"
			onwheel={handleWheel}
			onmousedown={handleMouseDown}
			onmousemove={handleMouseMove}
			onmouseleave={handleMouseLeave}
		></canvas>
	</div>

	{#if showTooltip && hoverX >= 0 && hoverY >= 0 && refSlice}
		{@const idx = hoverY * refSlice.w + hoverX}
		{@const diffVal = diffSlice.data[idx]}
		{@const band = diffVal > criticalThreshold ? 'CRITICAL' : diffVal > warningThreshold ? 'WARNING' : 'OK'}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="font-mono text-gray-400">[{hoverY}, {hoverX}]</div>
			<div><span class="text-gray-400">{refLabel}:</span> {formatValue(refSlice.data[idx])}</div>
			<div><span class="text-gray-400">{mainLabel}:</span> {formatValue(mainSlice.data[idx])}</div>
			<div><span class="text-gray-400">|Diff|:</span> <span class={band === 'CRITICAL' ? 'text-red-400' : band === 'WARNING' ? 'text-yellow-400' : 'text-green-400'}>{formatValue(diffVal)}</span></div>
			<div class={band === 'CRITICAL' ? 'text-red-400 font-bold' : band === 'WARNING' ? 'text-yellow-400 font-bold' : 'text-green-400'}>{band}</div>
		</div>
	{/if}
</div>
