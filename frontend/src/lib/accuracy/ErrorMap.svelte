<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		valueToImageData,
		normalizedValueToImageData,
		formatValue,
		drawColorbar,
		computeStats,
		computeHistogram,
		colormapRGB,
		ALL_COLORMAP_OPTIONS,
		ALL_NORM_MODE_OPTIONS,
		type ColormapName,
		type NormMode,
		type NormOptions,
	} from './tensorUtils';
	import { rangeScroll } from './rangeScroll';
	import { keyboardNav } from './keyboardNav';

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

	type ErrorMode = 'absolute' | 'relative' | 'log' | 'signed' | 'threshold';
	let mode = $state<ErrorMode>('absolute');

	// Shared state
	let canvas: HTMLCanvasElement;
	let batch = $state(0);
	let channel = $state(0);
	let colormap: ColormapName = $state('blueGreenRed');
	let normMode = $state<NormMode>('linear');
	let globalNorm = $state(false);
	let normOpts = $derived<NormOptions>({ mode: normMode });

	// Hover
	let hoverX = $state(-1);
	let hoverY = $state(-1);
	let showTooltip = $state(false);
	let tooltipScreenX = $state(0);
	let tooltipScreenY = $state(0);

	// Zoom/pan
	let zoom = $state(1);
	let panX = $state(0);
	let panY = $state(0);
	let dragging = $state(false);
	let dragStartX = 0;
	let dragStartY = 0;
	let panStartX = 0;
	let panStartY = 0;

	// Mode-specific state
	let relEpsilon = $state(1e-7);
	let relUseLog = $state(true);
	let logEpsilon = $state(1e-10);
	let symmetric = $state(true);
	let showTolerance = $state(false);
	let toleranceExp = $state(-4);
	let tolerance = $derived(Math.pow(10, toleranceExp));
	let warningExp = $state(-4);
	let criticalExp = $state(-2);
	let showRef = $state(true);
	let refColormap: ColormapName = $state('gray');
	let warningThreshold = $derived(Math.pow(10, warningExp));
	let criticalThreshold = $derived(Math.pow(10, criticalExp));

	// Histogram canvas for relative mode
	let histCanvas: HTMLCanvasElement;

	let dims = $derived(getSpatialDims(shape));

	// Switch default colormap per mode
	const MODE_COLORMAPS: Record<ErrorMode, ColormapName> = {
		absolute: 'blueGreenRed',
		relative: 'inferno',
		log: 'magma',
		signed: 'RdBu',
		threshold: 'gray',
	};
	$effect(() => { colormap = MODE_COLORMAPS[mode]; });

	// Computed data
	let absDiff = $derived.by(() => {
		const out = new Float32Array(main.length);
		for (let i = 0; i < main.length; i++) out[i] = Math.abs(main[i] - ref[i]);
		return out;
	});

	let relErrorRaw = $derived.by((): Float32Array | null => {
		if (mode !== 'relative') return null;
		const out = new Float32Array(main.length);
		for (let i = 0; i < main.length; i++) out[i] = Math.abs(main[i] - ref[i]) / (Math.abs(ref[i]) + relEpsilon);
		return out;
	});

	let displayData = $derived.by((): Float32Array => {
		switch (mode) {
			case 'absolute': return absDiff;
			case 'relative': {
				const raw = relErrorRaw!;
				if (!relUseLog) return raw;
				const out = new Float32Array(raw.length);
				for (let i = 0; i < raw.length; i++) out[i] = Math.log10(raw[i] + 1e-10);
				return out;
			}
			case 'log': {
				const out = new Float32Array(main.length);
				for (let i = 0; i < main.length; i++) out[i] = Math.log10(Math.abs(main[i] - ref[i]) + logEpsilon);
				return out;
			}
			case 'signed': {
				const out = new Float32Array(main.length);
				for (let i = 0; i < main.length; i++) out[i] = main[i] - ref[i];
				return out;
			}
			case 'threshold': return absDiff;
		}
	});

	// Global range
	let globalRange = $derived.by((): [number, number] | undefined => {
		if (mode === 'threshold') return undefined;
		if (mode === 'signed') {
			if (!globalNorm && !symmetric) return undefined;
			const stats = computeStats(displayData);
			if (symmetric) {
				const maxAbs = Math.max(Math.abs(stats.min), Math.abs(stats.max)) || 1;
				return [-maxAbs, maxAbs];
			}
			return [stats.min, stats.max];
		}
		if (!globalNorm) return undefined;
		return (() => { const s = computeStats(displayData); return [s.min, s.max] as [number, number]; })();
	});

	// Signed range (per-slice or global)
	let signedRange = $derived.by((): [number, number] => {
		if (globalRange) return globalRange;
		const stats = computeStats(sliceData.data);
		if (symmetric) {
			const maxAbs = Math.max(Math.abs(stats.min), Math.abs(stats.max)) || 1;
			return [-maxAbs, maxAbs];
		}
		return [stats.min, stats.max];
	});

	// Slices
	let sliceData = $derived(extractSlice(displayData, shape, batch, channel));
	let mainSlice = $derived(extractSlice(main, shape, batch, channel));
	let refSlice = $derived(extractSlice(ref, shape, batch, channel));
	let diffSlice = $derived(extractSlice(absDiff, shape, batch, channel));
	let relSlice = $derived.by(() => relErrorRaw ? extractSlice(relErrorRaw, shape, batch, channel) : null);

	// Stats
	let sliceStats = $derived(computeStats(sliceData.data));
	let diffStats = $derived(computeStats(diffSlice.data));

	let signCounts = $derived.by(() => {
		if (mode !== 'signed') return { pos: 0, neg: 0, zero: 0 };
		let pos = 0, neg = 0, zero = 0;
		for (let i = 0; i < sliceData.data.length; i++) {
			if (sliceData.data[i] > 0) pos++;
			else if (sliceData.data[i] < 0) neg++;
			else zero++;
		}
		return { pos, neg, zero };
	});

	let thresholdCounts = $derived.by(() => {
		if (mode !== 'threshold') return { warning: 0, critical: 0 };
		let warning = 0, critical = 0;
		for (let i = 0; i < diffSlice.data.length; i++) {
			if (diffSlice.data[i] > criticalThreshold) critical++;
			else if (diffSlice.data[i] > warningThreshold) warning++;
		}
		return { warning, critical };
	});
	let warningPct = $derived((thresholdCounts.warning / (diffSlice.data.length || 1) * 100));
	let criticalPct = $derived((thresholdCounts.critical / (diffSlice.data.length || 1) * 100));

	let relHist = $derived.by(() => {
		if (mode !== 'relative' || !relSlice) return null;
		return computeHistogram(relSlice.data, 64);
	});

	// Image data
	let offscreenImage = $derived.by((): ImageData | null => {
		if (!sliceData) return null;
		if (mode === 'threshold') return null;
		if (mode === 'signed') return valueToImageData(sliceData.data, sliceData.w, sliceData.h, colormap, signedRange);
		return normalizedValueToImageData(sliceData.data, sliceData.w, sliceData.h, colormap, normOpts, globalRange);
	});

	let baseScale = $derived.by(() => {
		if (!sliceData || !canvas) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh || !sliceData.w || !sliceData.h) return 1;
		return Math.min(dw / sliceData.w, dh / sliceData.h);
	});

	function redraw() {
		if (!canvas || !sliceData) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;
		const w = sliceData.w, h = sliceData.h;
		const es = baseScale * zoom;
		const ox = (dw - w * baseScale) / 2 + panX;
		const oy = (dh - h * baseScale) / 2 + panY;

		ctx.clearRect(0, 0, dw, dh);

		if (mode === 'threshold') {
			// Custom composite rendering
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
						pixels[px] = r; pixels[px + 1] = g; pixels[px + 2] = b;
					}
				} else {
					if (isCritical) {
						const intensity = Math.min(1, Math.log10(val / criticalThreshold + 1));
						pixels[px] = Math.round(55 + 200 * intensity); pixels[px + 1] = 30; pixels[px + 2] = 30;
					} else if (isWarning) {
						const intensity = Math.min(1, Math.log10(val / warningThreshold + 1));
						pixels[px] = Math.round(200 * intensity); pixels[px + 1] = Math.round(180 * intensity); pixels[px + 2] = Math.round(30 * intensity);
					} else {
						pixels[px] = 20; pixels[px + 1] = 20; pixels[px + 2] = 20;
					}
				}
				pixels[px + 3] = 255;
			}

			const offscreen = new OffscreenCanvas(w, h);
			offscreen.getContext('2d')!.putImageData(img, 0, 0);
			ctx.setTransform(es, 0, 0, es, ox, oy);
			ctx.imageSmoothingEnabled = false;
			ctx.drawImage(offscreen, 0, 0);
			ctx.resetTransform();
		} else {
			if (!offscreenImage) return;
			const offscreen = new OffscreenCanvas(w, h);
			offscreen.getContext('2d')!.putImageData(offscreenImage, 0, 0);
			ctx.setTransform(es, 0, 0, es, ox, oy);
			ctx.imageSmoothingEnabled = false;
			ctx.drawImage(offscreen, 0, 0);
			ctx.resetTransform();

			// Signed tolerance overlay
			if (mode === 'signed' && showTolerance) {
				ctx.fillStyle = 'rgba(0, 200, 100, 0.3)';
				for (let i = 0; i < sliceData.data.length; i++) {
					if (Math.abs(sliceData.data[i]) <= tolerance) {
						const x = i % w;
						const y = Math.floor(i / w);
						ctx.fillRect(x * es + ox, y * es + oy, es, es);
					}
				}
			}

			// Colorbar
			const range: [number, number] = mode === 'signed'
				? signedRange
				: (globalRange || [sliceStats.min, sliceStats.max]);
			drawColorbar(ctx, 10, dh - 30, Math.min(200, dw - 20), 12, colormap, range[0], range[1]);
		}

		// Crosshair
		if (showTooltip && hoverX >= 0 && hoverY >= 0) {
			const sx = hoverX * es + ox, sy = hoverY * es + oy;
			ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1;
			ctx.beginPath(); ctx.moveTo(sx + 0.5, 0); ctx.lineTo(sx + 0.5, dh); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(0, sy + 0.5); ctx.lineTo(dw, sy + 0.5); ctx.stroke();
		}
	}

	$effect(() => {
		offscreenImage; zoom; panX; panY; showTooltip; hoverX; hoverY;
		mode; showTolerance; tolerance; signedRange;
		refSlice; diffSlice; warningThreshold; criticalThreshold; showRef; refColormap;
		redraw();
	});

	// Histogram for relative mode
	$effect(() => {
		if (!histCanvas || !relHist) return;
		const ctx = histCanvas.getContext('2d');
		if (!ctx) return;
		const w = histCanvas.clientWidth, h = histCanvas.clientHeight;
		histCanvas.width = w; histCanvas.height = h;
		ctx.clearRect(0, 0, w, h);
		const maxCount = Math.max(...relHist.counts);
		if (maxCount === 0) return;
		const barW = w / relHist.counts.length;
		for (let i = 0; i < relHist.counts.length; i++) {
			const barH = (relHist.counts[i] / maxCount) * h;
			const t = i / (relHist.counts.length - 1);
			const [r, g, b] = colormapRGB(t, colormap);
			ctx.fillStyle = `rgb(${r},${g},${b})`;
			ctx.fillRect(i * barW, h - barH, barW, barH);
		}
	});

	// Event handlers
	function screenToData(cx: number, cy: number): [number, number] {
		if (!canvas || !sliceData) return [-1, -1];
		const rect = canvas.getBoundingClientRect();
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const ox = (dw - sliceData.w * baseScale) / 2 + panX;
		const oy = (dh - sliceData.h * baseScale) / 2 + panY;
		const es = baseScale * zoom;
		return [(cx - rect.left - ox) / es, (cy - rect.top - oy) / es];
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
		const rect = canvas.getBoundingClientRect();
		const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const cxOff = (dw - (sliceData?.w ?? 1) * baseScale) / 2;
		const cyOff = (dh - (sliceData?.h ?? 1) * baseScale) / 2;
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
		if (sliceData && ix >= 0 && ix < sliceData.w && iy >= 0 && iy < sliceData.h) {
			hoverX = ix; hoverY = iy; tooltipScreenX = e.clientX; tooltipScreenY = e.clientY; showTooltip = true;
		} else { showTooltip = false; }
	}
	function handleMouseUp() { dragging = false; }
	function handleMouseLeave() { dragging = false; showTooltip = false; }

	$effect(() => { shape; zoom = 1; panX = 0; panY = 0; });
	function resetView() { zoom = 1; panX = 0; panY = 0; }

	const relEpsilonOptions = [
		{ value: 1e-10, label: '1e-10' },
		{ value: 1e-7, label: '1e-7' },
		{ value: 1e-5, label: '1e-5' },
		{ value: 1e-3, label: '1e-3' },
	];
	const logEpsilonOptions = [
		{ value: 1e-15, label: '1e-15' },
		{ value: 1e-10, label: '1e-10' },
		{ value: 1e-7, label: '1e-7' },
		{ value: 1e-5, label: '1e-5' },
	];
</script>

<svelte:window onmouseup={handleMouseUp} />

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div class="flex flex-col gap-3 relative h-full" tabindex="0" use:keyboardNav={{
	onResetZoom: resetView,
	onNextChannel: () => { if (channel < dims.channels - 1) channel++; },
	onPrevChannel: () => { if (channel > 0) channel--; },
	onNextBatch: () => { if (batch < dims.batches - 1) batch++; },
	onPrevBatch: () => { if (batch > 0) batch--; },
}}>
	<!-- Controls -->
	<div class="flex flex-wrap gap-3 items-center text-xs w-full">
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Mode:</span>
			<select bind:value={mode} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				<option value="absolute">Absolute</option>
				<option value="relative">Relative</option>
				<option value="log">Log</option>
				<option value="signed">Signed</option>
				<option value="threshold">Threshold</option>
			</select>
		</label>

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

		{#if mode !== 'threshold'}
			<label class="flex items-center gap-2">
				<span class="text-gray-400">Colormap:</span>
				<select use:rangeScroll bind:value={colormap} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
					{#each ALL_COLORMAP_OPTIONS as opt}
						<option value={opt.value}>{opt.label}</option>
					{/each}
				</select>
			</label>
		{/if}

		{#if mode === 'absolute' || mode === 'relative' || mode === 'log'}
			<label class="flex items-center gap-2">
				<span class="text-gray-400">Norm:</span>
				<select use:rangeScroll bind:value={normMode} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
					{#each ALL_NORM_MODE_OPTIONS as opt}
						<option value={opt.value}>{opt.label}</option>
					{/each}
				</select>
			</label>
			<label class="flex items-center gap-1.5 text-gray-400">
				<input type="checkbox" bind:checked={globalNorm} /> Global norm
			</label>
		{/if}

		{#if mode === 'relative'}
			<label class="flex items-center gap-2">
				<span class="text-gray-400">Epsilon:</span>
				<select use:rangeScroll bind:value={relEpsilon} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
					{#each relEpsilonOptions as opt}
						<option value={opt.value}>{opt.label}</option>
					{/each}
				</select>
			</label>
			<label class="flex items-center gap-1.5 text-gray-400">
				<input type="checkbox" bind:checked={relUseLog} /> Log scale
			</label>
		{/if}

		{#if mode === 'log'}
			<label class="flex items-center gap-2">
				<span class="text-gray-400" title="Minimum value added before log10 to avoid log(0)">Epsilon (floor):</span>
				<select use:rangeScroll bind:value={logEpsilon} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
					{#each logEpsilonOptions as opt}
						<option value={opt.value}>{opt.label}</option>
					{/each}
				</select>
			</label>
		{/if}

		{#if mode === 'signed'}
			<label class="flex items-center gap-1.5 text-gray-400">
				<input type="checkbox" bind:checked={symmetric} /> Symmetric range
			</label>
			<label class="flex items-center gap-1.5 text-gray-400">
				<input type="checkbox" bind:checked={globalNorm} /> Global norm
			</label>
			<label class="flex items-center gap-1.5 text-gray-400">
				<input type="checkbox" bind:checked={showTolerance} /> Tolerance zone
			</label>
			{#if showTolerance}
				<label class="flex items-center gap-2">
					<input use:rangeScroll type="range" min="-8" max="0" step="0.5" bind:value={toleranceExp} />
					<span class="font-mono text-gray-300 text-xs">&plusmn;{tolerance.toExponential(1)}</span>
				</label>
			{/if}
		{/if}

		{#if mode === 'threshold'}
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
		{/if}

		<button class="px-2 py-0.5 text-gray-400 hover:text-gray-200 border border-edge rounded text-xs" onclick={resetView}>Reset view</button>
		<span class="text-gray-500">Shape: [{shape.join(', ')}] | Slice: {dims.height}&times;{dims.width}</span>
	</div>

	<!-- Stats line -->
	{#if mode === 'relative' && relSlice}
		{@const relStats = computeStats(relSlice.data)}
		<div class="flex gap-4 text-xs text-gray-400">
			<span>Rel. error — min: {formatValue(relStats.min)}, max: {formatValue(relStats.max)}, mean: {formatValue(relStats.mean)}</span>
			<span class="text-gray-500">|test-ref|/(|ref|+eps)</span>
		</div>
	{:else if mode === 'log'}
		<div class="flex gap-4 text-xs text-gray-400">
			<span>log10(|diff|+eps) — range: [{formatValue(sliceStats.min)}, {formatValue(sliceStats.max)}] | mean |diff|: {formatValue(diffStats.mean)}</span>
		</div>
	{:else if mode === 'signed'}
		<div class="flex gap-4 text-xs text-gray-400">
			<span>Signed diff (test-ref) — min: {formatValue(sliceStats.min)}, max: {formatValue(sliceStats.max)}, mean: {formatValue(sliceStats.mean)}</span>
			<span class="text-blue-400">+{signCounts.pos}</span>
			<span class="text-red-400">-{signCounts.neg}</span>
			<span class="text-gray-500">={signCounts.zero}</span>
		</div>
	{:else if mode === 'threshold'}
		<div class="flex gap-4 text-xs">
			<span class="text-yellow-400">{thresholdCounts.warning} warning ({warningPct.toFixed(1)}%)</span>
			<span class="text-red-400">{thresholdCounts.critical} critical ({criticalPct.toFixed(1)}%)</span>
			<span class="text-gray-500">{diffSlice.data.length - thresholdCounts.warning - thresholdCounts.critical} within tolerance</span>
		</div>
	{/if}

	<!-- Canvas -->
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

	{#if mode === 'relative'}
		<canvas bind:this={histCanvas} class="w-full h-[60px] bg-surface-base rounded"></canvas>
	{/if}

	<!-- Tooltip -->
	{#if showTooltip && hoverX >= 0 && hoverY >= 0 && sliceData}
		{@const idx = hoverY * sliceData.w + hoverX}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="font-mono text-gray-400">[{hoverY}, {hoverX}]</div>
			<div><span class="text-gray-400">{refLabel}:</span> {formatValue(refSlice.data[idx])}</div>
			<div><span class="text-gray-400">{mainLabel}:</span> {formatValue(mainSlice.data[idx])}</div>
			{#if mode === 'absolute'}
				<div><span class="text-gray-400">Diff:</span> {formatValue(sliceData.data[idx])}</div>
			{:else if mode === 'relative' && relSlice}
				<div><span class="text-gray-400">Rel Error:</span> {formatValue(relSlice.data[idx])}</div>
			{:else if mode === 'log'}
				<div><span class="text-gray-400">Abs Diff:</span> {formatValue(diffSlice.data[idx])}</div>
				<div><span class="text-gray-400">log10:</span> {formatValue(sliceData.data[idx])}</div>
			{:else if mode === 'signed'}
				<div><span class="text-gray-400">Diff:</span> <span class={sliceData.data[idx] > 0 ? 'text-blue-400' : sliceData.data[idx] < 0 ? 'text-red-400' : ''}>{formatValue(sliceData.data[idx])}</span></div>
			{:else if mode === 'threshold'}
				{@const diffVal = diffSlice.data[idx]}
				{@const band = diffVal > criticalThreshold ? 'CRITICAL' : diffVal > warningThreshold ? 'WARNING' : 'OK'}
				<div><span class="text-gray-400">|Diff|:</span> <span class={band === 'CRITICAL' ? 'text-red-400' : band === 'WARNING' ? 'text-yellow-400' : 'text-green-400'}>{formatValue(diffVal)}</span></div>
				<div class={band === 'CRITICAL' ? 'text-red-400 font-bold' : band === 'WARNING' ? 'text-yellow-400 font-bold' : 'text-green-400'}>{band}</div>
			{/if}
		</div>
	{/if}
</div>
