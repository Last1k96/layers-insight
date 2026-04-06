<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		valueToImageData,
		formatValue,
		drawColorbar,
		computeStats,
		ALL_COLORMAP_OPTIONS,
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
	let colormap: ColormapName = $state('turbo');
	let windowSize = $state(7);
	let component: 'ssim' | 'luminance' | 'contrast' | 'structure' = $state('ssim');

	let ssimThreshold = $state(0.95);
	let highlightLow = $state(false);
	let multiScale = $state(false);

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

	// Compute SSIM map
	let ssimResult = $derived.by(() => {
		const rSlice = extractSlice(ref, shape, batch, channel);
		const mSlice = extractSlice(main, shape, batch, channel);
		const w = rSlice.w, h = rSlice.h;
		const half = Math.floor(windowSize / 2);

		// Dynamic range constants
		const rStats = computeStats(rSlice.data);
		const L = rStats.max - rStats.min || 1;
		const C1 = (0.01 * L) ** 2;
		const C2 = (0.03 * L) ** 2;
		const C3 = C2 / 2;

		const outW = Math.max(1, w - windowSize + 1);
		const outH = Math.max(1, h - windowSize + 1);
		const ssimMap = new Float32Array(outW * outH);
		const lumMap = new Float32Array(outW * outH);
		const conMap = new Float32Array(outW * outH);
		const strMap = new Float32Array(outW * outH);

		for (let oy = 0; oy < outH; oy++) {
			for (let ox = 0; ox < outW; ox++) {
				let sumR = 0, sumM = 0, sumR2 = 0, sumM2 = 0, sumRM = 0;
				let n = 0;
				for (let wy = 0; wy < windowSize; wy++) {
					for (let wx = 0; wx < windowSize; wx++) {
						const py = oy + wy, px = ox + wx;
						const rv = rSlice.data[py * w + px];
						const mv = mSlice.data[py * w + px];
						sumR += rv; sumM += mv;
						sumR2 += rv * rv; sumM2 += mv * mv;
						sumRM += rv * mv;
						n++;
					}
				}
				const muR = sumR / n;
				const muM = sumM / n;
				const sigR2 = sumR2 / n - muR * muR;
				const sigM2 = sumM2 / n - muM * muM;
				const sigRM = sumRM / n - muR * muM;

				const sigR = Math.sqrt(Math.max(0, sigR2));
				const sigM = Math.sqrt(Math.max(0, sigM2));

				const luminance = (2 * muR * muM + C1) / (muR * muR + muM * muM + C1);
				const contrast = (2 * sigR * sigM + C2) / (sigR2 + sigM2 + C2);
				const structure = (sigRM + C3) / (sigR * sigM + C3);

				const idx = oy * outW + ox;
				lumMap[idx] = luminance;
				conMap[idx] = contrast;
				strMap[idx] = structure;
				ssimMap[idx] = luminance * contrast * structure;
			}
		}

		// Aggregate SSIM
		let sum = 0;
		for (let i = 0; i < ssimMap.length; i++) sum += ssimMap[i];
		const meanSSIM = sum / (ssimMap.length || 1);

		return { ssimMap, lumMap, conMap, strMap, outW, outH, meanSSIM };
	});

	// Multi-scale SSIM: compute SSIM at multiple window sizes and take weighted geometric mean
	let msSSIM = $derived.by(() => {
		if (!multiScale) return 0;
		const windowSizes = [3, 5, 7, 11];
		const weights = [0.0448, 0.2856, 0.3001, 0.3695];
		const rSlice = extractSlice(ref, shape, batch, channel);
		const mSlice = extractSlice(main, shape, batch, channel);
		const w = rSlice.w, h = rSlice.h;

		const means: number[] = [];
		for (const ws of windowSizes) {
			const half = Math.floor(ws / 2);
			const rStats = computeStats(rSlice.data);
			const L = rStats.max - rStats.min || 1;
			const C1 = (0.01 * L) ** 2;
			const C2 = (0.03 * L) ** 2;

			const outW = Math.max(1, w - ws + 1);
			const outH = Math.max(1, h - ws + 1);
			let sum = 0;
			let count = 0;

			for (let oy = 0; oy < outH; oy++) {
				for (let ox = 0; ox < outW; ox++) {
					let sumR = 0, sumM = 0, sumR2 = 0, sumM2 = 0, sumRM = 0;
					let n = 0;
					for (let wy = 0; wy < ws; wy++) {
						for (let wx = 0; wx < ws; wx++) {
							const py = oy + wy, px = ox + wx;
							const rv = rSlice.data[py * w + px];
							const mv = mSlice.data[py * w + px];
							sumR += rv; sumM += mv;
							sumR2 += rv * rv; sumM2 += mv * mv;
							sumRM += rv * mv;
							n++;
						}
					}
					const muR = sumR / n;
					const muM = sumM / n;
					const sigR2 = sumR2 / n - muR * muR;
					const sigM2 = sumM2 / n - muM * muM;
					const sigRM = sumRM / n - muR * muM;
					const sigR = Math.sqrt(Math.max(0, sigR2));
					const sigM = Math.sqrt(Math.max(0, sigM2));

					const luminance = (2 * muR * muM + C1) / (muR * muR + muM * muM + C1);
					const contrast = (2 * sigR * sigM + C2) / (sigR2 + sigM2 + C2);
					const structure = (sigRM + C2 / 2) / (sigR * sigM + C2 / 2);
					sum += luminance * contrast * structure;
					count++;
				}
			}
			means.push(sum / (count || 1));
		}

		// Weighted geometric mean
		let logSum = 0;
		for (let i = 0; i < means.length; i++) {
			logSum += weights[i] * Math.log(Math.max(1e-10, means[i]));
		}
		return Math.exp(logSum);
	});

	let displayMap = $derived.by(() => {
		switch (component) {
			case 'luminance': return ssimResult.lumMap;
			case 'contrast': return ssimResult.conMap;
			case 'structure': return ssimResult.strMap;
			default: return ssimResult.ssimMap;
		}
	});

	let offscreenImage = $derived.by(() => {
		return valueToImageData(displayMap, ssimResult.outW, ssimResult.outH, colormap, [0, 1]);
	});

	let baseScale = $derived.by(() => {
		if (!canvas || !ssimResult.outW) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh) return 1;
		return Math.min(dw / ssimResult.outW, dh / ssimResult.outH);
	});

	let mapStats = $derived(computeStats(displayMap));

	function resetView() { zoom = 1; panX = 0; panY = 0; }

	function redraw() {
		if (!canvas || !offscreenImage) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const w = ssimResult.outW, h = ssimResult.outH;
		const offscreen = new OffscreenCanvas(w, h);
		offscreen.getContext('2d')!.putImageData(offscreenImage, 0, 0);

		const es = baseScale * zoom;
		const ox = (dw - w * baseScale) / 2 + panX;
		const oy = (dh - h * baseScale) / 2 + panY;

		ctx.clearRect(0, 0, dw, dh);
		ctx.setTransform(es, 0, 0, es, ox, oy);
		ctx.imageSmoothingEnabled = false;
		ctx.drawImage(offscreen, 0, 0);
		ctx.resetTransform();

		// Threshold highlighting: overlay red on pixels below threshold
		if (highlightLow) {
			ctx.fillStyle = 'rgba(255, 60, 60, 0.4)';
			for (let y = 0; y < h; y++) {
				for (let x = 0; x < w; x++) {
					if (ssimResult.ssimMap[y * w + x] < ssimThreshold) {
						const sx = x * es + ox;
						const sy = y * es + oy;
						ctx.fillRect(sx, sy, es, es);
					}
				}
			}
		}

		if (showTooltip && hoverX >= 0 && hoverY >= 0) {
			const sx = hoverX * es + ox, sy = hoverY * es + oy;
			ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1;
			ctx.beginPath(); ctx.moveTo(sx + 0.5, 0); ctx.lineTo(sx + 0.5, dh); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(0, sy + 0.5); ctx.lineTo(dw, sy + 0.5); ctx.stroke();
		}

		drawColorbar(ctx, 10, dh - 30, Math.min(200, dw - 20), 12, colormap, 0, 1);
	}

	$effect(() => { offscreenImage; zoom; panX; panY; showTooltip; hoverX; hoverY; highlightLow; ssimThreshold; redraw(); });

	function screenToData(cx: number, cy: number): [number, number] {
		if (!canvas) return [-1, -1];
		const rect = canvas.getBoundingClientRect();
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const w = ssimResult.outW, h = ssimResult.outH;
		const ox = (dw - w * baseScale) / 2 + panX;
		const oy = (dh - h * baseScale) / 2 + panY;
		const es = baseScale * zoom;
		return [(cx - rect.left - ox) / es, (cy - rect.top - oy) / es];
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
		const rect = canvas.getBoundingClientRect();
		const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const cxOff = (dw - ssimResult.outW * baseScale) / 2;
		const cyOff = (dh - ssimResult.outH * baseScale) / 2;
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
		if (ix >= 0 && ix < ssimResult.outW && iy >= 0 && iy < ssimResult.outH) {
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
			<label class="flex items-center gap-2">
				<span class="text-gray-400 shrink-0">Batch:</span>
				<input use:rangeScroll type="range" min="0" max={dims.batches - 1} bind:value={batch} class="w-20" />
				<span class="text-gray-300 w-6 shrink-0">{batch}</span>
			</label>
		{/if}
		{#if dims.channels > 1}
			<label class="flex items-center gap-2">
				<span class="text-gray-400 shrink-0">Channel:</span>
				<input use:rangeScroll type="range" min="0" max={dims.channels - 1} bind:value={channel} class="w-24" />
				<span class="text-gray-300 w-8 shrink-0">{channel}/{dims.channels}</span>
			</label>
		{/if}
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Component:</span>
			<select use:rangeScroll bind:value={component} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				<option value="ssim">SSIM</option>
				<option value="luminance">Luminance</option>
				<option value="contrast">Contrast</option>
				<option value="structure">Structure</option>
			</select>
		</label>
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Window:</span>
			<select use:rangeScroll bind:value={windowSize} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				{#each [3, 5, 7, 11] as s}
					<option value={s}>{s}x{s}</option>
				{/each}
			</select>
		</label>
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Colormap:</span>
			<select use:rangeScroll bind:value={colormap} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				{#each ALL_COLORMAP_OPTIONS as opt}
					<option value={opt.value}>{opt.label}</option>
				{/each}
			</select>
		</label>
		<label class="flex items-center gap-1.5 text-gray-400">
			<input type="checkbox" bind:checked={highlightLow} /> Highlight low
		</label>
		{#if highlightLow}
			<label class="flex items-center gap-2">
				<input use:rangeScroll type="range" min="0" max="1" step="0.01" bind:value={ssimThreshold} class="w-20" />
				<span class="font-mono text-gray-300 text-xs">&lt;{ssimThreshold.toFixed(2)}</span>
			</label>
		{/if}
		<label class="flex items-center gap-1.5 text-gray-400">
			<input type="checkbox" bind:checked={multiScale} /> MS-SSIM
		</label>
		<button onclick={resetView} class="px-2 py-0.5 rounded bg-surface-base border border-edge text-gray-400 hover:text-gray-200 text-xs">Reset zoom</button>
	</div>

	<div class="flex gap-4 text-xs text-gray-400">
		<span class="font-medium">Mean SSIM: <span class={ssimResult.meanSSIM > 0.99 ? 'text-green-400' : ssimResult.meanSSIM > 0.95 ? 'text-yellow-400' : 'text-red-400'}>{ssimResult.meanSSIM.toFixed(6)}</span></span>
		{#if multiScale}
			<span class="font-medium">MS-SSIM: <span class={msSSIM > 0.99 ? 'text-green-400' : msSSIM > 0.95 ? 'text-yellow-400' : 'text-red-400'}>{msSSIM.toFixed(6)}</span></span>
		{/if}
		<span>Map min: {formatValue(mapStats.min)}, max: {formatValue(mapStats.max)}</span>
		<span class="text-gray-500">{ssimResult.outW}x{ssimResult.outH} (padded by {Math.floor(windowSize / 2)})</span>
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

	{#if showTooltip && hoverX >= 0 && hoverY >= 0}
		{@const idx = hoverY * ssimResult.outW + hoverX}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="font-mono text-gray-400">[{hoverY}, {hoverX}]</div>
			<div><span class="text-gray-400">SSIM:</span> {formatValue(ssimResult.ssimMap[idx])}</div>
			<div><span class="text-gray-400">Lum:</span> {formatValue(ssimResult.lumMap[idx])}</div>
			<div><span class="text-gray-400">Con:</span> {formatValue(ssimResult.conMap[idx])}</div>
			<div><span class="text-gray-400">Str:</span> {formatValue(ssimResult.strMap[idx])}</div>
		</div>
	{/if}
</div>
