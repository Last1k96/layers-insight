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
	let colormap: ColormapName = $state('magma');
	let epsilon = $state(1e-10);
	let globalNorm = $state(false);

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

	// log10(|main - ref| + eps)
	let logError = $derived.by(() => {
		const out = new Float32Array(main.length);
		for (let i = 0; i < main.length; i++) {
			out[i] = Math.log10(Math.abs(main[i] - ref[i]) + epsilon);
		}
		return out;
	});

	let globalRange = $derived.by((): [number, number] | undefined => {
		if (!globalNorm) return undefined;
		const stats = computeStats(logError);
		return [stats.min, stats.max];
	});

	let sliceData = $derived(extractSlice(logError, shape, batch, channel));
	let mainSlice = $derived(extractSlice(main, shape, batch, channel));
	let refSlice = $derived(extractSlice(ref, shape, batch, channel));

	// Abs diff for tooltip
	let absDiff = $derived.by(() => {
		const out = new Float32Array(main.length);
		for (let i = 0; i < main.length; i++) out[i] = Math.abs(main[i] - ref[i]);
		return out;
	});
	let diffSlice = $derived(extractSlice(absDiff, shape, batch, channel));

	let offscreenImage = $derived.by(() => {
		if (!sliceData) return null;
		return valueToImageData(sliceData.data, sliceData.w, sliceData.h, colormap, globalRange);
	});

	let baseScale = $derived.by(() => {
		if (!sliceData || !canvas) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh || !sliceData.w || !sliceData.h) return 1;
		return Math.min(dw / sliceData.w, dh / sliceData.h);
	});

	let sliceStats = $derived(computeStats(sliceData.data));

	function redraw() {
		if (!canvas || !offscreenImage || !sliceData) return;
		const t0 = performance.now();
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const offscreen = new OffscreenCanvas(sliceData.w, sliceData.h);
		offscreen.getContext('2d')!.putImageData(offscreenImage, 0, 0);

		const es = baseScale * zoom;
		const ox = (dw - sliceData.w * baseScale) / 2 + panX;
		const oy = (dh - sliceData.h * baseScale) / 2 + panY;

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

		const range = globalRange || [sliceStats.min, sliceStats.max];
		drawColorbar(ctx, 10, dh - 30, Math.min(200, dw - 20), 12, colormap, range[0], range[1]);
	}

	$effect(() => { offscreenImage; zoom; panX; panY; showTooltip; hoverX; hoverY; redraw(); });

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

	const epsilonOptions = [
		{ value: 1e-15, label: '1e-15' },
		{ value: 1e-10, label: '1e-10' },
		{ value: 1e-7, label: '1e-7' },
		{ value: 1e-5, label: '1e-5' },
	];
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
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Colormap:</span>
			<select use:rangeScroll bind:value={colormap} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				{#each ALL_COLORMAP_OPTIONS as opt}
					<option value={opt.value}>{opt.label}</option>
				{/each}
			</select>
		</label>
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Floor:</span>
			<select use:rangeScroll bind:value={epsilon} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				{#each epsilonOptions as opt}
					<option value={opt.value}>{opt.label}</option>
				{/each}
			</select>
		</label>
		<label class="flex items-center gap-1.5 text-gray-400">
			<input type="checkbox" bind:checked={globalNorm} /> Global norm
		</label>
	</div>

	<div class="flex gap-4 text-xs text-gray-400">
		<span>log10(|diff|) — range: [{formatValue(sliceStats.min)}, {formatValue(sliceStats.max)}]</span>
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

	{#if showTooltip && hoverX >= 0 && hoverY >= 0 && sliceData}
		{@const idx = hoverY * sliceData.w + hoverX}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="font-mono text-gray-400">[{hoverY}, {hoverX}]</div>
			<div><span class="text-gray-400">{refLabel}:</span> {formatValue(refSlice.data[idx])}</div>
			<div><span class="text-gray-400">{mainLabel}:</span> {formatValue(mainSlice.data[idx])}</div>
			<div><span class="text-gray-400">Abs Diff:</span> {formatValue(diffSlice.data[idx])}</div>
			<div><span class="text-gray-400">log10:</span> {formatValue(sliceData.data[idx])}</div>
		</div>
	{/if}
</div>
