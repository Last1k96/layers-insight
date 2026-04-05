<script lang="ts">
	import {
		getSpatialDims,
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
	let colormap: ColormapName = $state('turbo');

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

	// Spatial cosine similarity: for each (h,w), cosine sim of C-dimensional vector
	let cosineMap = $derived.by(() => {
		const { channels, height, width } = dims;
		const spatialSize = height * width;
		const batchStride = channels * spatialSize;
		const offset = batch * batchStride;
		const map = new Float32Array(spatialSize);

		for (let hw = 0; hw < spatialSize; hw++) {
			let dot = 0, normR = 0, normM = 0;
			for (let c = 0; c < channels; c++) {
				const idx = offset + c * spatialSize + hw;
				const rv = ref[idx];
				const mv = main[idx];
				dot += rv * mv;
				normR += rv * rv;
				normM += mv * mv;
			}
			const denom = Math.sqrt(normR) * Math.sqrt(normM);
			map[hw] = denom > 0 ? dot / denom : 0;
		}
		return map;
	});

	let mapStats = $derived(computeStats(cosineMap));

	let offscreenImage = $derived.by(() => {
		return valueToImageData(cosineMap, dims.width, dims.height, colormap, [
			Math.min(mapStats.min, 0.9),
			1,
		]);
	});

	let baseScale = $derived.by(() => {
		if (!canvas || !dims.width) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh) return 1;
		return Math.min(dw / dims.width, dh / dims.height);
	});

	function redraw() {
		if (!canvas || !offscreenImage) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const w = dims.width, h = dims.height;
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

		if (showTooltip && hoverX >= 0 && hoverY >= 0) {
			const sx = hoverX * es + ox, sy = hoverY * es + oy;
			ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1;
			ctx.beginPath(); ctx.moveTo(sx + 0.5, 0); ctx.lineTo(sx + 0.5, dh); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(0, sy + 0.5); ctx.lineTo(dw, sy + 0.5); ctx.stroke();
		}

		const rangeMin = Math.min(mapStats.min, 0.9);
		drawColorbar(ctx, 10, dh - 30, Math.min(200, dw - 20), 12, colormap, rangeMin, 1);
	}

	$effect(() => { offscreenImage; zoom; panX; panY; showTooltip; hoverX; hoverY; redraw(); });

	function screenToData(cx: number, cy: number): [number, number] {
		if (!canvas) return [-1, -1];
		const rect = canvas.getBoundingClientRect();
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const ox = (dw - dims.width * baseScale) / 2 + panX;
		const oy = (dh - dims.height * baseScale) / 2 + panY;
		const es = baseScale * zoom;
		return [(cx - rect.left - ox) / es, (cy - rect.top - oy) / es];
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
		const rect = canvas.getBoundingClientRect();
		const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const cxOff = (dw - dims.width * baseScale) / 2;
		const cyOff = (dh - dims.height * baseScale) / 2;
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
		if (ix >= 0 && ix < dims.width && iy >= 0 && iy < dims.height) {
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
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Colormap:</span>
			<select use:rangeScroll bind:value={colormap} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				{#each ALL_COLORMAP_OPTIONS as opt}
					<option value={opt.value}>{opt.label}</option>
				{/each}
			</select>
		</label>
		<span class="text-gray-500">{dims.channels} channels per spatial position</span>
	</div>

	<div class="flex gap-4 text-xs text-gray-400">
		<span>Spatial cosine similarity — min: <span class={mapStats.min > 0.999 ? 'text-green-400' : mapStats.min > 0.99 ? 'text-yellow-400' : 'text-red-400'}>{mapStats.min.toFixed(6)}</span></span>
		<span>mean: {mapStats.mean.toFixed(6)}</span>
		<span>max: {mapStats.max.toFixed(6)}</span>
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
		{@const idx = hoverY * dims.width + hoverX}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="font-mono text-gray-400">[{hoverY}, {hoverX}]</div>
			<div><span class="text-gray-400">Cosine sim:</span> <span class={cosineMap[idx] > 0.999 ? 'text-green-400' : cosineMap[idx] > 0.99 ? 'text-yellow-400' : 'text-red-400'}>{cosineMap[idx].toFixed(6)}</span></div>
		</div>
	{/if}
</div>
