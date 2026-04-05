<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		valueToImageData,
		formatValue,
		drawColorbar,
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
	let colormap: ColormapName = $state('viridis');
	let blockSize = $state(4);
	let sharedRange = $state(true);

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
	let refSlice = $derived(extractSlice(ref, shape, batch, channel));
	let mainSlice = $derived(extractSlice(main, shape, batch, channel));

	let range = $derived.by((): [number, number] | undefined => {
		if (!sharedRange) return undefined;
		let lo = refSlice.data[0], hi = refSlice.data[0];
		for (let i = 1; i < refSlice.data.length; i++) {
			if (refSlice.data[i] < lo) lo = refSlice.data[i];
			if (refSlice.data[i] > hi) hi = refSlice.data[i];
		}
		for (let i = 0; i < mainSlice.data.length; i++) {
			if (mainSlice.data[i] < lo) lo = mainSlice.data[i];
			if (mainSlice.data[i] > hi) hi = mainSlice.data[i];
		}
		return [lo, hi];
	});

	let refImage = $derived(valueToImageData(refSlice.data, refSlice.w, refSlice.h, colormap, range));
	let mainImage = $derived(valueToImageData(mainSlice.data, mainSlice.w, mainSlice.h, colormap, range));

	// Build checkerboard composite
	let compositeImage = $derived.by(() => {
		const w = refSlice.w, h = refSlice.h;
		const img = new ImageData(w, h);
		for (let y = 0; y < h; y++) {
			for (let x = 0; x < w; x++) {
				const px = (y * w + x) * 4;
				const useRef = (Math.floor(x / blockSize) + Math.floor(y / blockSize)) % 2 === 0;
				const src = useRef ? refImage : mainImage;
				img.data[px] = src.data[px];
				img.data[px + 1] = src.data[px + 1];
				img.data[px + 2] = src.data[px + 2];
				img.data[px + 3] = 255;
			}
		}
		return img;
	});

	let baseScale = $derived.by(() => {
		if (!refSlice || !canvas) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh || !refSlice.w || !refSlice.h) return 1;
		return Math.min(dw / refSlice.w, dh / refSlice.h);
	});

	function redraw() {
		if (!canvas || !compositeImage || !refSlice) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const w = refSlice.w, h = refSlice.h;
		const offscreen = new OffscreenCanvas(w, h);
		offscreen.getContext('2d')!.putImageData(compositeImage, 0, 0);

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

			// Highlight which source this pixel comes from
			const useRef = (Math.floor(hoverX / blockSize) + Math.floor(hoverY / blockSize)) % 2 === 0;
			ctx.fillStyle = useRef ? 'rgba(96,165,250,0.15)' : 'rgba(248,113,113,0.15)';
			const bx = Math.floor(hoverX / blockSize) * blockSize;
			const by = Math.floor(hoverY / blockSize) * blockSize;
			ctx.fillRect(ox + bx * es, oy + by * es, blockSize * es, blockSize * es);
		}

		drawColorbar(ctx, 10, dh - 30, Math.min(200, dw - 20), 12, colormap, range?.[0] ?? 0, range?.[1] ?? 1);
	}

	$effect(() => { compositeImage; zoom; panX; panY; showTooltip; hoverX; hoverY; blockSize; redraw(); });

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
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Block:</span>
			<select use:rangeScroll bind:value={blockSize} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				{#each [1, 2, 4, 8, 16, 32] as s}
					<option value={s}>{s}px</option>
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
			<input type="checkbox" bind:checked={sharedRange} /> Shared range
		</label>
		<span class="text-gray-500">
			<span class="text-blue-400">{refLabel}</span> / <span class="text-red-400">{mainLabel}</span> interleaved
		</span>
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
		{@const useRef = (Math.floor(hoverX / blockSize) + Math.floor(hoverY / blockSize)) % 2 === 0}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="font-mono text-gray-400">[{hoverY}, {hoverX}] <span class={useRef ? 'text-blue-400' : 'text-red-400'}>({useRef ? refLabel : mainLabel})</span></div>
			<div><span class="text-blue-400">{refLabel}:</span> {formatValue(refSlice.data[idx])}</div>
			<div><span class="text-red-400">{mainLabel}:</span> {formatValue(mainSlice.data[idx])}</div>
			<div><span class="text-gray-400">|Diff|:</span> {formatValue(Math.abs(refSlice.data[idx] - mainSlice.data[idx]))}</div>
		</div>
	{/if}
</div>
