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

	// --- Spatial dims ---
	let dims = $derived(getSpatialDims(shape));

	// --- Controls state ---
	let batch = $state(0);
	let channel = $state(0);
	let colormap: ColormapName = $state('viridis');
	let sharedRange = $state(false);
	let showDiffThreshold = $state(false);
	let diffThresholdExp = $state(-3);
	let diffThreshold = $derived(Math.pow(10, diffThresholdExp));

	// --- Canvas refs ---
	let canvasRef: HTMLCanvasElement | undefined = $state();
	let canvasMain: HTMLCanvasElement | undefined = $state();
	let canvasDiff: HTMLCanvasElement | undefined = $state();

	// --- Zoom/Pan state (synchronized) ---
	let zoom = $state(1);
	let panX = $state(0);
	let panY = $state(0);

	// --- Drag state ---
	let dragging = $state(false);
	let dragStartX = 0;
	let dragStartY = 0;
	let panStartX = 0;
	let panStartY = 0;

	// --- Crosshair / tooltip state ---
	let hoverX = $state(-1);
	let hoverY = $state(-1);
	let tooltipScreenX = $state(0);
	let tooltipScreenY = $state(0);
	let showTooltip = $state(false);

	// --- Slice extraction ---
	let refSlice = $derived(extractSlice(ref, shape, batch, channel));
	let mainSlice = $derived(extractSlice(main, shape, batch, channel));

	let diffData = $derived.by(() => {
		const r = refSlice.data;
		const m = mainSlice.data;
		const d = new Float32Array(r.length);
		for (let i = 0; i < r.length; i++) {
			d[i] = Math.abs(m[i] - r[i]);
		}
		return d;
	});

	// --- Shared range computation ---
	let globalRange = $derived.by((): [number, number] | undefined => {
		if (!sharedRange) return undefined;
		const r = refSlice.data;
		const m = mainSlice.data;
		const d = diffData;
		let lo = Infinity;
		let hi = -Infinity;
		for (const arr of [r, m, d]) {
			for (let i = 0; i < arr.length; i++) {
				if (arr[i] < lo) lo = arr[i];
				if (arr[i] > hi) hi = arr[i];
			}
		}
		return [lo, hi];
	});

	// --- Auto-fit scale ---
	let baseScale = $derived.by(() => {
		const c = canvasRef;
		if (!c || !refSlice.w || !refSlice.h) return 1;
		const displayW = c.clientWidth;
		const displayH = c.clientHeight;
		if (!displayW || !displayH) return 1;
		return Math.min(displayW / refSlice.w, displayH / refSlice.h);
	});

	// --- Rendering ---
	function renderCanvas(
		canvas: HTMLCanvasElement | undefined,
		data: Float32Array,
		w: number,
		h: number,
		range: [number, number] | undefined,
		isDiff = false,
	) {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const displayW = canvas.clientWidth;
		const displayH = canvas.clientHeight;
		canvas.width = displayW;
		canvas.height = displayH;

		// Create offscreen imagedata at native resolution
		const imgData = valueToImageData(data, w, h, colormap, range);

		// Apply diff threshold overlay on diff panel
		if (isDiff && showDiffThreshold) {
			const px = imgData.data;
			for (let i = 0; i < data.length; i++) {
				if (data[i] > diffThreshold) {
					const base = i * 4;
					px[base] = Math.min(255, px[base] + 100);
					px[base + 1] = Math.max(0, px[base + 1] - 60);
					px[base + 2] = Math.max(0, px[base + 2] - 60);
				}
			}
		}

		// Create offscreen canvas to hold the image
		const offscreen = new OffscreenCanvas(w, h);
		const offCtx = offscreen.getContext('2d')!;
		offCtx.putImageData(imgData, 0, 0);

		// Auto-fit: center and scale to fill
		const effectiveScale = baseScale * zoom;
		const offsetX = (displayW - w * baseScale) / 2 + panX;
		const offsetY = (displayH - h * baseScale) / 2 + panY;

		ctx.resetTransform();
		ctx.clearRect(0, 0, displayW, displayH);
		ctx.setTransform(effectiveScale, 0, 0, effectiveScale, offsetX, offsetY);
		ctx.imageSmoothingEnabled = false;
		ctx.drawImage(offscreen, 0, 0);

		// Draw colorbar
		ctx.resetTransform();
		const stats = computeStats(data);
		const cbRange = range || [stats.min, stats.max];
		drawColorbar(ctx, 4, displayH - 24, Math.min(140, displayW - 8), 10, colormap, cbRange[0], cbRange[1]);
	}

	function drawCrosshair(canvas: HTMLCanvasElement | undefined, dataX: number, dataY: number) {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const displayW = canvas.width;
		const displayH = canvas.height;
		const w = refSlice.w;
		const h = refSlice.h;

		// Convert data coords to screen coords using auto-fit transform
		const effectiveScale = baseScale * zoom;
		const offsetX = (displayW - w * baseScale) / 2 + panX;
		const offsetY = (displayH - h * baseScale) / 2 + panY;
		const screenX = dataX * effectiveScale + offsetX;
		const screenY = dataY * effectiveScale + offsetY;

		ctx.resetTransform();
		ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
		ctx.lineWidth = 1;

		// Vertical line
		ctx.beginPath();
		ctx.moveTo(screenX + 0.5, 0);
		ctx.lineTo(screenX + 0.5, displayH);
		ctx.stroke();

		// Horizontal line
		ctx.beginPath();
		ctx.moveTo(0, screenY + 0.5);
		ctx.lineTo(displayW, screenY + 0.5);
		ctx.stroke();
	}

	// Redraw all canvases when dependencies change
	$effect(() => {
		const w = refSlice.w;
		const h = refSlice.h;
		const range = globalRange;
		const _zoom = zoom;
		const _panX = panX;
		const _panY = panY;
		const _showDiffThreshold = showDiffThreshold;
		const _diffThreshold = diffThreshold;

		renderCanvas(canvasRef, refSlice.data, w, h, range);
		renderCanvas(canvasMain, mainSlice.data, w, h, range);
		renderCanvas(canvasDiff, diffData, w, h, range, true);

		// Draw crosshair if hovering
		if (showTooltip && hoverX >= 0 && hoverY >= 0) {
			drawCrosshair(canvasRef, hoverX, hoverY);
			drawCrosshair(canvasMain, hoverX, hoverY);
			drawCrosshair(canvasDiff, hoverX, hoverY);
		}
	});

	// --- Event handlers ---
	function screenToData(canvas: HTMLCanvasElement, clientX: number, clientY: number): [number, number] {
		const rect = canvas.getBoundingClientRect();
		const sx = clientX - rect.left;
		const sy = clientY - rect.top;
		const displayW = canvas.clientWidth;
		const displayH = canvas.clientHeight;
		const w = refSlice.w;
		const h = refSlice.h;
		const effectiveScale = baseScale * zoom;
		const offsetX = (displayW - w * baseScale) / 2 + panX;
		const offsetY = (displayH - h * baseScale) / 2 + panY;
		const dataX = (sx - offsetX) / effectiveScale;
		const dataY = (sy - offsetY) / effectiveScale;
		return [dataX, dataY];
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
		panX *= factor;
		panY *= factor;
		zoom *= factor;
	}

	function handleMouseDown(e: MouseEvent) {
		if (e.button !== 0) return;
		dragging = true;
		dragStartX = e.clientX;
		dragStartY = e.clientY;
		panStartX = panX;
		panStartY = panY;
	}

	function handleMouseMove(e: MouseEvent) {
		const canvas = e.currentTarget as HTMLCanvasElement;

		if (dragging) {
			panX = panStartX + (e.clientX - dragStartX);
			panY = panStartY + (e.clientY - dragStartY);
			return;
		}

		// Update crosshair position
		const [dx, dy] = screenToData(canvas, e.clientX, e.clientY);
		const ix = Math.floor(dx);
		const iy = Math.floor(dy);

		if (ix >= 0 && ix < refSlice.w && iy >= 0 && iy < refSlice.h) {
			hoverX = ix;
			hoverY = iy;
			tooltipScreenX = e.clientX;
			tooltipScreenY = e.clientY;
			showTooltip = true;
		} else {
			showTooltip = false;
		}
	}

	function handleMouseUp() {
		dragging = false;
	}

	function handleMouseLeave() {
		dragging = false;
		showTooltip = false;
		hoverX = -1;
		hoverY = -1;
	}

	// Reset zoom/pan when shape changes
	$effect(() => {
		const _shape = shape;
		zoom = 1;
		panX = 0;
		panY = 0;
		batch = 0;
		channel = 0;
	});

	function resetView() { zoom = 1; panX = 0; panY = 0; }
</script>

<svelte:window onmouseup={handleMouseUp} />

<div class="flex w-full flex-col gap-2">
	<!-- Controls row -->
	<div class="flex flex-wrap items-center gap-4 text-sm text-gray-300 w-full">
		{#if dims.batches > 1}
			<label class="flex items-center gap-1 flex-1 min-w-[10rem]">
				<span class="shrink-0">Batch</span>
				<input
					use:rangeScroll
					type="range"
					min="0"
					max={dims.batches - 1}
					bind:value={batch}
					class="flex-1"
				/>
				<span class="w-6 text-center font-mono shrink-0">{batch}</span>
			</label>
		{/if}

		{#if dims.channels > 1}
			<label class="flex items-center gap-1 flex-1 min-w-[10rem]">
				<span class="shrink-0">Channel</span>
				<input
					use:rangeScroll
					type="range"
					min="0"
					max={dims.channels - 1}
					bind:value={channel}
					class="flex-1"
				/>
				<span class="w-6 text-center font-mono shrink-0">{channel}</span>
			</label>
		{/if}

		<label class="flex items-center gap-1">
			Colormap
			<select
				use:rangeScroll
				bind:value={colormap}
				class="rounded border border-gray-600 bg-gray-800 px-2 py-0.5 text-gray-200"
			>
				{#each ALL_COLORMAP_OPTIONS as opt}
					<option value={opt.value}>{opt.label}</option>
				{/each}
			</select>
		</label>

		<label class="flex items-center gap-1">
			<input type="checkbox" bind:checked={sharedRange} />
			Shared range
		</label>

		<label class="flex items-center gap-1">
			<input type="checkbox" bind:checked={showDiffThreshold} />
			Diff threshold
		</label>
		{#if showDiffThreshold}
			<label class="flex items-center gap-1">
				<input use:rangeScroll type="range" min="-8" max="2" step="0.5" bind:value={diffThresholdExp} />
				<span class="font-mono w-16 text-xs">{diffThreshold.toExponential(1)}</span>
			</label>
		{/if}

		<button
			class="px-2 py-0.5 text-gray-400 hover:text-gray-200 border border-gray-600 rounded text-xs"
			onclick={resetView}
		>Reset view</button>
	</div>

	<!-- Canvases -->
	<div class="grid grid-cols-3 gap-2">
		<div class="flex flex-col gap-1">
			<span class="text-center text-xs font-medium text-gray-400">{refLabel}</span>
			<canvas
				bind:this={canvasRef}
				class="h-96 w-full rounded border border-gray-700 bg-gray-900"
				style="image-rendering: pixelated"
				onwheel={handleWheel}
				onmousedown={handleMouseDown}
				onmousemove={handleMouseMove}
				onmouseleave={handleMouseLeave}
			></canvas>
		</div>

		<div class="flex flex-col gap-1">
			<span class="text-center text-xs font-medium text-gray-400">{mainLabel}</span>
			<canvas
				bind:this={canvasMain}
				class="h-96 w-full rounded border border-gray-700 bg-gray-900"
				style="image-rendering: pixelated"
				onwheel={handleWheel}
				onmousedown={handleMouseDown}
				onmousemove={handleMouseMove}
				onmouseleave={handleMouseLeave}
			></canvas>
		</div>

		<div class="flex flex-col gap-1">
			<span class="text-center text-xs font-medium text-gray-400">Abs Diff</span>
			<canvas
				bind:this={canvasDiff}
				class="h-96 w-full rounded border border-gray-700 bg-gray-900"
				style="image-rendering: pixelated"
				onwheel={handleWheel}
				onmousedown={handleMouseDown}
				onmousemove={handleMouseMove}
				onmouseleave={handleMouseLeave}
			></canvas>
		</div>
	</div>

	<!-- Tooltip -->
	{#if showTooltip && hoverX >= 0 && hoverY >= 0}
		{@const idx = hoverY * refSlice.w + hoverX}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="font-mono text-gray-400">
				[{hoverY}, {hoverX}]
			</div>
			<div>
				<span class="text-gray-400">{refLabel}:</span>
				{formatValue(refSlice.data[idx])}
			</div>
			<div>
				<span class="text-gray-400">{mainLabel}:</span>
				{formatValue(mainSlice.data[idx])}
			</div>
			<div>
				<span class="text-gray-400">Diff:</span>
				{formatValue(diffData[idx])}
			</div>
		</div>
	{/if}
</div>
