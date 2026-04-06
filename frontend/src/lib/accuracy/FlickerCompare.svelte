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

	let canvas: HTMLCanvasElement;
	let batch = $state(0);
	let channel = $state(0);
	let colormap: ColormapName = $state('viridis');
	let sharedRange = $state(true);
	let playing = $state(true);
	let speed = $state(2); // Hz
	let showingRef = $state(true);

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

	// Shared range for consistent comparison
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

	let currentSlice = $derived(showingRef ? refSlice : mainSlice);
	let currentImage = $derived(showingRef ? refImage : mainImage);

	let baseScale = $derived.by(() => {
		if (!refSlice || !canvas) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh || !refSlice.w || !refSlice.h) return 1;
		return Math.min(dw / refSlice.w, dh / refSlice.h);
	});

	// Flicker timer
	let intervalId: ReturnType<typeof setInterval> | null = null;

	function startFlicker() {
		stopFlicker();
		if (playing && speed > 0) {
			intervalId = setInterval(() => { showingRef = !showingRef; }, 1000 / speed);
		}
	}

	function stopFlicker() {
		if (intervalId !== null) { clearInterval(intervalId); intervalId = null; }
	}

	$effect(() => {
		playing; speed;
		startFlicker();
		return () => stopFlicker();
	});

	function redraw() {
		if (!canvas || !currentImage || !currentSlice) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const w = currentSlice.w, h = currentSlice.h;
		const offscreen = new OffscreenCanvas(w, h);
		offscreen.getContext('2d')!.putImageData(currentImage, 0, 0);

		const es = baseScale * zoom;
		const ox = (dw - w * baseScale) / 2 + panX;
		const oy = (dh - h * baseScale) / 2 + panY;

		ctx.clearRect(0, 0, dw, dh);
		ctx.setTransform(es, 0, 0, es, ox, oy);
		ctx.imageSmoothingEnabled = false;
		ctx.drawImage(offscreen, 0, 0);
		ctx.resetTransform();

		// Label
		ctx.fillStyle = showingRef ? '#60a5fa' : '#f87171';
		ctx.font = 'bold 14px monospace';
		ctx.textAlign = 'left';
		ctx.textBaseline = 'top';
		ctx.fillText(showingRef ? refLabel : mainLabel, 10, 10);

		if (showTooltip && hoverX >= 0 && hoverY >= 0) {
			const sx = hoverX * es + ox, sy = hoverY * es + oy;
			ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1;
			ctx.beginPath(); ctx.moveTo(sx + 0.5, 0); ctx.lineTo(sx + 0.5, dh); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(0, sy + 0.5); ctx.lineTo(dw, sy + 0.5); ctx.stroke();
		}

		drawColorbar(ctx, 10, dh - 30, Math.min(200, dw - 20), 12, colormap, range?.[0] ?? 0, range?.[1] ?? 1);
	}

	$effect(() => { currentImage; zoom; panX; panY; showTooltip; hoverX; hoverY; redraw(); });

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

	function resetView() { zoom = 1; panX = 0; panY = 0; }
</script>

<svelte:window onmouseup={handleMouseUp} />

<div class="flex flex-col gap-4 relative h-full" tabindex="0" use:keyboardNav={{
	onResetZoom: resetView,
	onTogglePlay: () => { playing = !playing; },
	onNextChannel: () => { if (channel < dims.channels - 1) channel++; },
	onPrevChannel: () => { if (channel > 0) channel--; },
}}>
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
			<span class="text-gray-400">Speed:</span>
			<input use:rangeScroll type="range" min="0.5" max="10" step="0.5" bind:value={speed} class="w-20" />
			<span class="text-gray-300">{speed}Hz</span>
		</label>
		<button
			class="px-2 py-0.5 rounded text-xs border border-edge hover:bg-surface-hover"
			class:text-green-400={playing}
			class:text-yellow-400={!playing}
			onclick={() => playing = !playing}
		>
			{playing ? 'Pause' : 'Play'}
		</button>
		<button
			class="px-2 py-0.5 rounded text-xs border border-edge hover:bg-surface-hover text-gray-400"
			onclick={() => showingRef = !showingRef}
		>
			Toggle
		</button>
		<label class="flex items-center gap-1.5 text-gray-400">
			<input type="checkbox" bind:checked={sharedRange} /> Shared range
		</label>
		<button
			class="px-2 py-0.5 text-gray-400 hover:text-gray-200 border border-edge rounded text-xs"
			onclick={resetView}
		>Reset view</button>
	</div>

	<!-- Frame indicator -->
	<div class="flex items-center gap-2 text-xs">
		<span class="inline-block w-2.5 h-2.5 rounded-full" style:background={showingRef ? '#60a5fa' : '#f87171'}></span>
		<span class={showingRef ? 'text-blue-400' : 'text-red-400'}>{showingRef ? refLabel : mainLabel}</span>
		<span class="text-gray-500">Frame {showingRef ? '1' : '2'}/2</span>
		<span class="text-gray-600 text-[10px]">[Space] play/pause</span>
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
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="font-mono text-gray-400">[{hoverY}, {hoverX}]</div>
			<div><span class="text-blue-400">{refLabel}:</span> {formatValue(refSlice.data[idx])}</div>
			<div><span class="text-red-400">{mainLabel}:</span> {formatValue(mainSlice.data[idx])}</div>
			<div><span class="text-gray-400">|Diff|:</span> {formatValue(Math.abs(refSlice.data[idx] - mainSlice.data[idx]))}</div>
		</div>
	{/if}
</div>
