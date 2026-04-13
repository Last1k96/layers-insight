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

	type CompareMode = 'flicker' | 'swipe' | 'blend' | 'checkerboard';
	let mode = $state<CompareMode>('flicker');

	let canvas: HTMLCanvasElement;
	let batch = $state(0);
	let channel = $state(0);
	let colormap: ColormapName = $state('viridis');
	let sharedRange = $state(true);

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

	// Flicker state
	let playing = $state(true);
	let speed = $state(2);
	let showingRef = $state(true);

	// Swipe state
	let swipePos = $state(0.5);
	let swipeDragging = $state(false);
	let animating = $state(false);
	let animDir = $state(1);

	// Blend state
	let blendAlpha = $state(0.5);

	// Checkerboard state
	let blockSize = $state(4);
	let showGrid = $state(false);

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
		if (playing && speed > 0 && mode === 'flicker') {
			intervalId = setInterval(() => { showingRef = !showingRef; }, 1000 / speed);
		}
	}
	function stopFlicker() {
		if (intervalId !== null) { clearInterval(intervalId); intervalId = null; }
	}
	$effect(() => { playing; speed; mode; startFlicker(); return () => stopFlicker(); });

	// Swipe animation
	$effect(() => {
		if (!animating || mode !== 'swipe') return;
		let frameId: number;
		function tick() {
			swipePos += animDir * 0.005;
			if (swipePos >= 1) { swipePos = 1; animDir = -1; }
			if (swipePos <= 0) { swipePos = 0; animDir = 1; }
			frameId = requestAnimationFrame(tick);
		}
		frameId = requestAnimationFrame(tick);
		return () => cancelAnimationFrame(frameId);
	});

	function redraw() {
		if (!canvas || !refSlice) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;
		const w = refSlice.w, h = refSlice.h;
		const es = baseScale * zoom;
		const ox = (dw - w * baseScale) / 2 + panX;
		const oy = (dh - h * baseScale) / 2 + panY;

		ctx.clearRect(0, 0, dw, dh);
		ctx.imageSmoothingEnabled = false;

		if (mode === 'flicker') {
			const img = showingRef ? refImage : mainImage;
			const offscreen = new OffscreenCanvas(w, h);
			offscreen.getContext('2d')!.putImageData(img, 0, 0);
			ctx.setTransform(es, 0, 0, es, ox, oy);
			ctx.drawImage(offscreen, 0, 0);
			ctx.resetTransform();
			// Label
			ctx.fillStyle = showingRef ? '#60a5fa' : '#f87171';
			ctx.font = 'bold 14px monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
			ctx.fillText(showingRef ? refLabel : mainLabel, 10, 10);
		} else if (mode === 'swipe') {
			const splitX = ox + w * es * swipePos;
			const offRef = new OffscreenCanvas(w, h);
			offRef.getContext('2d')!.putImageData(refImage, 0, 0);
			ctx.save(); ctx.beginPath(); ctx.rect(0, 0, splitX, dh); ctx.clip();
			ctx.setTransform(es, 0, 0, es, ox, oy); ctx.drawImage(offRef, 0, 0); ctx.resetTransform(); ctx.restore();
			const offMain = new OffscreenCanvas(w, h);
			offMain.getContext('2d')!.putImageData(mainImage, 0, 0);
			ctx.save(); ctx.beginPath(); ctx.rect(splitX, 0, dw - splitX, dh); ctx.clip();
			ctx.setTransform(es, 0, 0, es, ox, oy); ctx.drawImage(offMain, 0, 0); ctx.resetTransform(); ctx.restore();
			// Divider + handle
			ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 2;
			ctx.beginPath(); ctx.moveTo(splitX, 0); ctx.lineTo(splitX, dh); ctx.stroke();
			ctx.fillStyle = 'white'; ctx.beginPath(); ctx.arc(splitX, dh / 2, 8, 0, Math.PI * 2); ctx.fill();
			ctx.strokeStyle = '#333'; ctx.lineWidth = 1; ctx.stroke();
			ctx.fillStyle = '#60a5fa'; ctx.font = 'bold 12px monospace'; ctx.textAlign = 'left';
			ctx.fillText(refLabel, ox + 4, oy + 16);
			ctx.fillStyle = '#f87171'; ctx.textAlign = 'right';
			ctx.fillText(mainLabel, ox + w * es - 4, oy + 16);
		} else if (mode === 'blend') {
			const blended = new ImageData(w, h);
			const a = blendAlpha;
			for (let i = 0; i < w * h; i++) {
				const px = i * 4;
				blended.data[px] = Math.round(refImage.data[px] * (1 - a) + mainImage.data[px] * a);
				blended.data[px + 1] = Math.round(refImage.data[px + 1] * (1 - a) + mainImage.data[px + 1] * a);
				blended.data[px + 2] = Math.round(refImage.data[px + 2] * (1 - a) + mainImage.data[px + 2] * a);
				blended.data[px + 3] = 255;
			}
			const offscreen = new OffscreenCanvas(w, h);
			offscreen.getContext('2d')!.putImageData(blended, 0, 0);
			ctx.setTransform(es, 0, 0, es, ox, oy); ctx.drawImage(offscreen, 0, 0); ctx.resetTransform();
		} else if (mode === 'checkerboard') {
			const img = new ImageData(w, h);
			for (let y = 0; y < h; y++) {
				for (let x = 0; x < w; x++) {
					const px = (y * w + x) * 4;
					const useRef = (Math.floor(x / blockSize) + Math.floor(y / blockSize)) % 2 === 0;
					const src = useRef ? refImage : mainImage;
					img.data[px] = src.data[px]; img.data[px + 1] = src.data[px + 1];
					img.data[px + 2] = src.data[px + 2]; img.data[px + 3] = 255;
				}
			}
			const offscreen = new OffscreenCanvas(w, h);
			offscreen.getContext('2d')!.putImageData(img, 0, 0);
			ctx.setTransform(es, 0, 0, es, ox, oy); ctx.drawImage(offscreen, 0, 0); ctx.resetTransform();
			// Grid overlay
			if (showGrid) {
				ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.lineWidth = 1;
				for (let gx = 0; gx <= w; gx += blockSize) {
					const sx = ox + gx * es;
					ctx.beginPath(); ctx.moveTo(sx, oy); ctx.lineTo(sx, oy + h * es); ctx.stroke();
				}
				for (let gy = 0; gy <= h; gy += blockSize) {
					const sy = oy + gy * es;
					ctx.beginPath(); ctx.moveTo(ox, sy); ctx.lineTo(ox + w * es, sy); ctx.stroke();
				}
			}
			// Block highlight on hover
			if (showTooltip && hoverX >= 0 && hoverY >= 0) {
				const useRef = (Math.floor(hoverX / blockSize) + Math.floor(hoverY / blockSize)) % 2 === 0;
				ctx.fillStyle = useRef ? 'rgba(96,165,250,0.15)' : 'rgba(248,113,113,0.15)';
				const bx = Math.floor(hoverX / blockSize) * blockSize;
				const by = Math.floor(hoverY / blockSize) * blockSize;
				ctx.fillRect(ox + bx * es, oy + by * es, blockSize * es, blockSize * es);
			}
		}

		// Crosshair
		if (showTooltip && hoverX >= 0 && hoverY >= 0) {
			const sx = hoverX * es + ox, sy = hoverY * es + oy;
			ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1;
			ctx.beginPath(); ctx.moveTo(sx + 0.5, 0); ctx.lineTo(sx + 0.5, dh); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(0, sy + 0.5); ctx.lineTo(dw, sy + 0.5); ctx.stroke();
		}

		drawColorbar(ctx, 10, dh - 30, Math.min(200, dw - 20), 12, colormap, range?.[0] ?? 0, range?.[1] ?? 1);
	}

	$effect(() => {
		refImage; mainImage; zoom; panX; panY; showTooltip; hoverX; hoverY;
		mode; showingRef; swipePos; blendAlpha; blockSize; showGrid;
		redraw();
	});

	// Event handlers
	function screenToData(cx: number, cy: number): [number, number] {
		if (!canvas || !refSlice) return [-1, -1];
		const rect = canvas.getBoundingClientRect();
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const ox = (dw - refSlice.w * baseScale) / 2 + panX;
		const oy = (dh - refSlice.h * baseScale) / 2 + panY;
		const es = baseScale * zoom;
		return [(cx - rect.left - ox) / es, (cy - rect.top - oy) / es];
	}

	function isNearSwipeLine(cx: number): boolean {
		if (!canvas || !refSlice || mode !== 'swipe') return false;
		const rect = canvas.getBoundingClientRect();
		const dw = canvas.clientWidth;
		const ox = (dw - refSlice.w * baseScale) / 2 + panX;
		const es = baseScale * zoom;
		const splitX = ox + refSlice.w * es * swipePos + rect.left;
		return Math.abs(cx - splitX) < 12;
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
		if (mode === 'swipe' && isNearSwipeLine(e.clientX)) {
			swipeDragging = true;
		} else {
			dragging = true; dragStartX = e.clientX; dragStartY = e.clientY;
			panStartX = panX; panStartY = panY;
		}
	}
	function handleMouseMove(e: MouseEvent) {
		if (swipeDragging && refSlice) {
			const rect = canvas.getBoundingClientRect();
			const dw = canvas.clientWidth;
			const ox = (dw - refSlice.w * baseScale) / 2 + panX;
			const es = baseScale * zoom;
			swipePos = Math.max(0, Math.min(1, (e.clientX - rect.left - ox) / (refSlice.w * es)));
			return;
		}
		if (dragging) { panX = panStartX + (e.clientX - dragStartX); panY = panStartY + (e.clientY - dragStartY); return; }
		const [dx, dy] = screenToData(e.clientX, e.clientY);
		const ix = Math.floor(dx), iy = Math.floor(dy);
		if (refSlice && ix >= 0 && ix < refSlice.w && iy >= 0 && iy < refSlice.h) {
			hoverX = ix; hoverY = iy; tooltipScreenX = e.clientX; tooltipScreenY = e.clientY; showTooltip = true;
		} else { showTooltip = false; }
	}
	function handleMouseUp() { dragging = false; swipeDragging = false; }
	function handleMouseLeave() { dragging = false; swipeDragging = false; showTooltip = false; }

	$effect(() => { shape; zoom = 1; panX = 0; panY = 0; });
	function resetView() { zoom = 1; panX = 0; panY = 0; }

	function blockMeanDiff(bx: number, by: number): number {
		if (!refSlice || !mainSlice) return 0;
		const w = refSlice.w, h = refSlice.h;
		let sum = 0, count = 0;
		for (let dy = 0; dy < blockSize && by + dy < h; dy++) {
			for (let dx = 0; dx < blockSize && bx + dx < w; dx++) {
				sum += Math.abs(refSlice.data[(by + dy) * w + (bx + dx)] - mainSlice.data[(by + dy) * w + (bx + dx)]);
				count++;
			}
		}
		return count > 0 ? sum / count : 0;
	}
</script>

<svelte:window onmouseup={handleMouseUp} />

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div class="flex flex-col gap-3 relative h-full" tabindex="0" use:keyboardNav={{
	onResetZoom: resetView,
	onTogglePlay: () => { if (mode === 'flicker') playing = !playing; else if (mode === 'swipe') animating = !animating; },
	onNextChannel: () => { if (channel < dims.channels - 1) channel++; },
	onPrevChannel: () => { if (channel > 0) channel--; },
}}>
	<div class="flex flex-wrap gap-3 items-center text-xs">
		<!-- Mode selector -->
		<div class="flex items-center gap-1">
			{#each [['flicker', 'Flicker'], ['swipe', 'Swipe'], ['blend', 'Blend'], ['checkerboard', 'Checker']] as [m, label]}
				<button
					class="px-2 py-0.5 rounded text-xs border border-edge"
					class:bg-accent={mode === m}
					class:text-white={mode === m}
					class:text-gray-400={mode !== m}
					onclick={() => mode = m as CompareMode}
				>{label}</button>
			{/each}
		</div>

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

		<!-- Flicker controls -->
		{#if mode === 'flicker'}
			<label class="flex items-center gap-2">
				<span class="text-gray-400">Speed:</span>
				<input use:rangeScroll type="range" min="0.5" max="10" step="0.5" bind:value={speed} class="w-20" />
				<span class="text-gray-300">{speed}Hz</span>
			</label>
			<button
				class="px-2 py-0.5 rounded text-xs border border-edge hover:bg-surface-hover"
				class:text-green-400={playing} class:text-yellow-400={!playing}
				onclick={() => playing = !playing}
			>{playing ? 'Pause' : 'Play'}</button>
			<button class="px-2 py-0.5 rounded text-xs border border-edge hover:bg-surface-hover text-gray-400"
				onclick={() => showingRef = !showingRef}>Toggle</button>
		{/if}

		<!-- Blend controls -->
		{#if mode === 'blend'}
			<label class="flex items-center gap-2">
				<span class="text-blue-400">{refLabel}</span>
				<input use:rangeScroll type="range" min="0" max="1" step="0.01" bind:value={blendAlpha} class="w-24" />
				<span class="text-red-400">{mainLabel}</span>
			</label>
		{/if}

		<!-- Swipe controls -->
		{#if mode === 'swipe'}
			<button
				class="px-2 py-0.5 rounded text-xs border border-edge hover:bg-surface-hover"
				class:text-green-400={animating} class:text-gray-400={!animating}
				onclick={() => animating = !animating}
			>{animating ? 'Stop' : 'Animate'}</button>
		{/if}

		<!-- Checkerboard controls -->
		{#if mode === 'checkerboard'}
			<label class="flex items-center gap-2">
				<span class="text-gray-400">Block:</span>
				<select use:rangeScroll bind:value={blockSize} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
					{#each [1, 2, 4, 8, 16, 32] as s}
						<option value={s}>{s}px</option>
					{/each}
				</select>
			</label>
			<label class="flex items-center gap-1.5 text-gray-400">
				<input type="checkbox" bind:checked={showGrid} /> Grid
			</label>
		{/if}

		<label class="flex items-center gap-1.5 text-gray-400">
			<input type="checkbox" bind:checked={sharedRange} /> Shared range
		</label>
		<button class="px-2 py-0.5 text-gray-400 hover:text-gray-200 border border-edge rounded text-xs" onclick={resetView}>Reset view</button>
	</div>

	<!-- Status line -->
	<div class="text-xs text-gray-500">
		{#if mode === 'flicker'}
			<span class="inline-block w-2.5 h-2.5 rounded-full align-middle" style:background={showingRef ? '#60a5fa' : '#f87171'}></span>
			<span class={showingRef ? 'text-blue-400' : 'text-red-400'}>{showingRef ? refLabel : mainLabel}</span>
			<span>Frame {showingRef ? '1' : '2'}/2</span>
			<span class="text-gray-600 text-[10px] ml-1">[Space] play/pause</span>
		{:else if mode === 'swipe'}
			Swipe: {(swipePos * 100).toFixed(0)}% — <span class="text-blue-400">{refLabel}</span> | <span class="text-red-400">{mainLabel}</span>
		{:else if mode === 'blend'}
			Blend: {((1 - blendAlpha) * 100).toFixed(0)}% <span class="text-blue-400">{refLabel}</span> / {(blendAlpha * 100).toFixed(0)}% <span class="text-red-400">{mainLabel}</span>
		{:else if mode === 'checkerboard'}
			<span class="text-blue-400">{refLabel}</span> / <span class="text-red-400">{mainLabel}</span> interleaved
		{/if}
	</div>

	<!-- Canvas -->
	<div class="flex-1 flex justify-center bg-surface-base rounded-lg p-4 overflow-hidden min-h-0">
		<canvas
			bind:this={canvas}
			class="w-full h-full"
			class:cursor-col-resize={mode === 'swipe'}
			class:cursor-crosshair={mode !== 'swipe'}
			style="image-rendering: pixelated;"
			onwheel={handleWheel}
			onmousedown={handleMouseDown}
			onmousemove={handleMouseMove}
			onmouseleave={handleMouseLeave}
		></canvas>
	</div>

	<!-- Tooltip -->
	{#if showTooltip && hoverX >= 0 && hoverY >= 0 && refSlice}
		{@const idx = hoverY * refSlice.w + hoverX}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="font-mono text-gray-400">
				[{hoverY}, {hoverX}]
				{#if mode === 'checkerboard'}
					{@const useRef = (Math.floor(hoverX / blockSize) + Math.floor(hoverY / blockSize)) % 2 === 0}
					<span class={useRef ? 'text-blue-400' : 'text-red-400'}>({useRef ? refLabel : mainLabel})</span>
				{/if}
			</div>
			<div><span class="text-blue-400">{refLabel}:</span> {formatValue(refSlice.data[idx])}</div>
			<div><span class="text-red-400">{mainLabel}:</span> {formatValue(mainSlice.data[idx])}</div>
			<div><span class="text-gray-400">|Diff|:</span> {formatValue(Math.abs(refSlice.data[idx] - mainSlice.data[idx]))}</div>
			{#if mode === 'checkerboard'}
				<div><span class="text-gray-400">Block mean |diff|:</span> {formatValue(blockMeanDiff(Math.floor(hoverX / blockSize) * blockSize, Math.floor(hoverY / blockSize) * blockSize))}</div>
			{/if}
		</div>
	{/if}
</div>
