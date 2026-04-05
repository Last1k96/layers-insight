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
	let sharedRange = $state(true);
	let mode: 'swipe' | 'blend' = $state('swipe');
	let swipePos = $state(0.5); // 0-1, left=ref, right=main
	let blendAlpha = $state(0.5);

	let hoverX = $state(-1);
	let hoverY = $state(-1);
	let showTooltip = $state(false);
	let tooltipScreenX = $state(0);
	let tooltipScreenY = $state(0);

	let zoom = $state(1);
	let panX = $state(0);
	let panY = $state(0);
	let dragging = $state(false);
	let swipeDragging = $state(false);
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
		const es = baseScale * zoom;
		const ox = (dw - w * baseScale) / 2 + panX;
		const oy = (dh - h * baseScale) / 2 + panY;

		ctx.clearRect(0, 0, dw, dh);
		ctx.imageSmoothingEnabled = false;

		if (mode === 'swipe') {
			// Draw ref on the left, main on the right, split at swipePos
			const splitX = ox + w * es * swipePos;

			// Ref side (left)
			const offRef = new OffscreenCanvas(w, h);
			offRef.getContext('2d')!.putImageData(refImage, 0, 0);
			ctx.save();
			ctx.beginPath();
			ctx.rect(0, 0, splitX, dh);
			ctx.clip();
			ctx.setTransform(es, 0, 0, es, ox, oy);
			ctx.drawImage(offRef, 0, 0);
			ctx.resetTransform();
			ctx.restore();

			// Main side (right)
			const offMain = new OffscreenCanvas(w, h);
			offMain.getContext('2d')!.putImageData(mainImage, 0, 0);
			ctx.save();
			ctx.beginPath();
			ctx.rect(splitX, 0, dw - splitX, dh);
			ctx.clip();
			ctx.setTransform(es, 0, 0, es, ox, oy);
			ctx.drawImage(offMain, 0, 0);
			ctx.resetTransform();
			ctx.restore();

			// Divider line
			ctx.strokeStyle = 'rgba(255,255,255,0.8)';
			ctx.lineWidth = 2;
			ctx.beginPath();
			ctx.moveTo(splitX, 0);
			ctx.lineTo(splitX, dh);
			ctx.stroke();

			// Handle
			ctx.fillStyle = 'white';
			ctx.beginPath();
			ctx.arc(splitX, dh / 2, 8, 0, Math.PI * 2);
			ctx.fill();
			ctx.strokeStyle = '#333';
			ctx.lineWidth = 1;
			ctx.stroke();

			// Labels
			ctx.fillStyle = '#60a5fa'; ctx.font = 'bold 12px monospace'; ctx.textAlign = 'left';
			ctx.fillText(refLabel, ox + 4, oy + 16);
			ctx.fillStyle = '#f87171'; ctx.textAlign = 'right';
			ctx.fillText(mainLabel, ox + w * es - 4, oy + 16);
		} else {
			// Blend mode: alpha composite
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
			ctx.setTransform(es, 0, 0, es, ox, oy);
			ctx.drawImage(offscreen, 0, 0);
			ctx.resetTransform();
		}

		if (showTooltip && hoverX >= 0 && hoverY >= 0) {
			const sx = hoverX * es + ox, sy = hoverY * es + oy;
			ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1;
			ctx.beginPath(); ctx.moveTo(sx + 0.5, 0); ctx.lineTo(sx + 0.5, dh); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(0, sy + 0.5); ctx.lineTo(dw, sy + 0.5); ctx.stroke();
		}

		drawColorbar(ctx, 10, dh - 30, Math.min(200, dw - 20), 12, colormap, range?.[0] ?? 0, range?.[1] ?? 1);
	}

	$effect(() => { refImage; mainImage; swipePos; blendAlpha; mode; zoom; panX; panY; showTooltip; hoverX; hoverY; redraw(); });

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
		if (isNearSwipeLine(e.clientX)) {
			swipeDragging = true;
		} else {
			dragging = true;
			dragStartX = e.clientX; dragStartY = e.clientY;
			panStartX = panX; panStartY = panY;
		}
	}
	function handleMouseMove(e: MouseEvent) {
		if (swipeDragging && refSlice) {
			const rect = canvas.getBoundingClientRect();
			const dw = canvas.clientWidth;
			const ox = (dw - refSlice.w * baseScale) / 2 + panX;
			const es = baseScale * zoom;
			const localX = e.clientX - rect.left - ox;
			swipePos = Math.max(0, Math.min(1, localX / (refSlice.w * es)));
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
		<div class="flex items-center gap-1">
			<button
				class="px-2 py-0.5 rounded text-xs border border-edge"
				class:bg-accent={mode === 'swipe'}
				class:text-white={mode === 'swipe'}
				class:text-gray-400={mode !== 'swipe'}
				onclick={() => mode = 'swipe'}
			>Swipe</button>
			<button
				class="px-2 py-0.5 rounded text-xs border border-edge"
				class:bg-accent={mode === 'blend'}
				class:text-white={mode === 'blend'}
				class:text-gray-400={mode !== 'blend'}
				onclick={() => mode = 'blend'}
			>Blend</button>
		</div>
		{#if mode === 'blend'}
			<label class="flex items-center gap-2">
				<span class="text-blue-400">{refLabel}</span>
				<input use:rangeScroll type="range" min="0" max="1" step="0.01" bind:value={blendAlpha} class="w-24" />
				<span class="text-red-400">{mainLabel}</span>
			</label>
		{/if}
		<label class="flex items-center gap-1.5 text-gray-400">
			<input type="checkbox" bind:checked={sharedRange} /> Shared range
		</label>
	</div>

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
