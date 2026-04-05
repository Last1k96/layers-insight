<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		formatValue,
		drawColorbar,
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
	let maxULP = $state(100);

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

	// Compute ULP distance using DataView for bit manipulation
	function floatToInt(f: number): number {
		const buf = new ArrayBuffer(4);
		new Float32Array(buf)[0] = f;
		return new Int32Array(buf)[0];
	}

	function ulpDistance(a: number, b: number): number {
		if (Number.isNaN(a) || Number.isNaN(b)) return -1; // NaN marker
		if (!Number.isFinite(a) || !Number.isFinite(b)) return -2; // Inf marker
		if (a === b) return 0;

		let ia = floatToInt(a);
		let ib = floatToInt(b);

		// Handle sign difference: flip negative float's int representation
		if (ia < 0) ia = 0x80000000 - ia;
		if (ib < 0) ib = 0x80000000 - ib;

		return Math.abs(ia - ib);
	}

	// ULP distance map
	let ulpMap = $derived.by(() => {
		const rSlice = extractSlice(ref, shape, batch, channel);
		const mSlice = extractSlice(main, shape, batch, channel);
		const out = new Float32Array(rSlice.data.length);
		for (let i = 0; i < out.length; i++) {
			out[i] = ulpDistance(rSlice.data[i], mSlice.data[i]);
		}
		return { data: out, w: rSlice.w, h: rSlice.h };
	});

	let refSlice = $derived(extractSlice(ref, shape, batch, channel));
	let mainSlice = $derived(extractSlice(main, shape, batch, channel));

	// Stats
	let ulpStats = $derived.by(() => {
		let exact = 0, low = 0, med = 0, high = 0, vhigh = 0, special = 0;
		for (let i = 0; i < ulpMap.data.length; i++) {
			const v = ulpMap.data[i];
			if (v < 0) special++;
			else if (v === 0) exact++;
			else if (v <= 1) low++;
			else if (v <= 10) med++;
			else if (v <= 100) high++;
			else vhigh++;
		}
		return { exact, low, med, high, vhigh, special, total: ulpMap.data.length };
	});

	let baseScale = $derived.by(() => {
		if (!canvas || !ulpMap.w) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh) return 1;
		return Math.min(dw / ulpMap.w, dh / ulpMap.h);
	});

	// Color mapping for ULP
	function ulpToRGB(ulp: number): [number, number, number] {
		if (ulp < 0) return [255, 0, 255]; // NaN/Inf
		if (ulp === 0) return [20, 120, 20]; // exact match
		if (ulp <= 1) return [50, 200, 50]; // 1 ULP
		if (ulp <= 10) return [200, 200, 50]; // 2-10 ULP
		if (ulp <= 100) return [220, 130, 30]; // 10-100 ULP
		return [220, 50, 50]; // >100 ULP
	}

	function redraw() {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const { data, w, h } = ulpMap;
		const img = new ImageData(w, h);
		for (let i = 0; i < data.length; i++) {
			const [r, g, b] = ulpToRGB(data[i]);
			img.data[i * 4] = r;
			img.data[i * 4 + 1] = g;
			img.data[i * 4 + 2] = b;
			img.data[i * 4 + 3] = 255;
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

	$effect(() => { ulpMap; zoom; panX; panY; showTooltip; hoverX; hoverY; redraw(); });

	function screenToData(cx: number, cy: number): [number, number] {
		if (!canvas) return [-1, -1];
		const rect = canvas.getBoundingClientRect();
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const ox = (dw - ulpMap.w * baseScale) / 2 + panX;
		const oy = (dh - ulpMap.h * baseScale) / 2 + panY;
		const es = baseScale * zoom;
		return [(cx - rect.left - ox) / es, (cy - rect.top - oy) / es];
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
		const rect = canvas.getBoundingClientRect();
		const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const cxOff = (dw - ulpMap.w * baseScale) / 2;
		const cyOff = (dh - ulpMap.h * baseScale) / 2;
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
		if (ix >= 0 && ix < ulpMap.w && iy >= 0 && iy < ulpMap.h) {
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
	</div>

	<div class="flex flex-wrap gap-4 text-xs">
		<div class="flex items-center gap-3">
			<span class="flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background: rgb(20,120,20)"></span> <span class="text-green-600">Exact (0)</span>: {ulpStats.exact}</span>
			<span class="flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background: rgb(50,200,50)"></span> <span class="text-green-400">1 ULP</span>: {ulpStats.low}</span>
			<span class="flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background: rgb(200,200,50)"></span> <span class="text-yellow-400">2-10</span>: {ulpStats.med}</span>
			<span class="flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background: rgb(220,130,30)"></span> <span class="text-orange-400">10-100</span>: {ulpStats.high}</span>
			<span class="flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background: rgb(220,50,50)"></span> <span class="text-red-400">>100</span>: {ulpStats.vhigh}</span>
			{#if ulpStats.special > 0}
				<span class="flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background: rgb(255,0,255)"></span> <span class="text-fuchsia-400">NaN/Inf</span>: {ulpStats.special}</span>
			{/if}
		</div>
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
		{@const idx = hoverY * ulpMap.w + hoverX}
		{@const ulp = ulpMap.data[idx]}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="font-mono text-gray-400">[{hoverY}, {hoverX}]</div>
			<div><span class="text-blue-400">{refLabel}:</span> {formatValue(refSlice.data[idx])}</div>
			<div><span class="text-red-400">{mainLabel}:</span> {formatValue(mainSlice.data[idx])}</div>
			<div><span class="text-gray-400">ULP dist:</span> <span class={ulp <= 1 ? 'text-green-400' : ulp <= 10 ? 'text-yellow-400' : ulp <= 100 ? 'text-orange-400' : 'text-red-400'}>{ulp < 0 ? 'NaN/Inf' : ulp.toLocaleString()}</span></div>
			<div><span class="text-gray-400">|Diff|:</span> {formatValue(Math.abs(refSlice.data[idx] - mainSlice.data[idx]))}</div>
		</div>
	{/if}
</div>
