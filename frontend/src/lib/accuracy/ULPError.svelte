<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		formatValue,
		drawColorbar,
		valueToImageData,
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
	let histCanvas: HTMLCanvasElement;
	let batch = $state(0);
	let channel = $state(0);
	let maxULP = $state(100);

	let binPreset = $state<'default' | 'fine' | 'coarse'>('default');
	const BIN_PRESETS = {
		default: [0, 1, 10, 100],
		fine: [0, 1, 5, 10, 50, 100],
		coarse: [0, 10, 1000],
	};
	let bins = $derived(BIN_PRESETS[binPreset]);

	let viewMode = $state<'categorical' | 'continuous'>('categorical');
	let continuousColormap: ColormapName = $state('inferno');

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

	// Stats — dynamic bins
	let ulpStats = $derived.by(() => {
		const counts = new Array(bins.length + 1).fill(0); // bins.length buckets + 1 overflow + special
		let special = 0;
		for (let i = 0; i < ulpMap.data.length; i++) {
			const v = ulpMap.data[i];
			if (v < 0) { special++; continue; }
			let placed = false;
			for (let b = 0; b < bins.length; b++) {
				if (v <= bins[b]) { counts[b]++; placed = true; break; }
			}
			if (!placed) counts[bins.length]++;
		}
		return { counts, special, total: ulpMap.data.length, bins };
	});

	let baseScale = $derived.by(() => {
		if (!canvas || !ulpMap.w) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh) return 1;
		return Math.min(dw / ulpMap.w, dh / ulpMap.h);
	});

	// Categorical color palette for dynamic bins
	const BIN_COLORS: [number, number, number][] = [
		[20, 120, 20],   // exact match (0)
		[50, 200, 50],   // 1st bin boundary
		[200, 200, 50],  // 2nd bin boundary
		[220, 170, 30],  // 3rd bin boundary
		[220, 130, 30],  // 4th bin boundary
		[220, 80, 30],   // 5th bin boundary
		[220, 50, 50],   // overflow
	];

	// Color mapping for ULP — dynamic bins
	function ulpToRGB(ulp: number): [number, number, number] {
		if (ulp < 0) return [255, 0, 255]; // NaN/Inf
		for (let b = 0; b < bins.length; b++) {
			if (ulp <= bins[b]) return BIN_COLORS[Math.min(b, BIN_COLORS.length - 1)];
		}
		return BIN_COLORS[Math.min(bins.length, BIN_COLORS.length - 1)]; // overflow
	}

	function resetView() { zoom = 1; panX = 0; panY = 0; }

	function redraw() {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const { data, w, h } = ulpMap;

		let offscreen: OffscreenCanvas;
		if (viewMode === 'continuous') {
			// Continuous mode: log of ULP values through a colormap
			const logData = new Float32Array(data.length);
			for (let i = 0; i < data.length; i++) {
				logData[i] = data[i] > 0 ? Math.log(data[i] + 1) : (data[i] < 0 ? -1 : 0);
			}
			const img = valueToImageData(logData, w, h, continuousColormap);
			offscreen = new OffscreenCanvas(w, h);
			offscreen.getContext('2d')!.putImageData(img, 0, 0);
		} else {
			const img = new ImageData(w, h);
			for (let i = 0; i < data.length; i++) {
				const [r, g, b] = ulpToRGB(data[i]);
				img.data[i * 4] = r;
				img.data[i * 4 + 1] = g;
				img.data[i * 4 + 2] = b;
				img.data[i * 4 + 3] = 255;
			}
			offscreen = new OffscreenCanvas(w, h);
			offscreen.getContext('2d')!.putImageData(img, 0, 0);
		}

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

	// ULP histogram data: 20 log-spaced bins
	let histBins = $derived.by(() => {
		const { data } = ulpMap;
		let maxVal = 1;
		for (let i = 0; i < data.length; i++) {
			if (data[i] > maxVal) maxVal = data[i];
		}
		const numBins = 20;
		const logMax = Math.log(maxVal + 1);
		const edges: number[] = [];
		for (let i = 0; i <= numBins; i++) {
			edges.push(Math.exp((i / numBins) * logMax) - 1);
		}
		const counts = new Array(numBins).fill(0);
		for (let i = 0; i < data.length; i++) {
			if (data[i] < 0) continue; // skip NaN/Inf
			for (let b = 0; b < numBins; b++) {
				if (data[i] <= edges[b + 1] || b === numBins - 1) { counts[b]++; break; }
			}
		}
		return { edges, counts, maxVal };
	});

	function drawHistogram() {
		if (!histCanvas) return;
		const ctx = histCanvas.getContext('2d');
		if (!ctx) return;
		const dw = histCanvas.clientWidth, dh = histCanvas.clientHeight;
		histCanvas.width = dw; histCanvas.height = dh;

		const { counts, edges } = histBins;
		const maxCount = Math.max(...counts, 1);
		const logMax = Math.log(maxCount + 1);
		const barW = dw / counts.length;

		ctx.clearRect(0, 0, dw, dh);
		for (let i = 0; i < counts.length; i++) {
			const barH = counts[i] > 0 ? (Math.log(counts[i] + 1) / logMax) * (dh - 4) : 0;
			const midUlp = (edges[i] + edges[i + 1]) / 2;
			const [r, g, b] = ulpToRGB(midUlp);
			ctx.fillStyle = `rgb(${r},${g},${b})`;
			ctx.fillRect(i * barW + 1, dh - barH, barW - 2, barH);
		}
	}

	$effect(() => { ulpMap; zoom; panX; panY; showTooltip; hoverX; hoverY; viewMode; bins; continuousColormap; redraw(); });
	$effect(() => { histBins; drawHistogram(); });

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
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Bins:</span>
			<select use:rangeScroll bind:value={binPreset} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				<option value="default">Default</option>
				<option value="fine">Fine</option>
				<option value="coarse">Coarse</option>
			</select>
		</label>
		<div class="flex items-center gap-1">
			<span class="text-gray-400">View:</span>
			<button onclick={() => viewMode = 'categorical'} class="px-2 py-0.5 rounded text-xs {viewMode === 'categorical' ? 'bg-blue-600 text-white' : 'bg-surface-base border border-edge text-gray-400'}">Categorical</button>
			<button onclick={() => viewMode = 'continuous'} class="px-2 py-0.5 rounded text-xs {viewMode === 'continuous' ? 'bg-blue-600 text-white' : 'bg-surface-base border border-edge text-gray-400'}">Continuous</button>
		</div>
		{#if viewMode === 'continuous'}
			<label class="flex items-center gap-2">
				<span class="text-gray-400">Colormap:</span>
				<select use:rangeScroll bind:value={continuousColormap} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
					{#each ALL_COLORMAP_OPTIONS as opt}
						<option value={opt.value}>{opt.label}</option>
					{/each}
				</select>
			</label>
		{/if}
		<button onclick={resetView} class="px-2 py-0.5 rounded bg-surface-base border border-edge text-gray-400 hover:text-gray-200 text-xs">Reset zoom</button>
	</div>

	<div class="flex flex-wrap gap-4 text-xs">
		<div class="flex items-center gap-3">
			{#each ulpStats.bins as boundary, i}
				{@const color = BIN_COLORS[Math.min(i, BIN_COLORS.length - 1)]}
				{@const label = i === 0 ? `Exact (0)` : `${ulpStats.bins[i - 1]}-${boundary}`}
				<span class="flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background: rgb({color[0]},{color[1]},{color[2]})"></span> <span class="text-gray-300">{label}</span>: {ulpStats.counts[i]}</span>
			{/each}
			{#if true}
				{@const overflowColor = BIN_COLORS[Math.min(bins.length, BIN_COLORS.length - 1)]}
				<span class="flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background: rgb({overflowColor[0]},{overflowColor[1]},{overflowColor[2]})"></span> <span class="text-gray-300">&gt;{ulpStats.bins[ulpStats.bins.length - 1]}</span>: {ulpStats.counts[ulpStats.bins.length]}</span>
			{/if}
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

	<div class="bg-surface-base rounded-lg px-4 py-2">
		<div class="text-xs text-gray-500 mb-1">ULP distribution (log scale)</div>
		<canvas bind:this={histCanvas} class="w-full" style="height: 60px;"></canvas>
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
