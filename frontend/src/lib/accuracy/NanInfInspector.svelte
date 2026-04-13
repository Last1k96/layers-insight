<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		formatValue,
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
	let showDenormals = $state(true);
	let filterClass = $state<'all' | 'nan' | 'inf' | 'negInf' | 'denorm'>('all');

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

	// Classify each element
	type ValueClass = 'normal' | 'nan' | 'inf' | 'ninf' | 'denormal' | 'zero';

	function classify(v: number): ValueClass {
		if (Number.isNaN(v)) return 'nan';
		if (v === Infinity) return 'inf';
		if (v === -Infinity) return 'ninf';
		if (v === 0) return 'zero';
		if (Math.abs(v) < 1.17549435e-38) return 'denormal'; // FP32 min normal
		return 'normal';
	}

	const classColors: Record<ValueClass, [number, number, number]> = {
		normal: [40, 40, 40],
		nan: [255, 50, 50],
		inf: [255, 50, 255],
		ninf: [50, 220, 255],
		denormal: [255, 220, 50],
		zero: [30, 30, 30],
	};

	// Global scan
	interface ScanResult {
		nan: number;
		inf: number;
		ninf: number;
		denormal: number;
		zero: number;
		total: number;
	}

	function scan(data: Float32Array): ScanResult {
		const r: ScanResult = { nan: 0, inf: 0, ninf: 0, denormal: 0, zero: 0, total: data.length };
		for (let i = 0; i < data.length; i++) {
			const c = classify(data[i]);
			if (c === 'nan') r.nan++;
			else if (c === 'inf') r.inf++;
			else if (c === 'ninf') r.ninf++;
			else if (c === 'denormal') r.denormal++;
			else if (c === 'zero') r.zero++;
		}
		return r;
	}

	let mainScan = $derived(scan(main));
	let refScan = $derived(scan(ref));

	let mainSlice = $derived(extractSlice(main, shape, batch, channel));
	let refSlice = $derived(extractSlice(ref, shape, batch, channel));

	// Anomaly list: first 50 anomalous positions across both tensors
	interface AnomalyEntry {
		source: 'ref' | 'main';
		batch: number;
		channel: number;
		y: number;
		x: number;
		value: number;
		cls: string;
	}

	let anomalyList = $derived.by((): AnomalyEntry[] => {
		const limit = 50;
		const bCount = dims.batches;
		const cCount = dims.channels;
		const h = dims.height;
		const w = dims.width;
		const sliceSize = h * w;

		const results: AnomalyEntry[] = [];
		for (const [tensor, source] of [[ref, 'ref' as const], [main, 'main' as const]]) {
			const data = tensor as Float32Array;
			const src = source as 'ref' | 'main';
			for (let b = 0; b < bCount && results.length < limit; b++) {
				for (let c = 0; c < cCount && results.length < limit; c++) {
					const baseIdx = (b * cCount + c) * sliceSize;
					for (let yi = 0; yi < h && results.length < limit; yi++) {
						for (let xi = 0; xi < w && results.length < limit; xi++) {
							const v = data[baseIdx + yi * w + xi];
							const cls = classify(v);
							if (cls === 'nan' || cls === 'inf' || cls === 'ninf' || cls === 'denormal') {
								const clsLabel = cls === 'nan' ? 'NaN' : cls === 'inf' ? '+Inf' : cls === 'ninf' ? '-Inf' : 'Denorm';
								results.push({ source: src, batch: b, channel: c, y: yi, x: xi, value: v, cls: clsLabel });
							}
						}
					}
				}
			}
		}
		return results;
	});

	// Find first anomaly across both tensors
	let firstAnomaly = $derived.by((): { batch: number; channel: number; y: number; x: number } | null => {
		const bCount = dims.batches;
		const cCount = dims.channels;
		const h = dims.height;
		const w = dims.width;
		const sliceSize = h * w;

		for (const data of [main, ref]) {
			for (let b = 0; b < bCount; b++) {
				for (let c = 0; c < cCount; c++) {
					const baseIdx = (b * cCount + c) * sliceSize;
					for (let yi = 0; yi < h; yi++) {
						for (let xi = 0; xi < w; xi++) {
							const v = data[baseIdx + yi * w + xi];
							const cls = classify(v);
							if (cls === 'nan' || cls === 'inf' || cls === 'ninf' || cls === 'denormal') {
								return { batch: b, channel: c, y: yi, x: xi };
							}
						}
					}
				}
			}
		}
		return null;
	});

	function jumpToAnomaly(b: number, c: number, y: number, x: number) {
		batch = b;
		channel = c;
		zoom = 4;
		// Center the pixel in the canvas after next tick
		requestAnimationFrame(() => {
			if (!canvas || !refSlice) return;
			const dw = canvas.clientWidth, dh = canvas.clientHeight;
			const w = refSlice.w, h = refSlice.h;
			const gap = 4;
			const totalW = w * 2 + gap;
			const es = baseScale * zoom;
			const defaultOx = (dw - totalW * baseScale) / 2;
			const defaultOy = (dh - h * baseScale) / 2;
			// We want the pixel (x, y) centered in canvas
			panX = dw / 2 - (x + 0.5) * es - defaultOx;
			panY = dh / 2 - (y + 0.5) * es - defaultOy;
		});
	}

	function jumpToFirstAnomaly() {
		if (firstAnomaly) {
			jumpToAnomaly(firstAnomaly.batch, firstAnomaly.channel, firstAnomaly.y, firstAnomaly.x);
		}
	}

	let baseScale = $derived.by(() => {
		if (!refSlice || !canvas) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh) return 1;
		// Show two images side by side
		const totalW = refSlice.w * 2 + 4;
		return Math.min((dw) / totalW, dh / refSlice.h);
	});

	function redraw() {
		if (!canvas || !refSlice || !mainSlice) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const w = refSlice.w, h = refSlice.h;
		const gap = 4;

		// Map filterClass to ValueClass
		const filterMap: Record<string, ValueClass | null> = {
			all: null,
			nan: 'nan',
			inf: 'inf',
			negInf: 'ninf',
			denorm: 'denormal',
		};
		const activeFilter = filterMap[filterClass];
		const dimColor: [number, number, number] = [30, 30, 30];

		// Build images
		function buildImage(data: Float32Array): ImageData {
			const img = new ImageData(w, h);
			const px = img.data;
			for (let i = 0; i < data.length; i++) {
				const c = classify(data[i]);
				if (c === 'denormal' && !showDenormals) {
					px[i * 4] = 40; px[i * 4 + 1] = 40; px[i * 4 + 2] = 40;
				} else if (activeFilter !== null && c !== activeFilter) {
					// Dim non-matching classes when filter is active
					px[i * 4] = dimColor[0]; px[i * 4 + 1] = dimColor[1]; px[i * 4 + 2] = dimColor[2];
				} else {
					const rgb = classColors[c];
					px[i * 4] = rgb[0]; px[i * 4 + 1] = rgb[1]; px[i * 4 + 2] = rgb[2];
				}
				px[i * 4 + 3] = 255;
			}
			return img;
		}

		const refImg = buildImage(refSlice.data);
		const mainImg = buildImage(mainSlice.data);

		const offRef = new OffscreenCanvas(w, h);
		offRef.getContext('2d')!.putImageData(refImg, 0, 0);
		const offMain = new OffscreenCanvas(w, h);
		offMain.getContext('2d')!.putImageData(mainImg, 0, 0);

		const totalW = w * 2 + gap;
		const es = baseScale * zoom;
		const ox = (dw - totalW * baseScale) / 2 + panX;
		const oy = (dh - h * baseScale) / 2 + panY;

		ctx.clearRect(0, 0, dw, dh);
		ctx.imageSmoothingEnabled = false;

		// Ref image
		ctx.setTransform(es, 0, 0, es, ox, oy);
		ctx.drawImage(offRef, 0, 0);
		ctx.resetTransform();

		// Main image
		ctx.setTransform(es, 0, 0, es, ox + (w + gap) * es, oy);
		ctx.drawImage(offMain, 0, 0);
		ctx.resetTransform();

		// Labels
		ctx.fillStyle = '#60a5fa'; ctx.font = 'bold 12px monospace'; ctx.textAlign = 'left';
		ctx.fillText(refLabel, ox + 2, oy - 4);
		ctx.fillStyle = '#f87171';
		ctx.fillText(mainLabel, ox + (w + gap) * es + 2, oy - 4);

		// Crosshair
		if (showTooltip && hoverX >= 0 && hoverY >= 0) {
			const sx = hoverX * es + ox, sy = hoverY * es + oy;
			ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1;
			ctx.beginPath(); ctx.moveTo(0, sy + 0.5); ctx.lineTo(dw, sy + 0.5); ctx.stroke();
			// Vertical for both panels
			ctx.beginPath(); ctx.moveTo(sx + 0.5, 0); ctx.lineTo(sx + 0.5, dh); ctx.stroke();
			const sx2 = sx + (w + gap) * es;
			ctx.beginPath(); ctx.moveTo(sx2 + 0.5, 0); ctx.lineTo(sx2 + 0.5, dh); ctx.stroke();
		}
	}

	$effect(() => { refSlice; mainSlice; showDenormals; filterClass; zoom; panX; panY; showTooltip; hoverX; hoverY; redraw(); });

	function screenToData(cx: number, cy: number): [number, number] {
		if (!canvas || !refSlice) return [-1, -1];
		const rect = canvas.getBoundingClientRect();
		const w = refSlice.w, h = refSlice.h, gap = 4;
		const totalW = w * 2 + gap;
		const es = baseScale * zoom;
		const ox = (canvas.clientWidth - totalW * baseScale) / 2 + panX;
		const oy = (canvas.clientHeight - h * baseScale) / 2 + panY;
		return [(cx - rect.left - ox) / es, (cy - rect.top - oy) / es];
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
		const rect = canvas.getBoundingClientRect();
		const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const w = refSlice?.w ?? 1, h = refSlice?.h ?? 1;
		const totalW = w * 2 + 4;
		const cxOff = (dw - totalW * baseScale) / 2;
		const cyOff = (dh - h * baseScale) / 2;
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

	function pct(count: number, total: number): string {
		if (total === 0) return '0%';
		return (count / total * 100).toFixed(3) + '%';
	}
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
		<label class="flex items-center gap-1.5 text-gray-400">
			<input type="checkbox" bind:checked={showDenormals} /> Show denormals
		</label>
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Filter:</span>
			<select use:rangeScroll bind:value={filterClass} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				<option value="all">All classes</option>
				<option value="nan">NaN only</option>
				<option value="inf">+Inf only</option>
				<option value="negInf">-Inf only</option>
				<option value="denorm">Denormal only</option>
			</select>
		</label>
		<button
			class="px-2 py-0.5 rounded text-xs border border-edge text-gray-300 hover:bg-surface-overlay disabled:opacity-40 disabled:cursor-not-allowed"
			disabled={!firstAnomaly}
			onclick={jumpToFirstAnomaly}
		>Jump to first anomaly</button>
	</div>

	<!-- Legend + Stats -->
	<div class="flex gap-6 text-xs">
		<div class="flex flex-col gap-1">
			<span class="text-gray-500 font-medium">{refLabel}</span>
			{#if refScan.nan > 0}<span class="text-red-400">NaN: {refScan.nan} ({pct(refScan.nan, refScan.total)})</span>{/if}
			{#if refScan.inf > 0}<span class="text-fuchsia-400">+Inf: {refScan.inf} ({pct(refScan.inf, refScan.total)})</span>{/if}
			{#if refScan.ninf > 0}<span class="text-cyan-400">-Inf: {refScan.ninf} ({pct(refScan.ninf, refScan.total)})</span>{/if}
			{#if refScan.denormal > 0}<span class="text-yellow-400">Denormal: {refScan.denormal} ({pct(refScan.denormal, refScan.total)})</span>{/if}
			{#if refScan.nan === 0 && refScan.inf === 0 && refScan.ninf === 0 && refScan.denormal === 0}
				<span class="text-green-400">Clean (no special values)</span>
			{/if}
		</div>
		<div class="flex flex-col gap-1">
			<span class="text-gray-500 font-medium">{mainLabel}</span>
			{#if mainScan.nan > 0}<span class="text-red-400">NaN: {mainScan.nan} ({pct(mainScan.nan, mainScan.total)})</span>{/if}
			{#if mainScan.inf > 0}<span class="text-fuchsia-400">+Inf: {mainScan.inf} ({pct(mainScan.inf, mainScan.total)})</span>{/if}
			{#if mainScan.ninf > 0}<span class="text-cyan-400">-Inf: {mainScan.ninf} ({pct(mainScan.ninf, mainScan.total)})</span>{/if}
			{#if mainScan.denormal > 0}<span class="text-yellow-400">Denormal: {mainScan.denormal} ({pct(mainScan.denormal, mainScan.total)})</span>{/if}
			{#if mainScan.nan === 0 && mainScan.inf === 0 && mainScan.ninf === 0 && mainScan.denormal === 0}
				<span class="text-green-400">Clean (no special values)</span>
			{/if}
		</div>
		<div class="flex flex-col gap-1">
			<span class="text-gray-500 font-medium">Legend</span>
			<div class="flex gap-3">
				<span class="flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background: rgb(255,50,50)"></span> NaN</span>
				<span class="flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background: rgb(255,50,255)"></span> +Inf</span>
				<span class="flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background: rgb(50,220,255)"></span> -Inf</span>
				<span class="flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-sm" style="background: rgb(255,220,50)"></span> Denormal</span>
			</div>
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

	{#if anomalyList.length > 0}
		<div class="max-h-[120px] overflow-y-auto rounded border border-edge bg-surface-base text-xs font-mono">
			<table class="w-full">
				<thead class="sticky top-0 bg-surface-overlay">
					<tr class="text-gray-500">
						<th class="px-2 py-0.5 text-left">Source</th>
						<th class="px-2 py-0.5 text-left">Batch</th>
						<th class="px-2 py-0.5 text-left">Ch</th>
						<th class="px-2 py-0.5 text-left">Y</th>
						<th class="px-2 py-0.5 text-left">X</th>
						<th class="px-2 py-0.5 text-left">Value</th>
						<th class="px-2 py-0.5 text-left">Class</th>
					</tr>
				</thead>
				<tbody>
					{#each anomalyList as entry}
						<tr
							class="hover:bg-surface-overlay cursor-pointer text-gray-300"
							onclick={() => jumpToAnomaly(entry.batch, entry.channel, entry.y, entry.x)}
						>
							<td class="px-2 py-0.5" class:text-blue-400={entry.source === 'ref'} class:text-red-400={entry.source === 'main'}>{entry.source}</td>
							<td class="px-2 py-0.5">{entry.batch}</td>
							<td class="px-2 py-0.5">{entry.channel}</td>
							<td class="px-2 py-0.5">{entry.y}</td>
							<td class="px-2 py-0.5">{entry.x}</td>
							<td class="px-2 py-0.5">{formatValue(entry.value)}</td>
							<td class="px-2 py-0.5"
								class:text-red-400={entry.cls === 'NaN'}
								class:text-fuchsia-400={entry.cls === '+Inf'}
								class:text-cyan-400={entry.cls === '-Inf'}
								class:text-yellow-400={entry.cls === 'Denorm'}
							>{entry.cls}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}

	{#if showTooltip && hoverX >= 0 && hoverY >= 0 && refSlice}
		{@const idx = hoverY * refSlice.w + hoverX}
		{@const rv = refSlice.data[idx]}
		{@const mv = mainSlice.data[idx]}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="font-mono text-gray-400">[{hoverY}, {hoverX}]</div>
			<div><span class="text-blue-400">{refLabel}:</span> {formatValue(rv)} <span class="text-gray-500">({classify(rv)})</span></div>
			<div><span class="text-red-400">{mainLabel}:</span> {formatValue(mv)} <span class="text-gray-500">({classify(mv)})</span></div>
		</div>
	{/if}
</div>
