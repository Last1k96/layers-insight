<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		valueToImageData,
		computeStats,
		formatValue,
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
	let colormap: ColormapName = $state('viridis');
	let showMode: 'ref' | 'main' | 'diff' = $state('diff');
	let sortBy: 'index' | 'error' | 'activation' = $state('index');
	let globalNorm = $state(true);
	let selectedChannel = $state<number | null>(null);

	// Hover stats overlay
	let hoveredCell = $state<number | null>(null);

	// Error threshold filter
	let hideGoodChannels = $state(false);
	let errorThresholdExp = $state(-3);
	let errorThreshold = $derived(Math.pow(10, errorThresholdExp));

	// Double-click zoom
	let zoomedChannel = $state<number | null>(null);
	let zoomPanX = $state(0);
	let zoomPanY = $state(0);
	let zoomLevel = $state(1);
	let zoomDragging = $state(false);
	let zoomDragStartX = 0;
	let zoomDragStartY = 0;
	let zoomPanStartX = 0;
	let zoomPanStartY = 0;

	let dims = $derived(getSpatialDims(shape));

	// Compute per-channel error stats for sorting
	let channelErrors = $derived.by(() => {
		const errors: { idx: number; mse: number; maxAbs: number; meanAct: number }[] = [];
		for (let c = 0; c < dims.channels; c++) {
			const mSlice = extractSlice(main, shape, batch, c);
			const rSlice = extractSlice(ref, shape, batch, c);
			let mse = 0, maxAbs = 0, sumAct = 0;
			for (let i = 0; i < mSlice.data.length; i++) {
				const d = mSlice.data[i] - rSlice.data[i];
				mse += d * d;
				const a = Math.abs(d);
				if (a > maxAbs) maxAbs = a;
				sumAct += Math.abs(mSlice.data[i]);
			}
			mse /= mSlice.data.length || 1;
			errors.push({ idx: c, mse, maxAbs, meanAct: sumAct / (mSlice.data.length || 1) });
		}
		return errors;
	});

	let sortedChannels = $derived.by(() => {
		let arr = [...channelErrors];
		if (sortBy === 'error') arr.sort((a, b) => b.mse - a.mse);
		else if (sortBy === 'activation') arr.sort((a, b) => b.meanAct - a.meanAct);
		// else keep index order
		if (hideGoodChannels) {
			arr = arr.filter(ch => ch.mse >= errorThreshold);
		}
		return arr;
	});

	// Global range across all channels for consistent coloring
	let globalRange = $derived.by((): [number, number] | undefined => {
		if (!globalNorm) return undefined;
		const tensor = showMode === 'ref' ? ref : showMode === 'main' ? main : null;
		if (tensor) {
			const stats = computeStats(tensor);
			return [stats.min, stats.max];
		}
		// For diff mode, compute global diff range
		let lo = Infinity, hi = -Infinity;
		for (let i = 0; i < main.length; i++) {
			const d = Math.abs(main[i] - ref[i]);
			if (d < lo) lo = d;
			if (d > hi) hi = d;
		}
		return [lo, hi];
	});

	// Grid layout
	let gridCols = $derived(Math.ceil(Math.sqrt(dims.channels)));
	let gridRows = $derived(Math.ceil(dims.channels / gridCols));

	function getSliceData(chIdx: number): Float32Array {
		if (showMode === 'diff') {
			const mSlice = extractSlice(main, shape, batch, chIdx);
			const rSlice = extractSlice(ref, shape, batch, chIdx);
			const data = new Float32Array(mSlice.data.length);
			for (let j = 0; j < data.length; j++) data[j] = Math.abs(mSlice.data[j] - rSlice.data[j]);
			return data;
		} else {
			const tensor = showMode === 'ref' ? ref : main;
			return extractSlice(tensor, shape, batch, chIdx).data;
		}
	}

	function redraw() {
		if (!canvas || dims.channels === 0) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		// Zoomed single-channel mode
		if (zoomedChannel !== null) {
			const ch = channelErrors[zoomedChannel];
			if (!ch) return;
			const sliceData = getSliceData(ch.idx);
			const imgData = valueToImageData(sliceData, dims.width, dims.height, colormap, globalRange);
			const offscreen = new OffscreenCanvas(dims.width, dims.height);
			offscreen.getContext('2d')!.putImageData(imgData, 0, 0);

			const baseScale = Math.min(dw / dims.width, dh / dims.height);
			const es = baseScale * zoomLevel;
			const ox = (dw - dims.width * baseScale) / 2 + zoomPanX;
			const oy = (dh - dims.height * baseScale) / 2 + zoomPanY;

			ctx.clearRect(0, 0, dw, dh);
			ctx.imageSmoothingEnabled = false;
			ctx.setTransform(es, 0, 0, es, ox, oy);
			ctx.drawImage(offscreen, 0, 0);
			ctx.resetTransform();

			// Title
			ctx.fillStyle = '#e5e7eb';
			ctx.font = 'bold 12px monospace';
			ctx.textAlign = 'left';
			ctx.textBaseline = 'top';
			ctx.fillText(`Channel ${ch.idx} — MSE: ${formatValue(ch.mse)}, Max|Diff|: ${formatValue(ch.maxAbs)}`, 8, 8);
			return;
		}

		// Grid mode
		const padding = 2;
		const labelH = 12;
		const cellW = Math.floor((dw - padding * (gridCols + 1)) / gridCols);
		const cellH = Math.floor((dh - padding * (gridRows + 1) - labelH * gridRows) / gridRows);
		if (cellW < 2 || cellH < 2) return;

		ctx.clearRect(0, 0, dw, dh);
		ctx.font = '9px monospace';
		ctx.textBaseline = 'top';

		for (let i = 0; i < sortedChannels.length; i++) {
			const ch = sortedChannels[i];
			const col = i % gridCols;
			const row = Math.floor(i / gridCols);
			const x = padding + col * (cellW + padding);
			const y = padding + row * (cellH + padding + labelH);

			// Get slice data
			const sliceData = getSliceData(ch.idx);

			// Render to small ImageData then scale
			const imgData = valueToImageData(sliceData, dims.width, dims.height, colormap, globalRange);
			const offscreen = new OffscreenCanvas(dims.width, dims.height);
			offscreen.getContext('2d')!.putImageData(imgData, 0, 0);

			// Scale to fit cell
			const scaleX = cellW / dims.width;
			const scaleY = cellH / dims.height;
			const scale = Math.min(scaleX, scaleY);
			const drawW = dims.width * scale;
			const drawH = dims.height * scale;
			const drawX = x + (cellW - drawW) / 2;
			const drawY = y + labelH;

			ctx.imageSmoothingEnabled = false;
			ctx.drawImage(offscreen, drawX, drawY, drawW, drawH);

			// Selected highlight
			if (selectedChannel === ch.idx) {
				ctx.strokeStyle = '#60a5fa';
				ctx.lineWidth = 2;
				ctx.strokeRect(x - 1, y - 1, cellW + 2, cellH + labelH + 2);
			}

			// Label
			ctx.fillStyle = ch.mse > 0.01 ? '#f87171' : ch.mse > 0.001 ? '#fbbf24' : '#6ee7b7';
			ctx.textAlign = 'center';
			ctx.fillText(`ch${ch.idx}`, x + cellW / 2, y);

			// Hover stats overlay
			if (hoveredCell === i) {
				ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
				ctx.fillRect(x, y + labelH, cellW, cellH);
				ctx.fillStyle = '#e5e7eb';
				ctx.font = '9px monospace';
				ctx.textAlign = 'center';
				ctx.textBaseline = 'middle';
				const cy = y + labelH + cellH / 2;
				ctx.fillText(`MSE: ${formatValue(ch.mse)}`, x + cellW / 2, cy - 6);
				ctx.fillText(`Max|d|: ${formatValue(ch.maxAbs)}`, x + cellW / 2, cy + 6);
			}
		}
	}

	$effect(() => { sortedChannels; showMode; colormap; globalNorm; globalRange; selectedChannel; hoveredCell; zoomedChannel; zoomLevel; zoomPanX; zoomPanY; redraw(); });

	function hitTestGrid(e: MouseEvent): number | null {
		if (!canvas) return null;
		const rect = canvas.getBoundingClientRect();
		const mx = e.clientX - rect.left;
		const my = e.clientY - rect.top;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;

		const padding = 2;
		const labelH = 12;
		const cellW = Math.floor((dw - padding * (gridCols + 1)) / gridCols);
		const cellH = Math.floor((dh - padding * (gridRows + 1) - labelH * gridRows) / gridRows);

		const col = Math.floor(mx / (cellW + padding));
		const row = Math.floor(my / (cellH + padding + labelH));
		const idx = row * gridCols + col;

		if (idx >= 0 && idx < sortedChannels.length) return idx;
		return null;
	}

	function handleClick(e: MouseEvent) {
		if (zoomedChannel !== null) return; // clicks handled differently in zoom mode
		const idx = hitTestGrid(e);
		if (idx !== null) {
			selectedChannel = selectedChannel === sortedChannels[idx].idx ? null : sortedChannels[idx].idx;
		}
	}

	function handleDblClick(e: MouseEvent) {
		if (zoomedChannel !== null) return;
		const idx = hitTestGrid(e);
		if (idx !== null) {
			zoomedChannel = sortedChannels[idx].idx;
			zoomLevel = 1;
			zoomPanX = 0;
			zoomPanY = 0;
		}
	}

	function handleMouseMove(e: MouseEvent) {
		if (zoomedChannel !== null) {
			if (zoomDragging) {
				zoomPanX = zoomPanStartX + (e.clientX - zoomDragStartX);
				zoomPanY = zoomPanStartY + (e.clientY - zoomDragStartY);
			}
			return;
		}
		const idx = hitTestGrid(e);
		hoveredCell = idx;
	}

	function handleMouseLeave() {
		hoveredCell = null;
		zoomDragging = false;
	}

	function handleZoomWheel(e: WheelEvent) {
		if (zoomedChannel === null) return;
		e.preventDefault();
		const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
		const rect = canvas.getBoundingClientRect();
		const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const baseScale = Math.min(dw / dims.width, dh / dims.height);
		const cxOff = (dw - dims.width * baseScale) / 2;
		const cyOff = (dh - dims.height * baseScale) / 2;
		zoomPanX = cx - (cx - cxOff - zoomPanX) * factor - cxOff;
		zoomPanY = cy - (cy - cyOff - zoomPanY) * factor - cyOff;
		zoomLevel *= factor;
	}

	function handleZoomMouseDown(e: MouseEvent) {
		if (zoomedChannel === null || e.button !== 0) return;
		zoomDragging = true;
		zoomDragStartX = e.clientX; zoomDragStartY = e.clientY;
		zoomPanStartX = zoomPanX; zoomPanStartY = zoomPanY;
	}

	function handleZoomMouseUp() { zoomDragging = false; }

	function resetZoom() {
		zoomedChannel = null;
		zoomLevel = 1;
		zoomPanX = 0;
		zoomPanY = 0;
	}
</script>

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
			<span class="text-gray-400">Show:</span>
			<select use:rangeScroll bind:value={showMode} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				<option value="diff">|Diff|</option>
				<option value="ref">{refLabel}</option>
				<option value="main">{mainLabel}</option>
			</select>
		</label>
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Sort:</span>
			<select use:rangeScroll bind:value={sortBy} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				<option value="index">Channel Index</option>
				<option value="error">Error (MSE)</option>
				<option value="activation">Activation</option>
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
			<input type="checkbox" bind:checked={globalNorm} /> Global norm
		</label>
		<label class="flex items-center gap-1.5 text-gray-400">
			<input type="checkbox" bind:checked={hideGoodChannels} /> Hide low-error
		</label>
		{#if hideGoodChannels}
			<label class="flex items-center gap-2">
				<span class="text-gray-400">MSE &ge;</span>
				<input use:rangeScroll type="range" min="-6" max="-1" step="0.5" bind:value={errorThresholdExp} class="w-20" />
				<span class="text-gray-300 font-mono text-xs">{errorThreshold.toExponential(1)}</span>
			</label>
		{/if}
		<span class="text-gray-500">{sortedChannels.length}{hideGoodChannels ? `/${dims.channels}` : ''} channels | {gridCols}x{gridRows} grid</span>
	</div>

	{#if selectedChannel !== null}
		{@const ch = channelErrors[selectedChannel]}
		<div class="text-xs text-gray-400 bg-surface-panel p-2 rounded border border-edge">
			<span class="text-content-primary font-medium">Channel {selectedChannel}</span>
			 — MSE: {formatValue(ch.mse)}, Max|Diff|: {formatValue(ch.maxAbs)}, Mean Activation: {formatValue(ch.meanAct)}
			<button class="ml-2 text-gray-500 hover:text-gray-300" onclick={() => selectedChannel = null}>x</button>
		</div>
	{/if}

	<div class="flex-1 bg-surface-base rounded-lg p-2 overflow-hidden min-h-0 relative">
		{#if zoomedChannel !== null}
			<div class="absolute top-3 right-3 z-10 flex gap-1">
				<button class="px-2 py-0.5 rounded text-xs border border-edge bg-surface-panel text-gray-300 hover:text-white"
					onclick={() => { zoomLevel = 1; zoomPanX = 0; zoomPanY = 0; }}>Reset Zoom</button>
				<button class="px-2 py-0.5 rounded text-xs border border-edge bg-surface-panel text-gray-300 hover:text-white"
					onclick={resetZoom}>Back to Grid</button>
			</div>
		{/if}
		<canvas
			bind:this={canvas}
			class="w-full h-full"
			class:cursor-pointer={zoomedChannel === null}
			class:cursor-crosshair={zoomedChannel !== null}
			style="image-rendering: pixelated;"
			onclick={handleClick}
			ondblclick={handleDblClick}
			onmousemove={handleMouseMove}
			onmouseleave={handleMouseLeave}
			onwheel={handleZoomWheel}
			onmousedown={handleZoomMouseDown}
			onmouseup={handleZoomMouseUp}
		></canvas>
	</div>
</div>
