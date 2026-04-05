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
		const arr = [...channelErrors];
		if (sortBy === 'error') arr.sort((a, b) => b.mse - a.mse);
		else if (sortBy === 'activation') arr.sort((a, b) => b.meanAct - a.meanAct);
		// else keep index order
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

	function redraw() {
		if (!canvas || dims.channels === 0) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

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
			let sliceData: Float32Array;
			if (showMode === 'diff') {
				const mSlice = extractSlice(main, shape, batch, ch.idx);
				const rSlice = extractSlice(ref, shape, batch, ch.idx);
				sliceData = new Float32Array(mSlice.data.length);
				for (let j = 0; j < sliceData.length; j++) sliceData[j] = Math.abs(mSlice.data[j] - rSlice.data[j]);
			} else {
				const tensor = showMode === 'ref' ? ref : main;
				sliceData = extractSlice(tensor, shape, batch, ch.idx).data;
			}

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
		}
	}

	$effect(() => { sortedChannels; showMode; colormap; globalNorm; globalRange; selectedChannel; redraw(); });

	function handleClick(e: MouseEvent) {
		if (!canvas) return;
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

		if (idx >= 0 && idx < sortedChannels.length) {
			selectedChannel = selectedChannel === sortedChannels[idx].idx ? null : sortedChannels[idx].idx;
		}
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
		<span class="text-gray-500">{dims.channels} channels | {gridCols}x{gridRows} grid</span>
	</div>

	{#if selectedChannel !== null}
		{@const ch = channelErrors[selectedChannel]}
		<div class="text-xs text-gray-400 bg-surface-panel p-2 rounded border border-edge">
			<span class="text-content-primary font-medium">Channel {selectedChannel}</span>
			 — MSE: {formatValue(ch.mse)}, Max|Diff|: {formatValue(ch.maxAbs)}, Mean Activation: {formatValue(ch.meanAct)}
			<button class="ml-2 text-gray-500 hover:text-gray-300" onclick={() => selectedChannel = null}>x</button>
		</div>
	{/if}

	<div class="flex-1 bg-surface-base rounded-lg p-2 overflow-hidden min-h-0">
		<canvas
			bind:this={canvas}
			class="w-full h-full cursor-pointer"
			style="image-rendering: pixelated;"
			onclick={handleClick}
		></canvas>
	</div>
</div>
