<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		computeStats,
		formatValue,
		COLORMAPS,
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
	let colormap: ColormapName = $state('coolwarm');
	let maxChannels = $state(64); // Limit for performance

	let hoverX = $state(-1);
	let hoverY = $state(-1);
	let showTooltip = $state(false);
	let tooltipScreenX = $state(0);
	let tooltipScreenY = $state(0);

	let dims = $derived(getSpatialDims(shape));
	let numChannels = $derived(Math.min(dims.channels, maxChannels));

	// Compute error vectors per channel, then correlation matrix
	let corrMatrix = $derived.by(() => {
		const n = numChannels;
		const spatialSize = dims.height * dims.width;

		// Compute per-channel error vectors
		const errors: Float32Array[] = [];
		for (let c = 0; c < n; c++) {
			const rSlice = extractSlice(ref, shape, batch, c);
			const mSlice = extractSlice(main, shape, batch, c);
			const err = new Float32Array(spatialSize);
			for (let i = 0; i < spatialSize; i++) {
				err[i] = mSlice.data[i] - rSlice.data[i];
			}
			errors.push(err);
		}

		// Compute means and stds
		const means = new Float32Array(n);
		const stds = new Float32Array(n);
		for (let c = 0; c < n; c++) {
			const stats = computeStats(errors[c]);
			means[c] = stats.mean;
			stds[c] = stats.std;
		}

		// Pearson correlation matrix
		const matrix = new Float32Array(n * n);
		for (let i = 0; i < n; i++) {
			for (let j = i; j < n; j++) {
				if (i === j) {
					matrix[i * n + j] = 1;
					continue;
				}
				let sum = 0;
				for (let k = 0; k < spatialSize; k++) {
					sum += (errors[i][k] - means[i]) * (errors[j][k] - means[j]);
				}
				const corr = (stds[i] > 0 && stds[j] > 0)
					? sum / (spatialSize * stds[i] * stds[j])
					: 0;
				matrix[i * n + j] = corr;
				matrix[j * n + i] = corr;
			}
		}

		return matrix;
	});

	function redraw() {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const n = numChannels;
		const margin = { top: 30, right: 10, bottom: 10, left: 30 };
		const plotW = dw - margin.left - margin.right;
		const plotH = dh - margin.top - margin.bottom;
		const cellW = plotW / n;
		const cellH = plotH / n;

		ctx.clearRect(0, 0, dw, dh);

		const lut = COLORMAPS[colormap];

		for (let i = 0; i < n; i++) {
			for (let j = 0; j < n; j++) {
				const v = corrMatrix[i * n + j];
				// Map [-1, 1] to [0, 255]
				const t = (v + 1) / 2;
				const idx = Math.max(0, Math.min(255, Math.round(t * 255)));
				const base = idx * 3;
				ctx.fillStyle = `rgb(${lut[base]},${lut[base + 1]},${lut[base + 2]})`;
				ctx.fillRect(margin.left + j * cellW, margin.top + i * cellH, Math.ceil(cellW), Math.ceil(cellH));
			}
		}

		// Labels (show every Nth)
		const labelStep = Math.max(1, Math.floor(n / 16));
		ctx.fillStyle = '#888';
		ctx.font = '9px monospace';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'bottom';
		for (let i = 0; i < n; i += labelStep) {
			ctx.fillText(`${i}`, margin.left + (i + 0.5) * cellW, margin.top - 2);
		}
		ctx.textAlign = 'right';
		ctx.textBaseline = 'middle';
		for (let i = 0; i < n; i += labelStep) {
			ctx.fillText(`${i}`, margin.left - 2, margin.top + (i + 0.5) * cellH);
		}

		// Highlight hover
		if (showTooltip && hoverX >= 0 && hoverY >= 0 && hoverX < n && hoverY < n) {
			ctx.strokeStyle = 'rgba(255,255,255,0.7)';
			ctx.lineWidth = 2;
			ctx.strokeRect(
				margin.left + hoverX * cellW,
				margin.top + hoverY * cellH,
				cellW,
				cellH,
			);
		}
	}

	$effect(() => { corrMatrix; colormap; showTooltip; hoverX; hoverY; redraw(); });

	function handleMouseMove(e: MouseEvent) {
		if (!canvas) return;
		const rect = canvas.getBoundingClientRect();
		const mx = e.clientX - rect.left;
		const my = e.clientY - rect.top;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const margin = { top: 30, right: 10, bottom: 10, left: 30 };
		const plotW = dw - margin.left - margin.right;
		const plotH = dh - margin.top - margin.bottom;
		const n = numChannels;
		const col = Math.floor((mx - margin.left) / (plotW / n));
		const row = Math.floor((my - margin.top) / (plotH / n));
		if (col >= 0 && col < n && row >= 0 && row < n) {
			hoverX = col; hoverY = row;
			tooltipScreenX = e.clientX; tooltipScreenY = e.clientY;
			showTooltip = true;
		} else { showTooltip = false; }
	}
	function handleMouseLeave() { showTooltip = false; }
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
			<span class="text-gray-400">Max channels:</span>
			<select use:rangeScroll bind:value={maxChannels} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				<option value={32}>32</option>
				<option value={64}>64</option>
				<option value={128}>128</option>
				<option value={256}>256</option>
			</select>
		</label>
		<span class="text-gray-500">
			{numChannels}x{numChannels} correlation matrix of error vectors
			{#if dims.channels > maxChannels}
				<span class="text-yellow-400">(showing first {maxChannels} of {dims.channels})</span>
			{/if}
		</span>
	</div>

	<div class="text-xs text-gray-500">
		Pearson correlation of error (main-ref) between channel pairs. Strong correlation = shared error source.
	</div>

	<div class="flex-1 bg-surface-base rounded-lg p-2 overflow-hidden min-h-0">
		<canvas
			bind:this={canvas}
			class="w-full h-full cursor-crosshair"
			onmousemove={handleMouseMove}
			onmouseleave={handleMouseLeave}
		></canvas>
	</div>

	{#if showTooltip && hoverX >= 0 && hoverY >= 0 && hoverX < numChannels && hoverY < numChannels}
		{@const corr = corrMatrix[hoverY * numChannels + hoverX]}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="text-gray-400">Channel {hoverY} x Channel {hoverX}</div>
			<div><span class="text-gray-400">Correlation:</span> <span class={Math.abs(corr) > 0.7 ? 'text-yellow-400' : 'text-gray-300'}>{corr.toFixed(4)}</span></div>
		</div>
	{/if}
</div>
