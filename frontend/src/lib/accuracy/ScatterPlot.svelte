<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		computeStats,
		computeHistogram,
		formatValue,
		computeR2,
		COLORMAPS,
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
	let channel = $state(-1); // -1 = all channels
	let logAxes = $state(false);
	let resolution = $state(256); // 2D histogram bins
	let colormap: ColormapName = $state('inferno');

	let hoverX = $state(-1);
	let hoverY = $state(-1);
	let showTooltip = $state(false);
	let tooltipScreenX = $state(0);
	let tooltipScreenY = $state(0);

	let dims = $derived(getSpatialDims(shape));

	// Get data for scatter
	let scatterData = $derived.by(() => {
		if (channel === -1) {
			// All channels in this batch
			const batchSize = dims.channels * dims.height * dims.width;
			const offset = batch * batchSize;
			return {
				refData: ref.subarray(offset, offset + batchSize),
				mainData: main.subarray(offset, offset + batchSize),
			};
		}
		const rSlice = extractSlice(ref, shape, batch, channel);
		const mSlice = extractSlice(main, shape, batch, channel);
		return { refData: rSlice.data, mainData: mSlice.data };
	});

	// Transform data for log axes
	let plotData = $derived.by(() => {
		const { refData, mainData } = scatterData;
		if (!logAxes) return { refData, mainData };
		const eps = 1e-10;
		const rOut = new Float32Array(refData.length);
		const mOut = new Float32Array(mainData.length);
		for (let i = 0; i < refData.length; i++) {
			rOut[i] = Math.log10(Math.abs(refData[i]) + eps);
			mOut[i] = Math.log10(Math.abs(mainData[i]) + eps);
		}
		return { refData: rOut, mainData: mOut };
	});

	// Compute 2D histogram for density coloring
	let histogram2D = $derived.by(() => {
		const { refData, mainData } = plotData;
		const refStats = computeStats(refData);
		const mainStats = computeStats(mainData);

		let rMin = Math.min(refStats.min, mainStats.min);
		let rMax = Math.max(refStats.max, mainStats.max);
		if (rMin === rMax) { rMin -= 1; rMax += 1; }

		const bins = resolution;
		const hist = new Uint32Array(bins * bins);
		const span = rMax - rMin || 1;

		// Also compute marginals
		const margRef = new Uint32Array(bins);
		const margMain = new Uint32Array(bins);

		for (let i = 0; i < refData.length; i++) {
			let rx = refData[i], my = mainData[i];
			if (!isFinite(rx) || !isFinite(my)) continue;

			let bx = Math.floor((rx - rMin) / span * (bins - 1));
			let by = Math.floor((my - rMin) / span * (bins - 1));
			bx = Math.max(0, Math.min(bins - 1, bx));
			by = Math.max(0, Math.min(bins - 1, by));
			// Flip Y so low values are at bottom
			hist[(bins - 1 - by) * bins + bx]++;
			margRef[bx]++;
			margMain[bins - 1 - by]++;
		}

		let maxCount = 0;
		for (let i = 0; i < hist.length; i++) {
			if (hist[i] > maxCount) maxCount = hist[i];
		}

		let maxMargRef = 0, maxMargMain = 0;
		for (let i = 0; i < bins; i++) {
			if (margRef[i] > maxMargRef) maxMargRef = margRef[i];
			if (margMain[i] > maxMargMain) maxMargMain = margMain[i];
		}

		return { hist, maxCount, rMin, rMax, bins, margRef, margMain, maxMargRef, maxMargMain };
	});

	// Best-fit line (least squares)
	let bestFit = $derived.by(() => {
		const { refData, mainData } = plotData;
		let n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
		for (let i = 0; i < refData.length; i++) {
			const x = refData[i], y = mainData[i];
			if (!isFinite(x) || !isFinite(y)) continue;
			sx += x; sy += y; sxx += x * x; sxy += x * y; n++;
		}
		if (n < 2) return null;
		const denom = n * sxx - sx * sx;
		if (denom === 0) return null;
		const slope = (n * sxy - sx * sy) / denom;
		const intercept = (sy - slope * sx) / n;
		return { slope, intercept };
	});

	// R-squared
	let r2 = $derived(computeR2(plotData.refData, plotData.mainData));

	// Stats
	let correlation = $derived.by(() => {
		const { refData, mainData } = scatterData;
		let sumR = 0, sumM = 0, sumR2 = 0, sumM2 = 0, sumRM = 0, n = 0;
		for (let i = 0; i < refData.length; i++) {
			const r = refData[i], m = mainData[i];
			if (!isFinite(r) || !isFinite(m)) continue;
			sumR += r; sumM += m; sumR2 += r * r; sumM2 += m * m; sumRM += r * m;
			n++;
		}
		if (n === 0) return 0;
		const num = n * sumRM - sumR * sumM;
		const den = Math.sqrt((n * sumR2 - sumR * sumR) * (n * sumM2 - sumM * sumM));
		return den === 0 ? 0 : num / den;
	});

	function redraw() {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const { hist, maxCount, rMin, rMax, bins } = histogram2D;

		const margin = { top: 20, right: 20, bottom: 40, left: 50 };
		const plotW = dw - margin.left - margin.right;
		const plotH = dh - margin.top - margin.bottom;

		ctx.clearRect(0, 0, dw, dh);

		// Draw 2D histogram
		const lut = COLORMAPS[colormap];
		const cellW = plotW / bins;
		const cellH = plotH / bins;
		const logMax = Math.log(maxCount + 1);

		for (let row = 0; row < bins; row++) {
			for (let col = 0; col < bins; col++) {
				const count = hist[row * bins + col];
				if (count === 0) continue;
				const t = Math.log(count + 1) / logMax; // log density
				const idx = Math.min(255, Math.round(t * 255));
				const base = idx * 3;
				ctx.fillStyle = `rgb(${lut[base]},${lut[base + 1]},${lut[base + 2]})`;
				ctx.fillRect(
					margin.left + col * cellW,
					margin.top + row * cellH,
					Math.ceil(cellW),
					Math.ceil(cellH),
				);
			}
		}

		// Marginal histograms
		const margH = 24;
		const { margRef, margMain, maxMargRef, maxMargMain } = histogram2D;
		// Top marginal (ref distribution)
		ctx.fillStyle = 'rgba(96, 165, 250, 0.3)';
		for (let i = 0; i < bins; i++) {
			const barH = maxMargRef > 0 ? (margRef[i] / maxMargRef) * margH : 0;
			ctx.fillRect(margin.left + i * cellW, margin.top - barH, Math.ceil(cellW), barH);
		}
		// Right marginal (main distribution)
		ctx.fillStyle = 'rgba(248, 113, 113, 0.3)';
		for (let i = 0; i < bins; i++) {
			const barW = maxMargMain > 0 ? (margMain[i] / maxMargMain) * margH : 0;
			ctx.fillRect(margin.left + plotW, margin.top + i * cellH, barW, Math.ceil(cellH));
		}

		// Diagonal line (perfect agreement)
		ctx.strokeStyle = 'rgba(255,255,255,0.3)';
		ctx.lineWidth = 1;
		ctx.setLineDash([4, 4]);
		ctx.beginPath();
		ctx.moveTo(margin.left, margin.top + plotH);
		ctx.lineTo(margin.left + plotW, margin.top);
		ctx.stroke();
		ctx.setLineDash([]);

		// Best-fit line
		if (bestFit) {
			const { slope, intercept } = bestFit;
			const { rMin: bMin, rMax: bMax } = histogram2D;
			const span = bMax - bMin || 1;
			// Convert data space to plot space
			const y0 = slope * bMin + intercept;
			const y1 = slope * bMax + intercept;
			const px0 = margin.left;
			const px1 = margin.left + plotW;
			const py0 = margin.top + plotH - ((y0 - bMin) / span) * plotH;
			const py1 = margin.top + plotH - ((y1 - bMin) / span) * plotH;
			ctx.strokeStyle = 'rgba(250, 204, 21, 0.6)';
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			ctx.moveTo(px0, py0);
			ctx.lineTo(px1, py1);
			ctx.stroke();
		}

		// Axes
		ctx.strokeStyle = '#666';
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(margin.left, margin.top);
		ctx.lineTo(margin.left, margin.top + plotH);
		ctx.lineTo(margin.left + plotW, margin.top + plotH);
		ctx.stroke();

		// Labels
		ctx.fillStyle = '#999';
		ctx.font = '11px monospace';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'top';
		ctx.fillText(`${refLabel} value`, margin.left + plotW / 2, dh - 14);

		ctx.save();
		ctx.translate(12, margin.top + plotH / 2);
		ctx.rotate(-Math.PI / 2);
		ctx.textAlign = 'center';
		ctx.textBaseline = 'top';
		ctx.fillText(`${mainLabel} value`, 0, 0);
		ctx.restore();

		// Tick labels
		ctx.fillStyle = '#888';
		ctx.font = '9px monospace';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'top';
		ctx.fillText(formatValue(rMin), margin.left, margin.top + plotH + 4);
		ctx.fillText(formatValue(rMax), margin.left + plotW, margin.top + plotH + 4);

		ctx.textAlign = 'right';
		ctx.textBaseline = 'middle';
		ctx.fillText(formatValue(rMin), margin.left - 4, margin.top + plotH);
		ctx.fillText(formatValue(rMax), margin.left - 4, margin.top);

		// Crosshair
		if (showTooltip && hoverX >= 0 && hoverY >= 0) {
			ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1;
			const sx = margin.left + hoverX * cellW, sy = margin.top + hoverY * cellH;
			ctx.beginPath(); ctx.moveTo(sx, margin.top); ctx.lineTo(sx, margin.top + plotH); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(margin.left, sy); ctx.lineTo(margin.left + plotW, sy); ctx.stroke();
		}
	}

	$effect(() => { histogram2D; colormap; logAxes; bestFit; showTooltip; hoverX; hoverY; redraw(); });

	function handleMouseMove(e: MouseEvent) {
		if (!canvas) return;
		const rect = canvas.getBoundingClientRect();
		const mx = e.clientX - rect.left;
		const my = e.clientY - rect.top;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const margin = { top: 20, right: 20, bottom: 40, left: 50 };
		const plotW = dw - margin.left - margin.right;
		const plotH = dh - margin.top - margin.bottom;
		const bins = histogram2D.bins;
		const cellW = plotW / bins;
		const cellH = plotH / bins;
		const col = Math.floor((mx - margin.left) / cellW);
		const row = Math.floor((my - margin.top) / cellH);

		if (col >= 0 && col < bins && row >= 0 && row < bins) {
			hoverX = col; hoverY = row;
			tooltipScreenX = e.clientX; tooltipScreenY = e.clientY;
			showTooltip = true;
		} else {
			showTooltip = false;
		}
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
			<span class="text-gray-400">Channel:</span>
			<select use:rangeScroll bind:value={channel} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				<option value={-1}>All</option>
				{#each Array(dims.channels) as _, i}
					<option value={i}>{i}</option>
				{/each}
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
			<input type="checkbox" bind:checked={logAxes} /> Log axes
		</label>
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Resolution:</span>
			<select use:rangeScroll bind:value={resolution} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				<option value={128}>128</option>
				<option value={256}>256</option>
				<option value={512}>512</option>
			</select>
		</label>
	</div>

	<div class="flex gap-4 text-xs text-gray-400">
		<span>Pearson r = {correlation.toFixed(6)}</span>
		<span>R² = {r2.toFixed(6)}</span>
		{#if bestFit}
			<span class="text-yellow-400/60">fit: y = {bestFit.slope.toFixed(4)}x + {formatValue(bestFit.intercept)}</span>
		{/if}
		<span>{scatterData.refData.length.toLocaleString()} elements</span>
		<span class="text-gray-500">{logAxes ? 'Log-scale axes' : 'Diagonal = perfect agreement'}</span>
	</div>

	<div class="flex-1 bg-surface-base rounded-lg p-2 overflow-hidden min-h-0">
		<canvas
			bind:this={canvas}
			class="w-full h-full cursor-crosshair"
			onmousemove={handleMouseMove}
			onmouseleave={handleMouseLeave}
		></canvas>
	</div>

	{#if showTooltip && hoverX >= 0 && hoverY >= 0}
		{@const { rMin, rMax, bins, hist } = histogram2D}
		{@const span = rMax - rMin}
		{@const refVal = rMin + (hoverX / bins) * span}
		{@const mainVal = rMin + ((bins - 1 - hoverY) / bins) * span}
		{@const count = hist[hoverY * bins + hoverX]}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div><span class="text-blue-400">{refLabel}:</span> ~{formatValue(refVal)}</div>
			<div><span class="text-red-400">{mainLabel}:</span> ~{formatValue(mainVal)}</div>
			<div><span class="text-gray-400">Count:</span> {count}</div>
		</div>
	{/if}
</div>
