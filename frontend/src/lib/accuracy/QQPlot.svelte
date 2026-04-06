<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		formatValue,
		computeKSStatistic,
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
	let numQuantiles = $state(200);

	let hoverX = $state(-1);
	let hoverY = $state(-1);
	let showTooltip = $state(false);
	let tooltipScreenX = $state(0);
	let tooltipScreenY = $state(0);

	let dims = $derived(getSpatialDims(shape));

	// Get data
	let plotData = $derived.by(() => {
		let refData: Float32Array, mainData: Float32Array;
		if (channel === -1) {
			const batchSize = dims.channels * dims.height * dims.width;
			const offset = batch * batchSize;
			refData = ref.subarray(offset, offset + batchSize);
			mainData = main.subarray(offset, offset + batchSize);
		} else {
			const rSlice = extractSlice(ref, shape, batch, channel);
			const mSlice = extractSlice(main, shape, batch, channel);
			refData = rSlice.data;
			mainData = mSlice.data;
		}

		// Sort both
		const sortedRef = Float32Array.from(refData).sort();
		const sortedMain = Float32Array.from(mainData).sort();

		// Sample quantiles
		const n = Math.min(numQuantiles, sortedRef.length);
		const refQ = new Float32Array(n);
		const mainQ = new Float32Array(n);
		for (let i = 0; i < n; i++) {
			const p = i / (n - 1);
			const idx = Math.min(sortedRef.length - 1, Math.floor(p * sortedRef.length));
			refQ[i] = sortedRef[idx];
			mainQ[i] = sortedMain[Math.min(sortedMain.length - 1, Math.floor(p * sortedMain.length))];
		}

		let lo = Infinity, hi = -Infinity;
		for (let i = 0; i < n; i++) {
			if (refQ[i] < lo) lo = refQ[i];
			if (refQ[i] > hi) hi = refQ[i];
			if (mainQ[i] < lo) lo = mainQ[i];
			if (mainQ[i] > hi) hi = mainQ[i];
		}
		if (lo === hi) { lo -= 1; hi += 1; }

		return { refQ, mainQ, n, lo, hi };
	});

	let ksValue = $derived.by(() => {
		let refData: Float32Array, mainData: Float32Array;
		if (channel === -1) {
			const batchSize = dims.channels * dims.height * dims.width;
			const offset = batch * batchSize;
			refData = ref.subarray(offset, offset + batchSize);
			mainData = main.subarray(offset, offset + batchSize);
		} else {
			const rSlice = extractSlice(ref, shape, batch, channel);
			const mSlice = extractSlice(main, shape, batch, channel);
			refData = rSlice.data;
			mainData = mSlice.data;
		}
		return computeKSStatistic(refData, mainData);
	});

	function redraw() {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const { refQ, mainQ, n, lo, hi } = plotData;
		const margin = { top: 20, right: 20, bottom: 40, left: 50 };
		const plotW = dw - margin.left - margin.right;
		const plotH = dh - margin.top - margin.bottom;
		const span = hi - lo;

		ctx.clearRect(0, 0, dw, dh);

		// Background
		ctx.fillStyle = '#1a1a2e';
		ctx.fillRect(margin.left, margin.top, plotW, plotH);

		// Diagonal (identity line)
		ctx.strokeStyle = 'rgba(255,255,255,0.2)';
		ctx.lineWidth = 1;
		ctx.setLineDash([4, 4]);
		ctx.beginPath();
		ctx.moveTo(margin.left, margin.top + plotH);
		ctx.lineTo(margin.left + plotW, margin.top);
		ctx.stroke();
		ctx.setLineDash([]);

		// 95% confidence bands: diagonal ± 1.36 / sqrt(n)
		if (n > 1) {
			const ksEnvelope = 1.36 / Math.sqrt(n);
			const envPixels = (ksEnvelope / span) * plotH;
			ctx.strokeStyle = 'rgba(255, 200, 60, 0.3)';
			ctx.lineWidth = 1;
			ctx.setLineDash([3, 3]);

			// Upper band (diagonal + offset)
			ctx.beginPath();
			ctx.moveTo(margin.left, margin.top + plotH - envPixels);
			ctx.lineTo(margin.left + plotW, margin.top - envPixels);
			ctx.stroke();

			// Lower band (diagonal - offset)
			ctx.beginPath();
			ctx.moveTo(margin.left, margin.top + plotH + envPixels);
			ctx.lineTo(margin.left + plotW, margin.top + envPixels);
			ctx.stroke();

			ctx.setLineDash([]);
		}

		// Plot points — color by deviation from identity line
		let maxDev = 0;
		for (let i = 0; i < n; i++) {
			const d = Math.abs(refQ[i] - mainQ[i]);
			if (d > maxDev) maxDev = d;
		}
		if (maxDev === 0) maxDev = 1;

		for (let i = 0; i < n; i++) {
			const x = margin.left + ((refQ[i] - lo) / span) * plotW;
			const y = margin.top + plotH - ((mainQ[i] - lo) / span) * plotH;
			const t = Math.min(1, Math.abs(refQ[i] - mainQ[i]) / maxDev);
			const r = Math.round(100 + (250 - 100) * t);
			const g = Math.round(200 + (80 - 200) * t);
			const b = Math.round(100 + (80 - 100) * t);
			ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
			ctx.beginPath();
			ctx.arc(x, y, 2, 0, Math.PI * 2);
			ctx.fill();
		}

		// Connect with line
		ctx.strokeStyle = 'rgba(96,165,250,0.5)';
		ctx.lineWidth = 1;
		ctx.beginPath();
		for (let i = 0; i < n; i++) {
			const x = margin.left + ((refQ[i] - lo) / span) * plotW;
			const y = margin.top + plotH - ((mainQ[i] - lo) / span) * plotH;
			if (i === 0) ctx.moveTo(x, y);
			else ctx.lineTo(x, y);
		}
		ctx.stroke();

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
		ctx.fillText(`${refLabel} quantiles`, margin.left + plotW / 2, dh - 14);

		ctx.save();
		ctx.translate(12, margin.top + plotH / 2);
		ctx.rotate(-Math.PI / 2);
		ctx.textAlign = 'center';
		ctx.fillText(`${mainLabel} quantiles`, 0, 0);
		ctx.restore();

		// Tick labels
		ctx.fillStyle = '#888';
		ctx.font = '9px monospace';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'top';
		ctx.fillText(formatValue(lo), margin.left, margin.top + plotH + 4);
		ctx.fillText(formatValue(hi), margin.left + plotW, margin.top + plotH + 4);

		ctx.textAlign = 'right';
		ctx.textBaseline = 'middle';
		ctx.fillText(formatValue(lo), margin.left - 4, margin.top + plotH);
		ctx.fillText(formatValue(hi), margin.left - 4, margin.top);

		// Crosshair
		if (showTooltip && hoverX >= 0) {
			const i = Math.min(n - 1, Math.max(0, hoverX));
			const x = margin.left + ((refQ[i] - lo) / span) * plotW;
			const y = margin.top + plotH - ((mainQ[i] - lo) / span) * plotH;
			ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1;
			ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + plotH); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + plotW, y); ctx.stroke();
			ctx.fillStyle = '#fff';
			ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2); ctx.fill();
		}
	}

	$effect(() => { plotData; ksValue; showTooltip; hoverX; redraw(); });

	function handleMouseMove(e: MouseEvent) {
		if (!canvas) return;
		const rect = canvas.getBoundingClientRect();
		const mx = e.clientX - rect.left;
		const dw = canvas.clientWidth;
		const margin = { left: 50, right: 20 };
		const plotW = dw - margin.left - margin.right;
		const t = (mx - margin.left) / plotW;
		if (t >= 0 && t <= 1) {
			hoverX = Math.round(t * (plotData.n - 1));
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
			<span class="text-gray-400">Channel:</span>
			<select use:rangeScroll bind:value={channel} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				<option value={-1}>All</option>
				{#each Array(dims.channels) as _, i}
					<option value={i}>{i}</option>
				{/each}
			</select>
		</label>
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Points:</span>
			<select use:rangeScroll bind:value={numQuantiles} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				<option value={100}>100</option>
				<option value={200}>200</option>
				<option value={500}>500</option>
				<option value={1000}>1000</option>
			</select>
		</label>
	</div>

	<div class="text-xs text-gray-500">
		Diagonal = identical distribution. Deviations show: shift (offset), scale (slope change), tail differences (curvature at ends)
		<span class="ml-2 text-gray-400">KS = {ksValue.toFixed(6)}</span>
	</div>

	<div class="flex-1 bg-surface-base rounded-lg p-2 overflow-hidden min-h-0">
		<canvas
			bind:this={canvas}
			class="w-full h-full cursor-crosshair"
			onmousemove={handleMouseMove}
			onmouseleave={handleMouseLeave}
		></canvas>
	</div>

	{#if showTooltip && hoverX >= 0}
		{@const { refQ, mainQ } = plotData}
		{@const i = Math.min(plotData.n - 1, Math.max(0, hoverX))}
		{@const percentile = (i / (plotData.n - 1) * 100).toFixed(1)}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="text-gray-400">P{percentile}</div>
			<div><span class="text-blue-400">{refLabel}:</span> {formatValue(refQ[i])}</div>
			<div><span class="text-red-400">{mainLabel}:</span> {formatValue(mainQ[i])}</div>
			<div><span class="text-gray-400">Diff:</span> {formatValue(mainQ[i] - refQ[i])}</div>
		</div>
	{/if}
</div>
