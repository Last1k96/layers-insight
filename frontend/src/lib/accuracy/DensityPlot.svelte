<script lang="ts">
	import { COLORMAPS, formatValue, type ColormapName } from './tensorUtils';

	let {
		main,
		ref,
		signed = true,
		colormap = 'viridis' as ColormapName,
		bins = 128,
	}: {
		main: Float32Array;
		ref: Float32Array;
		signed?: boolean;
		colormap?: ColormapName;
		bins?: number;
	} = $props();

	let BINS = $derived(bins);
	let BIN2 = $derived(BINS * BINS);

	// Power-0.4 LUT for perceptual brightness
	const POW04 = new Uint8Array(256);
	for (let i = 0; i < 256; i++) POW04[i] = (Math.pow(i / 255, 0.4) * 255 + 0.5) | 0;

	let canvas: HTMLCanvasElement;
	let canvasSize = $state(0); // tracks resize for reactivity

	// Hover state
	let hoverBinX = $state(-1);
	let hoverBinY = $state(-1);
	let showTooltip = $state(false);
	let tooltipScreenX = $state(0);
	let tooltipScreenY = $state(0);

	// Compute density histogram
	let density = $derived.by(() => {
		const n = main.length;
		let mainMin = Infinity, mainMax = -Infinity;
		let diffMax = 0;
		for (let i = 0; i < n; i++) {
			const mv = main[i];
			if (mv < mainMin) mainMin = mv;
			if (mv > mainMax) mainMax = mv;
			const d = Math.abs(main[i] - ref[i]);
			if (d > diffMax) diffMax = d;
		}
		if (diffMax === 0) diffMax = 1;
		const mainSpan = mainMax - mainMin || 1;
		const invMainSpan = (BINS - 1) / mainSpan;
		const halfBins = BINS / 2;

		const buf = new Float32Array(BIN2);
		for (let i = 0; i < n; i++) {
			const mv = main[i];
			const d = mv - ref[i];
			let xi = ((mv - mainMin) * invMainSpan) | 0;
			let yi: number;
			if (signed) {
				const invSigned = (BINS / 2 - 1) / diffMax;
				yi = ((d * invSigned) + halfBins) | 0;
			} else {
				const invAbs = (BINS - 1) / diffMax;
				yi = BINS - 1 - (((d < 0 ? -d : d) * invAbs) | 0);
			}
			if (xi < 0) xi = 0; else if (xi >= BINS) xi = BINS - 1;
			if (yi < 0) yi = 0; else if (yi >= BINS) yi = BINS - 1;
			buf[yi * BINS + xi]++;
		}

		let maxCount = 1;
		for (let i = 0; i < BIN2; i++) if (buf[i] > maxCount) maxCount = buf[i];

		return { buf, maxCount, mainMin, mainMax, diffMax };
	});

	function redraw() {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const dpr = window.devicePixelRatio || 1;
		const w = canvas.clientWidth;
		const h = canvas.clientHeight;
		if (w === 0 || h === 0) return;
		canvas.width = w * dpr;
		canvas.height = h * dpr;
		ctx.scale(dpr, dpr);

		const lut = COLORMAPS[colormap];
		const { buf, maxCount } = density;
		const inv = 255 / maxCount;

		// Measure Y-axis labels to compute left margin dynamically
		ctx.font = '9px monospace';
		const yTopLabel = signed ? `+${formatValue(density.diffMax)}` : formatValue(density.diffMax);
		const yBotLabel = signed ? `-${formatValue(density.diffMax)}` : '0';
		const yLabelW = Math.max(ctx.measureText(yTopLabel).width, ctx.measureText(yBotLabel).width);
		const margin = { top: 6, right: 2, bottom: 20, left: Math.ceil(yLabelW + 14) };
		lastMargin = margin;
		const plotW = w - margin.left - margin.right;
		const plotH = h - margin.top - margin.bottom;
		const cellW = plotW / BINS;
		const cellH = plotH / BINS;

		// Background
		ctx.fillStyle = '#111827';
		ctx.fillRect(margin.left, margin.top, plotW, plotH);

		// Draw density cells
		for (let row = 0; row < BINS; row++) {
			for (let col = 0; col < BINS; col++) {
				const count = buf[row * BINS + col];
				if (count === 0) continue;
				const lin = Math.min(255, (count * inv) | 0);
				const idx = POW04[lin] * 3;
				ctx.fillStyle = `rgb(${lut[idx]},${lut[idx + 1]},${lut[idx + 2]})`;
				ctx.fillRect(
					margin.left + col * cellW,
					margin.top + row * cellH,
					Math.ceil(cellW),
					Math.ceil(cellH),
				);
			}
		}

		// Crosshair
		if (showTooltip && hoverBinX >= 0 && hoverBinY >= 0) {
			const sx = margin.left + hoverBinX * cellW + cellW / 2;
			const sy = margin.top + hoverBinY * cellH + cellH / 2;
			ctx.strokeStyle = 'rgba(255,255,255,0.3)';
			ctx.lineWidth = 1;
			ctx.beginPath(); ctx.moveTo(sx, margin.top); ctx.lineTo(sx, margin.top + plotH); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(margin.left, sy); ctx.lineTo(margin.left + plotW, sy); ctx.stroke();
		}

		// Axes
		ctx.strokeStyle = '#4b5563';
		ctx.lineWidth = 0.5;
		ctx.beginPath();
		ctx.moveTo(margin.left, margin.top);
		ctx.lineTo(margin.left, margin.top + plotH);
		ctx.lineTo(margin.left + plotW, margin.top + plotH);
		ctx.stroke();

		// Axis labels
		ctx.fillStyle = '#6b7280';
		ctx.font = '9px monospace';
		ctx.textBaseline = 'top';
		ctx.textAlign = 'left';
		ctx.fillText(formatValue(density.mainMin), margin.left, margin.top + plotH + 3);
		ctx.textAlign = 'right';
		ctx.fillText(formatValue(density.mainMax), margin.left + plotW, margin.top + plotH + 3);

		// Y-axis labels
		ctx.textBaseline = 'middle';
		ctx.textAlign = 'right';
		if (signed) {
			ctx.fillText(yTopLabel, margin.left - 4, margin.top);
			ctx.fillText('0', margin.left - 4, margin.top + plotH / 2);
			ctx.fillText(yBotLabel, margin.left - 4, margin.top + plotH);
		} else {
			ctx.fillText(yTopLabel, margin.left - 4, margin.top);
			ctx.fillText('0', margin.left - 4, margin.top + plotH);
		}

		// Center line for signed mode
		if (signed) {
			ctx.strokeStyle = 'rgba(255,255,255,0.15)';
			ctx.setLineDash([3, 3]);
			ctx.beginPath();
			ctx.moveTo(margin.left, margin.top + plotH / 2);
			ctx.lineTo(margin.left + plotW, margin.top + plotH / 2);
			ctx.stroke();
			ctx.setLineDash([]);
		}

		// Title labels
		ctx.fillStyle = '#9ca3af';
		ctx.font = '9px monospace';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'top';
		ctx.fillText('value', margin.left + plotW / 2, h - 9);
		ctx.save();
		ctx.translate(8, margin.top + plotH / 2);
		ctx.rotate(-Math.PI / 2);
		ctx.fillText(signed ? 'diff ±' : '|diff|', 0, 0);
		ctx.restore();
	}

	// ResizeObserver to redraw when canvas container resizes
	$effect(() => {
		if (!canvas) return;
		const obs = new ResizeObserver(() => {
			canvasSize = canvas.clientWidth + canvas.clientHeight;
		});
		obs.observe(canvas);
		return () => obs.disconnect();
	});

	$effect(() => { density; colormap; signed; showTooltip; hoverBinX; hoverBinY; canvasSize; redraw(); });

	// Cache the last computed margin so handleMouseMove uses the same layout as redraw
	let lastMargin = { top: 6, right: 2, bottom: 20, left: 50 };

	function handleMouseMove(e: MouseEvent) {
		if (!canvas) return;
		const rect = canvas.getBoundingClientRect();
		const mx = e.clientX - rect.left;
		const my = e.clientY - rect.top;
		const w = canvas.clientWidth, h = canvas.clientHeight;
		const margin = lastMargin;
		const plotW = w - margin.left - margin.right;
		const plotH = h - margin.top - margin.bottom;
		const cellW = plotW / BINS;
		const cellH = plotH / BINS;
		const col = Math.floor((mx - margin.left) / cellW);
		const row = Math.floor((my - margin.top) / cellH);

		if (col >= 0 && col < BINS && row >= 0 && row < BINS) {
			hoverBinX = col; hoverBinY = row;
			tooltipScreenX = e.clientX; tooltipScreenY = e.clientY;
			showTooltip = true;
		} else {
			showTooltip = false;
		}
	}

	function handleMouseLeave() { showTooltip = false; }
</script>

<div class="relative w-full h-full">
	<canvas
		bind:this={canvas}
		class="w-full h-full cursor-crosshair rounded"
		onmousemove={handleMouseMove}
		onmouseleave={handleMouseLeave}
	></canvas>

	{#if showTooltip && hoverBinX >= 0 && hoverBinY >= 0}
		{@const { mainMin, mainMax, diffMax, buf } = density}
		{@const mainSpan = mainMax - mainMin}
		{@const valLo = mainMin + (hoverBinX / BINS) * mainSpan}
		{@const valHi = mainMin + ((hoverBinX + 1) / BINS) * mainSpan}
		{@const count = buf[hoverBinY * BINS + hoverBinX]}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-2 py-1 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 12}px; top: {tooltipScreenY + 12}px;"
		>
			<div>value: [{formatValue(valLo)}, {formatValue(valHi)}]</div>
			{#if signed}
				{@const diffLo = (((hoverBinY + 1 - BINS / 2) / (BINS / 2)) * -diffMax)}
				{@const diffHi = (((hoverBinY - BINS / 2) / (BINS / 2)) * -diffMax)}
				<div>diff: [{formatValue(diffLo)}, {formatValue(diffHi)}]</div>
			{:else}
				{@const diffLo = ((BINS - 1 - hoverBinY) / BINS) * diffMax}
				{@const diffHi = ((BINS - hoverBinY) / BINS) * diffMax}
				<div>|diff|: [{formatValue(diffLo)}, {formatValue(diffHi)}]</div>
			{/if}
			<div class="text-gray-400">count: {count}</div>
		</div>
	{/if}
</div>
