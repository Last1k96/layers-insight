<script lang="ts">
	import { getSpatialDims, extractSlice, formatValue, colormapRGB } from './tensorUtils';

	let { main, ref, shape, mainLabel = 'Main', refLabel = 'Reference' }: {
		main: Float32Array;
		ref: Float32Array;
		shape: number[];
		mainLabel?: string;
		refLabel?: string;
	} = $props();

	let canvas: HTMLCanvasElement | undefined = $state();
	let selectedChannel = $state<number | null>(null);
	let hoveredIndex = $state<number | null>(null);
	let tooltipX = $state(0);
	let tooltipY = $state(0);

	// Current layout rectangles for hit testing
	let currentRects: { x: number; y: number; w: number; h: number; index: number }[] = [];

	const CANVAS_HEIGHT = 500;

	// ---------------------------------------------------------------------------
	// Per-channel error data
	// ---------------------------------------------------------------------------

	interface ChannelError {
		channel: number;
		error: number;
		size: number;
	}

	let dims = $derived(getSpatialDims(shape));
	let spatialSize = $derived(dims.height * dims.width);

	let channelErrors = $derived.by((): ChannelError[] => {
		const d = dims;
		const sSize = spatialSize;
		const result: ChannelError[] = [];

		for (let c = 0; c < d.channels; c++) {
			let totalError = 0;
			// Compute for batch 0
			const offset = c * sSize;
			for (let i = 0; i < sSize; i++) {
				totalError += Math.abs(main[offset + i] - ref[offset + i]);
			}
			result.push({ channel: c, error: totalError, size: sSize });
		}

		result.sort((a, b) => b.error - a.error);
		return result;
	});

	let totalError = $derived(channelErrors.reduce((s, e) => s + e.error, 0));

	// Max error density for colormap normalization
	let maxDensity = $derived.by(() => {
		let mx = 0;
		for (const ce of channelErrors) {
			const density = ce.size > 0 ? ce.error / ce.size : 0;
			if (density > mx) mx = density;
		}
		return mx || 1;
	});

	// ---------------------------------------------------------------------------
	// Spatial block data for drill-down
	// ---------------------------------------------------------------------------

	interface BlockError {
		blockIdx: number;
		row: number;
		col: number;
		error: number;
		size: number;
		rStart: number;
		rEnd: number;
		cStart: number;
		cEnd: number;
	}

	let blockErrors = $derived.by((): BlockError[] | null => {
		if (selectedChannel === null) return null;

		const ch = selectedChannel;
		const h = dims.height;
		const w = dims.width;
		const blockRows = Math.min(8, h);
		const blockCols = Math.min(8, w);
		const blockH = Math.ceil(h / blockRows);
		const blockW = Math.ceil(w / blockCols);

		const blocks: BlockError[] = [];
		const chOffset = ch * spatialSize;

		let idx = 0;
		for (let br = 0; br < blockRows; br++) {
			for (let bc = 0; bc < blockCols; bc++) {
				let err = 0;
				let count = 0;
				const rStart = br * blockH;
				const rEnd = Math.min(rStart + blockH, h);
				const cStart = bc * blockW;
				const cEnd = Math.min(cStart + blockW, w);
				for (let r = rStart; r < rEnd; r++) {
					for (let c = cStart; c < cEnd; c++) {
						const i = chOffset + r * w + c;
						err += Math.abs(main[i] - ref[i]);
						count++;
					}
				}
				blocks.push({ blockIdx: idx++, row: br, col: bc, error: err, size: count, rStart, rEnd: rEnd - 1, cStart, cEnd: cEnd - 1 });
			}
		}

		blocks.sort((a, b) => b.error - a.error);
		return blocks;
	});

	let blockTotalError = $derived(
		blockErrors ? blockErrors.reduce((s, b) => s + b.error, 0) : 0,
	);

	let blockMaxDensity = $derived.by(() => {
		if (!blockErrors) return 1;
		let mx = 0;
		for (const b of blockErrors) {
			const density = b.size > 0 ? b.error / b.size : 0;
			if (density > mx) mx = density;
		}
		return mx || 1;
	});

	// ---------------------------------------------------------------------------
	// Squarified treemap layout
	// ---------------------------------------------------------------------------

	interface TreemapItem {
		value: number;
		index: number;
	}

	interface Rect {
		x: number;
		y: number;
		w: number;
		h: number;
	}

	interface TreemapRect extends Rect {
		index: number;
	}

	function squarify(
		items: TreemapItem[],
		rect: Rect,
	): TreemapRect[] {
		if (items.length === 0) return [];

		const totalValue = items.reduce((s, it) => s + it.value, 0);
		if (totalValue <= 0) {
			// All zero errors — distribute equally
			const result: TreemapRect[] = [];
			const n = items.length;
			const cellW = rect.w / n;
			for (let i = 0; i < n; i++) {
				result.push({
					x: rect.x + i * cellW,
					y: rect.y,
					w: cellW,
					h: rect.h,
					index: items[i].index,
				});
			}
			return result;
		}

		const totalArea = rect.w * rect.h;
		// Normalized areas proportional to value
		const areas = items.map((it) => (it.value / totalValue) * totalArea);

		const result: TreemapRect[] = [];
		let remaining = { ...rect };

		let i = 0;
		while (i < items.length) {
			const row: number[] = [];
			const rowIndices: number[] = [];

			// Determine the shorter side of remaining rect
			const shorter = Math.min(remaining.w, remaining.h);

			row.push(areas[i]);
			rowIndices.push(i);
			let rowArea = areas[i];
			let bestWorst = worstRatio(row, shorter, rowArea);
			i++;

			while (i < items.length) {
				const candidate = [...row, areas[i]];
				const candidateArea = rowArea + areas[i];
				const candidateWorst = worstRatio(candidate, shorter, candidateArea);
				if (candidateWorst <= bestWorst) {
					row.push(areas[i]);
					rowIndices.push(i);
					rowArea = candidateArea;
					bestWorst = candidateWorst;
					i++;
				} else {
					break;
				}
			}

			// Lay out the row
			const horizontal = remaining.w >= remaining.h;
			if (horizontal) {
				const rowWidth = rowArea / remaining.h;
				let cy = remaining.y;
				for (let j = 0; j < row.length; j++) {
					const cellH = row[j] / rowWidth;
					result.push({
						x: remaining.x,
						y: cy,
						w: rowWidth,
						h: cellH,
						index: items[rowIndices[j]].index,
					});
					cy += cellH;
				}
				remaining = {
					x: remaining.x + rowWidth,
					y: remaining.y,
					w: remaining.w - rowWidth,
					h: remaining.h,
				};
			} else {
				const rowHeight = rowArea / remaining.w;
				let cx = remaining.x;
				for (let j = 0; j < row.length; j++) {
					const cellW = row[j] / rowHeight;
					result.push({
						x: cx,
						y: remaining.y,
						w: cellW,
						h: rowHeight,
						index: items[rowIndices[j]].index,
					});
					cx += cellW;
				}
				remaining = {
					x: remaining.x,
					y: remaining.y + rowHeight,
					w: remaining.w,
					h: remaining.h - rowHeight,
				};
			}
		}

		return result;
	}

	function worstRatio(row: number[], sideLength: number, rowArea: number): number {
		if (sideLength === 0 || rowArea === 0) return Infinity;
		const s2 = sideLength * sideLength;
		let worst = 0;
		for (const area of row) {
			const r1 = (s2 * area) / (rowArea * rowArea);
			const r2 = (rowArea * rowArea) / (s2 * area);
			const ratio = Math.max(r1, r2);
			if (ratio > worst) worst = ratio;
		}
		return worst;
	}

	// ---------------------------------------------------------------------------
	// Hover tooltip data
	// ---------------------------------------------------------------------------

	let tooltipData = $derived.by(() => {
		if (hoveredIndex === null) return null;

		if (selectedChannel === null) {
			// Level 0: channel view
			const ce = channelErrors.find((e) => e.channel === hoveredIndex);
			if (!ce) return null;
			const pct = totalError > 0 ? (ce.error / totalError) * 100 : 0;
			return {
				label: `Channel ${ce.channel}`,
				error: formatValue(ce.error),
				pct: pct.toFixed(1),
			};
		} else {
			// Level 1: block view
			if (!blockErrors) return null;
			const be = blockErrors.find((b) => b.blockIdx === hoveredIndex);
			if (!be) return null;
			const pct = blockTotalError > 0 ? (be.error / blockTotalError) * 100 : 0;
			return {
				label: `H:${be.rStart}-${be.rEnd}, W:${be.cStart}-${be.cEnd}`,
				error: formatValue(be.error),
				pct: pct.toFixed(1),
			};
		}
	});

	/** Pick black or white text for readability on a given background. */
	function contrastText(r: number, g: number, b: number): string {
		const lum = 0.299 * r + 0.587 * g + 0.114 * b;
		return lum > 140 ? '#000000' : '#ffffff';
	}

	// ---------------------------------------------------------------------------
	// Canvas rendering
	// ---------------------------------------------------------------------------

	$effect(() => {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const dpr = window.devicePixelRatio || 1;
		const displayW = canvas.clientWidth;
		const displayH = CANVAS_HEIGHT;
		canvas.width = displayW * dpr;
		canvas.height = displayH * dpr;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

		ctx.clearRect(0, 0, displayW, displayH);

		const boundingRect: Rect = { x: 0, y: 0, w: displayW, h: displayH };

		if (selectedChannel === null) {
			// Level 0: all channels
			const items: TreemapItem[] = channelErrors.map((ce) => ({
				value: ce.error,
				index: ce.channel,
			}));
			const rects = squarify(items, boundingRect);
			currentRects = rects;
			const mxD = maxDensity;

			for (const r of rects) {
				const ce = channelErrors.find((e) => e.channel === r.index);
				const density = ce && ce.size > 0 ? ce.error / ce.size : 0;
				const t = mxD > 0 ? density / mxD : 0;
				const [cr, cg, cb] = colormapRGB(t, 'magma');

				ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
				ctx.fillRect(r.x, r.y, r.w, r.h);

				// Border
				const isHovered = hoveredIndex === r.index;
				ctx.strokeStyle = isHovered ? '#f59e0b' : '#374151';
				ctx.lineWidth = isHovered ? 2 : 1;
				ctx.strokeRect(r.x, r.y, r.w, r.h);

				// Label
				if (r.w > 30 && r.h > 30) {
					ctx.fillStyle = contrastText(cr, cg, cb);
					ctx.font = '11px monospace';
					ctx.textAlign = 'center';
					ctx.textBaseline = 'middle';
					ctx.fillText(`${r.index}`, r.x + r.w / 2, r.y + r.h / 2);
				}
			}
		} else {
			// Level 1: spatial blocks within selected channel
			if (!blockErrors) return;
			const items: TreemapItem[] = blockErrors.map((be) => ({
				value: be.error,
				index: be.blockIdx,
			}));
			const rects = squarify(items, boundingRect);
			currentRects = rects;
			const mxD = blockMaxDensity;

			for (const r of rects) {
				const be = blockErrors.find((b) => b.blockIdx === r.index);
				const density = be && be.size > 0 ? be.error / be.size : 0;
				const t = mxD > 0 ? density / mxD : 0;
				const [cr, cg, cb] = colormapRGB(t, 'magma');

				ctx.fillStyle = `rgb(${cr},${cg},${cb})`;
				ctx.fillRect(r.x, r.y, r.w, r.h);

				const isHovered = hoveredIndex === r.index;
				ctx.strokeStyle = isHovered ? '#f59e0b' : '#374151';
				ctx.lineWidth = isHovered ? 2 : 1;
				ctx.strokeRect(r.x, r.y, r.w, r.h);

				if (r.w > 40 && r.h > 30 && be) {
					ctx.fillStyle = contrastText(cr, cg, cb);
					ctx.font = '11px monospace';
					ctx.textAlign = 'center';
					ctx.textBaseline = 'middle';
					const rangeLabel = `${be.rStart}-${be.rEnd}, ${be.cStart}-${be.cEnd}`;
					ctx.fillText(rangeLabel, r.x + r.w / 2, r.y + r.h / 2);
				}
			}
		}
	});

	// ---------------------------------------------------------------------------
	// Hit test
	// ---------------------------------------------------------------------------

	function hitTest(clientX: number, clientY: number): number | null {
		if (!canvas) return null;
		const rect = canvas.getBoundingClientRect();
		const x = clientX - rect.left;
		const y = clientY - rect.top;

		for (const r of currentRects) {
			if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) {
				return r.index;
			}
		}
		return null;
	}

	function handleClick(e: MouseEvent) {
		const idx = hitTest(e.clientX, e.clientY);
		if (idx === null) return;

		if (selectedChannel === null) {
			selectedChannel = idx;
			hoveredIndex = null;
		}
	}

	function handleMouseMove(e: MouseEvent) {
		const idx = hitTest(e.clientX, e.clientY);
		hoveredIndex = idx;
		tooltipX = e.clientX;
		tooltipY = e.clientY;
	}

	function handleMouseLeave() {
		hoveredIndex = null;
	}

	function resetToTop() {
		selectedChannel = null;
		hoveredIndex = null;
	}
</script>

<div class="relative">
	<!-- Breadcrumb -->
	<div class="mb-2 flex items-center gap-1 text-sm text-gray-300">
		<button
			class="hover:text-white underline-offset-2 {selectedChannel === null
				? 'font-semibold text-white'
				: 'hover:underline'}"
			onclick={resetToTop}
		>
			All Channels
		</button>
		{#if selectedChannel !== null}
			<span class="text-gray-500">&gt;</span>
			<span class="font-semibold text-white">Channel {selectedChannel}</span>
		{/if}
	</div>

	<!-- Canvas -->
	<canvas
		bind:this={canvas}
		class="w-full cursor-pointer"
		style="height: {CANVAS_HEIGHT}px"
		onclick={handleClick}
		onmousemove={handleMouseMove}
		onmouseleave={handleMouseLeave}
	></canvas>

	<!-- Tooltip -->
	{#if tooltipData && hoveredIndex !== null}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-700 bg-gray-900 px-3 py-2 text-xs text-gray-300 shadow-lg"
			style="left: {tooltipX + 12}px; top: {tooltipY - 10}px"
		>
			<div class="font-semibold text-white">{tooltipData.label}</div>
			<div>Error: {tooltipData.error}</div>
			<div>{tooltipData.pct}% of total</div>
		</div>
	{/if}
</div>
