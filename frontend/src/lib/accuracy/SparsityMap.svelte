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
	let sparsityBarCanvas: HTMLCanvasElement;
	let batch = $state(0);
	let channel = $state(0);
	let thresholdExp = $state(-7);
	let showMode: 'sidebyside' | 'xor' = $state('sidebyside');

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
	let threshold = $derived(Math.pow(10, thresholdExp));

	let refSlice = $derived(extractSlice(ref, shape, batch, channel));
	let mainSlice = $derived(extractSlice(main, shape, batch, channel));

	// Sparsity stats
	let refSparse = $derived.by(() => {
		let zero = 0;
		for (let i = 0; i < refSlice.data.length; i++) {
			if (Math.abs(refSlice.data[i]) < threshold) zero++;
		}
		return zero;
	});
	let mainSparse = $derived.by(() => {
		let zero = 0;
		for (let i = 0; i < mainSlice.data.length; i++) {
			if (Math.abs(mainSlice.data[i]) < threshold) zero++;
		}
		return zero;
	});

	// Per-channel sparsity for bar chart
	let channelSparsity = $derived.by(() => {
		const result: { refPct: number; mainPct: number }[] = [];
		for (let c = 0; c < dims.channels; c++) {
			const rSlice = extractSlice(ref, shape, batch, c);
			const mSlice = extractSlice(main, shape, batch, c);
			let rZero = 0, mZero = 0;
			for (let i = 0; i < rSlice.data.length; i++) {
				if (Math.abs(rSlice.data[i]) < threshold) rZero++;
				if (Math.abs(mSlice.data[i]) < threshold) mZero++;
			}
			const total = rSlice.data.length || 1;
			result.push({ refPct: rZero / total * 100, mainPct: mZero / total * 100 });
		}
		return result;
	});

	let totalPixels = $derived(refSlice.data.length);

	let baseScale = $derived.by(() => {
		if (!refSlice || !canvas) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh) return 1;
		const totalW = showMode === 'sidebyside' ? refSlice.w * 2 + 4 : refSlice.w;
		return Math.min(dw / totalW, dh / refSlice.h);
	});

	function redraw() {
		if (!canvas || !refSlice || !mainSlice) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const w = refSlice.w, h = refSlice.h;

		if (showMode === 'sidebyside') {
			const gap = 4;
			const totalW = w * 2 + gap;
			const es = baseScale * zoom;
			const ox = (dw - totalW * baseScale) / 2 + panX;
			const oy = (dh - h * baseScale) / 2 + panY;

			function buildSparsityImage(data: Float32Array, color: [number, number, number]): ImageData {
				const img = new ImageData(w, h);
				for (let i = 0; i < data.length; i++) {
					const isZero = Math.abs(data[i]) < threshold;
					const px = i * 4;
					if (isZero) {
						img.data[px] = 20; img.data[px + 1] = 20; img.data[px + 2] = 25;
					} else {
						img.data[px] = color[0]; img.data[px + 1] = color[1]; img.data[px + 2] = color[2];
					}
					img.data[px + 3] = 255;
				}
				return img;
			}

			const refImg = buildSparsityImage(refSlice.data, [96, 165, 250]);
			const mainImg = buildSparsityImage(mainSlice.data, [248, 113, 113]);

			const offRef = new OffscreenCanvas(w, h);
			offRef.getContext('2d')!.putImageData(refImg, 0, 0);
			const offMain = new OffscreenCanvas(w, h);
			offMain.getContext('2d')!.putImageData(mainImg, 0, 0);

			ctx.clearRect(0, 0, dw, dh);
			ctx.imageSmoothingEnabled = false;
			ctx.setTransform(es, 0, 0, es, ox, oy);
			ctx.drawImage(offRef, 0, 0);
			ctx.resetTransform();
			ctx.setTransform(es, 0, 0, es, ox + (w + gap) * es, oy);
			ctx.drawImage(offMain, 0, 0);
			ctx.resetTransform();

			ctx.fillStyle = '#60a5fa'; ctx.font = 'bold 11px monospace'; ctx.textAlign = 'left';
			ctx.fillText(refLabel, ox + 2, oy - 4);
			ctx.fillStyle = '#f87171';
			ctx.fillText(mainLabel, ox + (w + gap) * es + 2, oy - 4);
		} else {
			// XOR mode: show where sparsity differs, with magnitude scaling
			const es = baseScale * zoom;
			const ox = (dw - w * baseScale) / 2 + panX;
			const oy = (dh - h * baseScale) / 2 + panY;

			// Find max absolute value in each slice for magnitude normalization
			let maxAbsRef = 0, maxAbsMain = 0;
			for (let i = 0; i < refSlice.data.length; i++) {
				const ar = Math.abs(refSlice.data[i]);
				const am = Math.abs(mainSlice.data[i]);
				if (ar > maxAbsRef) maxAbsRef = ar;
				if (am > maxAbsMain) maxAbsMain = am;
			}
			if (maxAbsRef === 0) maxAbsRef = 1;
			if (maxAbsMain === 0) maxAbsMain = 1;

			const img = new ImageData(w, h);
			for (let i = 0; i < refSlice.data.length; i++) {
				const rZero = Math.abs(refSlice.data[i]) < threshold;
				const mZero = Math.abs(mainSlice.data[i]) < threshold;
				const px = i * 4;
				if (rZero && mZero) {
					// Both zero
					img.data[px] = 20; img.data[px + 1] = 20; img.data[px + 2] = 25;
				} else if (!rZero && !mZero) {
					// Both non-zero
					img.data[px] = 60; img.data[px + 1] = 60; img.data[px + 2] = 60;
				} else if (rZero && !mZero) {
					// Ref zero, main non-zero (activation appeared) - scale by main magnitude
					const brightness = Math.abs(mainSlice.data[i]) / maxAbsMain;
					const b = 0.2 + 0.8 * brightness; // min 20% brightness
					img.data[px] = Math.round(248 * b);
					img.data[px + 1] = Math.round(113 * b);
					img.data[px + 2] = Math.round(113 * b);
				} else {
					// Ref non-zero, main zero (activation died) - scale by ref magnitude
					const brightness = Math.abs(refSlice.data[i]) / maxAbsRef;
					const b = 0.2 + 0.8 * brightness; // min 20% brightness
					img.data[px] = Math.round(96 * b);
					img.data[px + 1] = Math.round(165 * b);
					img.data[px + 2] = Math.round(250 * b);
				}
				img.data[px + 3] = 255;
			}

			const offscreen = new OffscreenCanvas(w, h);
			offscreen.getContext('2d')!.putImageData(img, 0, 0);
			ctx.clearRect(0, 0, dw, dh);
			ctx.imageSmoothingEnabled = false;
			ctx.setTransform(es, 0, 0, es, ox, oy);
			ctx.drawImage(offscreen, 0, 0);
			ctx.resetTransform();
		}

		if (showTooltip && hoverX >= 0 && hoverY >= 0) {
			const es = baseScale * zoom;
			const totalW = showMode === 'sidebyside' ? refSlice.w * 2 + 4 : refSlice.w;
			const ox = (dw - totalW * baseScale) / 2 + panX;
			const oy = (dh - h * baseScale) / 2 + panY;
			const sx = hoverX * es + ox, sy = hoverY * es + oy;
			ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1;
			ctx.beginPath(); ctx.moveTo(0, sy + 0.5); ctx.lineTo(dw, sy + 0.5); ctx.stroke();
			ctx.beginPath(); ctx.moveTo(sx + 0.5, 0); ctx.lineTo(sx + 0.5, dh); ctx.stroke();
		}
	}

	$effect(() => { refSlice; mainSlice; showMode; threshold; zoom; panX; panY; showTooltip; hoverX; hoverY; redraw(); });

	function screenToData(cx: number, cy: number): [number, number] {
		if (!canvas || !refSlice) return [-1, -1];
		const rect = canvas.getBoundingClientRect();
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const totalW = showMode === 'sidebyside' ? refSlice.w * 2 + 4 : refSlice.w;
		const ox = (dw - totalW * baseScale) / 2 + panX;
		const oy = (dh - refSlice.h * baseScale) / 2 + panY;
		const es = baseScale * zoom;
		return [(cx - rect.left - ox) / es, (cy - rect.top - oy) / es];
	}

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
		const rect = canvas.getBoundingClientRect();
		const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const totalW = showMode === 'sidebyside' ? (refSlice?.w ?? 1) * 2 + 4 : (refSlice?.w ?? 1);
		const cxOff = (dw - totalW * baseScale) / 2;
		const cyOff = (dh - (refSlice?.h ?? 1) * baseScale) / 2;
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

	// Sparsity bar chart rendering
	$effect(() => {
		if (!sparsityBarCanvas || channelSparsity.length === 0) return;
		const ctx = sparsityBarCanvas.getContext('2d');
		if (!ctx) return;
		const dw = sparsityBarCanvas.clientWidth, dh = sparsityBarCanvas.clientHeight;
		sparsityBarCanvas.width = dw; sparsityBarCanvas.height = dh;
		ctx.clearRect(0, 0, dw, dh);

		const n = channelSparsity.length;
		const barGroupW = dw / n;
		const barW = Math.max(1, barGroupW * 0.4);
		const maxPct = 100;

		for (let c = 0; c < n; c++) {
			const { refPct, mainPct } = channelSparsity[c];
			const x = c * barGroupW;
			const refH = (refPct / maxPct) * dh;
			const mainH = (mainPct / maxPct) * dh;

			// Highlight current channel
			if (c === channel) {
				ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
				ctx.fillRect(x, 0, barGroupW, dh);
			}

			// Ref bar (blue)
			ctx.fillStyle = c === channel ? '#60a5fa' : '#3b82f680';
			ctx.fillRect(x, dh - refH, barW, refH);

			// Main bar (red)
			ctx.fillStyle = c === channel ? '#f87171' : '#ef444480';
			ctx.fillRect(x + barW, dh - mainH, barW, mainH);
		}

		// Baseline
		ctx.strokeStyle = '#444';
		ctx.lineWidth = 0.5;
		ctx.beginPath(); ctx.moveTo(0, dh - 0.5); ctx.lineTo(dw, dh - 0.5); ctx.stroke();
	});

	function resetView() { zoom = 1; panX = 0; panY = 0; }
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
		<label class="flex items-center gap-2 flex-1 min-w-[12rem]">
			<span class="text-gray-400 shrink-0">Zero threshold: 10^</span>
			<input use:rangeScroll type="range" min="-10" max="0" step="0.5" bind:value={thresholdExp} class="flex-1" />
			<span class="text-gray-300 shrink-0 font-mono">{threshold.toExponential(1)}</span>
		</label>
		<div class="flex items-center gap-1">
			<button class="px-2 py-0.5 rounded text-xs border border-edge"
				class:bg-accent={showMode === 'sidebyside'} class:text-white={showMode === 'sidebyside'} class:text-gray-400={showMode !== 'sidebyside'}
				onclick={() => showMode = 'sidebyside'}>Side-by-Side</button>
			<button class="px-2 py-0.5 rounded text-xs border border-edge"
				class:bg-accent={showMode === 'xor'} class:text-white={showMode === 'xor'} class:text-gray-400={showMode !== 'xor'}
				onclick={() => showMode = 'xor'}>XOR Diff</button>
		</div>
		<button
			class="px-2 py-0.5 text-gray-400 hover:text-gray-200 border border-edge rounded text-xs"
			onclick={resetView}
		>Reset view</button>
	</div>

	<div class="flex gap-6 text-xs">
		<span class="text-blue-400">{refLabel}: {refSparse}/{totalPixels} zero ({(refSparse / (totalPixels || 1) * 100).toFixed(1)}%)</span>
		<span class="text-red-400">{mainLabel}: {mainSparse}/{totalPixels} zero ({(mainSparse / (totalPixels || 1) * 100).toFixed(1)}%)</span>
		{#if showMode === 'xor'}
			<span class="text-gray-500">
				<span class="text-blue-400">Blue</span>=died in {mainLabel},
				<span class="text-red-400">Red</span>=appeared in {mainLabel}
			</span>
		{/if}
	</div>

	{#if dims.channels > 1}
		<div class="bg-surface-base rounded-lg px-2 pt-1 pb-0 overflow-hidden" style="height: 60px;">
			<div class="text-[9px] text-gray-500 mb-0.5 flex justify-between">
				<span>Per-channel sparsity (<span class="text-blue-400">{refLabel}</span> / <span class="text-red-400">{mainLabel}</span>)</span>
				<span class="text-gray-600">ch {channel}</span>
			</div>
			<canvas
				bind:this={sparsityBarCanvas}
				class="w-full cursor-pointer"
				style="height: 44px;"
				onclick={(e: MouseEvent) => {
					const rect = sparsityBarCanvas.getBoundingClientRect();
					const mx = e.clientX - rect.left;
					const idx = Math.floor((mx / rect.width) * dims.channels);
					if (idx >= 0 && idx < dims.channels) channel = idx;
				}}
			></canvas>
		</div>
	{/if}

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

	{#if showTooltip && hoverX >= 0 && hoverY >= 0 && refSlice}
		{@const idx = hoverY * refSlice.w + hoverX}
		<div
			class="pointer-events-none fixed z-50 rounded border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs text-gray-200 shadow-lg"
			style="left: {tooltipScreenX + 16}px; top: {tooltipScreenY + 16}px;"
		>
			<div class="font-mono text-gray-400">[{hoverY}, {hoverX}]</div>
			<div><span class="text-blue-400">{refLabel}:</span> {formatValue(refSlice.data[idx])} {Math.abs(refSlice.data[idx]) < threshold ? '(zero)' : ''}</div>
			<div><span class="text-red-400">{mainLabel}:</span> {formatValue(mainSlice.data[idx])} {Math.abs(mainSlice.data[idx]) < threshold ? '(zero)' : ''}</div>
		</div>
	{/if}
</div>
