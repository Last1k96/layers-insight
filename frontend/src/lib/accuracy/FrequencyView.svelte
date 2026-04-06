<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		valueToImageData,
		formatValue,
		drawColorbar,
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
	let batch = $state(0);
	let channel = $state(0);
	let colormap: ColormapName = $state('inferno');
	let showMode = $state<'ref' | 'main' | 'diff' | 'refPhase' | 'mainPhase' | 'phaseDiff'>('diff');

	let zoom = $state(1);
	let panX = $state(0);
	let panY = $state(0);
	let dragging = $state(false);
	let dragStartX = 0;
	let dragStartY = 0;
	let panStartX = 0;
	let panStartY = 0;

	let dims = $derived(getSpatialDims(shape));

	// Cooley-Tukey radix-2 FFT (in-place)
	function fft1D(re: Float32Array, im: Float32Array): void {
		const n = re.length;
		if (n <= 1) return;

		// Bit-reversal permutation
		let j = 0;
		for (let i = 1; i < n; i++) {
			let bit = n >> 1;
			while (j & bit) {
				j ^= bit;
				bit >>= 1;
			}
			j ^= bit;
			if (i < j) {
				let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
				tmp = im[i]; im[i] = im[j]; im[j] = tmp;
			}
		}

		// FFT butterfly
		for (let len = 2; len <= n; len *= 2) {
			const halfLen = len / 2;
			const angle = -2 * Math.PI / len;
			const wRe = Math.cos(angle);
			const wIm = Math.sin(angle);
			for (let i = 0; i < n; i += len) {
				let curRe = 1, curIm = 0;
				for (let k = 0; k < halfLen; k++) {
					const evenIdx = i + k;
					const oddIdx = i + k + halfLen;
					const tRe = curRe * re[oddIdx] - curIm * im[oddIdx];
					const tIm = curRe * im[oddIdx] + curIm * re[oddIdx];
					re[oddIdx] = re[evenIdx] - tRe;
					im[oddIdx] = im[evenIdx] - tIm;
					re[evenIdx] += tRe;
					im[evenIdx] += tIm;
					const newCurRe = curRe * wRe - curIm * wIm;
					curIm = curRe * wIm + curIm * wRe;
					curRe = newCurRe;
				}
			}
		}
	}

	function nextPow2(v: number): number {
		let p = 1;
		while (p < v) p <<= 1;
		return p;
	}

	interface FFT2DResult {
		magnitude: Float32Array;
		phase: Float32Array;
		w: number;
		h: number;
	}

	function fft2D(data: Float32Array, w: number, h: number): FFT2DResult {
		const maxDim = 256;
		let useW = w, useH = h;
		let useData = data;
		if (w > maxDim || h > maxDim) {
			useW = Math.min(w, maxDim);
			useH = Math.min(h, maxDim);
			useData = new Float32Array(useW * useH);
			for (let y = 0; y < useH; y++) {
				for (let x = 0; x < useW; x++) {
					const sy = Math.floor(y * h / useH);
					const sx = Math.floor(x * w / useW);
					useData[y * useW + x] = data[sy * w + sx];
				}
			}
		}

		// Zero-pad to next power of 2
		const fftW = nextPow2(useW);
		const fftH = nextPow2(useH);

		// Allocate real/imag arrays for 2D transform
		const re = new Float32Array(fftW * fftH);
		const im = new Float32Array(fftW * fftH);

		// Copy data into real part (zero-padded)
		for (let y = 0; y < useH; y++) {
			for (let x = 0; x < useW; x++) {
				re[y * fftW + x] = useData[y * useW + x];
			}
		}

		// FFT on each row
		const rowRe = new Float32Array(fftW);
		const rowIm = new Float32Array(fftW);
		for (let y = 0; y < fftH; y++) {
			const offset = y * fftW;
			for (let x = 0; x < fftW; x++) {
				rowRe[x] = re[offset + x];
				rowIm[x] = im[offset + x];
			}
			fft1D(rowRe, rowIm);
			for (let x = 0; x < fftW; x++) {
				re[offset + x] = rowRe[x];
				im[offset + x] = rowIm[x];
			}
		}

		// FFT on each column
		const colRe = new Float32Array(fftH);
		const colIm = new Float32Array(fftH);
		for (let x = 0; x < fftW; x++) {
			for (let y = 0; y < fftH; y++) {
				colRe[y] = re[y * fftW + x];
				colIm[y] = im[y * fftW + x];
			}
			fft1D(colRe, colIm);
			for (let y = 0; y < fftH; y++) {
				re[y * fftW + x] = colRe[y];
				im[y * fftW + x] = colIm[y];
			}
		}

		// Compute magnitude and phase
		const magnitude = new Float32Array(fftW * fftH);
		const phase = new Float32Array(fftW * fftH);
		for (let i = 0; i < fftW * fftH; i++) {
			magnitude[i] = Math.log(Math.sqrt(re[i] * re[i] + im[i] * im[i]) + 1);
			phase[i] = Math.atan2(im[i], re[i]);
		}

		// Shift zero frequency to center
		const shiftedMag = new Float32Array(fftW * fftH);
		const shiftedPhase = new Float32Array(fftW * fftH);
		const hw = Math.floor(fftW / 2), hh = Math.floor(fftH / 2);
		for (let y = 0; y < fftH; y++) {
			for (let x = 0; x < fftW; x++) {
				const ny = (y + hh) % fftH;
				const nx = (x + hw) % fftW;
				shiftedMag[ny * fftW + nx] = magnitude[y * fftW + x];
				shiftedPhase[ny * fftW + nx] = phase[y * fftW + x];
			}
		}

		return { magnitude: shiftedMag, phase: shiftedPhase, w: fftW, h: fftH };
	}

	// Turbo colormap for phase visualization
	function turboColormap(t: number): [number, number, number] {
		// Attempt at turbo approximation
		t = Math.max(0, Math.min(1, t));
		const r = Math.max(0, Math.min(255, Math.round(
			34.61 + t * (1172.33 + t * (-10793.56 + t * (33300.12 + t * (-38394.49 + t * 14825.05))))
		)));
		const g = Math.max(0, Math.min(255, Math.round(
			23.31 + t * (557.33 + t * (1225.33 + t * (-3574.96 + t * (1073.77 + t * 707.56))))
		)));
		const b = Math.max(0, Math.min(255, Math.round(
			27.2 + t * (3211.1 + t * (-15327.97 + t * (27814 + t * (-22569.18 + t * 6838.66))))
		)));
		return [r, g, b];
	}

	function phaseToImageData(data: Float32Array, w: number, h: number): ImageData {
		const img = new ImageData(w, h);
		const px = img.data;
		for (let i = 0; i < data.length; i++) {
			// Phase is in [-pi, pi], normalize to [0, 1]
			const t = (data[i] + Math.PI) / (2 * Math.PI);
			const [r, g, b] = turboColormap(t);
			px[i * 4] = r;
			px[i * 4 + 1] = g;
			px[i * 4 + 2] = b;
			px[i * 4 + 3] = 255;
		}
		return img;
	}

	let refSlice = $derived(extractSlice(ref, shape, batch, channel));
	let mainSlice = $derived(extractSlice(main, shape, batch, channel));

	let computing = $state(false);
	let refResult = $state<FFT2DResult | null>(null);
	let mainResult = $state<FFT2DResult | null>(null);

	// Compute FFT when slice changes (debounced)
	$effect(() => {
		const rSlice = refSlice;
		const mSlice = mainSlice;
		computing = true;

		requestAnimationFrame(() => {
			refResult = fft2D(rSlice.data, rSlice.w, rSlice.h);
			mainResult = fft2D(mSlice.data, mSlice.w, mSlice.h);
			computing = false;
		});
	});

	let fftW = $derived(refResult?.w ?? 0);
	let fftH = $derived(refResult?.h ?? 0);

	let diffMag = $derived.by(() => {
		if (!refResult || !mainResult) return null;
		const out = new Float32Array(refResult.magnitude.length);
		for (let i = 0; i < out.length; i++) out[i] = Math.abs(mainResult.magnitude[i] - refResult.magnitude[i]);
		return out;
	});

	let diffPhase = $derived.by(() => {
		if (!refResult || !mainResult) return null;
		const out = new Float32Array(refResult.phase.length);
		for (let i = 0; i < out.length; i++) {
			// Phase difference wrapped to [-pi, pi]
			let d = mainResult.phase[i] - refResult.phase[i];
			while (d > Math.PI) d -= 2 * Math.PI;
			while (d < -Math.PI) d += 2 * Math.PI;
			out[i] = d;
		}
		return out;
	});

	let isPhaseMode = $derived(showMode === 'refPhase' || showMode === 'mainPhase' || showMode === 'phaseDiff');

	let displayData = $derived.by(() => {
		if (!refResult || !mainResult) return null;
		switch (showMode) {
			case 'ref': return refResult.magnitude;
			case 'main': return mainResult.magnitude;
			case 'diff': return diffMag;
			case 'refPhase': return refResult.phase;
			case 'mainPhase': return mainResult.phase;
			case 'phaseDiff': return diffPhase;
			default: return diffMag;
		}
	});

	let offscreenImage = $derived.by(() => {
		if (!displayData || fftW === 0) return null;
		if (isPhaseMode) {
			return phaseToImageData(displayData, fftW, fftH);
		}
		return valueToImageData(displayData, fftW, fftH, colormap);
	});

	let baseScale = $derived.by(() => {
		if (!canvas || !fftW) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh) return 1;
		return Math.min(dw / fftW, dh / fftH);
	});

	function resetView() {
		zoom = 1;
		panX = 0;
		panY = 0;
	}

	function redraw() {
		if (!canvas || !offscreenImage) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const offscreen = new OffscreenCanvas(fftW, fftH);
		offscreen.getContext('2d')!.putImageData(offscreenImage, 0, 0);

		const es = baseScale * zoom;
		const ox = (dw - fftW * baseScale) / 2 + panX;
		const oy = (dh - fftH * baseScale) / 2 + panY;

		ctx.clearRect(0, 0, dw, dh);
		ctx.imageSmoothingEnabled = false;

		ctx.setTransform(es, 0, 0, es, ox, oy);
		ctx.drawImage(offscreen, 0, 0);
		ctx.resetTransform();

		// Cross at center
		const centerX = ox + fftW / 2 * es;
		const centerY = oy + fftH / 2 * es;
		ctx.strokeStyle = 'rgba(255,255,255,0.15)';
		ctx.lineWidth = 1;
		ctx.beginPath(); ctx.moveTo(centerX, 0); ctx.lineTo(centerX, dh); ctx.stroke();
		ctx.beginPath(); ctx.moveTo(0, centerY); ctx.lineTo(dw, centerY); ctx.stroke();

		if (!isPhaseMode) {
			const stats = displayData ? computeStats(displayData) : null;
			if (stats) {
				drawColorbar(ctx, 10, dh - 30, Math.min(200, dw - 20), 12, colormap, stats.min, stats.max);
			}
		} else {
			// Draw phase colorbar manually
			const barX = 10, barY = dh - 30, barW = Math.min(200, dw - 20), barH = 12;
			for (let i = 0; i < barW; i++) {
				const t = i / barW;
				const [r, g, b] = turboColormap(t);
				ctx.fillStyle = `rgb(${r},${g},${b})`;
				ctx.fillRect(barX + i, barY, 1, barH);
			}
			ctx.fillStyle = '#888'; ctx.font = '9px monospace'; ctx.textAlign = 'left';
			ctx.fillText('-\u03C0', barX, barY - 2);
			ctx.textAlign = 'right';
			ctx.fillText('+\u03C0', barX + barW, barY - 2);
			ctx.textAlign = 'center';
			ctx.fillText('0', barX + barW / 2, barY - 2);
		}

		// Labels
		ctx.fillStyle = '#888'; ctx.font = '10px monospace';
		ctx.textAlign = 'center';
		ctx.fillText('Low freq', ox + fftW * es / 2, oy - 4);
		ctx.fillText('High freq', ox + fftW * es - 10, oy + fftH * es / 2);
	}

	$effect(() => { offscreenImage; zoom; panX; panY; redraw(); });

	function handleWheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
		const rect = canvas.getBoundingClientRect();
		const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		const cxOff = (dw - fftW * baseScale) / 2;
		const cyOff = (dh - fftH * baseScale) / 2;
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
		if (dragging) {
			panX = panStartX + (e.clientX - dragStartX);
			panY = panStartY + (e.clientY - dragStartY);
		}
	}
	function handleMouseUp() { dragging = false; }
	function handleMouseLeave() { dragging = false; }

	$effect(() => { shape; zoom = 1; panX = 0; panY = 0; });
</script>

<svelte:window onmouseup={handleMouseUp} />

<div class="flex flex-col gap-4 relative h-full">
	<div class="flex flex-wrap gap-4 items-center text-xs">
		{#if dims.batches > 1}
			<label class="flex items-center gap-2">
				<span class="text-gray-400 shrink-0">Batch:</span>
				<input use:rangeScroll type="range" min="0" max={dims.batches - 1} bind:value={batch} class="w-20" />
				<span class="text-gray-300 w-6 shrink-0">{batch}</span>
			</label>
		{/if}
		{#if dims.channels > 1}
			<label class="flex items-center gap-2">
				<span class="text-gray-400 shrink-0">Channel:</span>
				<input use:rangeScroll type="range" min="0" max={dims.channels - 1} bind:value={channel} class="w-24" />
				<span class="text-gray-300 w-8 shrink-0">{channel}/{dims.channels}</span>
			</label>
		{/if}
		<label class="flex items-center gap-2">
			<span class="text-gray-400">Show:</span>
			<select use:rangeScroll bind:value={showMode} class="bg-surface-base border border-edge rounded px-1.5 py-0.5 text-xs text-gray-300">
				<option value="diff">Spectral Difference</option>
				<option value="ref">{refLabel} Spectrum</option>
				<option value="main">{mainLabel} Spectrum</option>
				<option value="refPhase">{refLabel} Phase</option>
				<option value="mainPhase">{mainLabel} Phase</option>
				<option value="phaseDiff">Phase Difference</option>
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
		<button
			class="px-2 py-0.5 rounded text-xs border border-edge text-gray-300 hover:bg-surface-overlay"
			onclick={resetView}
		>Reset zoom</button>
	</div>

	<div class="text-xs text-gray-500">
		2D FFT {isPhaseMode ? 'phase' : 'log-magnitude'} spectrum (center=DC). High-freq diff = added noise. Low-freq diff = structural changes.
		{#if refSlice.w > 256 || refSlice.h > 256}
			<span class="text-yellow-400">Downsampled to 256x256 for performance.</span>
		{/if}
	</div>

	<div class="flex-1 flex justify-center items-center bg-surface-base rounded-lg p-4 overflow-hidden min-h-0">
		{#if computing}
			<div class="text-gray-500 text-sm">Computing FFT...</div>
		{:else}
			<canvas
				bind:this={canvas}
				class="w-full h-full cursor-crosshair"
				style="image-rendering: pixelated;"
				onwheel={handleWheel}
				onmousedown={handleMouseDown}
				onmousemove={handleMouseMove}
				onmouseleave={handleMouseLeave}
			></canvas>
		{/if}
	</div>
</div>
