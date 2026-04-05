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
	let showMode: 'ref' | 'main' | 'diff' = $state('diff');

	let dims = $derived(getSpatialDims(shape));

	// Simple 2D DFT (for small-to-medium tensors)
	// For larger tensors, this will be slow — consider warning
	function dft2D(data: Float32Array, w: number, h: number): Float32Array {
		// Compute magnitude spectrum using basic DFT
		// For performance, downsample if too large
		const maxDim = 128;
		let useW = w, useH = h;
		let useData = data;
		if (w > maxDim || h > maxDim) {
			// Downsample
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

		const magnitude = new Float32Array(useW * useH);
		const TWO_PI = 2 * Math.PI;

		// 1D FFT would be better, but simple DFT for correctness
		// Optimization: separate row and column transforms
		// Row transforms
		const realRows = new Float32Array(useW * useH);
		const imagRows = new Float32Array(useW * useH);

		for (let y = 0; y < useH; y++) {
			for (let u = 0; u < useW; u++) {
				let re = 0, im = 0;
				for (let x = 0; x < useW; x++) {
					const angle = -TWO_PI * u * x / useW;
					const v = useData[y * useW + x];
					re += v * Math.cos(angle);
					im += v * Math.sin(angle);
				}
				realRows[y * useW + u] = re;
				imagRows[y * useW + u] = im;
			}
		}

		// Column transforms
		for (let u = 0; u < useW; u++) {
			for (let v = 0; v < useH; v++) {
				let re = 0, im = 0;
				for (let y = 0; y < useH; y++) {
					const angle = -TWO_PI * v * y / useH;
					const cos = Math.cos(angle);
					const sin = Math.sin(angle);
					re += realRows[y * useW + u] * cos - imagRows[y * useW + u] * sin;
					im += realRows[y * useW + u] * sin + imagRows[y * useW + u] * cos;
				}
				magnitude[v * useW + u] = Math.log(Math.sqrt(re * re + im * im) + 1);
			}
		}

		// Shift zero frequency to center
		const shifted = new Float32Array(useW * useH);
		const hw = Math.floor(useW / 2), hh = Math.floor(useH / 2);
		for (let y = 0; y < useH; y++) {
			for (let x = 0; x < useW; x++) {
				const ny = (y + hh) % useH;
				const nx = (x + hw) % useW;
				shifted[ny * useW + nx] = magnitude[y * useW + x];
			}
		}

		return shifted;
	}

	let refSlice = $derived(extractSlice(ref, shape, batch, channel));
	let mainSlice = $derived(extractSlice(main, shape, batch, channel));

	let computing = $state(false);
	let refFFT = $state<Float32Array | null>(null);
	let mainFFT = $state<Float32Array | null>(null);
	let fftW = $state(0);
	let fftH = $state(0);

	// Compute FFT when slice changes (debounced)
	$effect(() => {
		const rSlice = refSlice;
		const mSlice = mainSlice;
		computing = true;

		// Use requestAnimationFrame to avoid blocking
		requestAnimationFrame(() => {
			const maxDim = 128;
			fftW = Math.min(rSlice.w, maxDim);
			fftH = Math.min(rSlice.h, maxDim);
			refFFT = dft2D(rSlice.data, rSlice.w, rSlice.h);
			mainFFT = dft2D(mSlice.data, mSlice.w, mSlice.h);
			computing = false;
		});
	});

	let diffFFT = $derived.by(() => {
		if (!refFFT || !mainFFT) return null;
		const out = new Float32Array(refFFT.length);
		for (let i = 0; i < out.length; i++) out[i] = Math.abs(mainFFT[i] - refFFT[i]);
		return out;
	});

	let displayData = $derived(showMode === 'ref' ? refFFT : showMode === 'main' ? mainFFT : diffFFT);

	let offscreenImage = $derived.by(() => {
		if (!displayData || fftW === 0) return null;
		return valueToImageData(displayData, fftW, fftH, colormap);
	});

	let baseScale = $derived.by(() => {
		if (!canvas || !fftW) return 1;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		if (!dw || !dh) return 1;
		return Math.min(dw / fftW, dh / fftH);
	});

	function redraw() {
		if (!canvas || !offscreenImage) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dw = canvas.clientWidth, dh = canvas.clientHeight;
		canvas.width = dw; canvas.height = dh;

		const offscreen = new OffscreenCanvas(fftW, fftH);
		offscreen.getContext('2d')!.putImageData(offscreenImage, 0, 0);

		const scale = baseScale;
		const ox = (dw - fftW * scale) / 2;
		const oy = (dh - fftH * scale) / 2;

		ctx.clearRect(0, 0, dw, dh);
		ctx.imageSmoothingEnabled = false;
		ctx.drawImage(offscreen, ox, oy, fftW * scale, fftH * scale);

		// Cross at center
		ctx.strokeStyle = 'rgba(255,255,255,0.15)';
		ctx.lineWidth = 1;
		ctx.beginPath(); ctx.moveTo(dw / 2, 0); ctx.lineTo(dw / 2, dh); ctx.stroke();
		ctx.beginPath(); ctx.moveTo(0, dh / 2); ctx.lineTo(dw, dh / 2); ctx.stroke();

		const stats = displayData ? computeStats(displayData) : null;
		if (stats) {
			drawColorbar(ctx, 10, dh - 30, Math.min(200, dw - 20), 12, colormap, stats.min, stats.max);
		}

		// Labels
		ctx.fillStyle = '#888'; ctx.font = '10px monospace';
		ctx.textAlign = 'center';
		ctx.fillText('Low freq', dw / 2, oy - 4);
		ctx.fillText('High freq', dw - 10, oy + fftH * scale / 2);
	}

	$effect(() => { offscreenImage; redraw(); });
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
	</div>

	<div class="text-xs text-gray-500">
		2D FFT log-magnitude spectrum (center=DC). High-freq diff = added noise. Low-freq diff = structural changes.
		{#if refSlice.w > 128 || refSlice.h > 128}
			<span class="text-yellow-400">Downsampled to 128x128 for performance.</span>
		{/if}
	</div>

	<div class="flex-1 flex justify-center items-center bg-surface-base rounded-lg p-4 overflow-hidden min-h-0">
		{#if computing}
			<div class="text-gray-500 text-sm">Computing FFT...</div>
		{:else}
			<canvas
				bind:this={canvas}
				class="w-full h-full"
				style="image-rendering: pixelated;"
			></canvas>
		{/if}
	</div>
</div>
