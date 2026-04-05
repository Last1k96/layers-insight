<script lang="ts">
	import {
		getSpatialDims,
		extractSlice,
		computeStats,
		cosineSimilarity,
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

	let batch = $state(0);
	let sortBy: 'index' | 'psnr' | 'mse' | 'cosine' | 'maxdiff' | 'relL2' = $state('psnr');
	let sortAsc = $state(true);

	let dims = $derived(getSpatialDims(shape));

	interface ChannelMetrics {
		idx: number;
		mse: number;
		psnr: number;
		sqnr: number;
		cosineSim: number;
		maxAbsDiff: number;
		relL2: number;
		refMean: number;
		mainMean: number;
	}

	let metrics = $derived.by((): ChannelMetrics[] => {
		const result: ChannelMetrics[] = [];
		for (let c = 0; c < dims.channels; c++) {
			const rSlice = extractSlice(ref, shape, batch, c);
			const mSlice = extractSlice(main, shape, batch, c);
			const rData = rSlice.data;
			const mData = mSlice.data;

			let mse = 0, maxAbs = 0, signalPow = 0, noisePow = 0;
			let rMax = -Infinity;
			for (let i = 0; i < rData.length; i++) {
				const d = mData[i] - rData[i];
				mse += d * d;
				noisePow += d * d;
				signalPow += rData[i] * rData[i];
				const a = Math.abs(d);
				if (a > maxAbs) maxAbs = a;
				if (rData[i] > rMax) rMax = rData[i];
			}
			mse /= rData.length || 1;

			const rStats = computeStats(rData);
			const mStats = computeStats(mData);

			// PSNR
			const maxVal = Math.max(Math.abs(rStats.min), Math.abs(rStats.max)) || 1;
			const psnr = mse > 0 ? 10 * Math.log10(maxVal * maxVal / mse) : Infinity;

			// SQNR
			const sqnr = noisePow > 0 ? 10 * Math.log10(signalPow / noisePow) : Infinity;

			// Cosine similarity
			const cosSim = cosineSimilarity(rData, mData);

			// Relative L2
			const refNorm = Math.sqrt(signalPow);
			const relL2 = refNorm > 0 ? Math.sqrt(noisePow) / refNorm : 0;

			result.push({
				idx: c,
				mse,
				psnr,
				sqnr,
				cosineSim: cosSim,
				maxAbsDiff: maxAbs,
				relL2,
				refMean: rStats.mean,
				mainMean: mStats.mean,
			});
		}
		return result;
	});

	let sortedMetrics = $derived.by(() => {
		const arr = [...metrics];
		const dir = sortAsc ? 1 : -1;
		arr.sort((a, b) => {
			switch (sortBy) {
				case 'psnr': return dir * (a.psnr - b.psnr);
				case 'mse': return dir * (a.mse - b.mse);
				case 'cosine': return dir * (a.cosineSim - b.cosineSim);
				case 'maxdiff': return dir * (a.maxAbsDiff - b.maxAbsDiff);
				case 'relL2': return dir * (a.relL2 - b.relL2);
				default: return dir * (a.idx - b.idx);
			}
		});
		return arr;
	});

	// Aggregate metrics
	let aggregate = $derived.by(() => {
		if (metrics.length === 0) return null;
		let totalMse = 0, worstPsnr = Infinity, worstCosine = 1, worstMaxDiff = 0;
		for (const m of metrics) {
			totalMse += m.mse;
			if (m.psnr < worstPsnr) worstPsnr = m.psnr;
			if (m.cosineSim < worstCosine) worstCosine = m.cosineSim;
			if (m.maxAbsDiff > worstMaxDiff) worstMaxDiff = m.maxAbsDiff;
		}
		return {
			avgMse: totalMse / metrics.length,
			worstPsnr,
			worstCosine,
			worstMaxDiff,
		};
	});

	function psnrColor(psnr: number): string {
		if (!isFinite(psnr)) return 'text-green-400';
		if (psnr > 40) return 'text-green-400';
		if (psnr > 30) return 'text-yellow-400';
		return 'text-red-400';
	}

	function cosineColor(cos: number): string {
		if (cos > 0.9999) return 'text-green-400';
		if (cos > 0.999) return 'text-yellow-400';
		return 'text-red-400';
	}

	function toggleSort(col: typeof sortBy) {
		if (sortBy === col) sortAsc = !sortAsc;
		else { sortBy = col; sortAsc = col === 'psnr' || col === 'cosine' ? true : true; }
	}

	// Bar chart canvas
	let barCanvas: HTMLCanvasElement;

	function drawBars() {
		if (!barCanvas || metrics.length === 0) return;
		const ctx = barCanvas.getContext('2d');
		if (!ctx) return;
		const dw = barCanvas.clientWidth, dh = barCanvas.clientHeight;
		barCanvas.width = dw; barCanvas.height = dh;

		ctx.clearRect(0, 0, dw, dh);

		const margin = { left: 40, right: 10, top: 10, bottom: 20 };
		const plotW = dw - margin.left - margin.right;
		const plotH = dh - margin.top - margin.bottom;
		const barW = Math.max(1, plotW / metrics.length - 1);

		// Draw PSNR bars
		let maxPsnr = 0;
		for (const m of metrics) {
			const p = isFinite(m.psnr) ? m.psnr : 100;
			if (p > maxPsnr) maxPsnr = p;
		}
		maxPsnr = Math.max(maxPsnr, 50);

		for (let i = 0; i < sortedMetrics.length; i++) {
			const m = sortedMetrics[i];
			const p = isFinite(m.psnr) ? m.psnr : maxPsnr;
			const h = (p / maxPsnr) * plotH;
			const x = margin.left + i * (barW + 1);
			const y = margin.top + plotH - h;

			if (p > 40) ctx.fillStyle = '#34d399';
			else if (p > 30) ctx.fillStyle = '#fbbf24';
			else ctx.fillStyle = '#f87171';

			ctx.fillRect(x, y, barW, h);
		}

		// Threshold lines
		ctx.strokeStyle = 'rgba(255,255,255,0.2)';
		ctx.lineWidth = 1;
		ctx.setLineDash([4, 4]);
		for (const threshold of [30, 40]) {
			const y = margin.top + plotH - (threshold / maxPsnr) * plotH;
			ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(dw - margin.right, y); ctx.stroke();
			ctx.fillStyle = '#888'; ctx.font = '9px monospace'; ctx.textAlign = 'right';
			ctx.fillText(`${threshold}dB`, margin.left - 4, y + 3);
		}
		ctx.setLineDash([]);

		// Axis
		ctx.fillStyle = '#888'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
		ctx.fillText('PSNR per channel', margin.left + plotW / 2, dh - 2);
	}

	$effect(() => { sortedMetrics; drawBars(); });
</script>

<div class="flex flex-col gap-3 relative h-full overflow-auto">
	<div class="flex flex-wrap gap-4 items-center text-xs">
		{#if dims.batches > 1}
			<label class="flex items-center gap-2">
				<span class="text-gray-400 shrink-0">Batch:</span>
				<input use:rangeScroll type="range" min="0" max={dims.batches - 1} bind:value={batch} class="w-20" />
				<span class="text-gray-300 w-6 shrink-0">{batch}</span>
			</label>
		{/if}
		<span class="text-gray-500">{dims.channels} channels</span>
	</div>

	{#if aggregate}
		<div class="flex gap-6 text-xs bg-surface-panel p-2 rounded border border-edge">
			<span><span class="text-gray-500">Avg MSE:</span> {formatValue(aggregate.avgMse)}</span>
			<span><span class="text-gray-500">Worst PSNR:</span> <span class={psnrColor(aggregate.worstPsnr)}>{isFinite(aggregate.worstPsnr) ? aggregate.worstPsnr.toFixed(1) + ' dB' : 'Inf'}</span></span>
			<span><span class="text-gray-500">Worst Cosine:</span> <span class={cosineColor(aggregate.worstCosine)}>{aggregate.worstCosine.toFixed(6)}</span></span>
			<span><span class="text-gray-500">Worst Max|Diff|:</span> {formatValue(aggregate.worstMaxDiff)}</span>
		</div>
	{/if}

	<!-- PSNR bar chart -->
	<div class="h-32 bg-surface-base rounded border border-edge shrink-0">
		<canvas bind:this={barCanvas} class="w-full h-full"></canvas>
	</div>

	<!-- Metrics table -->
	<div class="flex-1 overflow-auto min-h-0">
		<table class="w-full text-xs text-gray-300">
			<thead class="sticky top-0 bg-surface-panel">
				<tr class="text-gray-500 text-left">
					<th class="px-2 py-1 cursor-pointer hover:text-gray-300" onclick={() => toggleSort('index')}>Ch {sortBy === 'index' ? (sortAsc ? '↑' : '↓') : ''}</th>
					<th class="px-2 py-1 cursor-pointer hover:text-gray-300" onclick={() => toggleSort('psnr')}>PSNR (dB) {sortBy === 'psnr' ? (sortAsc ? '↑' : '↓') : ''}</th>
					<th class="px-2 py-1">SQNR (dB)</th>
					<th class="px-2 py-1 cursor-pointer hover:text-gray-300" onclick={() => toggleSort('mse')}>MSE {sortBy === 'mse' ? (sortAsc ? '↑' : '↓') : ''}</th>
					<th class="px-2 py-1 cursor-pointer hover:text-gray-300" onclick={() => toggleSort('cosine')}>Cosine {sortBy === 'cosine' ? (sortAsc ? '↑' : '↓') : ''}</th>
					<th class="px-2 py-1 cursor-pointer hover:text-gray-300" onclick={() => toggleSort('maxdiff')}>Max|Diff| {sortBy === 'maxdiff' ? (sortAsc ? '↑' : '↓') : ''}</th>
					<th class="px-2 py-1 cursor-pointer hover:text-gray-300" onclick={() => toggleSort('relL2')}>Rel L2 {sortBy === 'relL2' ? (sortAsc ? '↑' : '↓') : ''}</th>
					<th class="px-2 py-1">Ref Mean</th>
					<th class="px-2 py-1">{mainLabel} Mean</th>
				</tr>
			</thead>
			<tbody>
				{#each sortedMetrics as m (m.idx)}
					<tr class="border-t border-edge/30 hover:bg-surface-hover">
						<td class="px-2 py-1 font-mono">{m.idx}</td>
						<td class="px-2 py-1 font-mono {psnrColor(m.psnr)}">{isFinite(m.psnr) ? m.psnr.toFixed(1) : 'Inf'}</td>
						<td class="px-2 py-1 font-mono {psnrColor(m.sqnr)}">{isFinite(m.sqnr) ? m.sqnr.toFixed(1) : 'Inf'}</td>
						<td class="px-2 py-1 font-mono">{formatValue(m.mse)}</td>
						<td class="px-2 py-1 font-mono {cosineColor(m.cosineSim)}">{m.cosineSim.toFixed(6)}</td>
						<td class="px-2 py-1 font-mono">{formatValue(m.maxAbsDiff)}</td>
						<td class="px-2 py-1 font-mono">{formatValue(m.relL2)}</td>
						<td class="px-2 py-1 font-mono text-gray-500">{formatValue(m.refMean)}</td>
						<td class="px-2 py-1 font-mono text-gray-500">{formatValue(m.mainMean)}</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
</div>
