<script lang="ts">
	import ChannelView from './ChannelView.svelte';
	import MetricsDashboard from './MetricsDashboard.svelte';
	import DensityPlot from './DensityPlot.svelte';
	import { getSpatialDims, extractSlice, cosineSimilarity, formatValue } from './tensorUtils';
	import { rangeScroll } from './rangeScroll';
	import { keyboardNav } from './keyboardNav';

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

	let mode: 'density' | 'channel' | 'metrics' = $state('density');
	let channelIndex = $state(0);
	let batch = $state(0);
	let densityBins = $state(128);
	const BIN_OPTIONS = [32, 64, 128, 256] as const;

	let dims = $derived(getSpatialDims(shape));

	function getChannelData(tensor: Float32Array, ch: number): Float32Array {
		return extractSlice(tensor, shape, batch, ch).data;
	}

	let mainChannel = $derived(getChannelData(main, channelIndex));
	let refChannel = $derived(getChannelData(ref, channelIndex));
	let cosSim = $derived(cosineSimilarity(mainChannel, refChannel));

	// Per-channel summary for the table (order matches node status panel: MSE, Max Abs Diff, Cosine Sim)
	let channelSummary = $derived.by(() => {
		const summary: { ch: number; mse: number; maxAbsDiff: number; cosSim: number }[] = [];
		for (let c = 0; c < dims.channels; c++) {
			const m = getChannelData(main, c);
			const r = getChannelData(ref, c);
			let sumSqDiff = 0, maxD = 0;
			for (let i = 0; i < m.length; i++) {
				const d = m[i] - r[i];
				sumSqDiff += d * d;
				const ad = d < 0 ? -d : d;
				if (ad > maxD) maxD = ad;
			}
			summary.push({ ch: c, mse: sumSqDiff / m.length, maxAbsDiff: maxD, cosSim: cosineSimilarity(m, r) });
		}
		return summary;
	});

	// Sort state
	type SortKey = 'ch' | 'mse' | 'maxAbsDiff' | 'cosSim';
	let sortKey = $state<SortKey>('mse');
	let sortAsc = $state(false);

	function toggleSort(key: SortKey) {
		if (sortKey === key) sortAsc = !sortAsc;
		else { sortKey = key; sortAsc = key === 'ch' || key === 'cosSim'; }
	}

	let sortedSummary = $derived.by(() => {
		const sorted = [...channelSummary];
		const dir = sortAsc ? 1 : -1;
		sorted.sort((a, b) => (a[sortKey] - b[sortKey]) * dir);
		return sorted;
	});

	let columnRanges = $derived.by(() => {
		if (channelSummary.length === 0) return { mse: [0, 1] as [number, number], maxAbsDiff: [0, 1] as [number, number], cosSim: [0, 1] as [number, number] };
		let mseMin = Infinity, mseMax = -Infinity, maxMin = Infinity, maxMax = -Infinity, cosMin = Infinity, cosMax = -Infinity;
		for (const row of channelSummary) {
			if (row.mse < mseMin) mseMin = row.mse;
			if (row.mse > mseMax) mseMax = row.mse;
			if (row.maxAbsDiff < maxMin) maxMin = row.maxAbsDiff;
			if (row.maxAbsDiff > maxMax) maxMax = row.maxAbsDiff;
			if (row.cosSim < cosMin) cosMin = row.cosSim;
			if (row.cosSim > cosMax) cosMax = row.cosSim;
		}
		return { mse: [mseMin, mseMax] as [number, number], maxAbsDiff: [maxMin, maxMax] as [number, number], cosSim: [cosMin, cosMax] as [number, number] };
	});

	function errorColor(value: number, range: [number, number]): string {
		const [lo, hi] = range;
		const span = hi - lo;
		if (span === 0) return 'rgb(156, 163, 175)';
		const t = (value - lo) / span;
		return `rgb(${Math.round(55 + t * 200)}, ${Math.round(200 - t * 160)}, 50)`;
	}

	function cosSimColor(value: number, range: [number, number]): string {
		const [lo, hi] = range;
		const span = hi - lo;
		if (span === 0) return 'rgb(156, 163, 175)';
		const t = (value - lo) / span;
		return `rgb(${Math.round(220 - t * 170)}, ${Math.round(60 + t * 150)}, 50)`;
	}
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div class="flex flex-col gap-3 h-full" tabindex="0" use:keyboardNav={{
	onNextChannel: () => { channelIndex = Math.min(channelIndex + 1, dims.channels - 1); },
	onPrevChannel: () => { channelIndex = Math.max(channelIndex - 1, 0); },
}}>
	<!-- Sub-tabs + batch -->
	<div class="flex items-center gap-2 shrink-0">
		<div class="flex items-center gap-1">
			{#each [['density', 'Density'], ['channel', 'Per-Channel'], ['metrics', 'Metrics']] as [m, label]}
				<button
					class="px-2.5 py-1 rounded text-xs border border-edge"
					class:bg-accent={mode === m} class:text-white={mode === m}
					class:text-gray-400={mode !== m}
					onclick={() => mode = m as typeof mode}
				>{label}</button>
			{/each}
		</div>
		{#if dims.batches > 1}
			<span class="border-l border-gray-600 h-4"></span>
			<div class="flex items-center gap-1 text-xs text-gray-300">
				<span class="text-gray-400">Batch:</span>
				<input use:rangeScroll type="range" min="0" max={dims.batches - 1} bind:value={batch} class="w-20" />
				<span>{batch} / {dims.batches}</span>
			</div>
		{/if}
	</div>

	<!-- Content -->
	<div class="flex-1 min-h-0">
		{#if mode === 'density'}
			<div class="h-full flex flex-col gap-3">
				<!-- Channel selector + bins -->
				<div class="flex items-center gap-2 text-xs shrink-0">
					{#if dims.channels > 1}
						<span class="text-gray-400 shrink-0">Channel:</span>
						<input use:rangeScroll type="range" min="0" max={dims.channels - 1} bind:value={channelIndex} class="flex-1" />
						<span class="text-gray-300 shrink-0">{channelIndex} / {dims.channels}</span>
						<span class="border-l border-gray-600 h-4"></span>
					{/if}
					<span class="text-gray-400 shrink-0">Bins:</span>
					<select bind:value={densityBins} class="bg-gray-800 border border-gray-600 rounded px-1.5 py-0.5 text-xs text-gray-200">
						{#each BIN_OPTIONS as b}
							<option value={b}>{b}</option>
						{/each}
					</select>
				</div>

				<!-- Big density plot + stats -->
				<div class="flex gap-4 shrink-0">
					<div class="flex-1 bg-surface-base/50 rounded-lg p-3" style="height: 426px; min-height: 200px; resize: vertical; overflow: hidden;">
						<DensityPlot main={mainChannel} ref={refChannel} bins={densityBins} />
					</div>
					<div class="w-[160px] shrink-0 text-xs text-gray-400 space-y-2 pt-1">
						{#if channelSummary[channelIndex]}
							<div>
								<div class="text-gray-500 text-[10px] uppercase tracking-wider">MSE</div>
								<div class="font-mono text-lg text-gray-200">{formatValue(channelSummary[channelIndex].mse)}</div>
							</div>
							<div>
								<div class="text-gray-500 text-[10px] uppercase tracking-wider">Max Abs Diff</div>
								<div class="font-mono text-gray-200">{formatValue(channelSummary[channelIndex].maxAbsDiff)}</div>
							</div>
						{/if}
						<div>
							<div class="text-gray-500 text-[10px] uppercase tracking-wider">Cosine Sim</div>
							<div class="font-mono text-lg" class:text-green-400={cosSim > 0.999} class:text-yellow-400={cosSim > 0.99 && cosSim <= 0.999} class:text-red-400={cosSim <= 0.99}>{cosSim.toFixed(6)}</div>
						</div>
						<div>
							<div class="text-gray-500 text-[10px] uppercase tracking-wider">Channel</div>
							<div class="font-mono text-gray-200">{channelIndex} / {dims.channels}</div>
						</div>
						<div>
							<div class="text-gray-500 text-[10px] uppercase tracking-wider">Spatial</div>
							<div class="font-mono text-gray-200">{dims.height}&times;{dims.width}</div>
						</div>
					</div>
				</div>

				<!-- Summary table -->
				{#if dims.channels > 1}
					<div class="flex-1 min-h-0 flex flex-col">
						<div class="flex-1 min-h-0 overflow-y-auto rounded border border-edge">
							<table class="w-full text-xs table-fixed">
								<colgroup>
									<col class="w-[50px]" />
									<col />
									<col />
									<col />
								</colgroup>
								<thead class="sticky top-0 bg-surface-panel">
									<tr class="text-gray-500 border-b border-edge">
										<th class="px-2 py-1 text-left font-medium cursor-pointer hover:text-gray-300 select-none" onclick={() => toggleSort('ch')}>Ch{sortKey === 'ch' ? (sortAsc ? ' \u25B2' : ' \u25BC') : ''}</th>
										<th class="px-2 py-1 text-right font-medium cursor-pointer hover:text-gray-300 select-none" onclick={() => toggleSort('mse')}>MSE{sortKey === 'mse' ? (sortAsc ? ' \u25B2' : ' \u25BC') : ''}</th>
										<th class="px-2 py-1 text-right font-medium cursor-pointer hover:text-gray-300 select-none" onclick={() => toggleSort('maxAbsDiff')}>Max Abs Diff{sortKey === 'maxAbsDiff' ? (sortAsc ? ' \u25B2' : ' \u25BC') : ''}</th>
										<th class="px-2 py-1 text-right font-medium cursor-pointer hover:text-gray-300 select-none" onclick={() => toggleSort('cosSim')}>Cos Sim{sortKey === 'cosSim' ? (sortAsc ? ' \u25B2' : ' \u25BC') : ''}</th>
									</tr>
								</thead>
								<tbody>
									{#each sortedSummary as row (row.ch)}
										<tr
											class="border-b border-edge/50 cursor-pointer hover:bg-white/5 {row.ch === channelIndex ? 'bg-white/10' : ''}"
											onclick={() => channelIndex = row.ch}
										>
											<td class="px-2 py-1 font-mono text-gray-300">{row.ch}</td>
											<td class="px-2 py-1 text-right font-mono" style="color: {errorColor(row.mse, columnRanges.mse)}">{formatValue(row.mse)}</td>
											<td class="px-2 py-1 text-right font-mono" style="color: {errorColor(row.maxAbsDiff, columnRanges.maxAbsDiff)}">{formatValue(row.maxAbsDiff)}</td>
											<td class="px-2 py-1 text-right font-mono" style="color: {cosSimColor(row.cosSim, columnRanges.cosSim)}">{row.cosSim.toFixed(4)}</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					</div>
				{/if}
			</div>
		{:else if mode === 'channel'}
			<ChannelView {main} {ref} {shape} {mainLabel} {refLabel} {batch} bind:channelIndex />
		{:else}
			<MetricsDashboard {main} {ref} {shape} {mainLabel} {refLabel} {batch} />
		{/if}
	</div>
</div>
