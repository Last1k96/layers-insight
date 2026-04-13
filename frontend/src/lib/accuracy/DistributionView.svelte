<script lang="ts">
	import ScatterPlot from './ScatterPlot.svelte';
	import QQPlot from './QQPlot.svelte';

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

	let mode: 'scatter' | 'qq' = $state('scatter');
</script>

<div class="flex flex-col gap-3 h-full">
	<div class="flex items-center gap-1 shrink-0">
		<button
			class="px-2.5 py-1 rounded text-xs border border-edge"
			class:bg-accent={mode === 'scatter'} class:text-white={mode === 'scatter'}
			class:text-gray-400={mode !== 'scatter'}
			onclick={() => mode = 'scatter'}
		>Scatter Plot</button>
		<button
			class="px-2.5 py-1 rounded text-xs border border-edge"
			class:bg-accent={mode === 'qq'} class:text-white={mode === 'qq'}
			class:text-gray-400={mode !== 'qq'}
			onclick={() => mode = 'qq'}
		>Q-Q Plot</button>
	</div>

	<div class="flex-1 min-h-0">
		{#if mode === 'scatter'}
			<ScatterPlot {main} {ref} {shape} {mainLabel} {refLabel} />
		{:else}
			<QQPlot {main} {ref} {shape} {mainLabel} {refLabel} />
		{/if}
	</div>
</div>
