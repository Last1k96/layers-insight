<script lang="ts">
	import ChannelView from './ChannelView.svelte';
	import MetricsDashboard from './MetricsDashboard.svelte';

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

	let mode: 'channel' | 'metrics' = $state('channel');
</script>

<div class="flex flex-col gap-3 h-full">
	<div class="flex items-center gap-1 shrink-0">
		<button
			class="px-2.5 py-1 rounded text-xs border border-edge"
			class:bg-accent={mode === 'channel'} class:text-white={mode === 'channel'}
			class:text-gray-400={mode !== 'channel'}
			onclick={() => mode = 'channel'}
		>Per-Channel</button>
		<button
			class="px-2.5 py-1 rounded text-xs border border-edge"
			class:bg-accent={mode === 'metrics'} class:text-white={mode === 'metrics'}
			class:text-gray-400={mode !== 'metrics'}
			onclick={() => mode = 'metrics'}
		>Metrics</button>
	</div>

	<div class="flex-1 min-h-0">
		{#if mode === 'channel'}
			<ChannelView {main} {ref} {shape} {mainLabel} {refLabel} />
		{:else}
			<MetricsDashboard {main} {ref} {shape} {mainLabel} {refLabel} />
		{/if}
	</div>
</div>
