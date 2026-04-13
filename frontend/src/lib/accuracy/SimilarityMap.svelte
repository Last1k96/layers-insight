<script lang="ts">
	import SSIMMap from './SSIMMap.svelte';
	import SpatialCosine from './SpatialCosine.svelte';
	import { getSpatialDims } from './tensorUtils';

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

	let dims = $derived(getSpatialDims(shape));
	let hasSpatial = $derived(dims.height > 1 && dims.width > 1);
	let hasMultiChannel = $derived(dims.channels > 1);

	let mode: 'ssim' | 'cosine' = $state('ssim');
</script>

<div class="flex flex-col gap-3 h-full">
	<div class="flex items-center gap-1 shrink-0">
		{#if hasSpatial}
			<button
				class="px-2.5 py-1 rounded text-xs border border-edge"
				class:bg-accent={mode === 'ssim'} class:text-white={mode === 'ssim'}
				class:text-gray-400={mode !== 'ssim'}
				onclick={() => mode = 'ssim'}
			>SSIM</button>
		{/if}
		{#if hasMultiChannel}
			<button
				class="px-2.5 py-1 rounded text-xs border border-edge"
				class:bg-accent={mode === 'cosine'} class:text-white={mode === 'cosine'}
				class:text-gray-400={mode !== 'cosine'}
				onclick={() => mode = 'cosine'}
			>Spatial Cosine</button>
		{/if}
	</div>

	<div class="flex-1 min-h-0">
		{#if mode === 'ssim' && hasSpatial}
			<SSIMMap {main} {ref} {shape} {mainLabel} {refLabel} />
		{:else if mode === 'cosine' && hasMultiChannel}
			<SpatialCosine {main} {ref} {shape} {mainLabel} {refLabel} />
		{/if}
	</div>
</div>
