/**
 * Creates a 256x1 RGBA8Unorm texture by interpolating colormap stops.
 */
export function createColormapTexture(
	device: GPUDevice,
	stops: [number, number, number][],
): GPUTexture {
	const width = 256;
	const data = new Uint8Array(width * 4);

	for (let i = 0; i < width; i++) {
		const t = i / (width - 1);
		const segment = t * (stops.length - 1);
		const idx = Math.min(Math.floor(segment), stops.length - 2);
		const local = segment - idx;
		const a = stops[idx];
		const b = stops[idx + 1];

		data[i * 4 + 0] = Math.round(a[0] + (b[0] - a[0]) * local);
		data[i * 4 + 1] = Math.round(a[1] + (b[1] - a[1]) * local);
		data[i * 4 + 2] = Math.round(a[2] + (b[2] - a[2]) * local);
		data[i * 4 + 3] = 255;
	}

	const texture = device.createTexture({
		size: { width, height: 1 },
		format: 'rgba8unorm',
		usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
	});

	device.queue.writeTexture(
		{ texture },
		data,
		{ bytesPerRow: width * 4 },
		{ width, height: 1 },
	);

	return texture;
}
