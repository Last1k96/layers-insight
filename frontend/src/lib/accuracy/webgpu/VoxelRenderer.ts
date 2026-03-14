/**
 * WebGPU voxel renderer — orchestrates voxel + overlay pipelines.
 *
 * Follows WebGPURenderer.ts patterns: static create(), 4x MSAA,
 * dirty flag + RAF loop, ResizeObserver.
 */

import { createColormapTexture } from './colormapTexture';
import {
	createVoxelPipeline,
	rebuildBindGroup,
	type VoxelPipelineState,
} from './voxelPipeline';
import {
	createOverlayPipeline,
	type OverlayPipelineState,
} from './overlayPipeline';

const CLEAR_COLOR: GPUColor = { r: 0.067, g: 0.094, b: 0.153, a: 1.0 }; // #111827

export class VoxelRenderer {
	private device: GPUDevice;
	private context: GPUCanvasContext;
	private format: GPUTextureFormat;
	private canvas: HTMLCanvasElement;

	private voxelState: VoxelPipelineState;
	private overlayState: OverlayPipelineState;
	private colormapTexture: GPUTexture;

	private msaaTexture: GPUTexture | null = null;
	private msaaView: GPUTextureView | null = null;
	private dirty = true;
	private animFrameId = 0;
	private resizeObserver: ResizeObserver;
	private destroyed = false;

	// Current volume state
	private voxelCount = 0;
	private showHighlight = false;

	// Cached sort orders: 6 possible traversal directions
	private sortOrders: Map<string, Uint32Array> = new Map();
	private currentSortKey = '';
	private volumeC = 0;
	private volumeH = 0;
	private volumeW = 0;

	// Cached camera direction for sort key
	private camDirX = 0;
	private camDirY = 0;
	private camDirZ = 1;

	private constructor(
		device: GPUDevice,
		context: GPUCanvasContext,
		format: GPUTextureFormat,
		canvas: HTMLCanvasElement,
		colormapTexture: GPUTexture,
		voxelState: VoxelPipelineState,
		overlayState: OverlayPipelineState,
	) {
		this.device = device;
		this.context = context;
		this.format = format;
		this.canvas = canvas;
		this.colormapTexture = colormapTexture;
		this.voxelState = voxelState;
		this.overlayState = overlayState;

		this.resizeObserver = new ResizeObserver(() => {
			this.handleResize();
			this.markDirty();
		});
		this.resizeObserver.observe(canvas);

		this.handleResize();
		this.frameLoop();
	}

	static async create(canvas: HTMLCanvasElement): Promise<VoxelRenderer | null> {
		if (!navigator.gpu) {
			console.warn('WebGPU not supported');
			return null;
		}

		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) {
			console.warn('No GPU adapter found');
			return null;
		}

		const device = await adapter.requestDevice();
		const context = canvas.getContext('webgpu');
		if (!context) {
			console.warn('Could not get WebGPU context');
			return null;
		}

		const format = navigator.gpu.getPreferredCanvasFormat();
		context.configure({ device, format, alphaMode: 'opaque' });

		// Default colormap (plasma)
		const defaultStops: [number, number, number][] = [
			[70, 10, 150], [170, 30, 180], [230, 60, 120],
			[250, 130, 50], [250, 210, 60], [240, 250, 130],
		];
		const colormapTexture = createColormapTexture(device, defaultStops);

		const voxelState = createVoxelPipeline(device, format, colormapTexture);
		const overlayState = createOverlayPipeline(device, format, voxelState.cameraBuffer);

		return new VoxelRenderer(
			device, context, format, canvas,
			colormapTexture, voxelState, overlayState,
		);
	}

	private handleResize(): void {
		const dpr = window.devicePixelRatio || 1;
		const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
		const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));

		if (this.canvas.width === w && this.canvas.height === h && this.msaaTexture) return;

		this.canvas.width = w;
		this.canvas.height = h;

		this.msaaTexture?.destroy();
		this.msaaTexture = this.device.createTexture({
			size: { width: w, height: h },
			format: this.format,
			sampleCount: 4,
			usage: GPUTextureUsage.RENDER_ATTACHMENT,
		});
		this.msaaView = this.msaaTexture.createView();
	}

	markDirty(): void {
		this.dirty = true;
	}

	private frameLoop(): void {
		if (this.destroyed) return;
		this.animFrameId = requestAnimationFrame(() => this.frameLoop());
		if (!this.dirty) return;
		this.dirty = false;

		const w = this.canvas.width;
		const h = this.canvas.height;
		if (w === 0 || h === 0 || !this.msaaView) return;

		this.render();
	}

	private render(): void {
		const textureView = this.context.getCurrentTexture().createView();
		const encoder = this.device.createCommandEncoder();

		const pass = encoder.beginRenderPass({
			colorAttachments: [{
				view: this.msaaView!,
				resolveTarget: textureView,
				clearValue: CLEAR_COLOR,
				loadOp: 'clear',
				storeOp: 'discard',
			}],
		});

		// Draw voxels
		if (this.voxelCount > 0) {
			pass.setPipeline(this.voxelState.pipeline);
			pass.setBindGroup(0, this.voxelState.bindGroup);
			pass.draw(18, this.voxelCount); // 3 faces x 6 verts
		}

		// Draw highlight wireframe
		if (this.showHighlight) {
			pass.setPipeline(this.overlayState.pipeline);
			pass.setBindGroup(0, this.overlayState.bindGroup);
			pass.draw(6, 12); // 6 verts per edge quad, 12 edges
		}

		pass.end();
		this.device.queue.submit([encoder.finish()]);
	}

	/** Upload volume diff data and rebuild sort orders. */
	setVolumeData(
		diff: Float32Array,
		C: number, H: number, W: number,
		minVal: number, maxVal: number,
		voxW: number, voxH: number, voxD: number,
	): void {
		this.voxelCount = diff.length;
		this.volumeC = C;
		this.volumeH = H;
		this.volumeW = W;

		// Grow diff buffer if needed
		if (diff.length > this.voxelState.diffCapacity) {
			this.voxelState.diffBuffer.destroy();
			const newCap = Math.max(diff.length, this.voxelState.diffCapacity * 2);
			this.voxelState.diffBuffer = this.device.createBuffer({
				size: newCap * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			this.voxelState.diffCapacity = newCap;
			this.voxelState.bindGroup = rebuildBindGroup(
				this.device, this.voxelState, this.colormapTexture,
			);
		}
		this.device.queue.writeBuffer(this.voxelState.diffBuffer, 0, diff.buffer, diff.byteOffset, diff.byteLength);

		// Grow sort buffer if needed
		if (diff.length > this.voxelState.sortCapacity) {
			this.voxelState.sortBuffer.destroy();
			const newCap = Math.max(diff.length, this.voxelState.sortCapacity * 2);
			this.voxelState.sortBuffer = this.device.createBuffer({
				size: newCap * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			this.voxelState.sortCapacity = newCap;
			this.voxelState.bindGroup = rebuildBindGroup(
				this.device, this.voxelState, this.colormapTexture,
			);
		}

		// Build all 6 sort orders
		this.buildSortOrders(C, H, W);

		// Force re-upload: dimensions changed so cached sort key is stale
		this.currentSortKey = '';
		this.updateSortOrder();

		this.markDirty();
	}

	/** Build 6 cached index buffers for all axis traversal directions. */
	private buildSortOrders(C: number, H: number, W: number): void {
		this.sortOrders.clear();
		const total = C * H * W;

		for (const cFwd of [0, 1]) {
			for (const hFwd of [0, 1]) {
				for (const wFwd of [0, 1]) {
					const key = `${cFwd}${hFwd}${wFwd}`;
					const order = new Uint32Array(total);
					let idx = 0;

					const cStart = cFwd ? 0 : C - 1, cEnd = cFwd ? C : -1, cStep = cFwd ? 1 : -1;
					const hStart = hFwd ? 0 : H - 1, hEnd = hFwd ? H : -1, hStep = hFwd ? 1 : -1;
					const wStart = wFwd ? W - 1 : 0, wEnd = wFwd ? -1 : W, wStep = wFwd ? -1 : 1;

					for (let ci = cStart; ci !== cEnd; ci += cStep) {
						for (let yi = hStart; yi !== hEnd; yi += hStep) {
							for (let xi = wStart; xi !== wEnd; xi += wStep) {
								order[idx++] = ci * H * W + yi * W + xi;
							}
						}
					}

					this.sortOrders.set(key, order);
				}
			}
		}
	}

	/** Select and upload the correct sort order based on camera direction. */
	private updateSortOrder(): void {
		// Back-to-front: traverse from back towards camera
		// cosY > 0 → C increases toward camera → start from 0 (cFwd=1)
		// sinX > 0 → H increases toward camera → start from 0 (hFwd=1)
		// sinY > 0 → W decreases toward camera → start from W-1 (wFwd=1)
		const cFwd = this.camDirZ > 0 ? 1 : 0;
		const hFwd = this.camDirY > 0 ? 1 : 0;
		const wFwd = this.camDirX > 0 ? 1 : 0;
		const key = `${cFwd}${hFwd}${wFwd}`;

		if (key === this.currentSortKey) return;
		this.currentSortKey = key;

		const order = this.sortOrders.get(key);
		if (order) {
			this.device.queue.writeBuffer(this.voxelState.sortBuffer, 0, order.buffer, order.byteOffset, order.byteLength);
		}
	}

	/** Set camera rotation and zoom — builds orthographic viewProj matrix. */
	setCamera(
		rotXDeg: number, rotYDeg: number, zoom: number,
		pivotX: number, pivotY: number, pivotZ: number,
		canvasW: number, canvasH: number,
	): void {
		const radX = rotXDeg * Math.PI / 180;
		const radY = rotYDeg * Math.PI / 180;
		const cosX = Math.cos(radX);
		const sinX = Math.sin(radX);
		const cosY = Math.cos(radY);
		const sinY = Math.sin(radY);

		this.camDirX = sinY;
		this.camDirY = sinX;
		this.camDirZ = cosX * cosY;

		// Compute scale (same as existing code)
		const origW = this.volumeW; // these are already in world-space terms
		const origH = this.volumeH;
		const origC = this.volumeC;
		// We need the actual world-space extents — but those depend on vox sizes
		// which are set via setUniforms. For the matrix, we replicate project3D:
		// screenX = centerX + (x*cosY - z*sinY) * scale
		// screenY = centerY + (y*cosX - (x*sinY + z*cosY)*sinX) * scale
		//
		// This is equivalent to: Ortho * Translate(center) * [rotation matrix] * Scale(scale)
		// But we can just build the 4x4 directly.

		// Build rotation matrix (Y then X):
		// RotY: [cosY 0 -sinY; 0 1 0; sinY 0 cosY]
		// RotX: [1 0 0; 0 cosX -sinX; 0 sinX cosX]
		// Combined = RotX * RotY:
		// row0: cosY, 0, -sinY
		// row1: sinX*sinY, cosX, sinX*cosY
		// row2: cosX*sinY, -sinX, cosX*cosY
		//
		// But our project3D does:
		// rx = x*cosY - z*sinY  (row0 dot [x,y,z] but only x,z)
		// rz = x*sinY + z*cosY
		// ry = y*cosX - rz*sinX = y*cosX - (x*sinY + z*cosY)*sinX
		//
		// So screen coords:
		// sx = centerX + rx * scale = centerX + (x*cosY - z*sinY) * scale
		// sy = centerY + ry * scale = centerY + (y*cosX - x*sinY*sinX - z*cosY*sinX) * scale

		// Pivot offset: centerX = cw/2 - rpx*scale, centerY = ch/2 - rpy*scale
		const rpx = pivotX * cosY - pivotZ * sinY;
		const rpz = pivotX * sinY + pivotZ * cosY;
		const rpy = pivotY * cosX - rpz * sinX;

		// Orthographic projection maps to clip space [-1,1]:
		// clipX = (sx / (cw/2)) - 1 = (centerX + rx*scale) / (cw/2) - 1
		// clipY = -((sy / (ch/2)) - 1) = -(centerY + ry*scale) / (ch/2) + 1  (flip Y)

		// Let's build this as a single mat4x4.
		// clipX = (2/cw) * (cw/2 - rpx*scale + (x*cosY - z*sinY)*scale) - 1
		//       = 1 - (2*rpx*scale)/cw + (2*scale/cw)*(x*cosY - z*sinY) - 1
		//       = (2*scale/cw) * (x*cosY - z*sinY - rpx)
		//
		// clipY = -[(2/ch) * (ch/2 - rpy*scale + (y*cosX - x*sinY*sinX - z*cosY*sinX)*scale) - 1]
		//       = -[(2*scale/ch) * (y*cosX - x*sinY*sinX - z*cosY*sinX - rpy)]
		//       = (2*scale/ch) * (x*sinY*sinX + z*cosY*sinX - y*cosX + rpy)

		const sx = 2 * zoom / canvasW;
		const sy = 2 * zoom / canvasH;

		// Column-major mat4x4
		const m = new Float32Array(20); // 16 for mat4 + 2 for canvasSize + 2 pad

		// Column 0 (x coefficients)
		m[0] = sx * cosY;                    // clipX from x
		m[1] = sy * sinY * sinX;             // clipY from x (negated above)
		m[2] = 0;
		m[3] = 0;

		// Column 1 (y coefficients)
		m[4] = 0;                             // clipX from y
		m[5] = -sy * cosX;                    // clipY from y
		m[6] = 0;
		m[7] = 0;

		// Column 2 (z coefficients)
		m[8] = -sx * sinY;                    // clipX from z
		m[9] = sy * cosY * sinX;              // clipY from z
		m[10] = 0;
		m[11] = 0;

		// Column 3 (translation)
		m[12] = -sx * rpx;                    // clipX offset
		m[13] = sy * rpy;                     // clipY offset
		m[14] = 0;
		m[15] = 1;

		// Canvas size for overlay pipeline
		m[16] = canvasW;
		m[17] = canvasH;
		m[18] = 0; // pad
		m[19] = 0; // pad

		this.device.queue.writeBuffer(this.voxelState.cameraBuffer, 0, m);

		// Write updated camDir to filter buffer (floats 19-21 = bytes 76-87)
		const camDirData = new Float32Array([this.camDirX, this.camDirY, this.camDirZ]);
		this.device.queue.writeBuffer(this.voxelState.filterBuffer, 76, camDirData);

		// Update sort order based on new camera direction
		this.updateSortOrder();

		this.markDirty();
	}

	/** Update filter/volume uniforms. */
	setUniforms(
		threshold: number, opacity: number,
		sliceC: [number, number], sliceH: [number, number], sliceW: [number, number],
		C: number, H: number, W: number,
		voxW: number, voxH: number, voxD: number,
		halfOrigW: number, halfOrigH: number, halfDepth: number,
		minVal: number, maxVal: number,
		alphaPower: number,
	): void {
		const data = new Float32Array(24);
		data[0] = threshold;
		data[1] = opacity;
		data[2] = minVal;
		data[3] = maxVal;

		data[4] = sliceC[0];
		data[5] = sliceC[1];
		data[6] = sliceH[0];
		data[7] = sliceH[1];

		data[8] = sliceW[0];
		data[9] = sliceW[1];
		data[10] = C;
		data[11] = H;

		data[12] = W;
		data[13] = voxW;
		data[14] = voxH;
		data[15] = voxD;

		data[16] = halfOrigW;
		data[17] = halfOrigH;
		data[18] = halfDepth;
		data[19] = this.camDirX;

		data[20] = this.camDirY;
		data[21] = this.camDirZ;
		data[22] = alphaPower;
		data[23] = 0; // pad

		this.device.queue.writeBuffer(this.voxelState.filterBuffer, 0, data);
		this.markDirty();
	}

	/** Update colormap texture. */
	setColormap(stops: [number, number, number][]): void {
		this.colormapTexture.destroy();
		this.colormapTexture = createColormapTexture(this.device, stops);
		this.voxelState.bindGroup = rebuildBindGroup(
			this.device, this.voxelState, this.colormapTexture,
		);
		this.markDirty();
	}

	/** Set hovered voxel wireframe highlight. */
	setHoveredVoxel(
		ox: number, oy: number, oz: number,
		sizeX: number, sizeY: number, sizeZ: number,
	): void {
		const data = new Float32Array(8);
		data[0] = ox;
		data[1] = oy;
		data[2] = oz;
		data[3] = 0; // pad
		data[4] = sizeX;
		data[5] = sizeY;
		data[6] = sizeZ;
		data[7] = 2.0; // lineWidth in pixels
		this.device.queue.writeBuffer(this.overlayState.highlightBuffer, 0, data);
		this.showHighlight = true;
		this.markDirty();
	}

	clearHoveredVoxel(): void {
		this.showHighlight = false;
		this.markDirty();
	}

	destroy(): void {
		this.destroyed = true;
		cancelAnimationFrame(this.animFrameId);
		this.resizeObserver.disconnect();
		this.msaaTexture?.destroy();
		this.colormapTexture.destroy();
		this.voxelState.diffBuffer.destroy();
		this.voxelState.sortBuffer.destroy();
		this.voxelState.cameraBuffer.destroy();
		this.voxelState.filterBuffer.destroy();
		this.overlayState.highlightBuffer.destroy();
		this.device.destroy();
	}
}
