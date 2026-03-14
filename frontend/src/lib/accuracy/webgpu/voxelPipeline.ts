/**
 * Voxel instanced rendering pipeline.
 *
 * Each voxel instance draws 3 visible faces (18 vertices = 3 quads x 6 verts).
 * Position is derived from instance_index; the storage buffer holds only the
 * raw diff Float32Array (4 bytes per voxel).
 */

const WGSL = /* wgsl */ `

// Camera uniforms
struct Camera {
	viewProj: mat4x4<f32>,
	canvasSize: vec2<f32>,
};

// Filter / volume uniforms
struct Filter {
	threshold: f32,
	opacity: f32,
	minVal: f32,
	maxVal: f32,

	sliceCMin: f32,
	sliceCMax: f32,
	sliceHMin: f32,
	sliceHMax: f32,

	sliceWMin: f32,
	sliceWMax: f32,
	volumeC: f32,
	volumeH: f32,

	volumeW: f32,
	voxW: f32,
	voxH: f32,
	voxD: f32,

	halfOrigW: f32,
	halfOrigH: f32,
	halfDepth: f32,
	camDirX: f32,

	camDirY: f32,
	camDirZ: f32,
	alphaPower: f32,
	_pad1: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: Filter;
@group(0) @binding(2) var<storage, read> diffData: array<f32>;
@group(0) @binding(3) var colormapTex: texture_2d<f32>;
@group(0) @binding(4) var colormapSampler: sampler;
@group(0) @binding(5) var<storage, read> sortOrder: array<u32>;

struct VsOut {
	@builtin(position) pos: vec4<f32>,
	@location(0) norm: f32,
	@location(1) brightness: f32,
	@location(2) alpha: f32,
};

// Face corner offsets: 3 faces x 6 vertices x 3 components
// Face 0: X-face, Face 1: Y-face, Face 2: Z-face
// Each face is a quad: 2 triangles = 6 vertices
// Offsets are in [0,1] range, will be scaled by vox size

// Corner table for a unit quad on each face
// Face 0 (X): x=fx, yz varies
const FACE_X_CORNERS = array<vec3<f32>, 6>(
	vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0),
	vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), vec3(1.0, 0.0, 1.0),
);
// Face 1 (Y): y=fy, xz varies
const FACE_Y_CORNERS = array<vec3<f32>, 6>(
	vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0),
	vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0), vec3(0.0, 1.0, 1.0),
);
// Face 2 (Z): z=fz, xy varies
const FACE_Z_CORNERS = array<vec3<f32>, 6>(
	vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 1.0), vec3(1.0, 1.0, 1.0),
	vec3(0.0, 0.0, 1.0), vec3(1.0, 1.0, 1.0), vec3(0.0, 1.0, 1.0),
);

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VsOut {
	var out: VsOut;

	let realIdx = sortOrder[instanceIndex];

	let W = u32(params.volumeW);
	let H = u32(params.volumeH);
	let HW = H * W;

	let xi = realIdx % W;
	let yi = (realIdx / W) % H;
	let ci = realIdx / HW;

	// Read diff value
	let val = diffData[realIdx];
	let span = params.maxVal - params.minVal;
	var normVal: f32;
	if (span > 0.0) {
		normVal = (val - params.minVal) / span;
	} else {
		normVal = 0.0;
	}

	// Filter: threshold and slice range
	let fci = f32(ci);
	let fyi = f32(yi);
	let fxi = f32(xi);

	if (normVal < params.threshold
		|| fci < params.sliceCMin || fci > params.sliceCMax
		|| fyi < params.sliceHMin || fyi > params.sliceHMax
		|| fxi < params.sliceWMin || fxi > params.sliceWMax) {
		// Degenerate triangle
		out.pos = vec4(0.0, 0.0, 0.0, 1.0);
		out.norm = 0.0;
		out.brightness = 0.0;
		out.alpha = 0.0;
		return out;
	}

	// Determine face and vertex within face
	let faceIdx = vertexIndex / 6u;
	let vertInFace = vertexIndex % 6u;

	// Camera direction signs determine which faces are visible
	let showPosX = params.camDirX > 0.0;
	let showPosY = params.camDirY > 0.0;
	let showPosZ = params.camDirZ > 0.0;

	var corner: vec3<f32>;
	var brightness: f32;

	if (faceIdx == 0u) {
		// X face
		corner = FACE_X_CORNERS[vertInFace];
		if (!showPosX) { corner.x = 0.0; }
		brightness = 0.8;
	} else if (faceIdx == 1u) {
		// Y face
		corner = FACE_Y_CORNERS[vertInFace];
		if (!showPosY) { corner.y = 0.0; }
		brightness = select(1.0, 0.6, showPosY);
	} else {
		// Z face
		corner = FACE_Z_CORNERS[vertInFace];
		if (!showPosZ) { corner.z = 0.0; }
		brightness = 0.9;
	}

	// World-space position
	let wx = (fxi + corner.x) * params.voxW - params.halfOrigW;
	let wy = (fyi + corner.y) * params.voxH - params.halfOrigH;
	let wz = (fci + corner.z) * params.voxD - params.halfDepth;

	out.pos = camera.viewProj * vec4(wx, wy, wz, 1.0);
	out.norm = normVal;
	out.brightness = brightness;
	out.alpha = params.opacity * pow(normVal, params.alphaPower);

	return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
	if (in.alpha < 0.004) {
		discard;
	}
	let color = textureSample(colormapTex, colormapSampler, vec2(in.norm, 0.5));
	return vec4(color.rgb * in.brightness, in.alpha);
}
`;

export interface VoxelPipelineState {
	pipeline: GPURenderPipeline;
	bindGroupLayout: GPUBindGroupLayout;
	cameraBuffer: GPUBuffer;
	filterBuffer: GPUBuffer;
	diffBuffer: GPUBuffer;
	sortBuffer: GPUBuffer;
	bindGroup: GPUBindGroup;
	diffCapacity: number;
	sortCapacity: number;
}

export function createVoxelPipeline(
	device: GPUDevice,
	format: GPUTextureFormat,
	colormapTexture: GPUTexture,
): VoxelPipelineState {
	const shaderModule = device.createShaderModule({ code: WGSL });

	const bindGroupLayout = device.createBindGroupLayout({
		entries: [
			{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
			{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
			{ binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
			{ binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
			{ binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
			{ binding: 5, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
		],
	});

	const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

	const pipeline = device.createRenderPipeline({
		layout: pipelineLayout,
		vertex: {
			module: shaderModule,
			entryPoint: 'vs',
		},
		fragment: {
			module: shaderModule,
			entryPoint: 'fs',
			targets: [{
				format,
				blend: {
					color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
					alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
				},
			}],
		},
		primitive: { topology: 'triangle-list' },
		multisample: { count: 4 },
	});

	// Camera uniform: mat4x4 (64) + vec2 (8) + pad (8) = 80 bytes
	const cameraBuffer = device.createBuffer({
		size: 80,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	// Filter uniform: 24 floats = 96 bytes
	const filterBuffer = device.createBuffer({
		size: 96,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	// Initial storage buffers
	const initCap = 4096;
	const diffBuffer = device.createBuffer({
		size: initCap * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	const sortBuffer = device.createBuffer({
		size: initCap * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});

	const sampler = device.createSampler({
		magFilter: 'linear',
		minFilter: 'linear',
	});

	const bindGroup = device.createBindGroup({
		layout: bindGroupLayout,
		entries: [
			{ binding: 0, resource: { buffer: cameraBuffer } },
			{ binding: 1, resource: { buffer: filterBuffer } },
			{ binding: 2, resource: { buffer: diffBuffer } },
			{ binding: 3, resource: colormapTexture.createView() },
			{ binding: 4, resource: sampler },
			{ binding: 5, resource: { buffer: sortBuffer } },
		],
	});

	return {
		pipeline,
		bindGroupLayout,
		cameraBuffer,
		filterBuffer,
		diffBuffer,
		sortBuffer,
		bindGroup,
		diffCapacity: initCap,
		sortCapacity: initCap,
	};
}

/** Rebuild bind group (needed when buffers or textures change). */
export function rebuildBindGroup(
	device: GPUDevice,
	state: VoxelPipelineState,
	colormapTexture: GPUTexture,
): GPUBindGroup {
	const sampler = device.createSampler({
		magFilter: 'linear',
		minFilter: 'linear',
	});
	return device.createBindGroup({
		layout: state.bindGroupLayout,
		entries: [
			{ binding: 0, resource: { buffer: state.cameraBuffer } },
			{ binding: 1, resource: { buffer: state.filterBuffer } },
			{ binding: 2, resource: { buffer: state.diffBuffer } },
			{ binding: 3, resource: colormapTexture.createView() },
			{ binding: 4, resource: sampler },
			{ binding: 5, resource: { buffer: state.sortBuffer } },
		],
	});
}
