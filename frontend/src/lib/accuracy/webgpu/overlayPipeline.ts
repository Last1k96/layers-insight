/**
 * Wireframe highlight cube for hovered voxel.
 * 12 edges rendered as thin quads (WebGPU line-list is only 1px).
 */

const WGSL = /* wgsl */ `

struct Camera {
	viewProj: mat4x4<f32>,
	canvasSize: vec2<f32>,
};

struct Highlight {
	origin: vec3<f32>,
	_pad0: f32,
	size: vec3<f32>,
	lineWidth: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> highlight: Highlight;

struct VsOut {
	@builtin(position) pos: vec4<f32>,
};

// 12 edges of a unit cube, each as 2 endpoints
const EDGE_A = array<vec3<f32>, 12>(
	vec3(0,0,0), vec3(0,1,0), vec3(0,0,1), vec3(0,1,1), // X edges
	vec3(0,0,0), vec3(1,0,0), vec3(0,0,1), vec3(1,0,1), // Y edges
	vec3(0,0,0), vec3(1,0,0), vec3(0,1,0), vec3(1,1,0), // Z edges
);
const EDGE_B = array<vec3<f32>, 12>(
	vec3(1,0,0), vec3(1,1,0), vec3(1,0,1), vec3(1,1,1),
	vec3(0,1,0), vec3(1,1,0), vec3(0,1,1), vec3(1,1,1),
	vec3(0,0,1), vec3(1,0,1), vec3(0,1,1), vec3(1,1,1),
);

@vertex
fn vs(@builtin(vertex_index) vid: u32, @builtin(instance_index) edgeIdx: u32) -> VsOut {
	var out: VsOut;

	// 6 vertices per edge (quad = 2 triangles)
	let quadVert = vid % 6u;
	// Side: 0 or 1
	let sideMap = array<f32, 6>(0.0, 0.0, 1.0, 0.0, 1.0, 1.0);
	// Offset direction: -1 or +1
	let offsetMap = array<f32, 6>(-1.0, 1.0, 1.0, -1.0, -1.0, 1.0);
	// Wait, we need: tri0 = (A-,A+,B+), tri1 = (A-,B+,B-)
	// So side=0,0,1,0,1,1 and offset=-1,+1,+1,-1,-1,+1... no
	// Let me use: 6 verts defining a quad from A to B with width
	let side = sideMap[quadVert]; // 0=endpoint A, 1=endpoint B
	let offsetDir = offsetMap[quadVert]; // -1 or +1 perpendicular

	let a3d = EDGE_A[edgeIdx] * highlight.size + highlight.origin;
	let b3d = EDGE_B[edgeIdx] * highlight.size + highlight.origin;

	let aClip = camera.viewProj * vec4(a3d, 1.0);
	let bClip = camera.viewProj * vec4(b3d, 1.0);

	// Screen-space direction
	let aScreen = aClip.xy / aClip.w * camera.canvasSize * 0.5;
	let bScreen = bClip.xy / bClip.w * camera.canvasSize * 0.5;
	let dir = normalize(bScreen - aScreen);
	let perp = vec2(-dir.y, dir.x);

	// Pixel offset in clip space
	let pixelOffset = perp * offsetDir * highlight.lineWidth / camera.canvasSize;

	let p = mix(aClip, bClip, side);
	out.pos = vec4(p.xy + pixelOffset * p.w * 2.0, p.z, p.w);

	return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
	return vec4(1.0, 1.0, 1.0, 1.0);
}
`;

export interface OverlayPipelineState {
	pipeline: GPURenderPipeline;
	cameraBuffer: GPUBuffer; // shared with voxel pipeline
	highlightBuffer: GPUBuffer;
	bindGroup: GPUBindGroup;
}

export function createOverlayPipeline(
	device: GPUDevice,
	format: GPUTextureFormat,
	cameraBuffer: GPUBuffer,
): OverlayPipelineState {
	const shaderModule = device.createShaderModule({ code: WGSL });

	const bindGroupLayout = device.createBindGroupLayout({
		entries: [
			{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
			{ binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
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

	// Highlight uniform: origin(3)+pad + size(3)+lineWidth = 32 bytes
	const highlightBuffer = device.createBuffer({
		size: 32,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	const bindGroup = device.createBindGroup({
		layout: bindGroupLayout,
		entries: [
			{ binding: 0, resource: { buffer: cameraBuffer } },
			{ binding: 1, resource: { buffer: highlightBuffer } },
		],
	});

	return { pipeline, cameraBuffer, highlightBuffer, bindGroup };
}
