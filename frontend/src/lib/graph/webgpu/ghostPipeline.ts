/**
 * Screen-space WebGPU pipeline for rendering ghost node indicators.
 * Ghost nodes are small arrows + labels at viewport edges pointing to off-screen
 * nodes connected to the currently selected edge.
 */
import type { TextAtlasData } from './textAtlas';
import { GHOST_VERTEX_FLOATS, GHOST_VERTEX_BYTES, ALPHA_BLEND } from './types';

const SHADER = /* wgsl */ `
@group(0) @binding(0) var<uniform> viewport: vec2<f32>;
@group(0) @binding(1) var atlasTex: texture_2d<f32>;
@group(0) @binding(2) var atlasSampler: sampler;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) isText: f32,
};

@vertex
fn vertexMain(
  @location(0) pos: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) color: vec4<f32>,
  @location(3) isText: f32,
) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4(
    pos.x / viewport.x * 2.0 - 1.0,
    1.0 - pos.y / viewport.y * 2.0,
    0.0, 1.0
  );
  out.uv = uv;
  out.color = color;
  out.isText = isText;
  return out;
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4<f32> {
  // Use textureSampleLevel (explicit LOD) instead of textureSample to avoid
  // the uniform-control-flow requirement — isText varies per fragment.
  let texAlpha = textureSampleLevel(atlasTex, atlasSampler, in.uv, 0.0).a;
  let alpha = select(in.color.a, texAlpha * in.color.a, in.isText > 0.5);
  if (alpha < 0.01) { discard; }
  return vec4(in.color.rgb, alpha);
}
`;

export interface GhostPipelineState {
  pipeline: GPURenderPipeline;
  vertexBuffer: GPUBuffer;
  viewportBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  vertexCount: number;
  capacity: number;
}

export function createGhostPipeline(
  device: GPUDevice,
  format: GPUTextureFormat,
  atlas: TextAtlasData,
): GhostPipelineState {
  const module = device.createShaderModule({ code: SHADER });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: {
      module,
      entryPoint: 'vertexMain',
      buffers: [{
        arrayStride: GHOST_VERTEX_BYTES,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x2' },   // pos
          { shaderLocation: 1, offset: 8, format: 'float32x2' },   // uv
          { shaderLocation: 2, offset: 16, format: 'float32x4' },  // color
          { shaderLocation: 3, offset: 32, format: 'float32' },    // isText
        ],
      }],
    },
    fragment: {
      module,
      entryPoint: 'fragmentMain',
      targets: [{ format, blend: ALPHA_BLEND }],
    },
    primitive: { topology: 'triangle-list' },
    multisample: { count: 4 },
  });

  const viewportBuffer = device.createBuffer({
    size: 8, // vec2<f32>
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
  });

  const initialCapacity = 256;
  const vertexBuffer = device.createBuffer({
    size: initialCapacity * GHOST_VERTEX_BYTES,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: viewportBuffer } },
      { binding: 1, resource: atlas.texture.createView() },
      { binding: 2, resource: sampler },
    ],
  });

  return { pipeline, vertexBuffer, viewportBuffer, bindGroup, vertexCount: 0, capacity: initialCapacity };
}

export function updateGhostVertices(
  state: GhostPipelineState,
  device: GPUDevice,
  atlas: TextAtlasData,
  data: Float32Array,
  vertexCount: number,
): GhostPipelineState {
  if (vertexCount > state.capacity) {
    state.vertexBuffer.destroy();
    const newCapacity = Math.max(vertexCount, state.capacity * 2);
    const vertexBuffer = device.createBuffer({
      size: newCapacity * GHOST_VERTEX_BYTES,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    const bindGroup = device.createBindGroup({
      layout: state.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state.viewportBuffer } },
        { binding: 1, resource: atlas.texture.createView() },
        { binding: 2, resource: sampler },
      ],
    });

    state = { ...state, vertexBuffer, bindGroup, capacity: newCapacity };
  }

  if (vertexCount > 0) {
    device.queue.writeBuffer(state.vertexBuffer, 0, data as Float32Array<ArrayBuffer>, 0, vertexCount * GHOST_VERTEX_FLOATS);
  }
  state.vertexCount = vertexCount;
  return state;
}

export function updateGhostViewport(
  state: GhostPipelineState,
  device: GPUDevice,
  width: number,
  height: number,
): void {
  device.queue.writeBuffer(state.viewportBuffer, 0, new Float32Array([width, height]) as Float32Array<ArrayBuffer>);
}

export function drawGhosts(pass: GPURenderPassEncoder, state: GhostPipelineState): void {
  if (state.vertexCount === 0) return;
  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, state.bindGroup);
  pass.setVertexBuffer(0, state.vertexBuffer);
  pass.draw(state.vertexCount);
}
