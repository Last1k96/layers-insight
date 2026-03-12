/**
 * WebGPU pipeline for rendering text labels using a bitmap font atlas.
 * Each glyph is rendered as an instanced textured quad.
 */
import type { TextAtlasData } from './textAtlas';
import { GLYPH_FLOATS, GLYPH_BYTES, ALPHA_BLEND } from './types';

const SHADER = /* wgsl */ `
struct GlyphInstance {
  pos: vec2<f32>,
  size: vec2<f32>,
  uvRect: vec4<f32>,
  color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<storage, read> glyphs: array<GlyphInstance>;
@group(0) @binding(2) var atlasTex: texture_2d<f32>;
@group(0) @binding(3) var atlasSampler: sampler;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
};

var<private> quadPositions: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
  vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
  vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0),
);

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let glyph = glyphs[instanceIndex];
  let uv = quadPositions[vertexIndex];

  let worldPos = uv * glyph.size + glyph.pos;

  var out: VertexOutput;
  out.position = camera * vec4(worldPos, 0.0, 1.0);
  // Map UV from unit quad to atlas UV rect
  out.uv = mix(glyph.uvRect.xy, glyph.uvRect.zw, uv);
  out.color = glyph.color;
  return out;
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4<f32> {
  let texColor = textureSample(atlasTex, atlasSampler, in.uv);
  let alpha = texColor.a * in.color.a;
  if (alpha < 0.01) { discard; }
  return vec4(in.color.rgb, alpha);
}
`;

export interface TextPipelineState {
  pipeline: GPURenderPipeline;
  storageBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  instanceCount: number;
  capacity: number;
}

export function createTextPipeline(
  device: GPUDevice,
  format: GPUTextureFormat,
  cameraBuffer: GPUBuffer,
  atlas: TextAtlasData,
): TextPipelineState {
  const module = device.createShaderModule({ code: SHADER });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module, entryPoint: 'vertexMain' },
    fragment: {
      module,
      entryPoint: 'fragmentMain',
      targets: [{ format, blend: ALPHA_BLEND }],
    },
    primitive: { topology: 'triangle-list' },
    multisample: { count: 4 },
  });

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    mipmapFilter: 'linear',
  });

  const initialCapacity = 4096;
  const storageBuffer = device.createBuffer({
    size: initialCapacity * GLYPH_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: { buffer: storageBuffer } },
      { binding: 2, resource: atlas.texture.createView() },
      { binding: 3, resource: sampler },
    ],
  });

  return { pipeline, storageBuffer, bindGroup, instanceCount: 0, capacity: initialCapacity };
}

export function updateGlyphInstances(
  state: TextPipelineState,
  device: GPUDevice,
  cameraBuffer: GPUBuffer,
  atlas: TextAtlasData,
  data: Float32Array,
  count: number,
): TextPipelineState {
  if (count > state.capacity) {
    state.storageBuffer.destroy();
    const newCapacity = Math.max(count, state.capacity * 2);
    const storageBuffer = device.createBuffer({
      size: newCapacity * GLYPH_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    });

    const bindGroup = device.createBindGroup({
      layout: state.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cameraBuffer } },
        { binding: 1, resource: { buffer: storageBuffer } },
        { binding: 2, resource: atlas.texture.createView() },
        { binding: 3, resource: sampler },
      ],
    });

    state = { ...state, storageBuffer, bindGroup, capacity: newCapacity };
  }

  if (count > 0) {
    device.queue.writeBuffer(state.storageBuffer, 0, data as Float32Array<ArrayBuffer>, 0, count * GLYPH_FLOATS);
  }
  state.instanceCount = count;
  return state;
}

export function drawText(pass: GPURenderPassEncoder, state: TextPipelineState): void {
  if (state.instanceCount === 0) return;
  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, state.bindGroup);
  pass.draw(6, state.instanceCount);
}
