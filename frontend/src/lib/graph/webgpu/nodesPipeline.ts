/**
 * WebGPU pipeline for rendering nodes as instanced SDF rounded rectangles.
 */
import { NODE_FLOATS, NODE_BYTES, NODE_RADIUS, ALPHA_BLEND } from './types';

const SHADER = /* wgsl */ `
struct NodeInstance {
  pos: vec2<f32>,
  size: vec2<f32>,
  fillColor: vec4<f32>,
  strokeColor: vec4<f32>,
  strokeWidth: f32,
  cornerRadius: f32,
  opacity: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;
@group(0) @binding(1) var<storage, read> nodes: array<NodeInstance>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) localPos: vec2<f32>,
  @location(1) fillColor: vec4<f32>,
  @location(2) strokeColor: vec4<f32>,
  @location(3) size: vec2<f32>,
  @location(4) strokeWidth: f32,
  @location(5) cornerRadius: f32,
  @location(6) opacity: f32,
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
  let node = nodes[instanceIndex];
  let uv = quadPositions[vertexIndex];

  let margin = max(node.strokeWidth, 1.0) + 1.0;
  let expandedSize = node.size + vec2(2.0 * margin);
  let expandedPos = node.pos - vec2(margin);
  let worldPos = uv * expandedSize + expandedPos;

  var out: VertexOutput;
  out.position = camera * vec4(worldPos, 0.0, 1.0);
  out.localPos = uv * expandedSize - vec2(margin);
  out.fillColor = node.fillColor;
  out.strokeColor = node.strokeColor;
  out.size = node.size;
  out.strokeWidth = node.strokeWidth;
  out.cornerRadius = node.cornerRadius;
  out.opacity = node.opacity;
  return out;
}

fn sdRoundedRect(p: vec2<f32>, size: vec2<f32>, radius: f32) -> f32 {
  let halfSize = size * 0.5 - vec2(radius);
  let d = abs(p - size * 0.5) - halfSize;
  return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - radius;
}

@fragment
fn fragmentMain(in: VertexOutput) -> @location(0) vec4<f32> {
  let dist = sdRoundedRect(in.localPos, in.size, in.cornerRadius);
  let fw = fwidth(dist);
  let halfStroke = in.strokeWidth * 0.5;

  // Blend between fill and stroke at the inner edge
  let innerT = smoothstep(-halfStroke - fw, -halfStroke + fw, dist);
  let baseColor = mix(in.fillColor, in.strokeColor, innerT);

  // Fade to transparent at the outer edge
  let outerAlpha = 1.0 - smoothstep(halfStroke - fw, halfStroke + fw, dist);

  let alpha = baseColor.a * outerAlpha * in.opacity;
  if (alpha < 0.004) { discard; }
  return vec4(baseColor.rgb, alpha);
}
`;

export interface NodesPipelineState {
  pipeline: GPURenderPipeline;
  storageBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  instanceCount: number;
  capacity: number;
}

export function createNodesPipeline(
  device: GPUDevice,
  format: GPUTextureFormat,
  cameraBuffer: GPUBuffer,
): NodesPipelineState {
  const module = device.createShaderModule({ code: SHADER });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
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
  });

  // Initial empty storage buffer (will be resized on first update)
  const initialCapacity = 256;
  const storageBuffer = device.createBuffer({
    size: initialCapacity * NODE_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: cameraBuffer } },
      { binding: 1, resource: { buffer: storageBuffer } },
    ],
  });

  return { pipeline, storageBuffer, bindGroup, instanceCount: 0, capacity: initialCapacity };
}

export function updateNodeInstances(
  state: NodesPipelineState,
  device: GPUDevice,
  cameraBuffer: GPUBuffer,
  data: Float32Array,
  count: number,
): NodesPipelineState {
  if (count > state.capacity) {
    // Need a larger buffer
    state.storageBuffer.destroy();
    const newCapacity = Math.max(count, state.capacity * 2);
    const storageBuffer = device.createBuffer({
      size: newCapacity * NODE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const bindGroup = device.createBindGroup({
      layout: state.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: cameraBuffer } },
        { binding: 1, resource: { buffer: storageBuffer } },
      ],
    });

    state = { ...state, storageBuffer, bindGroup, capacity: newCapacity };
  }

  device.queue.writeBuffer(state.storageBuffer, 0, data as Float32Array<ArrayBuffer>, 0, count * NODE_FLOATS);
  state.instanceCount = count;
  return state;
}

export function drawNodes(pass: GPURenderPassEncoder, state: NodesPipelineState): void {
  if (state.instanceCount === 0) return;
  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, state.bindGroup);
  pass.draw(6, state.instanceCount);
}
