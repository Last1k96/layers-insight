/**
 * Custom Sigma.js node program: Netron-style rounded rectangle.
 *
 * Renders nodes as rounded rectangles with a 1px dark border (#333),
 * 5px corner radius, matching Netron's visual style exactly.
 */
import { NodeProgram } from 'sigma/rendering';
import { floatColor } from 'sigma/utils';
import type { NodeDisplayData, RenderParams } from 'sigma/types';
import type { ProgramInfo } from 'sigma/rendering';

const ASPECT = 4.0;

// Same angles as NodeCircleProgram
const ANGLE_1 = 0;
const ANGLE_2 = (2 * Math.PI) / 3;
const ANGLE_3 = (4 * Math.PI) / 3;

// language=GLSL
const VERTEX_SHADER = /*glsl*/ `
attribute vec2 a_position;
attribute float a_size;
attribute vec4 a_color;
attribute vec4 a_id;
attribute float a_angle;

uniform mat3 u_matrix;
uniform float u_sizeRatio;
uniform float u_correctionRatio;

varying vec4 v_color;
varying vec2 v_diffVector;
varying float v_halfW;
varying float v_halfH;
varying float v_border;

const float bias = 255.0 / 254.0;
const float ASPECT = ${ASPECT.toFixed(1)};

void main() {
  float baseSize = a_size * u_correctionRatio / u_sizeRatio * 4.0;
  float halfH = baseSize;
  float halfW = halfH * ASPECT;

  float triRadius = baseSize * ASPECT * 2.4;

  vec2 diffVector = triRadius * vec2(cos(a_angle), sin(a_angle));
  vec2 position = a_position + diffVector;

  gl_Position = vec4(
    (u_matrix * vec3(position, 1)).xy,
    0,
    1
  );

  v_diffVector = diffVector;
  v_halfW = halfW;
  v_halfH = halfH;
  v_border = u_correctionRatio;

  #ifdef PICKING_MODE
  v_color = a_id;
  #else
  v_color = a_color;
  #endif

  v_color.a *= bias;
}
`;

// language=GLSL
const FRAGMENT_SHADER = /*glsl*/ `
precision highp float;

varying vec4 v_color;
varying vec2 v_diffVector;
varying float v_halfW;
varying float v_halfH;
varying float v_border;

uniform float u_correctionRatio;

const vec4 transparent = vec4(0.0, 0.0, 0.0, 0.0);
const vec4 borderColor = vec4(0.2, 0.2, 0.2, 1.0); // #333

float roundedRectSDF(vec2 p, vec2 halfSize, float radius) {
  vec2 d = abs(p) - halfSize + radius;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - radius;
}

void main(void) {
  float aa = u_correctionRatio * 1.5;  // anti-alias width
  float borderW = u_correctionRatio * 1.0;  // 1px border
  float radius = min(5.0 * u_correctionRatio, min(v_halfW, v_halfH));
  float dist = roundedRectSDF(v_diffVector, vec2(v_halfW, v_halfH), radius);

  #ifdef PICKING_MODE
  if (dist > aa)
    gl_FragColor = transparent;
  else
    gl_FragColor = v_color;
  #else
  // Outside: fully transparent
  if (dist > aa) {
    gl_FragColor = transparent;
  }
  // Outer anti-alias edge
  else if (dist > 0.0) {
    float outerAlpha = 1.0 - dist / aa;
    gl_FragColor = vec4(borderColor.rgb, outerAlpha);
  }
  // Border band
  else if (dist > -borderW) {
    gl_FragColor = borderColor;
  }
  // Inner anti-alias (border to fill transition)
  else if (dist > -borderW - aa * 0.5) {
    float t = (-borderW - dist) / (aa * 0.5);
    gl_FragColor = mix(borderColor, v_color, t);
  }
  // Fill
  else {
    gl_FragColor = v_color;
  }
  #endif
}
`;

const { UNSIGNED_BYTE, FLOAT } = WebGLRenderingContext;
const UNIFORMS = ['u_sizeRatio', 'u_correctionRatio', 'u_matrix'] as const;

export default class NodeRectProgram extends NodeProgram<(typeof UNIFORMS)[number]> {
  getDefinition() {
    return {
      VERTICES: 3,
      VERTEX_SHADER_SOURCE: VERTEX_SHADER,
      FRAGMENT_SHADER_SOURCE: FRAGMENT_SHADER,
      METHOD: WebGLRenderingContext.TRIANGLES as 4,
      UNIFORMS,
      ATTRIBUTES: [
        { name: 'a_position', size: 2, type: FLOAT },
        { name: 'a_size', size: 1, type: FLOAT },
        { name: 'a_color', size: 4, type: UNSIGNED_BYTE, normalized: true },
        { name: 'a_id', size: 4, type: UNSIGNED_BYTE, normalized: true },
      ],
      CONSTANT_ATTRIBUTES: [{ name: 'a_angle', size: 1, type: FLOAT }],
      CONSTANT_DATA: [[ANGLE_1], [ANGLE_2], [ANGLE_3]],
    };
  }

  // Identical to NodeCircleProgram
  processVisibleItem(nodeIndex: number, startIndex: number, data: NodeDisplayData) {
    const array = this.array;
    const color = floatColor(data.color);
    array[startIndex++] = data.x;
    array[startIndex++] = data.y;
    array[startIndex++] = data.size;
    array[startIndex++] = color;
    array[startIndex++] = nodeIndex;
  }

  setUniforms(params: RenderParams, { gl, uniformLocations }: ProgramInfo) {
    const { u_sizeRatio, u_correctionRatio, u_matrix } = uniformLocations;
    gl.uniform1f(u_correctionRatio, params.correctionRatio);
    gl.uniform1f(u_sizeRatio, params.sizeRatio);
    gl.uniformMatrix3fv(u_matrix, false, params.matrix);
  }
}

export { ASPECT as NODE_RECT_ASPECT };
