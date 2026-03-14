/**
 * Shared types and constants for WebGPU rendering pipelines.
 */

/** Floats per node instance in the storage buffer */
export const NODE_FLOATS = 16;
/** Bytes per node instance */
export const NODE_BYTES = NODE_FLOATS * 4;

/** Floats per glyph instance in the storage buffer */
export const GLYPH_FLOATS = 12;
/** Bytes per glyph instance */
export const GLYPH_BYTES = GLYPH_FLOATS * 4;

/** Graph-space font size */
export const GRAPH_FONT_SIZE = 16;

/** Node corner radius in graph units */
export const NODE_RADIUS = 5;

/** Background clear color (#1B1E2B) */
export const CLEAR_COLOR: GPUColor = { r: 0.106, g: 0.118, b: 0.169, a: 1.0 };

/** Default edge color (#5A6080) — brighter for contrast on dark bg */
export const EDGE_COLOR = { r: 0.353, g: 0.376, b: 0.502, a: 1.0 };

/** Dimmed edge color during search */
export const EDGE_COLOR_DIMMED = { r: 0.184, g: 0.200, b: 0.255, a: 0.3 };

/** Standard alpha blend state for all pipelines */
export const ALPHA_BLEND: GPUBlendState = {
  color: {
    srcFactor: 'src-alpha',
    dstFactor: 'one-minus-src-alpha',
    operation: 'add',
  },
  alpha: {
    srcFactor: 'one',
    dstFactor: 'one-minus-src-alpha',
    operation: 'add',
  },
};

/** Build orthographic projection matrix incorporating pan/zoom */
export function buildCameraMatrix(
  width: number, height: number,
  tx: number, ty: number, scale: number,
): Float32Array {
  // Maps graph coords → NDC:
  //   ndcX = (graphX * scale + tx) * 2/W - 1
  //   ndcY = 1 - (graphY * scale + ty) * 2/H
  const mat = new Float32Array(16);
  mat[0] = 2 * scale / width;
  mat[5] = -2 * scale / height;
  mat[10] = 1;
  mat[12] = 2 * tx / width - 1;
  mat[13] = 1 - 2 * ty / height;
  mat[15] = 1;
  return mat;
}
