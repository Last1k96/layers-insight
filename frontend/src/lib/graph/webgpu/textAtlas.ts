/**
 * Canvas-based bitmap font atlas for WebGPU text rendering.
 * Renders ASCII printable characters (32-126) into a texture.
 */

export interface GlyphInfo {
  /** X position in atlas (pixels) */
  x: number;
  /** Y position in atlas (pixels) */
  y: number;
  /** Width in atlas (pixels) */
  w: number;
  /** Height in atlas (pixels) */
  h: number;
  /** Horizontal advance (pixels) */
  advance: number;
}

export interface TextAtlasData {
  texture: GPUTexture;
  glyphs: GlyphInfo[];  // indexed by (charCode - 32)
  atlasWidth: number;
  atlasHeight: number;
  fontSize: number;
  lineHeight: number;
}

const ATLAS_FONT_SIZE = 48;
const ATLAS_FONT = `${ATLAS_FONT_SIZE}px -apple-system, BlinkMacSystemFont, "Segoe UI", Ubuntu, sans-serif`;
const FIRST_CHAR = 32;
const LAST_CHAR = 126;
const CHAR_COUNT = LAST_CHAR - FIRST_CHAR + 1;
const COLS = 16;
const PAD = 4;

export function createTextAtlas(device: GPUDevice): TextAtlasData {
  // Measure characters
  const measureCanvas = document.createElement('canvas');
  const measureCtx = measureCanvas.getContext('2d')!;
  measureCtx.font = ATLAS_FONT;

  const advances: number[] = [];
  let maxWidth = 0;
  for (let i = 0; i < CHAR_COUNT; i++) {
    const ch = String.fromCharCode(FIRST_CHAR + i);
    const m = measureCtx.measureText(ch);
    const w = Math.ceil(m.width);
    advances.push(w);
    maxWidth = Math.max(maxWidth, w);
  }

  const cellW = maxWidth + PAD * 2;
  const cellH = ATLAS_FONT_SIZE + PAD * 2;
  const rows = Math.ceil(CHAR_COUNT / COLS);
  const atlasWidth = nextPow2(cellW * COLS);
  const atlasHeight = nextPow2(cellH * rows);

  // Render atlas
  const canvas = document.createElement('canvas');
  canvas.width = atlasWidth;
  canvas.height = atlasHeight;
  const ctx = canvas.getContext('2d')!;

  ctx.clearRect(0, 0, atlasWidth, atlasHeight);
  ctx.font = ATLAS_FONT;
  ctx.textBaseline = 'top';
  ctx.fillStyle = 'white';

  const glyphs: GlyphInfo[] = [];
  for (let i = 0; i < CHAR_COUNT; i++) {
    const col = i % COLS;
    const row = Math.floor(i / COLS);
    const x = col * cellW + PAD;
    const y = row * cellH + PAD;
    const ch = String.fromCharCode(FIRST_CHAR + i);

    ctx.fillText(ch, x, y);

    glyphs.push({
      x,
      y,
      w: advances[i] || cellW - PAD * 2,
      h: ATLAS_FONT_SIZE,
      advance: advances[i],
    });
  }

  // Upload to GPU texture
  const imageData = ctx.getImageData(0, 0, atlasWidth, atlasHeight);
  const texture = device.createTexture({
    size: { width: atlasWidth, height: atlasHeight },
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });

  device.queue.writeTexture(
    { texture },
    imageData.data as unknown as Uint8Array<ArrayBuffer>,
    { bytesPerRow: atlasWidth * 4, rowsPerImage: atlasHeight },
    { width: atlasWidth, height: atlasHeight },
  );

  return {
    texture,
    glyphs,
    atlasWidth,
    atlasHeight,
    fontSize: ATLAS_FONT_SIZE,
    lineHeight: cellH,
  };
}

/** Measure text width in graph-space units */
export function measureText(atlas: TextAtlasData, text: string, graphFontSize: number): number {
  const scale = graphFontSize / atlas.fontSize;
  let width = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i) - FIRST_CHAR;
    if (code >= 0 && code < CHAR_COUNT) {
      width += atlas.glyphs[code].advance * scale;
    }
  }
  return width;
}

function nextPow2(v: number): number {
  v--;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  return v + 1;
}
