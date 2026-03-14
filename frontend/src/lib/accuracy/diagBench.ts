// Benchmark harness for renderDiagnostics
import { renderDiagnostics } from './diagRenderer';

const SHAPE = [1, 160, 56, 56];
const TOTAL = SHAPE.reduce((a, b) => a * b, 1);

function syntheticData(): { main: Float32Array; ref: Float32Array } {
  const ref = new Float32Array(TOTAL);
  const main = new Float32Array(TOTAL);
  for (let i = 0; i < TOTAL; i++) {
    ref[i] = Math.sin(i * 0.01) * 2.0;
    main[i] = ref[i] + (Math.random() - 0.5) * 0.3;
  }
  return { main, ref };
}

export async function runBench(log: (msg: string) => void): Promise<void> {
  log('Generating synthetic data [1, 160, 56, 56]...');
  const { main, ref } = syntheticData();

  const offscreen = new OffscreenCanvas(1, 1);
  const ctx = offscreen.getContext('2d')!;
  const canvasProxy = {
    get width() { return offscreen.width; },
    set width(v: number) { offscreen.width = v; },
    get height() { return offscreen.height; },
    set height(v: number) { offscreen.height = v; },
    style: { width: '', height: '' },
  };

  const RUNS = 6;
  const times: number[] = [];

  for (let i = 0; i < RUNS; i++) {
    const t0 = performance.now();
    renderDiagnostics(ctx as unknown as CanvasRenderingContext2D, canvasProxy, main, ref, SHAPE, 1200);
    const dt = performance.now() - t0;
    if (i === 0) {
      log(`Warmup: ${dt.toFixed(1)} ms`);
    } else {
      times.push(dt);
      log(`Run ${i}: ${dt.toFixed(1)} ms`);
    }
  }

  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  log('');
  log(`Median: ${median.toFixed(1)} ms`);
  log(`Min:    ${times[0].toFixed(1)} ms`);
  log(`Max:    ${times[times.length - 1].toFixed(1)} ms`);
  log(median < 1000 ? '✓ PASS (<1s target)' : '✗ FAIL (>1s target)');
}
