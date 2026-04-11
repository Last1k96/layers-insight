import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';
import { resolve } from 'path';

export default defineConfig({
  plugins: [svelte()],
  server: {
    // Cross-origin isolation: unlocks 5 µs precision on performance.now()
    // (without these headers Chromium clamps to 100 µs as a Spectre mitigation,
    // which makes the bench harness unable to distinguish sub-millisecond phases).
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        bench: resolve(__dirname, 'bench.html'),
        perf: resolve(__dirname, 'perf.html'),
      },
      output: {
        manualChunks(id) {
          if (id.includes('/lib/accuracy/')) return 'accuracy';
          if (id.includes('/lib/perf/')) return 'perf';
        },
      },
    },
  },
});
