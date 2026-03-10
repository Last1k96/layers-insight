/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}', './index.html'],
  theme: {
    extend: {
      colors: {
        // Op category colors
        'op-conv': '#4A90D9',
        'op-norm': '#9B59B6',
        'op-act': '#E67E22',
        'op-pool': '#1ABC9C',
        'op-elem': '#2ECC71',
        'op-matmul': '#5C6BC0',
        'op-data': '#95A5A6',
        'op-quant': '#F39C12',
        'op-reduce': '#E91E63',
        'op-param': '#607D8B',
        'op-other': '#78909C',
        // Status colors
        'status-waiting': '#F59E0B',
        'status-executing': '#3B82F6',
        'status-success': '#10B981',
        'status-failed': '#EF4444',
      },
    },
  },
  plugins: [],
};
