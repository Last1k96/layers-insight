/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}', './index.html'],
  theme: {
    extend: {
      colors: {
        // Surface palette (cool slate)
        surface: {
          base: '#1B1E2B',
          panel: '#232636',
          elevated: '#2A2E41',
          input: '#1E2130',
        },
        edge: {
          DEFAULT: '#2F3347',
        },
        content: {
          primary: '#E1E4ED',
          secondary: '#9BA1B5',
        },
        accent: {
          DEFAULT: '#4C8DFF',
          hover: '#6BA1FF',
        },
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
        'status-waiting': '#E5A820',
        'status-executing': '#4C8DFF',
        'status-success': '#34C77B',
        'status-failed': '#E54D4D',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms')({ strategy: 'class' }),
  ],
};
