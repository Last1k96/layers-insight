/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}', './index.html'],
  theme: {
    extend: {
      colors: {
        // Surface palette (cool slate) — kept in sync with CSS tokens in app.css
        surface: {
          base: 'var(--bg-primary)',
          panel: 'var(--bg-panel)',
          elevated: 'var(--bg-menu)',
          hover: 'var(--bg-elevated)',
          input: 'var(--bg-input)',
        },
        edge: {
          DEFAULT: 'var(--border-color)',
          soft: 'var(--border-soft)',
          strong: 'var(--border-strong)',
        },
        content: {
          primary: 'var(--text-primary)',
          secondary: 'var(--text-secondary)',
        },
        muted: {
          strong: 'var(--text-muted-strong)',
          DEFAULT: 'var(--text-muted)',
          soft: 'var(--text-muted-soft)',
          faint: 'var(--text-muted-faint)',
        },
        accent: {
          DEFAULT: 'var(--accent)',
          hover: 'var(--accent-hover)',
          soft: 'var(--accent-soft)',
          bg: 'var(--accent-bg)',
          'bg-strong': 'var(--accent-bg-strong)',
          border: 'var(--accent-border)',
        },
        // Status aliases
        status: {
          info: 'var(--status-info)',
          warn: 'var(--status-warn)',
          ok: 'var(--status-ok)',
          err: 'var(--status-err)',
          hint: 'var(--status-hint)',
        },
        'status-waiting': 'var(--status-warn)',
        'status-executing': 'var(--status-info)',
        'status-success': 'var(--status-ok)',
        'status-failed': 'var(--status-err)',
      },
      borderRadius: {
        xs: 'var(--radius-xs)',
        sm: 'var(--radius-sm)',
        md: 'var(--radius-md)',
        lg: 'var(--radius-lg)',
        xl: 'var(--radius-xl)',
      },
      boxShadow: {
        panel:    'var(--shadow-panel)',
        modal:    'var(--shadow-modal)',
        elevated: 'var(--shadow-elevated)',
        accent:   'var(--shadow-accent)',
      },
      fontFamily: {
        sans:    ['DM Sans', 'system-ui', 'sans-serif'],
        display: ['Outfit', 'system-ui', 'sans-serif'],
        mono:    ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms')({ strategy: 'class' }),
  ],
};
