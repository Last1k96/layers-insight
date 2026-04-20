<script lang="ts">
  import { configStore } from '../stores/config.svelte';
  import { refreshRenderer } from './renderer';
  import { DEFAULT_RANGES, type AccuracyMetricKey } from '../utils/accuracyColors';

  const METRIC_LABELS: Record<AccuracyMetricKey, string> = {
    mse: 'MSE',
    max_abs_diff: 'Max Abs',
    cosine_similarity: 'Cosine',
  };

  function setMetric(m: AccuracyMetricKey) {
    configStore.setAccuracyMetric(m);
    refreshRenderer();
  }

  function handleRangeInput(field: 'min' | 'max', e: Event) {
    const input = e.target as HTMLInputElement;
    const val = parseFloat(input.value);
    if (isNaN(val)) return;
    const metric = configStore.accuracyMetric;
    const current = configStore.accuracyRanges[metric];
    configStore.setAccuracyRange(metric, { ...current, [field]: val });
    refreshRenderer();
  }

  function resetRange() {
    const metric = configStore.accuracyMetric;
    configStore.setAccuracyRange(metric, { ...DEFAULT_RANGES[metric] });
    refreshRenderer();
  }

  function handleRangeWheel(field: 'min' | 'max', e: WheelEvent) {
    e.preventDefault();
    const step = e.shiftKey ? 0.1 : 0.01;
    const actual = e.deltaY < 0 ? step : -step;
    const metric = configStore.accuracyMetric;
    const current = configStore.accuracyRanges[metric];
    const val = Math.round((current[field] + actual) * 1000) / 1000;
    configStore.setAccuracyRange(metric, { ...current, [field]: val });
    refreshRenderer();
  }
</script>

<section class="accuracy-settings">
  <header class="ass-header">
    <span class="ass-eyebrow">Accuracy</span>
    <button class="reset-link" onclick={resetRange} title="Reset color range to defaults">
      Reset
    </button>
  </header>

  <div class="ass-row">
    <span class="ass-label">Metric</span>
    <div class="metric-pills">
      {#each Object.entries(METRIC_LABELS) as [key, label] (key)}
        <button
          type="button"
          class="pill"
          class:active={configStore.accuracyMetric === key}
          onclick={() => setMetric(key as AccuracyMetricKey)}
        >
          {label}
        </button>
      {/each}
    </div>
  </div>

  <div class="ass-row">
    <span class="ass-label">Color range</span>
    <div class="range-grid">
      <label for="acc-range-min" class="range-tag">Min</label>
      <div class="gradient-bar"></div>
      <label for="acc-range-max" class="range-tag">Max</label>
      <input
        id="acc-range-min"
        type="number"
        step="0.01"
        value={configStore.activeRange.min}
        oninput={(e) => handleRangeInput('min', e)}
        onwheel={(e) => handleRangeWheel('min', e)}
        class="range-input"
      />
      <span class="ends-hint">
        {configStore.accuracyMetric === 'cosine_similarity' ? 'Bad → Good' : 'Good → Bad'}
      </span>
      <input
        id="acc-range-max"
        type="number"
        step="0.01"
        value={configStore.activeRange.max}
        oninput={(e) => handleRangeInput('max', e)}
        onwheel={(e) => handleRangeWheel('max', e)}
        class="range-input"
      />
    </div>
  </div>
</section>

<style>
  .accuracy-settings {
    flex-shrink: 0;
    background: linear-gradient(180deg, rgba(76, 141, 255, 0.04), transparent 60%), var(--bg-panel);
    border-bottom: 1px solid var(--border-color);
    padding: 12px 14px 14px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    animation: settings-slide 220ms cubic-bezier(0.32, 0.72, 0, 1);
  }
  @keyframes settings-slide {
    from { opacity: 0; transform: translateY(-6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .ass-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .ass-eyebrow {
    font-family: var(--font-display);
    font-size: 9.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    color: var(--accent);
  }
  .reset-link {
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
    font-size: 10px;
    color: var(--text-muted-soft);
    transition: color 140ms ease;
  }
  .reset-link:hover { color: var(--text-primary); }

  .ass-row { display: flex; flex-direction: column; gap: 6px; }
  .ass-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text-muted-soft);
  }

  .metric-pills {
    display: flex;
    gap: 4px;
    background: var(--bg-input);
    padding: 3px;
    border-radius: 8px;
    border: 1px solid var(--border-soft);
  }
  .pill {
    flex: 1;
    border: none;
    background: transparent;
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 500;
    padding: 5px 8px;
    border-radius: 5px;
    cursor: pointer;
    transition: background 140ms ease, color 140ms ease;
  }
  .pill:hover { color: var(--text-primary); }
  .pill.active {
    background: var(--accent);
    color: white;
    box-shadow: 0 1px 6px rgba(76, 141, 255, 0.45);
  }

  .range-grid {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 6px 8px;
    align-items: center;
  }
  .range-tag {
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--text-muted-soft);
  }
  .gradient-bar {
    grid-row: 1 / span 1;
    width: 36px;
    height: 8px;
    border-radius: 99px;
    background: linear-gradient(to right, #ef4444, #e5b010, #10b981);
    box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.25);
  }
  .range-input {
    background: var(--bg-input);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 4px 7px;
    color: var(--text-primary);
    font-family: var(--font-mono);
    font-size: 11px;
    outline: none;
    transition: border-color 140ms ease;
    min-width: 0;
  }
  .range-input:focus { border-color: var(--accent); }
  .ends-hint {
    font-family: var(--font-mono);
    font-size: 9px;
    text-align: center;
    color: var(--text-muted-faint);
    letter-spacing: 0.05em;
    white-space: nowrap;
  }
</style>
