/**
 * Entry point for the WebGPU bench harness page (frontend/perf.html).
 *
 * Wires the page UI to the runner: dataset dropdown, scenario checkboxes,
 * Run button, log output, and a JSON download link.
 */
import { runBench, DATASETS, summarize, type BenchResult } from './lib/perf/runner';
import { DEFAULT_SCENARIO_ORDER } from './lib/perf/scenarios';

const datasetEl = document.getElementById('dataset') as HTMLSelectElement;
const scenarioListEl = document.getElementById('scenario-list') as HTMLDivElement;
const runBtn = document.getElementById('run') as HTMLButtonElement;
const clearBtn = document.getElementById('clear') as HTMLButtonElement;
const logEl = document.getElementById('log') as HTMLDivElement;
const summaryEl = document.getElementById('summary') as HTMLPreElement;
const downloadEl = document.getElementById('download') as HTMLAnchorElement;
const disableMinimapEl = document.getElementById('disable-minimap') as HTMLInputElement;

// Populate dataset dropdown
for (const d of DATASETS) {
  const opt = document.createElement('option');
  opt.value = d.label;
  opt.textContent = d.label;
  datasetEl.appendChild(opt);
}

// Populate scenario checkboxes
const scenarioCheckboxes: Record<string, HTMLInputElement> = {};
for (const name of DEFAULT_SCENARIO_ORDER) {
  const label = document.createElement('label');
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = true;
  cb.value = name;
  label.appendChild(cb);
  label.appendChild(document.createTextNode(name));
  scenarioListEl.appendChild(label);
  scenarioCheckboxes[name] = cb;
}

function appendLog(msg: string): void {
  logEl.textContent += msg + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

function clearLog(): void {
  logEl.textContent = '';
  summaryEl.style.display = 'none';
  downloadEl.style.display = 'none';
}

clearBtn.onclick = () => clearLog();

runBtn.onclick = async () => {
  if (!('gpu' in navigator)) {
    appendLog('ERROR: navigator.gpu is not available in this browser.');
    return;
  }

  clearLog();
  runBtn.disabled = true;
  try {
    const datasetLabel = datasetEl.value;
    const dataset = DATASETS.find(d => d.label === datasetLabel);
    if (!dataset) {
      appendLog(`ERROR: unknown dataset ${datasetLabel}`);
      return;
    }

    const selectedScenarios = Object.entries(scenarioCheckboxes)
      .filter(([, cb]) => cb.checked)
      .map(([name]) => name);

    if (selectedScenarios.length === 0) {
      appendLog('ERROR: no scenarios selected');
      return;
    }

    appendLog(`Running ${selectedScenarios.length} scenarios on dataset "${datasetLabel}"...`);
    appendLog('');

    const result: BenchResult = await runBench({
      dataset,
      scenarios: selectedScenarios,
      disableMinimap: disableMinimapEl.checked,
      log: appendLog,
    });

    appendLog('');
    appendLog('── Summary ──');
    const summary = summarize(result);
    summaryEl.textContent = summary;
    summaryEl.style.display = 'block';

    // Build downloadable JSON
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    downloadEl.href = url;
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    downloadEl.download = `bench-${datasetLabel}-${stamp}.json`;
    downloadEl.style.display = 'inline-block';
    downloadEl.textContent = `Download ${downloadEl.download}`;
  } catch (e) {
    appendLog('ERROR: ' + (e as Error).message);
    console.error(e);
  } finally {
    runBtn.disabled = false;
  }
};
