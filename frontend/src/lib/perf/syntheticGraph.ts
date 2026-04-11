/**
 * Deterministic synthetic graph generator for the WebGPU bench harness.
 *
 * Produces GraphData matching the real schema (nodes with x/y/width/height/
 * color/category, edges with waypoints), laid out as a DAG of stacked layers
 * with light per-row jitter. No ELK, no async — fast enough to call from a
 * benchmark scenario at sizes up to ~50k nodes.
 *
 * The same seed always produces the same graph, so cross-run comparisons stay
 * meaningful.
 */
import type { GraphData, GraphNode, GraphEdge } from '../stores/types';
import { OP_CATEGORIES, getOpColor } from '../graph/opColors';

export interface SyntheticGraphOptions {
  /** Approximate target node count (will be rounded to fit the layer grid). */
  nodeCount: number;
  /** Average outgoing edges per non-source node (will be clamped to layer width). */
  avgFanOut?: number;
  /** Number of DAG layers. If omitted, derived from nodeCount with sqrt heuristic. */
  layers?: number;
  /** Seed for the deterministic RNG. Same seed → same graph. */
  seed?: number;
}

const NODE_W = 120;
const NODE_H = 32;
const COL_SPACING = 60; // horizontal gap between nodes within a layer
const ROW_SPACING = 120; // vertical gap between layers

const OP_TYPES = Object.keys(OP_CATEGORIES);

/** Tiny seeded LCG — sufficient for graph generation, deterministic across runs. */
class Rng {
  private state: number;
  constructor(seed: number) {
    this.state = (seed | 0) || 1;
  }
  next(): number {
    // mulberry32-ish
    this.state = (this.state + 0x6D2B79F5) | 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  intRange(lo: number, hi: number): number {
    return lo + Math.floor(this.next() * (hi - lo + 1));
  }
  pick<T>(arr: T[]): T {
    return arr[Math.floor(this.next() * arr.length)];
  }
}

export function generateSyntheticGraph(opts: SyntheticGraphOptions): GraphData {
  const target = Math.max(2, opts.nodeCount | 0);
  const fanOut = Math.max(1, opts.avgFanOut ?? 2);
  const seed = opts.seed ?? 1;
  const rng = new Rng(seed);

  const layerCount = opts.layers ?? Math.max(2, Math.round(Math.sqrt(target * 2)));
  const perLayer = Math.max(1, Math.round(target / layerCount));

  const nodes: GraphNode[] = [];

  for (let layer = 0; layer < layerCount; layer++) {
    const layerY = layer * (NODE_H + ROW_SPACING);
    const layerWidth = perLayer * (NODE_W + COL_SPACING) - COL_SPACING;
    const layerX0 = -layerWidth / 2;

    for (let col = 0; col < perLayer; col++) {
      const id = `n${layer}_${col}`;
      const opType = layer === 0
        ? 'Parameter'
        : layer === layerCount - 1 && col === 0
          ? 'Result'
          : rng.pick(OP_TYPES);

      const color = getOpColor(opType);
      const category = OP_CATEGORIES[opType]?.category ?? 'Other';

      // Light jitter to mimic ELK output (within ±20px so the layout stays readable)
      const jitterX = (rng.next() - 0.5) * 20;
      const jitterY = (rng.next() - 0.5) * 12;

      nodes.push({
        id,
        name: `${opType}_${layer}_${col}`,
        type: opType,
        category,
        color,
        attributes: {},
        x: layerX0 + col * (NODE_W + COL_SPACING) + jitterX,
        y: layerY + jitterY,
        width: NODE_W,
        height: NODE_H,
      });
    }
  }

  const edges: GraphEdge[] = [];
  // Connect each non-source-layer node to `fanOut` random nodes from the previous layer.
  for (let layer = 1; layer < layerCount; layer++) {
    for (let col = 0; col < perLayer; col++) {
      const targetId = `n${layer}_${col}`;
      const seenSources = new Set<number>();
      const desiredFanOut = Math.min(fanOut, perLayer);
      let attempts = 0;
      while (seenSources.size < desiredFanOut && attempts < desiredFanOut * 4) {
        attempts++;
        const srcCol = rng.intRange(0, perLayer - 1);
        if (seenSources.has(srcCol)) continue;
        seenSources.add(srcCol);
        const sourceId = `n${layer - 1}_${srcCol}`;

        const srcNode = nodes[(layer - 1) * perLayer + srcCol];
        const tgtNode = nodes[layer * perLayer + col];

        edges.push({
          source: sourceId,
          target: targetId,
          source_port: 0,
          target_port: 0,
          waypoints: [
            { x: srcNode.x + srcNode.width / 2, y: srcNode.y + srcNode.height },
            { x: tgtNode.x + tgtNode.width / 2, y: tgtNode.y },
          ],
        });
      }
    }
  }

  return { nodes, edges };
}

/** Preset graph sizes the bench harness exposes through its UI. */
export const SYNTHETIC_PRESETS: Array<{ label: string; opts: SyntheticGraphOptions }> = [
  { label: 'synthetic-1k',  opts: { nodeCount: 1_000,  avgFanOut: 2, seed: 1 } },
  { label: 'synthetic-5k',  opts: { nodeCount: 5_000,  avgFanOut: 2, seed: 1 } },
  { label: 'synthetic-10k', opts: { nodeCount: 10_000, avgFanOut: 2, seed: 1 } },
  { label: 'synthetic-25k', opts: { nodeCount: 25_000, avgFanOut: 2, seed: 1 } },
  { label: 'synthetic-50k', opts: { nodeCount: 50_000, avgFanOut: 2, seed: 1 } },
];
