#!/usr/bin/env node
/**
 * ELK layout engine wrapper — Netron-style graph layout.
 *
 * Matches Netron's dagre parameters:
 *   - Top-to-bottom (DOWN) direction
 *   - nodesep: 20 (node-node spacing in same rank)
 *   - ranksep: 20 (inter-layer spacing)
 *   - Network simplex node placement
 *
 * Input:  { nodes: [{id, width, height}], edges: [{source, target}] }
 * Output: { nodes: [{id, x, y}] }
 */
const ELK = require('elkjs');

const elk = new ELK();

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', async () => {
    try {
        const data = JSON.parse(input);
        const graph = {
            id: 'root',
            layoutOptions: {
                'elk.algorithm': 'layered',
                'elk.direction': 'DOWN',
                // Match Netron dagre spacing: nodesep=20, ranksep=20
                'elk.layered.spacing.nodeNodeBetweenLayers': '20',
                'elk.spacing.nodeNode': '20',
                'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
                'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
                // Spline edge routing for Netron-style curved edges
                'elk.edgeRouting': 'SPLINES',
            },
            children: data.nodes.map(n => ({
                id: n.id,
                width: n.width || 180,
                height: n.height || 50,
            })),
            edges: data.edges.map((e, i) => ({
                id: `e${i}`,
                sources: [e.source],
                targets: [e.target],
            })),
        };

        const result = await elk.layout(graph);
        const positions = {};
        for (const child of result.children || []) {
            // Negate Y so Sigma.js (Y-up) renders the graph top-to-bottom
            positions[child.id] = { x: child.x, y: -child.y };
        }
        process.stdout.write(JSON.stringify(positions));
    } catch (err) {
        process.stderr.write(`ELK layout error: ${err.message}\n`);
        process.exit(1);
    }
});
