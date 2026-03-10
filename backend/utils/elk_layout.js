#!/usr/bin/env node
/**
 * ELK layout engine wrapper.
 * Reads graph JSON from stdin, outputs positioned graph JSON to stdout.
 * Input: { nodes: [{id, width, height}], edges: [{source, target}] }
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
                'elk.layered.spacing.nodeNodeBetweenLayers': '80',
                'elk.spacing.nodeNode': '40',
                'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
                'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
                'elk.edgeRouting': 'ORTHOGONAL',
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
            positions[child.id] = { x: child.x, y: child.y };
        }
        process.stdout.write(JSON.stringify(positions));
    } catch (err) {
        process.stderr.write(`ELK layout error: ${err.message}\n`);
        process.exit(1);
    }
});
