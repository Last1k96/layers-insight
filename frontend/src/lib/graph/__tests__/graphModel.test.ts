import { describe, it, expect } from 'vitest';
import { GraphModel } from '../graphModel';
import type { NodeAttributes } from '../graphModel';

function makeAttrs(overrides: Partial<NodeAttributes> = {}): NodeAttributes {
  return {
    x: 0, y: 0, label: 'test', color: '#333', opType: 'Relu',
    nodeName: 'test', category: 'Activation', attributes: {},
    ...overrides,
  };
}

describe('GraphModel', () => {
  it('addNode and hasNode', () => {
    const g = new GraphModel();
    g.addNode('n1', makeAttrs());
    expect(g.hasNode('n1')).toBe(true);
    expect(g.hasNode('n2')).toBe(false);
  });

  it('addEdge and neighbors', () => {
    const g = new GraphModel();
    g.addNode('a', makeAttrs());
    g.addNode('b', makeAttrs());
    g.addEdge('a', 'b');

    expect(g.outNeighbors('a')).toEqual(['b']);
    expect(g.inNeighbors('b')).toEqual(['a']);
    expect(g.inNeighbors('a')).toEqual([]);
    expect(g.outNeighbors('b')).toEqual([]);
  });

  it('getNodeAttributes returns attrs', () => {
    const g = new GraphModel();
    const attrs = makeAttrs({ label: 'hello' });
    g.addNode('n1', attrs);
    expect(g.getNodeAttributes('n1').label).toBe('hello');
  });

  it('getNodeAttributes throws for missing node', () => {
    const g = new GraphModel();
    expect(() => g.getNodeAttributes('missing')).toThrow('Node not found');
  });

  it('nodes() returns all node IDs', () => {
    const g = new GraphModel();
    g.addNode('a', makeAttrs());
    g.addNode('b', makeAttrs());
    expect(g.nodes()).toEqual(['a', 'b']);
  });

  it('size() returns node count', () => {
    const g = new GraphModel();
    expect(g.size()).toBe(0);
    g.addNode('a', makeAttrs());
    expect(g.size()).toBe(1);
    g.addNode('b', makeAttrs());
    expect(g.size()).toBe(2);
  });
});
