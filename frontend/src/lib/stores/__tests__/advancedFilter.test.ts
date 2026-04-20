import { describe, it, expect, beforeEach } from 'vitest';
import { advancedFilterStore } from '../advancedFilter.svelte';
import type { GraphNode, InferenceTask } from '../types';

function node(id: string, name: string, type: string): GraphNode {
  return { id, name, type } as GraphNode;
}

function task(node_id: string, mse: number | undefined): InferenceTask {
  return {
    task_id: 't-' + node_id,
    node_id,
    node_name: 'name-' + node_id,
    node_type: 'T',
    status: mse === undefined ? 'waiting' : 'success',
    metrics: mse === undefined ? undefined : { mse, cosine_similarity: 1, max_abs_diff: 0 },
  } as InferenceTask;
}

describe('applyFilterToNodes - MSE', () => {
  beforeEach(() => {
    advancedFilterStore.rules = [];
    advancedFilterStore.connectors = [];
  });

  it('excludes non-inferred nodes when filtering by MSE > 0', () => {
    advancedFilterStore.rules = [
      { id: 'r1', field: 'mse', operator: '>', value: '0' } as any,
    ];
    const nodes = [
      node('a', 'A', 'Conv'),
      node('b', 'B', 'Conv'),
      node('c', 'C', 'Conv'),
    ];
    const taskByNodeId = new Map<string, InferenceTask>();
    taskByNodeId.set('a', task('a', 0.5));
    // b and c NOT inferred

    const result = advancedFilterStore.applyFilterToNodes(nodes, taskByNodeId);
    expect(result.map(n => n.id)).toEqual(['a']);
  });

  it('excludes nodes with waiting task (no metrics) when filtering by MSE', () => {
    advancedFilterStore.rules = [
      { id: 'r1', field: 'mse', operator: '>', value: '0' } as any,
    ];
    const nodes = [node('a', 'A', 'Conv')];
    const taskByNodeId = new Map<string, InferenceTask>();
    taskByNodeId.set('a', task('a', undefined));
    const result = advancedFilterStore.applyFilterToNodes(nodes, taskByNodeId);
    expect(result).toEqual([]);
  });

  it('excludes non-inferred nodes even with OR of node_type and mse', () => {
    advancedFilterStore.rules = [
      { id: 'r1', field: 'node_type', operator: 'equals', value: 'Conv' } as any,
      { id: 'r2', field: 'mse', operator: '>', value: '0' } as any,
    ];
    advancedFilterStore.connectors = ['OR'];
    const nodes = [
      node('a', 'A', 'Conv'),
      node('b', 'B', 'Conv'),
    ];
    const taskByNodeId = new Map<string, InferenceTask>();
    taskByNodeId.set('a', task('a', 0.5));

    const result = advancedFilterStore.applyFilterToNodes(nodes, taskByNodeId);
    expect(result.map(n => n.id)).toEqual(['a']);
  });

  it('does not require inference when only node-bound fields are filtered', () => {
    advancedFilterStore.rules = [
      { id: 'r1', field: 'node_type', operator: 'equals', value: 'Conv' } as any,
    ];
    advancedFilterStore.connectors = [];
    const nodes = [
      node('a', 'A', 'Conv'),
      node('b', 'B', 'Conv'),
      node('c', 'C', 'Relu'),
    ];
    const result = advancedFilterStore.applyFilterToNodes(nodes, new Map());
    expect(result.map(n => n.id)).toEqual(['a', 'b']);
  });

  it('treats empty numeric value as 0 (MSE > with no input filters non-zero MSE)', () => {
    advancedFilterStore.rules = [
      { id: 'r1', field: 'mse', operator: '>', value: '' } as any,
    ];
    advancedFilterStore.connectors = [];
    const nodes = [
      node('a', 'A', 'Conv'),
      node('b', 'B', 'Conv'),
      node('c', 'C', 'Conv'),
    ];
    const taskByNodeId = new Map<string, InferenceTask>();
    taskByNodeId.set('a', task('a', 0.5));
    taskByNodeId.set('b', task('b', 0));
    // c not inferred
    expect(advancedFilterStore.hasActiveRules).toBe(true);
    const result = advancedFilterStore.applyFilterToNodes(nodes, taskByNodeId);
    expect(result.map(n => n.id)).toEqual(['a']);
  });

  it('empty string rule stays inactive (no filter applied)', () => {
    advancedFilterStore.rules = [
      { id: 'r1', field: 'node_name', operator: 'contains', value: '' } as any,
    ];
    advancedFilterStore.connectors = [];
    expect(advancedFilterStore.hasActiveRules).toBe(false);
  });

  it('cosine similarity empty value defaults to 1 (shows any deviation)', () => {
    // Build task with explicit cosine similarity < 1 (imperfect match) vs 1 (perfect).
    function taskWithCosine(id: string, cosine: number): InferenceTask {
      return {
        task_id: 't-' + id, node_id: id, node_name: id, node_type: 'T',
        status: 'success',
        metrics: { cosine_similarity: cosine, mse: 0.1, max_abs_diff: 0.2 },
      } as InferenceTask;
    }
    advancedFilterStore.rules = [
      { id: 'r1', field: 'cosine_similarity', operator: '<', value: '' } as any,
    ];
    advancedFilterStore.connectors = [];
    const nodes = [node('a', 'A', 'Conv'), node('b', 'B', 'Conv')];
    const taskByNodeId = new Map<string, InferenceTask>();
    taskByNodeId.set('a', taskWithCosine('a', 0.95));  // imperfect — should match < 1
    taskByNodeId.set('b', taskWithCosine('b', 1.0));   // perfect — should NOT match < 1
    const result = advancedFilterStore.applyFilterToNodes(nodes, taskByNodeId);
    expect(result.map(n => n.id)).toEqual(['a']);
  });
});
