import { describe, it, expect } from 'vitest';
import {
  getOpColor,
  getOpCategory,
  getStatusColor,
  isLightNodeColor,
  STATUS_COLORS,
  LIGHT_NODE_CATEGORIES,
} from '../opColors';

describe('getOpColor', () => {
  it('returns correct color for known ops', () => {
    expect(getOpColor('Convolution')).toBe('#335588');
    expect(getOpColor('Relu')).toBe('#702921');
    expect(getOpColor('Add')).toBe('#335544');
    expect(getOpColor('Reshape')).toBe('#6C4F47');
  });

  it('returns default for unknown ops', () => {
    expect(getOpColor('UnknownOp')).toBe('#333333');
  });
});

describe('getOpCategory', () => {
  it('returns correct category for known ops', () => {
    expect(getOpCategory('Convolution')).toBe('Convolution');
    expect(getOpCategory('Relu')).toBe('Activation');
    expect(getOpCategory('MaxPool')).toBe('Pooling');
  });

  it('returns Other for unknown ops', () => {
    expect(getOpCategory('UnknownOp')).toBe('Other');
  });
});

describe('getStatusColor', () => {
  it('returns correct colors for all statuses', () => {
    expect(getStatusColor('waiting')).toBe('#E5A820');
    expect(getStatusColor('executing')).toBe('#4C8DFF');
    expect(getStatusColor('success')).toBe('#34C77B');
    expect(getStatusColor('failed')).toBe('#E54D4D');
  });

  it('returns transparent for unknown status', () => {
    expect(getStatusColor('unknown')).toBe('transparent');
  });
});

describe('STATUS_COLORS', () => {
  it('has all four status colors', () => {
    expect(Object.keys(STATUS_COLORS)).toHaveLength(4);
    expect(STATUS_COLORS).toHaveProperty('waiting');
    expect(STATUS_COLORS).toHaveProperty('executing');
    expect(STATUS_COLORS).toHaveProperty('success');
    expect(STATUS_COLORS).toHaveProperty('failed');
  });
});

describe('isLightNodeColor', () => {
  it('identifies light colors', () => {
    expect(isLightNodeColor('#eeeeee')).toBe(true);
    expect(isLightNodeColor('#ffffff')).toBe(true);
  });

  it('identifies dark colors', () => {
    expect(isLightNodeColor('#335588')).toBe(false);
    expect(isLightNodeColor('#702921')).toBe(false);
    expect(isLightNodeColor('#333333')).toBe(false);
  });

  it('returns false for invalid hex', () => {
    expect(isLightNodeColor('#fff')).toBe(false);
  });
});

describe('LIGHT_NODE_CATEGORIES', () => {
  it('contains Parameter, Constant, Result', () => {
    expect(LIGHT_NODE_CATEGORIES.has('Constant')).toBe(true);
    expect(LIGHT_NODE_CATEGORIES.has('Parameter')).toBe(true);
    expect(LIGHT_NODE_CATEGORIES.has('Result')).toBe(true);
  });
});
