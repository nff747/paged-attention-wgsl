import { describe, it, expect } from 'vitest';
import { ShaderBuilder } from '../src/shaders/shader_builder';

describe('ShaderBuilder', () => {
  it('generates valid WGSL with specified block size and head dimension', () => {
    const wgsl = ShaderBuilder.build({ blockSize: 16, headDim: 64 });
    expect(wgsl).toContain('@workgroup_size(16, 1, 1)');
    expect(wgsl).toContain('array<f32, 64>');
    expect(wgsl).toContain('q_tensor');
    expect(wgsl).toContain('block_tables');
  });

  it('throws on unsupported block sizes', () => {
    expect(() => ShaderBuilder.build({ blockSize: 24, headDim: 64 })).toThrow();
  });

  it('validates headDim divisibility', () => {
    expect(() => ShaderBuilder.build({ blockSize: 16, headDim: 65 })).toThrow();
  });

  it('generates valid WGSL for head dimension 128', () => {
    const wgsl = ShaderBuilder.build({ blockSize: 32, headDim: 128 });
    expect(wgsl).toContain('@workgroup_size(32, 1, 1)');
    expect(wgsl).toContain('array<f32, 128>');
  });
});
