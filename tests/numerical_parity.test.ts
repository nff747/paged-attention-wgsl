import { describe, it, expect } from 'vitest';
import { naiveScaledDotProductAttention } from '../src/math/naive_attention';
import { pagedCpuAttentionDecode } from '../src/math/paged_cpu_attention';

describe('Numerical Parity: PagedAttention vs Naive Attention', () => {
  it('achieves exact parity across varying context lengths and block sizes', () => {
    const headDim = 32;
    const blockSize = 4;
    const contextLengths = [1, 4, 7, 16, 27, 32];

    for (const seqLen of contextLengths) {
      // Deterministic synthetic inputs
      const query = new Float32Array(headDim);
      for (let d = 0; d < headDim; d++) query[d] = Math.sin(d + 1);

      const keys: Float32Array[] = [];
      const values: Float32Array[] = [];
      for (let t = 0; t < seqLen; t++) {
        const k = new Float32Array(headDim);
        const v = new Float32Array(headDim);
        for (let d = 0; d < headDim; d++) {
          k[d] = Math.cos(t * headDim + d);
          v[d] = Math.sin((t + 1) * (d + 1));
        }
        keys.push(k);
        values.push(v);
      }

      // Ground truth naive attention
      const expected = naiveScaledDotProductAttention(query, keys, values, headDim);

      // Pack into non-contiguous physical blocks
      const numBlocks = Math.ceil(seqLen / blockSize);
      const totalBlocks = numBlocks + 4; // Add slack
      const physicalKeys: Float32Array[][] = Array.from({ length: totalBlocks }, () =>
        Array.from({ length: blockSize }, () => new Float32Array(headDim))
      );
      const physicalValues: Float32Array[][] = Array.from({ length: totalBlocks }, () =>
        Array.from({ length: blockSize }, () => new Float32Array(headDim))
      );

      // Map logical blocks to arbitrary non-contiguous physical IDs
      const blockTable: number[] = [];
      const shuffledIds = [3, 1, 4, 0, 2, 5, 7, 6];
      for (let b = 0; b < numBlocks; b++) {
        const pId = shuffledIds[b % shuffledIds.length];
        blockTable.push(pId);
        for (let i = 0; i < blockSize; i++) {
          const tokenIdx = b * blockSize + i;
          if (tokenIdx < seqLen) {
            physicalKeys[pId][i].set(keys[tokenIdx]);
            physicalValues[pId][i].set(values[tokenIdx]);
          }
        }
      }

      const actual = pagedCpuAttentionDecode(
        query,
        physicalKeys,
        physicalValues,
        blockTable,
        seqLen,
        blockSize,
        headDim
      );

      // Compare max absolute error
      let maxDiff = 0.0;
      for (let d = 0; d < headDim; d++) {
        const diff = Math.abs(expected[d] - actual[d]);
        if (diff > maxDiff) maxDiff = diff;
      }

      expect(maxDiff).toBeLessThan(1e-5);
    }
  });
});

  it('handles extreme dynamic range without numerical overflow', () => {
    const headDim = 16;
    const query = new Float32Array(headDim).fill(50.0);
    const keys = [new Float32Array(headDim).fill(50.0), new Float32Array(headDim).fill(-50.0)];
    const values = [new Float32Array(headDim).fill(1.0), new Float32Array(headDim).fill(2.0)];

    const out = naiveScaledDotProductAttention(query, keys, values, headDim);
    expect(Number.isFinite(out[0])).toBe(true);
    expect(out[0]).toBeCloseTo(1.0, 4);
  });
