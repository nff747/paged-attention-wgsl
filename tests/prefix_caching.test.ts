import { describe, it, expect } from 'vitest';
import { BlockAllocator } from '../src/core/block_allocator';
import { SequenceManager } from '../src/core/sequence_manager';
import { pagedCpuAttentionDecode } from '../src/math/paged_cpu_attention';

describe('Prefix Caching Invariance', () => {
  it('yields bit-exact identical output when prompt blocks are shared across sequences', () => {
    const allocator = new BlockAllocator(30, 4);
    const manager = new SequenceManager(allocator);

    const prompt = [10, 20, 30, 40, 50, 60, 70, 80]; // 8 tokens = 2 blocks
    const seq1 = manager.createSequence(prompt);
    const seq2 = manager.forkSequence(seq1.seqId);

    // Both sequences share the exact same physical blocks
    expect(seq1.blockTable).toEqual(seq2.blockTable);

    // Populate mock physical key/value data
    const headDim = 16;
    const physicalKeys: Float32Array[][] = Array.from({ length: 30 }, () =>
      Array.from({ length: 4 }, () => new Float32Array(headDim).fill(1.5))
    );
    const physicalValues: Float32Array[][] = Array.from({ length: 30 }, () =>
      Array.from({ length: 4 }, () => new Float32Array(headDim).fill(2.5))
    );

    const query = new Float32Array(headDim).fill(0.5);

    const out1 = pagedCpuAttentionDecode(
      query, physicalKeys, physicalValues, seq1.blockTable, seq1.contextLen, 4, headDim
    );
    const out2 = pagedCpuAttentionDecode(
      query, physicalKeys, physicalValues, seq2.blockTable, seq2.contextLen, 4, headDim
    );

    expect(out1).toEqual(out2);
  });
});
