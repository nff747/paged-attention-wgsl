import { describe, it, expect, beforeEach } from 'vitest';
import { BlockAllocator } from '../src/core/block_allocator';
import { SequenceManager } from '../src/core/sequence_manager';

describe('SequenceManager', () => {
  let allocator: BlockAllocator;
  let manager: SequenceManager;

  beforeEach(() => {
    allocator = new BlockAllocator(20, 4); // 4 tokens per block
    manager = new SequenceManager(allocator);
  });

  it('allocates required blocks for initial prompt', () => {
    const prompt = [101, 2045, 1037, 2812, 102]; // 5 tokens -> requires 2 blocks
    const seq = manager.createSequence(prompt);

    expect(seq.contextLen).toBe(5);
    expect(seq.blockTable.length).toBe(2);
    expect(allocator.getAllocatedCount()).toBe(2);
  });

  it('dynamically expands block table on token generation boundaries', () => {
    const seq = manager.createSequence([1, 2, 3]); // 3 tokens -> 1 block
    expect(seq.blockTable.length).toBe(1);

    manager.appendToken(seq.seqId, 4); // 4 tokens -> fills block 0
    expect(seq.blockTable.length).toBe(1);

    manager.appendToken(seq.seqId, 5); // 5 tokens -> allocates block 1
    expect(seq.blockTable.length).toBe(2);
    expect(seq.contextLen).toBe(5);
  });

  it('supports sequence forking with shared block tables', () => {
    const parent = manager.createSequence([1, 2, 3, 4]); // 1 block
    const child = manager.forkSequence(parent.seqId);

    expect(child.blockTable).toEqual(parent.blockTable);
    expect(allocator.isShared(parent.blockTable[0])).toBe(true);

    manager.freeSequence(parent.seqId);
    expect(allocator.isShared(child.blockTable[0])).toBe(false);
    expect(allocator.getAllocatedCount()).toBe(1);

    manager.freeSequence(child.seqId);
    expect(allocator.getAllocatedCount()).toBe(0);
  });

  it('handles empty prompt initialization gracefully', () => {
    const seq = manager.createSequence([]);
    expect(seq.contextLen).toBe(0);
    expect(seq.blockTable.length).toBe(1);
    expect(seq.status).toBe('prefill');
  });
});
