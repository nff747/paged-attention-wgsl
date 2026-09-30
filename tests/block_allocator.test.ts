import { describe, it, expect, beforeEach } from 'vitest';
import { BlockAllocator } from '../src/core/block_allocator';

describe('BlockAllocator', () => {
  let allocator: BlockAllocator;

  beforeEach(() => {
    allocator = new BlockAllocator(10, 16);
  });

  it('initializes with correct capacity and all blocks free', () => {
    expect(allocator.totalBlocks).toBe(10);
    expect(allocator.getFreeBlockCount()).toBe(10);
    expect(allocator.getAllocatedCount()).toBe(0);
  });

  it('allocates and frees physical blocks in O(1)', () => {
    const id1 = allocator.allocate();
    const id2 = allocator.allocate();
    expect(allocator.getAllocatedCount()).toBe(2);
    expect(allocator.getFreeBlockCount()).toBe(8);

    allocator.free(id1);
    expect(allocator.getAllocatedCount()).toBe(1);
    expect(allocator.getFreeBlockCount()).toBe(9);

    allocator.free(id2);
    expect(allocator.getAllocatedCount()).toBe(0);
    expect(allocator.getFreeBlockCount()).toBe(10);
  });

  it('supports Copy-On-Write reference counting', () => {
    const id = allocator.allocate();
    expect(allocator.isShared(id)).toBe(false);

    allocator.retain(id);
    expect(allocator.isShared(id)).toBe(true);
    expect(allocator.getRefCount(id)).toBe(2);

    allocator.free(id);
    expect(allocator.isShared(id)).toBe(false);
    expect(allocator.getAllocatedCount()).toBe(1);

    allocator.free(id);
    expect(allocator.getAllocatedCount()).toBe(0);
  });

  it('throws when exhausting pool capacity', () => {
    for (let i = 0; i < 10; i++) {
      allocator.allocate();
    }
    expect(allocator.getFreeBlockCount()).toBe(0);
    expect(() => allocator.allocate()).toThrow('Out of physical KV-cache blocks');
  });

  it('throws on double free', () => {
    const id = allocator.allocate();
    allocator.free(id);
    expect(() => allocator.free(id)).toThrow('Double free detected');
  });

  it('validates block bounds', () => {
    expect(() => allocator.free(-1)).toThrow(RangeError);
    expect(() => allocator.free(100)).toThrow(RangeError);
  });
});
