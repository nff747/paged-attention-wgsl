import { PhysicalBlock } from './types';

/**
 * High-performance free-list physical block allocator with Copy-On-Write support.
 */
export class BlockAllocator {
  public readonly blockSize: number;
  public readonly totalBlocks: number;
  private readonly blocks: PhysicalBlock[];
  private readonly freeList: number[];
  private allocatedCount: number = 0;

  constructor(totalBlocks: number, blockSize: number = 16) {
    if (totalBlocks <= 0) throw new Error('totalBlocks must be greater than 0');
    if (blockSize <= 0) throw new Error('blockSize must be greater than 0');

    this.totalBlocks = totalBlocks;
    this.blockSize = blockSize;
    this.blocks = new Array<PhysicalBlock>(totalBlocks);
    this.freeList = new Array<number>(totalBlocks);

    for (let i = 0; i < totalBlocks; i++) {
      this.blocks[i] = {
        blockId: i,
        refCount: 0,
        lastAccessed: 0,
      };
      this.freeList[i] = totalBlocks - 1 - i;
    }
  }

  public getFreeBlockCount(): number {
    return this.freeList.length;
  }

  public getAllocatedCount(): number {
    return this.allocatedCount;
  }

  public allocateBulk(count: number): number[] {
    if (this.freeList.length < count) {
      throw new Error(`Insufficient free blocks. Requested ${count}, available ${this.freeList.length}`);
    }
    const allocated: number[] = [];
    for (let i = 0; i < count; i++) {
      allocated.push(this.allocate());
    }
    return allocated;
  }

  public allocate(): number {
    if (this.freeList.length === 0) {
      throw new Error(`Out of physical KV-cache blocks. Total capacity: ${this.totalBlocks}`);
    }
    const blockId = this.freeList.pop()!;
    const block = this.blocks[blockId];
    block.refCount = 1;
    block.lastAccessed = Date.now();
    this.allocatedCount++;
    return blockId;
  }

  public free(blockId: number): void {
    this.validateBlockId(blockId);
    const block = this.blocks[blockId];
    if (block.refCount <= 0) {
      throw new Error(`Double free detected for blockId ${blockId}`);
    }

    block.refCount--;
    if (block.refCount === 0) {
      this.freeList.push(blockId);
      this.allocatedCount--;
    }
  }

  public retain(blockId: number): void {
    this.validateBlockId(blockId);
    const block = this.blocks[blockId];
    if (block.refCount <= 0) {
      throw new Error(`Cannot retain unallocated blockId ${blockId}`);
    }
    block.refCount++;
  }

  public isShared(blockId: number): boolean {
    this.validateBlockId(blockId);
    return this.blocks[blockId].refCount > 1;
  }

  public getRefCount(blockId: number): number {
    this.validateBlockId(blockId);
    return this.blocks[blockId].refCount;
  }

  public reset(): void {
    this.freeList.length = 0;
    this.allocatedCount = 0;
    for (let i = 0; i < this.totalBlocks; i++) {
      this.blocks[i].refCount = 0;
      this.blocks[i].lastAccessed = 0;
      this.freeList[i] = this.totalBlocks - 1 - i;
    }
  }

  private validateBlockId(blockId: number): void {
    if (blockId < 0 || blockId >= this.totalBlocks) {
      throw new RangeError(`Invalid blockId ${blockId}. Valid range: [0, ${this.totalBlocks - 1}]`);
    }
  }
}
