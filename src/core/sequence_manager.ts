import { BlockAllocator } from './block_allocator';
import { Sequence, BlockTable } from './types';

/**
 * Manages active LLM sequences, prompt allocation, token appends, and Copy-On-Write branching.
 */
export class SequenceManager {
  private readonly allocator: BlockAllocator;
  private readonly sequences: Map<number, Sequence> = new Map();
  private nextSeqId: number = 1;

  constructor(allocator: BlockAllocator) {
    this.allocator = allocator;
  }

  public get blockSize(): number {
    return this.allocator.blockSize;
  }

  public createSequence(promptTokens: number[] = []): Sequence {
    const seqId = this.nextSeqId++;
    const numBlocksNeeded = Math.ceil(promptTokens.length / this.blockSize) || 1;
    const blockTable: BlockTable = [];

    for (let i = 0; i < numBlocksNeeded; i++) {
      blockTable.push(this.allocator.allocate());
    }

    const seq: Sequence = {
      seqId,
      tokens: [...promptTokens],
      blockTable,
      contextLen: promptTokens.length,
      status: promptTokens.length > 0 ? 'decode' : 'prefill',
    };

    this.sequences.set(seqId, seq);
    return seq;
  }

  public appendToken(seqId: number, token: number): void {
    const seq = this.sequences.get(seqId);
    if (!seq) throw new Error(`Sequence ${seqId} not found`);

    const currentLen = seq.contextLen;
    // Check if new block is needed
    if (currentLen > 0 && currentLen % this.blockSize === 0) {
      seq.blockTable.push(this.allocator.allocate());
    }

    seq.tokens.push(token);
    seq.contextLen++;
  }

  public forkSequence(parentSeqId: number): Sequence {
    const parent = this.sequences.get(parentSeqId);
    if (!parent) throw new Error(`Parent sequence ${parentSeqId} not found`);

    const childSeqId = this.nextSeqId++;
    // Retain parent blocks for Copy-On-Write sharing
    for (const blockId of parent.blockTable) {
      this.allocator.retain(blockId);
    }

    const child: Sequence = {
      seqId: childSeqId,
      tokens: [...parent.tokens],
      blockTable: [...parent.blockTable],
      contextLen: parent.contextLen,
      status: parent.status,
    };

    this.sequences.set(childSeqId, child);
    return child;
  }

  public freeSequence(seqId: number): void {
    const seq = this.sequences.get(seqId);
    if (!seq) return;

    for (const blockId of seq.blockTable) {
      this.allocator.free(blockId);
    }
    this.sequences.delete(seqId);
  }

  public getSequence(seqId: number): Sequence | undefined {
    return this.sequences.get(seqId);
  }

  public getActiveCount(): number {
    return this.sequences.size;
  }

  public clear(): void {
    for (const seqId of Array.from(this.sequences.keys())) {
      this.freeSequence(seqId);
    }
  }
}
