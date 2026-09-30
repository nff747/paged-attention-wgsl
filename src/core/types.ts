/**
 * Core type definitions for PagedAttention WebGPU compute engine.
 */

export interface CacheConfig {
  /** Number of tokens stored per physical block (typically 8, 16, or 32). */
  blockSize: number;
  /** Total physical blocks pre-allocated in the GPU buffer pool. */
  numBlocks: number;
  /** Number of transformer attention layers. */
  numLayers: number;
  /** Number of query attention heads. */
  numHeads: number;
  /** Number of Key-Value attention heads (for GQA/MQA). */
  numKvHeads: number;
  /** Hidden dimension per attention head (typically 64 or 128). */
  headDim: number;
  /** Numerical datatype for KV cache elements. */
  dtype: 'f32' | 'f16';
}

export interface PhysicalBlock {
  /** Unique index within the physical block pool [0, numBlocks - 1]. */
  blockId: number;
  /** Reference count for Copy-On-Write sharing (e.g. prefix caching). */
  refCount: number;
  /** Epoch timestamp of last access for LRU eviction policies. */
  lastAccessed: number;
}

export type BlockTable = number[];

export interface Sequence {
  /** Unique sequence identifier. */
  seqId: number;
  /** Ordered array of token IDs belonging to this sequence. */
  tokens: number[];
  /** Array of physical block IDs mapping logical blocks to GPU memory. */
  blockTable: BlockTable;
  /** Number of tokens currently cached. */
  contextLen: number;
  /** Generation status. */
  status: 'prefill' | 'decode' | 'finished';
}

export interface AttentionMetadata {
  /** Maximum context length in the current batch. */
  maxContextLen: number;
  /** Flat array of block tables per sequence [batchSize, maxBlocksPerSeq]. */
  blockTables: Int32Array;
  /** Array of sequence lengths in the current batch [batchSize]. */
  seqLens: Int32Array;
  /** Batch size currently executing. */
  batchSize: number;
}
