/**
 * Capacity planner estimating maximum concurrent sequences in WebGPU storage limits.
 */
export class ThroughputPlanner {
  public static calculateMaxBatchSize(
    availableGpuBytes: number,
    avgContextLen: number,
    blockSize: number,
    numKvHeads: number,
    headDim: number,
    dtypeBytes: number = 4
  ): { maxConcurrentPaged: number; maxConcurrentContiguous: number } {
    const bytesPerToken = 2 * numKvHeads * headDim * dtypeBytes;

    // Contiguous assumes worst-case maxContextLen (e.g. 2048)
    const maxContextLen = 2048;
    const contiguousBytesPerSeq = maxContextLen * bytesPerToken;
    const maxConcurrentContiguous = Math.floor(availableGpuBytes / contiguousBytesPerSeq);

    // Paged only allocates for avgContextLen + 1 partial block
    const pagedBlocksPerSeq = Math.ceil(avgContextLen / blockSize);
    const pagedBytesPerSeq = pagedBlocksPerSeq * blockSize * bytesPerToken;
    const maxConcurrentPaged = Math.floor(availableGpuBytes / pagedBytesPerSeq);

    return {
      maxConcurrentPaged,
      maxConcurrentContiguous,
    };
  }
}

export function formatThroughputTable(
  capacity: { maxConcurrentPaged: number; maxConcurrentContiguous: number }
): string {
  return `| Strategy | Max Concurrent Sequences | Gain |
| :--- | :---: | :---: |
| Contiguous Pre-allocation | ${capacity.maxConcurrentContiguous} | 1.0x |
| PagedAttention (Block 16) | ${capacity.maxConcurrentPaged} | ${(capacity.maxConcurrentPaged / Math.max(1, capacity.maxConcurrentContiguous)).toFixed(1)}x |`;
}
