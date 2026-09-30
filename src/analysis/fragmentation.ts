/**
 * Mathematical model analyzing internal & external memory fragmentation:
 * Contiguous pre-allocation vs PagedAttention block-tiling.
 */

export interface FragmentationMetrics {
  totalAllocatedBytes: number;
  actuallyUsedBytes: number;
  wastedBytes: number;
  wastePercentage: number;
}

export class FragmentationAnalyzer {
  /**
   * Models contiguous pre-allocation memory waste where each sequence
   * is reserved up to maxContextLen tokens.
   */
  public static analyzeContiguous(
    seqLens: number[],
    maxContextLen: number,
    bytesPerToken: number
  ): FragmentationMetrics {
    const batchSize = seqLens.length;
    const totalAllocatedBytes = batchSize * maxContextLen * bytesPerToken;
    let actuallyUsedBytes = 0;

    for (const len of seqLens) {
      actuallyUsedBytes += len * bytesPerToken;
    }

    const wastedBytes = totalAllocatedBytes - actuallyUsedBytes;
    const wastePercentage = totalAllocatedBytes === 0 ? 0 : (wastedBytes / totalAllocatedBytes) * 100;

    return {
      totalAllocatedBytes,
      actuallyUsedBytes,
      wastedBytes,
      wastePercentage,
    };
  }

  /**
   * Models PagedAttention memory waste where only the final block of each sequence
   * can have un-filled slots.
   */
  /**
   * Models multi-turn conversation where prefix blocks are retained across user queries.
   */
  public static analyzeMultiTurnSharing(
    numTurns: number,
    systemPromptTokens: number,
    tokensPerTurn: number,
    blockSize: number
  ): { sharedBlocks: number; savedBytes: number } {
    const sharedBlocks = Math.ceil(systemPromptTokens / blockSize);
    const savedBytes = (numTurns - 1) * sharedBlocks * blockSize * 16384;
    return { sharedBlocks, savedBytes };
  }

  public static analyzePaged(
    seqLens: number[],
    blockSize: number,
    bytesPerToken: number
  ): FragmentationMetrics {
    let totalAllocatedBlocks = 0;
    let actuallyUsedBytes = 0;

    for (const len of seqLens) {
      const blocksNeeded = Math.ceil(len / blockSize);
      totalAllocatedBlocks += blocksNeeded;
      actuallyUsedBytes += len * bytesPerToken;
    }

    const totalAllocatedBytes = totalAllocatedBlocks * blockSize * bytesPerToken;
    const wastedBytes = totalAllocatedBytes - actuallyUsedBytes;
    const wastePercentage = totalAllocatedBytes === 0 ? 0 : (wastedBytes / totalAllocatedBytes) * 100;

    return {
      totalAllocatedBytes,
      actuallyUsedBytes,
      wastedBytes,
      wastePercentage,
    };
  }
}
