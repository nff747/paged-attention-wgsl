import { OnlineSoftmaxAccumulator } from './online_softmax';
import { BlockTable } from '../core/types';

/**
 * CPU Reference simulation of PagedAttention kernel reading from a physical block pool.
 */
export function pagedCpuAttentionDecode(
  query: Float32Array,               // [headDim]
  physicalKeys: Float32Array[][],    // [totalBlocks][blockSize][headDim]
  physicalValues: Float32Array[][],  // [totalBlocks][blockSize][headDim]
  blockTable: BlockTable,
  contextLen: number,
  blockSize: number,
  headDim: number,
  scale?: number
): Float32Array {
  const tau = scale ?? 1.0 / Math.sqrt(headDim);
  const accumulator = new OnlineSoftmaxAccumulator(headDim);
  const numBlocks = blockTable.length;

  for (let b = 0; b < numBlocks; b++) {
    const physicalBlockId = blockTable[b];
    const blockStartToken = b * blockSize;
    const tokensInThisBlock = Math.min(blockSize, Math.max(0, contextLen - blockStartToken));

    if (tokensInThisBlock <= 0) break;

    const kBlock = physicalKeys[physicalBlockId];
    const vBlock = physicalValues[physicalBlockId];
    const scores = new Float32Array(tokensInThisBlock);

    // Compute Q . K for tokens in this physical block
    for (let i = 0; i < tokensInThisBlock; i++) {
      const k = kBlock[i];
      let dot = 0.0;
      for (let d = 0; d < headDim; d++) {
        dot += query[d] * k[d];
      }
      scores[i] = dot * tau;
    }

    // Accumulate via online softmax
    accumulator.update(scores, vBlock, tokensInThisBlock);
  }

  return accumulator.finalize();
}
