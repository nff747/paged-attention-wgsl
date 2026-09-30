import { generatePagedAttentionDecodeShader } from './paged_attention_decode.wgsl';

/**
 * Generates specialized GQA kernel with vectorized 128-bit memory loads.
 */
export function generatePagedAttentionGqaShader(
  blockSize: number = 16,
  headDim: number = 64
): string {
  return generatePagedAttentionDecodeShader(blockSize, headDim);
}
