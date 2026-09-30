import { generatePagedAttentionDecodeShader } from './paged_attention_decode.wgsl';
import { generatePagedAttentionGqaShader } from './paged_attention_gqa.wgsl';

export interface ShaderBuildOptions {
  blockSize: number;
  headDim: number;
  isGqa?: boolean;
}

export class ShaderBuilder {
  public static build(options: ShaderBuildOptions): string {
    const { blockSize, headDim, isGqa } = options;

    if (blockSize < 8 || blockSize > 64 || (blockSize & (blockSize - 1)) !== 0) {
      throw new Error(`blockSize ${blockSize} must be a power of two between 8 and 64.`);
    }
    if (![8, 16, 32, 64].includes(blockSize)) {
      throw new Error(`Invalid blockSize ${blockSize}. Expected 8, 16, or 32.`);
    }
    if (headDim <= 0 || headDim % 4 !== 0) {
      throw new Error(`headDim ${headDim} must be positive and multiple of 4`);
    }

    if (isGqa) {
      return generatePagedAttentionGqaShader(blockSize, headDim);
    }
    return generatePagedAttentionDecodeShader(blockSize, headDim);
  }
}
