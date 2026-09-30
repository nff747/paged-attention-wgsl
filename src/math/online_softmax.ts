/**
 * Online Softmax accumulator for incremental tile-based reduction.
 * Enables streaming attention without materializing intermediate score matrices.
 */

export interface OnlineSoftmaxState {
  maxScore: number;
  sumExp: number;
  output: Float32Array;
}

export class OnlineSoftmaxAccumulator {
  public maxScore: number = -Infinity;
  public sumExp: number = 0.0;
  public readonly output: Float32Array;

  constructor(public readonly headDim: number) {
    this.output = new Float32Array(headDim);
  }

  /**
   * Updates accumulator state with a newly computed block of attention scores and values.
   */
  public update(blockScores: Float32Array, blockValues: Float32Array[], count: number): void {
    if (count <= 0) return;

    // Find maximum score in the incoming block
    let blockMax = -Infinity;
    for (let i = 0; i < count; i++) {
      if (blockScores[i] > blockMax) {
        blockMax = blockScores[i];
      }
    }

    const newMax = Math.max(this.maxScore, blockMax);
    // Underflow safeguard: skip scaling if diff exceeds numerical float range
    const diff = this.maxScore - newMax;
    const alpha = this.maxScore === -Infinity ? 0.0 : (diff < -88.0 ? 0.0 : Math.exp(diff));

    // Rescale previous accumulator sum and output vector
    this.sumExp *= alpha;
    for (let d = 0; d < this.headDim; d++) {
      this.output[d] *= alpha;
    }

    // Accumulate incoming block
    let blockSum = 0.0;
    for (let i = 0; i < count; i++) {
      const expVal = Math.exp(blockScores[i] - newMax);
      blockSum += expVal;
      const v = blockValues[i];
      for (let d = 0; d < this.headDim; d++) {
        this.output[d] += expVal * v[d];
      }
    }

    this.sumExp += blockSum;
    this.maxScore = newMax;
  }

  /**
   * Finalizes reduction by dividing accumulated output by the softmax denominator.
   */
  public finalize(): Float32Array {
    const result = new Float32Array(this.headDim);
    const normalizer = 1.0 / (this.sumExp || 1e-12);
    for (let d = 0; d < this.headDim; d++) {
      result[d] = this.output[d] * normalizer;
    }
    return result;
  }

  public reset(): void {
    this.maxScore = -Infinity;
    this.sumExp = 0.0;
    this.output.fill(0);
  }
}
