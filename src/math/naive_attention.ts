/**
 * Mathematical reference implementation of Un-paged Scaled Dot-Product Attention.
 * Used for bit-exact ground truth verification against PagedAttention.
 */

export function naiveScaledDotProductAttention(
  query: Float32Array,   // [headDim]
  keys: Float32Array[],   // seqLen array of [headDim]
  values: Float32Array[], // seqLen array of [headDim]
  headDim: number,
  scale?: number
): Float32Array {
  const seqLen = keys.length;
  if (seqLen === 0) return new Float32Array(headDim);

  const tau = scale ?? 1.0 / Math.sqrt(headDim);
  const scores = new Float32Array(seqLen);
  let maxScore = -Infinity;

  // 1. Compute dot-product scores: S_i = (Q . K_i) * tau
  for (let i = 0; i < seqLen; i++) {
    const k = keys[i];
    let dot = 0.0;
    // Unroll 4x for SIMD efficiency
    let d = 0;
    for (; d + 3 < headDim; d += 4) {
      dot += query[d] * k[d] + query[d + 1] * k[d + 1] + query[d + 2] * k[d + 2] + query[d + 3] * k[d + 3];
    }
    for (; d < headDim; d++) {
      dot += query[d] * k[d];
    }
    const s = dot * tau;
    scores[i] = s;
    if (s > maxScore) maxScore = s;
  }

  // 2. Compute softmax denominator and probabilities: P_i = exp(S_i - maxScore) / sum
  let sumExp = 0.0;
  const probs = new Float32Array(seqLen);
  for (let i = 0; i < seqLen; i++) {
    const expVal = Math.exp(scores[i] - maxScore);
    probs[i] = expVal;
    sumExp += expVal;
  }

  const invSum = 1.0 / (sumExp || 1e-12);
  for (let i = 0; i < seqLen; i++) {
    probs[i] *= invSum;
  }

  // 3. Compute weighted output: O = sum(P_i * V_i)
  const output = new Float32Array(headDim);
  for (let i = 0; i < seqLen; i++) {
    const p = probs[i];
    const v = values[i];
    for (let d = 0; d < headDim; d++) {
      output[d] += p * v[d];
    }
  }

  return output;
}
