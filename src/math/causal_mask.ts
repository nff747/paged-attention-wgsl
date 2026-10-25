/**
 * Causal mask calculation for autoregressive prefill & decode steps.
 */
export function isCausalMasked(queryIdx: number, keyIdx: number): boolean {
  return keyIdx > queryIdx;
}

/**
 * Generates an N x N lower-triangular causal attention mask matrix.
 * Returns 0.0 for allowed attention and -1e9 (-Infinity surrogate) for masked positions.
 */
export function generateCausalMaskMatrix(seqLen: number): Float32Array {
  const mask = new Float32Array(seqLen * seqLen);
  for (let q = 0; q < seqLen; q++) {
    for (let k = 0; k < seqLen; k++) {
      mask[q * seqLen + k] = isCausalMasked(q, k) ? -1e9 : 0.0;
    }
  }
  return mask;
}
