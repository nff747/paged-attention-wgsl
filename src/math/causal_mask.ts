/**
 * Causal mask calculation for autoregressive prefill & decode steps.
 */
export function isCausalMasked(queryIdx: number, keyIdx: number): boolean {
  return keyIdx > queryIdx;
}
