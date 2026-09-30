/**
 * Grouped-Query Attention (GQA) routing math.
 * Maps Query head index to corresponding Key-Value head index:
 * kv_head_idx = floor(query_head_idx / (num_query_heads / num_kv_heads))
 */

export function mapQueryHeadToKvHead(
  queryHeadIdx: number,
  numQueryHeads: number,
  numKvHeads: number
): number {
  if (numKvHeads <= 0 || numQueryHeads <= 0) {
    throw new Error('Number of heads must be positive');
  }
  if (numQueryHeads % numKvHeads !== 0) {
    throw new Error(`numQueryHeads (${numQueryHeads}) must be divisible by numKvHeads (${numKvHeads})`);
  }
  const groupSize = Math.floor(numQueryHeads / numKvHeads);
  return Math.floor(queryHeadIdx / groupSize);
}
