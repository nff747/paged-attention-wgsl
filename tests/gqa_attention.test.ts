import { describe, it, expect } from 'vitest';
import { mapQueryHeadToKvHead } from '../src/math/gqa';

describe('Grouped-Query Attention (GQA)', () => {
  it('maps 32 query heads to 8 KV heads correctly (4:1 ratio)', () => {
    // Heads 0..3 map to KV 0
    expect(mapQueryHeadToKvHead(0, 32, 8)).toBe(0);
    expect(mapQueryHeadToKvHead(3, 32, 8)).toBe(0);
    // Heads 4..7 map to KV 1
    expect(mapQueryHeadToKvHead(4, 32, 8)).toBe(1);
    expect(mapQueryHeadToKvHead(7, 32, 8)).toBe(1);
    // Heads 28..31 map to KV 7
    expect(mapQueryHeadToKvHead(28, 32, 8)).toBe(7);
    expect(mapQueryHeadToKvHead(31, 32, 8)).toBe(7);
  });

  it('handles standard Multi-Head Attention (1:1 ratio)', () => {
    for (let h = 0; h < 16; h++) {
      expect(mapQueryHeadToKvHead(h, 16, 16)).toBe(h);
    }
  });

  it('throws on non-divisible head configurations', () => {
    expect(() => mapQueryHeadToKvHead(0, 10, 3)).toThrow();
  });
});
