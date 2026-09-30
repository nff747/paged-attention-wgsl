import { describe, it, expect } from 'vitest';
import { FragmentationAnalyzer } from '../src/analysis/fragmentation';

describe('FragmentationAnalyzer', () => {
  it('demonstrates massive memory savings of PagedAttention over contiguous pre-allocation', () => {
    // 8 sequences with variable lengths between 64 and 512 tokens
    const seqLens = [64, 128, 210, 340, 150, 480, 95, 310];
    const maxContext = 2048; // Common pre-allocation reservation
    const bytesPerToken = 2 * 32 * 64 * 4; // 2 (K+V) * 32 heads * 64 dim * 4 bytes = 16 KB

    const contiguous = FragmentationAnalyzer.analyzeContiguous(seqLens, maxContext, bytesPerToken);
    const paged = FragmentationAnalyzer.analyzePaged(seqLens, 16, bytesPerToken);

    // Contiguous wastes ~89% of memory
    expect(contiguous.wastePercentage).toBeGreaterThan(85);

    // Paged wastes < 4% of memory
    expect(paged.wastePercentage).toBeLessThan(5);

    // Paged saves > 80% total allocated VRAM
    expect(paged.totalAllocatedBytes).toBeLessThan(contiguous.totalAllocatedBytes * 0.2);
  });
});
