import { CacheConfig } from '../core/types';

/**
 * Manages physical GPU storage buffers for Key and Value block caches.
 */
export class PagedKvCache {
  public readonly config: CacheConfig;
  public keyBuffer: GPUBuffer | null = null;
  public valueBuffer: GPUBuffer | null = null;
  public readonly totalBytesPerBuffer: number;

  constructor(config: CacheConfig) {
    this.config = config;
    const elementsPerBlock = config.numKvHeads * config.blockSize * config.headDim;
    const bytesPerElement = config.dtype === 'f16' ? 2 : 4;
    this.totalBytesPerBuffer = config.numBlocks * elementsPerBlock * bytesPerElement;
  }

  public allocateBuffers(device: GPUDevice): void {
    this.keyBuffer = device.createBuffer({
      label: 'paged_attention_key_cache',
      size: this.totalBytesPerBuffer,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.valueBuffer = device.createBuffer({
      label: 'paged_attention_value_cache',
      size: this.totalBytesPerBuffer,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  public destroy(): void {
    if (this.keyBuffer) {
      this.keyBuffer.destroy();
      this.keyBuffer = null;
    }
    if (this.valueBuffer) {
      this.valueBuffer.destroy();
      this.valueBuffer = null;
    }
  }
}
