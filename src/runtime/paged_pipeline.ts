import { CacheConfig, AttentionMetadata } from '../core/types';
import { ShaderBuilder } from '../shaders/shader_builder';
import { PagedKvCache } from './paged_kv_cache';

/**
 * Orchestrates compute pipeline, bind groups, and dispatch grid for PagedAttention.
 */
export class PagedAttentionPipeline {
  private cachedBindGroup: GPUBindGroup | null = null;

  private pipeline: GPUComputePipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;

  constructor(
    public readonly device: GPUDevice,
    public readonly config: CacheConfig
  ) {
    this.initPipeline();
  }

  private initPipeline(): void {
    const wgsl = ShaderBuilder.build({
      blockSize: this.config.blockSize,
      headDim: this.config.headDim,
      isGqa: this.config.numHeads !== this.config.numKvHeads,
    });

    const shaderModule = this.device.createShaderModule({
      label: 'paged_attention_shader',
      code: wgsl,
    });

    this.pipeline = this.device.createComputePipeline({
      label: 'paged_attention_pipeline',
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });

    this.uniformBuffer = this.device.createBuffer({
      label: 'paged_attention_uniforms',
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  public dispatch(
    commandEncoder: GPUCommandEncoder,
    qBuffer: GPUBuffer,
    kvCache: PagedKvCache,
    blockTableBuffer: GPUBuffer,
    contextLenBuffer: GPUBuffer,
    outBuffer: GPUBuffer,
    metadata: AttentionMetadata
  ): void {
    if (!this.pipeline || !this.uniformBuffer || !kvCache.keyBuffer || !kvCache.valueBuffer) {
      throw new Error('Pipeline or buffers not initialized');
    }

    // Uniform buffer updates
    const uniformData = new ArrayBuffer(32);
    const u32View = new Uint32Array(uniformData);
    const f32View = new Float32Array(uniformData);

    u32View[0] = metadata.batchSize;
    u32View[1] = this.config.numHeads;
    u32View[2] = this.config.numKvHeads;
    u32View[3] = this.config.headDim;
    u32View[4] = this.config.blockSize;
    u32View[5] = Math.ceil(metadata.maxContextLen / this.config.blockSize);
    f32View[6] = 1.0 / Math.sqrt(this.config.headDim);

    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: qBuffer } },
        { binding: 2, resource: { buffer: kvCache.keyBuffer } },
        { binding: 3, resource: { buffer: kvCache.valueBuffer } },
        { binding: 4, resource: { buffer: blockTableBuffer } },
        { binding: 5, resource: { buffer: contextLenBuffer } },
        { binding: 6, resource: { buffer: outBuffer } },
      ],
    });

    const pass = commandEncoder.beginComputePass({ label: 'paged_attention_pass' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.config.numHeads, metadata.batchSize, 1);
    pass.end();
  }
}
