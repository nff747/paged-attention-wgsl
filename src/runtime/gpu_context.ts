/**
 * WebGPU hardware initialization and feature/limit probing.
 */
export interface GpuCapabilities {
  hasF16: boolean;
  maxStorageBufferBindingSize: number;
  maxComputeWorkgroupStorageSize: number;
}

export class GpuContext {
  public device: GPUDevice | null = null;
  public adapter: GPUAdapter | null = null;
  public capabilities: GpuCapabilities | null = null;

  public async init(customDevice?: GPUDevice): Promise<boolean> {
    if (customDevice) {
      this.device = customDevice;
      this.capabilities = {
        hasF16: customDevice.features.has('shader-f16'),
        maxStorageBufferBindingSize: customDevice.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: customDevice.limits.maxComputeWorkgroupStorageSize,
      };
      return true;
    }

    if (typeof navigator === 'undefined' || !navigator.gpu) {
      return false;
    }

    try {
      this.adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
      if (!this.adapter) return false;

      const requiredFeatures: GPUFeatureName[] = [];
      if (this.adapter.features.has('shader-f16')) {
        requiredFeatures.push('shader-f16');
      }

      this.device = await this.adapter.requestDevice({
        requiredFeatures,
      });

      this.capabilities = {
        hasF16: this.device.features.has('shader-f16'),
        maxStorageBufferBindingSize: this.device.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: this.device.limits.maxComputeWorkgroupStorageSize,
      };
      return true;
    } catch {
      return false;
    }
  }

  public isAvailable(): boolean {
    return this.device !== null;
  }
}
