interface Device { destroy(): void }
interface Adapter { features?: { has(feature: string): boolean }; requestDevice(options: { requiredFeatures: string[] }): Promise<Device> }
interface GPU { requestAdapter(options: { powerPreference: string }): Promise<Adapter | null> }
const currentGPU = (): GPU | undefined => (globalThis.navigator as unknown as {gpu?:GPU} | undefined)?.gpu;

export async function checkWebGPU(gpu: GPU | undefined = currentGPU(), requiredFeatures: string[] = []): Promise<boolean> {
  if (!gpu?.requestAdapter) return false;
  for (let attempt = 0; attempt < 3; attempt++) {
    if (attempt) await new Promise(resolve => setTimeout(resolve, attempt * 400));
    try {
      const adapter = await gpu.requestAdapter({powerPreference:'high-performance'});
      if (!adapter) continue;
      if (requiredFeatures.some(feature => !adapter.features?.has(feature))) return false;
      const device = await adapter.requestDevice({requiredFeatures});
      device.destroy();
      return true;
    } catch {}
  }
  return false;
}
export async function probe(): Promise<{webgpu:boolean;shaderF16:boolean}> {
  const webgpu = await checkWebGPU();
  return {webgpu,shaderF16:webgpu && await checkWebGPU(undefined,['shader-f16'])};
}
