export async function checkWebGPU(gpu = globalThis.navigator?.gpu, requiredFeatures = []) {
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
    } catch {
    }
  }
  return false;
}
