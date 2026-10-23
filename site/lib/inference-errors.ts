export function inferenceErrorKey(code: string | undefined, engine?: string): string | undefined {
  switch (code) {
    case 'UNSUPPORTED_MODEL':
    case 'UNSUPPORTED_BACKEND':
    case 'UNSUPPORTED_FORMAT':
    case 'MODEL_NOT_COMPILED':
    case 'BACKEND_NOT_COMPILED':
    case 'FORMAT_NOT_COMPILED': return 'modelUnsupported';
    case 'MISSING_GPU_FEATURE': return 'gpuPrecisionRequired';
    case 'GPU_UNAVAILABLE': return 'webgpuUnavailable';
    case 'RESOURCE_LIMIT': return 'modelMemoryLimit';
    case 'CANCELLED': return 'paused';
    case 'INPUT_TOO_LONG': return engine === 'pplx-q4' ? 'pplxInputTooLong' : 'inputTooLong';
    default: return undefined;
  }
}
