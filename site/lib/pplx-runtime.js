const importRuntime = async (...args) => (await import('../runtime/runtime.js')).loadFlash(...args);
export const PPLX_MAX_TOKENS = 1024;

export async function loadPplxFlash(api, files, progress = () => {}, loadRuntime = importRuntime) {
  let activeSignal;
  const runtime = await loadRuntime(api, files, message => {
    activeSignal?.throwIfAborted();
    progress(message);
  }, 'webgpu', {mode:'q4-e4', resident:false, profile:false});
  return {
    dispose() { runtime.dispose(); },
    async predict(request, signal) {
      activeSignal = signal;
      try {
        signal?.throwIfAborted();
        const output = await runtime.predict(request, PPLX_MAX_TOKENS);
        signal?.throwIfAborted();
        return {response:output.response, backend:output.backend, model:'Pplx Flash · Q4', elapsed_ms:output.elapsed_ms};
      } finally { activeSignal = null; }
    },
  };
}
