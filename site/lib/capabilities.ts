export async function probeCapabilities() {
  const state = {wasm:false, simd:false, gpu:false, storage:false, secure:isSecureContext};
  try {
    state.wasm = WebAssembly.validate(new Uint8Array([0,97,115,109,1,0,0,0]));
    state.simd = WebAssembly.validate(new Uint8Array([
      0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,10,10,1,8,0,65,0,253,15,253,98,11,
    ]));
  } catch {}
  try {
    const root = await navigator.storage.getDirectory();
    state.storage = !!root && typeof (FileSystemFileHandle.prototype as FileSystemFileHandle & {createSyncAccessHandle?:unknown}).createSyncAccessHandle === 'function' && !!navigator.locks;
  } catch {}
  if (!state.storage && typeof navigator.storage?.getDirectory === 'function' && navigator.locks && state.secure) {
    state.storage = true;
  }
  return state;
}
