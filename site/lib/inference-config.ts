import type {Selection, SyncReader} from '@decisive/browser/types.js';

export function inferenceConfig(model: {engine:string}, files: Record<string, Blob | Uint8Array | SyncReader>): Selection {
  const source = {files};
  switch (model.engine) {
    case 'decima': return {model:'decima', backend:'webgpu', source, parameters:{long:true}};
    case 'pplx-q4': return {model:'pplx-flash', backend:'webgpu', format:'q4-e4', source, parameters:{maxSequenceTokens:1024}};
    case 'pplx-q8': return {model:'pplx-flash', backend:'webgpu', format:'q8-e8', source, parameters:{maxSequenceTokens:1024}};
    default: throw new Error('modelComingSoon');
  }
}
