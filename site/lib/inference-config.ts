import type {Json, Selection, SyncReader} from '@decisive/browser/types.js';

export function inferenceConfig(model: {engine:string}, files: Record<string, Blob | Uint8Array | SyncReader>, tuningCache: Json = null): Selection {
  const source = {files}, tuning = tuningCache ? {cache:tuningCache} : undefined;
  switch (model.engine) {
    case 'decima': return {model:'decima', backend:'webgpu', source, parameters:{long:true}};
    case 'pplx-q4': return {model:'pplx-flash', backend:'webgpu', format:'q4-e4', source, tuning, parameters:{maxSequenceTokens:1024}};
    case 'pplx-q8': return {model:'pplx-flash', backend:'webgpu', format:'q8-e8', source, tuning, parameters:{maxSequenceTokens:1024}};
    default: throw new Error('modelComingSoon');
  }
}
