import type {Selection} from '@decisive/browser/types.js';

export function inferenceConfig(model: {engine:string}, files: Record<string, Blob | Uint8Array>): Selection {
  const source = {files};
  switch (model.engine) {
    case 'decima': return {model:'decima', backend:'webgpu', source, parameters:{long:true}};
    case 'pplx-q4': return {model:'pplx-flash', backend:'webgpu', format:'q4-e4', source, parameters:{maxSequenceTokens:1024}};
    default: throw new Error('modelComingSoon');
  }
}
