export function selectModel(catalog, id = catalog.default) {
  const model = catalog.models.find(model => model.id === id);
  if (!model) throw new Error('unknownModel');
  return model;
}
export function requirePublished(model) {
  if (!model.available || !model.repository || !model.revision || !Object.keys(model.files).length) {
    throw new Error('modelComingSoon');
  }
  return model;
}
export function emptyModelState(model) {
  const files=Object.entries(model.files).map(([name,f])=>({
    name,expected:f.bytes,bytes:0,verified:false,state:'missing',
  }));
  const total=files.reduce((n,f)=>n+f.expected,0);
  return {modelId:model.id,revision:model.revision,files,total,stored:0,verified:0,remaining:total,ready:false,legacyStored:0};
}

export function validateModelState(model, state) {
  const expected=emptyModelState(model);
  if(state?.modelId!==model.id || state.revision!==model.revision || state.total!==expected.total ||
     !Array.isArray(state.files) || state.files.length!==expected.files.length ||
     new Set(state.files.map(f=>f.name)).size!==expected.files.length ||
     !state.files.every(f=>model.files[f.name]?.bytes===f.expected)) throw new Error('versionMismatch');
  return state;
}
