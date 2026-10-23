export async function loadDecima(api, files, progress = () => {}) {
  progress('Loading Decima');
  const bytes=async name=>new Uint8Array(await files[name].arrayBuffer());
  const names=['pytorch/decima.json','pytorch/encoder/config.json','pytorch/encoder/tokenizer.json','pytorch/head.safetensors'];
  const [config,encoder,tokenizer,head]=await Promise.all(names.map(bytes));
  const model=await api.DecimaWebGpuModel.create(config,encoder,tokenizer,files['pytorch/encoder/model.safetensors'],head);
  return {
    dispose(){model.free();},
    async predict(request){
      const start=performance.now();
      const response=JSON.parse(await model.predict(JSON.stringify(request)));
      return {response,backend:'webgpu',model:'Decima Small',elapsed_ms:performance.now()-start};
    },
  };
}
