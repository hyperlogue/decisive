import init,* as api from '../dist/pkg/decisive.js';
import {loadDecima} from '../dist/lib/model-runtime.js';
import {buildRequest,readScores} from '../dist/lib/privacy.js';
self.onmessage=async({data})=>{
  try{
    await init();
    const engine=await loadDecima(api,data.files,message=>self.postMessage({message}));
    const filler='I grew up loving books and spent most summers reading in the garden. These days I mostly cook, hike, and take far too many photos of clouds. '.repeat(14);
    for(const draft of ['My name is Alex Rivera. My email is alex@example.com.', '我住在北京。我有糖尿病。','Hello everyone! Happy to join this community.',
      filler+'My Wi-Fi password is Tr0ub4dor&3, I keep forgetting it. '+filler]){
      const request=buildRequest(draft);const output=await engine.predict(request);
      self.postMessage({result:{draft,scores:readScores(output.response),response:output.response,elapsed_ms:output.elapsed_ms,backend:output.backend}});
    }
    engine.dispose();
  }catch(error){self.postMessage({error:String(error)+'\n'+error.stack});}
};
