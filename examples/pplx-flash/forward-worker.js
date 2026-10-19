import init, * as api from '/pkg/decisive.js';
import {loadFlash} from '/runtime/runtime.js';

let engine,request,busy=false;
const progress=message=>self.postMessage({type:'progress',message});

function input512(base,model){
  const words=Array(60).fill(base.state).join(' ').split(/\s+/);
  const chunks=state=>JSON.parse(model.prepare(JSON.stringify({...base,state}),2048));
  const length=state=>chunks(state).reduce((sum,chunk)=>sum+chunk.ids.length,0);
  let lo=1,hi=words.length;
  while(lo<hi){
    const mid=Math.ceil((lo+hi)/2);
    if(length(words.slice(0,mid).join(' '))<=512)lo=mid;else hi=mid-1;
  }
  let state=words.slice(0,lo).join(' '),tokens=length(state);
  while(tokens<512){
    const next=state+' x',count=length(next);
    if(count<=tokens||count>512)throw Error('Could not construct exactly 512 input tokens');
    state=next;tokens=count;
  }
  const plan=chunks(state);
  if(plan.length!==1||plan[0].ids.length!==512)throw Error('Expected one sequence of exactly 512 tokens');
  return {...base,state};
}

self.onmessage=async({data})=>{
  if(busy)return;
  busy=true;
  try{
    if(data.type==='prepare'){
      if(engine)throw Error('Model is already prepared');
      await init();
      engine=await loadFlash(api,'/model',message=>{
        if(/Layer (8|16|24|32)\//.test(message))progress(message);
      },'webgpu',{mode:'q4-e4',resident:false,profile:false});
      const response=await fetch('/fixtures/request.json');
      if(!response.ok)throw Error('Input fixture is unavailable');
      request=input512(await response.json(),engine.model);
      self.postMessage({type:'prepared',tokens:512,request});
    }else if(data.type==='run'){
      if(!engine||!request)throw Error('Model is not ready');
      const label=`Pplx worker forward pass ${data.number}`;
      performance.mark(label+' start');console.timeStamp?.(label+' start');
      const result=await engine.predict(request,512,true);
      performance.mark(label+' end');performance.measure(label,label+' start',label+' end');
      console.timeStamp?.(label+' end');
      if(result.response.usage.input_tokens!==512||result.response.usage.sequences!==1||result.layers.length!==32)throw Error('Forward-pass shape mismatch');
      if(!result.queued_execution||!result.double_buffered)throw Error('Expected queued execution with double buffering');
      self.postMessage({type:'result',result});
    }else throw Error('Unknown command');
  }catch(error){
    self.postMessage({type:'error',message:String(error),prepared:Boolean(engine&&request)});
  }finally{busy=false;}
};
self.postMessage({type:'ready'});
