import test from 'node:test';
import assert from 'node:assert/strict';
import {WorkerClient} from '../lib/worker-client.js';
const tick = () => new Promise(resolve=>setTimeout(resolve,0));

test('storage refresh and download are ordered, but pause bypasses the queue',async()=>{
  const previous=globalThis.Worker, sent=[];
  class StubWorker {
    postMessage(message){sent.push(message);}
    terminate(){}
  }
  globalThis.Worker=StubWorker;
  try{
    const client=new WorkerClient(()=>{});
    const inspect=client.call('inspect'),download=client.call('download');
    await tick();assert.deepEqual(sent.map(m=>m.method),['inspect']);
    const pause=client.call('pause');
    assert.deepEqual(sent.map(m=>m.method),['inspect','pause']);
    client.worker.onmessage({data:{id:sent[1].id,result:true}});await pause;
    client.worker.onmessage({data:{id:sent[0].id,result:{ready:false}}});await inspect;
    await tick();assert.deepEqual(sent.map(m=>m.method),['inspect','pause','download']);
    client.worker.onmessage({data:{id:sent[2].id,result:{ready:true}}});
    assert.deepEqual(await download,{ready:true});client.close();
  }finally{globalThis.Worker=previous;}
});
