import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {webcrypto} from 'node:crypto';
import {Blob,atob} from 'node:buffer';
import {setTimeout,clearTimeout} from 'node:timers';
import {performance} from 'node:perf_hooks';
import {pathToFileURL} from 'node:url';
import vm from 'node:vm';

export async function checkSyntheticQueue(page) {
  const html=await readFile(page,'utf8');
  const source=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
  const bytes=html.match(/id="bytes"[^>]*>([^<]*)<\/script>/)[1];
  for(const outcome of ['success','failure','stop']) {
    let active=0,created=0,terminated=0,blobs=0;const messages=[],storage=new Map(),listeners={};
    const element=()=>({textContent:'',disabled:false,children:[],append(child){this.children.push(child);},replaceChildren(){this.children=[];}});
    const elements=Object.fromEntries(['run','stop','download','plan','status','rows','report','bytes','worker'].map(id=>['#'+id,element()]));elements['#bytes'].textContent=bytes;
    const document={hidden:false,querySelector:selector=>elements[selector],createElement:element,addEventListener:(name,fn)=>listeners[name]=fn};
    class Worker {
      constructor(){this.index=created++;active++;assert.equal(active,1,'Phases must not overlap');Promise.resolve().then(()=>this.onmessage({data:{type:'ready'}}));}
      postMessage(data){
        assert.equal(data.audit,true);assert.equal(data.safariTune,true);assert.equal(data.optimization,true);assert.ok(data.bytes.byteLength>1000);
        if(outcome==='stop'){Promise.resolve().then(()=>elements['#stop'].onclick());return;}
        const start=performance.timeOrigin+performance.now();document.hidden=this.index===0;listeners.visibilitychange();
        const result={experiment:'pplx-synthetic-full-v1',completed:!(outcome==='failure'&&this.index===1),error:outcome==='failure'&&this.index===1?'injected failure':undefined,runs:[{warmup:false,started_epoch_ms:start-1,finished_epoch_ms:start+10}],cases:[]};
        Promise.resolve().then(()=>this.onmessage({data:{type:'result',report:result}}));
      }
      terminate(){assert.ok(!this.terminated);this.terminated=true;active--;terminated++;}
    }
    const context=vm.createContext({document,window:{},navigator:{userAgent:'test'},location:{hostname:'127.0.0.1',pathname:'/test',search:'?report=/results/'+'a'.repeat(32)},URLSearchParams,URL:{createObjectURL:()=>{blobs++;return 'blob:test';},revokeObjectURL:()=>blobs--},Blob,Worker,performance,crypto:webcrypto,atob,setTimeout,clearTimeout,sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},fetch:async(url,options)=>{messages.push(JSON.parse(options.body));return {ok:true};}});
    new vm.Script(source).runInContext(context);await elements['#run'].onclick();
    const report=messages.at(-1);assert.equal(report.experiment,'pplx-synthetic-queue-v1');assert.equal(active,0);assert.equal(created,terminated);assert.equal(blobs,0);assert.equal(storage.size,0);
    if(outcome==='success'){assert.equal(report.completed,true);assert.equal(report.phases.length,report.expected_phases);assert.equal(report.phases[0].report.measured_runs_hidden,true);}
    else {assert.equal(report.completed,false);assert.match(report.error,outcome==='stop'?/Stopped by user/:/injected failure/);assert.equal(created,outcome==='stop'?1:2);}
  }
  return {sequential_workers:true,failure_stops_queue:true,stop_cleanup:true,visibility_preserved:true};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (!process.argv[2]) throw Error('Usage: node scripts/pplx_flash/check_synthetic_queue.mjs packaged.html');
  console.log(JSON.stringify(await checkSyntheticQueue(process.argv[2]), null, 2));
}
