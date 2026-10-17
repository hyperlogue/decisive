const q=new URLSearchParams(location.search);
try{
 const pkg=q.get('package')||'/artifacts/web/pkg-decima-threads';
 if(!pkg.startsWith('/artifacts/web/')||pkg.includes('..'))throw Error('Package must be local');
 const api=await import(pkg+'/decisive.js'),wasm=await api.default(),records=[];
 for(const count of [0,65]){
  let rejected=false;
  try{await api.initDecimaThreadPool(count);}catch(e){rejected=String(e).includes('1..=64');}
  if(!rejected)throw Error('Invalid count did not reject');records.push({case:'reject-count-'+count});
 }
 await api.initDecimaThreadPool(2);records.push({case:'valid-after-invalid'});
 await api.initDecimaThreadPool(2);records.push({case:'same-count-idempotent'});
 let rejected=false;
 try{await api.initDecimaThreadPool(3);}catch(e){rejected=String(e).includes('cannot change');}
 if(!rejected)throw Error('Changing live pool size did not reject');records.push({case:'changed-count-rejected'});
 await api.initDecimaThreadPool(2);records.push({case:'original-count-after-rejection'});
 if(!(wasm.memory.buffer instanceof SharedArrayBuffer))throw Error('Expected shared memory');
 postMessage({records,shared_memory:true});
}catch(e){postMessage({error:String(e),stack:e.stack});}
