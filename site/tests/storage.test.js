import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {ModelStore, STORE_NAME, validateRange} from '../lib/model-store.js';
const hash = data => createHash('sha256').update(data).digest('hex');
const hasher = () => {
  const h=createHash('sha256');
  return {update:bytes=>h.update(bytes),finish:()=>h.digest('hex'),free:()=>{}};
};
const notFound=()=>new DOMException('Missing','NotFoundError');
class FileHandle {
  kind='file'; bytes=new Uint8Array(); modified=1; locked=false;
  constructor(name){this.name=name;}
  async getFile(){return new File([this.bytes],this.name,{lastModified:this.modified});}
  async createSyncAccessHandle(){
    if(this.locked)throw new DOMException('Locked','NoModificationAllowedError');
    this.locked=true;
    return {
      getSize:()=>this.bytes.length,
      read:(target,{at})=>{const data=this.bytes.subarray(at,at+target.length);target.set(data);return data.length;},
      write:(source,{at})=>{const data=new Uint8Array(Math.max(this.bytes.length,at+source.length));data.set(this.bytes);data.set(source,at);this.bytes=data;this.modified++;return source.length;},
      truncate:n=>{this.bytes=this.bytes.slice(0,n);this.modified++;},
      flush:()=>{},close:()=>{this.locked=false;},
    };
  }
}
class Directory {
  kind='directory'; children=new Map();
  async getDirectoryHandle(name,{create=false}={}){if(!this.children.has(name)){if(!create)throw notFound();this.children.set(name,new Directory());}return this.children.get(name);}
  async getFileHandle(name,{create=false}={}){if(!this.children.has(name)){if(!create)throw notFound();this.children.set(name,new FileHandle(name));}return this.children.get(name);}
  async removeEntry(name){if(!this.children.delete(name))throw notFound();}
  async *entries(){yield* this.children.entries();}
}
function fixture(fetcher){
  const body=new TextEncoder().encode('real bytes for integrity checks');
  const manifest={repository:'example/model',revision:'pinned',files:{'weights.bin':{bytes:body.length,sha256:hash(body)}}};
  const root=new Directory();
  const store=new ModelStore({manifest,root,createHasher:hasher,fetcher:fetcher??(async()=>new Response(body))});
  return {body,root,store};
}
test('streamed download verifies files; inspect counts actual bytes; deletion is scoped',async()=>{
  const {store,root,body}=fixture();
  await root.getDirectoryHandle('unrelated',{create:true});
  assert.equal((await store.inspect()).stored,0);
  const state=await store.download({});
  assert.equal(state.ready,true);assert.equal(state.verified,body.length);assert.ok(state.stored>body.length);
  assert.equal((await store.files())['weights.bin'].size,body.length);
  await store.remove();assert.equal((await store.inspect()).stored,0);
  assert.ok(root.children.has('unrelated'));assert.ok(!root.children.has(STORE_NAME));
});
test('partial downloads resume at the exact offset and hash the existing prefix',async()=>{
  const {store,body}=fixture();const dir=await store.directory(true),handle=await dir.getFileHandle('weights.bin',{create:true});
  handle.bytes=body.slice(0,8);
  store.fetcher=async(url,options)=>{
    assert.equal(options.credentials,'omit');assert.equal(options.headers.Range,'bytes=8-');
    return new Response(body.slice(8),{status:206,headers:{'Content-Range':`bytes 8-${body.length-1}/${body.length}`}});
  };
  assert.equal((await store.download({})).ready,true);
  assert.deepEqual(handle.bytes,body);
});
test('a server ignoring Range safely replaces the partial prefix',async()=>{
  const {store,body}=fixture();const dir=await store.directory(true),handle=await dir.getFileHandle('weights.bin',{create:true});
  handle.bytes=body.slice(0,5);
  assert.equal((await store.download({})).ready,true);assert.deepEqual(handle.bytes,body);
});
test('bad ranges and corrupt complete files cannot become ready',async()=>{
  assert.throws(()=>validateRange(new Response(null,{status:206,headers:{'content-range':'bytes 1-9/20'}}),2,20),/invalidRange/);
  const {store,body}=fixture();store.fetcher=async()=>new Response(new Uint8Array(body.length).fill(1));
  await assert.rejects(store.download({}),/integrityFailed/);
  const status=await store.inspect();assert.equal(status.ready,false);assert.equal(status.files[0].bytes,0);
});
test('pause keeps a partial file and releases its write handle',async()=>{
  const controller=new AbortController(),{store,body}=fixture();
  store.fetcher=async()=>new Response(new ReadableStream({
    start(c){c.enqueue(body.slice(0,5));},
    pull(){controller.abort();},
  }));
  await assert.rejects(store.download({signal:controller.signal}),{name:'AbortError'});
  const file=await (await store.directory()).getFileHandle('weights.bin');
  assert.equal(file.locked,false);assert.equal(file.bytes.length,5);
});
test('changed files invalidate receipts and already complete files reverify without network',async()=>{
  const {store,body}=fixture();await store.download({});
  const handle=await (await store.directory()).getFileHandle('weights.bin');handle.modified++;
  assert.equal((await store.inspect()).ready,false);
  store.fetcher=async()=>assert.fail('must not download an already complete file');
  assert.equal((await store.download({})).ready,true);
  assert.deepEqual(handle.bytes,body);
});
test('nested Hugging Face paths are stored safely and models delete independently',async()=>{
  const root=new Directory(),body=new TextEncoder().encode('nested weights');
  const manifest={repository:'example/model',revision:'pinned',files:{'pytorch/encoder/model.safetensors':{bytes:body.length,sha256:hash(body)}}};
  const make=storeName=>new ModelStore({manifest,root,storeName,createHasher:hasher,fetcher:async()=>new Response(body)});
  const decima=make('decima'),q4=make('q4');
  await decima.download({});await q4.download({});
  assert.ok((await decima.directory()).children.has('pytorch%2Fencoder%2Fmodel.safetensors'));
  assert.equal((await decima.files())['pytorch/encoder/model.safetensors'].size,body.length);
  await decima.remove();
  assert.equal((await decima.inspect()).stored,0);
  assert.equal((await q4.inspect()).ready,true);
});
