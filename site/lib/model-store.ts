export interface Hasher { update(bytes: Uint8Array): void; finish(): string; free?(): void }
interface SyncFile { getSize(): number; read(bytes: Uint8Array, options: {at:number}): number; write(bytes: Uint8Array, options: {at:number}): number; truncate(size:number): void; flush(): void; close(): void }
interface FileHandle { kind:'file'; getFile(): Promise<File>; createSyncAccessHandle(options?: {mode?: 'readwrite' | 'read-only'}): Promise<SyncFile> }
export interface StorageDirectory { kind:'directory'; getDirectoryHandle(name:string, options?:{create?:boolean}): Promise<StorageDirectory>; getFileHandle(name:string, options?:{create?:boolean}): Promise<FileHandle>; entries(): AsyncIterableIterator<[string,FileHandle|StorageDirectory]>; removeEntry(name:string,options:{recursive:boolean}): Promise<void> }
interface FileSpec { bytes:number; sha256:string }
interface Manifest { repository:string; revision:string; files:Record<string,FileSpec> }
interface Receipt { revision:string; files:Record<string,{sha256:string;modified:number}> }
type DownloadProgress = {phase:string;name:string;bytes:number;total:number};

export const STORE_NAME = 'decisive-privacy-pplx-v1';
export const LOCK_NAME = 'decisive-privacy-model-files';
const CHUNK = 4 * 1024 * 1024;

export function validateRange(response: Pick<Response, "status"|"headers">, offset: number, expectedSize: number) {
  if (response.status === 200) return 0;
  if (response.status !== 206) throw new Error('downloadFailed');
  const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get('content-range') || '');
  if (!match || Number(match[1]) !== offset || Number(match[3]) !== expectedSize ||
      Number(match[2]) < offset || Number(match[2]) >= expectedSize) throw new Error('invalidRange');
  return offset;
}

export class ModelStore {
  manifest: Manifest;
  root: StorageDirectory;
  createHasher: () => Hasher;
  fetcher: typeof fetch;
  storeName: string;
  entries: [string,FileSpec][];
  total: number;
  constructor({manifest, root, createHasher, storeName = STORE_NAME, fetcher = (...args: Parameters<typeof fetch>) => fetch(...args)}: {manifest:Manifest;root:StorageDirectory;createHasher:()=>Hasher;storeName?:string;fetcher?:typeof fetch}) {
    this.manifest=manifest;this.root=root;this.createHasher=createHasher;this.fetcher=fetcher;this.storeName=storeName;
    this.entries = Object.entries(manifest.files);
    this.total = this.entries.reduce((sum, [,f]) => sum + f.bytes, 0);
  }
  async directory(create = false): Promise<StorageDirectory | null> {
    try { return await this.root.getDirectoryHandle(this.storeName, {create}); }
    catch (e) { if ((e as Error)?.name === 'NotFoundError' && !create) return null; throw e; }
  }
  async receipt(dir: StorageDirectory): Promise<Receipt> {
    try {
      const data = JSON.parse(await (await (await dir.getFileHandle('verified.json')).getFile()).text());
      return data.revision === this.manifest.revision && data.files && typeof data.files === 'object' ?
        data : {revision:this.manifest.revision, files:{}};
    } catch { return {revision:this.manifest.revision, files:{}}; }
  }
  async inspect() {
    const dir = await this.directory(), receipt: Receipt = dir ? await this.receipt(dir) : {revision:this.manifest.revision,files:{}};
    const files = [];
    let stored = 0;
    if (dir) {
      for await (const [,handle] of dir.entries()) {
        if (handle.kind === 'file') stored += (await handle.getFile()).size;
      }
    }
    for (const [name, spec] of this.entries) {
      let file;
      try { file = await (await dir!.getFileHandle(encodeURIComponent(name))).getFile(); } catch {}
      const record = receipt.files[name];
      const verified = !!file && file.size === spec.bytes && record?.sha256 === spec.sha256 &&
        record.modified === file.lastModified;
      files.push({name, expected:spec.bytes, bytes:file?.size ?? 0, verified,
        state:verified ? 'verified' : file?.size ? (file.size === spec.bytes ? 'unverified' : 'partial') : 'missing'});
    }
    return {files, total:this.total, stored, verified:files.reduce((sum,f)=>sum+(f.verified?f.bytes:0),0),
      remaining:files.reduce((sum,f)=>sum+Math.max(0,f.expected-Math.min(f.bytes,f.expected)),0),
      ready:files.length>0&&files.every(f=>f.verified)};
  }
  async saveReceipt(dir: StorageDirectory, receipt: Receipt) {
    const handle = await (await dir.getFileHandle('verified.json', {create:true})).createSyncAccessHandle();
    try {
      const bytes = new TextEncoder().encode(JSON.stringify(receipt));
      if (handle.write(bytes, {at:0}) !== bytes.length) throw new Error('storageFailed');
      handle.truncate(bytes.length);
      handle.flush();
    } finally { handle.close(); }
  }
  async download({signal, progress = () => {}}: {signal?:AbortSignal;progress?:(event:DownloadProgress)=>void}) {
    const dir = (await this.directory(true))!, receipt = await this.receipt(dir);
    const aborted = () => { if (signal?.aborted) throw new DOMException('Paused', 'AbortError'); };
    for (const [name, spec] of this.entries) {
      aborted();
      const fileHandle = await dir.getFileHandle(encodeURIComponent(name), {create:true});
      if (typeof fileHandle.createSyncAccessHandle !== 'function') throw new Error('storageUnsupported');
      let file = await fileHandle.getFile();
      const record = receipt.files[name];
      if (file.size === spec.bytes && record?.sha256 === spec.sha256 && record.modified === file.lastModified) continue;
      delete receipt.files[name];
      const handle = await fileHandle.createSyncAccessHandle();
      let hasher: Hasher | null = this.createHasher();
      let reader: ReadableStreamDefaultReader<Uint8Array> | null = null, success = false, offset = 0;
      try {
        offset = handle.getSize();
        if (offset > spec.bytes) { handle.truncate(0); offset = 0; }
        const scratch = new Uint8Array(Math.min(CHUNK, offset));
        for (let at = 0; at < offset;) {
          aborted();
          const bytes = handle.read(scratch.subarray(0, Math.min(CHUNK, offset-at)), {at});
          if (!bytes) throw new Error('storageFailed');
          hasher.update(scratch.subarray(0,bytes)); at += bytes;
          progress({phase:'verifying', name, bytes:at, total:spec.bytes});
          await new Promise(resolve => setTimeout(resolve, 0));
        }
        if (offset < spec.bytes) {
          const url = `https://huggingface.co/${this.manifest.repository}/resolve/${this.manifest.revision}/${name}`;
          let response;
          try {
            response = await this.fetcher(url, {
              credentials:'omit', cache:'no-store', signal,
              headers:offset ? {Range:`bytes=${offset}-`} : {},
            });
          } catch (error) {
            if ((error as Error)?.name === 'AbortError') throw error;
            throw new Error('downloadFailed');
          }
          if (!response.ok || !response.body) throw new Error('downloadFailed');
          const start = validateRange(response, offset, spec.bytes);
          if (start !== offset) {
            hasher.free?.(); hasher = this.createHasher();
            handle.truncate(0); offset = 0;
          }
          reader = response.body.getReader();
          let flushed = offset, reported = 0;
          for (;;) {
            aborted();
            const {value, done} = await reader.read();
            if (done) break;
            if (offset + value.length > spec.bytes) throw new Error('invalidSize');
            if (handle.write(value, {at:offset}) !== value.length) throw new Error('storageFailed');
            hasher.update(value); offset += value.length;
            if (offset - flushed >= 64 * 1024 * 1024) { handle.flush(); flushed = offset; }
            if (performance.now() - reported > 120) {
              progress({phase:'downloading', name, bytes:offset, total:spec.bytes});
              reported = performance.now();
              await new Promise(resolve=>setTimeout(resolve,0));
            }
          }
        }
        aborted();
        if (offset !== spec.bytes) throw new Error('invalidSize');
        const hash = hasher.finish(); hasher = null;
        if (hash !== spec.sha256) { handle.truncate(0); throw new Error('integrityFailed'); }
        success = true;
      } finally {
        try { await reader?.cancel(); } catch {}
        hasher?.free?.();
        try { handle.flush(); } finally { handle.close(); }
      }
      if (success) {
        file = await fileHandle.getFile();
        receipt.files[name] = {sha256:spec.sha256, modified:file.lastModified};
        await this.saveReceipt(dir, receipt);
        progress({phase:'verified', name, bytes:spec.bytes, total:spec.bytes});
      }
    }
    return this.inspect();
  }
  async files() {
    if (!(await this.inspect()).ready) throw new Error('modelMissing');
    const dir = (await this.directory())!;
    return Object.fromEntries(await Promise.all(this.entries.map(async ([name])=>
      [name, await (await dir.getFileHandle(encodeURIComponent(name))).getFile()])));
  }
  // Sync handles let inference read OPFS directly into WASM memory, several times faster than File
  // reads. Chromium shares read-only handles; Safari ignores the mode and locks each file, which still
  // permits the getFile() calls storage inspection makes. If any handle fails, File objects are used.
  async readers(): Promise<{files: Record<string, File | SyncFile>; close(): void}> {
    const files = await this.files(), dir = (await this.directory())!, handles: SyncFile[] = [];
    const close = () => { for (const handle of handles.splice(0)) handle.close(); };
    try {
      const readers: Record<string, SyncFile> = {};
      for (const [name] of this.entries) {
        const handle = await dir.getFileHandle(encodeURIComponent(name));
        readers[name] = await handle.createSyncAccessHandle({mode:'read-only'});handles.push(readers[name]);
      }
      return {files:readers, close};
    } catch { close(); return {files, close() {}}; }
  }
  async remove() {
    try { await this.root.removeEntry(this.storeName, {recursive:true}); }
    catch(e) { if ((e as Error)?.name !== 'NotFoundError') throw e; }
    return this.inspect();
  }
}
