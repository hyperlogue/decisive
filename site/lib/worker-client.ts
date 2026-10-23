interface Reply { id?:number; event?:string; result?:unknown; error?:string; detail?:string }
export class WorkerClient {
  worker: Worker;
  pending: Map<number,{resolve:(value:unknown)=>void;reject:(error:Error)=>void}>;
  sequence: number;
  queue: Promise<unknown>;
  closed: boolean;
  constructor(onProgress: (event: Reply) => void) {
    this.worker = new Worker(new URL('../workers/model-worker.js', import.meta.url), {type:'module'});
    this.pending = new Map();
    this.sequence = 0;
    this.queue = Promise.resolve();
    this.closed = false;
    this.worker.onmessage = ({data}: MessageEvent<Reply>) => {
      if (data.event) { onProgress(data); return; }
      const call = this.pending.get(data.id!);
      if (!call) return;
      this.pending.delete(data.id!);
      if (data.error) call.reject(Object.assign(new Error(data.error), {detail:data.detail}));
      else call.resolve(data.result);
    };
    this.worker.onerror = () => {
      this.closed = true;
      for (const call of this.pending.values()) call.reject(new Error('workerFailed'));
      this.pending.clear();
      onProgress({event:'fatal'});
    };
  }
  call(method: string, args: Record<string,unknown> = {}) {
    if (method === 'pause') return this.send(method, args);
    const result = this.queue.then(() => this.send(method, args));
    this.queue = result.catch(() => {});
    return result;
  }
  send(method: string, args: Record<string,unknown>) {
    if (this.closed) return Promise.reject(new Error('workerFailed'));
    const id = ++this.sequence;
    return new Promise<unknown>((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
      this.worker.postMessage({id, method, args});
    });
  }
  close() {
    this.closed = true;
    this.worker.terminate();
    for (const call of this.pending.values()) call.reject(new Error('cancelled'));
    this.pending.clear();
  }
}
