export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export interface Request { state: string | Json[] | Record<string, Json>; questions: Record<string, Json> }
export interface DecimaParameters { long?: boolean; window?: number; overlap?: number; lang?: string; minConfidence?: number }
export interface LayaParameters { long?: boolean; window?: number; stride?: number; lang?: string; minConfidence?: number }
export interface PplxParameters { maxSequenceTokens?: number }
export type EmptyParameters = Record<string, never>;
export interface Registry {
  decima: { wasm: DecimaParameters; webgpu: DecimaParameters };
  laya: { wasm: LayaParameters; webgpu: LayaParameters };
  'pplx-flash': { webgpu: PplxParameters; wasm: PplxParameters };
  kev: { wasm: EmptyParameters };
  'jev-style': { wasm: EmptyParameters };
}
export type Family = keyof Registry;
export type Backend<M extends Family> = keyof Registry[M] & string;
export interface LoadOptions<M extends Family, B extends Backend<M>> {
  model: M;
  backend: B;
  source: { files: Record<string, Blob | Uint8Array> };
  format?: string;
  parameters?: Registry[NoInfer<M>][NoInfer<B>];
  tuning?: { mode?: 'auto' | 'off'; budgetMs?: number };
}
export type Selection = { [M in Family]: { [B in Backend<M>]: LoadOptions<M, B> }[Backend<M>] }[Family];
export interface CacheOptions { maxBytes?: number; requirePrefix?: boolean }
export interface Progress { phase: string; completed: number; total: number }
export interface Controls { signal?: AbortSignal; onProgress?: (event: Progress) => void }
export interface InferenceResult {
  response: { answers: Record<string, Json>; usage: Record<string, Json>; [key: string]: Json };
  model: string;
  backend: string;
  elapsed_ms: number;
}
export interface Failure { code: string; message: string }
export type BatchResult = ({ status: 'success'; result: InferenceResult } | { status: 'error'; error: Failure })[];
export interface ModelInfo { model: string; backend: string; caches: string[]; tuning: Json; parameters: Json }
export interface OpaqueCache { clear(): Promise<void>; dispose(): Promise<void> }
export interface PredictOptions<P> extends Controls { parameters?: P; cache?: OpaqueCache }
export interface Predictor {
  readonly info: ModelInfo;
  createCache(options?: CacheOptions): Promise<OpaqueCache>;
  predict(request: Request, options?: PredictOptions<never>): Promise<InferenceResult>;
  predictBatch(requests: readonly Request[], options?: PredictOptions<never>): Promise<BatchResult>;
  dispose(): Promise<void>;
}
export interface Model<P> extends Omit<Predictor, 'predict' | 'predictBatch'> {
  predict(request: Request, options?: PredictOptions<P>): Promise<InferenceResult>;
  predictBatch(requests: readonly Request[], options?: PredictOptions<P>): Promise<BatchResult>;
}
export class InferenceError extends Error {
  constructor(readonly code: string, message: string) { super(message); this.name = 'InferenceError'; }
}
export function failure(error: unknown): Failure {
  if (error && typeof error === 'object') {
    const value = error as { code?: unknown; message?: unknown; name?: unknown };
    return { code: typeof value.code === 'string' ? value.code : value.name === 'AbortError' ? 'CANCELLED' : 'INFERENCE_FAILED', message: typeof value.message === 'string' ? value.message : String(error) };
  }
  return { code: 'INFERENCE_FAILED', message: String(error) };
}
