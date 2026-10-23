import type { Failure, Progress } from './types.js';
export interface Command { id: number; method: 'load' | 'createCache' | 'clearCache' | 'disposeCache' | 'predict' | 'predictBatch' | 'dispose' | 'cancel'; args?: unknown }
export type Reply = { id: number; result: unknown } | { id: number; error: Failure } | { id: number; progress: Progress };
