import {loadModel} from '../src/index.js';
import type {DecimaParameters, PplxParameters, Predictor, Selection} from '../src/types.js';

type Assert<T extends true> = T;
type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false;
const decima = loadModel({model:'decima',backend:'webgpu',source:{files:{}}});
const pplx = loadModel({model:'pplx-flash',backend:'webgpu',source:{files:{}},parameters:{maxSequenceTokens:1024}});
type Options<T extends Promise<{predict: (...args: never[]) => unknown}>> = NonNullable<Parameters<Awaited<T>['predict']>[1]>;
type DecimaCheck = Assert<Equal<NonNullable<Options<typeof decima>['parameters']>, DecimaParameters>>;
type PplxCheck = Assert<Equal<NonNullable<Options<typeof pplx>['parameters']>, PplxParameters>>;
type NoCrossModelOptions = Assert<Equal<'long' extends keyof PplxParameters ? true : false, false>>;
async function configured(selection: Selection): Promise<Predictor> {
  const model = await loadModel(selection);
  const cache = await model.createCache();
  await model.predict({state:'data',questions:{q:{type:'noul'}}},{cache});
  return model;
}
void configured;
