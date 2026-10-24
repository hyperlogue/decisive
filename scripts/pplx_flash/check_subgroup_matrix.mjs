import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {fileURLToPath, pathToFileURL} from 'node:url';
import vm from 'node:vm';

export async function checkSubgroupMatrix(root,page) {
  const source=await readFile(`${root}/examples/pplx-flash/subgroup-matrix.js`,'utf8');
  const html=await readFile(page,'utf8');
  new vm.Script(html.match(/<script>([\s\S]*)<\/script>/)[1]);
  let deviceRequested=false;
  const context=vm.createContext({navigator:{userAgent:'test',gpu:{requestAdapter:async()=>({features:new Set(['subgroups','shader-f16']),info:{subgroupMinSize:32,subgroupMaxSize:32},requestDevice(){deviceRequested=true;throw Error('Unexpected device request');}})}},PplxQuantizationProbe:{},PplxDp4aCore:{}});
  new vm.Script(source).runInContext(context);
  const result=await context.PplxSubgroupMatrix.run({},{});
  assert.equal(result.completed,true);
  assert.equal(result.supported,false);
  assert.equal(deviceRequested,false);
  assert.throws(()=>context.PplxSubgroupMatrix.shader('i8'));
  assert.throws(()=>context.PplxSubgroupMatrix.shader('f16',64));
  return {javascript_syntax:true,unsupported_feature_gate:true,unsupported_configuration_rejected:true};
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (!process.argv[2]) throw Error('Usage: node scripts/pplx_flash/check_subgroup_matrix.mjs packaged.html');
  const root = fileURLToPath(new URL('../..', import.meta.url));
  console.log(JSON.stringify(await checkSubgroupMatrix(root, process.argv[2]), null, 2));
}
