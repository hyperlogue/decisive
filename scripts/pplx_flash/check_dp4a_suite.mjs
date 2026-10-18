import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

export async function checkSuite(page){
  const context=vm.createContext({self:{postMessage(){}},performance:{now:()=>0},isSecureContext:false,navigator:{}});
  const html=await readFile(page,'utf8');
  for(const match of html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)){
    const script=new vm.Script(match[2]);
    if(match[1].includes('id="suite-library"')||match[1].includes('id="probe"'))script.runInContext(context);
  }
  const suite=context.PplxDp4aSuite;
  const timing=t=>({median_ms:t,relative_mad:.02,samples_ms:Array(20).fill(t)});
  const variant=(id,t)=>({id,gpu:timing(t),wall:timing(t+1),kernel_check:{passed:true},raw_integer_check:{passed:true},configuration:{kind:id.startsWith('fp16')?'fp16':'int8',tile:64,transpose:false,pad:0,register:null}});
  const make=()=>({completed:true,capabilities:{webgpu:true,adapter:true,packed_dot:true,fp16:true},selection:{selected:{117:{id:'int8-tile64'},1024:{id:'int8-tile64'}}},confirmation:{completed:true,cases:[117,1024].map(rows=>({rows,variants:[variant('fp16-tile64',10),variant('int8-tile64',5)]}))}});
  assert.equal(suite.summarize(make()).code,'faster');
  const slow=make();for(const c of slow.confirmation.cases)c.variants[1].gpu=timing(15);
  assert.equal(suite.summarize(slow).code,'use-fp16');
  for(const c of slow.confirmation.cases)c.variants.push(variant('int8-not-selected',1));
  assert.equal(suite.summarize(slow).code,'use-fp16');
  const mixed=make();mixed.confirmation.cases[1].variants[1].gpu=timing(15);
  assert.equal(suite.summarize(mixed).code,'mixed');
  const close=make();for(const c of close.confirmation.cases)c.variants[1].gpu=timing(9.5);
  assert.equal(suite.summarize(close).code,'use-fp16');
  const noisy=make();noisy.confirmation.cases[0].variants[1].gpu.relative_mad=.2;
  assert.equal(suite.summarize(noisy).code,'noisy');
  const unstable=make();for(const c of unstable.confirmation.cases)c.variants[1].gpu.samples_ms=Array.from({length:20},(_,i)=>i%2?11:5);
  assert.equal(suite.summarize(unstable).code,'use-fp16');
  assert.equal(suite.summarize(make(),true).code,'interrupted');
  const bad=make();bad.confirmation.cases[1].variants[1].raw_integer_check.passed=false;
  assert.equal(suite.summarize(bad).code,'failed');
  const nan=make();nan.confirmation.cases[0].variants[1].gpu.samples_ms[0]=NaN;
  assert.equal(suite.summarize(nan).code,'failed');
  assert.equal(suite.summarize({completed:false}).code,'incomplete');
  assert.equal(suite.summarize({capabilities:{packed_dot:false}}).code,'unsupported');
  assert.equal(suite.summarize({capabilities:{fp16:false,packed_dot:true}}).code,'unavailable');
  const wall=make();for(const c of wall.confirmation.cases)for(const v of c.variants)v.gpu=null;
  assert.equal(suite.summarize(wall).code,'faster');
  assert.equal(suite.summarize(wall).comparisons[0].timer,'Synchronized elapsed time');
  const missing=make();missing.confirmation.cases.pop();assert.equal(suite.summarize(missing).code,'failed');
  const phase={id:'core',report:{cases:make().confirmation.cases}};
  const selection=suite.select([phase]);assert.equal(selection.selected[1024].id,'int8-tile64');
  assert.equal(selection.rounds%selection.specs.length,0);
  const unavailable=await suite.run({},{});assert.equal(unavailable.verdict.code,'unavailable');
  return {syntax:true,verdict_cases:16,selected_kernel_only:true,preflight:true,balanced_confirmation:true};
}
