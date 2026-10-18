import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const directory=resolve(process.argv[2]);
const api=await import(pathToFileURL(directory+'/decisive.js'));
await api.default({module_or_path:await readFile(directory+'/decisive_bg.wasm')});
const quantize=(values,columns,bits)=>[...api.pplxFlashQuantizeRows(new Float32Array(values),columns,bits)];
for(const bits of [4,8]){
  const limit=2**(bits-1)-1;
  const input=[limit,-limit,.5,-.5,1.5,-1.5,2.5,-2.5];
  const expected=[limit,-limit,0,0,2,-2,2,-2];
  assert.deepEqual(quantize(input,8,bits),expected);
  assert.deepEqual(quantize([...input,...input.map(x=>x/16)],8,bits),[...expected,...expected.map(x=>x/16)]);
  assert.deepEqual(quantize([0,-0,0,-0],4,bits),[0,0,0,0]);
}
for(const [values,columns,bits] of [[[1],0,8],[[1],2,4],[[1],1,3],[[Infinity],1,8],[[NaN],1,4]]){
  assert.throws(()=>quantize(values,columns,bits));
}
let count=0;
if(process.argv[3]){
  for(const row of JSON.parse(await readFile(process.argv[3])))for(const bits of [4,8]){
    const actual=quantize(row.values,row.values.length,bits),expected=row.expected[bits];
    assert.equal(actual.length,expected.length);
    for(let i=0;i<actual.length;i++)assert.equal(actual[i],expected[i],`${row.id} Q${bits} index ${i}`);
    count+=actual.length;
  }
}
console.log(JSON.stringify({passed:true,real_and_golden_values:count,ties_to_even:true,signed_zero:true,independent_row_scales:true,invalid_inputs_rejected:true}));
