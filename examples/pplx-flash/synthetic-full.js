self.postMessage({type: 'ready'});
self.onmessage = async ({data}) => {
  const report = {schema_version: 1, experiment: 'pplx-synthetic-full-v1', completed: false,
    seed: data.seed, started_at: new Date().toISOString(), runs: [],
    scope: 'Full 32-layer text decoder, final RMS norm and 255-class readout at two markers; deterministic random weights; no trained-model accuracy claim',
    precision: {baseline: 'BF16 storage / FP32 compute', candidate: 'FP16 projection storage and 32-element FP16 FMA partials with FP32 outer sums; remaining weights BF16 and operations FP32'},
    methodology: {warmups: data.warmups, rounds: data.rounds, lengths: data.rows,
      order: 'Rotate the first optimization engine by measured round; two-engine comparisons alternate baseline-first and candidate-first.',
      weights: 'Independent stateless hash per tensor/element; round to BF16 first, then convert projection weights to FP16 for the candidate. Generated on GPU one layer at a time.',
      distribution: 'Uniform fan-in-scaled projections; residual output projections scaled by 0.125; near-identity norms and stable recurrent constants. Not representative of trained weight/activation distributions.',
      timed_execution: 'Sum of embedding setup, synchronized production layer calls and final CPU readout. Generation, compilation, audits, and progress callbacks are excluded.',
      wall_time: 'Contiguous run including synthetic generation, audits and progress callbacks; not real checkpoint inference latency.',
      excluded: 'Tokenizer and question protocol, checkpoint download, storage I/O, CPU-to-GPU weight uploads, vision; input uses selected rows of an implicit random embedding table',
      validation: 'CPU-reference edge-tile projection check, every layer hidden state finite, all logits finite; cross-precision drift and repeatability recorded without a model-quality pass threshold'},
  };
  const progress = message => self.postMessage({type: 'progress', message});
  const engines = [];
  const baselineId=data.safariTune?'tuning-baseline':data.cache?'input-reference':data.optimization?'fp16-baseline':'bf16-f32';
  report.baseline_id=baselineId;
  report.profile_enabled=Boolean(data.profile);
  report.layer_fingerprint_audit=Boolean(data.audit&&!data.cache);
  report.input_dependency_experiment=Boolean(data.cache);
  if(data.cache)report.methodology.input_dependency='Identical weights and positions; change only first or second half of synthetic input embeddings. Fingerprint each half after every layer to locate cross-input dependencies. Not a cache implementation or trained-model quality test.';
  if(data.audit)report.methodology.validation+='; all 32 layer fingerprints must match for candidates with unchanged precision';
  if(data.profile)report.methodology.profile='Per-dispatch GPU timestamps; instrumented diagnostic runs must not be compared with uninstrumented latency results.';
  if(data.optimization){report.precision={baseline:'FP16 projection storage / FP16 tile FMA / FP32 outer sums',candidate:'Identical projection and non-projection precision; kernel and memory configurations are recorded per engine'};report.methodology.validation+='; optimization candidates require bit-identical logits to the FP16 baseline';}
  if(data.safari){
    report.precision.candidate='FP16 workgroup padding or W8A8 unsigned packed projections with exact INT32 correction, per-row activation and per-output-channel weight scales; Fallback policy is recorded per engine: pureInt8 quantizes every decoder projection; other INT8 paths retain FP16 for small projections and <=8 rows. Hybrid retains fused FP16 MLP gate/up. int8FloatPartials uses exact FP32 partial arithmetic over the same INT8 operands with INT32 outer accumulation.';
    report.methodology.validation='FP16 padding requires bit-identical logits and optional layer fingerprints. INT8 requires CPU-checked quantization, every raw integer dot on edge/real-K tests, scaled/fused CPU agreement, finite full-model outputs and reported synthetic drift. No trained-model quality claim.';
    report.methodology.timed_execution+=' Activation max reduction, INT8 packing, row sums, correction and shared-input quantization reuse are included.';
    report.methodology.weights+=' INT8 weights and metadata are prepared on GPU outside execution; Original FP16 weights remain allocated even in pureInt8, so this experiment does not establish an INT8 model-memory reduction.';
  }
  if(data.cache)report.methodology.validation='Finite outputs and 64-bit hidden-state fingerprints; require unchanged prefix through the first three causal layers, changed prefix at the first full-attention layer, and state-dependent question states in the first layer.';
  const started = performance.now();
  const sum = (items, key) => items.reduce((s, x) => s + x[key], 0);
  const median = items => {const sorted = [...items].sort((a,b)=>a-b); return (sorted[(sorted.length-1)>>1]+sorted[sorted.length>>1])/2;};
  const compare = (a,b) => {
    const x=a.flat(),y=b.flat();
    if(x.length!==y.length || !x.every(Number.isFinite) || !y.every(Number.isFinite)) throw Error('Invalid output comparison');
    let max=0,delta=0,ref=0;
    for(let i=0;i<x.length;i++){const d=x[i]-y[i];max=Math.max(max,Math.abs(d));delta+=d*d;ref+=y[i]*y[i];}
    return {max_absolute_error:max,relative_l2:Math.sqrt(delta/Math.max(ref,1e-30))};
  };
  const argmax = row => row.indexOf(Math.max(...row));
  try {
    const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
    if(!adapter)throw Error('No WebGPU adapter');
    const info=adapter.info||{};
    report.browser_adapter={vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,is_fallback_adapter:info.isFallbackAdapter??adapter.isFallbackAdapter??null};
    report.software_gpu=report.browser_adapter.is_fallback_adapter===true||/swiftshader|llvmpipe|software/i.test(JSON.stringify(report.browser_adapter));
    report.features=[...adapter.features].sort();
    const init=performance.now();
    const wasm=await wasm_bindgen({module_or_path:data.bytes});
    report.wasm_init_ms=performance.now()-init;
    const config=(id,options={})=>({id,fp16:true,block:16,reuse:false,...options});
    const baseline=()=>config('fp16-baseline');
    const tile64={tileM:64,tileN:64};
    const combined={...tile64,block:128,reuse:true,mlpTile:64,attention:true};
    const tunedBase={...combined,pad:true,...(data.path==='fp16'?{}:{int8:true,fp16Mlp:data.path==='hybrid',pureInt8:data.path==='pure-int8'})};
    const tuningOptions={scan4:{scanUnroll:4},i8f32r8:{int8FloatPartials:8},combined:{scanUnroll:4},
      'combined-f32r8':{scanUnroll:4,int8FloatPartials:8}};
    if(data.safariTune&&!tuningOptions[data.candidate])throw Error('Unknown tuning candidate');
    if(data.safariTune&&(data.candidate.includes('f32')&&data.path==='fp16'))throw Error('Tuning candidate does not apply to this precision path');
    const experiments={
      safariTune:[config('tuning-baseline',tunedBase),config(data.candidate,{...tunedBase,...tuningOptions[data.candidate]})],
      safari:[config('fp16-baseline',{...combined,pad:data.comparison!=='padding'}),
        data.comparison==='padding'?config('fp16-padded',{...combined,pad:true}):
        config(data.comparison==='pure'?'pure-int8':data.comparison==='hybrid'?'int8-hybrid':data.comparison==='fusion'?'int8-fused':'int8-unfused',{...combined,pad:true,int8:true,int8Fused:data.comparison==='fusion',fp16Mlp:data.comparison==='hybrid',pureInt8:data.comparison==='pure'})],
      cache:[config('input-reference'),config('changed-state'),config('changed-questions')],
      profile:[baseline()],
      smallrows:[baseline(),config('fp16-fused-attention-reuse',combined),
        config('fp16-small4',{...combined,small:4}),config('fp16-small8',{...combined,small:8})],
      selected:[baseline(),config('fp16-selected',combined)],
      norm:[baseline(),config('fp16-fused-attention-reuse',combined),
        config('fp16-cooperative-norm',{...combined,norm:true}),config('fp16-norm-batched',{...combined,norm:true,batch:true})],
      operands:[baseline(),config('fp16-fused-attention-reuse',combined),
        config('fp16-packed-inputs',{...combined,packed:true}),config('fp16-batched-passes',{...combined,batch:true})],
      combo:[baseline(),config('fp16-attention128',{...tile64,block:128,attention:true}),
        config('fp16-fused-attention',{...combined,reuse:false}),config('fp16-fused-attention-reuse',combined)],
      subtile:[baseline(),config('fp16-tile64x64',tile64),
        config('fp16-load8',{...tile64,subtile:8}),config('fp16-load16',{...tile64,subtile:16})],
      k64:[baseline(),config('fp16-tile64x64',tile64),
        config('fp16-k64-tile32',{tileM:32,tileN:32,k64:32}),config('fp16-k64-tile64',{...tile64,k64:64})],
      attention:[baseline(),config('fp16-tile64x64',tile64),
        config('fp16-attention',{...tile64,attention:true}),config('fp16-attention128',{...tile64,block:128,attention:true})],
      fusion:[baseline(),config('fp16-tile64x64',tile64),
        config('fp16-fused32',{...tile64,mlpTile:32}),config('fp16-fused64',{...tile64,mlpTile:64})],
      unroll:[baseline(),config('fp16-tile64x64',tile64),
        config('fp16-unroll32',{tileM:32,tileN:32,unroll:true}),config('fp16-unroll64',{...tile64,unroll:true})],
      tiles:[baseline(),config('fp16-contiguous',{tileM:32,tileN:32}),
        config('fp16-tile64x32',{tileM:64,tileN:32}),config('fp16-tile64x64',tile64)],
      optimization:[baseline(),config('fp16-batched',{block:128}),
        config('fp16-reuse',{reuse:true}),config('fp16-combined',{block:128,reuse:true})],
      precision:[config('bf16-f32',{fp16:false}),config('fp16-f16')],
    };
    const mode=Object.keys(experiments).find(key=>data[key])||'precision';
    report.configuration_mode=mode;
    if(data.safari)report.comparison=data.comparison||'int8';
    if(data.safariTune){report.path=data.path;report.candidate=data.candidate;report.methodology.validation+='; tuning requires bit-identical logits and audited hidden states against the same-precision baseline';}
    const configurations=experiments[mode];
    for(const configuration of configurations) {
      const {id,fp16,block,reuse,tileM,tileN,unroll,mlpTile,attention,k64,subtile,packed,batch,norm,small,pad,int8,int8Fused,fp16Mlp,pureInt8,int8FloatPartials,scanUnroll}=configuration;
      progress(`Initializing ${id}…`);
      const engine=data.profile?await wasm_bindgen.PplxFlashSyntheticGpu.createProfiled(data.seed):await wasm_bindgen.PplxFlashSyntheticGpu.create(fp16,data.seed);
      const entry={id,engine,configuration};engines.push(entry);
      if(data.optimization)engine.configure(block,reuse);
      if(tileM)await engine.configureGemm(tileM,tileN,Boolean(unroll));
      if(k64)await engine.configureGemmK64(k64);
      if(subtile)await engine.configureGemmSubtile(tileM,subtile);
      if(mlpTile)await engine.configureMlp(mlpTile);
      if(pad)await engine.configurePadding();
      if(int8)await engine.configureInt8(Boolean(int8Fused),Boolean(fp16Mlp));
      if(pureInt8)engine.configurePureInt8();
      if(int8FloatPartials)await engine.configureInt8FloatPartials(int8FloatPartials);
      if(scanUnroll)await engine.configureDeltaScan(scanUnroll);
      if(attention)await engine.configureAttention();
      if(norm)await engine.configureNorm();
      if(small)await engine.configureSmallRows(small);
      if(packed)await engine.configurePackedInputs();
      if(batch)engine.configureBatchedPasses();
      Object.assign(entry,{adapter:JSON.parse(engine.adapter()),projection_check:JSON.parse(await engine.checkProjection()),mlp_check:JSON.parse(await engine.checkMlp()),attention_check:JSON.parse(await engine.checkAttention())});
      if(int8)entry.int8_check=JSON.parse(await engine.checkInt8());
    }
    report.engines=engines.map(({engine,...entry})=>entry);
    for(const rows of data.rows) {
      for(let round=-data.warmups;round<data.rounds;round++) {
        const offset=data.optimization?((round%engines.length)+engines.length)%engines.length:0;
        const order=[...engines.slice(offset),...engines.slice(0,offset)];
        if(!data.optimization&&round%2!==0)order.reverse();
        for(const {id,engine,configuration} of order) {
          const prefix=`${rows} tokens · ${id} · ${round<0?'warmup':`run ${round+1}/${data.rounds}`}`;
          progress(`${prefix}: starting`);
          const start=performance.now();
          if(data.cache)engine.configureInputChange(id==='changed-questions'?Math.floor(rows/2):0,id==='changed-questions'?rows:Math.floor(rows/2),id==='input-reference'?0:1729);
          if(data.audit&&!data.cache)engine.configureInputChange(0,1,0);
          const prep=JSON.parse(engine.prepare(rows));
          const run={id,rows,round,warmup:round<0,started_epoch_ms:performance.timeOrigin+start,embedding_ms:prep.embedding_ms,layers:[]};
          for(let layer=0;layer<32;layer++) {
            const weights=JSON.parse(await engine.prepareLayer());
            const result=JSON.parse(await engine.runLayer());
            if(result.layer!==layer || !result.finite) throw Error('Invalid decoder layer result');
            if(configuration.pureInt8&&result.profile?.dispatches.some(d=>['gemm','mlp_fused','gemm_small','mlp_small'].includes(d.kernel)))throw Error('Pure INT8 dispatched an FP16 projection');
            run.layers.push({...weights,...result});
            if((layer+1)%8===0) progress(`${prefix}: layer ${layer+1}/32 complete`);
          }
          run.output=JSON.parse(await engine.finish());
          run.wall_ms=performance.now()-start;
          run.finished_epoch_ms=performance.timeOrigin+performance.now();
          run.generation_ms=sum(run.layers,'generation_ms');
          run.audit_ms=sum(run.layers,'audit_ms');
          run.decoder_ms=sum(run.layers,'compute_ms');
          run.execution_ms=run.embedding_ms+run.decoder_ms+run.output.readout_ms;
          run.wasm_memory_bytes=wasm.memory.buffer.byteLength;
          if(run.output.completed_layers!==32 || !run.output.finite) throw Error('Incomplete full decoder');
          report.runs.push(run);
          progress(`${prefix}: execution ${(run.execution_ms/1000).toFixed(2)} s, generation ${(run.generation_ms/1000).toFixed(2)} s`);
        }
      }
    }
    report.cases=data.rows.map(rows=>{
      const runs=report.runs.filter(r=>r.rows===rows&&!r.warmup);
      const variants=Object.fromEntries(engines.map(({id})=>{
        const samples=runs.filter(r=>r.id===id), times=samples.map(r=>r.execution_ms), center=median(times);
        return [id,{samples:times,median_execution_ms:center,relative_mad:median(times.map(t=>Math.abs(t-center)))/center,
          median_decoder_ms:median(samples.map(r=>r.decoder_ms)),median_generation_ms:median(samples.map(r=>r.generation_ms)),
          median_wall_ms:median(samples.map(r=>r.wall_ms)),
          max_hidden_abs:Math.max(...samples.flatMap(r=>r.layers.map(l=>l.hidden_max_abs))),
          repeatability: samples.map(r=>compare(r.output.logits,samples[0].output.logits))}];
      }));
      const paired=Array.from({length:data.rounds},(_,round)=>engines.filter(e=>e.id!==baselineId).map(({id})=>{
        const baseline=runs.find(r=>r.round===round&&r.id===baselineId);
        const candidate=runs.find(r=>r.round===round&&r.id===id);
        const difference=compare(candidate.output.logits,baseline.output.logits);
        const candidateConfiguration=engines.find(e=>e.id===id).configuration;
        const samePrecision=data.safariTune||!candidateConfiguration.int8||(rows<=8&&!candidateConfiguration.pureInt8);
        if(data.optimization&&!data.cache&&samePrecision&&difference.max_absolute_error!==0)throw Error(`${id}: graph optimization changed logits by ${difference.max_absolute_error}`);
        if(data.audit&&!data.cache&&samePrecision&&candidate.layers.some((l,i)=>l.dependency.prefix_hash!==baseline.layers[i].dependency.prefix_hash||l.dependency.suffix_hash!==baseline.layers[i].dependency.suffix_hash))throw Error(`${id}: layer fingerprint mismatch`);
        return {round,id,speedup:data.cache?null:baseline.execution_ms/candidate.execution_ms,logits:difference,
          probabilities:compare(candidate.output.probabilities,baseline.output.probabilities),
          argmax_changes:candidate.output.logits.filter((row,i)=>argmax(row)!==argmax(baseline.output.logits[i])).length};
      })).flat();
      const speedups=Object.fromEntries(engines.filter(e=>e.id!==baselineId).map(({id})=>[id,variants[baselineId].median_execution_ms/variants[id].median_execution_ms]));
      let dependency;
      if(data.cache){
        const ref=runs.find(r=>r.id==='input-reference'),state=runs.find(r=>r.id==='changed-state'),questions=runs.find(r=>r.id==='changed-questions');
        dependency=ref.layers.map((l,i)=>({layer:i,prefix_unchanged_when_questions_change:l.dependency.prefix_hash===questions.layers[i].dependency.prefix_hash,questions_unchanged_when_state_changes:l.dependency.suffix_hash===state.layers[i].dependency.suffix_hash}));
        if(!dependency.slice(0,3).every(l=>l.prefix_unchanged_when_questions_change)||dependency[3].prefix_unchanged_when_questions_change||dependency[0].questions_unchanged_when_state_changes)throw Error('Unexpected causal/full-attention dependency boundary');
      }
      return {rows,variants,paired,speedups:data.cache?{}:speedups,speedup:data.cache?null:Object.values(speedups)[0],dependency};
    });
    report.completed=true;
  } catch(error) {report.error=String(error);}
  finally {for(const {engine} of engines) engine.free();}
  report.elapsed_ms=performance.now()-started;
  self.postMessage({type:'result',report});
};
