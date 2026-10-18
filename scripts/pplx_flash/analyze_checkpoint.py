import argparse
import json
import math
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parents[2]


def precision(mode):
    mode=mode.removesuffix('-reuse')
    if mode=='bf16':return 'bf16'
    if mode.startswith('fp16'):return 'fp16'
    if mode.startswith('hybrid'):return 'hybrid'
    if mode in ['int8','int8-tuned','w8a8-f32']:return 'w8a8'
    if mode in ['q4-fused','q4-buffered']:return 'w4a8'
    raise ValueError(f'Unknown configuration: {mode}')


def analyze(path):
    report=json.loads(path.read_text())
    assert report['experiment']=='pplx-checkpoint-phase-v1' and report['completed'],report.get('error')
    spec=report['configuration'];modes=spec['modes']
    assert 1<=len(modes)<=2 and len(set(modes))==len(modes)
    assert len(report['runs'])==len(modes)*len(spec['fixtures'])*(spec['warmups']+spec['rounds'])
    resident=report.get('timing_scope')=='warm-request'
    if resident:
        assert spec.get('resident',True) is True
        assert {s['mode'] for s in report['startup']}==set(modes)
        for startup in report['startup']:
            assert startup['weight_residency']=='all-decoder-layers'
            assert startup['resident_decoder_weight_bytes']>0 and startup['elapsed_ms']>0
            assert startup['bytes_read']>=7_140_099_072
    def raw(run):
        return [v for chunk in run['diagnostics']['logits'] for row in chunk['values'] for v in row]
    for run in report['runs']:
        assert run['mode'] in modes and run['fixture'] in spec['fixtures']
        assert run['warmup']==(run['round']<0)
        assert len(run['layers'])==32*run['response']['usage']['sequences']
        if resident:
            assert run['configuration']['resident'] is True
            assert run['timing_scope']=='warm-request' and run['request_ms']==run['elapsed_ms']
            assert run['decoder_weight_bytes_read']==0
            assert run['adapter']['weight_residency']=='all-decoder-layers'
            assert all(l['pack_ms']==0 and l['read_ms']==0 for l in run['layers'])
        else:
            assert run['bytes_read']>=7_140_099_072*run['response']['usage']['sequences']
        assert run['elapsed_ms']>0 and math.isfinite(run['elapsed_ms'])
        assert raw(run) and all(math.isfinite(v) for v in raw(run))
        for chunk in run['diagnostics']['logits']:
            assert len(chunk['keys'])==len(chunk['values'])
            assert all(1<=len(row)<=255 for row in chunk['values'])
        assert sum(l['compute_ms']+l['pack_ms'] for l in run['layers'])<=run['elapsed_ms']+2
        if precision(run['mode']) in ['w8a8','w4a8']:
            assert not any(d['kernel'] in ['gemm','mlp_fused','gemm_small','mlp_small']
                           for layer in run['layers'] for d in (layer.get('profile') or {}).get('dispatches',[])), 'FP16 projection fallback'
    result={'source':str(path.relative_to(ROOT)),'verified_runs':len(report['runs']),
            'checkpoint_revision':report['metadata']['revision'],'synthetic_weights':False,
            'timing_scope':'warm-request' if resident else 'streamed-request-diagnostic',
            'startup':report.get('startup',[]),
            'hidden_timings_provisional':any(r.get('page_hidden') for r in report['runs'] if not r['warmup']),
            'confirmation':resident and spec['rounds']>=3 and spec['warmups']>=1 and not spec.get('profile'),'cases':[]}
    for fixture in spec['fixtures']:
        runs=[r for r in report['runs'] if r['fixture']==fixture and not r['warmup']]
        baseline={r['round']:r for r in runs if r['mode']==modes[0]}
        candidate={r['round']:r for r in runs if r['mode']==modes[-1]}
        assert set(baseline)==set(candidate)==set(range(spec['rounds']))
        variants={}
        for mode in modes:
            samples=[r for r in runs if r['mode']==mode];times=[r['elapsed_ms'] for r in samples];center=median(times)
            variants[mode]={'median_ms':center,'samples_ms':times,'relative_mad':median(abs(t-center) for t in times)/center,
                'input_tokens':samples[0]['response']['usage']['input_tokens'],
                'median_compute_ms':median(sum(l['compute_ms'] for l in r['layers']) for r in samples),
                'median_read_and_upload_ms':median(sum(l['read_ms'] for l in r['layers']) for r in samples),
                'median_weight_preparation_ms':median(sum(l['pack_ms'] for l in r['layers']) for r in samples)}
        if len(modes)==1:
            result['cases'].append({'fixture':fixture,'variants':variants,'clears_complexity_threshold':False})
            continue
        differences=[]
        for round in baseline:
            a,b=raw(candidate[round]),raw(baseline[round]);assert len(a)==len(b)
            maximum=max(abs(x-y) for x,y in zip(a,b))
            if precision(modes[0])==precision(modes[1]):assert maximum==0,'Same-precision logits changed'
            differences.append({'max_absolute_error':maximum,'relative_l2':math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/max(sum(y*y for y in b),1e-30))})
        ratio=variants[modes[0]]['median_ms']/variants[modes[1]]['median_ms']
        wins=sum(candidate[i]['elapsed_ms']<baseline[i]['elapsed_ms'] for i in baseline)
        result['cases'].append({'fixture':fixture,'variants':variants,'speedup':ratio,'latency_reduction':1-1/ratio,
            'paired_wins':wins,'rounds':spec['rounds'],'logit_differences':differences,
            'clears_complexity_threshold':result['confirmation'] and not result['hidden_timings_provisional'] and
                1-1/ratio>=.10 and wins/spec['rounds']>=.8 and all(v['relative_mad']<=.1 for v in variants.values())})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    results=[analyze(p.resolve()) for p in args.reports]
    with args.output.open('x') as output:json.dump(results,output,indent=2);output.write('\n')
    print(json.dumps({'verified_reports':len(results),'verified_runs':sum(r['verified_runs'] for r in results)}))
