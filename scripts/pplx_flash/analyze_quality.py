import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean

MODES=['fp16-scan','int8-tuned','q4-fused']


def distribution(answer):
    if answer['type']=='noul':
        p=answer['noul'];values={'false':1-p,'true':p}
    else:values=answer['probabilities']
    assert values and all(isinstance(p,(int,float)) and math.isfinite(p) and 0<=p<=1 for p in values.values())
    assert abs(sum(values.values())-1)<1e-6
    return values


def prediction(answer):
    values=distribution(answer)
    if answer['type']=='noul':return 'true' if values['true']>=.5 else 'false'
    label=max(values,key=values.get)
    if answer['type']=='choice':assert answer['choice']==label
    if answer['type']=='score':assert abs(answer['score']-sum(int(k)*v for k,v in values.items()))<1e-6
    return label


def primary(case):
    return {'emotion':'emotion','sentiment':'sentiment','banking':'intent','controlled-tone':'tone','stress':'refund'}.get(case['family'],
        'team' if 'team' in case['request']['questions'] else 'department')


def metric(rows,mode):
    scored=[(r,r['answers'][mode]) for r in rows if r['gold'] is not None]
    homogeneous=len({(r.get('family'),r.get('question')) for r,_ in scored})<=1
    labels=sorted({r['gold'] for r,_ in scored}|{prediction(a) for _,a in scored})
    correct=sum(prediction(a)==r['gold'] for r,a in scored)
    f1=[]
    for label in labels:
        tp=sum(r['gold']==label and prediction(a)==label for r,a in scored)
        fp=sum(r['gold']!=label and prediction(a)==label for r,a in scored)
        fn=sum(r['gold']==label and prediction(a)!=label for r,a in scored)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    ordinal=[a['score']-r['ordinal'] for r,a in scored if r['ordinal'] is not None]
    confusion={label:dict(collections.Counter(prediction(a) for r,a in scored if r['gold']==label)) for label in labels}
    return {'n':len(scored),'correct':correct,'accuracy':correct/len(scored) if scored else None,
            'macro_f1':mean(f1) if f1 and homogeneous else None,
            'nll':mean(-math.log(max(distribution(a)[r['gold']],1e-15)) for r,a in scored) if scored else None,
            'brier':mean(sum((p-(label==r['gold']))**2 for label,p in distribution(a).items()) for r,a in scored) if scored else None,
            'ordinal_mae':mean(abs(v) for v in ordinal) if ordinal else None,
            'ordinal_bias':mean(ordinal) if ordinal else None,'confusion':confusion if homogeneous else None}


def paired(rows,candidate,reference='fp16-scan',bootstrap=False):
    labeled=[r for r in rows if r['gold'] is not None]
    regressions=sum(prediction(r['answers'][reference])==r['gold'] and prediction(r['answers'][candidate])!=r['gold'] for r in labeled)
    fixes=sum(prediction(r['answers'][reference])!=r['gold'] and prediction(r['answers'][candidate])==r['gold'] for r in labeled)
    flips=sum(prediction(r['answers'][reference])!=prediction(r['answers'][candidate]) for r in rows)
    high=sum(prediction(r['answers'][reference])!=prediction(r['answers'][candidate]) and max(distribution(r['answers'][reference]).values())>=.9 for r in rows)
    disagreements=regressions+fixes
    p=min(1,2*sum(math.comb(disagreements,i) for i in range(min(regressions,fixes)+1))/2**disagreements) if disagreements else 1
    result={'reference':reference,'candidate':candidate,'questions':len(rows),'labeled_questions':len(labeled),
            'decision_flips':flips,'high_confidence_reference_flips':high,'correct_to_wrong':regressions,'wrong_to_correct':fixes,
            'accuracy_delta':(fixes-regressions)/len(labeled) if labeled else None,
            'mcnemar_exact_p':p if len({r['id'] for r in labeled})==len(labeled) else None,
            'mean_total_variation':mean(.5*sum(abs(distribution(r['answers'][candidate])[k]-v) for k,v in distribution(r['answers'][reference]).items()) for r in rows) if rows else None}
    numeric={kind:[r['answers'][candidate][field]-r['answers'][reference][field] for r in rows if r['answers'][reference]['type']==kind]
             for kind,field in [('score','score'),('noul','noul')]}
    result['numeric_drift']={kind:{'n':len(values),'mean_absolute':mean(abs(v) for v in values),'max_absolute':max(abs(v) for v in values),'mean_signed':mean(values)} for kind,values in numeric.items() if values}
    if bootstrap and labeled:


        assert len({r['id'] for r in labeled})==len(labeled)
        delta=[int(prediction(r['answers'][candidate])==r['gold'])-int(prediction(r['answers'][reference])==r['gold']) for r in labeled]
        rng=random.Random(20261018);n=len(delta)
        samples=sorted(sum(delta[rng.randrange(n)] for _ in range(n))/n for _ in range(4000))
        result['paired_bootstrap_95_ci']=[samples[99],samples[3899]]
    return result


def analyze(corpus_path,paths,partial=False,modes=None):
    modes = MODES if modes is None else modes
    assert len(modes)==3 and len(set(modes))==3 and modes[0]=='fp16-scan'
    fp16,q8,q4=modes
    raw=corpus_path.read_bytes();corpus=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
    cases={c['id']:c for c in corpus['cases']};runs={m:{} for m in modes};builds=set();revisions=set()
    for path in paths:
        report=json.loads(path.read_text());assert report['completed'] and report['purpose']=='quality-evaluation'
        assert report['corpus_sha256']==report['metadata']['corpus_sha256']==digest
        builds.add(report['wasm_sha256']);revisions.add(report['metadata']['revision'])
        for run in report['runs']:
            mode=run['mode'];id=run['fixture'];assert id in cases and mode in modes and id not in runs[mode]
            assert not run['warmup'] and run['round']==0
            assert len(run['layers'])==32*run['response']['usage']['sequences']
            assert run['timing_scope']=='warm-request' and run['decoder_weight_bytes_read']==0
            assert all(l['pack_ms']==0 and l['read_ms']==0 for l in run['layers'])
            if mode in ['int8-all','q4-all']:
                auxiliary=run['adapter']['auxiliary_quantization']
                assert auxiliary['weight_bits']==(8 if mode=='int8-all' else 4)
                assert auxiliary['tensors']==['language_model.embed_tokens.weight','readout.weight']
                assert auxiliary['performance_eligible'] is False
            assert set(run['response']['answers'])==set(cases[id]['request']['questions'])
            assert all(math.isfinite(x) for c in run['diagnostics']['logits'] for row in c['values'] for x in row)
            for key,answer in run['response']['answers'].items():
                assert answer['type']==cases[id]['request']['questions'][key]['type']
                prediction(answer)
            runs[mode][id]=run
    assert len(builds)==len(revisions)==1
    matched=set.intersection(*(set(runs[m]) for m in modes))
    if not partial:assert all(set(runs[m])==set(cases) for m in modes),'Missing model/case outputs'
    rows=[];details=[]
    for id in [c['id'] for c in corpus['cases'] if c['id'] in matched]:
        case=cases[id]
        counts=[runs[m][id]['response']['usage'] for m in modes];assert all(c==counts[0] for c in counts)
        for question in case['request']['questions']:
            gold=case['expected'].get(question,{})
            answers={m:runs[m][id]['response']['answers'][question] for m in modes}
            assert all(set(distribution(a))==set(distribution(answers[modes[0]])) for a in answers.values())
            row={'id':id,'family':case['family'],'question':question,'gold':gold.get('label'),'ordinal':gold.get('ordinal'),
                 'primary':question==primary(case),'answers':answers}
            rows.append(row)
            preds={m:prediction(a) for m,a in answers.items()}
            probs={m:distribution(a) for m,a in answers.items()}
            tv=.5*sum(abs(probs[q4][k]-v) for k,v in probs[fp16].items())
            score_delta=answers[q4]['score']-answers[fp16]['score'] if answers[fp16]['type']=='score' else None
            details.append({**{k:v for k,v in row.items() if k!='answers'},'predictions':preds,'probabilities':probs,
                'numeric':{m:a.get('score',a.get('noul')) for m,a in answers.items()},
                'q4_total_variation_vs_fp16':tv,'q4_score_delta_vs_fp16':score_delta,
                'source':case['source'],'input_tokens':counts[0]['input_tokens']})
    primary_rows=[r for r in rows if r['primary']]
    tone_rows={r['id']:r for r in rows if r['family']=='controlled-tone' and r['question']=='tone'}
    tone_triplets=[]
    for base in sorted({id.rsplit('-',1)[0] for id in tone_rows}):
        ids=[base+'-'+str(i) for i in range(3)]
        if not all(id in tone_rows for id in ids):continue
        values={m:[tone_rows[id]['answers'][m]['score'] for id in ids] for m in modes}
        tone_triplets.append({'topic':base,'scores_calm_concerned_angry':values,'strict_order':{m:v[0]<v[1]<v[2] for m,v in values.items()}})
    groups={group:[r for r in rows if r['family']+'/'+r['question']==group] for group in sorted({r['family']+'/'+r['question'] for r in rows})}
    summary={'complete':not partial,'expected_inputs':len(cases),'matched_inputs':len(matched),'coverage':{m:len(r) for m,r in runs.items()},
        'modes':modes,'quantization':{m:next(iter(runs[m].values()))['adapter'].get('auxiliary_quantization') for m in modes},
        'corpus_sha256':digest,'wasm_sha256':next(iter(builds)),'checkpoint_revision':next(iter(revisions)),'seed':corpus['seed'],'dataset_sources':corpus['sources'],
        'methodology':{'public_examples':128,'authored_or_regression_examples':32,'labeled_decisions':sum(r['gold'] is not None for r in rows),
            'primary':'One predeclared primary question per input; auxiliary judgments reported separately.',
            'score':'Score-class accuracy uses probability argmax; ordinal MAE uses expected score against the annotated class index.',
            'binary':'noul is classified true at probability >=0.5.',
            'uncertainty':'Paired 4000-resample bootstrap over inputs, 95% percentile interval. Task subsets are small; authored labels are not independent human annotations.',
            'scope':'Frozen subset evaluation, not a full public benchmark or representative production accuracy. No latency claims from these non-warmed quality runs.'},
        'primary_metrics':{m:metric(primary_rows,m) for m in modes},
        'primary_comparisons':[paired(primary_rows,m,bootstrap=True) for m in modes[1:]],
        'q4_vs_q8_primary':paired(primary_rows,q4,q8,bootstrap=True),
        'controlled_tone_triplets':tone_triplets,
        'groups':{g:{'metrics':{m:metric(rs,m) for m in modes},'comparisons':[paired(rs,m) for m in modes[1:]]} for g,rs in groups.items()},
        'all_question_drift':[paired(rows,m) for m in modes[1:]],
        'largest_q4_distribution_changes':sorted(details,key=lambda d:d['q4_total_variation_vs_fp16'],reverse=True)[:20],
        'largest_q4_score_changes':sorted([d for d in details if d['q4_score_delta_vs_fp16'] is not None],key=lambda d:abs(d['q4_score_delta_vs_fp16']),reverse=True)[:20],
        'details':details}
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--partial',action='store_true')
    parser.add_argument('--modes',nargs=3,default=MODES)
    parser.add_argument('reports',nargs='+',type=Path);args=parser.parse_args()
    result=analyze(args.corpus,args.reports,args.partial,args.modes)
    with args.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ['complete','matched_inputs','coverage','primary_metrics','primary_comparisons']}))
