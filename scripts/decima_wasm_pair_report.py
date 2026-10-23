from pathlib import Path
import collections, json, math, statistics, sys
root=Path(sys.argv[1]); destination=Path(sys.argv[2])
matrix=json.loads((root/'matrix.json').read_text())
completed=json.loads((root/'summary.json').read_text())
assert completed['failures']==0
files=sorted(root.glob('rep-*.json'))
assert len(files)==matrix['repetitions']*len(matrix['cases'])
groups=collections.defaultdict(list); metadata={}
for path in files:
    run=json.loads(path.read_text()); case=path.stem.split('-',2)[2]
    names=[p['name'] for p in run['packages']]
    for package in run['packages']:
        key=package['name']; selected={k:package[k] for k in ['name','path','workers','wasm_sha256']}
        assert key not in metadata or metadata[key]==selected
        metadata[key]=selected
    for fixture in sorted({r['fixture'] for r in run['records']}):
        records={r['variant']:r for r in run['records'] if r['fixture']==fixture}
        assert list(records)==names
        a=records[names[0]]
        for name in names[1:]:
            b=records[name]
            assert a['output']==b['output']
            before=a['latencies_ms']; after=b['latencies_ms']
            assert len(before)==len(after)==matrix['rounds']
            assert all(math.isfinite(x) and x>0 for x in before+after)
            groups[(case,run['precision'],run['cache'],fixture,a.get('choices'),names[0],name)].append({
                'file':path.name,'before_ms':before,'after_ms':after,
                'paired_speedups':[x/y for x,y in zip(before,after)]})
rows=[]
for (case,precision,cache,fixture,choices,before_name,after_name),runs in groups.items():
    before=[x for r in runs for x in r['before_ms']]
    after=[x for r in runs for x in r['after_ms']]
    ratios=[x for r in runs for x in r['paired_speedups']]
    row={'case':case,'precision':precision,'cache':cache,'fixture':fixture,'choices':choices,
         'before':before_name,'after':after_name,'samples_per_variant':len(before),
         'before_ms':statistics.median(before),'after_ms':statistics.median(after),
         'median_process_before_ms':statistics.median(statistics.median(r['before_ms']) for r in runs),
         'median_process_after_ms':statistics.median(statistics.median(r['after_ms']) for r in runs),
         'before_p95_ms':sorted(before)[math.ceil(.95*len(before))-1],
         'after_p95_ms':sorted(after)[math.ceil(.95*len(after))-1],
         'paired_speedup':statistics.median(ratios),
         'per_run_speedup':[statistics.median(r['paired_speedups']) for r in runs],
         'per_run_medians_ms':[[statistics.median(r['before_ms']),statistics.median(r['after_ms'])] for r in runs]}
    rows.append(row)
result={'method':'Alternating complete requests from isolated persistent modules in one dedicated Worker; fresh browser per matrix case. All samples retained. Speedup is the median of paired before/after ratios; latency columns are pooled medians, with median process medians also recorded. p95 uses the empirical nearest rank. Process ratios expose variability; they are not confidence intervals.',
        'matrix':matrix,'packages':list(metadata.values()),'all_outputs_exact':True,'rows':rows}
destination.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'file':str(destination),'comparisons':len(rows),'paired_runs':len(files)}))
