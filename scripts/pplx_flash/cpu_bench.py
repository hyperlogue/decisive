import argparse
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess

from analyze_quality import distribution, metric, paired, prediction, primary

MODES = ['bf16', 'q8', 'q4', 'q4-e4', 'q4-mse', 'q4-e4-mse', 'q4-e4-nf4', 'q4-e4-affine', 'q4-all']
PRECISIONS = {
    'bf16': 'bf16-storage-f32-compute', 'q8': 'q8-w8a8-e8-r8-g64',
    'q4': 'q4-w4a8-e8-r4-g64', 'q4-e4': 'q4-w4a8-e4-r4-g64',
    'q4-mse': 'q4-w4a8-e8-r4-g64-mse', 'q4-e4-mse': 'q4-w4a8-e4-r4-g64-mse',
    'q4-e4-affine': 'q4-w4a8-e4-r4-g64-affine',
    'q4-e4-nf4': 'q4-w4a8-e4-r4-g64-mse-nf4', 'q4-all': 'q4-w4a4-e4-r4-g64',
}
LABELS = {
    'bf16': 'BF16 baseline', 'q8': 'W8A8 · E8', 'q4': 'W4A8 · E8',
    'q4-e4': 'W4A8 · E4', 'q4-mse': 'W4A8 · E8 · MSE',
    'q4-e4-mse': 'W4A8 · E4 · MSE', 'q4-e4-nf4': 'W4A8 · NF4 embeddings · MSE',
    'q4-e4-affine': 'W4A8 · E4 · affine',
    'q4-all': 'W4A4 · E4',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(args.corpus.read_text())
    binary = root / 'decisive'
    if not binary.exists():
        shutil.copy2(args.binary, binary)
    else:
        assert digest(binary) == digest(args.binary), 'binary changed; choose a fresh output directory'
    cases = corpus['cases']
    if args.performance:
        cases = [c for c in cases if c['family'] == 'regression']
        assert args.rounds >= 3, 'use at least one warmup and two measured rounds'
    requests = ''.join(json.dumps(c['request'], ensure_ascii=False) + '\n' for c in cases)
    prefix = 'performance-' if args.performance else ''
    path = root / (prefix + 'requests.jsonl')
    if path.exists():
        assert path.read_text() == requests
    else:
        path.write_text(requests)
    manifest = {
        'binary_sha256': digest(binary), 'binary_file': binary.name, 'corpus_sha256': digest(args.corpus),
        'requests_sha256': digest(path), 'modes': args.modes, 'threads': args.threads,
        'rounds': args.rounds, 'cases': [c['id'] for c in cases],
        'warmup_rounds': 1 if args.performance else 0,
        'timing_scope': 'resident weights; tokenization through response; model loading reported separately',
        'machine': {'architecture': platform.machine(), 'logical_cpus': os.cpu_count()},
        'source_sha256': {str(p): digest(p) for p in sorted(Path('src/models/pplx_flash').glob('*.rs'))},
        'capabilities': json.loads(subprocess.check_output([str(binary.resolve()), 'capabilities', '--json'])),
    }
    write(root / (prefix + '-'.join(args.modes) + '-manifest.json'), manifest)
    for mode in args.modes:
        command = [str(binary.resolve()), 'pplx-flash', '--resident', '--threads', str(args.threads),
                   '--quantization', mode, '--profile', 'evaluate', '--jsonl',
                   '--input', str(path), '--rounds', str(args.rounds)]
        print(f'{prefix}{mode}: {len(cases)} inputs, {args.rounds} rounds', flush=True)
        with (root / (prefix + mode + '.jsonl')).open('x') as stream:
            subprocess.run(command, stdout=stream, check=True)


def read_runs(path, cases, mode):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records and len(records) % len(cases) == 0, 'incomplete evaluation'
    rounds = len(records) // len(cases)
    assert [(r['round'], r['case']) for r in records] == [(round, i) for round in range(rounds) for i in range(len(cases))]
    for r in records:
        assert r['metrics']['precision'] == PRECISIONS[mode]
        assert r['metrics']['weight_loading'] == 'resident packed decoder weights'
        assert len(r['metrics']['layers']) == 32 * r['response']['usage']['sequences']
        case = cases[r['case']]
        assert set(r['response']['answers']) == set(case['request']['questions'])
        assert set(r['metrics']['logits']) == set(case['request']['questions'])
        for key, answer in r['response']['answers'].items():
            assert answer['type'] == case['request']['questions'][key]['type']
            prediction(answer)
            assert all(math.isfinite(x) for x in r['metrics']['logits'][key])
        assert r['response'] == records[r['case']]['response'], 'repeated inference changed outputs'
        assert r['metrics']['logits'] == records[r['case']]['metrics']['logits'], 'repeated inference changed logits'
    return records


def compare_mae(details, candidate, reference):

    clusters = {}
    for row in details:
        base = row['probabilities']['bf16']
        a, b = (row['probabilities'][m] for m in [candidate, reference])
        item = clusters.setdefault(row['id'], [0., 0., 0])
        item[0] += sum(abs(a[k] - v) for k, v in base.items())
        item[1] += sum(abs(b[k] - v) for k, v in base.items())
        item[2] += len(base)
    values = list(clusters.values())
    count = sum(v[2] for v in values)
    a, b = (sum(v[i] for v in values) / count for i in [0, 1])
    rng = random.Random(20261018)
    samples = []
    for _ in range(4000):
        selected = rng.choices(values, k=len(values))
        samples.append(sum(v[0] - v[1] for v in selected) / sum(v[2] for v in selected))
    samples.sort()
    return {'candidate': candidate, 'reference': reference, 'probability_entries': count,
            'candidate_mae': a, 'reference_mae': b, 'mae_delta': a - b,
            'relative_mae_reduction': 1 - a / b if b else None,
            'paired_input_bootstrap_95_ci': [samples[100], samples[3899]]}


def summarize(args):
    modes = args.modes
    assert modes[0] == "bf16", "summaries require BF16 as the first reference mode"
    corpus = json.loads(args.corpus.read_text())
    cases = corpus['cases']
    runs = {m: read_runs(args.output / (m + '.jsonl'), cases, m)[:len(cases)] for m in modes}
    manifests = {}
    for path in args.output.glob('*-manifest.json'):
        if path.name.startswith('performance-'):
            continue
        manifest = json.loads(path.read_text())
        assert manifest['binary_sha256'] == digest(args.output / manifest.get('binary_file', 'decisive'))
        assert manifest['corpus_sha256'] == digest(args.corpus)
        assert manifest['requests_sha256'] == digest(args.output / 'requests.jsonl')
        assert manifest['cases'] == [c['id'] for c in cases]
        for mode in manifest['modes']:
            if mode not in modes:
                continue
            assert mode not in manifests
            manifests[mode] = manifest
            assert all(r['threads'] == manifest['threads'] for r in runs[mode])
    assert set(manifests) == set(modes), 'missing build/corpus provenance'
    revisions = {r['metrics']['revision'] for mode in runs.values() for r in mode}
    assert len(revisions) == 1
    rows = []
    errors = {m: {'logit': [], 'probability': []} for m in modes[1:]}
    details = []
    for i, case in enumerate(cases):
        assert all(runs[m][i]['response']['usage'] == runs['bf16'][i]['response']['usage'] for m in modes)
        for key in case['request']['questions']:
            gold = case['expected'].get(key, {})
            answers = {m: runs[m][i]['response']['answers'][key] for m in modes}
            row = {'id': case['id'], 'family': case['family'], 'question': key,
                   'gold': gold.get('label'), 'ordinal': gold.get('ordinal'),
                   'primary': key == primary(case), 'answers': answers}
            rows.append(row)
            base = distribution(answers['bf16'])
            for mode in modes[1:]:
                probs = distribution(answers[mode])
                assert probs.keys() == base.keys()
                errors[mode]['probability'].extend(probs[k] - base[k] for k in base)
                a, b = (runs[m][i]['metrics']['logits'][key] for m in ['bf16', mode])
                assert len(a) == len(b)
                errors[mode]['logit'].extend(y-x for x, y in zip(a, b))
            details.append({k: v for k, v in row.items() if k != 'answers'} | {
                'predictions': {m: prediction(a) for m, a in answers.items()},
                'probabilities': {m: distribution(a) for m, a in answers.items()}})
    primary_rows = [r for r in rows if r['primary']]
    groups = {group: [r for r in rows if r['family']+'/'+r['question'] == group]
              for group in sorted({r['family']+'/'+r['question'] for r in rows})}
    result = {
        'complete': True, 'inputs': len(cases), 'questions': len(rows), 'corpus_sha256': digest(args.corpus),
        'checkpoint_revision': next(iter(revisions)), 'modes': modes,
        'quality_binary_sha256': {m: manifests[m]['binary_sha256'] for m in modes},
        'scheme': 'Groups of 64 with FP32 scales. Base modes use symmetric absmax. MSE presets fit signed INT4 scales and clipping using all 16 codes; NF4 preset uses fitted NormalFloat4 embeddings. Affine Q4 fits both scale and offset, requiring 5 effective bits per weight including metadata versus 4.5 for symmetric Q4. Activations retain symmetric absmax. Readout follows weights. No calibration inputs or labels are used.',
        'q4_all_scope': 'W4A4 projections including small gates and readout, E4 embeddings; FP32 normalization, convolution, attention, recurrent state, residuals and accumulation; activation codes held in byte lanes for AVX2 dots',
        'methodology': {
            'primary': 'One predeclared primary question per input; auxiliary judgments reported separately.',
            'score': 'Class accuracy and flips use probability argmax; numeric score drift uses the expected score returned by the API.',
            'binary': 'noul is classified true at probability >= 0.5.',
            'uncertainty': '4,000 paired bootstrap resamples over inputs. A zero-width interval on unchanged samples does not establish population equivalence.',
        },
        'limitations': ['160-input frozen subset, not full TweetEval/BANKING77 or production accuracy',
                        'Authored tone/stress labels are not independent human annotations',
                        'CPU grouped quantization differs from browser per-row quantization',
                        'Quality-run latency is not a warmed performance measurement',
                        'Latency is host-specific; raw samples are retained and small differences between quantized modes should not be overinterpreted'],
        'primary_metrics': {m: metric(primary_rows, m) for m in modes},
        'primary_comparisons': [paired(primary_rows, m, 'bf16', bootstrap=True) for m in modes[1:]],
        'all_question_drift': [paired(rows, m, 'bf16') for m in modes[1:]],
        'errors': {m: {kind: {'n': len(v), 'mae': statistics.mean(abs(x) for x in v),
                            'rmse': math.sqrt(statistics.mean(x*x for x in v)),
                            'max_absolute': max(abs(x) for x in v)} for kind, v in e.items()} for m, e in errors.items()},
        'groups': {g: {'metrics': {m: metric(rs, m) for m in modes},
                       'comparisons': [paired(rs, m, 'bf16') for m in modes[1:]]} for g, rs in groups.items()},
        'model_bytes': {m: {k: runs[m][0]['metrics'][k] for k in ['packed_projection_bytes', 'embedding_bytes', 'readout_bytes']} for m in modes},
        'dataset_sources': corpus['sources'], 'details': details,
        'weight_bytes_scope': 'packed decoder projections plus full vocabulary embeddings and readout; excludes small FP32 tensors, workspaces, tokenizer and allocator overhead',
    }
    result['largest_distribution_changes'] = {}
    pairs = [('q4-e4', 'q4'), ('q4-mse', 'q4'), ('q4-e4-mse', 'q4-e4'),
             ('q4-e4-nf4', 'q4-e4'), ('q4-e4-nf4', 'q4-e4-mse'), ('q4-e4-affine', 'q4-e4'), ('q4-e4-affine', 'q4'), ('q4-all', 'q4-e4')]
    pairs = [(a, b) for a, b in pairs if a in modes and b in modes]
    result['quantizer_mae_comparisons'] = [compare_mae(details, a, b) for a, b in pairs]
    result['quantizer_primary_comparisons'] = [paired(primary_rows, a, b, bootstrap=True) for a, b in pairs]
    if (args.output / 'experiment.json').exists():
        result['quantizer_experiment'] = json.loads((args.output / 'experiment.json').read_text())
    for mode in modes[1:]:
        changes = []
        for row in details:
            base, candidate = row['probabilities']['bf16'], row['probabilities'][mode]
            changes.append({k: row[k] for k in ['id', 'family', 'question', 'gold', 'primary']} | {
                'bf16_prediction': row['predictions']['bf16'], 'candidate_prediction': row['predictions'][mode],
                'total_variation': .5*sum(abs(base[k]-candidate[k]) for k in base)})
        result['largest_distribution_changes'][mode] = sorted(changes, key=lambda c: c['total_variation'], reverse=True)[:20]
    perf_cases = [c for c in cases if c['family'] == 'regression']
    perf_paths = [args.output / ('performance-' + m + '.jsonl') for m in modes]
    if any(p.exists() for p in perf_paths):
        perf = {m: read_runs(p, perf_cases, m) for m, p in zip(modes, perf_paths)}
        perf_manifests = [json.loads(p.read_text()) for p in args.output.glob('performance-*-manifest.json')]
        assert set(m for p in perf_manifests for m in p['modes']) >= set(modes)
        result['performance_binary_sha256'] = {}
        for p in perf_manifests:
            assert p['binary_sha256'] == digest(args.output / p.get('binary_file', 'decisive'))
            assert p['corpus_sha256'] == result['corpus_sha256']
            assert p['cases'] == [c['id'] for c in perf_cases] and p['warmup_rounds'] == 1
            assert p['requests_sha256'] == digest(args.output / 'performance-requests.jsonl')
            for mode in p['modes']:
                if mode not in modes:
                    continue
                assert mode not in result['performance_binary_sha256']
                result['performance_binary_sha256'][mode] = p['binary_sha256']
                assert all(r['threads'] == p['threads'] and r['metrics']['revision'] == result['checkpoint_revision'] for r in perf[mode])
        result['performance'] = []
        for i, case in enumerate(perf_cases):
            entry = {'case': case['id'], 'modes': {}}
            for mode in modes:
                measured = [r for r in perf[mode] if r['case'] == i and r['round'] > 0]
                assert len(measured) >= 2
                corpus_index = next(j for j, c in enumerate(cases) if c['id'] == case['id'])
                assert all(r['response'] == runs[mode][corpus_index]['response'] for r in measured)
                assert all(r['metrics']['logits'] == runs[mode][corpus_index]['metrics']['logits'] for r in measured)
                entry['modes'][mode] = {
                    'median_ms': statistics.median(r['metrics']['elapsed_ms'] for r in measured),
                    'samples_ms': [r['metrics']['elapsed_ms'] for r in measured],
                    'threads': measured[0]['threads'], 'tokens': measured[0]['response']['usage']['input_tokens'],
                    'load_ms': measured[0]['model_load_ms'],
                    'projection_ms': statistics.median(sum(l['projection_ms'] for l in r['metrics']['layers']) for r in measured),
                }
            result['performance'].append(entry)
    if (args.output / 'equivalence.json').exists():
        equivalence = json.loads((args.output / 'equivalence.json').read_text())
        assert equivalence['passed'] and equivalence['corpus_sha256'] == result['corpus_sha256']
        for mode in [m for m in ['q4', 'q4-all'] if m in modes]:
            assert equivalence['modes'][mode]['requests'] == len(cases)
            assert equivalence['modes'][mode]['responses_exact'] and equivalence['modes'][mode]['logits_exact']
            assert equivalence['optimized_binary_sha256'] == result['quality_binary_sha256'][mode]
        result['kernel_equivalence'] = equivalence
    write(args.output / 'summary.json', result)
    with (args.output / 'report.html').open('x') as stream:
        stream.write(render(result))
    print(json.dumps({k: result[k] for k in ['inputs', 'primary_metrics', 'primary_comparisons']}, indent=2))


def render(result):
    modes = result['modes']
    labels = {m: LABELS[m] for m in modes}
    comparisons = {r['candidate']: r for r in result['primary_comparisons']}
    drift = {r['candidate']: r for r in result['all_question_drift']}
    def table(headers, rows):
        return '<table><thead><tr>' + ''.join('<th>'+escape(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
            '<tr>' + ''.join('<td>'+escape(str(v))+'</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'
    def pct(v): return f'{100*v:.3f}%'
    quality = []
    for mode in modes:
        m = result['primary_metrics'][mode]
        c = comparisons.get(mode)
        quality.append([labels[mode], f"{m['correct']}/{m['n']}", pct(m['accuracy']),
                        '—' if not c else f"{100*c['accuracy_delta']:+.2f} pp",
                        '—' if not c else f"{c['decision_flips']} / {drift[mode]['decision_flips']}",
                        f"{m['nll']:.4f}", f"{sum(result['model_bytes'][mode].values())/2**30:.3f}"])
    numeric = [[labels[m], f"{result['errors'][m]['logit']['mae']:.6f}",
                f"{result['errors'][m]['logit']['rmse']:.6f}",
                f"{result['errors'][m]['probability']['mae']:.6f}",
                f"{result['errors'][m]['probability']['max_absolute']:.6f}",
                f"{drift[m]['mean_total_variation']:.6f}"] for m in modes[1:]]
    uncertainty = [[labels[m], f"[{100*comparisons[m]['paired_bootstrap_95_ci'][0]:+.2f}, {100*comparisons[m]['paired_bootstrap_95_ci'][1]:+.2f}] pp",
                    comparisons[m]['correct_to_wrong'], comparisons[m]['wrong_to_correct']] for m in modes[1:]]
    perf = []
    for case in result.get('performance', []):
        base = case['modes']['bf16']['median_ms']
        for mode, values in case['modes'].items():
            perf.append([case['case'].removeprefix('regression-'), labels[mode], values['tokens'],
                         values['threads'], f"{values['median_ms']:.1f}", f"{base/values['median_ms']:.2f}×",
                         f"{values['projection_ms']:.1f}"])
    groups = [[name, next(iter(g['metrics'].values()))['n']] + [pct(g['metrics'][m]['accuracy']) for m in modes]
              for name, g in result['groups'].items() if next(iter(g['metrics'].values()))['n']]
    sections = '<h2>Primary task quality and matrix storage</h2>' + table(
        ['Mode', 'Correct', 'Accuracy', 'Δ vs BF16', 'Flips: primary / all', 'NLL ↓', 'Weights GiB'], quality)
    sections += '<p>One primary question per input. Readout uses the same weight and activation precision as decoder projections. Matrix storage excludes small FP32 tensors and inference workspaces; it is not peak RSS.</p>'
    sections += '<p>Score-class accuracy and decision flips use probability argmax. Numeric score drift uses the expected score returned by the API. Binary answers use a 0.5 threshold.</p>'
    sections += '<h2>Numerical drift against BF16</h2>' + table(
        ['Mode', 'Logit MAE', 'Logit RMSE', 'Probability MAE', 'Max probability error', 'Mean total variation'], numeric)
    if result.get('quantizer_mae_comparisons'):
        sections += '<h2>Quantizer ablations</h2>' + table(
            ['Candidate', 'Reference', 'MAE reduction', '95% interval for MAE difference'],
            [[labels[c['candidate']], labels[c['reference']],
              '—' if c['relative_mae_reduction'] is None else pct(c['relative_mae_reduction']),
              f"[{c['paired_input_bootstrap_95_ci'][0]:+.6f}, {c['paired_input_bootstrap_95_ci'][1]:+.6f}]"]
             for c in result['quantizer_mae_comparisons']])
        sections += '<p>Both errors are measured against BF16. Positive reduction favors the candidate. The bootstrap resamples entire inputs, including their auxiliary questions; intervals are exploratory and are not corrected for multiple comparisons. Quantizers use weights only, with no calibration inputs or label fitting. This is exploratory method selection on a single evaluation corpus; confirm the best candidate on new data.</p>'
    numeric_answers = [[labels[m], kind, values['n'], f"{values['mean_absolute']:.6f}",
                        f"{values['max_absolute']:.6f}", f"{values['mean_signed']:+.6f}"]
                       for m in modes[1:] for kind, values in drift[m]['numeric_drift'].items()]
    sections += '<h3>Numeric answer drift</h3>' + table(
        ['Mode', 'Answer type', 'n', 'MAE', 'Maximum error', 'Mean signed change'], numeric_answers)
    sections += '<h2>Paired accuracy uncertainty</h2>' + table(
        ['Mode', '95% bootstrap interval for Δ', 'Correct → wrong', 'Wrong → correct'], uncertainty)
    sections += '<p>4,000 paired resamples over inputs. The small task subsets and authored examples limit generalization. A zero-width interval on unchanged samples does not establish population equivalence.</p>'
    if perf:
        sections += '<h2>Warmed end-to-end latency</h2>' + table(
            ['Fixture', 'Mode', 'Tokens', 'Threads', 'Median ms ↓', 'Speedup', 'Decoder projection ms'], perf)
        sections += '<p>Resident weights; tokenization through response. First round excluded. Loading is reported separately in summary.json. Quality runs are excluded from these latency measurements.</p>'
    sections += '<h2>Task breakdown</h2>' + table(['Task / question', 'Labeled n'] + list(labels.values()), groups)
    sections += '<h2>Largest distribution changes</h2>'
    for mode in modes[1:]:
        changes = result['largest_distribution_changes'][mode][:8]
        sections += '<h3>'+escape(labels[mode])+'</h3>' + table(
            ['Input / question', 'Gold', 'BF16 prediction', 'Quantized prediction', 'Total variation'],
            [[c['id']+' / '+c['question'], c['gold'] or 'unlabeled', c['bf16_prediction'], c['candidate_prediction'],
              f"{c['total_variation']:.4f}"] for c in changes])
    sections += '<h2>Scope and reproduction</h2><ul>' + ''.join('<li>'+escape(x)+'</li>' for x in result['limitations']) + '</ul>'
    sections += '<p>'+escape(result['q4_all_scope'])+'</p><p>'+escape(result['scheme'])+'</p>'
    sections += '<p>Quantized kernels retain packed weight panels, reuse shared activation quantization, and keep the existing tiled attention, recurrent scans, and decision-token pruning. Q4 expands a small panel into reusable thread-local scratch for longer inputs, and unpacks directly in AVX2 registers for short inputs.</p>'
    sections += '<p>Run <code>scripts/pplx_flash/cpu_bench.py</code> as described in its module docstring. <a href="summary.json">Full metrics and per-input decisions</a> · <a href="corpus.json">Frozen corpus and dataset provenance</a></p>'
    sections += '<p>Quality and performance build identities are recorded separately. Every warmed performance output must match the corresponding corpus output.</p>'
    if result.get('kernel_equivalence'):
        sections += '<p>The final Q4 panel-cache implementation was rerun on the complete corpus: all Q4 responses and logits match the earlier register-unpacking implementation exactly. BF16 and Q8 quality runs precede that optimization.</p>'
    if result.get('quantizer_experiment'):
        sections += '<p><a href="experiment.json">Quantizer settings and experiment selection history</a>. Integer MSE fitting searches clipping and signed scales; affine fitting jointly solves scale and offset; fitted <a href="https://arxiv.org/abs/2305.14314">NF4</a> applies to embeddings only. These are load-time, data-free methods. The affine follow-up was selected after interim scale-only results; confirmation on a new corpus is still needed.</p>'
    sections += '<p class="hash">Checkpoint: '+escape(result['checkpoint_revision'])+'<br>Corpus SHA-256: '+escape(result['corpus_sha256'])+'<br>'+'<br>'.join('Quality '+m+': '+v for m,v in result['quality_binary_sha256'].items())+'</p>'
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PPLX CPU precision audit</title><style>body{font:16px/1.5 system-ui,sans-serif;color:#182330;background:#f8fafc;max-width:1180px;margin:48px auto;padding:0 24px}h1{font-size:32px;margin-bottom:8px}h2{font-size:22px;margin-top:40px}table{width:100%;border-collapse:collapse;background:white;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:10px 12px;border-bottom:1px solid #dce3eb}th{font-size:13px;background:#eaf0f7}td:first-child,th:first-child{text-align:left}p,li{max-width:1000px;color:#465366}.hash{font:12px/1.8 monospace;overflow-wrap:anywhere}code{font-size:14px}a{color:#1855a6}</style><h1>PPLX CPU precision audit</h1><p>'+str(result['inputs'])+' frozen inputs · '+str(result['questions'])+' questions · BF16 storage with FP32 compute as reference</p>'+sections+'</html>'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Frozen-corpus CPU precision audit and separately warmed request benchmarks.\n\nPrepare the corpus with prepare_quality.py, then run this script with action run.\nAll arithmetic and inference run in the Rust CLI. Raw outputs are retained.\n\nExample (run from the repository root):\n  cargo build --locked --release --no-default-features --features cli,pplx-flash,int8\n  python3 scripts/pplx_flash/prepare_quality.py --output benchmark-runs/cpu/corpus.json\n  python3 scripts/pplx_flash/cpu_bench.py run --corpus benchmark-runs/cpu/corpus.json --output benchmark-runs/cpu\n  python3 scripts/pplx_flash/cpu_bench.py run --corpus benchmark-runs/cpu/corpus.json --output benchmark-runs/cpu --performance --rounds 5\n  python3 scripts/pplx_flash/cpu_bench.py summarize --corpus benchmark-runs/cpu/corpus.json --output benchmark-runs/cpu\n')
    parser.add_argument('action', choices=['run', 'summarize'])
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('target/release/decisive'))
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--performance', action='store_true')
    args = parser.parse_args()
    assert args.threads > 0 and args.rounds > 0
    assert len(set(args.modes)) == len(args.modes)
    {'run': run, 'summarize': summarize}[args.action](args)
