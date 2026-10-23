import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import shutil
import zipfile
from pathlib import Path


def write_lines(path, values):
    path.write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values))


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = Path('tests/fixtures/pplx_flash')
    requests = [json.loads((fixtures / f'{name}.json').read_text()) for name in ['request', 'multilingual']]
    cases = [{'name': 'english-routing'}, {'name': 'multilingual-types'}]
    for length in args.lengths:
        count = max(0, length - 117)
        for _ in range(4):
            request = json.loads(json.dumps(requests[0]))
            request['state'] += '\nUnrelated archived markers:' + ' x' * count
            path = args.output / f'length-{length}.json'
            path.write_text(json.dumps(request) + '\n')
            info = json.loads(subprocess.check_output([str(args.binary), 'pplx-flash', 'inspect', '--input', str(path)]))
            actual = sum(len(c['ids']) for c in info['chunks'])
            if actual == length:
                break
            count += length - actual
            assert count >= 0, 'requested length is below the request template'
        assert actual == length, 'could not produce exact token length'
        requests.append(request)
        cases.append({'name': f'english-padding-{length}', 'tokens': length,
                      'scope': 'repeated irrelevant filler; throughput case, not representative quality'})
    if args.xnli:
        digest = hashlib.sha256(args.xnli.read_bytes()).hexdigest()
        if args.xnli.suffix == '.zip':
            assert digest == '4ba1d5e1afdb7161f0f23c66dc787802ccfa8a25a3ddd3b165a35e50df346ab1', 'XNLI archive checksum mismatch'
            with zipfile.ZipFile(args.xnli) as archive:
                records = [json.loads(line) for line in archive.read('XNLI-1.0/xnli.test.jsonl').splitlines()]
        else:
            records = [json.loads(line) for line in args.xnli.read_text().splitlines()]
        labels = ['entailment', 'neutral', 'contradiction']
        if records and 'sentence1' in records[0]:
            records = [{'id': f"xnli-test-{r['language']}-{r['pairID']}", 'pair_id': r['pairID'],
                        'language': r['language'], 'state': r['sentence1'], 'gold': labels.index(r['gold_label']),
                        'question': {'text': 'Given the premise, how is this hypothesis related to it? Hypothesis: ' + r['sentence2'],
                                     'choices': ['The hypothesis follows from the premise.',
                                                 'The premise does not determine whether the hypothesis is true.',
                                                 'The hypothesis contradicts the premise.']}} for r in records]
        assert set(args.languages.split(',')) <= {r['language'] for r in records}
        ids = sorted({r['pair_id'] for r in records})
        assert 0 < args.pairs <= len(ids)
        chosen = {ids[(i * len(ids) + len(ids) // 2) // args.pairs] for i in range(args.pairs)}
        for row in records:
            if row['pair_id'] not in chosen or row['language'] not in args.languages.split(','):
                continue
            requests.append({'state': row['state'], 'questions': {'relation': {
                'type': 'choice', 'instructions': row['question']['text'],
                'criteria': dict(zip(labels, row['question']['choices'])),
            }}})
            cases.append({'name': row['id'], 'language': row['language'], 'gold': labels[row['gold']]})
        (args.output / 'dataset.json').write_text(json.dumps({
            'source': 'https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip',
            'input_sha256': digest,
            'input_format': 'official archive' if args.xnli.suffix == '.zip' else 'JSONL',
            'available_pairs': len(ids),
            'selection': 'midpoints of equal-width intervals of sorted available pair IDs',
            'pairs': args.pairs, 'languages': args.languages.split(','),
            'scope': 'small labeled quantization audit; translated pairs are correlated; not a leaderboard',
        }, indent=2) + '\n')
    write_lines(args.output / 'requests.jsonl', requests)
    (args.output / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')


def run(args):
    assert args.rounds > 0
    assert args.repetitions > 0
    binary = args.output / 'decisive'
    assert not binary.exists(), 'use a fresh run directory'
    shutil.copy2(args.binary, binary)
    manifest = {
        'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'threads': args.threads, 'rounds': args.rounds, 'repetitions': args.repetitions,
        'requests_sha256': hashlib.sha256((args.output / 'requests.jsonl').read_bytes()).hexdigest(),
        'warmup': 'first complete round excluded when rounds > 1',
        'machine': {'architecture': platform.machine(), 'os': platform.system(),
                    'logical_cpus': os.cpu_count(),
                    'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None},
        'load_average_start': os.getloadavg() if hasattr(os, 'getloadavg') else None,
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path('src/models/pplx_flash').glob('*.rs'))},
    }
    (args.output / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'capabilities.json').write_text(subprocess.check_output([str(binary), 'capabilities', '--json'], text=True))
    for threads in args.threads:
        for repetition in range(args.repetitions):
            modes = ['bf16', 'int8'] if repetition % 2 == 0 else ['int8', 'bf16']
            for mode in modes:
                target = args.output / f'{mode}-t{threads}-r{repetition}.jsonl'
                command = [str(binary), 'pplx-flash', '--threads', str(threads), '--resident',
                           '--quantization', mode, 'evaluate', '--input', str(args.output / 'requests.jsonl'),
                           '--jsonl', '--rounds', str(args.rounds)]
                print(' '.join(command), flush=True)
                with target.open('x') as file:
                    subprocess.run(command, stdout=file, check=True)
    summarize(args)


def summarize(args):
    cases = json.loads((args.output / 'cases.json').read_text())
    runs = {}
    for path in sorted(args.output.glob('*-t*-r*.jsonl')):
        mode = path.name.split('-')[0]
        for line in path.read_text().splitlines():
            record = json.loads(line)
            assert record['metrics']['precision'] == ('int8-w8a8-g64' if mode == 'int8' else 'bf16-storage-f32-compute')
            runs.setdefault((mode, record['threads'], record['case']), []).append(record)
    tables = []
    quality = []
    for threads in sorted({key[1] for key in runs}):
        for index, case in enumerate(cases):
            modes = {}
            for mode in ['bf16', 'int8']:
                records = runs[(mode, threads, index)]
                assert all(r['response'] == records[0]['response'] for r in records), 'repeated inference changed outputs'
                measured = [r for r in records if r['round'] > 0] or records
                samples = [r['metrics']['elapsed_ms'] for r in measured]
                modes[mode] = {
                    'median_ms': statistics.median(samples), 'samples_ms': samples,
                    'load_ms': [r['model_load_ms'] for r in records if r['round'] == 0],
                    'packed_projection_bytes': records[0]['metrics']['packed_projection_bytes'],
                }
            tables.append({'case': case['name'], 'threads': threads,
                           'tokens': runs[('bf16', threads, index)][0]['response']['usage']['input_tokens'],
                           'bf16': modes['bf16'], 'int8': modes['int8'],
                           'speedup': modes['bf16']['median_ms'] / modes['int8']['median_ms']})
            reference = runs[('bf16', threads, index)][0]['response']
            quantized = runs[('int8', threads, index)][0]['response']
            assert reference['usage'] == quantized['usage'], 'input/sequence semantics differ'
            delta = []
            changes = []
            def compare(a, b, name=''):
                if isinstance(a, dict):
                    assert a.keys() == b.keys()
                    for key in a: compare(a[key], b[key], f'{name}.{key}')
                elif isinstance(a, (int, float)) and not isinstance(a, bool):
                    assert math.isfinite(a) and math.isfinite(b)
                    delta.append({'field': name, 'absolute_error': abs(a - b)})
                elif a != b:
                    changes.append({'field': name, 'bf16': a, 'int8': b})
            compare(reference['answers'], quantized['answers'])
            item = {'case': case['name'], 'threads': threads, 'differences': delta, 'decision_changes': changes}
            if 'gold' in case:
                item.update(language=case['language'], gold=case['gold'],
                            bf16_correct=reference['answers']['relation']['choice'] == case['gold'],
                            int8_correct=quantized['answers']['relation']['choice'] == case['gold'])
                for mode, response in [('bf16', reference), ('int8', quantized)]:
                    probabilities = response['answers']['relation']['probabilities']
                    item[f'{mode}_log_loss'] = -math.log(max(probabilities[case['gold']], 1e-300))
                    item[f'{mode}_brier'] = sum((v - (key == case['gold'])) ** 2 for key, v in probabilities.items())
            quality.append(item)
    result = {'performance': tables, 'quality': quality,
              'max_absolute_numeric_error': max((d['absolute_error'] for q in quality for d in q['differences']), default=0),
              'decision_changes': sum(len(q['decision_changes']) for q in quality),
              'quality_note': 'Labeled sample accuracy is reported separately from numerical differences. No automatic production-quality approval.'}
    result['max_probability_error'] = max((d['absolute_error'] for q in quality for d in q['differences']
                                          if '.probabilities.' in d['field'] or d['field'].endswith('.noul')), default=0)
    result['max_score_error'] = max((d['absolute_error'] for q in quality for d in q['differences']
                                    if d['field'].endswith('.score')), default=0)
    labeled = [q for q in quality if 'gold' in q]
    result['labeled'] = {str(t): {'cases': len(rows), 'bf16_correct': sum(q['bf16_correct'] for q in rows),
                                 'int8_correct': sum(q['int8_correct'] for q in rows),
                                 'regressions': sum(q['bf16_correct'] and not q['int8_correct'] for q in rows),
                                 'improvements': sum(q['int8_correct'] and not q['bf16_correct'] for q in rows),
                                 **{key: sum(q[key] for q in rows) / len(rows)
                                    for key in ['bf16_log_loss', 'int8_log_loss', 'bf16_brier', 'int8_brier']}}
                         for t in sorted({q['threads'] for q in labeled})
                         if (rows := [q for q in labeled if q['threads'] == t])}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['max_probability_error', 'max_score_error', 'decision_changes', 'labeled']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run paired native BF16/INT8 benchmarks and labeled XNLI quality comparisons.\n\nInference runs in the Rust CLI. This script only prepares requests and reports.\n')
    parser.add_argument('action', choices=['prepare', 'run', 'summarize'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('target/release/decisive'))
    parser.add_argument('--xnli', type=Path)
    parser.add_argument('--pairs', type=int, default=12)
    parser.add_argument('--languages', default='en,fr,de,zh,ar')
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 3])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--lengths', nargs='+', type=int, default=[])
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'summarize': summarize}[args.action](args)
