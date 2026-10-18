import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]


def analyze(path):
    report = json.loads(path.read_text())
    assert report['experiment'] == 'pplx-synthetic-full-v1' and report['completed'], report.get('error')
    assert not report.get('reloaded_during_run'), 'Browser reloaded during measurement'
    assert report.get('external_validation', {}).get('timing_valid', True), 'Externally invalidated run'
    configurations = {e['id']: e['configuration'] for e in report['engines']}
    for engine in report['engines']:
        if engine['configuration'].get('int8'):
            assert engine['int8_check']['passed'], 'Missing INT8 kernel validation'
    methodology = report['methodology']
    expected = len(methodology['lengths']) * len(report['engines']) * (methodology['warmups'] + methodology['rounds'])
    assert len(report['runs']) == expected
    for run in report['runs']:
        assert len(run['layers']) == 32


        quantized_bytes = 0
        if configurations[run['id']].get('int8'):
            mlp = [(9216, 2560), (9216, 2560), (2560, 9216)]
            if configurations[run['id']].get('fp16Mlp'):
                mlp = [(2560, 9216)]
            delta = [(8192, 2560), (4096, 2560), (2560, 4096)] + mlp
            if configurations[run['id']].get('pureInt8'):
                delta += [(32, 2560), (32, 2560)]
            attention = [(8192, 2560), (1024, 2560), (1024, 2560), (2560, 4096)] + mlp
            quantized_bytes = 24*sum(n*(k+8) for n,k in delta) + 8*sum(n*(k+8) for n,k in attention)
        assert sum(layer['weight_bytes'] for layer in run['layers']) == 7140099072 + quantized_bytes
        for index, layer in enumerate(run['layers']):
            assert layer['layer'] == index and layer['finite']
            assert layer['hidden_values'] == (2 if index == 31 else run['rows']) * 2560
            assert math.isfinite(layer['hidden_max_abs']) and math.isfinite(layer['hidden_rms'])
            profile = layer.get('profile')
            if profile:
                assert len(profile['dispatches']) == layer['dispatches']
                assert profile['span_ms'] > 0
                if configurations[run['id']].get('pureInt8'):
                    assert not any(d['kernel'] in ['gemm', 'mlp_fused', 'gemm_small', 'mlp_small'] for d in profile['dispatches']), 'Pure INT8 dispatched an FP16 projection'
                assert all(math.isfinite(d['gpu_ms']) and d['gpu_ms'] >= 0 for d in profile['dispatches'])
        output = run['output']
        assert output['completed_layers'] == 32 and output['finite']
        for key in ['logits', 'probabilities']:
            assert len(output[key]) == 2
            assert all(len(row) == 255 and all(math.isfinite(x) for x in row) for row in output[key])
        assert all(abs(sum(row) - 1) < 1e-5 and all(0 <= x <= 1 for x in row) for row in output['probabilities'])
        execution = run['embedding_ms'] + sum(l['compute_ms'] for l in run['layers']) + output['readout_ms']
        assert abs(execution - run['execution_ms']) < 1e-5

    result = {'source': str(path.relative_to(ROOT)), 'verified_runs': expected,
              'synthetic_only': True, 'layer_fingerprint_audit': report.get('layer_fingerprint_audit', False),
              'input_dependency_experiment': report.get('input_dependency_experiment', False), 'hidden_timings_provisional': report['measured_runs_hidden'], 'cases': []}
    for case in report['cases']:
        runs = [r for r in report['runs'] if r['rows'] == case['rows'] and not r['warmup']]
        baseline = [r for r in runs if r['id'] == report['baseline_id']]
        entry = {'rows': case['rows'], 'variants': {}}
        for name in case['variants']:
            samples = [r for r in runs if r['id'] == name]
            times = [r['execution_ms'] for r in samples]
            center = median(times)
            identical = all(r['output']['logits'] == next(b for b in baseline if b['round'] == r['round'])['output']['logits'] for r in samples)
            same_precision = report.get('configuration_mode') == 'safariTune' or not configurations[name].get('int8') or (case['rows'] <= 8 and not configurations[name].get('pureInt8'))
            if report['baseline_id'] in ['fp16-baseline','tuning-baseline'] and same_precision:
                assert identical, f'{name}: changed full-model logits'
            layer_identical = None
            if report.get('layer_fingerprint_audit'):
                layer_identical = all(
                    all(a['dependency'] == b['dependency'] for a, b in zip(r['layers'], next(b for b in baseline if b['round'] == r['round'])['layers']))
                    for r in samples)
                if same_precision:
                    assert layer_identical, f'{name}: changed layer fingerprint'
            variant = {'execution_median_ms': center, 'samples_ms': times,
                       'relative_mad': median(abs(t-center) for t in times)/center,
                       'speedup': None if report.get('input_dependency_experiment') else median(r['execution_ms'] for r in baseline)/center,
                       'identical_layer_fingerprints': layer_identical,
                       'identical_logits': identical,
                       'repeatable_logits': all(r['output']['logits'] == samples[0]['output']['logits'] for r in samples),
                       'paired_wins': sum(r['execution_ms'] < next(b for b in baseline if b['round'] == r['round'])['execution_ms'] for r in samples),
                       'dispatches': [sum(l['dispatches'] for l in r['layers']) for r in samples],
                       'allocations': [sum(l['scratch_allocations'] for l in r['layers']) for r in samples],
                       'max_retained_scratch_bytes': max(l['retained_scratch_bytes'] for r in samples for l in r['layers'])}
            paired = [p for p in case['paired'] if p['id'] == name]
            if paired:
                variant['max_logit_absolute_difference'] = max(p['logits']['max_absolute_error'] for p in paired)
                variant['max_logit_relative_l2'] = max(p['logits']['relative_l2'] for p in paired)
                variant['argmax_changes'] = [p['argmax_changes'] for p in paired]
            if report.get('profile_enabled'):
                sums = defaultdict(list)
                projection_shapes = defaultdict(list)
                layer_spans = defaultdict(list)
                projection_names = {'gemm', 'gemm_small', 'mlp_fused', 'mlp_small',
                                    'int8_gemm', 'int8_gemm_small', 'int8_mlp', 'int8_mlp_small'}
                for run in samples:
                    kernels = defaultdict(float)
                    shapes = defaultdict(float)
                    for layer in run['layers']:
                        layer_spans[layer['layer']].append(layer['profile']['span_ms'])
                        for dispatch in layer['profile']['dispatches']:
                            kernels[dispatch['kernel']] += dispatch['gpu_ms']
                            if dispatch['kernel'] in projection_names:
                                shapes[(dispatch['kernel'], *dispatch['params'][:3])] += dispatch['gpu_ms']
                    for kernel, value in kernels.items():
                        sums[kernel].append(value)
                    for shape, value in shapes.items():
                        projection_shapes[shape].append(value)
                variant['kernel_interval_medians_ms'] = {k: median(v) for k,v in sorted(sums.items(), key=lambda item: -median(item[1]))}
                variant['projection_shape_interval_medians'] = [
                    {'kernel': key[0], 'rows': key[1], 'output': key[2], 'input': key[3],
                     'gpu_ms': median(values)}
                    for key, values in sorted(projection_shapes.items(), key=lambda item: -median(item[1]))]
                variant['layer_gpu_span_medians_ms'] = [median(layer_spans[i]) for i in range(32)]
                variant['profile_note'] = 'Instrumented pass intervals may overlap; their sum is not decoder latency or exclusive time. Use them to rank candidates, then verify uninstrumented.'
            entry['variants'][name] = variant
        result['cases'].append(entry)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    summary = [analyze(path.resolve()) for path in args.reports]
    with args.output.open('x') as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'verified_reports': len(summary), 'verified_runs': sum(s['verified_runs'] for s in summary)}))
