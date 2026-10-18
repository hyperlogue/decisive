import argparse
import json
import math
from pathlib import Path


def trace(report):
    if report.get('experiment')=='pplx-checkpoint-phase-v1':
        report={**report,'profile_enabled':report['configuration'].get('profile',False),
                'scope':('Actual checkpoint warm-request GPU passes; one-time weight preparation is excluded. Runtime Q4 expansion is included.'
                         if report.get('timing_scope')=='warm-request' else
                         'Actual checkpoint streaming diagnostic; local weight reads, uploads and preparation appear between compute intervals.'),
                'runs':[{**r,'id':r['mode'],'rows':r['response']['usage']['input_tokens']} for r in report['runs']]}
    if not report.get('completed') or not report.get('profile_enabled'):
        raise ValueError('A completed GPU-profiled report is required')
    events = []
    for pid, run in enumerate(report['runs'], 1):
        layers = run['layers']
        origin = int(layers[0]['profile']['origin_ticks'])
        period = layers[0]['profile']['tick_period_ns']
        if not math.isfinite(period) or period <= 0:
            raise ValueError('Invalid timestamp period')
        events.append({'ph': 'M', 'pid': pid, 'name': 'process_name',
                       'args': {'name': f"{run['id']} / {run['rows']} tokens / round {run['round']}"}})
        previous = origin
        for layer in layers:
            profile = layer['profile']
            at = int(profile['origin_ticks'])
            if at < previous or profile['tick_period_ns'] != period:
                raise ValueError('GPU clock is inconsistent across layers')
            previous = at
            tid = layer['layer']
            events.append({'ph': 'M', 'pid': pid, 'tid': tid, 'name': 'thread_name',
                           'args': {'name': f'Layer {tid}'}})
            offset_us = (at-origin)*period/1000
            for dispatch in profile['dispatches']:
                start, end = dispatch['start_ms'], dispatch['end_ms']
                if not all(math.isfinite(v) for v in [start, end]) or end < start:
                    raise ValueError('Invalid GPU interval')
                events.append({'ph': 'X', 'pid': pid, 'tid': tid, 'cat': 'WebGPU pass',
                               'name': dispatch['kernel'], 'ts': offset_us+start*1000,
                               'dur': (end-start)*1000,
                               'args': {'layer': tid, 'params': dispatch['params'],
                                        'gpu_interval_ms': dispatch['gpu_ms']}})
    return {'traceEvents': events, 'displayTimeUnit': 'ms',
            'metadata': {'scope': report['scope'], 'alignment': 'Each run starts at its own first GPU timestamp; clocks between different runs/devices are not aligned.',
                         'interpretation': 'Actual per-layer GPU clock gaps are preserved. Pass intervals can overlap; their sum is not exclusive compute time. Preparation and audits appear as gaps, not individual pass events.'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = trace(json.loads(args.report.read_text()))
    with args.output.open('x') as output:
        json.dump(result, output, separators=(',', ':'), allow_nan=False)
        output.write('\n')
    print(json.dumps({'trace_events': len(result['traceEvents']), 'output': str(args.output)}))
