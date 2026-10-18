import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BOUND = 1e-4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def package(path, kernels_only):
    build = read(path / 'build.json')
    required = {'pplx-flash', 'wasm', 'webgpu', 'kernel-checks'}
    require(required <= set(build['features'].split(',')),
            'Build with DECISIVE_PPLX_CHECKS=1; all WebGPU runs require kernel-checks')
    require((path / 'decisive.js').is_file(), 'Missing generated decisive.js')
    digest = hashlib.sha256((path / 'decisive_bg.wasm').read_bytes()).hexdigest()
    require(digest == build['wasm_sha256'], 'Wasm does not match build.json')
    if not kernels_only:
        manifest = read(ROOT / 'examples/pplx-flash/model.json')
        model = ROOT / 'models/pplx-decider-flash-v1.2'
        for name, entry in manifest['files'].items():
            file = model / name
            require(file.is_file() and file.stat().st_size == entry['bytes'],
                    f'Missing or incomplete model file: {name}. Run scripts/pplx_flash/download.sh, '
                    'or use --kernels-only')


def adapter(report):
    require(not report.get('error'), f"Adapter probe failed: {report.get('error')}")
    require(report.get('available') is True and report.get('compute_readback_passed') is True,
            'WebGPU adapter/compute unavailable; inspect adapter.json')
    require(isinstance(report.get('adapter'), dict) and report['adapter'], 'Missing adapter identity')


def compare(actual, expected, path='response'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(),
                f'{path}: response keys differ')
        return max((compare(actual[k], v, f'{path}.{k}') for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'{path}: shape differs')
        return max((compare(a, b, f'{path}[{i}]') for i, (a, b) in enumerate(zip(actual, expected))), default=0.)
    if type(expected) in (float, int):
        require(finite(actual), f'{path}: missing or non-finite numeric output')
        error = abs(actual - expected)
        require(error <= BOUND, f'{path}: output error {error} exceeds {BOUND}')
        return error
    require(type(actual) is type(expected) and actual == expected, f'{path}: decision/value differs')
    return 0.


def results(root, kernels_only):
    probe = read(root / 'adapter.json')
    adapter(probe)
    build = read(root / 'build.json')
    kernel_runs = [read(root / f'kernels-worker-{worker}.json') for worker in (0, 1)]
    for worker, run in enumerate(kernel_runs):
        require(not run.get('error') and run.get('checks'), f'Kernel checks failed in worker={worker}')
        require(run.get('worker') is bool(worker), 'Kernel execution context differs')
        require(run.get('wasm_sha256') == build['wasm_sha256'] and
                run.get('features') == build['features'], 'Kernel build provenance differs')
        require(run.get('bound') == BOUND, 'Unexpected kernel tolerance')
        require(isinstance(run.get('adapter'), dict) and run['adapter'], 'Missing kernel adapter')
        for check in run['checks']:
            require(isinstance(check.get('name'), str) and check['name'], 'Missing kernel name')
            require(type(check.get('values')) is int and check['values'] > 0, 'Empty kernel check')
            error = check.get('max_error')
            require(finite(error) and 0 <= error <= BOUND, f"Invalid kernel error: {check['name']}")
    require([(c['name'], c['values']) for c in kernel_runs[0]['checks']] ==
            [(c['name'], c['values']) for c in kernel_runs[1]['checks']], 'Kernel coverage differs between contexts')
    errors = []
    if not kernels_only:
        reference = read(ROOT / 'tests/fixtures/pplx_flash/reference.json')
        for fixture in ('request', 'multilingual'):
            run = read(root / f'{fixture}.json')
            require(not run.get('error') and run.get('backend') == 'webgpu', f'{fixture}: WebGPU run failed')
            require(run.get('fixture') == fixture, f'{fixture}: wrong fixture')
            require(run.get('wasm_sha256') == build['wasm_sha256'] and
                    run.get('features') == build['features'], f'{fixture}: build provenance differs')
            request = read(ROOT / f'tests/fixtures/pplx_flash/{fixture}.json')
            expected = next(c['response'] for c in reference['cases'] if c['request'] == request)
            errors.append(compare(run.get('response'), expected))
    info = probe['adapter']
    software = (probe.get('software_gpu_requested') is True or info.get('is_fallback_adapter') is True
                or any(name in json.dumps(info).lower() for name in ('swiftshader', 'llvmpipe', 'software')))
    summary = {
        'mode': 'kernels-only' if kernels_only else 'full',
        'kernel_contexts': 2,
        'kernel_checks_per_context': len(kernel_runs[0]['checks']),
        'max_absolute_kernel_error': max(c['max_error'] for r in kernel_runs for c in r['checks']),
        'cases': len(errors),
        'max_absolute_output_error': max(errors) if errors else None,
        'bound': BOUND,
        'software_gpu': software,
        'adapter': info,
        'wasm_sha256': build['wasm_sha256'],
        'scope': 'kernel correctness only; full-model parity not run' if kernels_only else
                 'synthetic reference parity, not a task-quality benchmark',
        'timing': 'wall-clock including layer streaming; software adapters are correctness checks only',
    }
    (root / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    try:
        action, path, *options = sys.argv[1:]
        kernels_only = options == ['--kernels-only']
        require(options in ([], ['--full'], ['--kernels-only']), 'Invalid verification mode')
        if action == 'package':
            package(Path(path), kernels_only)
        elif action == 'adapter':
            adapter(read(Path(path)))
        elif action == 'results':
            print(json.dumps(results(Path(path), kernels_only), indent=2))
        else:
            raise ValueError('Expected package, adapter or results')
    except (ValueError, KeyError, TypeError, OSError, StopIteration) as error:
        sys.exit(f'WebGPU verification failed: {error}')
