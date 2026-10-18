import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import verify_webgpu as check


class WebGpuVerificationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.build = {'features': 'pplx-flash,wasm,webgpu,kernel-checks',
                      'wasm_sha256': hashlib.sha256(b'wasm').hexdigest()}
        self.write('build.json', self.build)
        self.write('adapter.json', {'available': True, 'compute_readback_passed': True,
                                   'adapter': {'description': 'test adapter'}})
        for worker in (0, 1):
            self.write(f'kernels-worker-{worker}.json', {
                **self.build,
                'worker': bool(worker), 'bound': check.BOUND, 'adapter': {'name': 'test adapter'},
                'checks': [{'name': 'test kernel', 'values': 4, 'max_error': 0.}],
            })

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def full_reports(self):
        reference = check.read(check.ROOT / 'tests/fixtures/pplx_flash/reference.json')
        for name in ('request', 'multilingual'):
            request = check.read(check.ROOT / f'tests/fixtures/pplx_flash/{name}.json')
            response = next(c['response'] for c in reference['cases'] if c['request'] == request)
            self.write(f'{name}.json', {**self.build, 'backend': 'webgpu', 'fixture': name,
                                      'response': response, 'reference_max_error': 0.})

    def test_kernel_only_does_not_claim_full_model_parity(self):
        result = check.results(self.root, True)
        self.assertEqual(result['cases'], 0)
        self.assertIsNone(result['max_absolute_output_error'])
        self.assertEqual(result['kernel_contexts'], 2)

    def test_missing_worker_report_fails(self):
        (self.root / 'kernels-worker-1.json').unlink()
        with self.assertRaises(FileNotFoundError):
            check.results(self.root, True)

    def test_invalid_kernel_reports_fail(self):
        original = check.read(self.root / 'kernels-worker-1.json')
        for value in (float('nan'), float('inf'), -1, 1e-3, True, None):
            with self.subTest(value=value):
                report = copy.deepcopy(original)
                report['checks'][0]['max_error'] = value
                self.write('kernels-worker-1.json', report)
                with self.assertRaises(ValueError):
                    check.results(self.root, True)
        for field, value in [('worker', False), ('checks', []), ('error', 'device lost')]:
            with self.subTest(field=field):
                self.write('kernels-worker-1.json', {**original, field: value})
                with self.assertRaises(ValueError):
                    check.results(self.root, True)

    def test_no_adapter_fails(self):
        with self.assertRaises(ValueError):
            check.adapter({'available': False})

    def test_software_adapter_is_labelled_without_explicit_flag(self):
        self.write('adapter.json', {'available': True, 'compute_readback_passed': True,
                                   'adapter': {'description': 'SwiftShader'}})
        self.assertTrue(check.results(self.root, True)['software_gpu'])

    def test_full_model_is_compared_independently(self):
        self.full_reports()
        self.assertEqual(check.results(self.root, False)['cases'], 2)
        report = check.read(self.root / 'request.json')
        report['response']['answers']['team']['probabilities']['support'] = 0.
        self.write('request.json', report)
        with self.assertRaisesRegex(ValueError, 'output error'):
            check.results(self.root, False)

    def test_extra_keys_and_changed_decisions_fail(self):
        for actual, expected in [({'x': 1, 'extra': 0}, {'x': 1}),
                                 ({'choice': 'sales'}, {'choice': 'support'}),
                                 ({'x': None}, {'x': 1}), ({'x': True}, {'x': 1})]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                check.compare(actual, expected)

    def test_wrong_build_fails(self):
        self.full_reports()
        report = check.read(self.root / 'request.json')
        report['wasm_sha256'] = 'wrong build'
        self.write('request.json', report)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            check.results(self.root, False)

    def test_package_requires_checks_and_matching_hash(self):
        (self.root / 'decisive_bg.wasm').write_bytes(b'wasm')
        (self.root / 'decisive.js').write_text('')
        check.package(self.root, True)
        (self.root / 'decisive_bg.wasm').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            check.package(self.root, True)
        self.write('build.json', {**self.build, 'features': 'pplx-flash,wasm,webgpu'})
        with self.assertRaisesRegex(ValueError, 'DECISIVE_PPLX_CHECKS'):
            check.package(self.root, True)


if __name__ == '__main__':
    unittest.main()
