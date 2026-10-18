import copy
import unittest
from export_synthetic_trace import trace


class TraceTests(unittest.TestCase):
    def report(self):
        return {'completed': True, 'profile_enabled': True, 'scope': 'test', 'runs': [
            {'id': 'test', 'rows': 117, 'round': 0, 'layers': [
                {'layer': i, 'profile': {'origin_ticks': str(10**18+i*2_000_000),
                    'tick_period_ns': 1, 'dispatches': [{'kernel': 'gemm', 'params': [117, 9216, 2560],
                    'start_ms': 0.1, 'end_ms': 0.6, 'gpu_ms': 0.5}]}}
                for i in range(2)]}]}

    def test_preserves_gpu_gap_and_large_integer_clock(self):
        events = [e for e in trace(self.report())['traceEvents'] if e['ph'] == 'X']
        self.assertEqual([e['ts'] for e in events], [100, 2100])
        self.assertEqual([e['dur'] for e in events], [500, 500])
        self.assertEqual(events[0]['args']['params'], [117, 9216, 2560])

    def test_rejects_invalid_clock_or_interval(self):
        original = self.report()
        for field, value in [('origin_ticks', '0'), ('tick_period_ns', 2)]:
            report = copy.deepcopy(original)
            report['runs'][0]['layers'][1]['profile'][field] = value
            with self.assertRaises(ValueError):
                trace(report)
        original['runs'][0]['layers'][0]['profile']['dispatches'][0]['end_ms'] = 0
        with self.assertRaises(ValueError):
            trace(original)

    def test_rejects_unprofiled_or_incomplete(self):
        for flag in ['completed', 'profile_enabled']:
            report = self.report()
            report[flag] = False
            with self.assertRaises(ValueError):
                trace(report)
