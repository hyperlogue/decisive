import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from cpu_bench import MODES, PRECISIONS, compare_mae, digest, read_runs, summarize


class CpuReports(unittest.TestCase):
    def test_selected_modes_summary_counts_flips_and_numeric_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [{'id': str(i), 'family': 'stress',
                      'request': {'state': 'test', 'questions': {'refund': {'type': 'noul'}}},
                      'expected': {'refund': {'label': label}}} for i, label in enumerate(['true', 'false'])]
            corpus = root / 'corpus.json'
            corpus.write_text(json.dumps({'cases': cases, 'sources': []}))
            (root / 'decisive').write_bytes(b'test binary identity')
            requests = root / 'requests.jsonl'
            requests.write_text(''.join(json.dumps(c['request'])+'\n' for c in cases))
            (root / 'all-manifest.json').write_text(json.dumps({
                'binary_sha256': digest(root / 'decisive'), 'corpus_sha256': digest(corpus),
                'requests_sha256': digest(requests), 'cases': ['0', '1'], 'modes': MODES, 'threads': 3}))
            for mode in MODES:
                rows = []
                for i in range(2):
                    p = .9 if i == 0 else .1
                    if mode == 'q4-all':
                        p = 1-p
                    rows.append({'case': i, 'round': 0, 'threads': 3,
                                 'response': {'usage': {'sequences': 1}, 'answers': {'refund': {'type': 'noul', 'noul': p}}},
                                 'metrics': {'precision': PRECISIONS[mode], 'revision': 'test-checkpoint',
                                             'weight_loading': 'resident packed decoder weights', 'layers': [{}]*32,
                                             'logits': {'refund': [0., -1. if mode == 'q4-all' else 1.]},
                                             'packed_projection_bytes': 16, 'embedding_bytes': 8, 'readout_bytes': 4}})
                (root / (mode+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with redirect_stdout(io.StringIO()):
                summarize(SimpleNamespace(corpus=corpus, output=root, modes=MODES))
            result = json.loads((root / 'summary.json').read_text())
            self.assertEqual(result['primary_metrics']['bf16']['accuracy'], 1)
            self.assertEqual(result['primary_metrics']['q4-all']['accuracy'], 0)
            self.assertEqual(next(c for c in result['primary_comparisons'] if c['candidate'] == 'q4-all')['decision_flips'], 2)
            self.assertAlmostEqual(result['errors']['q4-all']['probability']['mae'], .8)
            self.assertAlmostEqual(result['errors']['q4-all']['logit']['rmse'], 2**.5)
            self.assertIn('W4A4', (root / 'report.html').read_text())
            self.assertIn('Quantizer ablations', (root / 'report.html').read_text())
            for name in ['summary.json', 'report.html']:
                (root / name).unlink()
            with redirect_stdout(io.StringIO()):
                summarize(SimpleNamespace(corpus=corpus, output=root, modes=['bf16', 'q4-all']))
            selected = json.loads((root / 'summary.json').read_text())
            self.assertEqual(set(selected['primary_metrics']), {'bf16', 'q4-all'})

    def test_mae_bootstrap_weights_probability_entries_and_clusters_questions(self):
        rows = [{'id': str(i), 'probabilities': {'bf16': {'x': .8, 'y': .2},
                 'a': {'x': .75, 'y': .25}, 'b': {'x': .6, 'y': .4}}}
                for i in [0, 0, 1]]
        result = compare_mae(rows, 'a', 'b')
        self.assertEqual(result['probability_entries'], 6)
        self.assertAlmostEqual(result['candidate_mae'], .05)
        self.assertAlmostEqual(result['reference_mae'], .2)
        self.assertAlmostEqual(result['relative_mae_reduction'], .75)
        for bound in result['paired_input_bootstrap_95_ci']:
            self.assertAlmostEqual(bound, -.15)

    def test_rejects_missing_duplicate_misidentified_and_nonfinite_runs(self):
        cases = [{'request': {'questions': {'answer': {'type': 'noul'}}}}] * 2
        row = {'round': 0, 'case': 0,
               'response': {'answers': {'answer': {'type': 'noul', 'noul': .6}},
                            'usage': {'sequences': 1}},
               'metrics': {'precision': PRECISIONS['q4-all'],
                           'weight_loading': 'resident packed decoder weights',
                           'layers': [{}] * 32, 'logits': {'answer': [0., 1.]}}}
        records = [row, dict(row, case=1)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs.jsonl'
            def check(records):
                path.write_text(''.join(json.dumps(r)+'\n' for r in records))
                return read_runs(path, cases, 'q4-all')
            self.assertEqual(len(check(records)), 2)
            bad_sets = [[], records[:1], [row, row]]
            for key, value in [('precision', PRECISIONS['q4']), ('layers', []),
                               ('logits', {'answer': [float('nan')]}),
                               ('weight_loading', 'mmap')]:
                bad = copy.deepcopy(records)
                bad[1]['metrics'][key] = value
                bad_sets.append(bad)
            repeated = copy.deepcopy(records) + [dict(copy.deepcopy(r), round=1) for r in records]
            self.assertEqual(len(check(repeated)), 4)
            # A constant logit shift can leave probabilities unchanged; the
            # numerical audit must still reject nondeterministic repeated logits.
            repeated[2]['metrics']['logits']['answer'] = [.1, 1.1]
            bad_sets.append(repeated)
            for bad in bad_sets:
                with self.assertRaises(AssertionError):
                    check(bad)


if __name__ == '__main__':
    unittest.main()
