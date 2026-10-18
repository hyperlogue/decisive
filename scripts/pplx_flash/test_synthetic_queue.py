import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


class QueueCollectorTests(unittest.TestCase):
    def test_checkpoint_rejections_and_final_report(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            page = directory / 'page.html'
            page.write_text('test')
            output = directory / 'queue.json'
            process = subprocess.Popen([sys.executable, str(ROOT/'scripts/pplx_flash/serve_quantization.py'),
                '--page', str(page), '--output', str(output), '--timeout', '20',
                '--experiment', 'pplx-synthetic-queue-v1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                url = json.loads(process.stdout.readline())['url']
                parts = urlsplit(url)
                origin = f'{parts.scheme}://{parts.netloc}'
                token = parts.path[1:]

                def post(route, value):
                    request = Request(f'{origin}/{route}/{token}', json.dumps(value).encode(),
                        {'Origin': origin, 'Content-Type': 'application/json'})
                    with urlopen(request, timeout=5) as response:
                        self.assertEqual(response.status, 200)

                child = {'experiment': 'pplx-synthetic-full-v1', 'completed': True, 'runs': []}
                checkpoint = {'message': 'completed phase', 'phase_report': {'index': 1, 'report': child}}
                post('progress', checkpoint)
                self.assertEqual(json.loads((directory/'queue.phase-01.json').read_text()), child)
                post('progress', checkpoint)
                checkpoint['phase_report']['report'] = {**child, 'completed': False}
                with self.assertRaises(HTTPError) as failure:
                    post('progress', checkpoint)
                failure.exception.close()
                report = {'experiment': 'pplx-synthetic-queue-v1', 'completed': True,
                          'expected_phases': 2, 'phases': [{'id': 'first', 'report': child}]}
                with self.assertRaises(HTTPError) as failure:
                    post('results', report)
                failure.exception.close()
                self.assertFalse(output.exists())
                report['phases'].append({'id': 'second', 'report': child})
                post('results', report)
                self.assertEqual(process.wait(timeout=5), 0)
                result = json.loads(output.read_text())
                self.assertEqual(result['saved_phase_reports'], ['queue.phase-01.json', 'queue.phase-02.json'])
                self.assertEqual(json.loads((directory/'queue.phase-02.json').read_text()), child)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                process.stdout.close()
                process.stderr.close()
