import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import tempfile
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]


def write_new_json(path, value):
    serialized = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as file:
        temporary = Path(file.name)
        file.write(serialized)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def serve(page, output, timeout, experiment='pplx-weight-only-quantization-v1'):
    html = page.read_bytes()
    if output.exists():
        raise ValueError('Choose a new report filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    result_path = f'/results/{token}'
    saved = False
    saved_phases = set()

    def save_phase(index, child):
        if (type(index) is not int or not 1 <= index <= 16 or not isinstance(child, dict)
                or child.get('experiment') != 'pplx-synthetic-full-v1'
                or not isinstance(child.get('completed'), bool)):
            raise ValueError('Invalid phase report')
        child_path = output.with_name(f'{output.stem}.phase-{index:02}.json')
        if child_path.exists():
            if json.loads(child_path.read_text()) != child:
                raise ValueError('Conflicting phase checkpoint')
        else:
            write_new_json(child_path, child)
        saved_phases.add(index)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_):
            pass

        def do_GET(self):
            if urlsplit(self.path).path != f'/{token}':
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        def do_POST(self):
            nonlocal saved
            progress = self.path == f'/progress/{token}'
            if (self.path != result_path and not progress) or self.headers.get('Origin') != origin:
                self.send_error(403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                limit = (32 if experiment == 'pplx-synthetic-queue-v1' else 2) * 1024 * 1024
                if not 0 < size <= limit:
                    raise ValueError('Invalid report size')
                report = json.loads(self.rfile.read(size))
                if progress:
                    if not isinstance(report.get('message'), str) or len(report['message']) > 1000:
                        raise ValueError('Invalid progress message')
                    checkpoint = report.get('phase_report')
                    if checkpoint is not None:
                        if experiment != 'pplx-synthetic-queue-v1' or not isinstance(checkpoint, dict):
                            raise ValueError('Unexpected phase checkpoint')
                        save_phase(checkpoint.get('index'), checkpoint.get('report'))
                    print(json.dumps({'progress': report['message']}), flush=True)
                    self.send_response(200)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                if report.get('experiment') != experiment or not isinstance(report.get('completed'), bool):
                    raise ValueError('Unexpected report schema')
                if experiment == 'pplx-synthetic-queue-v1':
                    phases = report.get('phases')
                    expected = report.get('expected_phases')
                    if (not isinstance(phases, list) or len(phases) > 16
                            or type(expected) is not int or not 1 <= expected <= 16):
                        raise ValueError('Invalid queue report')
                    if report['completed'] and (len(phases) != report.get('expected_phases') or not all(p.get('report', {}).get('completed') for p in phases)):
                        raise ValueError('Incomplete queue marked complete')
                    for phase in phases:
                        child = phase.get('report', {})
                        if child.get('experiment') != 'pplx-synthetic-full-v1' or not isinstance(child.get('completed'), bool):
                            raise ValueError('Invalid phase report')
                    for index, phase in enumerate(phases):
                        save_phase(index+1, phase['report'])
                    report['saved_phase_reports'] = [f'{output.stem}.phase-{i:02}.json' for i in sorted(saved_phases)]
                write_new_json(output, report)
            except (ValueError, TypeError, AttributeError, OSError) as error:
                self.send_error(400, str(error))
                return
            self.send_response(200)
            self.send_header('Content-Length', '0')
            self.end_headers()
            saved = True
            print(json.dumps({'saved': str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
                              'completed': report['completed'], 'error': report.get('error')}), flush=True)

    with HTTPServer(('127.0.0.1', 0), Handler) as server:
        server.timeout = 0.5
        origin = f'http://127.0.0.1:{server.server_port}'
        print(json.dumps({'url': f'{origin}/{token}?autorun=1&report={result_path}'}), flush=True)
        deadline = time.monotonic() + timeout
        while not saved and time.monotonic() < deadline:
            server.handle_request()
    if not saved:
        raise TimeoutError('No browser report received before the deadline')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--page', type=Path, default=ROOT / 'artifacts/web/pplx-quantization-check.html')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--experiment', choices=['pplx-weight-only-quantization-v1', 'pplx-synthetic-full-v1', 'pplx-synthetic-queue-v1', 'pplx-dp4a-v1'], default='pplx-weight-only-quantization-v1')
    args = parser.parse_args()
    try:
        serve(args.page.resolve(), args.output.resolve(), args.timeout, args.experiment)
    except (ValueError, OSError, TimeoutError) as error:
        parser.exit(1, f'Benchmark collection failed: {error}\n')
