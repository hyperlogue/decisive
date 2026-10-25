"""Serve the Pplx GEMM variant benchmark and save the reports it posts.

The page fetches the production kernels from src/models/pplx_flash/webgpu and posts its report
after every shape, so each run's file holds partial results until the run finishes. Expose it
to a phone with `tailscale serve --bg --https=8443 http://127.0.0.1:<port>`.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / 'examples/browser/pplx_gemm_variants.html'
KERNELS = ROOT / 'src/models/pplx_flash/webgpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8771)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark-runs/pplx-gemm-variants')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body, kind):
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            kernel = re.search(r'/kernels/(grouped_q4_[a-z0-9_]+\.wgsl)$', path)
            if kernel and (KERNELS / kernel[1]).is_file():
                self.send((KERNELS / kernel[1]).read_bytes(), 'text/plain; charset=utf-8')
            elif path.endswith('/') or path.endswith('/index.html'):
                self.send(PAGE.read_bytes(), 'text/html; charset=utf-8')
            else:
                self.send_error(404)

        def do_POST(self):
            if not urlsplit(self.path).path.endswith('/report'):
                self.send_error(404)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError('invalid report size')
                report = json.loads(self.rfile.read(size))
                if report.get('experiment') != 'pplx-gemm-variants-v1' or not re.fullmatch(r'[0-9a-f]{12}', str(report.get('run'))):
                    raise ValueError('unexpected report')
            except ValueError as error:
                self.send_error(400, str(error))
                return
            path = output / f"{report['started'][:19].replace(':', '')}-{report['run']}.json"
            path.write_text(json.dumps(report, indent=1))
            print(json.dumps({'saved': str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path), 'shapes': len(report.get('results', [])),
                              'final': report.get('final'), 'errors': len(report.get('errors', [])),
                              'summary': report.get('summary'), 'user_agent': report.get('user_agent')}), flush=True)
            self.send(json.dumps({'saved': path.name}).encode(), 'application/json')

    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(json.dumps({'url': f'http://127.0.0.1:{server.server_port}/'}), flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
