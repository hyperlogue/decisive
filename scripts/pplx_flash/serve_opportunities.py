import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / 'examples/browser/pplx_flash_opportunities.html'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark-runs/pplx-opportunities')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            path = urlsplit(self.path).path
            if not (path.endswith('/') or path.endswith('/index.html')):
                self.send_error(404)
                return
            body = PAGE.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if not urlsplit(self.path).path.endswith('/report'):
                self.send_error(404)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError('invalid report size')
                report = json.loads(self.rfile.read(size))
                if report.get('experiment') != 'pplx-webgpu-opportunities-v1':
                    raise ValueError('unexpected report')
            except ValueError as error:
                self.send_error(400, str(error))
                return
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            path = output / f'report-{stamp}-{secrets.token_hex(3)}.json'
            path.write_text(json.dumps(report, indent=1))
            print(json.dumps({'saved': str(path.relative_to(ROOT)), 'user_agent': report.get('user_agent'),
                              'errors': len(report.get('results', {}).get('errors', []))}), flush=True)
            body = json.dumps({'saved': path.name}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        print(json.dumps({'url': f'http://127.0.0.1:{server.server_port}/'}), flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
