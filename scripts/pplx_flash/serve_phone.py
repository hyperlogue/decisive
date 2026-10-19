import argparse
import errno
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]


def assets_for(package, model, packed, asset_dir=None):
    original = json.loads((ROOT / 'examples/pplx-flash/model.json').read_text())
    index_path = packed / 'packed-q4-e4.json'
    index = json.loads(index_path.read_text())
    if index['revision'] != original['revision'] or index['scheme'] != 'q4-w4a8-e4-r4-g64':
        raise ValueError('Packed model mismatch')
    assets = {'': asset_dir / 'index.html' if asset_dir else ROOT / 'examples/browser/pplx_flash_phone_forward.html'}
    for name in ['phone.js', 'phone-worker.js', 'packed-cache.js', 'runtime.js', 'embedding-cache.js', 'read-worker.js']:
        assets[name] = (asset_dir or ROOT / 'examples/pplx-flash') / name
    for name in ['build.json', 'decisive.js', 'decisive_bg.wasm']:
        assets['pkg/' + name] = package / name
    files = dict(index['files'])
    for name in files:
        if not re.fullmatch(r'q4-e4-(?:layer-\d{2}|embedding|readout|norm)\.bin', name):
            raise ValueError('Invalid packed filename')
        assets['model/' + name] = packed / name
    for name in ['config.json', 'decision_config.json', 'joint_config.json', 'tokenizer.json']:
        files[name] = original['files'][name]
        assets['model/' + name] = model / name
    files[index_path.name] = {'bytes': index_path.stat().st_size, 'sha256': hashlib.sha256(index_path.read_bytes()).hexdigest()}
    assets['model/' + index_path.name] = index_path
    for name, spec in files.items():
        if assets['model/' + name].stat().st_size != spec['bytes']:
            raise ValueError(f'File size mismatch: {name}')
    assets['request.json'] = ROOT / 'tests/fixtures/pplx_flash/request.json'
    metadata = {'build': json.loads((package / 'build.json').read_text()), 'revision': index['revision'],
                'scheme': index['scheme'], 'packed_manifest_sha256': files[index_path.name]['sha256'],
                'source_sha256': {route or 'index.html': hashlib.sha256(path.read_bytes()).hexdigest()
                                  for route, path in assets.items() if path.suffix in ['.js', '.html']}}
    generated = {'download.json': json.dumps({'revision': index['revision'], 'files': files}).encode(),
                 'metadata.json': json.dumps(metadata).encode()}
    return assets, generated


def serve(package, model, packed, output, port, route_token=None, asset_dir=None):
    assets, generated = assets_for(package, model, packed, asset_dir)
    output.mkdir(parents=True, exist_ok=True)
    route_token = route_token or secrets.token_hex(16)
    if not re.fullmatch(r'[a-f0-9]{32}', route_token):
        raise ValueError('Route token must be 32 hexadecimal characters')
    prefix = '/' + route_token + '/'
    kinds = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm'}
    frozen = {route: (path.read_bytes(), kinds[path.suffix]) for route, path in assets.items()
              if not route.startswith('model/') and path.suffix in ('.html', '.js', '.json')}
    frozen.update({route: (body, 'application/json') for route, body in generated.items()})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def write_block(self, block):
            sent = 0
            deadline = time.monotonic() + 30
            while sent < len(block):
                try:
                    count = self.connection.send(memoryview(block)[sent:])
                except OSError as error:
                    if error.errno not in (errno.ENOBUFS, errno.EAGAIN) or time.monotonic() >= deadline:
                        raise
                    time.sleep(.01)
                    continue
                if not count:
                    raise ConnectionResetError('Socket closed')
                sent += count
                deadline = time.monotonic() + 30

        def do_GET(self):
            self.connection.settimeout(30)
            route = urlsplit(self.path).path
            if not route.startswith(prefix):
                self.send_error(404)
                return
            route = route[len(prefix):]
            if route in frozen:
                body, kind = frozen[route]
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.write_block(body)
                return
            file = assets.get(route)
            if file is None:
                self.send_error(404)
                return
            size = file.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get('Range')
            if requested:
                match = re.fullmatch(r'bytes=(\d+)-(\d+)', requested)
                if not match or not 0 <= int(match[1]) <= int(match[2]) < size:
                    self.send_error(416)
                    return
                start, end = map(int, match.groups())
            self.send_response(206 if requested else 200)
            kind = kinds.get(file.suffix, 'application/octet-stream')
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if requested:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with file.open('rb') as stream:
                    stream.seek(start)
                    left = end - start + 1
                    while left:
                        block = stream.read(min(left, 1024 * 1024))
                        if not block:
                            raise OSError('File changed during transfer')
                        self.write_block(block)
                        left -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if urlsplit(self.path).path != prefix + 'results' or self.headers.get('Sec-Fetch-Site') not in (None, 'same-origin'):
                self.send_error(403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError('Report size invalid')
                report = json.loads(self.rfile.read(size))
                if report.get('experiment') != 'pplx-phone-forward-v1' or not isinstance(report.get('runs'), list):
                    raise ValueError('Invalid report')
                path = output / f'phone-{secrets.token_hex(8)}.json'
                with path.open('x') as stream:
                    json.dump(report, stream)
                print(json.dumps({'report': path.name, 'runs': len(report['runs'])}), flush=True)
            except (ValueError, OSError) as error:
                self.send_error(400, str(error))
                return
            self.send_response(204)
            self.end_headers()

    with ThreadingHTTPServer(('127.0.0.1', port), Handler) as server:
        print(json.dumps({'url': f'http://127.0.0.1:{server.server_port}{prefix}', 'port': server.server_port}), flush=True)
        server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True, type=Path)
    parser.add_argument('--packed', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/pplx-decider-flash-v1.2')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark-runs/pplx-phone')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--route-token')
    parser.add_argument('--assets', type=Path)
    args = parser.parse_args()
    serve(args.package.resolve(), args.model.resolve(), args.packed.resolve(), args.output.resolve(), args.port,
          args.route_token, args.assets.resolve() if args.assets else None)
