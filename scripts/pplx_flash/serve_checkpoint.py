import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import re
import secrets
import threading
import time
from urllib.parse import urlsplit

from serve_quantization import write_new_json

ROOT = Path(__file__).resolve().parents[2]


def serve(package, model, plan_path, output, timeout, corpus=None, packed=None, manual=False):
    if manual and packed is None:
        raise ValueError('Manual forward pass requires packed Q4-E4 weights')
    if output is not None and output.exists():
        raise ValueError('Choose a new report filename')
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'examples/pplx-flash/model.json').read_text())
    assets = {f'/model/{name}': model/name for name in manifest['files']}
    for name, spec in manifest['files'].items():
        if not (model/name).is_file() or (model/name).stat().st_size != spec['bytes']:
            raise ValueError(f'Missing or incomplete checkpoint file: {name}')
    assets.update({f'/pkg/{name}': package/name for name in ['build.json','decisive.js','decisive_bg.wasm']})
    if packed is not None:
        name='packed-q4-e4.json';index=json.loads((packed/name).read_text())
        if index['revision']!=manifest['revision'] or index['scheme']!='q4-w4a8-e4-r4-g64':raise ValueError('Packed model mismatch')
        assets['/model/'+name]=packed/name
        for name,spec in index['files'].items():
            if not re.fullmatch(r'q4-e4-(?:layer-\d{2}|embedding|readout|norm)\.bin',name):raise ValueError('Invalid packed path')
            if (packed/name).stat().st_size!=spec['bytes']:raise ValueError('Packed file size mismatch')
            assets['/model/'+name]=packed/name
    assets.update({f'/runtime/{name}': ROOT/'examples/pplx-flash'/name for name in ['runtime.js','embedding-cache.js','read-worker.js']})
    assets.update({f'/fixtures/{name}.json': ROOT/'tests/fixtures/pplx_flash'/f'{name}.json' for name in ['request','multilingual','reference']})
    worker_name='forward-worker.js' if manual else 'checkpoint-worker.js'
    assets['/'+worker_name] = ROOT/'examples/pplx-flash'/worker_name
    if corpus is not None:
        assets['/quality-corpus.json'] = corpus
    plan = [] if manual else json.loads(plan_path.read_text())
    if not manual and (not isinstance(plan,list) or not 1 <= len(plan) <= 16):
        raise ValueError('Expected 1–16 checkpoint phases')
    token = secrets.token_hex(16)
    page_path=ROOT/'examples/browser'/('pplx_flash_forward.html' if manual else 'pplx_flash_checkpoint.html')
    page = page_path.read_text().replace('__PLAN_JSON__',json.dumps(plan).replace('<','\\u003c'))
    metadata = {'build':json.loads((package/'build.json').read_text()),'repository':manifest['repository'],
                'revision':manifest['revision'],'plan':plan,'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [page_path,assets['/'+worker_name],*[assets[f'/runtime/{name}'] for name in ['runtime.js','embedding-cache.js','read-worker.js']]]}}
    if corpus is not None:
        metadata['corpus_sha256'] = hashlib.sha256(corpus.read_bytes()).hexdigest()
    if packed is not None:
        metadata['packed_manifest_sha256']=hashlib.sha256((packed/'packed-q4-e4.json').read_bytes()).hexdigest()
    frozen={route:file.read_bytes() for route,file in assets.items()
            if not route.startswith('/model/') and file.suffix in ['.js','.json']}
    saved = threading.Event()
    lock = threading.Lock()
    transfers={}

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self,*_):
            pass

        def body(self, content, content_type):
            self.send_response(200)
            self.send_header('Content-Type',content_type)
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path=urlsplit(self.path).path
            if path == f'/{token}':
                return self.body(page.encode(),'text/html; charset=utf-8')
            if path == '/metadata':
                return self.body(json.dumps(metadata).encode(),'application/json')
            if path in frozen:
                return self.body(frozen[path],'text/javascript' if path.endswith('.js') else 'application/json')
            file=assets.get(path)
            if file is None:
                self.send_error(404);return
            size=file.stat().st_size;start,end=0,size-1
            requested=self.headers.get('Range')
            if requested:
                match=re.fullmatch(r'bytes=(\d+)-(\d+)',requested)
                if not match or not 0 <= int(match[1]) <= int(match[2]) < size:
                    self.send_error(416);return
                start,end=map(int,match.groups())
            self.send_response(206 if requested else 200)
            self.send_header('Content-Type','text/javascript' if file.suffix=='.js' else 'application/wasm' if file.suffix=='.wasm' else 'application/octet-stream')
            self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
            if path.endswith('.bin'):self.send_header('Cache-Control','no-store')
            if requested:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            try:
                with file.open('rb') as stream:
                    stream.seek(start);left=end-start+1
                    while left:
                        block=stream.read(min(left,1024*1024))
                        if not block:raise OSError('Checkpoint file changed during read')
                        self.wfile.write(block);left-=len(block)
                if path.startswith('/model/'):
                    with lock:transfers[path]=transfers.get(path,0)+end-start+1
            except (BrokenPipeError,ConnectionResetError):
                pass

        def do_POST(self):
            if manual:
                self.send_error(404);return
            if self.headers.get('Origin') != origin or self.path not in [f'/progress/{token}',f'/results/{token}']:
                self.send_error(403);return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 32*1024*1024:raise ValueError('Invalid report size')
                report=json.loads(self.rfile.read(size))
                with lock:
                    if self.path.startswith('/progress/'):
                        if not isinstance(report.get('message'),str):raise ValueError('Invalid progress')
                        if 'phase' in report:
                            phase=report['phase'];index=phase['index']
                            if type(index) is not int or not 1<=index<=len(plan):raise ValueError('Invalid phase index')
                            write_new_json(output.with_name(f'{output.stem}.phase-{index:02}.json'),phase['report'])
                        print(json.dumps({'progress':report['message']}),flush=True)
                    else:
                        if report.get('experiment')!='pplx-checkpoint-v1' or type(report.get('completed')) is not bool:raise ValueError('Invalid report')
                        report['server_model_transfer_bytes']=dict(transfers)
                        write_new_json(output,report);saved.set();print(json.dumps({'saved':str(output.relative_to(ROOT)),'completed':report['completed'],'error':report.get('error')}),flush=True)
                self.send_response(200);self.send_header('Content-Length','0');self.end_headers()
            except (ValueError,KeyError,TypeError,OSError) as error:
                self.send_error(400,str(error))

    with ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
        origin=f'http://127.0.0.1:{server.server_port}'
        server.timeout=.5
        url=f'{origin}/{token}' if manual else f'{origin}/{token}?autorun=1&report=/results/{token}'
        print(json.dumps({'url':url}),flush=True)
        deadline=time.monotonic()+timeout
        while not saved.is_set() and time.monotonic()<deadline:server.handle_request()
    if not manual and not saved.is_set():raise TimeoutError('No checkpoint report received')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',required=True,type=Path)
    parser.add_argument('--model',type=Path,default=ROOT/'models/pplx-decider-flash-v1.2')
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--manual',action='store_true')
    parser.add_argument('--timeout',type=int,default=3600)
    parser.add_argument('--corpus',type=Path,help='Explicit local quality corpus to expose at /quality-corpus.json')
    parser.add_argument('--packed',type=Path)
    args=parser.parse_args()
    if not args.manual and (args.plan is None or args.output is None):
        parser.error('--plan and --output are required unless --manual is set')
    serve(args.package.resolve(),args.model.resolve(),args.plan.resolve() if args.plan else None,
          args.output.resolve() if args.output else None,args.timeout,args.corpus.resolve() if args.corpus else None,
          args.packed.resolve() if args.packed else None,args.manual)
