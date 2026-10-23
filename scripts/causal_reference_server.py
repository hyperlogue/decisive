#!/usr/bin/env python3

import argparse
import importlib.util
import importlib.metadata
import json
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--model', choices=['kev', 'jev'], required=True)
p.add_argument('--engine', choices=['torch', 'llama'], default='torch')
p.add_argument('--models', type=Path, default=Path('models'))
p.add_argument('--sources', type=Path, required=True)
p.add_argument('--threads', type=int, default=1)
p.add_argument('--port', type=int, default=18991)
p.add_argument('--scorer', type=Path)
p.add_argument('--gguf', type=Path, help='Override published F16 with an exact-checkpoint conversion')
p.add_argument('--attention', choices=['eager', 'sdpa'], default='eager')
p.add_argument('--flash-attn', choices=['off', 'on'], default='off', help='Jev llama.cpp Flash Attention setting')
p.add_argument('--ubatch', type=int, default=1024, help='Jev llama.cpp physical batch size')
p.add_argument('--generate', type=Path, help='Write reference fixtures instead of serving')
p.add_argument('--append', action='store_true', help='Include a third request with appended state in generated fixtures')
p.add_argument('--lengths', default='128,512,1024,2048')
p.add_argument('--languages', default='en,zh,fr,ar')
a = p.parse_args()
assert a.threads > 0
assert a.ubatch > 0
if a.flash_attn != 'off' and a.engine != 'llama':
    p.error('--flash-attn applies to the Jev llama.cpp adapter')
sys.path.insert(0, str(a.sources / (a.model + '-source')))

def module(path):
    spec = importlib.util.spec_from_file_location('official_runtime', path)
    rt = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rt
    spec.loader.exec_module(rt)
    return rt

meta = {'model': a.model, 'engine': a.engine, 'threads': a.threads,
        'device': 'cpu', 'benchmark_transport': 'single-lane HTTP', 'adapter': 1}
meta['python'] = sys.version.split()[0]
meta['packages'] = {}
for package in ['torch', 'transformers', 'peft', 'accelerate', 'numpy', 'safetensors',
                'tokenizers', 'pydantic', 'huggingface-hub', 'regex', 'sympy']:
    try:
        meta['packages'][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pass
started = time.perf_counter()
if a.engine == 'torch':
    import torch
    import transformers
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    meta.update(torch=torch.__version__, transformers=transformers.__version__, dtype='float32',
                cpu_capability=torch.backends.cpu.get_cpu_capability(), mkl=torch.backends.mkl.is_available(),
                mkldnn=torch.backends.mkldnn.enabled)

if a.model == 'kev':
    assert a.engine == 'torch', 'Use native llama-server for Kev'
    from transformers import AutoTokenizer
    from peft import PeftModel
    from kev.model import DecisionModel, encode
    from kev.api import SystemOneRequest, to_record, to_answers, output_tokens
    root = a.models / 'kev-0.8b'
    h = torch.load(root / 'head.pt', map_location='cpu', weights_only=True)
    tok = AutoTokenizer.from_pretrained(root)
    m = DecisionModel(str(a.models / 'qwen3.5-0.8b-base'), tok, 'cpu',
                      head_dim=h['head_dim'], dtype=torch.float32, attn=a.attention)
    m.lm = PeftModel.from_pretrained(m.lm, str(root), torch_device='cpu').merge_and_unload()
    m.head.load_state_dict(h['head'])
    m.head.temperature = h['temperature']
    m.eval()
    meta.update(attention=a.attention, delta_kernel='transformers PyTorch fallback', cross_request_cache='official prefix API, one state')
    cached_ids, cached_prefix = None, None
    def prepare(body):
        rec, info = to_record(SystemOneRequest.model_validate(body))
        return encode(tok, rec, max_state=65536, max_branch=73728, strict=True), info
    def token_count(body):
        enc, _ = prepare(body)
        return len(enc['ids']), enc['state_tokens']
    def infer(body):
        global cached_ids, cached_prefix
        enc, info = prepare(body)
        ids = enc['ids'][:enc['state_tokens']]
        hit = ids == cached_ids and cached_prefix is not None
        with torch.inference_mode():
            if hit:
                values = m.probs_with_prefix(enc, cached_prefix)
            else:
                values, cached_prefix = m.probs_and_prefix(enc)
                cached_ids = ids
            probs = [x.tolist() for x in values]
        answers = to_answers(probs, info)
        return {'answers': answers, 'probabilities': dict(zip(body['questions'], probs)),
                'benchmark_cache': {'hit': hit, 'prefix_tokens_reused': len(ids) if hit else 0},
                'usage': {'input_tokens': len(enc['ids']), 'state_tokens': enc['state_tokens'],
                          'output_tokens': output_tokens(tok, answers)}}
    def reset():
        global cached_ids, cached_prefix
        cached_ids, cached_prefix = None, None
        return {'reset': True}
else:
    from jev_style.adapter import Adapter
    from jev_style.schema import parse_request
    if a.engine == 'torch':
        rt = module(a.sources / 'jev' / 'jev_style_decision.py')
        m = rt.JevStyleDecision(a.models / 'jev-style-0.8b-v3', device='cpu', dtype='float32', cuda_graphs=False)
        meta.update(attention='official chunked SDPA', cross_request_cache=False)
    else:
        rt = module(a.sources / 'jev-gguf' / 'jev_style_decision_gguf.py')
        m = rt.JevStyleDecisionGGUF(a.models / 'jev-style-0.8b-v3-gguf', binary=a.scorer, gguf=a.gguf,
                                  n_gpu_layers=0, threads=a.threads, n_ubatch=a.ubatch,
                                  flash_attn=a.flash_attn, many_mode='exact', stderr=sys.stderr)
        meta.update(dtype=m.info['desc'] + ' / ' + m.info.get('cache_type_k', 'f16') + ' K / ' + m.info.get('cache_type_v', 'f16') + ' V', attention='flash '+a.flash_attn,
                    many_mode='exact', cross_request_cache='whole ubatches, multi-question only', scorer=m.info)
    adapter = Adapter(m, rt, a.engine)
    def token_count(body):
        parsed = parse_request(body)
        rows = [m.renderer.render(parsed.state, q.internal) for q in parsed.questions]
        return rows[0].prefix_len + sum(r.head_tokens for r in rows), rows[0].prefix_len
    def infer(body):
        return adapter.systemone(body)
    def reset():
        return m._request({'cmd': 'reset'}) if a.engine == 'llama' else {'reset': True, 'cross_request_cache': False}

meta['load_ms'] = (time.perf_counter() - started) * 1000

def close(*_):
    if a.model == 'jev':
        adapter.worker.shutdown()
        if a.engine == 'llama':
            m.close()
    sys.exit(0)
signal.signal(signal.SIGTERM, close)

if a.generate:
    lang = {
        'en': ('A previous shipment arrived on time. ', 'I was charged twice. Refund the duplicate charge.', 'Which team should help?', 'Who should handle this request?', ['billing', 'shipping', 'technical']),
        'zh': ('之前的包裹已经按时送达。', '我的订单被扣款两次。请退还重复的款项。', '哪个团队应该提供帮助？', '应该由谁处理这个请求？', ['账单', '物流', '技术']),
        'fr': ('Une livraison précédente est arrivée à temps. ', 'Mon compte a été débité deux fois. Remboursez le doublon.', 'Quelle équipe doit intervenir ?', 'Qui doit traiter cette demande ?', ['facturation', 'livraison', 'technique']),
        'ar': ('وصلت الشحنة السابقة في الوقت المحدد. ', 'تم خصم المبلغ مرتين. أرجو إعادة المبلغ المكرر.', 'أي فريق يجب أن يساعد؟', 'من يجب أن يتولى هذا الطلب؟', ['الفواتير', 'الشحن', 'التقنية']),
    }
    a.generate.parent.mkdir(parents=True, exist_ok=True)
    with a.generate.open('w') as f:
        for language in a.languages.split(','):
            filler, content, q1, q2, names = lang[language]
            for length in map(int, a.lengths.split(',')):
                body = {'state': content, 'questions': {'q': {'type': 'choice', 'instructions': q1, 'criteria': dict.fromkeys(names)}}}
                lo, hi = 0, length
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    body['state'] = filler * mid + content
                    if token_count(body)[0] <= length:
                        lo = mid
                    else:
                        hi = mid - 1
                body['state'] = filler * lo + content
                while token_count(body)[0] < length:
                    body['state'] += ' x'
                assert token_count(body)[0] == length, (language, length, token_count(body))
                requests = []
                append_text = {
                    'en': ' The customer also requested an update on the refund.',
                    'zh': '客户还希望了解退款进度。',
                    'fr': ' Le client demande aussi des nouvelles du remboursement.',
                    'ar': ' يطلب العميل أيضًا تحديثًا بشأن استرداد المبلغ.',
                }[language]
                sequence = [('initial', q1, body['state']), ('followup', q2, body['state'])]
                if a.append:
                    sequence.append(('append', q2, body['state'] + append_text))
                for name, question, state_text in sequence:
                    body['state'] = state_text
                    body['questions']['q']['instructions'] = question
                    request = json.loads(json.dumps(body))
                    n, state_n = token_count(request)
                    reset()
                    expected = infer(request)
                    for key in ['latency_ms', 'timing', 'backend', 'model']:
                        expected.pop(key, None)
                    requests.append({'kind': name, 'request': request, 'expected': expected,
                                     'input_tokens': n, 'state_tokens': state_n})
                row = {'id': f'{a.model}-{language}-{length}', 'model': a.model, 'language': language,
                       'length': length, 'requests': requests, 'reference': meta}
                f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush()
                print(row['id'], flush=True)
    close()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass
    def reply(self, status, value):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        self.reply(200, meta) if self.path == '/health' else self.reply(404, {})
    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/benchmark/reset':
                self.reply(200, reset())
            elif self.path == '/v1/systemone':
                self.reply(200, infer(body))
            else:
                self.reply(404, {})
        except Exception as e:
            self.reply(500, {'error': str(e)})
print(json.dumps(meta), flush=True)
HTTPServer(('127.0.0.1', a.port), Handler).serve_forever()
