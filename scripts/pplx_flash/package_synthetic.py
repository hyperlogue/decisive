import argparse
import base64
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def bundle(package, output, queue=None):
    build = json.loads((package / 'build.json').read_text())
    wasm = (package / 'decisive_bg.wasm').read_bytes()
    bindings = (package / 'decisive-classic.js').read_text()
    if hashlib.sha256(wasm).hexdigest() != build['wasm_sha256']:
        raise ValueError('Wasm hash mismatch')
    if hashlib.sha256(bindings.encode()).hexdigest() != build['classic_js_sha256']:
        raise ValueError('Classic bindings hash mismatch')
    if 'PplxFlashSyntheticGpu' not in bindings or 'kernel-checks' not in build['features']:
        raise ValueError('Build the synthetic runner with DECISIVE_PPLX_CHECKS=1')
    js_path = ROOT / 'examples/pplx-flash/synthetic-full.js'
    template_path = ROOT / ('examples/browser/pplx_flash_synthetic_queue.html' if queue else 'examples/browser/pplx_flash_synthetic.html')
    javascript = js_path.read_text()
    if re.search('</script', bindings + javascript, re.I):
        raise ValueError('Unsafe embedded script')
    build['synthetic_runner_sources'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in [js_path, template_path, Path(__file__).resolve()]}
    html = template_path.read_text()
    if queue:
        plan = json.loads(queue.read_text())
        if not isinstance(plan, list) or not 1 <= len(plan) <= 16:
            raise ValueError('Queue requires 1–16 phases')
        for phase in plan:
            if (not isinstance(phase, dict) or not re.fullmatch(r'[a-z0-9-]{1,64}', phase.get('id', ''))
                    or phase.get('path') not in ['fp16', 'hybrid', 'int8', 'pure-int8']
                    or not isinstance(phase.get('candidate'), str)
                    or not isinstance(phase.get('rows'), list) or not phase['rows']
                    or any(type(n) is not int or n < 3 or n > 2048 for n in phase['rows'])
                    or type(phase.get('warmups')) is not int or not 0 <= phase['warmups'] <= 2
                    or type(phase.get('rounds')) is not int or not 1 <= phase['rounds'] <= 7
                    or type(phase.get('seed', 137)) is not int or not 0 <= phase.get('seed',137) <= 0xffffffff):
                raise ValueError('Invalid queue phase')
        if len({p['id'] for p in plan}) != len(plan):
            raise ValueError('Duplicate queue phase ID')
        build['queue_plan_sha256'] = hashlib.sha256(queue.read_bytes()).hexdigest()
        if html.count('__QUEUE_JSON__') != 1:
            raise ValueError('Expected one queue plan marker')
        html = html.replace('__QUEUE_JSON__', json.dumps(plan).replace('<', '\\u003c'))
    for marker, replacement in {'__WASM_BASE64__': base64.b64encode(wasm).decode(),
                                '__BUILD_JSON__': json.dumps(build).replace('<', '\\u003c'),
                                '/*__WASM_BINDINGS__*/': bindings, '/*__SYNTHETIC_JS__*/': javascript}.items():
        if html.count(marker) != 1:
            raise ValueError(f'Expected one {marker}')
        html = html.replace(marker, replacement)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as file:
        file.write(html)
    print(f'Packaged {output} ({len(html) / 1024**2:.1f} MiB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--queue', type=Path, help='Run a bounded list of independent two-engine comparisons')
    args = parser.parse_args()
    bundle(args.package, args.output, args.queue)
