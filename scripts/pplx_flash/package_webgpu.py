import argparse
import base64
import hashlib
import json
from pathlib import Path
import re

from verify_webgpu import ROOT, package, read


def bundle(source, destination):
    package(source, kernels_only=True)
    classic = source / 'decisive-classic.js'
    build = read(source / 'build.json')
    if not classic.is_file() or 'classic_js_sha256' not in build:
        raise ValueError('Rebuild with DECISIVE_PPLX_CHECKS=1 to generate classic worker bindings')
    if hashlib.sha256(classic.read_bytes()).hexdigest() != build['classic_js_sha256']:
        raise ValueError('Classic bindings do not match build.json')
    bindings = classic.read_text()
    if re.search(r'^\s*(?:import|export)\s|import\.meta|</script', bindings, re.MULTILINE | re.IGNORECASE):
        raise ValueError('The generated bindings must be a self-contained classic script')
    if not bindings.startswith('let wasm_bindgen ='):
        raise ValueError('Expected wasm-bindgen --target no-modules bindings')
    template = (ROOT / 'examples/browser/pplx_flash_standalone.html').read_text()
    model = read(ROOT / 'examples/pplx-flash/model.json')
    replacements = {
        '__WASM_BASE64__': base64.b64encode((source / 'decisive_bg.wasm').read_bytes()).decode('ascii'),
        '__BUILD_JSON__': json.dumps(build).replace('<', '\\u003c'),
        '__MODEL_JSON__': json.dumps({k: model[k] for k in ('repository', 'revision')}),
        '/*__WASM_BINDINGS__*/': bindings,
    }
    for marker, content in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f'Expected one template marker: {marker}')
        template = template.replace(marker, content)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as output:
        output.write(template)
    print(f'Open in Chrome: {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('output', type=Path, nargs='?', default=Path('artifacts/web/pplx-flash-webgpu-check.html'))
    args = parser.parse_args()
    try:
        bundle(args.package, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'Packaging failed: {error}\n')
