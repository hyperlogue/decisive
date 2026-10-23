#!/usr/bin/env python3
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from pages import DEMO, normalize_base


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
        self.refresh = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for name in ('src', 'href'):
            if name in attrs:
                self.urls.append(attrs[name])
        if tag == 'meta' and attrs.get('http-equiv', '').lower() == 'refresh':
            self.refresh = attrs['content'].split('url=', 1)[1]
            self.urls.append(self.refresh)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_pages(root, base=''):
    root = Path(root).resolve()
    base = normalize_base(base)
    for file in root.rglob('*'):
        require(not file.is_symlink(), f'Symlink in Pages artifact: {file.relative_to(root)}')
        require(file.suffix not in ('.safetensors', '.gguf', '.bin'), 'Model weights must stay outside the Pages artifact')
    demo = root / DEMO
    for name in ['index.html', '404.html', 'demo/index.html', '.nojekyll', f'{DEMO}/index.html', f'{DEMO}/sw.js', f'{DEMO}/NOTICE']:
        require((root / name).is_file(), f'Missing Pages file: {name}')

    def asset(parent, url):
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return
        path = unquote(parsed.path)
        if path.startswith('/'):
            require(not base or path.startswith(base + '/'), f'URL omits project base: {url}')
            target = root / path[len(base):].lstrip('/')
        else:
            target = parent / path
        target = target.resolve()
        require(target.is_relative_to(root), f'Asset escapes Pages tree: {url}')
        if target.is_dir():
            target /= 'index.html'
        require(target.is_file(), f'Missing local asset: {url}')

    for file in root.rglob('*.html'):
        refs = References()
        refs.feed(file.read_text())
        for url in refs.urls:
            asset(file.parent, url)
        if file == root / '404.html':
            require(refs.refresh == f'{base}/{DEMO}/', '404 redirect does not point at the demo')

    version = json.loads((demo / 'release.json').read_text())['version']
    require(bool(re.fullmatch(r'[a-f0-9]{16}', version)), 'Invalid release version')
    release = demo / 'releases' / version
    index = (demo / 'index.html').read_text()
    for name in ['app.js', 'style.css']:
        require(f'./releases/{version}/{name}' in index, f'Entrypoint uses an unversioned {name}')
    for file in release.rglob('*.js'):
        for url in re.findall(r'''(?:\bfrom\s+|\bimport\s*\(\s*|\bimport\s+)['"](\.[^'"]+)['"]''', file.read_text()):
            asset(file.parent, url)
    for file in release.rglob('*.css'):
        for url in re.findall(r'''url\(\s*['"]?([^'"\)]+)['"]?\s*\)''', file.read_text()):
            asset(file.parent, url.strip())
    sw = (demo / 'sw.js').read_text()
    match = re.search(r'const ASSETS\s*=\s*(\[.*?\]);', sw, re.S)
    require(match is not None, 'Missing service-worker precache list')
    assets = json.loads(match[1])
    require('./index.html' in assets, 'Entrypoint missing from offline cache')
    for url in assets:
        asset(demo, url)
    require(f'./releases/{version}/workers/model-worker.js' in assets, 'Model worker missing from offline cache')
    metadata = json.loads((release / 'pkg/build.json').read_text())
    wasm = (release / 'pkg/decisive_bg.wasm').read_bytes()
    require(wasm.startswith(b'\0asm\x01\0\0\0'), 'Invalid WASM artifact')
    require(hashlib.sha256(wasm).hexdigest() == metadata['wasm_sha256'], 'WASM differs from its build metadata')
    return {'release': version, 'precache_files': len(assets), 'base': base}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--base', default='')
    args = parser.parse_args()
    print(json.dumps(check_pages(args.directory, args.base)))
