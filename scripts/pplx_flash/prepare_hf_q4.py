#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = json.loads((ROOT / 'examples/pplx-flash/model.json').read_text())
SCHEME = 'q4-w4a8-e4-r4-g64'
SUPPORT = ['config.json', 'decision_config.json', 'joint_config.json', 'tokenizer.json', 'LICENSE', 'NOTICE']
SHARDS = ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def spec(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def checked_file(directory, name):
    require(bool(re.fullmatch(r'[a-zA-Z0-9_.-]+', name)) and name not in ('.', '..'), 'Unsafe filename')
    path = directory / name
    require(path.is_file() and not path.is_symlink(), f'Missing file or symlink: {name}')
    return path


def verify_files(directory, files):
    for name, expected in files.items():
        require(spec(checked_file(directory, name)) == expected, f'Checksum or size mismatch: {name}')


def header(path):
    with path.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        require(length <= 16 * 1024 * 1024, 'Invalid safetensors header')
        return f.read(length).decode()


def packed_size(shape):
    if len(shape) == 2:
        require(shape[1] % 64 == 0, 'Matrix width is not a multiple of 64')
        return math.prod(shape) // 64 * 36
    return math.prod(shape) * 2


def validate_packed(directory):
    manifest = json.loads(checked_file(directory, 'packed-q4-e4.json').read_text())
    require(manifest['schema_version'] == 1 and manifest['scheme'] == SCHEME, 'Unsupported packed format')
    require(manifest['revision'] == SOURCE['revision'], 'Wrong source revision')
    require((manifest['group_size'], manifest['group_bytes'], manifest['embedding_row_bytes']) == (64, 36, 1440), 'Wrong group layout')
    require(len(manifest['layers']) == 32 and len(manifest['headers']) == 2, 'Incomplete checkpoint')
    tensors = {}
    for source_header in manifest['headers']:
        for name, value in json.loads(source_header).items():
            if name == '__metadata__':
                continue
            require(name not in tensors and value['dtype'] == 'BF16', 'Invalid source tensor index')
            tensors[name] = value
    expected_files = set()
    decoder_bytes = 0
    for index, layer in enumerate(manifest['layers']):
        name = f'q4-e4-layer-{index:02}.bin'
        require(layer['file'] == name, 'Wrong layer filename/order')
        expected_files.add(name)
        expected_tensors = {n for n in tensors if n.startswith(f'language_model.layers.{index}.')}
        seen, position = set(), 0
        for tensor in layer['tensors']:
            key = tensor['name']
            require(key in expected_tensors and key not in seen, 'Unknown or repeated tensor')
            seen.add(key)
            require(tensor['shape'] == tensors[key]['shape'], 'Tensor shape mismatch')
            require(tensor['start'] == position and tensor['end'] - position == packed_size(tensor['shape']), 'Invalid tensor range')
            position = tensor['end']
        require(seen == expected_tensors and seen, 'Missing layer tensors')
        require(position == layer['bytes'] == manifest['files'][name]['bytes'], 'Layer size mismatch')
        decoder_bytes += position
    require(decoder_bytes == manifest['decoder_bytes'], 'Decoder total mismatch')
    auxiliary = {
        'embedding': ('q4-e4-embedding.bin', packed_size(tensors['language_model.embed_tokens.weight']['shape'])),
        'readout': ('q4-e4-readout.bin', packed_size([255, 2560])),
        'norm': ('q4-e4-norm.bin', 2560 * 2),
    }
    for key, (name, size) in auxiliary.items():
        require(manifest[key] == name and manifest['files'][name]['bytes'] == size, f'Invalid {key} payload')
        expected_files.add(name)
    require(set(manifest['files']) == expected_files, 'Unexpected packed files')
    verify_files(directory, manifest['files'])
    return manifest


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def reference_group(raw):

    values = [struct.unpack('<f', b'\0\0' + raw[i:i + 2])[0] for i in range(0, 128, 2)]
    require(len(raw) == 128 and all(math.isfinite(v) for v in values), 'Invalid BF16 group')
    scale = max(f32(max(abs(v) for v in values) / 7), 2 ** -126)
    inverse = f32(1 / scale)
    codes = [max(-7, min(7, round(f32(v * inverse)))) + 8 for v in values]
    return bytes(a | b << 4 for a, b in zip(codes[::2], codes[1::2])) + struct.pack('<f', scale)


def verify_samples(source, directory, manifest):
    sources = {}
    for name, text in zip(SHARDS, manifest['headers']):
        require(header(source / name) == text, 'Exported headers differ from source')
        for key, value in json.loads(text).items():
            if key != '__metadata__':
                sources[key] = (source / name, 8 + len(text.encode()) + value['data_offsets'][0])
    checked = 0

    def check(name, filename, start, shape, source_info=None):
        nonlocal checked
        src, source_offset = source_info or sources[name]
        with src.open('rb') as original, (directory / filename).open('rb') as packed:
            if len(shape) == 2:
                groups = math.prod(shape) // 64
                for group in sorted({0, groups // 2, groups - 1}):
                    original.seek(source_offset + group * 128)
                    packed.seek(start + group * 36)
                    require(packed.read(36) == reference_group(original.read(128)), f'Q4 sample mismatch: {name}, group {group}')
                    checked += 1
            else:
                size = math.prod(shape) * 2
                original.seek(source_offset)
                packed.seek(start)
                require(original.read(size) == packed.read(size), f'Unquantized tensor changed: {name}')
                checked += 1

    for layer in manifest['layers']:
        for tensor in layer['tensors']:
            check(tensor['name'], layer['file'], tensor['start'], tensor['shape'])
    embedding = next(json.loads(text)['language_model.embed_tokens.weight']['shape']
                     for text in manifest['headers'] if 'language_model.embed_tokens.weight' in json.loads(text))
    check('language_model.embed_tokens.weight', manifest['embedding'], 0, embedding)
    check('language_model.norm.weight', manifest['norm'], 0, [2560])
    readout = source / 'readout.safetensors'
    text = header(readout)
    offset = 8 + len(text.encode()) + json.loads(text)['weight']['data_offsets'][0]
    check('readout', manifest['readout'], 0, [255, 2560], (readout, offset))
    return checked


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def prepare(args):
    source, out = args.source.resolve(), args.output.resolve()
    require(not out.exists(), 'Output already exists; choose a new directory')
    print('Verifying pinned source files…', flush=True)
    verify_files(source, SOURCE['files'])
    subprocess.run([str(args.exporter.resolve()), 'pplx-flash', '--model', str(source), 'export-webgpu-q4', '--output', str(out)], check=True)
    manifest = validate_packed(out)
    samples = verify_samples(source, out, manifest)
    for name in SUPPORT:
        shutil.copyfile(source / name, out / name)
    files = {**manifest['files'], **{name: spec(out / name) for name in ['packed-q4-e4.json', *SUPPORT]}}
    files = dict(sorted(files.items()))
    total = sum(f['bytes'] for f in files.values())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    template = (Path(__file__).with_name('hf_q4_model_card.md')).read_text()
    card = template.replace('{{SOURCE_REVISION}}', SOURCE['revision']).replace('{{RUNTIME_REVISION}}', commit)
    card = card.replace('{{TOTAL_BYTES}}', f'{total:,}').replace('{{TOTAL_GB}}', f'{total / 1e9:.3f}').replace('{{TOTAL_GIB}}', f'{total / 2**30:.3f}')
    card = card.replace('{{SAMPLE_CHECKS}}', str(samples))
    (out / 'README.md').write_text(card)
    release = {
        'schema_version': 1, 'format': SCHEME,
        'source': {'repository': SOURCE['repository'], 'revision': SOURCE['revision'], 'files': SOURCE['files']},
        'runtime': {'repository': 'https://github.com/hyperlogue/dicisive', 'revision': commit, 'mode': 'q4-e4', 'features': ['pplx-flash', 'wasm', 'webgpu', 'kernel-checks']},
        'validation': {'source_sha256': True, 'packed_sha256': True, 'layout': True, 'sample_checks': samples, 'full_inference': False},
        'download': {'total_bytes': total, 'files': files},
    }
    write_json(out / 'release-manifest.json', release)
    sums = dict(files)
    sums.update({name: spec(out / name) for name in ['README.md', 'release-manifest.json']})
    (out / 'SHA256SUMS').write_text(''.join(f"{value['sha256']}  {name}\n" for name, value in sorted(sums.items())))
    print(json.dumps({'runtime_download_bytes': total, 'upload_files': len(sums) + 1, 'sample_checks': samples, 'published': False}), flush=True)


def verify(args):
    out = args.output.resolve()
    manifest = validate_packed(out)
    release = json.loads(checked_file(out, 'release-manifest.json').read_text())
    require(release['source'] == {**SOURCE}, 'Source provenance mismatch')
    files = release['download']['files']
    require(set(files) == set(manifest['files']) | set(SUPPORT) | {'packed-q4-e4.json'}, 'Incomplete download catalog')
    require(all(files[name] == value for name, value in manifest['files'].items()), 'Download/packed checksum mismatch')
    require(all(files[name] == SOURCE['files'][name] for name in SUPPORT), 'Modified source metadata/license')
    require(release['download']['total_bytes'] == sum(v['bytes'] for v in files.values()), 'Download total mismatch')
    verify_files(out, {name: value for name, value in files.items() if name not in manifest['files']})
    expected = set(files) | {'README.md', 'release-manifest.json', 'SHA256SUMS'}
    require({p.name for p in out.iterdir() if p.name != '.cache'} == expected, 'Unexpected upload files')
    sums = {}
    for line in checked_file(out, 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        require(name not in sums and re.fullmatch(r'[a-f0-9]{64}', digest), 'Invalid SHA256SUMS')
        sums[name] = digest
    require(set(sums) == expected - {'SHA256SUMS'}, 'Incomplete SHA256SUMS')
    for name, digest in sums.items():
        actual = files[name] if name in files else spec(checked_file(out, name))
        require(digest == actual['sha256'], f'SHA256SUMS mismatch: {name}')
    print(json.dumps({'verified': True, 'runtime_download_bytes': release['download']['total_bytes'], 'upload_files': len(expected)}))


def main():
    parser = argparse.ArgumentParser(description='Prepare/verify an upload directory using the unchanged Decisive Q4 exporter.')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Verify, export, and package locally; never uploads')
    p.add_argument('--source', type=Path, default=ROOT / 'models/pplx-decider-flash-v1.2')
    p.add_argument('--exporter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(run=prepare)
    v = commands.add_parser('verify', help='Verify an existing upload folder')
    v.add_argument('--output', type=Path, required=True)
    v.set_defaults(run=verify)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
