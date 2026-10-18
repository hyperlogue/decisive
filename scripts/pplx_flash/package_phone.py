import argparse
import hashlib
import json
from pathlib import Path

from package_quantization import bundle as bundle_dots
from package_synthetic import bundle as bundle_model

ROOT = Path(__file__).resolve().parents[2]


def bundle(package, destination):
    destination.mkdir(parents=True, exist_ok=False)
    bundle_dots(destination / 'matmul.html', dp4a=True)
    bundle_dots(destination / 'check-int8.html', suite=True)
    bundle_model(package, destination / 'model.html')
    (destination / 'index.html').write_bytes((ROOT / 'examples/browser/pplx_flash_phone.html').read_bytes())
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(destination.glob('*.html'))}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Phone bundle: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path, help='Wasm package built with kernel-checks')
    parser.add_argument('destination', type=Path, help='New directory for the phone bundle')
    args = parser.parse_args()
    bundle(args.package, args.destination)
