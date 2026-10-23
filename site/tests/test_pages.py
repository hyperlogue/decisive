import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pages import build_pages, DEMO
from check_pages import check_pages


class PagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dist = self.root / 'dist'
        self.out = self.root / 'pages'
        self.version = '0123456789abcdef'
        release = self.dist / 'releases' / self.version
        (release / 'pkg').mkdir(parents=True)
        (release / 'workers').mkdir()
        (release / 'lib').mkdir()
        (release / 'app.js').write_text("import './lib/feature.js';\nexport {};")
        (release / 'lib/feature.js').write_text('export {};')
        (release / 'style.css').write_text('body { color: black; }')
        (release / 'workers/model-worker.js').write_text('self.onmessage = () => {};')
        wasm = b'\0asm\x01\0\0\0'
        (release / 'pkg/decisive_bg.wasm').write_bytes(wasm)
        (release / 'pkg/build.json').write_text(json.dumps({'wasm_sha256': hashlib.sha256(wasm).hexdigest()}))
        (self.dist / 'release.json').write_text(json.dumps({'version': self.version}))
        (self.dist / 'NOTICE').write_text('Attribution fixture')
        (self.dist / 'index.html').write_text(f'<link rel="stylesheet" href="./releases/{self.version}/style.css"><script type="module" src="./releases/{self.version}/app.js"></script>')
        assets = ['./index.html', *['./' + str(p.relative_to(self.dist)) for p in release.rglob('*') if p.is_file()]]
        (self.dist / 'sw.js').write_text('const ASSETS=' + json.dumps(assets) + ';')

    def test_custom_domain_and_project_base(self):
        for base in ['', '/project', '/project/']:
            with self.subTest(base=base):
                build_pages(self.dist, self.out, base)
                result = check_pages(self.out, base)
                self.assertEqual(result['base'], base.rstrip('/'))

    def test_rebuild_replaces_old_output(self):
        build_pages(self.dist, self.out)
        (self.out / 'stale.txt').write_text('old')
        build_pages(self.dist, self.out)
        self.assertFalse((self.out / 'stale.txt').exists())

    def test_rejects_source_overlap_before_deletion(self):
        for output in [self.root, self.dist, self.dist / 'nested']:
            with self.subTest(output=output), self.assertRaises(ValueError):
                build_pages(self.dist, output)
            self.assertTrue((self.dist / 'index.html').is_file())

    def test_rejects_invalid_base_before_deletion(self):
        for base in ['//example.com', '/../escape', '/x?y', '/x#y', 'relative', '/x\\y']:
            with self.subTest(base=base), self.assertRaises(ValueError):
                build_pages(self.dist, self.out, base)

    def test_finds_missing_import_and_worker(self):
        build_pages(self.dist, self.out)
        file = self.out / DEMO / 'releases' / self.version / 'lib/feature.js'
        file.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing local asset'):
            check_pages(self.out)
        build_pages(self.dist, self.out)
        (self.out / DEMO / 'sw.js').write_text('const ASSETS=["./index.html"];')
        with self.assertRaisesRegex(ValueError, 'worker missing'):
            check_pages(self.out)

    def test_rejects_symlinks_weights_and_corrupt_wasm(self):
        for kind in ['symlink', 'weights', 'wasm']:
            build_pages(self.dist, self.out)
            if kind == 'symlink':
                (self.out / 'link').symlink_to(self.dist / 'index.html')
            elif kind == 'weights':
                (self.out / 'weights.safetensors').write_bytes(b'unwanted model')
            else:
                (self.out / DEMO / 'releases' / self.version / 'pkg/decisive_bg.wasm').write_bytes(b'\0asm\x01\0\0\0corrupt')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                check_pages(self.out)


if __name__ == '__main__':
    unittest.main()
