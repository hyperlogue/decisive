import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from prepare_hf_q4 import SOURCE, SCHEME, reference_group, validate_packed, verify_files


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {'schema_version': 1, 'revision': SOURCE['revision'], 'scheme': SCHEME,
                         'group_size': 64, 'group_bytes': 36, 'embedding_row_bytes': 1440,
                         'layers': [], 'headers': [], 'files': {}}
        tensors = {}

        def write(name, size):
            data = bytes(size)
            (self.root / name).write_bytes(data)
            self.manifest['files'][name] = {'bytes': size, 'sha256': hashlib.sha256(data).hexdigest()}

        for layer in range(32):
            name = f'language_model.layers.{layer}.matrix.weight'
            tensors[name] = {'dtype': 'BF16', 'shape': [1, 64], 'data_offsets': [0, 128]}
            filename = f'q4-e4-layer-{layer:02}.bin'
            write(filename, 36)
            self.manifest['layers'].append({'file': filename, 'bytes': 36, 'tensors': [
                {'name': name, 'shape': [1, 64], 'start': 0, 'end': 36}]})
        tensors['language_model.embed_tokens.weight'] = {'dtype': 'BF16', 'shape': [2, 2560], 'data_offsets': [0, 10240]}
        tensors['language_model.norm.weight'] = {'dtype': 'BF16', 'shape': [2560], 'data_offsets': [0, 5120]}
        self.manifest['headers'] = [json.dumps(tensors), '{}']
        self.manifest['decoder_bytes'] = 32 * 36
        for key, size in [('embedding', 2880), ('readout', 255 * 1440), ('norm', 5120)]:
            name = f'q4-e4-{key}.bin'
            self.manifest[key] = name
            write(name, size)
        self.save()

    def save(self):
        (self.root / 'packed-q4-e4.json').write_text(json.dumps(self.manifest))

    def test_accepts_complete_layout(self):
        self.assertEqual(validate_packed(self.root)['decoder_bytes'], 32 * 36)

    def test_rejects_corrupt_bytes(self):
        (self.root / 'q4-e4-layer-07.bin').write_bytes(b'x' * 36)
        with self.assertRaisesRegex(ValueError, 'Checksum'):
            validate_packed(self.root)

    def test_rejects_missing_duplicate_and_overlapping_tensors(self):
        original = copy.deepcopy(self.manifest)
        for edit in ['missing', 'duplicate', 'overlap']:
            with self.subTest(edit=edit):
                self.manifest = copy.deepcopy(original)
                tensors = self.manifest['layers'][0]['tensors']
                if edit == 'missing':
                    tensors.clear()
                elif edit == 'duplicate':
                    tensors.append(copy.deepcopy(tensors[0]))
                else:
                    tensors[0]['start'] = 1
                self.save()
                with self.assertRaises(ValueError):
                    validate_packed(self.root)

    def test_rejects_wrong_source_and_extra_payload(self):
        for key, value in [('revision', '0' * 40), ('scheme', 'other')]:
            original = self.manifest[key]
            self.manifest[key] = value
            self.save()
            with self.assertRaises(ValueError):
                validate_packed(self.root)
            self.manifest[key] = original
        self.manifest['files']['unexpected.bin'] = {'bytes': 1, 'sha256': '0' * 64}
        self.save()
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            validate_packed(self.root)

    def test_rejects_traversal_and_symlink(self):
        with self.assertRaisesRegex(ValueError, 'Unsafe filename'):
            verify_files(self.root, {'../outside': {}})
        (self.root / 'link.bin').symlink_to('q4-e4-layer-00.bin')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            verify_files(self.root, {'link.bin': self.manifest['files']['q4-e4-layer-00.bin']})

    def test_independent_reference_rounding_and_zero(self):
        values = [7., -7., .5, -.5, 1.5, -1.5, 2.5, -2.5] * 8
        bf16 = b''.join(struct.pack('<f', n)[2:] for n in values)
        self.assertEqual(reference_group(bf16), bytes([0x1F, 0x88, 0x6A, 0x6A] * 8) + struct.pack('<f', 1.))
        self.assertEqual(reference_group(bytes(128)), bytes([0x88] * 32) + struct.pack('<f', 2**-126))

    def test_q8_reference_uses_signed_bytes(self):
        values = [127., -127., .5, -.5, 1.5, -1.5, 2.5, -2.5] * 8
        bf16 = b''.join(struct.pack('<f', n)[2:] for n in values)
        self.assertEqual(reference_group(bf16, 127), bytes([127, 129, 0, 0, 2, 254, 2, 254] * 8) + struct.pack('<f', 1.))
        self.assertEqual(reference_group(bytes(128), 127), bytes(64) + struct.pack('<f', 2**-126))


if __name__ == '__main__':
    unittest.main()
