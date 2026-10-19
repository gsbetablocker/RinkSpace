"""Original source cache authentication rejects changed identities and pixels."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from rinkspace.native_source_cache import (
    AuthenticatedNativeSourceCache, _sha, authenticate_missing_native_png)


class NativeSourceCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'original.mp4'
        self.source.write_bytes(b'original video fixture')
        self.pixels = np.full((1080, 1920, 3), (7, 11, 19), np.uint8)
        self.aliases = {256: 256, 512: 256, 768: 768}
        self.raw = self.root / 'raw.png'
        self.assertTrue(cv2.imwrite(str(self.raw), self.pixels))
        self.manifest = self.root / 'receipt.json'
        self.cache = dict(source_video=dict(path=str(self.source), sha256=_sha(self.source), time_base='1/15360'),
            frames=[dict(native_pts=256, path=str(self.raw), sha256=_sha(self.raw),
                native_bgr24_sha256=hashlib.sha256(self.pixels.tobytes()).hexdigest())])
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.cache), encoding='utf8')

    def reader(self, manifests=None):
        return AuthenticatedNativeSourceCache(manifests or [self.manifest],
            source_path=self.source, source_sha256=_sha(self.source), source_time_base='1/15360', aliases=self.aliases)

    def test_exact_native_read_never_substitutes_actual_alias_pixels(self):
        reader = self.reader()
        pixels, row = reader.read(256)
        self.assertTrue(np.array_equal(pixels, self.pixels))
        self.assertEqual(row['requested_native_pts'], 256)
        with self.assertRaisesRegex(ValueError, 'omits'):
            reader.read(512)

    def test_different_native_presentations_with_same_actual_can_differ(self):
        other = self.root / 'other.png'
        image = self.pixels.copy(); image[0, 0, 0] += 1
        self.assertTrue(cv2.imwrite(str(other), image))
        self.cache['frames'].append(dict(native_pts=512, actual_pts=256, path=str(other),
            sha256=_sha(other), native_bgr24_sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        self.write_manifest()
        reader = self.reader()
        self.assertTrue(np.array_equal(reader.read(512)[0], image))

    def test_source_sha_clock_and_alias_must_match(self):
        for key, value in (('sha256', '0' * 64), ('time_base', '1/60')):
            with self.subTest(key=key):
                original = self.cache['source_video'][key]
                self.cache['source_video'][key] = value; self.write_manifest()
                with self.assertRaisesRegex(ValueError, 'identity/clock'):
                    self.reader()
                self.cache['source_video'][key] = original
        self.cache['frames'][0]['actual_pts'] = 768; self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'alias differs'):
            self.reader()

    def test_changed_png_and_changed_decoded_BGR_rejected(self):
        reader = self.reader()
        self.raw.write_bytes(self.raw.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'PNG changed'):
            reader.read(256)
        self.cache['frames'][0]['sha256'] = _sha(self.raw)
        self.cache['frames'][0]['native_bgr24_sha256'] = '0' * 64; self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'BGR changed'):
            self.reader().read(256)

    def test_duplicate_conflict_and_manifest_mutation_rejected(self):
        reader = self.reader()
        self.manifest.write_text(self.manifest.read_text('utf8') + '\n', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            reader.assert_unchanged()
        other = self.root / 'second.json'
        changed = json.loads(json.dumps(self.cache))
        changed['frames'][0]['native_bgr24_sha256'] = '0' * 64
        other.write_text(json.dumps(changed), encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'pixels disagree'):
            self.reader([self.manifest, other])


class MissingNativePNGTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'original.mp4'
        self.source.write_bytes(b'complete original source fixture')
        self.missing = self.root / '256_native_original.png'
        self.manifest = self.root / 'complete_original_native_cache_manifest.json'
        self.pixels = np.full((1080, 1920, 3), (7, 11, 19), np.uint8)
        ok, png = cv2.imencode('.png', self.pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        self.assertTrue(ok)
        self.row = dict(native_pts=256, actual_pts=256, raw_png=str(self.missing),
            raw_png_sha256=hashlib.sha256(png.tobytes()).hexdigest(),
            source_BGR_sha256=hashlib.sha256(self.pixels.tobytes()).hexdigest())
        self.cache = dict(schema='rinkspace.complete-secondary-lossless-native-source-cache.v1',
            source=dict(path=str(self.source), sha256=_sha(self.source)), native_timebase='1/15360', frames=[self.row])
        self.write_manifest()
        self.stream = SimpleNamespace(thread_count=0, time_base='1/15360')
        self.frame = SimpleNamespace(pts=256, to_ndarray=lambda **_: self.pixels)
        self.container = SimpleNamespace(streams=SimpleNamespace(video=[self.stream]),
            seek=lambda *_, **__: None, decode=lambda _: iter([self.frame]))
        class Open:
            def __enter__(inner): return self.container
            def __exit__(inner, *_): return False
        self.fake_av = SimpleNamespace(open=lambda _: Open())

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.cache), encoding='utf8')

    def reproduce(self, **kwargs):
        with patch.dict('sys.modules', av=self.fake_av):
            return authenticate_missing_native_png(self.missing, **kwargs)

    def test_exact_reproduction_authenticates_files_without_writes(self):
        before = set(self.root.iterdir()); authenticated = []
        def auth(path): authenticated.append(Path(path)); return _sha(path)
        digest, receipt = self.reproduce(authenticate_file=auth)
        self.assertEqual(digest, self.row['raw_png_sha256'])
        self.assertEqual(authenticated, [self.manifest, self.source])
        self.assertEqual(receipt['new_source_images_written'], 0)
        self.assertEqual(before, set(self.root.iterdir()))
        self.assertFalse(self.missing.exists())

    def test_wrong_source_BGR_or_PNG_SHA_rejected(self):
        for target, key in ((self.cache['source'], 'sha256'), (self.row, 'source_BGR_sha256'), (self.row, 'raw_png_sha256')):
            with self.subTest(key=key):
                original = target[key]; target[key] = '0' * 64; self.write_manifest()
                with self.assertRaises(ValueError): self.reproduce()
                target[key] = original

    def test_missing_exact_native_pts_and_wrong_clock_rejected(self):
        self.frame.pts = 512
        with self.assertRaisesRegex(ValueError, 'presentation is unavailable'): self.reproduce()
        self.frame.pts = 256; self.stream.time_base = '1/60'
        with self.assertRaisesRegex(ValueError, 'time base changed'): self.reproduce()

    def test_duplicate_row_and_unsupported_missing_rejected(self):
        self.cache['frames'].append(dict(self.row)); self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'one exact original cache row'): self.reproduce()
        self.manifest.unlink()
        with self.assertRaises(FileNotFoundError): self.reproduce()

    def test_changed_dependency_during_reproduction_rejected(self):
        def auth(path):
            digest = _sha(path)
            if Path(path) == self.source: self.source.write_bytes(self.source.read_bytes() + b'changed')
            return digest
        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            self.reproduce(authenticate_file=auth)


if __name__ == '__main__':
    unittest.main()
