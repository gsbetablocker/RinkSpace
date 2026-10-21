"""Standard-library-only package checks; no original media or runtime deps."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_source_review', ROOT / 'scripts/export_source_review.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class SourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        files = {'README.md': '# Test project', 'pyproject.toml': '[project]\nname="fixture"',
                 'START_LOCAL.cmd': '@echo off', 'INSTALL_LOCAL.cmd': '@echo off',
                 'src/fixture.py': 'value = 1\n', 'tests/test_fixture.py': 'assert True\n',
                 'docs/ARCHITECTURE.md': '# Public architecture', 'docs/PRIVACY_REVIEW.md': 'LOCAL_WORK_RECORD',
                 'configs/project.example.json': '{"sources":{}}', 'configs/holdout_lock.json': '{}',
                 'configs/local.json': '{"private":"DO_NOT_PACKAGE"}',
                 'data/input/source.mp4': 'DO_NOT_PACKAGE', '.venv/private.py': 'DO_NOT_PACKAGE',
                 'outputs/private.json': 'DO_NOT_PACKAGE', '.git/private': 'DO_NOT_PACKAGE',
                 'evidence/tests/current.json': '{"passed":1}'}
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')

    def export(self, **kwargs):
        return exporter.export_source_review(self.root, test_records=['evidence/tests/current.json'], **kwargs)

    def test_explicit_snapshot_excludes_inputs_private_config_and_git_objects(self):
        result = self.export()
        with zipfile.ZipFile(result['path']) as archive:
            names = set(archive.namelist())
            self.assertIn('src/fixture.py', names)
            self.assertIn('docs/ARCHITECTURE.md', names)
            self.assertNotIn('docs/PRIVACY_REVIEW.md', names)
            self.assertIn('configs/holdout_lock.json', names)
            self.assertIn('evidence/tests/current.json', names)
            for forbidden in ['configs/local.json', 'data/input/source.mp4', '.venv/private.py',
                              'outputs/private.json', '.git/private']:
                self.assertNotIn(forbidden, names)
            self.assertTrue(all(b'DO_NOT_PACKAGE' not in archive.read(name) for name in names))
            manifest = json.loads(archive.read(exporter.MANIFEST))
            self.assertFalse(manifest['source_media_included'])
        self.assertTrue(exporter.verify_source_review(result['path'])['verified'])

    def test_extracted_source_tampering_is_detected(self):
        result = self.export()
        extracted = self.root / 'unpacked'
        with zipfile.ZipFile(result['path']) as archive:
            archive.extractall(extracted)  # The archive above was generated and verified locally.
        self.assertTrue(exporter.verify_source_review(extracted)['verified'])
        (extracted / 'src/fixture.py').write_text('value = 999\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            exporter.verify_source_review(extracted)

    def test_record_escape_and_oversize_do_not_publish(self):
        with self.assertRaises(ValueError):
            exporter.export_source_review(self.root, test_records=['../secret.json'])
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.export(limit_mib=0.00001)
        self.assertFalse((self.root / 'outputs/source_review.zip').exists())

    def test_archive_with_undeclared_file_is_rejected(self):
        result = self.export()
        with zipfile.ZipFile(result['path'], 'a') as archive:
            archive.writestr('extra.txt', 'not in manifest')
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            exporter.verify_source_review(result['path'])


if __name__ == '__main__':
    unittest.main()
