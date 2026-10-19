"""Missing disposable PNG authentication keeps original source SHA gates."""
import hashlib
import os

import pytest

from calibration.solver_runtime import startup_source_auth_reuse_runtime as runtime
from rinkspace import native_source_cache


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_file_still_uses_original_and_revalidates_hits(tmp_path, monkeypatch):
    image = tmp_path / 'original.png'
    image.write_bytes(b'original encoded PNG')
    calls = []
    def original(path):
        calls.append(path)
        return _sha(path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Existing PNG must use original authentication')
    monkeypatch.setattr(native_source_cache, 'authenticate_missing_native_png', forbidden)
    with runtime.SourceAuthenticationScope(tmp_path / 'output') as scope:
        assert scope.digest(image, original) == _sha(image)
        assert scope.digest(image, original) == _sha(image)
    assert len(calls) == 2
    assert scope.hits == 1
    assert scope.missing_native_png_authentications == []


def test_missing_png_uses_real_dependency_sha_without_writing_image(tmp_path, monkeypatch):
    missing = tmp_path / 'original.png'
    manifest = tmp_path / 'manifest.json'
    source = tmp_path / 'source.mp4'
    manifest.write_bytes(b'original manifest')
    source.write_bytes(b'original movie')
    digest = hashlib.sha256(b'exact reconstructed PNG').hexdigest()
    def authenticate(path, *, resolve_source, authenticate_file):
        assert path == missing
        assert authenticate_file(manifest) == _sha(manifest)
        assert authenticate_file(source) == _sha(source)
        return digest, dict(path=str(path), source_sha256=_sha(source), raw_png_sha256=digest)
    monkeypatch.setattr(native_source_cache, 'authenticate_missing_native_png', authenticate)
    def original(path):
        raise AssertionError('Missing disposable file must not be read')
    with runtime.SourceAuthenticationScope(tmp_path / 'output') as scope:
        assert scope.digest(missing, original) == digest
        assert scope.digest(missing, original) == digest
    assert not missing.exists()
    assert len(scope.missing_native_png_authentications) == 2
    assert scope.revalidated_files == 2
    assert scope.closed and not scope.files


def test_changed_dependency_is_rejected_at_scope_exit(tmp_path, monkeypatch):
    missing = tmp_path / 'original.png'
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'original')
    before = source.stat()
    def authenticate(path, *, resolve_source, authenticate_file):
        authenticate_file(source)
        authenticate_file(source)
        return 'a' * 64, dict(path=str(path))
    monkeypatch.setattr(native_source_cache, 'authenticate_missing_native_png', authenticate)
    with pytest.raises(ValueError, match='source changed'):
        with runtime.SourceAuthenticationScope(tmp_path / 'output') as scope:
            scope.digest(missing, _sha)
            source.write_bytes(b'changed!')
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not missing.exists()
    assert scope.closed and runtime._ACTIVE.get() is None


def test_unsupported_missing_reference_preserves_failure(tmp_path, monkeypatch):
    missing = tmp_path / 'unrecorded.png'
    def authenticate(path, **kwargs):
        raise FileNotFoundError(path)
    monkeypatch.setattr(native_source_cache, 'authenticate_missing_native_png', authenticate)
    with pytest.raises(FileNotFoundError):
        with runtime.SourceAuthenticationScope(tmp_path / 'output') as scope:
            scope.digest(missing, _sha)
    assert scope.closed and not scope.missing_native_png_authentications
