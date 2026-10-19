"""Read byte-exact project assets kept in compact, registered ZIP bundles.

Logical identities stay in OneDrive. Only a consumer requiring a real Path
materializes a verified disposable copy beneath the configured external Data.
No global pathlib hooks or source-document rewrites are used.
"""
from __future__ import annotations
from functools import lru_cache
import atexit
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

DEFAULT_PROJECT = Path(__file__).resolve().parents[2]


def _bounded(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative.replace('\\', '/'))
    if p.is_absolute() or '..' in p.parts or ':' in relative or '\x00' in relative:
        raise ValueError('Packed asset path must stay inside the project')
    target = (root / str(p)).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Packed asset path escapes the project')
    return target


@lru_cache(maxsize=4)
def _registry(root_string: str, signature: tuple) -> dict:
    root = Path(root_string)
    config = root / 'configs/packed_assets.json'
    if not config.is_file():
        return {}
    value = json.loads(config.read_text(encoding='utf-8-sig'))
    if value.get('schema') != 'rinkspace.packed-assets-registry.v1':
        raise ValueError('Unsupported packed asset registry')
    result = {}
    for index_name in value['indexes']:
        index = _bounded(root, index_name)
        doc = json.loads(index.read_text(encoding='utf-8-sig'))
        if doc.get('schema') != 'rinkspace.packed-assets.v1':
            raise ValueError('Unsupported packed asset index')
        archive = _bounded(root, doc['archive'])
        for row in doc['assets']:
            logical = _bounded(root, row['path'])
            member = row['member']
            if (PurePosixPath(member).is_absolute() or '..' in PurePosixPath(member).parts
                    or '\\' in member or ':' in member or '\x00' in member):
                raise ValueError('Unsafe packed asset member')
            if len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256']):
                raise ValueError('Invalid packed asset identity')
            if row['bytes'] < 0 or str(logical).casefold() in result:
                raise ValueError('Invalid or duplicate packed asset')
            result[str(logical).casefold()] = dict(row, archive_path=archive, logical_path=logical)
    return result


def _entries(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    config = root / 'configs/packed_assets.json'
    if not config.is_file():
        return {}
    stat = config.stat()
    value = json.loads(config.read_text(encoding='utf-8-sig'))
    signatures = [(stat.st_size, stat.st_mtime_ns)]
    for name in value.get('indexes', []):
        meta = _bounded(root, name).stat()
        signatures.append((meta.st_size, meta.st_mtime_ns))
    return _registry(str(root), tuple(signatures))


def _logical(path: str | Path, project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    p = Path(path)
    return (p if p.is_absolute() else root / p).resolve()


def packed_asset_reference(path: str | Path, *, project_root=DEFAULT_PROJECT) -> dict | None:
    return _entries(project_root).get(str(_logical(path, project_root)).casefold())


@lru_cache(maxsize=8)
def _archive(path: str, size: int, modified_ns: int):
    # Frozen archives are shared by many small observations. Reuse the parsed
    # central directory, while read_project_bytes still checks each raw SHA.
    handle = zipfile.ZipFile(path)
    atexit.register(handle.close)
    return handle


def read_project_bytes(path: str | Path, *, project_root=DEFAULT_PROJECT) -> bytes:
    logical = _logical(path, project_root)
    if logical.is_file():
        return logical.read_bytes()
    row = packed_asset_reference(logical, project_root=project_root)
    if row is None:
        raise FileNotFoundError(logical)
    meta = row['archive_path'].stat()
    archive = _archive(str(row['archive_path']), meta.st_size, meta.st_mtime_ns)
    info = archive.getinfo(row['member'])
    if info.file_size != row['bytes']:
        raise ValueError('Packed asset size changed')
    value = archive.read(info)
    if hashlib.sha256(value).hexdigest() != row['sha256']:
        raise ValueError('Packed asset identity changed')
    return value


def read_project_json(path: str | Path, *, project_root=DEFAULT_PROJECT):
    return json.loads(read_project_bytes(path, project_root=project_root).decode('utf-8-sig'))


def materialize_project_asset(path: str | Path, *, project_root=DEFAULT_PROJECT) -> Path:
    root = Path(project_root).resolve()
    logical = _logical(path, root)
    if logical.is_file():
        return logical
    row = packed_asset_reference(logical, project_root=root)
    if row is None:
        return logical
    config = json.loads((root / 'configs/storage.json').read_text(encoding='utf-8-sig'))
    configured_data = Path(config['data_root'])
    if not configured_data.is_absolute():
        raise ValueError('Packed assets require an absolute Data root')
    data = configured_data.resolve()
    if data.is_relative_to(root):
        raise ValueError('Packed assets require a separate external Data root')
    target = data / 'cache/packed_assets' / (row['sha256'] + logical.suffix)
    if target.is_file():
        if target.stat().st_size == row['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest() == row['sha256']:
            return target
        raise ValueError('Materialized packed asset identity changed')
    value = read_project_bytes(logical, project_root=root)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix='packed_', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(value)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def iter_project_assets(directory: str | Path, pattern: str='*', *, project_root=DEFAULT_PROJECT) -> list[Path]:
    directory = _logical(directory, project_root)
    result = {str(p).casefold(): p for p in directory.glob(pattern) if p.is_file()}
    for row in _entries(project_root).values():
        logical = row['logical_path']
        if logical.is_relative_to(directory):
            relative = logical.relative_to(directory)
            if ((len(relative.parts) == 1 and relative.match(pattern))
                    or ('/' in pattern and relative.match(pattern))):
                result.setdefault(str(logical).casefold(), logical)
    return sorted(result.values())
