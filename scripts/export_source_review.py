"""Build or verify an explicit source snapshot without media or secrets.

This script uses only Python's standard library. The media review ZIP is a
separate artifact: source snapshots intentionally contain no original source media,
derived videos, virtual environments, Git objects, local configuration, remote
URLs from Git, or raw diff content.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rinkspace.source-review.v1"
MANIFEST = "SOURCE_MANIFEST.json"
CHECKSUMS = "SHA256SUMS"
BLOCKED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", "outputs", ".private", ".codex", ".agents"}


def _json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _safe_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(ord(character) < 32 or ord(character) == 127 for character in name)):
        raise ValueError(f"Unsafe source-package path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or any(p in BLOCKED_PARTS for p in path.parts):
        raise ValueError(f"Forbidden source-package path: {name!r}")
    if name in {"configs/local.json", "configs/storage.json"} or path.parts[:2] == ("data", "input"):
        raise ValueError(f"Private configuration or original input is forbidden: {name}")
    return name


def _source_file(root, relative):
    _safe_name(relative)
    path = root / relative
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
        raise ValueError(f"Source file is missing or escapes project: {relative}")
    return path


def _selected_files(root, test_records):
    # Use the same Git candidate and local-artifact policy as publication review.
    # No filesystem walk can resurrect ignored tasks, machine roots or handoffs.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import audit_public_repository as inventory
    from check_github_readiness import tracked_review_reasons
    try:
        top = Path(inventory.decode(inventory.git(root, "rev-parse", "--show-toplevel")).strip()).resolve()
        if top != root:
            raise ValueError("Export requires a Git checkout rooted at --root")
        _, _, candidates = inventory.current_index_paths(root)
    except RuntimeError as error:
        raise ValueError("Export requires a Git checkout rooted at --root; --verify does not require Git") from error
    names = set()
    for name in candidates:
        path = inventory.allowed(root, name)
        if path is None or not path.is_file() or path.is_symlink() or tracked_review_reasons(name):
            continue
        try:
            _safe_name(name)
        except ValueError:
            continue
        if path.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
            raise ValueError("Original or derived video cannot enter a public source snapshot: " + name)
        names.add(name)
    names.difference_update({MANIFEST, CHECKSUMS, "SOURCE_REVIEW.md"})
    for required in ("README.md", "LICENSE.md", "pyproject.toml"):
        if required not in names:
            raise ValueError(f"Required public source entry is missing: {required}")
    for name in test_records or ():
        relative = PurePosixPath(_safe_name(name))
        if relative.parts[:2] != ("evidence", "tests") or relative.suffix.lower() not in {".json", ".xml", ".txt", ".log"}:
            raise ValueError("Test records must be explicit text records under evidence/tests/")
        if name not in names:
            raise ValueError("Test record is not a public Git candidate: " + name)
    return {name: _source_file(root, name) for name in sorted(names)}


def _git_metadata(root, included):
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False,
                              encoding="utf-8", errors="replace", timeout=30)
    try:
        top = git("rev-parse", "--show-toplevel")
        if top.returncode or Path(top.stdout.strip()).resolve() != root:
            return {"available": False, "reason": "No Git repository rooted at this source snapshot; parent or neighboring repositories are not consulted."}
        head = git("rev-parse", "--verify", "HEAD")
        if head.returncode:
            return {"available": False, "reason": "No readable Git HEAD; per-file SHA-256 remains authoritative."}
        # Do not capture author, email, remote URL, Git config, or patch text.
        diff = git("diff", "--no-renames", "--numstat", "-z", "HEAD", "--")
        untracked = git("ls-files", "--others", "--exclude-standard", "-z")
        if diff.returncode or untracked.returncode:
            return {"available": False, "head_commit": head.stdout.strip(), "reason": "Git working-tree summary unavailable."}
        changes = []
        for line in filter(None, diff.stdout.split("\0")):
            added, removed, name = line.split("\t", 2)
            if name in included:
                changes.append({"path": name, "added_lines": int(added) if added.isdigit() else None,
                                "removed_lines": int(removed) if removed.isdigit() else None})
        new = sorted(name for name in untracked.stdout.split("\0") if name in included)
        return {"available": True, "head_commit": head.stdout.strip(),
                "included_files_dirty": bool(changes or new), "tracked_diff_summary": changes,
                "untracked_included_files": new,
                "scope": "Only paths included in this snapshot; no raw diff, private paths, author identity, remote URLs, or Git objects."}
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "reason": "Git unavailable; per-file SHA-256 remains authoritative."}


def _guide():
    return """# 源码快照说明

本包保存当前公开 Git 候选文件（含必要公开数据），不是实片验收结果。
项目介绍、安装、合成 demo、基础测试及高级依赖统一见 [README.md](README.md)；
许可见 [LICENSE.md](LICENSE.md)，本机资源准备见 [MAINTENANCE.md](MAINTENANCE.md)。

无需安装项目或第三方包，可在解压目录运行：
~~~console
python scripts/export_source_review.py --verify .
~~~
此命令只用 Python 标准库，核对路径、清单和文件 SHA，不要求 Git。
创建新快照需要 Git；导出采用工作区候选字节，可能包含未提交修改。
SOURCE_MANIFEST.json 记录来源和身份，SHA256SUMS 同时覆盖清单。
完整性校验不是签名、许可或科学验收；原始比赛视频、截图和本机安装不随包提供。
""".encode("utf-8")


def export_source_review(root, out=None, test_records=None, limit_mib=None):
    root = Path(root).resolve(strict=True)
    # Lazy source loading keeps --help/--verify standard-library-only.
    sys.path.insert(0, str(ROOT / "src"))
    from rinkspace.storage import AssetStore
    store = AssetStore(root)
    output = store.output("source_review") if store.configured else root / "outputs"
    allowed_output = store.data if store.configured else output
    if store.configured and (allowed_output.is_relative_to(root) or root.is_relative_to(allowed_output)):
        raise ValueError("Data and source checkout must be separate")
    out = Path(out) if out is not None else output / "source_review.zip"
    if not out.is_absolute():
        out = output / out
    if out.is_symlink() or not out.resolve().is_relative_to(allowed_output.resolve()) or out.suffix.lower() != ".zip":
        raise ValueError("Source-review destination must be a ZIP within configured Data (or local outputs without Data)")
    if limit_mib is not None and limit_mib <= 0:
        raise ValueError("Size limit must be positive")
    sources = _selected_files(root, test_records)
    entries = []
    manifest = {"schema_version": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "entrypoint": "README.md", "source_media_included": False,
                "scope": "Current public Git candidate snapshot, including public data; separate from source media and scientific acceptance.",
                "git": _git_metadata(root, set(sources)), "files": entries,
                "excluded": ["data/input", "configs/local.json", "configs/storage.json", "outputs", ".venv", ".git objects/config/remotes", "neighboring projects"],
                "test_records": sorted(name for name in sources if name.startswith('evidence/tests/')),
                "generated_entry_note": "Installation and testing instructions remain in the repository README; no second HTML entry is generated.",
                "limit_mib": limit_mib}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".source-review-", suffix=".zip", dir=out.parent, delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in sources.items():
                before = path.stat()
                digest, length = hashlib.sha256(), 0
                with path.open("rb") as source, archive.open(name, "w", force_zip64=True) as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
                        digest.update(block)
                        length += len(block)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    raise ValueError("Source changed during export: " + name)
                entries.append(dict(path=name, sha256=digest.hexdigest(), size_bytes=length, origin="repository_verbatim"))
            guide = _guide()
            archive.writestr("SOURCE_REVIEW.md", guide)
            entries.append(dict(path="SOURCE_REVIEW.md", sha256=_hash(guide), size_bytes=len(guide), origin="generated_source_package_entry"))
            entries.sort(key=lambda row: row["path"])
            manifest_bytes = _json(manifest)
            archive.writestr(MANIFEST, manifest_bytes)
            checks = {row["path"]: row["sha256"] for row in entries}
            checks[MANIFEST] = _hash(manifest_bytes)
            archive.writestr(CHECKSUMS, "".join(f"{digest}  {name}\n" for name, digest in sorted(checks.items())).encode("utf-8"))
        size = temporary.stat().st_size
        if limit_mib is not None and size > limit_mib * 1024 ** 2:
            raise ValueError(f"Source package exceeds {limit_mib} MiB ({size} bytes); no files were silently omitted")
        verified = verify_source_review(temporary)
        temporary.replace(out)
        temporary = None
        return {"path": str(out), "size_bytes": size, "file_count": len(entries) + 2,
                "head_commit": manifest['git'].get('head_commit'), "verified": verified['verified']}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_source_review(path):
    """Verify a ZIP before extracting, or verify declared files after extraction."""
    path = Path(path).resolve(strict=True)
    archive = zipfile.ZipFile(path) if path.is_file() else None
    try:
        if archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate archive members")
            for name in names:
                _safe_name(name)
            read = archive.read
        else:
            read = lambda name: _source_file(path, name).read_bytes()
        manifest = json.loads(read(MANIFEST))
        if manifest.get('schema_version') != SCHEMA or manifest.get('source_media_included') is not False:
            raise ValueError("Unrecognized source manifest")
        checksums = {}
        for line in read(CHECKSUMS).decode('utf-8').splitlines():
            digest, name = line.split('  ', 1)
            _safe_name(name)
            if name in checksums:
                raise ValueError("Duplicate checksum path")
            checksums[name] = digest
        expected = {_safe_name(row['path']) for row in manifest['files']} | {MANIFEST}
        if len(expected) != len(manifest['files']) + 1 or set(checksums) != expected:
            raise ValueError("Manifest/checksum file-set mismatch")
        if archive and set(names) != expected | {CHECKSUMS}:
            raise ValueError("Archive has undeclared or missing files")
        for name, digest in checksums.items():
            if _hash(read(name)) != digest:
                raise ValueError(f"SHA-256 mismatch: {name}")
        for row in manifest['files']:
            data = read(row['path'])
            if len(data) != row['size_bytes'] or _hash(data) != row['sha256']:
                raise ValueError(f"Manifest content mismatch: {row['path']}")
        return {"verified": True, "file_count": len(expected) + 1,
                "scope": "Declared source snapshot integrity; no scientific accuracy claim."}
    finally:
        if archive:
            archive.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--test-record', action='append', default=None,
                        help='Explicit project-relative text test record under evidence/tests; repeatable')
    parser.add_argument('--limit-mib', type=float, default=None, help='Optional archive size limit in MiB')
    parser.add_argument('--verify', type=Path, help='Verify a ZIP or extracted source snapshot; no dependencies or Git required')
    args = parser.parse_args(argv)
    try:
        result = verify_source_review(args.verify) if args.verify else export_source_review(
            args.root, args.out, args.test_record, args.limit_mib)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        parser.exit(2, f'Source review failed: {error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
