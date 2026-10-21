"""Build or verify an explicit source snapshot without media or secrets.

This script uses only Python's standard library. The media review ZIP is a
separate artifact: source snapshots intentionally contain no original inputs,
derived videos, virtual environments, Git objects, local configuration, remote
URLs from Git, or raw diff content.
"""
from __future__ import annotations
from rinkspace.storage import resolve_asset as _resolve_asset

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rinkspace.source-review.v1"
MANIFEST = "SOURCE_MANIFEST.json"
CHECKSUMS = "SHA256SUMS"
ROOT_FILES = ("README.md", "pyproject.toml", "START_LOCAL.cmd", "INSTALL_LOCAL.cmd",
              "AGENTS.md", "START_CODEX.md", ".gitignore", ".gitattributes")
CONFIG_FILES = ("project.example.json", "rink_model.example.json", "benchmark_windows.json")
TREE_TYPES = {
    "src": {".py", ".html", ".css", ".js"}, "tests": {".py"}, "scripts": {".py"},
    "docs": {".md"}, "tasks": {".md"},
}
DEFAULT_TEST_RECORDS = (
    "evidence/tests/broadcast_correction_tests.json",
    "evidence/tests/broadcast_delivery_ui.json",
    "evidence/tests/delivery_a_junit.xml",
    "evidence/tests/long_shot_review.json",
    "evidence/tests/wide_static_protected_review.json",
)
LOCAL_DOCUMENTS = {"docs/REPOSITORY_GUIDE.md", "docs/PUBLIC_REPOSITORY_CONTENTS.md",
                   "docs/PRIVACY_REVIEW.md", "docs/RUNTIME_DEPENDENCIES.md", "docs/rink_model/VALIDATION.md"}
BLOCKED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", "outputs", "models"}


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
    if name == "configs/local.json" or path.parts[:2] == ("data", "input"):
        raise ValueError(f"Private configuration or original input is forbidden: {name}")
    return name


def _source_file(root, relative):
    _safe_name(relative)
    path = root / relative
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
        raise ValueError(f"Source file is missing or escapes project: {relative}")
    return path


def _selected_files(root, test_records):
    names = {name for name in ROOT_FILES if (root / name).is_file()}
    for required in ("README.md", "pyproject.toml", "START_LOCAL.cmd", "INSTALL_LOCAL.cmd"):
        if required not in names:
            raise ValueError(f"Required source entry is missing: {required}")
    for tree, suffixes in TREE_TYPES.items():
        for path in (root / tree).rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                relative = path.relative_to(root).as_posix()
                if not any(part in BLOCKED_PARTS for part in PurePosixPath(relative).parts):
                    names.add(relative)
    if not any(name.startswith("src/") for name in names) or not any(name.startswith("tests/") for name in names):
        raise ValueError("A source review needs both src/ and tests/ files")
    names.update("configs/" + name for name in CONFIG_FILES if (root / "configs" / name).is_file())
    # Explicit source-only annotation inputs, never video/model/iteration caches.
    for name in ('manifest.json','jin_inputs.json','okada_inputs.json','jin_observations.json.gz','okada_observations.json.gz'):
        relative = 'configs/continuous_inputs/' + name
        if (root / relative).is_file():
            names.add(relative)
    names.update(path.relative_to(root).as_posix() for path in (root / "configs").glob("*lock*.json") if path.is_file())
    if "configs/project.example.json" not in names:
        raise ValueError("Required public default configuration is missing")
    names.update(path.name for path in root.glob("requirements*.txt") if path.is_file())
    selected_records = list(test_records) if test_records is not None else [
        name for name in DEFAULT_TEST_RECORDS if (root / name).is_file()]
    for name in selected_records:
        relative = PurePosixPath(_safe_name(name))
        if relative.parts[:2] != ("evidence", "tests") or relative.suffix.lower() not in {".json", ".xml", ".txt", ".log"}:
            raise ValueError("Test records must be explicit text records under evidence/tests/")
        _source_file(root, name)
        names.add(name)
    return {name: _source_file(root, name) for name in sorted(names) if name not in LOCAL_DOCUMENTS}


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
        selected = sorted(included)
        diff = git("diff", "--no-renames", "--numstat", "-z", "HEAD", "--", *selected)
        untracked = git("ls-files", "--others", "--exclude-standard", "-z", "--", *selected)
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
    return """# RinkSpace 源码审阅包

这个包提供当前源码、测试、公开配置、锁定记录和测试记录，供独立审阅与复现。
它不是实片验收结果；`review.zip` 是另一个媒体证据包。没有输入原片、虚拟环境、
本机 `configs/local.json` 或 `.git` 数据库。README 和 docs 是该次源码快照的项目说明；
其中提到的本机已安装环境和媒体输出不会自动出现在本包。

## 不安装第三方依赖的检查（Python 3.11+）

在解压目录运行：

```
python scripts/export_source_review.py --verify .
python -m unittest discover -s tests -p test_source_review.py -v
```

第一条校验全部源文件 SHA-256 与安全路径；第二条检查打包排除、篡改检测和源码快照
边界，不依赖原片、NumPy、OpenCV、PyAV 或 pytest。它们不证明配准精度。

## 安装并运行合成 / 临时媒体测试

```
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -e ".[dev]"
.venv\\Scripts\\python.exe -m pytest -q
```

macOS/Linux 将 `.venv\\Scripts\\python.exe` 换成 `.venv/bin/python`。
安装依赖需要可用的软件包来源。测试自行创建临时夹具，不需要用户视频；系统未安装
FFmpeg/ffprobe 时，旧 CLI 解码测试会明确跳过。测试记录对应记录中注明的版本与时间，
不能视为后来源码改动已重新测试的证明。

## 启动工作台

安装后运行 `START_LOCAL.cmd`，或 `python -m rinkspace serve --root . --port 8765 --open`。
没有原片时工作台会显示缺少输入；须导入自己的授权本地视频才能做实片分析。可在
安装时另加 `[acquisition]` 以使用包内获取脚本。原片不上传。

`SOURCE_MANIFEST.json` 区分原样复制的文件与本包生成的入口，保存 Git HEAD、已打包
路径的 dirty diff 行数摘要和各文件哈希。源码快照可能含未提交修改，具体以清单为准。
`SHA256SUMS` 同时覆盖清单自身；摘要只证明完整性，不是签名或科学验收。
""".encode("utf-8")


def export_source_review(root, out=None, test_records=None, limit_mib=None):
    root = Path(root).resolve(strict=True)
    output = _resolve_asset(root, root / "outputs")
    output.mkdir(exist_ok=True)
    if not output.resolve().is_relative_to(root):
        raise ValueError("Project outputs directory must not redirect outside the project")
    out = Path(out) if out is not None else output / "source_review.zip"
    if not out.is_absolute():
        out = root / out
    if out.is_symlink() or not out.resolve().is_relative_to(output.resolve()) or out.suffix.lower() != ".zip":
        raise ValueError("Source-review destination must be a ZIP within this project's outputs/")
    if limit_mib is not None and limit_mib <= 0:
        raise ValueError("Size limit must be positive")
    sources = _selected_files(root, test_records)
    payload = {name: path.read_bytes() for name, path in sources.items()}
    payload["SOURCE_REVIEW.md"] = _guide()
    payload["START_HERE.html"] = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
        "<title>RinkSpace 源码审阅包</title><style>body{font:17px/1.8 system-ui;max-width:880px;margin:50px auto;padding:20px}"
        "pre{white-space:pre-wrap;background:#eef4f4;padding:20px}</style><h1>RinkSpace 源码审阅包</h1>"
        "<p>这个入口属于轻量源码包。实片证据在单独的 review.zip；本包不含原片或安装环境。</p>"
        "<p><a href='SOURCE_REVIEW.md'>完整安装与验证说明</a> · <a href='SOURCE_MANIFEST.json'>快照与哈希清单</a></p>"
        "<h2>无需第三方依赖，先检查源码包</h2><pre>python scripts/export_source_review.py --verify .\n"
        "python -m unittest discover -s tests -p test_source_review.py -v</pre>"
        "<h2>安装与合成测试</h2><pre>python -m venv .venv\n"
        + html.escape('.venv\\Scripts\\python.exe -m pip install -e ".[dev]"')
        + "\n.venv\\Scripts\\python.exe -m pytest -q</pre><p>安装后运行 START_LOCAL.cmd。没有输入原片时，"
        "工作台显示缺少媒体；合成测试不需要原片。测试通过不等于实片配准精度通过。</p></html>").encode("utf-8")
    entries = [{"path": name, "sha256": _hash(data), "size_bytes": len(data),
                "origin": "repository_verbatim" if name in sources else "generated_source_package_entry"}
               for name, data in sorted(payload.items())]
    manifest = {"schema_version": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "entrypoint": "START_HERE.html", "source_media_included": False,
                "scope": "Source snapshot, tests and explicit text evidence; separate from media review and scientific acceptance.",
                "git": _git_metadata(root, set(sources)), "files": entries,
                "excluded": ["data/input", "configs/local.json", "outputs", ".venv", ".git objects/config/remotes", "unselected evidence", "neighboring projects"],
                "test_records": sorted(name for name in sources if name.startswith('evidence/tests/')),
                "generated_entry_note": "START_HERE.html is a source-package-specific entry, not the repository media-report redirect.",
                "limit_mib": limit_mib}
    payload[MANIFEST] = _json(manifest)
    payload[CHECKSUMS] = "".join(f"{_hash(data)}  {name}\n" for name, data in sorted(payload.items())).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".source-review-", suffix=".zip", dir=out.parent, delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in sorted(payload.items()):
                archive.writestr(name, data)
        size = temporary.stat().st_size
        if limit_mib is not None and size > limit_mib * 1024 ** 2:
            raise ValueError(f"Source package exceeds {limit_mib} MiB ({size} bytes); no files were silently omitted")
        verified = verify_source_review(temporary)
        temporary.replace(out)
        temporary = None
        return {"path": str(out), "size_bytes": size, "file_count": len(payload),
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
