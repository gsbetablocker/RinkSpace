"""Read-only repository inventory. Never import or start a RinkSpace runtime."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from rinkspace.storage import load_storage_config, configured_data_root
EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
CODE_SUFFIXES = {".py", ".ps1", ".cmd", ".bat", ".c", ".cpp", ".h", ".cu"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bounded(root: Path, relative: str, *, allowed_root: Path | None = None) -> Path:
    normalized = relative.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute() or PureWindowsPath(normalized).drive:
        raise ValueError(f"Manifest path is not relative: {relative}")
    result = (root / candidate).resolve()
    if not result.is_relative_to((allowed_root or root).resolve()):
        raise ValueError(f"Manifest path leaves its source root: {relative}")
    return result


def inventory(root: Path):
    """Do not follow directory links or read virtual-environment/Git contents."""
    pending = [root]
    files, unreadable = [], []
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name)
        except OSError as error:
            unreadable.append({"path": str(directory), "error": str(error)})
            continue
        for path in children:
            if path.name in EXCLUDED or path.name.endswith(".egg-info"):
                continue
            try:
                if path.is_symlink():
                    unreadable.append({"path": str(path), "error": "link not followed"})
                elif path.is_dir():
                    pending.append(path)
                elif path.is_file():
                    files.append((path.relative_to(root).as_posix(), path.stat().st_size))
            except OSError as error:
                unreadable.append({"path": str(path), "error": str(error)})
    return sorted(files), unreadable


def check_sources(root: Path):
    result = []
    package = root / "calibration/solver_runtime"
    for filename in ("source_manifest.json", "event_factories_manifest.json"):
        manifest_path = package / filename
        if not manifest_path.is_file():
            result.append({"manifest": filename, "status": "missing"})
            continue
        manifest = read_json(manifest_path)
        rows = list(manifest.get("modules", {}).values()) + manifest.get("native_sources", [])
        if filename == "event_factories_manifest.json":
            # The event manifest uses a separate list of exact source aliases.
            rows = manifest.get("owned_sources", manifest.get("sources", []))
            if not rows:
                rows = list(manifest.get("modules", {}).values())
        checked, failures = 0, []
        for row in rows:
            path = bounded(package, row["path"], allowed_root=root)
            try:
                actual = digest(path)
                if actual != row["sha256"]:
                    failures.append({"path": row["path"], "status": "sha256_mismatch"})
            except OSError as error:
                failures.append({"path": row["path"], "status": "unreadable", "error": str(error)})
            checked += 1
        result.append({"manifest": filename, "sha256": digest(manifest_path),
                       "checked": checked, "failures": failures,
                       "status": ("identity_failures" if failures else "verified") if rows else "unrecognized_manifest_layout"})
    return result


def input_references(root: Path, data_root: Path):
    """Declared profile paths only; existence does not prove full dependency closure."""
    references = []
    old_data = load_storage_config(root).get("data_root", "")
    prefix = old_data.replace("\\", "/").rstrip("/") + "/"
    for base in (root / "configs/solver_runtime", root / "events"):
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            try:
                config = read_json(path)
            except (OSError, ValueError):
                continue
            if not isinstance(config, dict) or not isinstance(config.get("source_options"), dict):
                continue
            for option, value in sorted(config["source_options"].items()):
                if not isinstance(value, str):
                    continue
                normalized = value.replace("\\", "/")
                if not (Path(normalized).is_absolute() or PureWindowsPath(normalized).drive):
                    continue
                source = Path(normalized)
                # Translate this project's registered Data address for an audit
                # on another machine, without rewriting an active input profile.
                if old_data and normalized.lower().startswith(prefix.lower()):
                    source = data_root / normalized[len(prefix):]
                references.append({"profile": path.relative_to(root).as_posix(),
                                   "option": option, "path": str(source),
                                   "exists": source.exists(),
                                   "kind": "directory" if source.is_dir() else "file" if source.is_file() else "missing",
                                   "suffix": source.suffix.lower()})
    return references


def check_preserved_assets(root: Path):
    results = []
    manifests = sorted((root / "events").rglob("asset_preservation_manifest*.json"))
    for directory in ("events", "calibration", "environments"):
        manifests += sorted((root / directory).rglob("source_preservation_manifest.json"))
    manifests += sorted((root / "scripts").glob("*.provenance.json"))
    manifests += sorted((root / "calibration").rglob("*.provenance.json"))
    for path in manifests:
        manifest = read_json(path)
        rows = manifest.get("assets", manifest.get("records", []))
        if manifest.get("schema") == "rinkspace.recovered-source-provenance.v1":
            rows = [{"preserved_path": manifest["preserved_path"],
                     "sha256": manifest["preserved_sha256"]}]
        failures = []
        for row in rows:
            relative = row.get("project_path", row.get("preserved_path"))
            target = bounded(root, relative)
            try:
                if digest(target) != row["sha256"]:
                    failures.append({"path": relative, "status": "sha256_mismatch"})
            except OSError as error:
                failures.append({"path": relative, "status": "unreadable", "error": str(error)})
        results.append({"manifest": path.relative_to(root).as_posix(), "checked": len(rows),
                        "failures": failures, "status": "identity_failures" if failures else "verified",
                        "scope": "Project copies only; does not certify nested external inputs or runtime activation."})
    return results


def audit(root: Path, data_root: Path):
    files, unreadable = inventory(root)
    references = input_references(root, data_root)
    git = subprocess.run(["git", "--no-optional-locks", "-C", str(root), "ls-files", "-z"],
                         capture_output=True, check=False)
    tracked = set(git.stdout.decode("utf-8", errors="replace").split("\0")) if git.returncode == 0 else set()
    sources = [name for name, size in files if Path(name).suffix.lower() in CODE_SUFFIXES]
    return {
        "schema": "rinkspace.repository-assets-audit.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root), "data_root": str(data_root),
        "scope": "Read-only source/config inventory; no solve, native load, environment change or full media scan.",
        "snapshot_note": "Files may change in another active chat. Re-run after that work settles; this is not a release or accuracy certificate.",
        "file_count_excluding_runtime_directories": len(files),
        "file_bytes_excluding_runtime_directories": sum(size for name, size in files),
        "by_top_level": dict(sorted(Counter(name.split('/')[0] for name, size in files).items())),
        "source_file_count": len(sources),
        "source_files_not_yet_tracked": [name for name in sources if name not in tracked],
        "git_inventory_available": git.returncode == 0,
        "owned_source_manifests": check_sources(root),
        "preserved_asset_manifests": check_preserved_assets(root),
        "declared_profile_input_references": references,
        "missing_declared_profile_inputs": [row for row in references if not row["exists"]],
        "unreadable_or_skipped_links": unreadable,
        "limits": ["Nested JSON/NPZ/pickle dependencies are not certified by this inventory.",
                   "Old profiles may deliberately point to retired inputs; assess against the selected target.",
                   "Environment/native rebuild and source-video accuracy need separate verification."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report under Data; default is stdout.")
    options = parser.parse_args()
    root = options.project.resolve()
    data = configured_data_root(root, data_root=options.data_root)
    report = audit(root, data)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if options.output:
        destination = options.output.resolve()
        if not destination.is_relative_to(data) or destination.is_relative_to(root):
            parser.error("--output must be under Data and outside the project source folder")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        print(f"Audit saved: {destination}")
        print(f"Source files: {report['source_file_count']}; missing profile references: {len(report['missing_declared_profile_inputs'])}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
