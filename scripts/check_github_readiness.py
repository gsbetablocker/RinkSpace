"""Read-only GitHub preflight; reuse the publication inventory and scanner.

Run with an explicit external --data-root and --output. --review-only skips
reachable-history enumeration for a quick current-tree review. No command
stages, commits, changes Git configuration, contacts a remote, or hashes large
payloads. This is a snapshot, not a security or reproducibility certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from urllib.parse import urlsplit

# Also works when loaded directly by importlib for focused tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_public_repository as inventory


PROJECT = Path(__file__).resolve().parents[1]
CACHE_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", ".venv", "node_modules"}
LOCAL_NAMES = {"local.json", "credentials.json", "token.json", ".npmrc", ".pypirc", ".netrc", "id_rsa", "id_ed25519"}


LOCAL_ONLY = {
    "START_HERE.html", "calibration/rinkfit/START_HERE.md", "calibration/rinkfit/CODEX_START.md",
    "docs/REPOSITORY_GUIDE.md", "docs/PUBLIC_REPOSITORY_CONTENTS.md", "docs/PRIVACY_REVIEW.md",
    "docs/RUNTIME_DEPENDENCIES.md", "docs/rink_model/VALIDATION.md",
    "events/ankara_2026/diagnostics/external_review_20261007.html",
    "events/ankara_2026/feedback/index.html",
    "events/ankara_2026/programmes/okada_mei_sp/audit_1017.html",
    "events/ankara_2026/programmes/okada_mei_sp/fit_joint9/recovery_1019.html",
    "events/ankara_2026/programmes/okada_mei_sp/fit_joint9/review.html",
    "events/ankara_2026/programmes/okada_mei_sp/fit_v11/audit_1018.html",
    "events/ankara_2026/programmes/okada_mei_sp/fit_v11/status.html",
}
PUBLIC_GUIDES = ("README.md", "MAINTENANCE.md", "environments/README.md",
                 "docs/ARCHITECTURE.md", "docs/GITHUB_REVIEW.md", "docs/VALIDATION.md",
                 "calibration/rinkfit/README.md", "calibration/solver_runtime/README.md")
QUICKSTART_FILES = ("src/rinkspace/cli.py", "src/rinkspace/demo.py", "tests/test_geometry.py",
                    "tests/test_motion.py", "tests/test_mosaic.py", "tests/test_demo.py")


def local_publication_path(path: str) -> bool:
    path = path.replace("\\", "/")
    value = PurePosixPath(path)
    return (path in LOCAL_ONLY or bool(re.fullmatch(r"START_SHOT[2-9]_REVIEW\.cmd", path))
            or ("source_datasets" in value.parts and value.name in {"review.html", "whole_first_shot_review.html"}))


def public_entry_findings(root: Path, candidates: set[str]) -> list[dict]:
    findings = []
    for name in PUBLIC_GUIDES + QUICKSTART_FILES:
        if name not in candidates or not (root / name).is_file():
            findings.append({"path": name, "reason": "missing_public_entry_or_quickstart_file"})
    for name in PUBLIC_GUIDES:
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig")
        if re.search(r"file:/+|\]\([A-Za-z]:[/\\]", text, re.I):
            findings.append({"path": name, "reason": "public_guide_depends_on_machine_file_link"})
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
            if "://" in target:
                continue
            linked = (path.parent / target).resolve()
            if linked.is_relative_to(root) and local_publication_path(linked.relative_to(root).as_posix()):
                findings.append({"path": name, "reason": "public_guide_links_to_local_only_entry"})
    return findings


def size_class(size: int) -> str:
    if size > inventory.ORDINARY_LIMIT_BYTES:
        return "above_github_ordinary_file_limit"
    if size > inventory.ORDINARY_WARNING_BYTES:
        return "above_github_ordinary_file_warning"
    return "within_github_ordinary_file_warning"


def tracked_review_reasons(path: str) -> list[str]:
    """Review path classes; arrays and images are deliberately not excluded."""
    value = PurePosixPath(path.replace("\\", "/"))
    parts = {part.lower() for part in value.parts}
    suffix, name = value.suffix.lower(), value.name.lower()
    reasons = []
    if local_publication_path(path):
        reasons.append("tracked_local_panel_handoff_or_generated_review")
    if parts & CACHE_PARTS or any(part.endswith(".egg-info") for part in parts):
        reasons.append("tracked_runtime_or_cache")
    if suffix in inventory.COMPILED_SUFFIXES | {".pyc", ".pyo"}:
        reasons.append("tracked_compiled_artifact")
    if (name.startswith(".env") and name not in {".env.example", ".env.sample", ".env.template"}) or name in LOCAL_NAMES:
        reasons.append("tracked_local_configuration_or_credential_filename")
    if parts & {".codex", ".agents", ".idea", ".vscode"}:
        reasons.append("tracked_machine_or_agent_configuration_requires_review")
    return reasons


def safe_remote(url: str) -> dict:
    """Never return URL userinfo, query strings, or fragments."""
    try:
        if "://" in url:
            parsed = urlsplit(url)
            host = parsed.hostname
            if not host:
                return {"kind": "local_or_unrecognized", "endpoint": "redacted"}
            return {"kind": parsed.scheme, "host": host, "repository_path": parsed.path}
        match = re.fullmatch(r"(?:[^/@\s]+@)?([^/:\s]+):([^?#\s]+)(?:[?#].*)?", url)
        if match and not re.match(r"^[A-Za-z]:[\\/]", url):
            return {"kind": "ssh_scp", "host": match.group(1), "repository_path": match.group(2)}
    except ValueError:
        pass
    return {"kind": "local_or_unrecognized", "endpoint": "redacted"}


def optional_git(root: Path, *args: str) -> bytes | None:
    try:
        return inventory.git(root, *args)
    except RuntimeError:
        return None


def tree_entries(root: Path) -> list[dict]:
    raw = optional_git(root, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    entries = []
    for item in (raw or b"").split(b"\0"):
        if item:
            fields, path = item.split(b"\t", 1)
            mode, kind, oid = inventory.decode(fields).split()
            entries.append({"path": inventory.decode(path), "mode": mode, "type": kind, "object_id": oid})
    return entries


@contextmanager
def blob_reader(root: Path):
    """One bounded, on-demand cat-file process; no large payload reads."""
    process = subprocess.Popen(
        ["git", "--no-optional-locks", "-C", str(root), "cat-file", "--batch"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    def read(oid: str) -> bytes:
        process.stdin.write((oid + "\n").encode("ascii"))
        process.stdin.flush()
        header = process.stdout.readline().split()
        if len(header) != 3 or header[1] != b"blob" or not header[2].isdigit():
            raise RuntimeError("Git blob reader returned an unexpected header")
        size = int(header[2])
        value = process.stdout.read(size)
        if len(value) != size or process.stdout.read(1) != b"\n":
            raise RuntimeError("Git blob reader returned incomplete content")
        return value
    try:
        yield read
    finally:
        process.stdin.close()
        process.stdout.close()
        if process.wait() != 0:
            raise RuntimeError("Read-only Git blob reader failed")


def inspect_lfs_blob(size: int, pointer: dict | None, attribute: dict) -> str:
    if pointer:
        return "lfs_pointer" if attribute.get("filter") == "lfs" else "lfs_pointer_without_effective_lfs_filter"
    if attribute.get("filter") == "lfs":
        return "raw_blob_despite_lfs_attribute"
    return size_class(size)


def local_git_state(root: Path) -> dict:
    remotes = []
    for name in inventory.decode(inventory.git(root, "remote")).splitlines():
        urls = optional_git(root, "remote", "get-url", "--all", name)
        push_urls = optional_git(root, "remote", "get-url", "--push", "--all", name)
        remotes.append({"name": name,
                        "fetch": [safe_remote(url) for url in inventory.decode(urls or b"").splitlines()],
                        "push": [safe_remote(url) for url in inventory.decode(push_urls or b"").splitlines()]})
    lfs = optional_git(root, "lfs", "version")
    filters = {}
    for name in ("clean", "process", "smudge", "required"):
        value = optional_git(root, "config", "--get", "filter.lfs." + name)
        text = inventory.decode(value or b"").strip()
        filters[name] = {"configured": value is not None,
                         "git_lfs_command" if name != "required" else "required_true":
                             bool(re.search(r"(?:git-lfs|git lfs)(?:\.exe)?(?:\s|$)", text)) if name != "required" else text.lower() == "true"}
    commit = optional_git(root, "rev-parse", "--verify", "HEAD")
    branch = optional_git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    upstream = optional_git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    counts = optional_git(root, "rev-list", "--left-right", "--count", "HEAD...@{upstream}") if upstream else None
    return {"head_commit": inventory.decode(commit).strip() if commit else None,
            "branch": inventory.decode(branch).strip() if branch else None,
            "upstream": inventory.decode(upstream).strip() if upstream else None,
            "ahead_behind_local_refs": [int(n) for n in counts.split()] if counts else None,
            "remotes": remotes, "remote_contacted": False,
            "lfs_available": lfs is not None,
            "lfs_version": inventory.decode(lfs).strip() if lfs else None,
            "lfs_filter_configuration": filters,
            "lfs_remote_payloads": "not_checked_no_network"}


def reachable_large_blobs(root: Path) -> list[dict]:
    paths = {}
    for line in inventory.decode(inventory.git(root, "rev-list", "--objects", "--all")).splitlines():
        oid, _, path = line.partition(" ")
        paths[oid] = path
    metadata = inventory.object_metadata(root, list(paths))
    return sorted(({"object_id": oid, "path_hint": paths[oid], "bytes": info["bytes"]}
                   for oid, info in metadata.items()
                   if info["type"] == "blob" and info["bytes"] > inventory.ORDINARY_LIMIT_BYTES),
                  key=lambda row: row["bytes"], reverse=True)


def audit(root: Path, *, review_only: bool, max_text_bytes: int) -> dict:
    snapshot = inventory.audit_metadata(root)
    index = snapshot["index"]
    head = tree_entries(root)
    metadata = inventory.object_metadata(root, [row["object_id"] for row in index + head])
    candidates = snapshot["working_candidates"]
    attrs = {row["path"]: row["attributes_from_worktree"] for row in candidates}
    findings, skips = [], []
    scan_counts = Counter()
    analysis_cache = {}
    lfs_rows = []
    with blob_reader(root) as read:
        for layer, rows in (("index", index), ("HEAD", head)):
            for row in rows:
                oid, path = row["object_id"], row["path"]
                info = metadata.get(oid, {})
                row["bytes"] = info.get("bytes")
                if info.get("type") != "blob":
                    continue
                size = info["bytes"]
                attribute = row.get("attributes_from_index", {}) if layer == "index" else attrs.get(path, {})
                # HEAD attributes are not guessed: this comparison explicitly uses current attributes.
                text = inventory.text_file(path) and inventory.allowed(root, path) is not None and size <= max_text_bytes
                small = size <= 1024
                if text or small:
                    key = (oid, text)
                    if key not in analysis_cache:
                        value = read(oid)
                        found, _ = inventory.scan_text(value, path, layer) if text else ([], [])
                        analysis_cache[key] = (inventory.lfs_pointer(value), found)
                    pointer, found = analysis_cache[key]
                    if text:
                        findings.extend({**finding, "path": path, "layer": layer} for finding in found)
                        scan_counts[layer] += 1
                else:
                    pointer = None
                if not text:
                    skips.append({"path": path, "layer": layer, "reason": "binary_excluded_path_or_text_size_limit"})
                state = inspect_lfs_blob(size, pointer, attribute)
                if pointer or attribute.get("filter") == "lfs" or size > inventory.ORDINARY_WARNING_BYTES:
                    lfs_rows.append({"path": path, "layer": layer, "object_id": oid, "bytes": size,
                                     "state": state, "pointer": pointer,
                                     "attribute_basis": "index" if layer == "index" else "current_worktree_not_historical_HEAD",
                                     "attributes": attribute})
    for row in candidates:
        path = inventory.allowed(root, row["path"])
        size = row.get("bytes", 0)
        row["size_class"] = size_class(size) if "bytes" in row else "not_measured"
        if row.get("status") != "present" or path is None or path.is_symlink():
            skips.append({"path": row["path"], "layer": "worktree", "reason": row["status"]})
        elif inventory.text_file(row["path"]) and size <= max_text_bytes:
            try:
                before = path.stat()
                value = path.read_bytes()
                after = path.stat()
                row["changed_during_scan"] = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                found, _ = inventory.scan_text(value, row["path"], "worktree")
                findings.extend(found)
                scan_counts["worktree"] += 1
            except OSError:
                skips.append({"path": row["path"], "layer": "worktree", "reason": "unreadable_during_text_scan"})
        else:
            skips.append({"path": row["path"], "layer": "worktree", "reason": "binary_or_text_size_limit"})
    git_state = local_git_state(root)
    history = [] if review_only else reachable_large_blobs(root)
    index_large = [row for row in index if (row.get("bytes") or 0) > inventory.ORDINARY_LIMIT_BYTES]
    head_large = [row for row in head if (row.get("bytes") or 0) > inventory.ORDINARY_LIMIT_BYTES]
    work_large = [row for row in candidates if row.get("bytes", 0) > inventory.ORDINARY_LIMIT_BYTES]
    tracked_review = [{"path": row["path"], "reasons": reasons}
                      for row in index if (reasons := tracked_review_reasons(row["path"]))]
    lfs_raw = [row for row in lfs_rows if row["state"] == "raw_blob_despite_lfs_attribute"]
    public_findings = public_entry_findings(root, {row["path"] for row in candidates})
    local_tracked = [row for row in index if local_publication_path(row["path"])]
    blockers = []
    if public_findings:
        blockers.append("public_entry_or_quickstart_incomplete")
    if local_tracked:
        blockers.append("local_work_material_tracked_for_publication")
    if index_large:
        blockers.append("index_contains_blobs_above_github_ordinary_file_limit")
    if head_large or history:
        blockers.append("reachable_committed_blobs_above_github_ordinary_file_limit")
    if any(row["stage"] for row in index):
        blockers.append("unmerged_index_entries")
    lfs_candidates = [row for row in candidates if row["attributes_from_worktree"].get("filter") == "lfs"]
    if lfs_candidates and (not git_state["lfs_available"] or not any(git_state["lfs_filter_configuration"][name].get("git_lfs_command") for name in ("clean", "process"))):
        blockers.append("lfs_candidates_without_available_git_lfs_and_clean_filter")
    if any(row["attributes_from_worktree"].get("filter") != "lfs" for row in work_large):
        blockers.append("working_candidates_above_limit_without_lfs_attribute")
    return {
        "schema": "rinkspace.github-readiness.v1", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root), "mode": "current_tree_review" if review_only else "current_tree_and_reachable_history",
        "scope": "Current candidate metadata, index and HEAD bounded text/pointer reads, local Git/LFS configuration; history sizes only. No network or Git mutations.",
        "snapshot_note": "Active work can change this snapshot. No findings is not a publication, security, licence, model-accuracy or offline-reproducibility certificate.",
        "github_thresholds": {"warning_bytes": inventory.ORDINARY_WARNING_BYTES, "blocking_above_bytes": inventory.ORDINARY_LIMIT_BYTES, "source": inventory.SIZE_SOURCE},
        "summary": {"candidate_files": len(candidates), "candidate_bytes": sum(row.get("bytes", 0) for row in candidates),
                    "untracked_not_ignored": snapshot["summary"]["untracked_not_ignored_files"],
                    "working_above_ordinary_limit": len(work_large), "index_above_ordinary_limit": len(index_large),
                    "HEAD_above_ordinary_limit": len(head_large),
                    "reachable_history_above_ordinary_limit": None if review_only else len(history),
                    "suspected_credential_locations": len(findings), "tracked_artifact_or_local_config_paths": len(tracked_review),
                    "raw_lfs_attribute_blobs_in_index_or_HEAD": len(lfs_raw), "blocking_checks": blockers,
                    "remote_count": len(git_state["remotes"]), "lfs_available": git_state["lfs_available"]},
        "public_entry_findings": public_findings,
        "local_work_material_in_index": local_tracked,
        "git": git_state, "candidates": candidates,
        "index_above_ordinary_limit": index_large, "HEAD_above_ordinary_limit": head_large,
        "lfs_index_and_HEAD": lfs_rows, "lfs_attribute_differences": snapshot["lfs_attribute_differences"],
        "tracked_artifact_or_local_config_paths": tracked_review,
        "suspected_credentials": findings,
        "credential_scan": {"counts": dict(scan_counts), "max_text_bytes_per_file": max_text_bytes,
                            "output": "Locations and categories only, never matched text or values.",
                            "limits": "Pattern candidates can be examples. Binary, excluded runtime paths, oversized text and history credential contents are not scanned.", "skipped": skips},
        "reachable_history_above_ordinary_limit": history,
        "history_scope": "skipped_by_review_only" if review_only else "All local refs; metadata only, one path hint per object. Unreachable/reflog-only objects are outside this scope.",
        "lfs_limits": "Attributes do not convert existing commits. Pointer presence does not prove local/remote payload availability or rights. Large worktree payloads are never read or hashed.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path, help="Explicit external Data directory")
    parser.add_argument("--output", required=True, type=Path, help="JSON destination beneath Data, outside source worktree")
    parser.add_argument("--review-only", action="store_true", help="Skip reachable-history object enumeration")
    parser.add_argument("--max-text-bytes", type=int, default=4 * inventory.MIB, help="Bounded text scan per file; skipped locations are listed")
    args = parser.parse_args(argv)
    root, data, output = PROJECT.resolve(), args.data_root.resolve(), args.output.resolve()
    if data.is_relative_to(root) or root.is_relative_to(data):
        parser.error("Data must be separate from the source worktree")
    if not output.is_relative_to(data) or output.is_relative_to(root):
        parser.error("--output must be beneath external Data")
    if args.max_text_bytes < 1:
        parser.error("--max-text-bytes must be positive")
    report = audit(root, review_only=args.review_only, max_text_bytes=args.max_text_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "summary": report["summary"]}, ensure_ascii=False))
    return 1 if report["summary"]["blocking_checks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
