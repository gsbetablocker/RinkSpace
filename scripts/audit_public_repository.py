"""Read-only publication inventory; writes its report only beneath external Data.

This script never stages files, changes Git configuration, reads source media,
or scans other projects. Metadata-only mode needs no project runtime imports.
It is a preparation report, not a public-release security or licence certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import stat
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024
ORDINARY_WARNING_BYTES = 50 * MIB
ORDINARY_LIMIT_BYTES = 100 * MIB
SIZE_SOURCE = "https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".cache", "node_modules"}
CODE_SUFFIXES = {".py", ".ps1", ".cmd", ".bat", ".c", ".cpp", ".h", ".hpp", ".cu"}
TEXT_SUFFIXES = CODE_SUFFIXES | {".md", ".rst", ".txt", ".json", ".toml", ".yaml", ".yml", ".html", ".htm", ".css", ".js", ".ts", ".xml", ".svg", ".ini", ".cfg", ".env", ".pem", ".key", ".lock"}
TEXT_NAMES = {".gitignore", ".gitattributes", "LICENSE", "COPYING", "NOTICE", "Dockerfile", "Makefile"}
COMPILED_SUFFIXES = {".dll", ".lib", ".exe", ".obj", ".o", ".pdb", ".exp", ".ilk", ".so", ".dylib", ".a"}
MEDIA_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".mp3", ".wav", ".png", ".jpg", ".jpeg", ".webp", ".gif"}
SECRET_PATTERNS = {
    "private_key_header": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
    "github_token_format": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"),
    "aws_access_key_id_format": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "google_api_key_format": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "slack_token_format": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b"),
    "credential_in_url": re.compile(r"https?://[^\s/'\"<>:]+:[^\s/'\"<>@]+@"),
}
LITERAL_CREDENTIAL = re.compile(
    r"(?i)[\"']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|secret[_-]?key)[\"']?\s*[:=]\s*[\"']([^\"'\r\n]{12,})[\"']"
)
PATH_PATTERNS = {
    "windows_absolute_path": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    "user_home_absolute_path": re.compile(r"(?i)(?:[A-Za-z]:[\\/]+Users[\\/]|/(?:home|Users)/[A-Za-z0-9_.-]+/)"),
}


def git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "--no-optional-locks", "-c", "core.quotepath=false", "-C", str(root), *args],
        input=input_bytes, capture_output=True, check=False,
    )
    if result.returncode:
        # Do not echo arbitrary Git output into the report or tool transcript.
        raise RuntimeError(f"Read-only Git operation {args[0]} failed ({result.returncode})")
    return result.stdout


def decode(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


def names(value: bytes) -> list[str]:
    return [decode(item) for item in value.split(b"\0") if item]


def allowed(root: Path, relative: str) -> Path | None:
    path = Path(relative)
    if path.is_absolute() or any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in path.parts):
        return None
    candidate = root / path
    try:
        if not candidate.resolve().is_relative_to(root):
            return None
    except OSError:
        return None
    return candidate


def object_metadata(root: Path, object_ids: list[str]) -> dict[str, dict]:
    if not object_ids:
        return {}
    output = git(root, "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                 input_bytes=("\n".join(sorted(set(object_ids))) + "\n").encode("ascii"))
    result = {}
    for line in decode(output).splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[2].isdigit():
            result[fields[0]] = {"type": fields[1], "bytes": int(fields[2])}
    return result


def attributes(root: Path, paths: list[str], *, cached: bool) -> dict[str, dict]:
    if not paths:
        return {}
    arguments = ["check-attr"] + (["--cached"] if cached else []) + ["-z", "--stdin", "filter", "diff", "merge"]
    values = names(git(root, *arguments, input_bytes=("\0".join(paths) + "\0").encode("utf-8")))
    result: dict[str, dict] = {}
    for offset in range(0, len(values), 3):
        path, attribute, value = values[offset:offset + 3]
        result.setdefault(path, {})[attribute] = value
    return result


def lfs_pointer(value: bytes) -> dict | None:
    if len(value) > 1024 or not value.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        return None
    oid = re.search(rb"(?m)^oid sha256:([0-9a-f]{64})$", value)
    size = re.search(rb"(?m)^size ([0-9]+)$", value)
    if not oid or not size:
        return None
    return {"payload_sha256": decode(oid.group(1)), "payload_bytes": int(size.group(1))}


def text_file(relative: str) -> bool:
    path = Path(relative)
    return path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES or path.name.startswith(".env")


def scan_text(value: bytes, path: str, layer: str) -> tuple[list[dict], list[dict]]:
    if b"\0" in value[:8192]:
        return [], []
    secrets, absolute_paths = [], []
    for line_number, line in enumerate(decode(value).splitlines(), 1):
        for category, pattern in SECRET_PATTERNS.items():
            if pattern.search(line):
                secrets.append({"path": path, "line": line_number, "category": category, "layer": layer})
        for match in LITERAL_CREDENTIAL.finditer(line):
            literal = match.group(1)
            placeholder = re.search(r"(?i)(?:example|placeholder|your[_ -]|replace[_ -]|changeme|redacted|dummy|<|\{|\$)", literal)
            if not placeholder and len(set(literal)) >= 8 and not re.search(r"\s", literal):
                secrets.append({"path": path, "line": line_number, "category": "literal_credential_assignment_candidate", "layer": layer})
        for category, pattern in PATH_PATTERNS.items():
            if pattern.search(line):
                absolute_paths.append({"path": path, "line": line_number, "category": category, "layer": layer})
    return secrets, absolute_paths


def preservation_roles(root: Path) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    for path in (root / "events").glob("*/asset_preservation_manifest*.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        for asset in manifest.get("assets", []):
            relative = asset.get("project_path")
            if relative:
                relative = relative.replace("\\", "/")
                roles[relative] = sorted(set(roles.get(relative, [])) | set(asset.get("roles", [])))
    return roles


def role(path: str, preserved: dict[str, list[str]]) -> str:
    suffix = Path(path).suffix.lower()
    if path in preserved:
        return "preserved_source_packet_or_irreplaceable_fact"
    if suffix in CODE_SUFFIXES:
        return "source_code"
    if path.startswith("models/") and suffix in {".pt", ".pth"}:
        return "neural_weight_candidate_requires_provenance_and_licence"
    if suffix in COMPILED_SUFFIXES:
        return "compiled_artifact_requires_review"
    if suffix in MEDIA_SUFFIXES:
        return "media_requires_rights_and_role_review"
    if suffix in {".npz", ".npy"}:
        return "array_requires_semantic_review_not_extension_exclusion"
    if path.startswith("events/") and suffix == ".json":
        return "event_or_programme_parameter_or_evidence"
    return "configuration_documentation_or_other"


def local_review_reasons(path: str) -> list[str]:
    name = Path(path).name.lower()
    parts = {part.lower() for part in Path(path).parts}
    reasons = []
    if name.startswith(".env") or name in {"local.json", "credentials.json", "token.json"}:
        reasons.append("local_configuration_or_credential_filename")
    if parts & {".agents", ".codex", "diagnostics", "history", "review_20261005"}:
        reasons.append("agent_session_or_historical_evidence_context")
    if path.endswith(".provenance.json") or name.endswith("_status.json") or "preservation_manifest" in name:
        reasons.append("local_provenance_or_status_review")
    return reasons


def current_index_paths(root: Path) -> tuple[list[dict], list[str], list[str]]:
    """Read names and index identities only; never ask Git for blob contents."""
    raw_index = git(root, "ls-files", "--stage", "-z")
    index = []
    for item in raw_index.split(b"\0"):
        if not item:
            continue
        details, path_bytes = item.split(b"\t", 1)
        mode, oid, stage = decode(details).split()
        index.append({"path": decode(path_bytes), "mode": mode, "object_id": oid, "stage": int(stage)})
    untracked = names(git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    candidates = sorted(set(row["path"] for row in index) | set(untracked))
    return index, untracked, candidates


def audit_metadata(root: Path) -> dict:
    """Low-load snapshot: ls-files/check-attr plus filesystem metadata only.

    Do not call preservation_roles, object_metadata, scan_text, lfs_pointer,
    or any read/hash operation on candidate files from this function.
    """
    started = datetime.now(timezone.utc).isoformat()
    index, untracked, candidates = current_index_paths(root)
    worktree_attrs = attributes(root, candidates, cached=False)
    index_attrs = attributes(root, candidates, cached=True)
    for row in index:
        row["attributes_from_index"] = index_attrs.get(row["path"], {})
        row["blob_bytes_status"] = "not_checked_metadata_only"
    tracked_names = {row["path"] for row in index}
    working = []
    for relative in candidates:
        row = {"path": relative, "git_state": "tracked" if relative in tracked_names else "untracked_not_ignored",
               "role": role(relative, {}), "attributes_from_worktree": worktree_attrs.get(relative, {}),
               "local_review_reasons": local_review_reasons(relative)}
        path = allowed(root, relative)
        if path is None:
            row["status"] = "excluded_runtime_or_external_link"
        else:
            try:
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    row["status"] = "link_or_nonregular_metadata_only"
                else:
                    row.update(status="present", bytes=metadata.st_size, mtime_ns=metadata.st_mtime_ns)
            except FileNotFoundError:
                row["status"] = "missing_from_worktree"
            except OSError:
                row["status"] = "unreadable_metadata"
        working.append(row)
    local_presence = []
    for relative in ("configs/local.json", ".env", ".env.local", ".env.production"):
        path = allowed(root, relative)
        if path and path.is_file():
            local_presence.append({"path": relative, "candidate_for_add": relative in candidates,
                                   "content_scanned": False})
    finished = datetime.now(timezone.utc)
    large_working = [row for row in working if row.get("bytes", 0) > ORDINARY_WARNING_BYTES]
    compiled = [row for row in index if Path(row["path"]).suffix.lower() in COMPILED_SUFFIXES]
    return {
        "schema": "rinkspace.public-repository-metadata.v1", "mode": "metadata_only",
        "started_at_utc": started, "finished_at_utc": finished.isoformat(),
        "finished_at_shanghai": finished.astimezone(timezone(timedelta(hours=8))).isoformat(),
        "project_root": str(root),
        "scope": "Only Git ls-files/check-attr and candidate filesystem metadata; no object enumeration, blob contents, credential/path text scan, candidate hashes, or media/array/weight reads.",
        "snapshot_note": "Files in an active chat can change during or after this snapshot. A candidate is an inventory item, not a publication decision. This mode does not inherit results of older reports.",
        "publication_review_status": "not_certified",
        "checks_not_performed": {"credentials": "not_scanned", "hardcoded_path_contents": "not_scanned",
                                 "history_objects": "not_enumerated", "index_blob_sizes": "not_checked",
                                 "lfs_pointer_contents": "not_read", "candidate_hashes": "not_computed",
                                 "preservation_manifest_roles": "not_read", "licence_contents_or_third_party_rights": "not_reviewed"},
        "candidate_role_basis": "Path/extension heuristic only; manifests and array semantics were not read. NPZ/NPY candidates remain items requiring semantic review.",
        "ordinary_github_file_thresholds": {"warning_bytes": ORDINARY_WARNING_BYTES, "blocking_above_bytes": ORDINARY_LIMIT_BYTES,
                                            "source": SIZE_SOURCE, "verified_date": "2026-10-11", "rechecked_in_metadata_mode": False},
        "summary": {"index_entries": len(index), "unique_tracked_paths": len(tracked_names),
                    "untracked_not_ignored_files": len(untracked), "working_candidates": len(working),
                    "untracked_source_candidates": sum(row["git_state"] == "untracked_not_ignored" and row["role"] == "source_code" for row in working),
                    "by_candidate_role": dict(sorted(Counter(row["role"] for row in working).items())),
                    "by_worktree_status": dict(sorted(Counter(row["status"] for row in working).items())),
                    "large_working_files": len(large_working), "tracked_compiled_artifacts": len(compiled)},
        "index": index, "working_candidates": working, "working_files_above_warning": large_working,
        "tracked_compiled_artifacts": compiled,
        "lfs_attribute_differences": [{"path": name, "index": index_attrs.get(name, {}), "worktree": worktree_attrs.get(name, {})}
                                    for name in candidates if index_attrs.get(name) != worktree_attrs.get(name)],
        "root_project_licence_file_present": any((root / name).is_file() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.md")),
        "local_ignored_configuration_presence": local_presence,
        "lfs_scope": "Effective attributes only. Pointer content, payload identity, upload availability and rights were not checked.",
        "candidate_content_reads": False, "git_blob_content_reads": False, "history_objects_enumerated": False,
    }


def audit(root: Path, max_text_bytes: int) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    index, untracked, candidates = current_index_paths(root)
    index_meta = object_metadata(root, [row["object_id"] for row in index])
    worktree_attrs = attributes(root, candidates, cached=False)
    index_attrs = attributes(root, candidates, cached=True)
    preserved = preservation_roles(root)
    secrets, paths, scan_skips, scan_counts = [], [], [], Counter()
    index_cache: dict[str, bytes] = {}
    for row in index:
        relative, oid = row["path"], row["object_id"]
        meta = index_meta.get(oid, {})
        row["bytes"] = meta.get("bytes")
        row["attributes_from_index"] = index_attrs.get(relative, {})
        row["role"] = role(relative, preserved)
        if meta.get("type") != "blob":
            scan_skips.append({"path": relative, "layer": "index", "reason": "not_a_regular_blob"})
            continue
        needs_text = allowed(root, relative) is not None and text_file(relative) and meta["bytes"] <= max_text_bytes
        check_pointer = meta["bytes"] <= 1024 and (index_attrs.get(relative, {}).get("filter") == "lfs" or Path(relative).suffix.lower() in {".pt", ".pth"})
        if check_pointer or needs_text:
            if oid not in index_cache:
                index_cache[oid] = git(root, "cat-file", "blob", oid)
            value = index_cache[oid]
            row["lfs_pointer"] = lfs_pointer(value)
            if needs_text:
                found, absolute = scan_text(value, relative, "index")
                secrets.extend(found)
                paths.extend(absolute)
                scan_counts["index_text_files"] += 1
        if not needs_text:
            scan_skips.append({"path": relative, "layer": "index", "reason": "excluded_path_binary_or_text_size_limit"})
    working = []
    tracked_names = {row["path"] for row in index}
    for relative in candidates:
        row = {"path": relative, "git_state": "tracked" if relative in tracked_names else "untracked_not_ignored",
               "role": role(relative, preserved), "attributes_from_worktree": worktree_attrs.get(relative, {}),
               "preservation_roles": preserved.get(relative, []), "local_review_reasons": local_review_reasons(relative)}
        path = allowed(root, relative)
        if path is None:
            row["status"] = "excluded_runtime_or_external_link"
        else:
            try:
                before = path.stat()
                row.update(status="present", bytes=before.st_size, mtime_ns=before.st_mtime_ns)
                if path.is_symlink() or not path.is_file():
                    row["status"] = "link_or_nonregular_not_read"
                elif text_file(relative) and before.st_size <= max_text_bytes:
                    value = path.read_bytes()
                    after = path.stat()
                    row["changed_during_scan"] = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
                    found, absolute = scan_text(value, relative, "worktree")
                    secrets.extend(found)
                    paths.extend(absolute)
                    scan_counts["worktree_text_files"] += 1
                else:
                    scan_skips.append({"path": relative, "layer": "worktree", "reason": "binary_or_text_size_limit"})
            except FileNotFoundError:
                row["status"] = "missing_from_worktree"
            except OSError:
                row["status"] = "unreadable"
        working.append(row)
    # Object sizes only: no history blob content is loaded, decoded, or scanned.
    history_paths: dict[str, str] = {}
    for line in decode(git(root, "rev-list", "--objects", "--all")).splitlines():
        fields = line.split(" ", 1)
        history_paths[fields[0]] = fields[1] if len(fields) == 2 else ""
    history_meta = object_metadata(root, list(history_paths))
    large_history = [{"object_id": oid, "path_hint": history_paths[oid], "bytes": meta["bytes"]}
                     for oid, meta in history_meta.items()
                     if meta["type"] == "blob" and meta["bytes"] > ORDINARY_WARNING_BYTES]
    large_working = [row for row in working if row.get("bytes", 0) > ORDINARY_WARNING_BYTES]
    large_index = [row for row in index if (row.get("bytes") or 0) > ORDINARY_WARNING_BYTES]
    local_presence = []
    for relative in ("configs/local.json", ".env", ".env.local", ".env.production"):
        path = allowed(root, relative)
        if path and path.is_file():
            local_presence.append({"path": relative, "candidate_for_add": relative in candidates,
                                   "content_scanned": relative in candidates})
    licence = {"root_project_licence_present": any((root / name).is_file() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.md")),
               "decision": "Project owner must choose the project licence and separately establish third-party rights; this audit chooses none."}
    return {
        "schema": "rinkspace.public-repository-audit.v1", "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "project_root": str(root),
        "scope": "Current index, nonignored untracked candidates, working metadata, and sizes of objects reachable from --all refs; no staging, checkout, config changes, solve, media decoding, or history rewriting.",
        "snapshot_note": "An active chat can change files during or after this read. Re-run before publication; this is not a public-release security, accuracy, licence, or reproducibility certificate.",
        "ordinary_github_file_thresholds": {"warning_bytes": ORDINARY_WARNING_BYTES, "blocking_above_bytes": ORDINARY_LIMIT_BYTES, "source": SIZE_SOURCE, "verified_date": "2026-10-11"},
        "summary": {"index_entries": len(index), "unique_tracked_paths": len(tracked_names), "untracked_not_ignored_files": len(untracked),
                    "working_candidates": len(working), "by_candidate_role": dict(sorted(Counter(row["role"] for row in working).items())),
                    "large_working_files": len(large_working), "large_index_blobs": len(large_index), "large_reachable_history_blobs": len(large_history),
                    "suspected_credential_locations": len(secrets), "absolute_path_locations": len(paths), "text_scan_counts": dict(scan_counts)},
        "licence": licence,
        "index": index, "working_candidates": working,
        "working_files_above_warning": large_working, "index_blobs_above_warning": large_index,
        "reachable_history_blobs_above_warning": sorted(large_history, key=lambda row: row["bytes"], reverse=True),
        "tracked_compiled_artifacts": [row for row in index if Path(row["path"]).suffix.lower() in COMPILED_SUFFIXES],
        "lfs_attribute_differences": [{"path": name, "index": index_attrs.get(name, {}), "worktree": worktree_attrs.get(name, {})}
                                    for name in candidates if index_attrs.get(name) != worktree_attrs.get(name)],
        "suspected_credentials": secrets, "absolute_paths_requiring_portability_or_privacy_review": paths,
        "local_ignored_configuration_presence": local_presence, "text_scan_skips": scan_skips,
        "credential_scan_limits": {"max_text_bytes_per_file": max_text_bytes, "output": "File path, line number and category only; no matched text or values.",
                                   "excluded": "Runtime directories, external Data, other projects, binary/media content, oversized text, and history credential contents.",
                                   "interpretation": "Pattern findings may be examples or test fixtures; absence of a finding is not evidence that all secrets are absent. Absolute paths and session files are review items, not automatically credentials."},
        "history_size_scope": "Objects reachable from all local refs, with one path hint per object; unreachable/reflog-only objects, LFS remote availability and payload rights are not certified.",
        "lfs_scope": "Small index blobs with LFS attributes or weight suffixes are inspected for pointers. Attributes alone do not prove a payload was added, uploaded or licensed. Large payload bytes are not hashed or read.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="External Data root; required for --metadata-only. Full mode uses the configured local Data root.")
    parser.add_argument("--output", type=Path, help="JSON report destination beneath Data only.")
    parser.add_argument("--metadata-only", action="store_true", help="Low-load names/attributes/stat only; skip blob, history, credential/content scans and hashes.")
    parser.add_argument("--max-text-bytes", type=int, default=4 * MIB, help="Per-file bounded text scan; skipped files are listed.")
    options = parser.parse_args()
    if options.max_text_bytes < 1:
        parser.error("--max-text-bytes must be positive")
    if options.metadata_only and not options.data_root:
        parser.error("--metadata-only requires --data-root so no storage configuration content is opened")
    root = PROJECT.resolve()
    if options.data_root:
        data = options.data_root.resolve()
    else:
        sys.path.insert(0, str(PROJECT / "src"))
        from rinkspace.storage import configured_data_root
        data = configured_data_root(root)
    if data.is_relative_to(root) or root.is_relative_to(data):
        parser.error("Data and the source worktree must be separate directories")
    local_date = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    report_name = f"public_repository_metadata_{local_date}.json" if options.metadata_only else "public_repository_audit.json"
    output = (options.output or data / "maintenance/repository_preparation_20261011" / report_name).resolve()
    if not output.is_relative_to(data) or output.is_relative_to(root):
        parser.error("--output must be beneath external Data, outside the worktree")
    report = audit_metadata(root) if options.metadata_only else audit(root, options.max_text_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
