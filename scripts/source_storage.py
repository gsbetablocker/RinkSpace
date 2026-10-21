"""Commit reviewed source qualification changes without full-table staging files.

Request schema rinkspace.source_storage_commit_request.v1 requires:
  db, base_rows/base_frames={stream, stream_semantic_sha256},
  new_rows_stream, new_frames_stream,
  changes=[{shot_id,native_pts,remove/update/add/order}],
  source_evidence=[{file,sha256,role(optional)}].

This entry checks evidence bytes and complete stored record content. It does not
read video pixels, qualify material, fit a physical model, or certify point accuracy.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, closing
from datetime import datetime, timezone
import hashlib
import json
import os
import sqlite3
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "calibration/rinkfit/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from rinkfit_joint import source_record_store
from rinkfit_joint.source_record_store import SourceRecordStore, SourceStoreError
from rinkfit_joint.source_record_io import canonical_record, checked_records

REQUEST_SCHEMA = "rinkspace.source_storage_commit_request.v1"
MANIFEST_SCHEMA = "rinkspace.source_storage_commit.v1"


class CommitFailure(ValueError):
    def __init__(self, message, sealed_streams=()):
        super().__init__(message)
        self.sealed_streams = list(sealed_streams)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf8")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a lowercase SHA256 digest")
    return value


def _no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("Duplicate JSON object key: " + key)
        out[key] = value
    return out


def _load(path):
    raw = Path(path).read_bytes()
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_no_duplicate_keys), hashlib.sha256(raw).hexdigest()


def _same_path(a, b):
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def _new_path(path):
    path = Path(path).resolve()
    # Check the exact final Windows UTF-16 path before any file/directory creation.
    if len(str(path).encode("utf-16-le")) // 2 >= 260 or any(len(p.encode("utf-16-le")) // 2 > 255 for p in path.parts):
        raise ValueError("Output path exceeds the project's Windows/OneDrive path limits")
    return path


def _reference(db, ref, kind):
    if not isinstance(ref, dict) or not isinstance(ref.get("stream"), str) or not ref["stream"]:
        raise ValueError("Base reference requires an explicit logical stream")
    semantic = _digest(ref.get("stream_semantic_sha256"))
    if "store" in ref and not _same_path(db, ref["store"]):
        raise ValueError("Base reference points to a different database")
    if "kind" in ref and ref["kind"] != kind:
        raise ValueError("Base reference has a wrong stream kind")
    if "store_sha256" in ref and _sha(db) != _digest(ref["store_sha256"]):
        raise ValueError("Requested database byte SHA differs")
    return {"store": str(db), "store_content_addressed": True, "stream": ref["stream"],
            "kind": kind, "stream_semantic_sha256": semantic}


def _evidence(refs):
    if not isinstance(refs, list) or not refs:
        raise ValueError("At least one actual source-evidence file with SHA is required")
    result = []
    seen = set()
    for ref in refs:
        if not isinstance(ref, dict) or not isinstance(ref.get("file"), str) or set(ref) - {"file", "sha256", "role"}:
            raise ValueError("Evidence accepts only file, sha256 and optional role")
        path = Path(ref["file"]).resolve()
        expected = _digest(ref.get("sha256"))
        if not path.is_file() or _sha(path) != expected:
            raise ValueError("Source evidence bytes differ: " + str(path))
        key = os.path.normcase(str(path))
        if key in seen:
            raise ValueError("Duplicate source evidence file: " + str(path))
        seen.add(key)
        item = {"file": str(path), "sha256": expected, "bytes": path.stat().st_size}
        if "role" in ref:
            if not isinstance(ref["role"], str):
                raise ValueError("Evidence role must be text")
            item["role"] = ref["role"]
        result.append(item)
    return result


def _check_evidence_unchanged(refs):
    for ref in refs:
        p = Path(ref["file"])
        if not p.is_file() or p.stat().st_size != ref["bytes"] or _sha(p) != ref["sha256"]:
            raise ValueError("Source evidence changed during commit: " + str(p))


def _changes(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("A nonempty explicit frame-change list is required")
    result = {}
    geometry = source_record_store.GEOMETRY_KEYS | {"selected_row_id"}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"shot_id", "native_pts", "remove", "update", "add", "order"}:
            raise ValueError("Unknown source qualification change operation")
        shot, pts = entry.get("shot_id"), entry.get("native_pts")
        if not isinstance(shot, str) or not shot or not isinstance(pts, int) or isinstance(pts, bool):
            raise ValueError("Changes require a shot and integer native PTS")
        key = shot, pts
        if key in result:
            raise ValueError("Duplicate frame-change entry")
        change = {k: v for k, v in entry.items() if k not in {"shot_id", "native_pts"}}
        if not change:
            raise ValueError("Empty qualification change")
        for field in ("remove", "order"):
            if field in change and (not isinstance(change[field], list) or any(not isinstance(i, str) or not i for i in change[field]) or len(set(change[field])) != len(change[field])):
                raise ValueError("Change identities must be an ordered, unique string list")
        updates = change.get("update", {})
        if not isinstance(updates, dict):
            raise ValueError("Qualification updates must map selected-row IDs to partial objects")
        for ident, patch in updates.items():
            if not isinstance(ident, str) or not isinstance(patch, dict) or set(patch) & geometry:
                raise ValueError("Qualification update cannot change source geometry/clock/identity")
            if patch.get("independent_fit_weight_issued") is True or patch.get("physical_world_correspondence_issued") is True:
                raise ValueError("Source-only commit cannot issue physical correspondences or weights")
        additions = change.get("add", [])
        if not isinstance(additions, list) or any(not isinstance(r, dict) for r in additions):
            raise ValueError("Added rows must be explicit complete source-row objects")
        for row in additions:
            if row.get("independent_fit_weight_issued") is not False or row.get("physical_world_correspondence_issued") is not False:
                raise ValueError("Added source candidates must explicitly withhold physical correspondences and weights")
        _json(change)
        result[key] = change
    return result


def _frame_groups(records):
    current, block = None, []
    for row in records:
        key = row["shot_id"], row["native_pts"]
        if current is not None and key != current:
            yield current, block
            block = []
        current = key
        block.append(row)
    if current is not None:
        yield current, block


def _verify_pair(rows_ref, frames_ref, transition_parent=None):
    """Read every actual record through the production checked reader, at most one frame in memory."""
    counts = {"frames": 0, "point_rows": 0}
    transition = {"removed_selected_rows": 0, "added_selected_rows": 0, "changed_retained_qualification_rows": 0,
                  "retained_source_UV_unchanged": True}
    with ExitStack() as stack:
        rows = stack.enter_context(closing(checked_records(rows_ref)))
        groups = _frame_groups(rows)
        frame_records = stack.enter_context(closing(checked_records(frames_ref)))
        old_groups = None
        if transition_parent is not None:
            old = stack.enter_context(closing(checked_records(transition_parent)))
            old_groups = _frame_groups(old)
        sentinel = object()
        for frame in frame_records:
            item = next(groups, sentinel)
            if item is sentinel or item[0] != (frame["shot_id"], frame["native_pts"]):
                raise ValueError("Rows and preview disagree on native frame coverage/order")
            key, block = item
            ids = [r["selected_row_id"] for r in block]
            if len(ids) != len(set(ids)):
                raise ValueError("Repeated selected-row identity in actual source data")
            expected = [{k: r[k] for k in ("selected_row_id", "uv", "display_class")} for r in block]
            if _json(frame["points"]) != _json(expected):
                raise ValueError("Actual preview disagrees with source IDs/UV/classes/order")
            if any(r.get("independent_fit_weight_issued") is True or r.get("physical_world_correspondence_issued") is True for r in block):
                raise ValueError("Source-only commit requires rows without physical correspondences or weights")
            counts["frames"] += 1
            counts["point_rows"] += len(block)
            if old_groups is not None:
                prior = next(old_groups, sentinel)
                if prior is sentinel or prior[0] != key:
                    raise ValueError("Commit changed native exposure coverage")
                before = {r["selected_row_id"]: r for r in prior[1]}
                after = {r["selected_row_id"]: r for r in block}
                transition["removed_selected_rows"] += len(set(before) - set(after))
                transition["added_selected_rows"] += len(set(after) - set(before))
                transition["changed_retained_qualification_rows"] += sum(_json(before[i]) != _json(after[i]) for i in set(before) & set(after))
                old_source = {r["source_point_id"]: r for r in prior[1]}
                for r in block:
                    p = old_source.get(r["source_point_id"])
                    if p is not None and _json([p["uv"], p.get("time_base")]) != _json([r["uv"], r.get("time_base")]):
                        raise ValueError("Existing source ID changed original UV or clock")
        if next(groups, sentinel) is not sentinel:
            raise ValueError("Extra source-row native frame absent from preview")
        if old_groups is not None and next(old_groups, sentinel) is not sentinel:
            raise ValueError("Commit removed native source exposures")
    if transition_parent is not None:
        counts["actual_transition"] = transition
    counts["actual_record_semantics_verified"] = True
    counts["actual_preview_projection_verified"] = True
    return counts


def _publish(path, manifest):
    path = _new_path(path)
    if path.exists():
        raise FileExistsError("Immutable manifest already exists: " + str(path))
    temporary = _new_path(path.parent / (".source_storage-" + uuid.uuid4().hex + ".tmp"))
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf8")
    try:
        with temporary.open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        if os.name == "nt":
            # Windows rename is atomic and refuses an existing destination.
            os.rename(temporary, path)
        else:
            # POSIX rename would overwrite; link atomically refuses any collision.
            os.link(temporary, path)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(raw).hexdigest()


def commit(request_path, output_path):
    request_path = Path(request_path).resolve()
    out = _new_path(output_path)
    if out.exists():
        raise FileExistsError("Immutable manifest already exists: " + str(out))
    request, request_sha = _load(request_path)
    required = {"schema", "db", "base_rows", "base_frames", "new_rows_stream", "new_frames_stream", "changes", "source_evidence"}
    if not isinstance(request, dict) or set(request) != required or request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("Unknown/incomplete source-storage request schema; see source_storage.py --help")
    if not isinstance(request["db"], str):
        raise ValueError("Request database must be a file path")
    db = Path(request["db"]).resolve()
    if not db.is_file():
        raise FileNotFoundError("Existing source database required")
    for field in ("new_rows_stream", "new_frames_stream"):
        if not isinstance(request[field], str) or not request[field]:
            raise ValueError("New logical stream names must be explicit")
    if os.path.normcase(os.path.normpath(request["new_rows_stream"])) == os.path.normcase(os.path.normpath(request["new_frames_stream"])):
        raise ValueError("Rows and frames require distinct logical names")
    rows_ref = _reference(db, request["base_rows"], "selected_rows")
    frames_ref = _reference(db, request["base_frames"], "selected_frames")
    evidence = _evidence(request["source_evidence"])
    changes = _changes(request["changes"])
    with SourceRecordStore(db, readonly=True) as store:
        for ref, kind in ((rows_ref, "selected_rows"), (frames_ref, "selected_frames")):
            info = store.stream_info(ref["stream"])
            if info["kind"] != kind or info["semantic_sha256"] != ref["stream_semantic_sha256"]:
                raise ValueError("Base stored stream descriptor differs from request")
        for field in ("new_rows_stream", "new_frames_stream"):
            try:
                store.stream_info(request[field])
            except KeyError:
                pass
            else:
                raise FileExistsError("Immutable logical stream already exists: " + request[field])
    base_validation = _verify_pair(rows_ref, frames_ref)
    sealed = []
    try:
        with SourceRecordStore(db) as store:
            row_info = store.fork_rows_stream(request["new_rows_stream"], rows_ref["stream"], changes)
            sealed.append(request["new_rows_stream"])
            frame_info = store.project_selected_frames(request["new_frames_stream"], request["new_rows_stream"], frames_ref["stream"])
            sealed.append(request["new_frames_stream"])
        new_rows_ref = {"store": str(db), "store_content_addressed": True, "stream": request["new_rows_stream"],
                        "kind": "selected_rows", "stream_semantic_sha256": row_info["semantic_sha256"]}
        new_frames_ref = {"store": str(db), "store_content_addressed": True, "stream": request["new_frames_stream"],
                          "kind": "selected_frames", "stream_semantic_sha256": frame_info["semantic_sha256"]}
        actual = _verify_pair(new_rows_ref, new_frames_ref, transition_parent=rows_ref)
        _check_evidence_unchanged(evidence)
        if _sha(request_path) != request_sha:
            raise ValueError("Source-storage request changed during commit")
        module = Path(source_record_store.__file__).resolve()
        manifest = {"schema": MANIFEST_SCHEMA, "complete": True,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(), "quality_status": "candidate_source_only",
                    "store_content_addressed": True,
                    "content_address_scope": "compressed blobs and complete recovered stream semantics; SQLite may append later sealed streams",
                    "database": {"file": str(db), "sha256_at_commit": _sha(db)},
                    "selected_rows": new_rows_ref, "selected_frames": new_frames_ref,
                    "parent": {"selected_rows": rows_ref, "selected_frames": frames_ref},
                    "request": {"file": str(request_path), "sha256": request_sha},
                    "source_evidence": evidence,
                    "source_record_store_module": {"file": str(module), "sha256": _sha(module)},
                    "source_storage_entry": {"file": str(Path(__file__).resolve()), "sha256": _sha(__file__)},
                    "base_validation": base_validation, "actual_validation": actual,
                    "qualification_delta": {"requested_changed_frames": len(changes),
                        "stored_row_frame_overrides": row_info["stored_frame_overrides"],
                        "stored_preview_header_overrides": frame_info["stored_frame_overrides"],
                        "actual_transition": actual["actual_transition"]},
                    "source_UV_unchanged": True,
                    "source_UV_unchanged_scope": "all retained source IDs; additions copy explicit request UV without certifying pixel correctness",
                    "full_selected_rows_file_written": False, "full_selected_frames_file_written": False,
                    "original_source_readings_regenerated": False, "source_video_pixels_decoded": False,
                    "source_image_accuracy_or_material_qualification_validated": False,
                    "physical_fit_started": False, "physical_world_correspondence_issued": False,
                    "independent_fit_weights_issued": False}
        digest = _publish(out, manifest)
    except BaseException as error:
        raise CommitFailure(str(error), sealed) from error
    return {"ok": True, "manifest": str(out), "sha256": digest, "quality_status": manifest["quality_status"],
            "frames": actual["frames"], "point_rows": actual["point_rows"],
            "actual_transition": actual["actual_transition"], "full_point_tables_written": False}


def inspect_manifest(path, verify=False):
    path = Path(path).resolve()
    manifest, digest = _load(path)
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("complete") is not True:
        raise ValueError("Unknown/incomplete source-storage manifest")
    result = {"ok": True, "manifest": str(path), "sha256": digest, "schema": manifest["schema"],
              "quality_status": manifest["quality_status"], "database": manifest["database"]["file"],
              "selected_rows_stream": manifest["selected_rows"]["stream"],
              "selected_frames_stream": manifest["selected_frames"]["stream"],
              "frames": manifest["actual_validation"]["frames"], "point_rows": manifest["actual_validation"]["point_rows"],
              "evidence_files": len(manifest["source_evidence"]),
              "qualification_delta": manifest["qualification_delta"], "record_content_reverified": False,
              "physical_fit_started": manifest["physical_fit_started"]}
    if verify:
        _check_evidence_unchanged(manifest["source_evidence"])
        result["verification"] = _verify_pair(manifest["selected_rows"], manifest["selected_frames"])
        result["record_content_reverified"] = True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("commit", help="Commit explicit source qualification deltas and publish a new immutable candidate manifest")
    command.add_argument("--request", required=True)
    command.add_argument("--out", required=True)
    command = commands.add_parser("inspect", help="Show compact manifest counts, without dumping source points")
    command.add_argument("manifest", nargs="?")
    command.add_argument("--manifest", dest="manifest_option")
    command.add_argument("--verify", action="store_true", help="Re-read every actual source record and evidence file")
    args = parser.parse_args(argv)
    try:
        if args.command == "commit":
            result = commit(args.request, args.out)
        else:
            path = args.manifest or args.manifest_option
            if not path or (args.manifest and args.manifest_option):
                raise ValueError("Specify exactly one source manifest")
            result = inspect_manifest(path, verify=args.verify)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, sqlite3.Error, SourceStoreError) as error:
        result = {"ok": False, "error": type(error).__name__, "message": str(error), "manifest_published": False}
        if isinstance(error, CommitFailure):
            result["sealed_unpublished_streams"] = error.sealed_streams
            result["history_automatically_deleted"] = False
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
