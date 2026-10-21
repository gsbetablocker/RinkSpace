"""One checked record reader for legacy files and immutable logical source streams.

A source SQLite database may grow by appending sealed streams. A content-addressed
reference identifies recovered JSON values, not the changing container bytes.
"""
from pathlib import Path
import gzip
import hashlib
import json
from .io_utils import file_sha256


def canonical_record(row):
    return (json.dumps(row, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")


def pin_reference(ref, pins=None):
    """Pin physical bytes where requested; logical content is verified while read."""
    pins = {} if pins is None else pins
    path = Path(ref["store"] if "store" in ref else ref["file"]).resolve()
    if "store" in ref and ref.get("store_content_addressed") is True:
        if not ref.get("stream_semantic_sha256"):
            raise ValueError("Content-addressed source stream requires its expected semantic SHA")
        return path
    expected = ref.get("store_sha256") if "store" in ref else ref["sha256"]
    if not expected:
        raise ValueError("Physical source file SHA is required")
    if str(path) in pins:
        if pins[str(path)] != expected:
            raise ValueError("Conflicting physical source file pins")
    elif file_sha256(path) != expected:
        raise ValueError("Physical source bytes changed")
    pins[str(path)] = expected
    return path


def checked_records(ref, pins=None):
    """Yield exact records; reject changed bytes, metadata, count or actual content.

    A content-addressed stream is read in one read-only SQL transaction and every
    recovered row contributes to its expected canonical JSON SHA. Historical raw
    file SHA is optional provenance, separately verified against stream metadata.
    With an explicit frame selector, only that one record is hashed against its
    frame-semantic SHA; the whole-stream pin is checked as metadata, not rehashed.
    Frozen-store and legacy references continue to pin physical file bytes.
    """
    pins = {} if pins is None else pins
    path = pin_reference(ref, pins)
    frame = ref.get("frame")
    if "frame" in ref:
        if "store" not in ref or not isinstance(frame, (list, tuple)) or len(frame) != 2 or not isinstance(frame[0], str) or not isinstance(frame[1], int) or isinstance(frame[1], bool):
            raise ValueError("A local source frame requires a store and explicit [shot_id, native_pts]")
        if not ref.get("frame_semantic_sha256"):
            raise ValueError("A local source frame requires its actual frame-semantic SHA")
    elif "frame_semantic_sha256" in ref:
        raise ValueError("A frame-semantic SHA requires an explicit frame selector")
    if "store" in ref:
        from .source_record_store import SourceRecordStore
        original = ref.get("original_sha256")
        semantic = ref.get("stream_semantic_sha256")
        logical = ref.get("stream")
        if not logical or not (original or semantic):
            raise ValueError("Store reference requires an original-byte or stream-semantic identity")
        if original and ref.get("sha256", original) != original:
            raise ValueError("Logical expected SHA differs from original byte SHA")
        if not original and "sha256" in ref:
            raise ValueError("A new semantic stream must not invent an original file byte SHA")
        with SourceRecordStore(path, readonly=True) as store:
            store.db.execute("BEGIN")
            try:
                info = store.stream_info(logical)
                if not info.get("sealed"):
                    raise ValueError("Logical source stream is not sealed")
                if ref.get("kind") is not None and info["kind"] != ref["kind"]:
                    raise ValueError("Logical source stream kind differs")
                if original and info.get("original_sha256") != original:
                    raise ValueError("Original source byte identity changed")
                if semantic and info.get("semantic_sha256") != semantic:
                    raise ValueError("Source semantic stream identity changed")
                expected = ref["frame_semantic_sha256"] if frame is not None else semantic or info["semantic_sha256"]
                expected_count = 1 if frame is not None else info["record_count"]
                digest = hashlib.sha256()
                count = 0
                for row in store.iterate_records(logical, frame=tuple(frame) if frame is not None else None):
                    count += 1
                    if frame is not None and (count > 1 or not isinstance(row, dict) or (row.get("shot_id"), row.get("native_pts")) != tuple(frame)):
                        raise ValueError("Local source frame identity or record count differs")
                    digest.update(canonical_record(row))
                    yield row
                if count != expected_count:
                    raise ValueError("Logical source stream record count differs")
                if digest.hexdigest() != expected:
                    raise ValueError("Actual recovered source frame differs from its frame-semantic SHA" if frame is not None else "Actual recovered source records differ from the pinned semantic SHA")
                store.db.execute("COMMIT")
            except BaseException:
                if store.db.in_transaction:
                    store.db.execute("ROLLBACK")
                raise
    else:
        op = gzip.open if path.suffix.lower() == ".gz" else open
        with op(path, "rt", encoding="utf-8-sig") as records:
            for line in records:
                if line.strip():
                    yield json.loads(line)
    if str(path) in pins and file_sha256(path) != pins[str(path)]:
        raise ValueError("Physical source bytes changed during consumption")
