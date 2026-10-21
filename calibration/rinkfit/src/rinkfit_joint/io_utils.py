"""Portable, deterministic files. Original evidence is never edited in place."""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path


def read_json(path):
    if isinstance(path, dict):
        from .source_record_io import checked_records
        records = checked_records(path); missing = object()
        try:
            value = next(records, missing)
            if value is missing or next(records, missing) is not missing:
                raise ValueError("A JSON reference must contain exactly one checked record")
            return value
        finally:
            records.close()
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(value, dict) and value.get("schema") == "rinkfit.joint_packet.v1" and ("dense_source_binding" in value or "dense_parent_packet" in value):
        from .dense_joint_packet import expand_dense_packet
        return expand_dense_packet(value, path)
    return value


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    if isinstance(path, dict):
        from .source_record_io import checked_records
        yield from checked_records(path)
        return
    p = Path(path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt", encoding="utf-8") as f:
        yield from (json.loads(line) for line in f if line.strip())


def write_jsonl(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()
