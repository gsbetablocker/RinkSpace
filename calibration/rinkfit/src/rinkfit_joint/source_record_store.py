"""Lossless, immutable source-record streams with per-frame qualification deltas.

SQLite is a storage container, never a source measurement or a physical checkpoint.
Source IDs/UV are stored once per native frame; a derived stream stores only changed
membership/qualification. Original byte SHA is provenance, not a hash of this DB.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import contextmanager
import gzip
import hashlib
import json
import math
import ntpath
from pathlib import Path
import sqlite3
import zlib

FORMAT = "rinkspace.source_record_store.v1"
KINDS = {"selected_rows", "selected_frames", "coverage", "excluded", "consumer_bindings", "native_frames"}
FRAME_SCHEMAS = {"rinkfit.dense_native_board_source_frame.v1"}
GEOMETRY_KEYS = {"shot_id", "native_pts", "uv", "time_base", "source_point_id", "source_row_id"}


class SourceStoreError(ValueError):
    pass


class UnknownSchemaError(SourceStoreError):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _same(a, b):
    # bool/int and int/float must not collapse merely because Python compares equal.
    return _json(a) == _json(b)


def _logical(path):
    return ntpath.normcase(ntpath.normpath(str(path)))


def _sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _factor(rows):
    if not rows:
        return {"common": {}, "rows": []}
    common = dict(rows[0])
    for r in rows[1:]:
        for k in list(common):
            if k not in r or not _same(common[k], r[k]):
                del common[k]
    return {"common": common, "rows": [{k: v for k, v in r.items() if k not in common} for r in rows]}


def _unfactor(value):
    return [dict(value["common"], **r) for r in value["rows"]]


def _diff(a, b):
    if _same(a, b):
        return None
    if isinstance(a, dict) and isinstance(b, dict):
        changes = {}
        for k, v in b.items():
            d = _diff(a[k], v) if k in a else ["set", v]
            if d is not None:
                changes[k] = d
        return ["dict", [k for k in a if k not in b], changes]
    return ["set", b]


def _apply(a, delta):
    if delta is None:
        return a
    if delta[0] == "set":
        return delta[1]
    if delta[0] != "dict" or not isinstance(a, dict):
        raise SourceStoreError("Invalid dictionary delta")
    result = dict(a)
    for k in delta[1]:
        del result[k]
    for k, d in delta[2].items():
        result[k] = _apply(result.get(k), d)
    return result


def _frame_key(row):
    if not isinstance(row, dict) or not isinstance(row.get("shot_id"), str):
        raise UnknownSchemaError("Expected an object with string shot_id")
    pts = row.get("native_pts")
    if not isinstance(pts, int) or isinstance(pts, bool):
        raise UnknownSchemaError("native_pts must be an integer source PTS")
    return row["shot_id"], pts


def _validate(row, kind):
    _frame_key(row)
    # JSON finite values are checked even if no encoding change is needed.
    _json(row)
    if kind == "native_frames":
        if row.get("schema") not in FRAME_SCHEMAS or not isinstance(row.get("rows"), list):
            raise UnknownSchemaError("Unknown native frame schema")
        for r in row["rows"]:
            if _frame_key(r) != _frame_key(row) or not isinstance(r.get("source_row_id"), str):
                raise UnknownSchemaError("Native source row lacks its original identity/clock")
            _uv(r)
    elif "schema" in row:
        raise UnknownSchemaError("Unexpected schema tag in dense56 JSONL record")
    elif kind == "selected_rows":
        for k in ("selected_row_id", "source_point_id", "display_class"):
            if not isinstance(row.get(k), str):
                raise UnknownSchemaError("selected_rows requires string " + k)
        _uv(row)
    elif kind == "selected_frames":
        if not isinstance(row.get("points"), list) or not isinstance(row.get("decoded_native_BGR_sha256"), str):
            raise UnknownSchemaError("Unknown selected_frames projection")
        for r in row["points"]:
            if set(r) != {"selected_row_id", "uv", "display_class"}:
                raise UnknownSchemaError("Unsupported point-frame projection keys")
            _uv(r)
    elif kind == "coverage":
        if not isinstance(row.get("selected_rows_by_part"), dict):
            raise UnknownSchemaError("Unknown dense56 coverage")
    elif kind == "excluded":
        if not isinstance(row.get("reason"), str):
            raise UnknownSchemaError("Unknown dense56 exclusion")
    elif kind == "consumer_bindings":
        if not isinstance(row.get("source_BGR_sha256"), str):
            raise UnknownSchemaError("Unknown dense56 consumer binding")
    else:
        raise UnknownSchemaError("Unsupported stream kind: " + str(kind))


def _uv(row):
    uv = row.get("uv")
    if not isinstance(uv, list) or len(uv) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in uv):
        raise UnknownSchemaError("Expected finite, two-dimensional original UV")


def _jsonl(path):
    op = gzip.open if str(path).lower().endswith(".gz") else open
    with op(path, "rt", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _groups(rows, kind):
    last = None
    block = []
    visited_shots = set()
    for r in rows:
        _validate(r, kind)
        key = _frame_key(r)
        if key != last and last is not None:
            yield last, block
            block = []
            if key[0] == last[0] and key[1] <= last[1]:
                raise SourceStoreError("Stream is not ordered by increasing native PTS")
            if key[0] != last[0]:
                visited_shots.add(last[0])
                if key[0] in visited_shots:
                    raise SourceStoreError("Shot frame records are not contiguous")
        block.append(r)
        last = key
    if last is not None:
        yield last, block


class SourceRecordStore:
    """One connection; readers reconstruct at most one native frame at a time.

    Streams are sealed at transaction commit. No API mutates a sealed stream.
    The database may hold additional new streams until its owner freezes the DB.
    """
    def __init__(self, db_path, readonly=False):
        self.path = Path(db_path)
        self.readonly = readonly
        self._atom_cache = OrderedDict()
        self._read_atom_cache = OrderedDict()
        self._keys_cache = {}
        self._seen_strings = OrderedDict()
        self._source_cache = OrderedDict()
        if readonly:
            self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        if not readonly:
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS atoms(id INTEGER PRIMARY KEY, digest BLOB UNIQUE NOT NULL, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS blobs(id INTEGER PRIMARY KEY, digest BLOB UNIQUE NOT NULL, raw_bytes INTEGER NOT NULL, payload BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS streams(id INTEGER PRIMARY KEY, logical_key TEXT UNIQUE NOT NULL, logical_path TEXT NOT NULL,
                    kind TEXT NOT NULL, mode TEXT NOT NULL, parent_id INTEGER REFERENCES streams(id), projection_id INTEGER REFERENCES streams(id),
                    original_sha256 TEXT, original_bytes INTEGER, semantic_sha256 TEXT NOT NULL DEFAULT '',
                    frame_count INTEGER NOT NULL DEFAULT 0, record_count INTEGER NOT NULL DEFAULT 0,
                    order_blob INTEGER REFERENCES blobs(id), sealed INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS frames(stream_id INTEGER NOT NULL REFERENCES streams(id), shot TEXT NOT NULL, pts INTEGER NOT NULL,
                    blob_id INTEGER NOT NULL REFERENCES blobs(id), record_count INTEGER NOT NULL, semantic_sha256 TEXT NOT NULL,
                    PRIMARY KEY(stream_id, shot, pts));
                CREATE TABLE IF NOT EXISTS source_blocks(shot TEXT NOT NULL, pts INTEGER NOT NULL, start_slot INTEGER NOT NULL,
                    count INTEGER NOT NULL, blob_id INTEGER NOT NULL REFERENCES blobs(id), PRIMARY KEY(shot, pts, start_slot));
                CREATE TABLE IF NOT EXISTS concat_children(stream_id INTEGER NOT NULL REFERENCES streams(id), ordinal INTEGER NOT NULL,
                    child_id INTEGER NOT NULL REFERENCES streams(id), PRIMARY KEY(stream_id, ordinal));
            ''')
            old = self.db.execute("SELECT value FROM meta WHERE key='format'").fetchone()
            if old is None:
                self.db.execute("INSERT INTO meta VALUES('format', ?)", (FORMAT,))
                self.db.commit()
        row = self.db.execute("SELECT value FROM meta WHERE key='format'").fetchone()
        if row is None or row[0] != FORMAT:
            raise UnknownSchemaError("Unknown source-record database schema")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.db.close()

    @contextmanager
    def _transaction(self):
        if self.readonly:
            raise SourceStoreError("Read-only source-record store")
        if self.db.in_transaction:
            raise SourceStoreError("Nested source-record transaction")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.commit()
        except BaseException:
            self.db.rollback()
            self._atom_cache.clear()
            self._read_atom_cache.clear()
            self._keys_cache.clear()
            self._source_cache.clear()
            raise

    def _checked_stream(self, r):
        if r is None or r["sealed"] != 1 or r["kind"] not in KINDS or r["mode"] not in {"base", "delta", "projection", "concat"}:
            raise UnknownSchemaError("Unknown, incomplete or unsupported stored stream")
        if r["record_count"] < 0 or r["frame_count"] < 0:
            raise SourceStoreError("Negative sealed stream counts")
        if r["mode"] == "base" and r["parent_id"] is not None:
            raise SourceStoreError("Base stream has an unexpected parent")
        if r["mode"] == "delta" and r["parent_id"] is None:
            raise SourceStoreError("Delta stream has no parent")
        if r["parent_id"] is not None:
            p = self.db.execute("SELECT * FROM streams WHERE id=?", (r["parent_id"],)).fetchone()
            if p is None or p["id"] >= r["id"] or p["sealed"] != 1 or p["kind"] != r["kind"] or p["mode"] not in {"base", "delta", "projection"}:
                raise SourceStoreError("Invalid or cyclic stream parent")
        if r["mode"] == "projection":
            p = self.db.execute("SELECT * FROM streams WHERE id=?", (r["projection_id"],)).fetchone()
            if r["kind"] != "selected_frames" or p is None or p["id"] >= r["id"] or p["sealed"] != 1 or p["kind"] != "selected_rows":
                raise SourceStoreError("Invalid selected-frame projection source")
        elif r["projection_id"] is not None:
            raise SourceStoreError("Unexpected projection source")
        if r["mode"] == "concat":
            if r["parent_id"] is not None:
                raise SourceStoreError("Concat stream cannot have a parent")
            children = self.db.execute("SELECT s.* FROM concat_children c JOIN streams s ON s.id=c.child_id WHERE c.stream_id=? ORDER BY c.ordinal", (r["id"],)).fetchall()
            if not children or any(c["id"] >= r["id"] or c["sealed"] != 1 or c["kind"] != r["kind"] or c["mode"] not in {"base", "delta", "projection"} for c in children):
                raise SourceStoreError("Invalid or cyclic concat children")
        return r

    def _stream(self, logical):
        r = self.db.execute("SELECT * FROM streams WHERE logical_key=? AND sealed=1", (_logical(logical),)).fetchone()
        if r is None:
            raise KeyError("Unknown sealed stream: " + str(logical))
        return self._checked_stream(r)

    def stream_info(self, logical):
        r = dict(self._stream(logical))
        r["schema"] = FORMAT
        r["stored_frame_overrides"] = self.db.execute("SELECT count(*) FROM frames WHERE stream_id=?", (r["id"],)).fetchone()[0]
        return r

    def _atom(self, s):
        if s in self._atom_cache:
            self._atom_cache.move_to_end(s)
            return self._atom_cache[s]
        digest = hashlib.sha256(s.encode("utf-8")).digest()
        r = self.db.execute("SELECT id,value FROM atoms WHERE digest=?", (digest,)).fetchone()
        if r is None:
            ident = self.db.execute("INSERT INTO atoms(digest,value) VALUES(?,?)", (digest, s)).lastrowid
        else:
            if r[1] != s:
                raise SourceStoreError("String digest collision")
            ident = r[0]
        self._atom_cache[s] = ident
        if len(self._atom_cache) > 8192:
            self._atom_cache.popitem(last=False)
        return ident

    def _encode(self, value, counts):
        if isinstance(value, str) and len(value) >= 56:
            if counts[value] >= 2 or value in self._seen_strings:
                return ["@s", self._atom(value)]
            self._seen_strings[value] = True
            if len(self._seen_strings) > 8192:
                self._seen_strings.popitem(last=False)
            return value
        if isinstance(value, list):
            out = [self._encode(v, counts) for v in value]
            return ["@l", out] if value and isinstance(value[0], str) and value[0] in {"@s", "@l"} else out
        if isinstance(value, dict):
            return {k: self._encode(v, counts) for k, v in value.items()}
        return value

    def _decode(self, value):
        if isinstance(value, list):
            if len(value) == 2 and value[0] == "@s":
                ident = value[1]
                if ident in self._read_atom_cache:
                    self._read_atom_cache.move_to_end(ident)
                    return self._read_atom_cache[ident]
                r = self.db.execute("SELECT value,digest FROM atoms WHERE id=?", (ident,)).fetchone()
                if r is None or hashlib.sha256(r[0].encode("utf-8")).digest() != r[1]:
                    raise SourceStoreError("Missing or corrupt dictionary atom")
                self._read_atom_cache[ident] = r[0]
                if len(self._read_atom_cache) > 8192:
                    self._read_atom_cache.popitem(last=False)
                return r[0]
            if len(value) == 2 and value[0] == "@l":
                return [self._decode(v) for v in value[1]]
            return [self._decode(v) for v in value]
        if isinstance(value, dict):
            return {k: self._decode(v) for k, v in value.items()}
        return value

    def _put(self, value):
        counts = Counter()
        def collect(v):
            if isinstance(v, str) and len(v) >= 56:
                counts[v] += 1
            elif isinstance(v, list):
                for x in v:
                    collect(x)
            elif isinstance(v, dict):
                for x in v.values():
                    collect(x)
        collect(value)
        raw = _json(self._encode(value, counts))
        digest = hashlib.sha256(raw).digest()
        r = self.db.execute("SELECT id FROM blobs WHERE digest=?", (digest,)).fetchone()
        if r:
            return r[0]
        return self.db.execute("INSERT INTO blobs(digest,raw_bytes,payload) VALUES(?,?,?)", (digest, len(raw), zlib.compress(raw, 6))).lastrowid

    def _get(self, blob):
        r = self.db.execute("SELECT payload,raw_bytes,digest FROM blobs WHERE id=?", (blob,)).fetchone()
        if r is None:
            raise SourceStoreError("Missing compressed blob")
        raw = zlib.decompress(r[0])
        if len(raw) != r[1] or hashlib.sha256(raw).digest() != r[2]:
            raise SourceStoreError("Compressed blob integrity failure")
        return self._decode(json.loads(raw))

    def _catalog(self, key):
        if key in self._source_cache:
            self._source_cache.move_to_end(key)
            return self._source_cache[key]
        values = []
        for r in self.db.execute("SELECT start_slot,count,blob_id FROM source_blocks WHERE shot=? AND pts=? ORDER BY start_slot", key):
            if r[0] != len(values):
                raise SourceStoreError("Noncontiguous source geometry catalog")
            block = self._get(r[2])
            if len(block) != r[1]:
                raise SourceStoreError("Invalid source geometry count")
            values.extend(block)
        by_id = {v["source_id"]: i for i, v in enumerate(values)}
        if len(by_id) != len(values):
            raise SourceStoreError("Duplicate source geometry identity")
        cache = (values, by_id)
        self._source_cache[key] = cache
        if len(self._source_cache) > 2:
            self._source_cache.popitem(last=False)
        return cache

    def _encode_points(self, key, rows, id_field):
        values, by_id = self._catalog(key)
        start = len(values)
        out = []
        for r in rows:
            if "_source_id_field" in r or ("source_row_id" in r and "source_point_id" in r):
                raise UnknownSchemaError("Ambiguous or reserved source identity property")
            geo = {"source_id": r[id_field], "uv": r["uv"]}
            if "time_base" in r:
                geo["time_base"] = r["time_base"]
            slot = by_id.get(geo["source_id"])
            if slot is None:
                slot = len(values)
                values.append(geo)
                by_id[geo["source_id"]] = slot
            elif not _same(values[slot], geo):
                raise SourceStoreError("Existing source ID changed UV/clock; a new measurement requires a new source ID: " + geo["source_id"])
            qual = {k: v for k, v in r.items() if k not in GEOMETRY_KEYS}
            # The spelling of the old ID property is part of lossless reconstruction.
            qual["_source_id_field"] = id_field
            out.append({"source_slot": slot, "qualification": qual})
        if len(values) > start:
            blob = self._put(values[start:])
            self.db.execute("INSERT INTO source_blocks VALUES(?,?,?,?,?)", (*key, start, len(values) - start, blob))
        return out

    def _decode_points(self, key, encoded):
        values, unused = self._catalog(key)
        out = []
        for item in encoded:
            geo = values[item["source_slot"]]
            row = dict(item["qualification"])
            field = row.pop("_source_id_field")
            row.update(shot_id=key[0], native_pts=key[1], uv=list(geo["uv"]))
            row[field] = geo["source_id"]
            if "time_base" in geo:
                row["time_base"] = geo["time_base"]
            out.append(row)
        return out

    def _keys(self, stream):
        if stream["id"] not in self._keys_cache:
            self._keys_cache[stream["id"]] = self._load_keys(stream)
        return self._keys_cache[stream["id"]]

    def _load_keys(self, stream):
        if stream["mode"] == "concat":
            keys = []
            for r in self.db.execute("SELECT child_id FROM concat_children WHERE stream_id=? ORDER BY ordinal", (stream["id"],)):
                keys.extend(self._keys(self.db.execute("SELECT * FROM streams WHERE id=?", (r[0],)).fetchone()))
            return keys
        if stream["order_blob"] is not None:
            return [tuple(k) for k in self._get(stream["order_blob"])]
        if stream["parent_id"] is not None:
            return self._keys(self.db.execute("SELECT * FROM streams WHERE id=?", (stream["parent_id"],)).fetchone())
        return []

    def _encoded_frame(self, stream, key):
        found = self.db.execute("SELECT blob_id FROM frames WHERE stream_id=? AND shot=? AND pts=?", (stream["id"], *key)).fetchone()
        if found:
            payload = self._get(found[0])
            if payload["type"] == "full":
                return _unfactor(payload["records"])
            if payload["type"] == "full_points":
                qualifications = _unfactor(payload["qualification"])
                if len(qualifications) != len(payload["source_slots"]):
                    raise SourceStoreError("Invalid factored source-row count")
                return [{"source_slot": slot, "qualification": q} for slot, q in zip(payload["source_slots"], qualifications)]
            if payload["type"] == "delta":
                parent = self.db.execute("SELECT * FROM streams WHERE id=?", (stream["parent_id"],)).fetchone()
                base = self._encoded_frame(parent, key)
                if stream["kind"] == "selected_rows":
                    return self._apply_row_delta(base, payload["changes"])
                return _apply(base, payload["changes"])
            raise SourceStoreError("Unknown frame payload type")
        if stream["parent_id"] is not None:
            parent = self.db.execute("SELECT * FROM streams WHERE id=?", (stream["parent_id"],)).fetchone()
            return self._encoded_frame(parent, key)
        raise KeyError("Frame absent from stream: " + str(key))

    @staticmethod
    def _apply_row_delta(base, delta):
        records = {r["qualification"]["selected_row_id"]: r for r in base}
        for ident in delta["remove"]:
            del records[ident]
        for ident, change in delta["update"].items():
            records[ident] = _apply(records[ident], change)
        for r in delta["add"]:
            records[r["qualification"]["selected_row_id"]] = r
        order = delta.get("order")
        if order is None:
            order = [r["qualification"]["selected_row_id"] for r in base if r["qualification"]["selected_row_id"] in records]
            order.extend(r["qualification"]["selected_row_id"] for r in delta["add"])
        if len(order) != len(records) or len(set(order)) != len(order) or set(order) != set(records):
            raise SourceStoreError("Invalid source-row delta membership/order")
        return [records[i] for i in order]

    @staticmethod
    def _row_delta(base, rows):
        before = {r["qualification"]["selected_row_id"]: r for r in base}
        after = {r["qualification"]["selected_row_id"]: r for r in rows}
        if len(before) != len(base) or len(after) != len(rows):
            raise SourceStoreError("Duplicate selected_row_id in a native frame")
        remove = [i for i in before if i not in after]
        add = [r for i, r in after.items() if i not in before]
        update = {i: d for i in after if i in before and (d := _diff(before[i], after[i])) is not None}
        result = {"remove": remove, "add": add, "update": update}
        default = [i for i in before if i in after] + [r["qualification"]["selected_row_id"] for r in add]
        if default != list(after):
            result["order"] = list(after)
        return result

    def _read_frame(self, stream, key):
        self._checked_stream(stream)
        if stream["mode"] == "concat":
            matches = []
            for r in self.db.execute("SELECT child_id FROM concat_children WHERE stream_id=? ORDER BY ordinal", (stream["id"],)):
                child = self.db.execute("SELECT * FROM streams WHERE id=?", (r[0],)).fetchone()
                if key in self._keys(child):
                    matches.append(child)
            if len(matches) != 1:
                raise SourceStoreError("Ambiguous or absent concat native frame")
            return self._read_frame(matches[0], key)
        encoded = self._encoded_frame(stream, key)
        if stream["kind"] == "selected_rows":
            return self._decode_points(key, encoded)
        if stream["kind"] == "native_frames":
            out = []
            for r in encoded:
                frame = dict(r["header"])
                frame["rows"] = self._decode_points(key, _unfactor(r["source_rows"]))
                out.append(frame)
            return out
        if stream["mode"] == "projection":
            src = self.db.execute("SELECT * FROM streams WHERE id=?", (stream["projection_id"],)).fetchone()
            rows = self._read_frame(src, key)
            points = [{k: r[k] for k in ("selected_row_id", "uv", "display_class")} for r in rows]
            if len(encoded) != 1:
                raise SourceStoreError("Invalid selected-frame header")
            return [dict(encoded[0], points=points)]
        return encoded

    def iterate_records(self, logical_path, frame=None):
        stream = self._stream(logical_path)
        keys = self._keys(stream)
        if frame is not None:
            if isinstance(frame, int) and not isinstance(frame, bool):
                matches = [k for k in keys if k[1] == frame]
                if len(matches) > 1:
                    raise SourceStoreError("PTS belongs to multiple shots; specify (shot_id, native_pts)")
                if not matches:
                    raise KeyError("Native frame absent from stream: " + str(frame))
                keys = matches
            else:
                key = tuple(frame)
                if key not in keys:
                    raise KeyError("Native frame absent from stream: " + str(key))
                keys = [key]
        for key in keys:
            yield from self._read_frame(stream, key)

    def _new_stream(self, logical, kind, mode, parent=None, projection=None, original_sha=None, original_bytes=None):
        if kind not in KINDS:
            raise UnknownSchemaError("Unsupported stream kind")
        if self.db.execute("SELECT 1 FROM streams WHERE logical_key=?", (_logical(logical),)).fetchone():
            raise SourceStoreError("Stream already exists; immutable stream cannot be replaced: " + str(logical))
        return self.db.execute("INSERT INTO streams(logical_key,logical_path,kind,mode,parent_id,projection_id,original_sha256,original_bytes) VALUES(?,?,?,?,?,?,?,?)", (_logical(logical), str(logical), kind, mode, parent, projection, original_sha, original_bytes)).lastrowid

    def _write_frame(self, sid, kind, key, records, parent=None, projection=False):
        for record in records:
            _validate(record, kind)
        if kind == "selected_rows" and len({r["selected_row_id"] for r in records}) != len(records):
            raise SourceStoreError("Duplicate selected-row identity")
        if kind == "selected_rows":
            encoded = self._encode_points(key, records, "source_point_id")
        elif kind == "native_frames":
            encoded = [{"header": {k: v for k, v in r.items() if k != "rows"}, "source_rows": _factor(self._encode_points(key, r["rows"], "source_row_id"))} for r in records]
        elif projection:
            encoded = [{k: v for k, v in r.items() if k != "points"} for r in records]
        else:
            encoded = records
        semantic = hashlib.sha256(_json(records)).hexdigest()
        base = None
        if parent is not None and key in self._keys(parent):
            base = self._encoded_frame(parent, key)
            if _same(base, encoded):
                return
        if kind == "selected_rows":
            payload = {"type": "full_points", "source_slots": [r["source_slot"] for r in encoded],
                       "qualification": _factor([r["qualification"] for r in encoded])}
        else:
            payload = {"type": "full", "records": _factor(encoded)}
        if base is not None:
            delta = self._row_delta(base, encoded) if kind == "selected_rows" else _diff(base, encoded)
            candidate = {"type": "delta", "changes": delta}
            if len(_json(candidate)) < len(_json(payload)):
                payload = candidate
        blob = self._put(payload)
        self.db.execute("INSERT INTO frames VALUES(?,?,?,?,?,?)", (sid, *key, blob, len(records), semantic))

    def _seal(self, sid, keys, count, digest, parent=None):
        if parent is not None and keys == self._keys(parent):
            order_blob = None
        else:
            order_blob = self._put(keys)
        self.db.execute("UPDATE streams SET semantic_sha256=?,frame_count=?,record_count=?,order_blob=?,sealed=1 WHERE id=?", (digest, len(keys), count, order_blob, sid))

    def import_jsonl_stream(self, path, parent_stream=None, kind=None, logical_path=None):
        path = Path(path)
        kind = kind or path.name.removesuffix(".gz").removesuffix(".jsonl")
        if kind == "selected_frames":
            raise SourceStoreError("Use import_selected_frames(path, selected_rows_stream) to avoid storing UV twice")
        if kind not in KINDS or kind == "native_frames":
            raise UnknownSchemaError("Unsupported JSONL stream kind: " + str(kind))
        logical = logical_path or str(path)
        stat = path.stat()
        original_sha = _sha_file(path)
        parent = self._stream(parent_stream) if parent_stream is not None else None
        if parent is not None and parent["kind"] != kind:
            raise UnknownSchemaError("Parent stream kind differs")
        with self._transaction():
            sid = self._new_stream(logical, kind, "delta" if parent else "base", parent["id"] if parent else None, original_sha=original_sha, original_bytes=stat.st_size)
            keys, count, digest = [], 0, hashlib.sha256()
            for key, rows in _groups(_jsonl(path), kind):
                self._write_frame(sid, kind, key, rows, parent)
                keys.append(key)
                count += len(rows)
                for row in rows:
                    digest.update(_json(row) + b"\n")
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise SourceStoreError("Source file changed during import")
            self._seal(sid, keys, count, digest.hexdigest(), parent)
        return self.stream_info(logical)

    def import_selected_frames(self, path, selected_rows_stream, logical_path=None, parent_stream=None):
        path = Path(path)
        logical = logical_path or str(path)
        source = self._stream(selected_rows_stream)
        if source["kind"] != "selected_rows":
            raise UnknownSchemaError("Point-frame projection requires selected_rows")
        parent = self._stream(parent_stream) if parent_stream is not None else None
        if parent is not None and parent["kind"] != "selected_frames":
            raise UnknownSchemaError("Parent must be selected_frames")
        stat = path.stat()
        original_sha = _sha_file(path)
        with self._transaction():
            sid = self._new_stream(logical, "selected_frames", "projection", parent["id"] if parent else None, source["id"], original_sha, stat.st_size)
            keys, count, digest = [], 0, hashlib.sha256()
            for key, records in _groups(_jsonl(path), "selected_frames"):
                if len(records) != 1:
                    raise SourceStoreError("Multiple point-frame records for one source frame")
                expected = [{k: r[k] for k in ("selected_row_id", "uv", "display_class")} for r in self._read_frame(source, key)]
                if not _same(records[0]["points"], expected):
                    raise SourceStoreError("Selected frame disagrees with original selected-row IDs/UV/color/order")
                self._write_frame(sid, "selected_frames", key, records, parent, projection=True)
                keys.append(key)
                count += 1
                digest.update(_json(records[0]) + b"\n")
            if keys != self._keys(source):
                raise SourceStoreError("Point-frame native coverage differs from selected rows")
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise SourceStoreError("Source point-frame file changed during import")
            self._seal(sid, keys, count, digest.hexdigest(), parent)
        return self.stream_info(logical)

    def project_selected_frames(self, logical_path, selected_rows_stream, frame_header_stream):
        """Create a next-generation point preview from stored rows and old headers.

        No full points JSONL or second UV table is emitted. Header inheritance is
        explicit and same native coverage is required. No old byte hash is invented.
        """
        source = self._stream(selected_rows_stream)
        header = self._stream(frame_header_stream)
        if source["kind"] != "selected_rows" or header["kind"] != "selected_frames" or header["mode"] != "projection":
            raise UnknownSchemaError("Projection requires selected rows and an existing point-frame header stream")
        keys = self._keys(source)
        if keys != self._keys(header):
            raise SourceStoreError("Projection source/header native coverage differs")
        with self._transaction():
            sid = self._new_stream(logical_path, "selected_frames", "projection", header["id"], source["id"])
            digest = hashlib.sha256()
            for key in keys:
                headers = self._encoded_frame(header, key)
                if len(headers) != 1 or "points" in headers[0]:
                    raise SourceStoreError("Invalid inherited frame header")
                rows = self._read_frame(source, key)
                record = dict(headers[0], points=[{k: r[k] for k in ("selected_row_id", "uv", "display_class")} for r in rows])
                _validate(record, "selected_frames")
                digest.update(_json(record) + b"\n")
            self._seal(sid, keys, len(keys), digest.hexdigest(), header)
        return self.stream_info(logical_path)

    def import_frame_jsons(self, logical_path, frame_paths, parent_stream=None):
        """Import known native material frames; preserve every old JSON value.

        Original per-file SHA is kept in stream receipt via native_file_refs().
        The logical stream has no invented old aggregate compressed-file SHA.
        """
        parent = self._stream(parent_stream) if parent_stream is not None else None
        if parent is not None and parent["kind"] != "native_frames":
            raise UnknownSchemaError("Native frame parent kind differs")
        with self._transaction():
            sid = self._new_stream(logical_path, "native_frames", "delta" if parent else "base", parent["id"] if parent else None)
            keys, digest, refs = [], hashlib.sha256(), []
            def rows():
                for p in frame_paths:
                    p = Path(p)
                    raw = p.read_bytes()
                    value = json.loads(raw.decode("utf-8-sig"))
                    refs.append({"file": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
                    yield value
            for key, records in _groups(rows(), "native_frames"):
                if len(records) != 1:
                    raise SourceStoreError("Multiple native source files at one shot/PTS")
                self._write_frame(sid, "native_frames", key, records, parent)
                keys.append(key)
                digest.update(_json(records[0]) + b"\n")
            self.db.execute("INSERT INTO meta VALUES(?,?)", ("native_file_refs:" + str(sid), str(self._put(refs))))
            self._seal(sid, keys, len(keys), digest.hexdigest(), parent)
        return self.stream_info(logical_path)

    def native_file_refs(self, logical_path):
        stream = self._stream(logical_path)
        row = self.db.execute("SELECT value FROM meta WHERE key=?", ("native_file_refs:" + str(stream["id"]),)).fetchone()
        return self._get(int(row[0])) if row else []

    def register_concat(self, logical_path, children, original_sha256=None, original_bytes=None):
        sources = [self._stream(p) for p in children]
        if not sources or len({s["kind"] for s in sources}) != 1:
            raise SourceStoreError("Concat requires at least one child, all of the same kind")
        keys = [k for s in sources for k in self._keys(s)]
        if len(keys) != len(set(keys)):
            raise SourceStoreError("Concat children overlap native frames")
        digest, count = hashlib.sha256(), 0
        for source in sources:
            for row in self.iterate_records(source["logical_path"]):
                digest.update(_json(row) + b"\n")
                count += 1
        with self._transaction():
            sid = self._new_stream(logical_path, sources[0]["kind"], "concat", original_sha=original_sha256, original_bytes=original_bytes)
            self.db.executemany("INSERT INTO concat_children VALUES(?,?,?)", [(sid, i, s["id"]) for i, s in enumerate(sources)])
            self.db.execute("UPDATE streams SET semantic_sha256=?,frame_count=?,record_count=?,sealed=1 WHERE id=?", (digest.hexdigest(), len(keys), count, sid))
        return self.stream_info(logical_path)

    def fork_rows_stream(self, logical_path, parent_stream, changes):
        """Create a new selected-row stream directly, without a full JSONL staging file.

        changes[(shot_id, PTS)] = {'remove': [selected_row_id],
            'update': {selected_row_id: partial_qualification}, 'add': [full_rows],
            'order': [selected_row_id, ...] optional}.
        Geometry/clock cannot be patched. New measurement uses a new source ID.
        """
        parent = self._stream(parent_stream)
        if parent["kind"] != "selected_rows":
            raise UnknownSchemaError("Only selected_rows supports qualification fork")
        parent_keys = self._keys(parent)
        normalized = {tuple(k): v for k, v in changes.items()}
        if not set(normalized).issubset(set(parent_keys)):
            raise SourceStoreError("Fork cannot invent source frames; import new measured stream explicitly")
        with self._transaction():
            sid = self._new_stream(logical_path, "selected_rows", "delta", parent["id"])
            count, keys, digest = 0, [], hashlib.sha256()
            for key in parent_keys:
                rows = self._read_frame(parent, key)
                if key in normalized:
                    change = normalized[key]
                    if set(change) - {"remove", "update", "add", "order"}:
                        raise SourceStoreError("Unknown qualification change operation")
                    by_id = {r["selected_row_id"]: r for r in rows}
                    if len(by_id) != len(rows):
                        raise SourceStoreError("Duplicate original selected-row identity")
                    removed = change.get("remove", [])
                    if len(removed) != len(set(removed)):
                        raise SourceStoreError("Repeated removal")
                    for ident in removed:
                        if ident not in by_id:
                            raise SourceStoreError("Cannot remove absent source row")
                        del by_id[ident]
                    for ident, patch in change.get("update", {}).items():
                        if ident not in by_id or set(patch) & (GEOMETRY_KEYS | {"selected_row_id"}):
                            raise SourceStoreError("Qualification update changes geometry/identity or absent row")
                        by_id[ident] = dict(by_id[ident], **patch)
                    for added in change.get("add", []):
                        _validate(added, "selected_rows")
                        if _frame_key(added) != key or added["selected_row_id"] in by_id:
                            raise SourceStoreError("Added row has wrong clock/duplicate selected identity")
                        by_id[added["selected_row_id"]] = added
                    order = change.get("order", list(by_id))
                    if len(order) != len(by_id) or len(set(order)) != len(order) or set(order) != set(by_id):
                        raise SourceStoreError("Invalid qualification fork order")
                    rows = [by_id[i] for i in order]
                    if not rows:
                        raise SourceStoreError("Qualification fork cannot silently remove a complete native exposure")
                    if rows:
                        self._write_frame(sid, "selected_rows", key, rows, parent)
                if rows:
                    keys.append(key)
                    count += len(rows)
                    for row in rows:
                        digest.update(_json(row) + b"\n")
            self._seal(sid, keys, count, digest.hexdigest(), parent)
        return self.stream_info(logical_path)

    def stats(self):
        tables = {name: self.db.execute("SELECT count(*) FROM " + name).fetchone()[0] for name in ("streams", "frames", "source_blocks", "atoms", "blobs")}
        tables["compressed_blob_bytes"] = self.db.execute("SELECT coalesce(sum(length(payload)),0) FROM blobs").fetchone()[0]
        tables["source_points"] = self.db.execute("SELECT coalesce(sum(count),0) FROM source_blocks").fetchone()[0]
        tables["sqlite_bytes"] = self.path.stat().st_size
        tables["schema"] = FORMAT
        return tables
