"""Read the *original* support behind a selected source row, without qualifying it.

``source_pixel_or_annotation_ref`` is an address, not evidence that its pixels
were read correctly. This module authenticates and dereferences that address. It
does not alter UV, native PTS, acceptance, correlation groups or solver weights.
The original exposure/provenance context is retained alongside the selected
value; unrelated rows are not copied into every resolved point.

Supported addresses are pinned JSON + JSON Pointer, pinned JSONL physical line,
one SHA-pinned gzip member + member line, and an exact sealed native-frame ref.
Unpinned historical leaves need ``file_pins`` from an authenticated manifest;
hashing whatever happens to exist today does not create a trusted source pin.

NativeFrameAlias is an explicit registration API, not an assumed format for the
private native_board_aliases.json. Its real manifest must be inspected and
authenticated by its adapter before constructing these entries. No database is
unpacked, rewritten, or scanned to guess an old file's replacement.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping
import zlib


class MissingSupport(FileNotFoundError):
    """A registered file, pin, pointer, line, or exposure is unavailable."""


class InvalidSupport(ValueError):
    """Original source bytes, address or exposure identity did not match."""


class UnsupportedSupport(InvalidSupport):
    """The real source format needs an explicit adapter; do not guess it."""


@dataclass(frozen=True)
class NativeFrameAlias:
    source_file: str
    source_sha256: str
    frame_reference: Mapping[str, Any]


@dataclass(frozen=True)
class ResolvedSourceSupport:
    reference: Mapping[str, Any]
    value: Any
    source_context: Any
    authentication: Mapping[str, Any]


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise InvalidSupport(f"{name} must be an explicit SHA-256 digest")
    return value.lower()


def _json(raw: bytes) -> Any:
    def bad(value):
        raise InvalidSupport(f"Non-finite source JSON value: {value}")
    try:
        return json.loads(raw.decode("utf-8-sig"), parse_constant=bad)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidSupport(f"Original support is not valid JSON: {exc}") from exc


def _index(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise InvalidSupport(f"{name} must be a nonnegative integer")
    return value


def _pointer(document: Any, pointer: str) -> tuple[Any, list[Mapping[str, Any]]]:
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise InvalidSupport("json_pointer must be an RFC 6901 pointer")
    value, ancestors = document, []
    if not pointer:
        return value, [value] if isinstance(value, Mapping) else []
    for token in pointer[1:].split("/"):
        if re.search(r"~(?![01])", token):
            raise InvalidSupport(f"Invalid JSON Pointer escape: {pointer}")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, Mapping):
            ancestors.append(value)
            if token not in value:
                raise MissingSupport(f"Original support has no JSON Pointer: {pointer}")
            value = value[token]
        elif isinstance(value, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise InvalidSupport(f"Invalid array index in JSON Pointer: {pointer}")
            index = int(token)
            if index >= len(value):
                raise MissingSupport(f"Original support has no JSON Pointer: {pointer}")
            value = value[index]
        else:
            raise MissingSupport(f"Original support has no JSON Pointer: {pointer}")
    if isinstance(value, Mapping):
        ancestors.append(value)
    return value, ancestors


def _signature(path: Path) -> tuple[int, ...]:
    try:
        s = path.stat()
        if not path.is_file():
            raise MissingSupport(f"Original support is not a file: {path}")
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    except FileNotFoundError as exc:
        raise MissingSupport(f"Original support file is missing: {path}") from exc


class SourceSupportResolver:
    """Read only; hashes and address caches are local to this resolver instance.

    File identity caches include inode, size, mtime and ctime. A changed file is
    authenticated again. Only the latest JSON document/member is retained; JSONL
    offsets are advanced lazily, so adjacent rows do not rescan a large file.
    ``expected_binding`` can require shot_id, native_pts/source_pts, time_base,
    source_sha256/source_video_sha256 and decoded_native_rgb_sha256. Identity is
    compared without modifying legacy numeric shot IDs or any saved source field.
    """

    def __init__(self, *, repo_root: str | Path, data_root: str | Path | None = None,
                 file_pins: Iterable[Mapping[str, Any]] | Mapping[str, str] = (),
                 native_frame_aliases: Iterable[NativeFrameAlias] = ()):
        self.repo_root = Path(repo_root).expanduser().resolve(strict=True)
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else None
        self._pins: dict[str, str] = {}
        self._aliases: dict[str, NativeFrameAlias] = {}
        self._verified: dict[str, tuple[tuple[int, ...], str]] = {}
        self._last_document: tuple[Any, Any] | None = None
        self._line_offsets: dict[str, tuple[tuple[int, ...], list[int]]] = {}
        self._alias_proofs: set[tuple[Any, ...]] = set()
        self._core_modules = None
        refs = ({"file": k, "sha256": v} for k, v in file_pins.items()) if isinstance(file_pins, Mapping) else file_pins
        for ref in refs:
            self.register_file_pin(ref)
        for alias in native_frame_aliases:
            self.register_native_frame_alias(alias)

    def resolve_registered(self, value: str, *, owner: Path | None = None) -> Path:
        """Relocate physical paths only; opaque SQLite stream names stay intact."""
        if not isinstance(value, str) or not value:
            raise InvalidSupport("Original support needs a nonempty physical path")
        normalized = value.replace("\\", "/")
        roots = (("D:/OneDrive/codex_projs/RinkSpace", self.repo_root),
                 ("D:/CodexData/RinkSpace_Data", self.data_root))
        for prefix, root in roots:
            if normalized.casefold() == prefix.casefold() or normalized.casefold().startswith(prefix.casefold() + "/"):
                if root is None:
                    raise MissingSupport(f"Supply data_root for registered source: {value}")
                path = (root / normalized[len(prefix):].lstrip("/")).resolve()
                if not path.is_relative_to(root):
                    raise InvalidSupport(f"Source path escapes its explicit root: {value}")
                return path
        if re.match(r"^[A-Za-z]:", normalized) and not Path(normalized).is_absolute():
            raise MissingSupport(f"No explicit root mapping for source: {value}")
        path = Path(normalized)
        if not path.is_absolute():
            path = (Path(owner).parent if owner is not None else self.repo_root) / path
        path = path.resolve()
        if not any(root is not None and path.is_relative_to(root) for root in (self.repo_root, self.data_root)):
            raise InvalidSupport(f"Source path is outside explicit roots: {value}")
        return path

    def register_file_pin(self, reference: Mapping[str, Any], *, owner: Path | None = None) -> None:
        """Register an original manifest pin; never compute a new trusted pin."""
        path = str(self.resolve_registered(reference.get("file", reference.get("path")), owner=owner))
        digest = _sha(reference.get("sha256"), "Registered source sha256")
        if path in self._pins and self._pins[path] != digest:
            raise InvalidSupport(f"Conflicting registered source pins: {path}")
        self._pins[path] = digest

    def register_native_frame_alias(self, alias: NativeFrameAlias) -> None:
        if not isinstance(alias, NativeFrameAlias):
            raise UnsupportedSupport("Unknown native alias manifest schema; adapt authenticated entries to NativeFrameAlias explicitly")
        path = str(self.resolve_registered(alias.source_file))
        digest = _sha(alias.source_sha256, "Original aliased file sha256")
        ref = deepcopy(dict(alias.frame_reference))
        if "store" not in ref or ref.get("kind") != "native_frames" or "frame" not in ref:
            raise InvalidSupport("Native alias must name an exact sealed native_frames exposure")
        _sha(ref.get("frame_semantic_sha256"), "Aliased frame_semantic_sha256")
        _sha(ref.get("stream_semantic_sha256"), "Aliased stream_semantic_sha256")
        entry = NativeFrameAlias(alias.source_file, digest, ref)
        if path in self._aliases and self._aliases[path] != entry:
            raise InvalidSupport(f"Conflicting native frame aliases: {path}")
        self.register_file_pin({"file": alias.source_file, "sha256": digest})
        self._aliases[path] = entry

    def _core(self):
        if self._core_modules is None:
            root = self.repo_root / "calibration/rinkfit/src"
            if not (root / "rinkfit_joint/source_record_io.py").is_file():
                raise MissingSupport(f"Existing checked source reader is missing: {root}")
            previous = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            sys.path.insert(0, str(root))
            try:
                modules = (importlib.import_module("rinkfit_joint.source_record_io"),
                           importlib.import_module("rinkfit_joint.source_record_store"))
                if any(not Path(m.__file__).resolve().is_relative_to(root.resolve()) for m in modules):
                    raise InvalidSupport("Source reader was imported from another repository")
                self._core_modules = modules
            finally:
                sys.path.remove(str(root))
                sys.dont_write_bytecode = previous
        return self._core_modules

    def _pin(self, path: Path, ref: Mapping[str, Any]) -> str:
        declared, registered = ref.get("sha256"), self._pins.get(str(path))
        if declared is None and registered is None:
            raise MissingSupport(f"Original source SHA is missing: {path}; register file_pins from its authenticated manifest")
        digest = _sha(declared if declared is not None else registered, "Original source sha256")
        if registered is not None and digest != registered:
            raise InvalidSupport(f"Leaf SHA conflicts with registered original source: {path}")
        return digest

    def _authenticate_file(self, path: Path, digest: str) -> tuple[int, ...]:
        signature = _signature(path)
        if self._verified.get(str(path)) != (signature, digest):
            core, _ = self._core()
            core.pin_reference({"file": str(path), "sha256": digest})
            if _signature(path) != signature:
                raise InvalidSupport(f"Original support changed while being authenticated: {path}")
            self._verified[str(path)] = signature, digest
        return signature

    def _json_document(self, path: Path, ref: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        digest = self._pin(path, ref)
        signature = self._authenticate_file(path, digest)
        key = ("json", str(path), signature, digest)
        # On Windows two same-size writes can expose identical stat metadata.
        # A JSON support reference is authenticated by bytes, not that metadata.
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest or _signature(path) != signature:
            raise InvalidSupport(f"Original source bytes changed while being read: {path}")
        if self._last_document is None or self._last_document[0] != key:
            self._last_document = key, _json(raw)
        return self._last_document[1], {"kind": "file_sha256", "file": str(path), "sha256": digest}

    def _jsonl_document(self, path: Path, ref: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        line = _index(ref["jsonl_line_0based"], "jsonl_line_0based")
        digest = self._pin(path, ref)
        signature = self._authenticate_file(path, digest)
        old = self._line_offsets.get(str(path))
        offsets = old[1] if old is not None and old[0] == signature else [0]
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rb") as handle:
            # gzip seek offsets are uncompressed. For repeatedly accessed large
            # compressed streams, use their original indexed member references.
            start = min(line, len(offsets) - 1)
            handle.seek(offsets[start])
            for index in range(start, line + 1):
                raw = handle.readline()
                if not raw:
                    raise MissingSupport(f"Original JSONL has no physical line {line}: {path}")
                if index + 1 == len(offsets):
                    offsets.append(handle.tell())
        if _signature(path) != signature:
            raise InvalidSupport(f"Original support changed during JSONL read: {path}")
        self._line_offsets[str(path)] = signature, offsets
        if not raw.strip():
            raise MissingSupport(f"Original JSONL physical line {line} is blank: {path}")
        return _json(raw), {"kind": "file_sha256", "file": str(path), "sha256": digest, "jsonl_line_0based": line}

    def _member_document(self, path: Path, ref: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        offset = _index(ref["gzip_member_offset"], "gzip_member_offset")
        length = _index(ref["gzip_member_bytes"], "gzip_member_bytes")
        line = _index(ref["member_jsonl_line_0based"], "member_jsonl_line_0based")
        digest = _sha(ref["gzip_member_sha256"], "gzip_member_sha256")
        signature = _signature(path)
        if "sha256" in ref or str(path) in self._pins:
            signature = self._authenticate_file(path, self._pin(path, ref))
        if length == 0 or offset + length > signature[2]:
            raise MissingSupport(f"Original gzip member byte interval is unavailable: {path}")
        key = ("member", str(path), signature, offset, length, digest)
        if self._last_document is None or self._last_document[0] != key:
            with path.open("rb") as handle:
                handle.seek(offset)
                raw = handle.read(length)
            if len(raw) != length or hashlib.sha256(raw).hexdigest() != digest or _signature(path) != signature:
                raise InvalidSupport(f"Original gzip member bytes changed: {path} at {offset}")
            try:
                decoder = zlib.decompressobj(31)
                decoded = decoder.decompress(raw) + decoder.flush()
            except zlib.error as exc:
                raise InvalidSupport(f"Original gzip member is corrupt: {path}") from exc
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise InvalidSupport("gzip member reference must cover exactly one complete member")
            # dense234/assemble.py and dense56/assemble_source_generation.py
            # enumerate parsed nonblank member records. Their field name says
            # "line", but unlike jsonl_line_0based it excludes blank lines.
            self._last_document = key, [line for line in decoded.splitlines() if line.strip()]
        lines = self._last_document[1]
        if line >= len(lines) or not lines[line].strip():
            raise MissingSupport(f"Original gzip member has no nonblank record {line}: {path}")
        return _json(lines[line]), {"kind": "gzip_member_sha256", "file": str(path), "gzip_member_offset": offset,
            "gzip_member_bytes": length, "gzip_member_sha256": digest, "member_jsonl_line_0based": line}

    def _store_document(self, ref: Mapping[str, Any], owner: Path | None) -> tuple[Any, Mapping[str, Any]]:
        if "frame" not in ref or ref.get("kind") != "native_frames":
            raise UnsupportedSupport("Source support requires an exact native_frames reference; whole-store or selected-row scans are not support lookup")
        resolved = deepcopy(dict(ref))
        path = self.resolve_registered(ref["store"], owner=owner)
        signature = self._store_signature(path)
        resolved["store"] = str(path)
        resolved.pop("json_pointer", None)
        _sha(ref.get("frame_semantic_sha256"), "frame_semantic_sha256")
        _sha(ref.get("stream_semantic_sha256"), "stream_semantic_sha256")
        key = ("store", str(path), signature, json.dumps(resolved, sort_keys=True))
        if self._last_document is None or self._last_document[0] != key:
            core, _ = self._core()
            # checked_records verifies its semantic hash after the final yield.
            # Exhaust the exact single-frame iterator before returning any value.
            records = list(core.checked_records(resolved))
            if len(records) != 1:
                raise InvalidSupport("Native source frame must resolve to exactly one complete record")
            if self._store_signature(path) != signature:
                raise InvalidSupport("Source store changed during exact frame lookup; retry against a stable source")
            self._last_document = key, records[0]
        return self._last_document[1], {"kind": "sealed_native_frame", "reference": resolved}

    @staticmethod
    def _store_signature(path: Path) -> tuple[Any, ...]:
        wal = Path(str(path) + "-wal")
        return _signature(path), _signature(wal) if wal.exists() else None

    def _aliased_document(self, path: Path, ref: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        alias = self._aliases.get(str(path))
        if alias is None:
            raise MissingSupport(f"Original support is missing and has no authenticated native frame alias: {path}")
        if self._pin(path, ref) != alias.source_sha256:
            raise InvalidSupport(f"Original source SHA differs from native frame alias: {path}")
        document, authentication = self._store_document(alias.frame_reference, None)
        _, store_module = self._core()
        store_path = self.resolve_registered(alias.frame_reference["store"])
        signature = self._store_signature(store_path)
        proof = (str(path), alias.source_sha256, str(store_path), alias.frame_reference["stream"], signature)
        if proof not in self._alias_proofs:
            with store_module.SourceRecordStore(store_path, readonly=True) as store:
                originals = store.native_file_refs(alias.frame_reference["stream"])
                # import_frame_jsons appends one file receipt for each frame in
                # its sealed order. Read that existing metadata, not every frame.
                # Membership alone would let a wrong alias select another frame
                # in the same stream while still naming a real original file.
                keys = store._keys(store.stream_info(alias.frame_reference["stream"]))
            matches = [(i, r) for i, r in enumerate(originals) if self.resolve_registered(r["file"]) == path]
            if len(matches) != 1 or matches[0][1].get("sha256") != alias.source_sha256:
                raise InvalidSupport("Sealed native stream does not preserve the aliased original file and SHA")
            if len(originals) != len(keys) or tuple(keys[matches[0][0]]) != tuple(alias.frame_reference["frame"]):
                raise InvalidSupport("Native alias maps the original file to a different sealed exposure")
            if self._store_signature(store_path) != signature:
                raise InvalidSupport("Source store changed during native alias lookup")
            self._alias_proofs.add(proof)
        return document, {**authentication, "original_file": alias.source_file,
                           "original_file_sha256": alias.source_sha256, "explicit_native_alias_used": True}

    def resolve(self, reference: Mapping[str, Any], *, expected_binding: Mapping[str, Any] | None = None,
                owner: Path | None = None) -> ResolvedSourceSupport:
        """Authenticate the full addressed support, then select its original value.

        Failure raises a typed blocker. Success authenticates saved content only;
        it never claims visibility, measurement correctness or solver eligibility.
        """
        if not isinstance(reference, Mapping):
            raise UnsupportedSupport("source_pixel_or_annotation_ref must be an explicit reference object")
        ref = deepcopy(dict(reference))
        member_keys = {"gzip_member_offset", "gzip_member_bytes", "gzip_member_sha256", "member_jsonl_line_0based"}
        member = bool(member_keys.intersection(ref))
        if member and not member_keys.issubset(ref):
            raise InvalidSupport("Incomplete original gzip member reference")
        if (member and "jsonl_line_0based" in ref) or ("store" in ref and (member or "jsonl_line_0based" in ref)):
            raise InvalidSupport("Conflicting original support selectors")
        try:
            if "store" in ref:
                if "file" in ref or "path" in ref:
                    raise InvalidSupport("Store support cannot also name a physical source file")
                document, authentication = self._store_document(ref, owner)
            else:
                if "file" not in ref and "path" not in ref:
                    raise UnsupportedSupport("Unknown source support format: explicit file/path or exact sealed frame is required")
                path = self.resolve_registered(ref.get("file", ref.get("path")), owner=owner)
                if "file" in ref and "path" in ref and self.resolve_registered(ref["path"], owner=owner) != path:
                    raise InvalidSupport("Source file and path addresses disagree")
                if not path.exists() and not member and "jsonl_line_0based" not in ref:
                    document, authentication = self._aliased_document(path, ref)
                elif member:
                    document, authentication = self._member_document(path, ref)
                elif "jsonl_line_0based" in ref:
                    document, authentication = self._jsonl_document(path, ref)
                else:
                    document, authentication = self._json_document(path, ref)
            value, ancestors = _pointer(document, ref.get("json_pointer", ""))
            if expected_binding is not None:
                _check_binding(ancestors, expected_binding)
        except (MissingSupport, InvalidSupport):
            raise
        except FileNotFoundError as exc:
            raise MissingSupport(f"Original source support is unavailable: {exc}") from exc
        except KeyError as exc:
            raise MissingSupport(f"Original source record or metadata is absent: {exc}") from exc
        except ValueError as exc:
            raise InvalidSupport(f"Original source authentication failed: {exc}") from exc
        except (EOFError, zlib.error) as exc:
            raise InvalidSupport(f"Original compressed support is corrupt: {exc}") from exc
        # Deep copies keep caller edits from changing a later cached resolution.
        # A native frame can contain thousands of rows. Copying that whole frame
        # per selected point would turn a streaming lookup into quadratic work.
        # Retain exact root metadata and the selected value, without neighboring
        # rows/features. Call resolve(ref without json_pointer) for the full frame.
        context = ({k: deepcopy(v) for k, v in document.items()
                    if k not in {"features", "rows", "points"}}
                   if isinstance(document, Mapping) else None)
        return ResolvedSourceSupport(ref, deepcopy(value), context, deepcopy(authentication))


def _check_binding(ancestors: list[Mapping[str, Any]], expected: Mapping[str, Any]) -> None:
    names = {"native_pts": ("native_pts", "source_pts", "actual_pts"), "source_pts": ("native_pts", "source_pts", "actual_pts"),
             "shot_id": ("shot_id",), "time_base": ("time_base", "source_time_base"),
             "source_sha256": ("source_sha256", "source_video_sha256"),
             "source_video_sha256": ("source_sha256", "source_video_sha256"),
             "decoded_native_rgb_sha256": ("decoded_native_rgb_sha256", "decoded_native_RGB_sha256", "decoded_RGB_sha256")}
    scopes = list(ancestors)
    for ancestor in ancestors:
        scopes.extend(ancestor[k] for k in ("source", "source_binding", "source_clock") if isinstance(ancestor.get(k), Mapping))

    def normalized(name, value):
        if name in ("native_pts", "source_pts"):
            return _index(value, name)
        if name == "shot_id":
            if type(value) is int and value > 0:
                return value
            if isinstance(value, str) and re.fullmatch(r"s[0-9]+", value) and int(value[1:]) > 0:
                return int(value[1:])
            raise InvalidSupport("Source shot_id is neither a native shot ID nor a legacy numeric shot ID")
        if name == "time_base":
            try:
                time_base = Fraction(*value) if isinstance(value, (list, tuple)) else Fraction(value)
                if time_base <= 0:
                    raise ValueError()
                return time_base
            except (ValueError, TypeError, ZeroDivisionError) as exc:
                raise InvalidSupport("Source time_base is invalid") from exc
        return _sha(value, name)

    for name, wanted in expected.items():
        if name not in names:
            raise UnsupportedSupport(f"Unknown expected source identity field: {name}")
        values = [scope[k] for scope in scopes for k in names[name] if k in scope]
        if name in ("source_sha256", "source_video_sha256"):
            values.extend(ancestor["source"]["sha256"] for ancestor in ancestors
                          if isinstance(ancestor.get("source"), Mapping) and "sha256" in ancestor["source"])
        if not values:
            raise MissingSupport(f"Original support has no {name} to verify against the selected row")
        target = normalized(name, wanted)
        if any(normalized(name, value) != target for value in values):
            raise InvalidSupport(f"Original support {name} differs from the requested exposure")
