"""Read existing RinkSpace source evidence without emitting observations.

Public API::

    inputs = SourceInputs(repo_root=..., data_root=...)
    for envelope in inputs.iter_sources(["s07", "s08"]):
        packet = envelope.source_payload["packet"]
        binding = envelope.source_payload["binding"]

SourceEnvelope is deliberately not an Observation. No coordinate, source ID,
qualification, entity binding, uncertainty or acceptance is created or changed.
Packet envelopes preserve the complete original packet and binding, including
segments, raw readouts and dependence groups. Dense envelopes preserve full rows.

Dense streams use the repository's checked_records and remain lazy. Their final
semantic SHA check is completed only when the iterator is exhausted. Do not
commit derived results from a partly consumed stream as verified evidence.
The SQLite database is opened read-only; it is never extracted or rewritten.
S09 component schemas have not been supplied here and explicitly need an adapter.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Iterator, Mapping


DEFAULT_ROUTES = None


class SourceInputError(ValueError):
    """A source identity, clock or declared input does not match."""


class SourceInputUnavailable(FileNotFoundError):
    """A registered source is unavailable at the explicitly supplied roots."""


class NeedsAdapterError(SourceInputError):
    """An input's real schema must be inspected before it can be consumed."""


@dataclass(frozen=True)
class SourceEnvelope:
    source_ref: Mapping[str, Any]
    shot_id: str
    native_pts: int
    source_payload: Mapping[str, Any]
    latest_qualification_refs: tuple[Mapping[str, Any], ...]


def source_job_contract(job):
    """Exposure/source identity, independent of task prose or fitted geometry."""
    fields=('source','source_clock','selected_shot_ids','sequences')
    return dict(**{key:job.get(key) for key in fields},
        roots={key:str(Path(job['roots'][key]).resolve()) for key in ('repo','data')},
        routes_sha256=job['source_routes_ref']['sha256'])


def job_reference_path(job_path, job, name):
    ref=job[name];root=Path(job_path).resolve().parent if ref.get('root','package')=='package' else Path(job['roots'][ref['root']])
    return (root/ref['path']).resolve()


def _json(raw: bytes) -> Any:
    def invalid(value: str) -> None:
        raise SourceInputError(f"Non-finite JSON value: {value}")
    return json.loads(raw.decode("utf-8-sig"), parse_constant=invalid)


class SourceInputs:
    """Source-only adapters; repo_root and data_root relocate registered paths.

    iter_sources() deduplicates the combined S02-S04 stream when several shots
    are requested. iter_packet_frames() also exposes older packet backups for
    S02-S06, but these must not be double-counted with dense rows.
    iter_qualification_stream() reads existing coverage/excluded/binding data;
    these remain evidence records, never automatic eligibility decisions.
    """

    def __init__(self, *, repo_root: str | Path, data_root: str | Path | None = None,
                 routes_path: str | Path = DEFAULT_ROUTES):
        self.repo_root = Path(repo_root).expanduser().resolve(strict=True)
        self.data_root = Path(data_root).expanduser().resolve() if data_root is not None else None
        if routes_path is None:
            raise SourceInputError("The programme job must explicitly provide source routes")
        self.routes_path = Path(routes_path).expanduser().resolve(strict=True)
        self.routes = _json(self.routes_path.read_bytes())
        if self.routes.get("schema") != "rinkspace.handoff.source_routes.v1":
            raise SourceInputError("Unsupported source routes schema")
        self.completed_streams: list[dict[str, Any]] = []
        self._reader = None
        self._store_type = None

    def resolve_location(self, location: Mapping[str, Any]) -> Path:
        root_name = location.get("root")
        root = self.repo_root if root_name == "repo" else self.data_root if root_name == "data" else None
        if root is None:
            raise SourceInputUnavailable(f"No explicit root for source location: {dict(location)!r}")
        rel = str(location.get("path", "")).replace("\\", "/")
        if not rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
            raise SourceInputError(f"Expected a path relative to {root_name}: {rel!r}")
        path = (root / rel).resolve()
        if not path.is_relative_to(root):
            raise SourceInputError(f"Source path escapes declared {root_name} root: {rel}")
        return path

    def resolve_registered(self, value: str, *, owner: Path | None = None) -> Path:
        """Relocate physical paths only. Logical SQLite stream names stay intact."""
        if not isinstance(value, str) or not value:
            raise SourceInputError("A registered source path must be a nonempty string")
        normalized = value.replace("\\", "/")
        for root_name, definition in self.routes["path_roots"].items():
            for prefix in definition["registered_prefixes"]:
                prefix = prefix.replace("\\", "/").rstrip("/")
                if normalized.casefold() == prefix.casefold():
                    return self.resolve_location({"root": root_name, "path": "."})
                if normalized.casefold().startswith(prefix.casefold() + "/"):
                    return self.resolve_location({"root": root_name, "path": normalized[len(prefix) + 1:]})
        if re.match(r"^[A-Za-z]:", normalized):
            raise SourceInputUnavailable(f"No declared root mapping for {value}")
        path = Path(normalized)
        if not path.is_absolute():
            path = ((owner.parent if owner is not None else self.repo_root) / path)
        path = path.resolve()
        if not any(root is not None and path.is_relative_to(root) for root in (self.repo_root, self.data_root)):
            raise SourceInputError(f"Source path is outside explicit roots: {value}")
        return path

    def resolve_reference(self, reference: Mapping[str, Any], *, owner: Path | None = None) -> dict[str, Any]:
        ref = deepcopy(dict(reference))
        for key in ("file", "path", "store"):
            if key in ref:
                ref[key] = str(self.resolve_registered(ref[key], owner=owner))
        # Deliberately do not relocate ref['stream']; it is an opaque DB identity.
        return ref

    def read_json_ref(self, reference: Mapping[str, Any]) -> Any:
        """Read an explicitly pinned ordinary JSON reference or route wrapper."""
        if "reference" in reference:
            ref = reference["reference"]
            location = reference.get("physical_location")
            path = self.resolve_location(location) if location else self.resolve_registered(ref.get("file", ref.get("path")))
        elif "root" in reference:
            ref = reference
            path = self.resolve_location(reference)
        else:
            ref = reference
            path = self.resolve_registered(ref.get("file", ref.get("path")))
        if not path.is_file():
            raise SourceInputUnavailable(f"Registered JSON is missing: {path}")
        expected = ref.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise SourceInputError(f"JSON source has no exact SHA-256 pin: {path}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise SourceInputError(f"Registered JSON bytes changed: {path}")
        return _json(raw)

    def _core(self):
        if self._reader is not None:
            return self._reader, self._store_type
        package_root = self.repo_root / "calibration" / "rinkfit" / "src"
        if not (package_root / "rinkfit_joint" / "source_record_io.py").is_file():
            raise SourceInputUnavailable(f"Existing source reader is missing: {package_root}")
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(package_root))
        try:
            reader = importlib.import_module("rinkfit_joint.source_record_io")
            store = importlib.import_module("rinkfit_joint.source_record_store")
            for module in (reader, store):
                if not Path(module.__file__).resolve().is_relative_to(package_root.resolve()):
                    raise SourceInputError("rinkfit_joint was already imported from a different repository")
            self._reader, self._store_type = reader.checked_records, store.SourceRecordStore
        finally:
            sys.path.remove(str(package_root))
            sys.dont_write_bytecode = previous
        return self._reader, self._store_type

    def _shot(self, shot_id: str) -> Mapping[str, Any]:
        if shot_id not in self.routes["requested_shot_ids"]:
            raise SourceInputError(f"Shot is outside the requested S02-S09 task: {shot_id}")
        return self.routes["shots"][shot_id]

    def _clock(self, shot_id: str, native_pts: Any) -> None:
        scope = self._shot(shot_id)["scope"]["processing"]
        step = self.routes["source_clock"]["native_pts_step"]
        if (type(native_pts) is not int or not scope["first_pts"] <= native_pts < scope["stop_pts_exclusive"]
                or (native_pts - scope["first_pts"]) % step):
            raise SourceInputError(f"Invalid or excluded exposure: {shot_id}/{native_pts!r}")

    def iter_sources(self, shot_ids: Iterable[str]) -> Iterator[SourceEnvelope]:
        requested = list(dict.fromkeys(shot_ids))
        for sid in requested:
            route = self._shot(sid)
            if route["source_kind"] == "needs_adapter":
                raise NeedsAdapterError(f"{sid}: {route['needs_adapter_reason']} Registered directory: {route['registered_results_directory']}")
        seen_groups = set()
        for sid in requested:
            route = self._shot(sid)
            if route["source_kind"] == "packet_batch":
                yield from self.iter_packet_frames(sid)
            elif route["source_kind"] == "sealed_selected_rows":
                gid = route["source_group_id"]
                if gid not in seen_groups:
                    seen_groups.add(gid)
                    yield from self._iter_stream(gid, "selected_rows", set(requested))
            else:
                raise NeedsAdapterError(f"Unsupported existing source kind for {sid}")

    def _stream_reference(self, group_id: str, kind: str):
        group = self.routes["source_groups"][group_id]
        if kind in group["streams"]:
            wrapped = group["streams"][kind]
            return deepcopy(wrapped["reference"]), deepcopy(wrapped)
        link = next((v for v in group.get("linked_manifest_children", []) if v["stream_kind"] == kind), None)
        if link is None and kind == "eligibility_delta":
            coverage = next((v for v in group.get("linked_manifest_children", []) if v["stream_kind"] == "coverage"), None)
            if coverage:
                link = dict(coverage, stream_kind=kind, json_pointer="/source_files/eligibility_delta.jsonl.gz")
        if link is None:
            raise NeedsAdapterError(f"No registered {kind} route in {group_id}")
        manifest = self.read_json_ref(link["manifest_ref"])
        chain = [deepcopy(link["manifest_ref"])]
        filename = kind + ".jsonl.gz"
        # dense234 v2 names its table source_files and inherits unchanged v1
        # streams. Follow only SHA-pinned parent references; retain the chain.
        while True:
            table = manifest.get("source_files", manifest.get("files", {}))
            if filename in table:
                ref = table[filename]
                break
            parent = manifest.get("parent_generation")
            if not parent:
                raise SourceInputUnavailable(f"Pinned manifest chain has no {kind}: {chain!r}")
            if parent in chain:
                raise SourceInputError("Cyclic source manifest parent chain")
            chain.append(deepcopy(parent))
            manifest = self.read_json_ref(parent)
        link = dict(link, resolved_manifest_chain=chain,
                    resolved_table="source_files" if "source_files" in manifest else "files")
        if not isinstance(ref, dict) or not isinstance(ref.get("file"), str):
            raise NeedsAdapterError("Actual linked stream reference does not use the known file schema")
        physical = self.resolve_reference(ref)
        if Path(physical["file"]).is_file():
            return deepcopy(ref), {"registered_manifest_child": deepcopy(link), "actual_child_ref": deepcopy(ref)}
        # An exact-copy file may have been retired after source.sqlite migration.
        # Ask the read-only store for the registered logical file, and bind the
        # original-byte SHA to its sealed metadata before using the semantic SHA.
        store_ref = group["streams"]["selected_rows"]["reference"]
        store_path = self.resolve_registered(store_ref["store"])
        if not store_path.is_file():
            raise SourceInputUnavailable(f"Neither linked file nor source store is present: {physical['file']}; {store_path}")
        _, Store = self._core()
        with Store(store_path, readonly=True) as store:
            info = store.stream_info(ref["file"])
        if (not info.get("sealed") or info.get("kind") != kind
                or info.get("original_sha256") != ref.get("sha256")
                or not info.get("semantic_sha256")):
            raise SourceInputError("Sealed logical stream does not match the pinned original file")
        resolved = {"store": store_ref["store"], "store_content_addressed": True,
                    "stream": ref["file"], "kind": kind,
                    "original_sha256": ref["sha256"], "stream_semantic_sha256": info["semantic_sha256"]}
        return resolved, {"registered_manifest_child": deepcopy(link), "actual_child_ref": deepcopy(ref), "sealed_store_ref": resolved}

    def _iter_stream(self, group_id: str, kind: str, selected_shots: set[str]) -> Iterator[SourceEnvelope]:
        group = self.routes["source_groups"][group_id]
        qualification_refs = tuple(deepcopy(group["latest_qualification_refs"]))
        declared, provenance = self._stream_reference(group_id, kind)
        resolved = self.resolve_reference(declared)
        input_path = Path(resolved["store"] if "store" in resolved else resolved["file"])
        if not input_path.is_file():
            raise SourceInputUnavailable(f"Registered {kind} source is missing: {input_path}")
        checked_records, _ = self._core()
        count = 0
        records = checked_records(resolved)
        try:
            for number, payload in enumerate(records):
                if not isinstance(payload, dict):
                    raise SourceInputError(f"Expected an object in {group_id}/{kind}")
                sid, pts = payload.get("shot_id"), payload.get("native_pts")
                # Some old excluded streams omit shot_id. It may be recovered
                # uniquely from the registered interval, never from an entity.
                if sid is None and type(pts) is int:
                    matches = [s for s in group["shot_ids"] if self._shot(s)["scope"]["processing"]["first_pts"] <= pts < self._shot(s)["scope"]["processing"]["stop_pts_exclusive"]]
                    if len(matches) == 1:
                        sid = matches[0]
                if sid not in group["shot_ids"]:
                    raise SourceInputError(f"Stream contains an undeclared shot: {sid!r}")
                self._clock(sid, pts)
                if "time_base" in payload and payload["time_base"] != "1/12800":
                    raise SourceInputError("Source row uses a different exposure clock")
                count += 1
                if sid in selected_shots:
                    yield SourceEnvelope(
                        source_ref={"source_kind": "sealed_stream_record", "source_group_id": group_id,
                                    "stream_kind": kind, "record_index_0based": number,
                                    "registered_reference": provenance,
                                    "verification": "stream semantic check completes at iterator exhaustion"},
                        shot_id=sid, native_pts=pts, source_payload=payload,
                        latest_qualification_refs=qualification_refs)
        finally:
            records.close()
        self.completed_streams.append({"source_group_id": group_id, "stream_kind": kind,
                                       "record_count": count, "reference": declared})

    def iter_qualification_stream(self, group_id: str, kind: str) -> Iterator[SourceEnvelope]:
        if kind not in {"coverage", "excluded", "consumer_bindings"}:
            raise SourceInputError("Qualification stream kind must be coverage, excluded or consumer_bindings")
        group = self.routes["source_groups"][group_id]
        yield from self._iter_stream(group_id, kind, set(group["shot_ids"]))

    def iter_packet_frames(self, shot_id: str) -> Iterator[SourceEnvelope]:
        route = self._shot(shot_id)
        batch = route.get("packet_batch", route.get("historical_packet_batch"))
        if batch is None:
            raise NeedsAdapterError(f"No packet batch is available for {shot_id}")
        manifest = self.read_json_ref(batch["manifest_ref"])
        entries = manifest.get("frames")
        if not isinstance(entries, list) or len(entries) != batch["registered_frame_count"]:
            raise SourceInputError("Registered packet batch count differs")
        seen = set()
        for entry in entries:
            packet_ref, binding_ref = entry[batch["packet_ref_key"]], entry[batch["binding_ref_key"]]
            packet, binding = self.read_json_ref(packet_ref), self.read_json_ref(binding_ref)
            if packet.get("schema") != batch["expected_packet_schema"] or binding.get("schema") != batch["expected_binding_schema"]:
                raise NeedsAdapterError("Actual packet/binding schema is not the inspected schema")
            pts = entry["source_pts"]
            self._clock(shot_id, pts)
            if pts in seen:
                raise SourceInputError(f"Duplicate packet exposure: {shot_id}/{pts}")
            seen.add(pts)
            source = packet.get("source", {})
            expected_shot_index = route["scope"]["shot_index"]
            if (packet.get("source_pts") != pts or source.get("native_pts") != pts
                    or binding.get("source_pts") != pts or source.get("shot_index") != expected_shot_index
                    or source.get("time_base") != "1/12800" or binding.get("source_time_base") != "1/12800"
                    or source.get("source_video_sha256") != self.routes["source"]["sha256"]):
                raise SourceInputError(f"Packet/binding/source exposure differs: {shot_id}/{pts}")
            linked = binding.get("source_packet", {})
            if (linked.get("sha256") != packet_ref.get("sha256")
                    or self.resolve_registered(linked.get("path")) != self.resolve_registered(packet_ref["path"])):
                raise SourceInputError("Binding refers to a different source packet")
            qualification_refs = deepcopy(batch.get("latest_qualification_refs", []))
            # These are exact local dispositions, not fresh approval. Only
            # point to fields actually present in this original packet.
            for field in ("unresolved_source_regions", "unresolved_board_regions",
                          "excluded_features_by_user", "unbound_static_materials",
                          "board_material_ownership_guard", "source_correlated_groups"):
                if field in packet:
                    qualification_refs.append({"role": "packet_local_" + field,
                                               "reference": deepcopy(packet_ref),
                                               "json_pointer": "/" + field})
            yield SourceEnvelope(
                source_ref={"source_kind": "packet_batch", "granularity": "native_frame",
                            "manifest_ref": deepcopy(batch["manifest_ref"]),
                            "manifest_entry": deepcopy(entry), "packet_ref": deepcopy(packet_ref),
                            "binding_ref": deepcopy(binding_ref),
                            "verification": "manifest, packet and binding exact file SHA verified; original image not decoded"},
                shot_id=shot_id, native_pts=pts,
                source_payload={"packet": packet, "binding": binding},
                latest_qualification_refs=tuple(qualification_refs))


def _pointer(value, pointer):
    if not pointer:
        return value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise SourceInputError("Expected an RFC 6901 JSON pointer")
    for part in pointer[1:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not part.isdecimal(): raise SourceInputError("Array pointer needs an integer index")
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise SourceInputError("Source pointer traverses a scalar")
    return value


class SourceSupportResolver:
    """Read source evidence, never invoke a historical producer.

    Bounded per-file cache retains only the most recently accessed support.
    Unpinned physical files can be read as candidates, but are explicitly
    distinguished from checked original-byte or sealed semantic evidence.
    """
    def __init__(self, inputs, support_index=None):
        self.inputs = inputs
        self.support_index = support_index
        self.support_db = None
        if support_index is not None:
            import sqlite3
            self.support_db = sqlite3.connect(Path(support_index).resolve().as_uri()+"?mode=ro", uri=True)
        self.cache_key = self.cache_value = None
        self.aliases = {}
        binding_path = inputs.repo_root / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/storage_20261021/bindings.json"
        if binding_path.is_file():
            binding = _json(binding_path.read_bytes())
            alias = inputs.read_json_ref(binding["native_board_frame_aliases"])
            self.aliases = {str(k).replace("\\", "/").casefold(): v for k,v in alias["aliases"].items()}

    def resolve(self, reference):
        import gzip
        import itertools
        ref = deepcopy(dict(reference))
        file_name = ref.get("file",ref.get("path"))
        if not file_name and "store" not in ref:
            raise SourceInputError("Source support has no registered location")
        if self.support_index is not None and "jsonl_line_0based" in ref and "gzip_member_offset" not in ref:
            import sqlite3
            line = ref['jsonl_line_0based']
            if type(line) is not int or line < 0:
                raise SourceInputError("Invalid source line index")
            row = self.support_db.execute("select payload,sha256 from support_rows join support_files using(file) where file=? and line=? and complete=1",
                (str(file_name).replace("\\", "/").casefold(), line)).fetchone()
            if row is None:raise SourceInputUnavailable("Requested source line is absent from the verified support index")
            if ref.get('sha256') and ref['sha256'] != row[1]:
                raise SourceInputError("Indexed source support SHA differs from the declared source")
            return dict(payload=_pointer(_json(row[0].encode()), ref.get("json_pointer", "")),
                provenance=dict(reference=ref, actual_file_sha256=row[1],
                    exact_declared_identity_verified=ref.get("sha256") == row[1],
                    indexed_source_line=True))
        key = json.dumps({k:v for k,v in ref.items() if k in ("file","path","store","stream","frame","sha256","frame_semantic_sha256","stream_semantic_sha256","gzip_member_offset","gzip_member_bytes","gzip_member_sha256")},sort_keys=True)
        if key != self.cache_key:
            provenance = dict(reference=ref,exact_declared_identity_verified=False)
            alias = self.aliases.get(str(file_name).replace("\\","/").casefold())
            if alias is not None:
                reader,_ = self.inputs._core()
                actual_ref = self.inputs.resolve_reference(alias)
                values = list(reader(actual_ref))
                if len(values)!=1: raise SourceInputError("Native-frame alias must return one complete frame")
                if ref.get("sha256") and ref["sha256"] != alias["original_file_sha256"]:
                    raise SourceInputError("Support SHA differs from the sealed original file")
                value = values[0]
                provenance.update(sealed_alias=alias,exact_declared_identity_verified=True)
            elif "store" in ref:
                reader,_ = self.inputs._core()
                value = list(reader(self.inputs.resolve_reference(ref)))
                provenance["exact_declared_identity_verified"]=True
            else:
                path = self.inputs.resolve_registered(file_name)
                if not path.is_file():
                    # An original JSONL may have been sealed and retired.
                    reader,Store = self.inputs._core()
                    stores={v["streams"]["selected_rows"]["reference"].get("store")
                            for v in self.inputs.routes["source_groups"].values()}
                    found=None
                    for name in stores:
                        if not name:continue
                        store_path=self.inputs.resolve_registered(name)
                        try:
                            with Store(store_path,readonly=True) as store: meta=store.stream_info(file_name)
                        except (KeyError,ValueError):continue
                        found=dict(store=str(store_path),store_content_addressed=True,stream=file_name,
                            stream_semantic_sha256=meta["semantic_sha256"],kind=meta["kind"])
                        break
                    if found is None:raise SourceInputUnavailable(str(path))
                    value=list(reader(found))
                    provenance.update(sealed_alias=found,exact_declared_identity_verified=True)
                else:
                    if "gzip_member_offset" in ref:
                        offset,size=ref["gzip_member_offset"],ref["gzip_member_bytes"]
                        if type(offset) is not int or type(size) is not int or offset<0 or size<=0:
                            raise SourceInputError("Invalid gzip member range")
                        with path.open("rb") as f:f.seek(offset);raw=f.read(size)
                        if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=ref["gzip_member_sha256"]:
                            raise SourceInputError("Source gzip member identity changed")
                        value=[_json(line) for line in gzip.decompress(raw).splitlines() if line.strip()]
                        provenance["exact_declared_identity_verified"]=True
                    else:
                        line_index = ref.get("jsonl_line_0based")
                        if line_index is not None:
                            # Bounded fallback. Production primes the exact requested
                            # lines once into the new derived support index.
                            with path.open("rb") as f:digest=hashlib.file_digest(f,"sha256").hexdigest()
                            expected=ref.get("sha256")
                            if expected and expected!=digest:raise SourceInputError("Source support bytes changed: "+str(path))
                            opener = gzip.open if path.suffix.lower()==".gz" else open
                            with opener(path, "rb") as f:
                                raw_line = next(itertools.islice(f, line_index, line_index+1), None)
                            if raw_line is None:raise SourceInputError("Source line is out of range")
                            return dict(payload=_pointer(_json(raw_line),ref.get("json_pointer","")),
                                provenance=dict(reference=ref,actual_file_sha256=digest,
                                    exact_declared_identity_verified=bool(expected)))
                        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
                        expected=ref.get("sha256")
                        if expected and expected!=digest:raise SourceInputError("Source support bytes changed: "+str(path))
                        provenance.update(actual_file_sha256=digest,exact_declared_identity_verified=bool(expected))
                        payload=gzip.decompress(raw) if path.suffix.lower()==".gz" else raw
                        value=_json(payload)
            self.cache_key,self.cache_value=key,(value,provenance)
        value,provenance=self.cache_value
        line=ref.get("member_jsonl_line_0based",ref.get("jsonl_line_0based"))
        if line is not None:
            if type(line) is not int or line<0:raise SourceInputError("Invalid source line index")
            value=value[line]
        frame_context=None
        if isinstance(value,dict) and value.get('schema') in (
                'rinkfit.dense_native_board_true_material_boundaries_source_frame.v1',
                'rinkfit.dense_native_board_source_frame.v1'):
            names=('schema','native_pts','shot_id','time_base','source_video_sha256','decoded_native_BGR_sha256',
                'paired_blue_cap_and_white_board_body_semantics','yellow_contact_from_current_global_band_not_accumulated_extrapolation',
                'yellow_contact_from_current_native_material_not_reflection','mask_sha256')
            frame_context={key:value[key] for key in names if key in value}
            frame_context['material_roles']=list(value.get('roles',{}))
            qualification=value.get('local_material_requalification',{})
            frame_context['local_material_requalification']={key:qualification[key] for key in
                ('actual_current_RGB_reread','original_anchored_source_UV_preserved') if key in qualification}
        value=_pointer(value,ref.get("json_pointer",""))
        provenance=deepcopy(provenance)
        provenance['reference']=ref
        result=dict(payload=value,provenance=provenance)
        if frame_context is not None:result['source_frame_context']=frame_context
        return result


def resolve_source_support(reference, *, inputs, resolver=None):
    return (resolver or SourceSupportResolver(inputs)).resolve(reference)


def iter_s09_components(inputs):
    """The inspected S09 format consists of two same-exposure source packets."""
    route=inputs._shot("s09")
    root=inputs.resolve_location(route["registered_results_directory"])
    manifest_path=root/"frame_manifest.json"
    if not manifest_path.is_file():raise SourceInputUnavailable(str(manifest_path))
    raw=manifest_path.read_bytes();manifest=_json(raw)
    if manifest["source"]["sha256"]!=inputs.routes["source"]["sha256"]:
        raise SourceInputError("S09 frame manifest belongs to another video")
    guides=tuple(dict(role="original_S09_guide_and_withdrawal",reference=ref)
                 for ref in route.get("registered_guides",[]))
    seen=set()
    for frame in manifest["frames"]:
        source=frame["source"];pts=source["native_pts"];inputs._clock("s09",pts)
        if pts in seen:raise SourceInputError("Duplicate S09 exposure")
        seen.add(pts)
        for component in ("board","ice"):
            path=root/(component+"_candidates")/(str(frame["index"]).zfill(6)+".json")
            if not path.is_file():raise SourceInputUnavailable(str(path))
            data=path.read_bytes();packet=_json(data)
            original=packet["source"]
            if (original["native_pts"]!=pts or original["source_video_sha256"]!=inputs.routes["source"]["sha256"]
                or original["time_base"]!="1/12800" or original["decoded_RGB_sha256"]!=source["decoded_RGB_sha256"]):
                raise SourceInputError("S09 component changed its original exposure")
            ref=dict(path=str(path),sha256=hashlib.sha256(data).hexdigest())
            yield SourceEnvelope(
                source_ref=dict(source_kind="s09_component_packet",component=component,packet_ref=ref,
                    manifest_ref=dict(path=str(manifest_path),sha256=hashlib.sha256(raw).hexdigest()),
                    verification="original packet source SHA/PTS/RGB identity retained; no new human approval"),
                shot_id="s09",native_pts=pts,source_payload=dict(packet=packet),
                latest_qualification_refs=guides)


def iter_source_evidence(job, shot_ids=None):
    """Isolate sealed groups; a missing group does not erase available shots.

    Availability errors are returned as explicit gap envelopes and can never
    be mistaken for a successfully verified empty source stream.
    """
    job_path=Path(job).resolve() if not isinstance(job,dict) else None
    spec=_json(job_path.read_bytes()) if job_path else job
    package=job_path.parent if job_path else Path(spec["roots"]["package"])
    inputs=SourceInputs(repo_root=spec["roots"]["repo"],data_root=spec["roots"]["data"],
                        routes_path=package/spec["source_routes_ref"]["path"])
    requested=list(shot_ids or spec["selected_shot_ids"]);seen=set()
    for sid in requested:
        route=inputs._shot(sid)
        group=route.get("source_group_id",sid)
        if group in seen:continue
        seen.add(group)
        selected=[s for s in requested if inputs._shot(s).get("source_group_id",s)==group]
        try:
            iterator=iter_s09_components(inputs) if sid=="s09" else inputs.iter_sources(selected)
            yield from iterator
        except SourceInputUnavailable as exc:
            yield SourceEnvelope(dict(source_kind="unavailable_source_group",group=group,reason=str(exc)),
                                 sid,inputs._shot(sid)["scope"]["processing"]["first_pts"],{},())



class SourceEvidenceIndex:
    """Small frame/source locators. Original selected rows stay in their store."""
    def __init__(self, job_path, index_path, *, read_only=False, reuse_from_job=None):
        import sqlite3
        self.job_path=Path(job_path).resolve()
        job_bytes=self.job_path.read_bytes();self.job=_json(job_bytes)
        self.inputs=SourceInputs(repo_root=self.job["roots"]["repo"],data_root=self.job["roots"]["data"],
            routes_path=job_reference_path(self.job_path,self.job,'source_routes_ref'))
        routes_bytes=self.inputs.routes_path.read_bytes()
        if hashlib.sha256(routes_bytes).hexdigest()!=self.job['source_routes_ref']['sha256']:
            raise SourceInputError('Source routes differ from the job reference')
        self.path=Path(index_path).resolve();self.read_only=read_only;self.reuse_provenance=None
        if read_only:
            self.db=sqlite3.connect(self.path.as_uri()+'?mode=ro',uri=True)
        else:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.db=sqlite3.connect(self.path)
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS streams(id TEXT PRIMARY KEY,reference TEXT NOT NULL,verified INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS frames(stream TEXT,kind TEXT,shot TEXT,pts INTEGER,start_row INTEGER,
                row_count INTEGER,semantic_sha TEXT,PRIMARY KEY(stream,shot,pts));
            CREATE INDEX IF NOT EXISTS frame_lookup ON frames(shot,pts);
            CREATE TABLE IF NOT EXISTS packets(id TEXT PRIMARY KEY,shot TEXT,pts INTEGER,reference TEXT NOT NULL,qualification TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS packet_lookup ON packets(shot,pts);
            CREATE TABLE IF NOT EXISTS gaps(id TEXT PRIMARY KEY,shot TEXT,reason TEXT NOT NULL);
        """)
        identity=hashlib.sha256(job_bytes+routes_bytes).hexdigest()
        old=self.db.execute("select value from meta where key='input_identity'").fetchone()
        if old and old[0]!=identity:
            try:
                if reuse_from_job is None:
                    raise SourceInputError("Index belongs to another job/routes revision")
                previous=Path(reuse_from_job).resolve();previous_bytes=previous.read_bytes();spec=_json(previous_bytes)
                prior_routes=job_reference_path(previous,spec,'source_routes_ref').read_bytes()
                if (hashlib.sha256(previous_bytes+prior_routes).hexdigest()!=old[0] or
                    hashlib.sha256(prior_routes).hexdigest()!=spec['source_routes_ref']['sha256']):
                    raise SourceInputError('Prior job does not authenticate the existing source index')
                # Task instructions, native scene and camera revisions are not source
                # records. Reuse requires the same source, clock, exact routes and scope.
                if prior_routes!=routes_bytes or source_job_contract(spec)!=source_job_contract(self.job):
                    raise SourceInputError('Source index reuse changes source routes, clock or exposure scope')
                self.reuse_provenance=dict(previous_job=str(previous),previous_job_sha256=hashlib.sha256(previous_bytes).hexdigest(),
                    authenticated_index_identity=old[0],current_job_sha256=hashlib.sha256(job_bytes).hexdigest(),
                    source_records_unchanged=True,previous_task_instructions_used=False)
            except BaseException:
                self.db.close();raise
        if old is None:
            if read_only:
                self.db.close();raise SourceInputError('Source index has no authenticated input identity')
            self.db.execute("insert into meta values('input_identity',?)",(identity,));self.db.commit()

    def build(self, progress=None):
        if self.read_only:raise SourceInputError('Cannot build a read-only source index')
        import time
        started=time.perf_counter()
        selected=set(self.job["selected_shot_ids"])
        for gid,group in self.inputs.routes["source_groups"].items():
            shots=selected.intersection(group["shot_ids"])
            if not shots:continue
            kinds=["selected_rows","excluded","coverage","consumer_bindings"]
            if gid == "s02_s04":
                kinds.append("eligibility_delta")
            for kind in kinds:
                key=gid+"/"+kind
                if self.db.execute("select 1 from streams where id=? and verified=1",(key,)).fetchone():continue
                self.db.execute("delete from frames where stream=?",(key,))
                try:
                    declared,_=self.inputs._stream_reference(gid,kind)
                    self.db.execute("insert or replace into streams values(?,?,0)",(key,json.dumps(declared),))
                    current=None;count=0;start=0;digest=None
                    total=0
                    for envelope in self.inputs._iter_stream(gid,kind,shots):
                        frame=(envelope.shot_id,envelope.native_pts)
                        if frame!=current:
                            if current is not None:
                                self.db.execute("insert into frames values(?,?,?,?,?,?,?)",(key,kind,*current,start,count,digest.hexdigest()))
                            current=frame;start=envelope.source_ref["record_index_0based"];count=0;digest=hashlib.sha256()
                        # Same canonical record contract as checked_records.
                        digest.update((json.dumps(envelope.source_payload,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode())
                        count+=1;total+=1
                        if progress and total%50000==0:progress(dict(stage="source_index",stream=key,records=total,pts=frame[1]))
                    if current is not None:
                        self.db.execute("insert into frames values(?,?,?,?,?,?,?)",(key,kind,*current,start,count,digest.hexdigest()))
                    self.db.execute("update streams set verified=1 where id=?",(key,));self.db.commit()
                    if progress:progress(dict(stage="source_stream_verified",stream=key,records=total))
                except SourceInputUnavailable as exc:
                    self.db.rollback()
                    self.db.execute("insert or replace into gaps values(?,?,?)",(key,",".join(sorted(shots)),str(exc)));self.db.commit()
                except BaseException:
                    self.db.rollback();raise
        for shot in sorted(selected.intersection({"s07","s08","s09"})):
            marker="packets/"+shot
            if self.db.execute("select 1 from meta where key=?",(marker,)).fetchone():continue
            iterator=iter_s09_components(self.inputs) if shot=="s09" else self.inputs.iter_packet_frames(shot)
            try:
                count=0
                for e in iterator:
                    ref=e.source_ref
                    key=ref["packet_ref"]["path"]
                    self.db.execute("insert or replace into packets values(?,?,?,?,?)",
                        (key,shot,e.native_pts,json.dumps(ref),json.dumps(e.latest_qualification_refs)))
                    count+=1
                self.db.execute("insert into meta values(?,?)",(marker,str(count)));self.db.commit()
                if progress:progress(dict(stage="packet_batch_verified",shot=shot,packets=count))
            except SourceInputUnavailable as exc:
                self.db.rollback();self.db.execute("insert or replace into gaps values(?,?,?)",(marker,shot,str(exc)));self.db.commit()
        result=dict(streams=self.db.execute("select id,verified from streams").fetchall(),
                    frames=self.db.execute("select kind,shot,count(*),sum(row_count) from frames group by kind,shot").fetchall(),
                    packets=self.db.execute("select shot,count(*),count(distinct pts) from packets group by shot").fetchall(),
                    unavailable=self.db.execute("select * from gaps").fetchall(),elapsed_seconds=time.perf_counter()-started,
                    original_source_rows_rewritten=False)
        return result

    def frame(self, shot, pts, *, kinds=("selected_rows",)):
        reader,Store=self.inputs._core()
        for stream,kind,start,count,expected,rawref in self.db.execute(
            "select f.stream,f.kind,f.start_row,f.row_count,f.semantic_sha,s.reference from frames f "
            "join streams s on s.id=f.stream where f.shot=? and f.pts=? and s.verified=1",(shot,pts)):
            if kind not in kinds:continue
            ref=self.inputs.resolve_reference(json.loads(rawref));digest=hashlib.sha256();n=0
            if "store" in ref:
                with Store(ref["store"],readonly=True) as store:
                    payloads=list(store.iterate_records(ref["stream"],frame=(shot,pts)))
            else:
                # Legacy sequential stream is used only when no sealed store exists.
                payloads=[row for row in reader(ref) if row.get("native_pts")==pts and row.get("shot_id",shot)==shot]
            for n,payload in enumerate(payloads,1):
                digest.update((json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode())
            if n!=count or digest.hexdigest()!=expected:raise SourceInputError("Indexed source frame content changed")
            for i,payload in enumerate(payloads):
                yield SourceEnvelope(dict(source_kind="indexed_sealed_record",stream_id=stream,stream_kind=kind,
                    record_index_0based=start+i,frame_semantic_sha256=expected,reference=ref),shot,pts,payload,tuple(deepcopy(self.inputs.routes["source_groups"][stream.split("/")[0]]["latest_qualification_refs"])))
        if "selected_rows" not in kinds:return
        for rawref,rawqual in self.db.execute("select reference,qualification from packets where shot=? and pts=?",(shot,pts)):
            ref=json.loads(rawref);packet=self.inputs.read_json_ref(ref["packet_ref"]);payload=dict(packet=packet)
            if "binding_ref" in ref:payload["binding"]=self.inputs.read_json_ref(ref["binding_ref"])
            yield SourceEnvelope(ref,shot,pts,payload,tuple(json.loads(rawqual)))

    def close(self):
        self.db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--routes", default=str(DEFAULT_ROUTES))
    parser.add_argument("--shots", nargs="+", required=True)
    args = parser.parse_args()
    try:
        source = SourceInputs(repo_root=args.repo_root, data_root=args.data_root, routes_path=args.routes)
        counts = Counter()
        for envelope in source.iter_sources(args.shots):
            counts[envelope.shot_id] += 1
        print(json.dumps({"source_envelopes_by_shot": dict(counts),
                          "completed_sealed_streams": len(source.completed_streams),
                          "formal_observations_emitted": 0, "source_quality_accepted": False}, ensure_ascii=False))
        return 0
    except (SourceInputError, SourceInputUnavailable) as exc:
        print(json.dumps({"status": "blocked_input", "reason": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


def build_support_index(index, path, progress=None):
    """Stream only referenced JSONL rows into a derived, bounded-memory index."""
    import sqlite3, gzip
    db=sqlite3.connect(path)
    db.executescript("""
      CREATE TABLE IF NOT EXISTS wanted(file TEXT,line INTEGER,PRIMARY KEY(file,line));
      CREATE TABLE IF NOT EXISTS support_rows(file TEXT,line INTEGER,payload TEXT,PRIMARY KEY(file,line));
      CREATE TABLE IF NOT EXISTS support_files(file TEXT PRIMARY KEY,path TEXT,sha256 TEXT,complete INTEGER);
    """)
    for shot,pts in index.db.execute("select distinct shot,pts from frames where kind='selected_rows' order by shot,pts"):
        for e in index.frame(shot,pts):
            ref=e.source_payload.get('source_pixel_or_annotation_ref',{})
            if 'jsonl_line_0based' not in ref or 'gzip_member_offset' in ref:continue
            name=ref['file'];key=name.replace('\\','/').casefold()
            db.execute('insert or ignore into wanted values(?,?)',(key,ref['jsonl_line_0based']))
            db.execute('insert or ignore into support_files values(?,?,NULL,0)',(key,str(index.inputs.resolve_registered(name))))
        db.commit()
    for key,name in db.execute('select file,path from support_files where complete=0').fetchall():
        wanted={r[0] for r in db.execute('select line from wanted where file=?',(key,))}
        source=Path(name)
        if not source.is_file():raise SourceInputUnavailable(name)
        with source.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        opener=gzip.open if source.suffix.lower()=='.gz' else open
        found=0
        with opener(source,'rb') as f:
            for line,raw in enumerate(f):
                if line not in wanted:continue
                value=_json(raw)
                db.execute('insert or replace into support_rows values(?,?,?)',(key,line,json.dumps(value,ensure_ascii=False,separators=(',',':'))))
                found+=1
                if found==len(wanted):break
        if found!=len(wanted):db.rollback();raise SourceInputError('Source support index has out-of-range lines: '+name)
        db.execute('update support_files set sha256=?,complete=1 where file=?',(sha,key));db.commit()
        if progress:progress(dict(stage='source_support_verified',path=name,requested_rows=found,sha256=sha))
    result=dict(files=db.execute('select count(*) from support_files where complete=1').fetchone()[0],
                referenced_rows=db.execute('select count(*) from support_rows').fetchone()[0])
    db.close();return result
