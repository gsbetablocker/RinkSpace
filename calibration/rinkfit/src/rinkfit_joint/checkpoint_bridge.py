"""Non-destructive import of the latest complete s01 repair checkpoint.

No 'latest' directory is guessed.  The caller selects an immutable accepted
checkpoint.  Physics and all row nuisance variables travel together; historical
optimiser caches, duals and certificates are retained only as provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from .camera_graph import CameraGraph, GraphError, with_s01_event

REQUIRED_ROWS = {"station", "branch", "row", "alpha"}
REQUIRED_DEPENDENCIES = {"observations", "resolved_material", "resolved_variant",
                         "semantic_manifest", "objective_manifest", "source_clock_manifest"}
SEMANTIC_RUNTIME_KEYS = {"arrays", "assets", "original_material_index",
    "resolved_material_index", "resolved_variant_index", "confirmed_bindings",
    "source_exclusion", "active_source_mask", "active_statistics", "source_order_overlay",
    "source_generator_recovery", "source_component_recovery", "source_audit_revision"}
SEMANTIC_PROVENANCE_KEYS = {"source_snapshot", "source_arrays", "binding",
    "parent_semantic_manifest", "source_exclusion_parent_manifest",
    "source_order_parent_manifest", "source_generator_parent_manifest",
    "source_component_parent_manifest", "source_audit_parent_manifest"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(value: str) -> Path:
    p = PurePosixPath(value.replace("\\", "/"))
    if not p.parts or p.is_absolute() or any(x in ("..", "") for x in p.parts) or ":" in value:
        raise GraphError(f"unsafe portable relative path: {value}")
    return Path(*p.parts)


def _file_refs(value: Any, pointer: str = ""):
    if isinstance(value, dict):
        if "sha256" in value and ("file" in value or "path" in value):
            yield pointer, value
        else:
            for key, child in value.items():
                yield from _file_refs(child, pointer + "/" + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _file_refs(child, pointer + "/" + str(index))


def _jsonl_file_refs(path: Path):
    """Read immutable source/statistics record refs, including source-row packets.

    A JSONL asset is not an opaque blob: source_rows points at the actual
    per-frame source records that establish measurement definitions. Keep those
    records inside the verified source closure as well as the numeric arrays.
    """
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            for pointer, ref in _file_refs(json.loads(line), "/" + str(index)):
                yield pointer, ref


def _semantic_refs(semantic: Mapping[str, Any]):
    if semantic.get("schema") != "rinkfit.semantic_packet.v1":
        raise GraphError("unsupported semantic replay schema")
    for key, value in semantic.items():
        refs = list(_file_refs(value, "/" + key))
        if refs and key not in SEMANTIC_RUNTIME_KEYS | SEMANTIC_PROVENANCE_KEYS:
            raise GraphError(f"unclassified semantic file-reference key: {key}; extend closure policy explicitly")
        if key in SEMANTIC_RUNTIME_KEYS:
            yield from refs


def _verify_numeric_closure(semantic: Mapping[str, Any], objective: Mapping[str, Any],
                            files_by_sha: Mapping[str, Path], n: int) -> dict[str, Any]:
    """Verify actual W/E storage and coverage, not just six manifest role names."""
    if objective.get("schema") != "rinkfit.s01_objective_export.v1":
        raise GraphError("objective export must use the explicitly supported full-W closure schema")
    def resolve(ref, *, shape=None):
        digest = ref["sha256"]
        if digest not in files_by_sha:
            raise GraphError(f"runtime numeric/semantic reference is not in closure: {ref.get('file', ref.get('path'))}")
        path = files_by_sha[digest]
        if "dtype" in ref and "shape" in ref:
            expected = int(np.prod(ref["shape"])) * np.dtype(ref["dtype"]).itemsize
            if path.stat().st_size != expected:
                raise GraphError("referenced runtime array byte size differs from semantic manifest")
        if shape is not None and tuple(ref.get("shape", ())) != tuple(shape):
            raise GraphError("runtime array source row shape mismatch")
        return path
    references = list(_semantic_refs(semantic))
    for _, ref in references:
        resolve(ref)
    nested_runtime_count = 0
    if "source_order_overlay" in semantic:
        order = _read(resolve(semantic["source_order_overlay"]))
        if order.get("schema") != "rinkfit.source_order_evidence_overlay.v1":
            raise GraphError("unsupported source-order overlay; add its runtime closure adapter")
        for key in ("audit_source_runs", "classifications", "active_hard_source_runs"):
            resolve(order[key]); nested_runtime_count += 1
    for _, ref in _file_refs(objective):
        resolve(ref)
    required_arrays = {"kind", "variant_index", "query_uv", "profile_normal", "alpha_bounds",
                       "source_sigma", "weights", "group_rows", "ordinary_rows"}
    if not required_arrays.issubset(semantic.get("arrays", {})):
        raise GraphError("semantic packet lacks a required forward/profile/W array")
    arrays = semantic["arrays"]
    def read_array(key):
        ref = arrays[key]
        return np.fromfile(resolve(ref), dtype=ref["dtype"]).reshape(ref["shape"])
    kind = read_array("kind"); sigma = read_array("source_sigma")
    if kind.shape != (n,) or sigma.shape != (n,) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise GraphError("invalid original semantic kinds/independent scales")
    if not set(np.unique(kind).tolist()).issubset({0, 1, 2}):
        raise GraphError("unsupported observation semantics")
    for key in ("query_uv", "profile_normal", "alpha_bounds"):
        a = read_array(key)
        if a.shape != (n, 2) or not np.isfinite(a[kind == 2]).all():
            raise GraphError(f"invalid full source profile array: {key}")
    budget_ref = objective["engineering_budget"]
    budget = np.fromfile(resolve(budget_ref, shape=(n,)), dtype=budget_ref["dtype"])
    if not np.isfinite(budget).all() or np.any(budget <= 0):
        raise GraphError("invalid engineering error budget")
    weights = read_array("weights").ravel(); members = read_array("group_rows").astype(np.int64)
    independent = read_array("ordinary_rows").astype(np.int64)
    if not np.isfinite(weights).all():
        raise GraphError("nonfinite original full correlated W")
    if np.any(members < 0) or np.any(members >= n) or np.any(independent < 0) or np.any(independent >= n):
        raise GraphError("statistical group source row outside original ledger")
    counts = np.bincount(np.r_[members, independent], minlength=n)
    if len(counts) != n or not np.all(counts == 1):
        raise GraphError("full W groups plus independent rows do not count each original source exactly once")
    group_ref = semantic["assets"]["groups.jsonl"]
    groups = {}; rank_sum = 0; member_segments = np.zeros(len(members), np.int32)
    weight_segments = np.zeros(len(weights), np.int32)
    with resolve(group_ref).open(encoding="utf-8") as stream:
        for line in stream:
            group = json.loads(line); gid = group["group_id"]
            start, count = int(group["member_start"]), int(group["member_count"])
            wstart, rank = int(group["weight_start"]), int(group["rank"])
            size = rank * 2 * count
            if gid in groups or start < 0 or count <= 0 or start+count > len(members) or wstart < 0 or rank <= 0 or wstart+size > len(weights):
                raise GraphError("invalid original group range or repeated group ID")
            if list(group["matrix_shape"]) != [rank, 2*count]:
                raise GraphError("original grouped W matrix shape mismatch")
            digest = hashlib.sha256(weights[wstart:wstart+size].astype('<f8', copy=False).tobytes()).hexdigest()
            if digest != group["matrix_sha256"]:
                raise GraphError("original group W slice does not match its recorded matrix SHA")
            member_segments[start:start+count] += 1; weight_segments[wstart:wstart+size] += 1
            rank_sum += rank; groups[gid] = group
    if not np.all(member_segments == 1) or not np.all(weight_segments == 1):
        raise GraphError("original group storage has gaps/overlaps")
    active = np.ones(n, bool); replacement_count = 0
    if "active_source_mask" in semantic:
        mask_ref = semantic["active_source_mask"]
        mask = np.fromfile(resolve(mask_ref, shape=(n,)), dtype=mask_ref["dtype"])
        if not np.isin(mask, [0, 1]).all():
            raise GraphError("active-source mask is not binary")
        active = mask.astype(bool)
    affected_groups = set()
    for gid, group in groups.items():
        rows = members[group['member_start']:group['member_start']+group['member_count']]
        if not active[rows].all():
            affected_groups.add(gid)
    if affected_groups and "active_statistics" not in semantic:
        raise GraphError("excluded correlated source row has no exact marginal-W definition")
    if "active_statistics" in semantic:
        statistics = _read(resolve(semantic["active_statistics"]))
        for _, ref in _file_refs(statistics):
            resolve(ref)
        if statistics.get("original_weights_array_sha256") != arrays["weights"]["sha256"] or statistics.get("original_groups_asset_sha256") != group_ref["sha256"]:
            raise GraphError("active statistics belongs to a different original full W")
        seen_replacements = set()
        for replacement in statistics.get("replacements", []):
            gid = replacement["group_id"]
            if gid not in groups or gid in seen_replacements:
                raise GraphError("unknown/repeated marginal W group")
            group = groups[gid]
            rows = members[group['member_start']:group['member_start']+group['member_count']]
            if not np.array_equal(np.asarray(replacement["source_rows"], np.int64), rows[active[rows]]):
                raise GraphError("marginal W retained row order differs from current active-source mask")
            if replacement.get("original_matrix_sha256", group["matrix_sha256"]) != group["matrix_sha256"]:
                raise GraphError("marginal W does not refer to the corresponding original matrix")
            wref = replacement["W"]
            w = np.fromfile(resolve(wref, shape=(replacement["rank"], 2*len(replacement["source_rows"]))), dtype=wref["dtype"])
            if not np.isfinite(w).all():
                raise GraphError("nonfinite marginal full W")
            seen_replacements.add(gid)
        if seen_replacements != affected_groups:
            raise GraphError("marginal replacements do not cover exactly the affected original groups")
        replacement_count = len(seen_replacements)
        if statistics.get("active_observations") != int(active.sum()):
            raise GraphError("active statistical count differs from active-source mask")
    if not active.all():
        if "source_exclusion" not in semantic:
            raise GraphError("inactive original observations require an explicit exclusion record")
        exclusions = _read(resolve(semantic["source_exclusion"]))
        listed = sorted(int(e["source_row"]) for e in exclusions.get("exclusions", []))
        if "source_audit_revision" in semantic:
            audit = _read(resolve(semantic["source_audit_revision"]))
            if audit.get("schema") != "rinkfit.native-source-audit-revision.v1":
                raise GraphError("unsupported source audit revision schema")
            additions = audit.get("new_audit_exclusions", [])
            listed = sorted(set(listed) | {int(e["source_row"]) if isinstance(e,dict) else int(e) for e in additions})
        if listed != np.flatnonzero(~active).tolist():
            raise GraphError("exclusion record and active-source mask disagree")
        declared_budget = exclusions.get("original_engineering_budget_sha256")
        if declared_budget and declared_budget != budget_ref["sha256"]:
            raise GraphError("engineering budget differs from the preserved source-exclusion objective")
    query_profiles = 0
    if "frozen_query_manifest" in objective:
        query = _read(resolve(objective["frozen_query_manifest"]))
        if query.get("schema") != "rinkfit.immutable_native_query_curve_packet.v1":
            raise GraphError("unsupported native query packet")
        qa = {}
        for key,ref in query["arrays"].items():
            qa[key] = np.fromfile(resolve(ref),ref["dtype"]).reshape(ref["shape"])
        if not np.array_equal(qa["source_rows"],np.flatnonzero(kind==2)):
            raise GraphError("frozen queries differ from profile source row order")
        if not np.array_equal(qa["bounds"],read_array("alpha_bounds")[kind==2]):
            raise GraphError("frozen profile query domain changed")
        if qa["offsets"][0] != 0 or qa["offsets"][-1] != len(qa["alpha"]) or np.any(np.diff(qa["offsets"])<2):
            raise GraphError("invalid frozen query curve storage")
        if not np.isfinite(qa["uv"]).all() or not np.isfinite(qa["alpha"]).all():
            raise GraphError("nonfinite frozen native query curve")
        query_profiles = len(qa["source_rows"])
    return {"frozen_native_queries_verified":query_profiles,"runtime_reference_policy": "semantic_v1_numeric_and_finite_material_replay",
        "runtime_semantic_references_verified": len(references), "original_groups": len(groups),
        "nested_source_order_runtime_references_verified": nested_runtime_count,
        "grouped_source_rows": len(members), "independent_source_rows": len(independent),
        "every_original_source_counted_once": True, "original_W_slice_hashes_verified": len(groups),
        "active_rows": int(active.sum()), "exact_marginal_W_groups": replacement_count,
        "engineering_budget_min_px": float(budget.min()), "engineering_budget_max_px": float(budget.max()),
        "full_numeric_W_E_closure_verified": True, "historical_provenance_references_are_not_runtime_closure": True,
        "objective_cost_recomputed": False}


def material_registry(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Numeric indexes are local to this model hash, never universal identities."""
    blocks = model["blocks"]
    seen = set(); result = []
    for material_index, material in enumerate(model["materials"]):
        label = material["label"]
        if label in seen:
            raise GraphError(f"duplicate material label in imported checkpoint: {label}")
        seen.add(label)
        path_ids = set()
        for branch, path in enumerate(material["paths"]):
            path_id = path.get("id", f"branch:{branch}")
            if path_id in path_ids:
                raise GraphError(f"duplicate finite path ID: {label}/{path_id}")
            path_ids.add(path_id)
            topology = {k: v for k, v in path.items()
                        if k not in {"provenance", "geometry_block", "advertisement_block"}}
            def block_name(key):
                value = int(path.get(key, -1))
                if value < 0:
                    return None
                if value >= len(blocks):
                    raise GraphError(f"out-of-range block reference: {label}/{path_id}")
                return blocks[value]["name"]
            result.append({"material_key": label, "path_key": path_id,
                "model_local_material_index": material_index, "model_local_branch": branch,
                "geometry_block_name": block_name("geometry_block"),
                "advertisement_block_name": block_name("advertisement_block"),
                "physical_point": bool(path.get("physical_point", False)),
                "topology_sha256": hashlib.sha256(json.dumps(topology, sort_keys=True,
                    separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()})
    return result


def _checkpoint_root(value):
    value = Path(value).resolve()
    if value.is_file():
        if value.name not in {"manifest.json", "complete_state_manifest.json"}:
            raise GraphError("select an explicit complete checkpoint manifest")
        return value.parent, value
    manifest = value / "manifest.json"
    candidate = value / "complete_state_manifest.json"
    if manifest.is_file() and candidate.is_file():
        raise GraphError("ambiguous checkpoint directory; select its manifest explicitly")
    return value, candidate if candidate.is_file() else manifest


def _complete_inspection(checkpoint, manifest_path):
    from rinkfit_start.state_integrity import validate_receipt_bindings
    manifest = _read(manifest_path)
    validated = validate_receipt_bindings(manifest, checkpoint)
    model = validated["model"]
    n = int(manifest["source_rows"])
    names = {"station":("station.f64","<f8"), "branch":("path.u32","<u4"),
             "row":("initial_row.f64","<f8"), "alpha":("initial_alpha.f64","<f8")}
    rows = {role:{"file":name,"dtype":dtype,"shape":[n],
                  "sha256":manifest["row_state_files"][name]["sha256"]}
            for role,(name,dtype) in names.items()}
    registry = material_registry(model)
    return {"schema":"rinkfit.s01_checkpoint_inspection.v1",
        "checkpoint_manifest_filename":manifest_path.name,
        "checkpoint_manifest_sha256":sha256(manifest_path),"model_sha256":sha256(checkpoint/"model.json"),
        "source_identity":model["source_identity"],"raw_rows":n,"active_rows":manifest["active_rows"],
        "blocks":len(model["blocks"]),"camera_controls":int(model["camera"]["n_controls"]),
        "material_count":len(model["materials"]),"path_count":len(registry),"row_state_files":rows,
        "physics_and_nuisance_complete":True,"full_objective_dependencies_checked":False,
        "old_solver_certificates_transplanted":False,"model_accuracy_accepted_by_import":False,
        "state_integrity_schema":manifest["schema"],
        "preserved_original_W_half_cost":manifest["metrics"]["whole_original_W_half_cost"],
        "statistics_rank":validated["report"]["statistics_rank"]}


def inspect_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint, manifest_path = _checkpoint_root(checkpoint)
    model_path = checkpoint / "model.json"
    if manifest_path.is_file() and _read(manifest_path).get("schema") == "rinkfit.complete_recomputed_native_candidate.v1":
        return _complete_inspection(checkpoint, manifest_path)
    if not manifest_path.is_file() or not model_path.is_file():
        raise GraphError("need complete checkpoint directory containing manifest.json and model.json")
    manifest, model = _read(manifest_path), _read(model_path)
    if model.get("schema") != "rinkfit.event_model.v1.1":
        raise GraphError("unsupported latest model schema; add an adapter, never restore the old seed")
    expected = manifest.get("model", {}).get("sha256")
    actual = sha256(model_path)
    if expected != actual:
        raise GraphError("checkpoint manifest/model SHA mismatch")
    blocks = model.get("blocks", [])
    if len({b["name"] for b in blocks}) != len(blocks):
        raise GraphError("duplicate parameter block names")
    if not blocks or len(blocks[0].get("values", [])) != 32:
        raise GraphError("event v1.1 requires complete global32 physical parameters")
    for block in blocks:
        v = np.asarray(block["values"], float)
        if v.ndim != 1 or not np.isfinite(v).all():
            raise GraphError(f"invalid complete parameter block: {block['name']}")
        for k in ("lower", "upper"):
            if k in block and np.asarray(block[k]).shape != v.shape:
                raise GraphError(f"bound shape mismatch: {block['name']}/{k}")
    row_files = manifest.get("row_state_files", {})
    if not REQUIRED_ROWS.issubset(row_files):
        raise GraphError("partial row state: station, branch, row and alpha must be preserved together")
    n = int(manifest.get("raw_rows", 0))
    if n <= 0:
        raise GraphError("missing original source row count")
    checked = {}
    for role, entry in row_files.items():
        path = checkpoint / _relative(entry["file"])
        if not path.is_file() or path.is_symlink() or sha256(path) != entry["sha256"]:
            raise GraphError(f"missing or changed row-state artifact: {role}")
        dtype = np.dtype(entry["dtype"])
        shape = tuple(int(x) for x in entry["shape"])
        if role in REQUIRED_ROWS and shape != (n,):
            raise GraphError(f"incompatible source row order/count in {role}")
        if path.stat().st_size != int(np.prod(shape)) * dtype.itemsize:
            raise GraphError(f"array byte size differs from manifest: {role}")
        a = np.fromfile(path, dtype=dtype)
        if a.dtype.kind == "f" and not np.isfinite(a).all():
            raise GraphError(f"non-finite imported row-state values: {role}")
        checked[role] = {"file": str(path.relative_to(checkpoint)).replace("\\", "/"),
                         "sha256": entry["sha256"], "shape": list(shape), "dtype": dtype.str}
    camera = model["camera"]
    begin, count = int(camera.get("control_block_begin", 1)), int(camera["n_controls"])
    if begin < 1 or begin + count > len(blocks):
        raise GraphError("incomplete camera control blocks")
    if any(len(b["values"]) != 3 for b in blocks[begin:begin + count]):
        raise GraphError("camera control block shape is not alpha/beta/logf")
    registry = material_registry(model)
    return {"schema": "rinkfit.s01_checkpoint_inspection.v1",
            "checkpoint_manifest_filename": manifest_path.name,
            "checkpoint_manifest_sha256": sha256(manifest_path), "model_sha256": actual,
            "source_identity": model.get("source_identity"), "raw_rows": n,
            "active_rows": manifest.get("active_rows"), "blocks": len(blocks),
            "camera_controls": count, "material_count": len(model["materials"]),
            "path_count": len(registry), "row_state_files": checked,
            "physics_and_nuisance_complete": True,
            "full_objective_dependencies_checked": False,
            "old_solver_certificates_transplanted": False,
            "model_accuracy_accepted_by_import": False}


def _dependencies(path: Path, inspection: Mapping[str, Any], model: Mapping[str, Any]):
    """Validate an explicit file closure; Windows host paths remain usable locally."""
    manifest = _read(path)
    if manifest.get("schema") != "rinkfit.s01_dependency_manifest.v1":
        raise GraphError("unsupported s01 dependency manifest schema")
    if manifest.get("closure_complete") is not True:
        raise GraphError("W/E/semantic/reference closure must be explicitly complete")
    if manifest.get("source_checkpoint_manifest_sha256") != inspection["checkpoint_manifest_sha256"] or manifest.get("source_model_sha256") != inspection["model_sha256"]:
        raise GraphError("dependency closure was exported for a different committed checkpoint")
    entries = manifest.get("files", [])
    if not REQUIRED_DEPENDENCIES.issubset({e["role"] for e in entries}):
        raise GraphError("dependency closure omits observations/bindings/semantics/objective/source clock")
    targets = set(); role_entries = {}
    for e in entries:
        source = Path(e["source_path"])
        if not source.is_absolute():
            source = path.parent / source
        target = _relative(e["snapshot_path"])
        if str(target) in targets:
            raise GraphError("duplicate dependency snapshot path")
        targets.add(str(target))
        if not source.is_file() or source.is_symlink() or sha256(source) != e["sha256"]:
            raise GraphError(f"missing or changed dependency: {e['role']} / {source}")
        if e["role"] in REQUIRED_DEPENDENCIES:
            if e["role"] in role_entries:
                raise GraphError(f"ambiguous required dependency role: {e['role']}")
            role_entries[e["role"]] = source
        e["resolved_source_path"] = str(source.resolve())
    n = inspection["raw_rows"]
    observation_sha = sha256(role_entries["observations"])
    expected_sha = model.get("source_hashes", {}).get("observations.npz")
    if expected_sha and observation_sha != expected_sha:
        raise GraphError("imported physical model and original observation ledger have different identities")
    with np.load(role_entries["observations"], allow_pickle=False) as z:
        if z["source_uv"].shape != (n, 2) or z["native_pts"].shape != (n,):
            raise GraphError("observation source row count differs from complete checkpoint")
        if z["native_pts"].dtype.kind not in "iu" or not np.isfinite(z["source_uv"]).all():
            raise GraphError("invalid native observation data")
        obs_pts = z["native_pts"].copy()
    material = np.fromfile(role_entries["resolved_material"], dtype="<u4")
    variant = np.fromfile(role_entries["resolved_variant"], dtype="<u4")
    if material.shape != (n,) or variant.shape != (n,) or material.max(initial=0) >= len(model["materials"]):
        raise GraphError("resolved bindings are not from this complete model/source ledger")
    # resolved_variant is a legacy variant identifier, NOT a finite-path branch.
    clock = _read(role_entries["source_clock_manifest"])
    if clock.get("time_base") != [1, 12800] or clock.get("image_size") != [1920, 1080]:
        raise GraphError("source clock manifest must describe native official 50 fps pixels")
    if np.any(obs_pts < clock["first_pts"]) or np.any(obs_pts > clock["last_pts"]):
        raise GraphError("s01 observation time lies outside declared source scope")
    if np.any((obs_pts - clock["first_pts"]) % 256):
        raise GraphError("s01 observation time is not on the native source exposure lattice")
    semantic = _read(role_entries["semantic_manifest"])
    objective = _read(role_entries["objective_manifest"])
    for name, role in (("resolved_material_index", "resolved_material"), ("resolved_variant_index", "resolved_variant")):
        if semantic[name]["sha256"] != sha256(role_entries[role]):
            raise GraphError(f"{role} role and selected semantic manifest name different binding arrays")
    with np.load(role_entries["observations"], allow_pickle=False) as z:
        for name, spec in semantic.get("immutable_source_arrays", {}).items():
            a = z[name]
            if list(a.shape) != spec["shape"] or a.dtype.str != spec["dtype_str"]:
                raise GraphError("semantic immutable source array shape/dtype differs from original NPZ")
            if hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() != spec["sha256_c_contiguous_array_bytes"]:
                raise GraphError("semantic packet and original source array bytes differ")
    if objective.get("semantic_manifest_sha256") != sha256(role_entries["semantic_manifest"]):
        raise GraphError("objective closure and selected latest semantics have different identities")
    files_by_sha = {entry["sha256"]: Path(entry["resolved_source_path"]) for entry in entries}
    closure = _verify_numeric_closure(semantic, objective, files_by_sha, n)
    if inspection.get("active_rows") is not None and closure["active_rows"] != inspection["active_rows"]:
        raise GraphError("complete checkpoint and objective closure disagree on active-source membership")
    return manifest, entries, clock, closure


def export_dependencies(checkpoint: str | Path, observations: str | Path,
                        semantic_manifest: str | Path, engineering_budget: str | Path,
                        source_clock: str | Path, output: str | Path,
                        *, asset_roots: list[str | Path] | None = None,
                        example_i20_not_latest: bool = False) -> dict[str, Any]:
    """Resolve every supported runtime reference and write a verified closure.

    On the actual host, original absolute references normally resolve directly.
    --asset-root enables SHA-qualified lookup of relocated review/export assets.
    No basename-only match or guessed replacement is accepted.
    """
    checkpoint, selected_manifest_path = _checkpoint_root(checkpoint); output = Path(output).resolve()
    if output.exists():
        raise GraphError("dependency output must be a new file")
    inspection = inspect_checkpoint(selected_manifest_path)
    model = _read(checkpoint / "model.json")
    semantic_path = Path(semantic_manifest).resolve(); semantic = _read(semantic_path)
    roots = [Path(p).resolve() for p in (asset_roots or [])]
    # Relative inherited descriptors retain their declaring parent's directory.
    # Never reinterpret an inherited basename relative to the selected child.
    lineage_dirs = []
    lineage_paths = []
    def lineage(path):
        path = Path(path).resolve()
        if path in lineage_paths:
            return
        lineage_paths.append(path); lineage_dirs.append(path.parent)
        value = _read(path)
        for key in sorted(SEMANTIC_PROVENANCE_KEYS):
            if "parent_manifest" in key and key in value:
                ref = value[key]; raw = Path(ref.get("file",ref.get("path","")))
                target = raw if raw.is_absolute() else path.parent/raw
                if not target.is_file() or sha256(target) != ref["sha256"]:
                    raise GraphError("semantic parent is missing or changed: "+str(target))
                lineage(target)
    lineage(semantic_path)
    digest_cache = {}; basename_cache = {}
    def digest(path):
        path = path.resolve()
        if path not in digest_cache:
            digest_cache[path] = sha256(path)
        return digest_cache[path]
    def locate(ref, base):
        source_name = ref.get("file", ref.get("path"))
        raw = Path(source_name)
        candidates = [raw, base / raw] + [directory/raw for directory in lineage_dirs]
        basename = source_name.replace("\\", "/").rsplit("/", 1)[-1]
        for p in candidates:
            if p.is_file() and not p.is_symlink() and digest(p) == ref["sha256"]:
                return p.resolve()
        if basename not in basename_cache:
            try:
                found = subprocess.run(["rg","--files","--hidden","-g",basename,*map(str,roots)],
                    capture_output=True,text=True,encoding="utf-8",check=False)
                if found.returncode not in (0,1):
                    raise OSError(found.stderr)
                basename_cache[basename] = [Path(line) for line in found.stdout.splitlines() if line]
            except OSError:
                basename_cache[basename] = [p for root in roots for p in root.rglob(basename) if p.is_file()]
        for p in basename_cache[basename]:
            if p.is_file() and not p.is_symlink() and digest(p) == ref["sha256"]:
                return p.resolve()
        raise GraphError(f"unresolved runtime artifact with required SHA {ref['sha256']}: {source_name}")
    entries = []; by_source = {}
    def add(path, role):
        path = Path(path).resolve()
        if path in by_source:
            return by_source[path]
        if not path.is_file() or path.is_symlink():
            raise GraphError(f"missing export dependency: {path}")
        d = digest(path)
        entry = {"role": role, "source_path": str(path),
                 "snapshot_path": f"assets/{d[:12]}_{path.name}", "sha256": d}
        for root in roots:
            if path.is_relative_to(root):
                entry["review_or_export_root_relative_locator"] = str(path.relative_to(root)).replace("\\", "/")
                break
        entries.append(entry); by_source[path] = entry
        return entry
    add(observations, "observations"); add(semantic_path, "semantic_manifest")
    add(source_clock, "source_clock_manifest")
    add(locate(semantic["resolved_material_index"], semantic_path.parent), "resolved_material")
    add(locate(semantic["resolved_variant_index"], semantic_path.parent), "resolved_variant")
    for pointer, ref in _semantic_refs(semantic):
        add(locate(ref, semantic_path.parent), "semantic" + pointer)
    source_record_assets = []; current_source_records = {}; unique_record_refs = {}
    for name, ref in semantic.get("assets", {}).items():
        if name.endswith((".jsonl", ".jsonl.gz")):
            record_path = locate(ref, semantic_path.parent)
            referenced = 0
            for pointer, source_ref in _jsonl_file_refs(record_path):
                key = (source_ref["sha256"], source_ref.get("file",source_ref.get("path")))
                if key not in unique_record_refs:
                    target = locate(source_ref, record_path.parent)
                    unique_record_refs[key] = target
                    add(target, "semantic_source_records/" + name + pointer)
                else:
                    target = unique_record_refs[key]
                if name == "source_rows.jsonl" and pointer.endswith("/source_packet"):
                    current_source_records[source_ref["sha256"]] = target
                referenced += 1
            source_record_assets.append({"asset": name, "sha256": digest(record_path),
                                         "file_reference_occurrences_verified": referenced})
    # Per-frame representatives often refer to a separate current material
    # readout packet. These actual two-interface/line/circle records are source
    # dependencies; they must not disappear behind an opaque source_rows blob.
    current_material_records = {}; exposure_proofs = []
    for source_sha, source_path in sorted(current_source_records.items()):
        source_document = _read(source_path)
        for key in ("board_source_readout_packet", "ice_source_readout_packet"):
            child_ref = source_document.get(key)
            if isinstance(child_ref,dict) and "sha256" in child_ref:
                child = locate(child_ref,source_path.parent)
                add(child,"current_native_material_readout/"+key)
                current_material_records[child_ref["sha256"]] = str(child)
        exposure = source_document.get("source")
        if not isinstance(exposure,dict):
            continue
        proof = {"source_record_sha256":source_sha}
        for path_key, hash_key, role in (("image_path","png_sha256","native_source_PNG"),
                ("mask_path","mask_sha256","native_source_exclusion_mask"),
                ("source","source_video_sha256","native_source_video")):
            if isinstance(exposure.get(path_key),str) and isinstance(exposure.get(hash_key),str):
                target = locate({"path":exposure[path_key],"sha256":exposure[hash_key]},source_path.parent)
                entry = add(target,role);entry["snapshot_policy"]="verified_external_evidence"
                proof[role] = {"file":str(target),"sha256":entry["sha256"]}
        exposure_proofs.append(proof)
    if "active_statistics" in semantic:
        statistics_path = locate(semantic["active_statistics"], semantic_path.parent)
        for pointer, ref in _file_refs(_read(statistics_path)):
            add(locate(ref, statistics_path.parent), "active_statistics" + pointer)
    if "source_order_overlay" in semantic:
        order_path = locate(semantic["source_order_overlay"], semantic_path.parent)
        order = _read(order_path)
        if order.get("schema") != "rinkfit.source_order_evidence_overlay.v1":
            raise GraphError("unsupported source-order overlay; add an explicit runtime adapter")
        for key in ("audit_source_runs", "classifications", "active_hard_source_runs"):
            add(locate(order[key], order_path.parent), "source_order/" + key)
    # The selected parent/overlay chain is numerical authority, even when its
    # descriptors are inherited by the flattened selected manifest.
    for parent_path in lineage_paths:
        add(parent_path, "semantic_lineage")
        parent_value = _read(parent_path)
        for pointer, ref in _semantic_refs(parent_value):
            add(locate(ref,parent_path.parent), "semantic_lineage"+pointer)
    if "source_audit_revision" in semantic:
        from rinkfit_start.semantic_resolution import load_semantics
        load_semantics(semantic_path, inspection["raw_rows"], expected_sha256=digest(semantic_path))
    # Capture the files actually consumed by the strict source authority. This
    # closes nested proof/source dependencies without following unrelated old
    # camera reports merely mentioned by an original source packet.
    if inspection.get("state_integrity_schema"):
        from rinkfit_start.state_integrity import validate_receipt_bindings
        accessed = set()
        def trace_hash(frame,event,arg):
            if event == "call" and frame.f_code.co_name in {"sha256_file","sha256"}:
                value = frame.f_locals.get("path")
                if isinstance(value,(str,Path)):
                    target = Path(value)
                    if target.is_file(): accessed.add(target.resolve())
            return trace_hash
        previous = sys.getprofile()
        try:
            sys.setprofile(trace_hash)
            validate_receipt_bindings(_read(selected_manifest_path),checkpoint)
        finally:
            sys.setprofile(previous)
        for path in sorted(accessed):
            entry = add(path,"authority_verified_recursive_input")
            if path.suffix.lower() in {".mp4",".mkv",".avi",".mov"}:
                entry["snapshot_policy"] = "verified_external_evidence"
    native_query = None
    if inspection.get("state_integrity_schema"):
        complete = _read(selected_manifest_path)
        packet_path = locate(complete["packet"],selected_manifest_path.parent)
        packet_value = _read(packet_path); add(packet_path,"native_packet")
        query_ref = packet_value.get("query_packet")
        if not isinstance(query_ref,dict):
            raise GraphError("complete native state lacks its frozen query packet")
        query_ref = {"file":query_ref.get("manifest"),"sha256":query_ref["sha256"]}
        query_path = locate(query_ref,packet_path.parent);query = _read(query_path)
        query_entry = add(query_path,"query_manifest")
        if query.get("schema") != "rinkfit.immutable_native_query_curve_packet.v1":
            raise GraphError("unsupported frozen profile query schema")
        for pointer, ref in _file_refs(query.get("arrays",{})):
            add(locate(ref,query_path.parent),"frozen_query"+pointer)
        if "certificates" in query:
            add(locate(query["certificates"],query_path.parent),"frozen_query_certificates")
        native_query = {"file":query_entry["snapshot_path"],"sha256":query_entry["sha256"]}
    budget_entry = add(engineering_budget, "engineering_budget")
    objective_path = output.with_name(output.stem + ".objective.json")
    if objective_path.exists():
        raise GraphError("derived objective export file already exists")
    objective = {"schema": "rinkfit.s01_objective_export.v1",
        "example_i20_not_latest": example_i20_not_latest,
        "source_checkpoint_manifest_sha256": inspection["checkpoint_manifest_sha256"],
        "semantic_manifest_sha256": digest(semantic_path), "raw_rows": inspection["raw_rows"],
        "statistical_objective": "sum of squared original full grouped W residuals plus original independent scalar rows",
        "engineering_error": "Euclidean UV for ordinary/entity; recorded normal component for true profile",
        "engineering_budget": {"file": budget_entry["snapshot_path"],
            "sha256": budget_entry["sha256"], "dtype": "<f8", "shape": [inspection["raw_rows"]]},
        "runtime_reference_policy": "semantic_v1_numeric_and_finite_material_replay",
        "parameter_values_and_domains": "complete checkpoint/model.json; no source/domain substitution",
        "inactive_sources": "exact recorded active-source mask and marginal group replacements only"}
    if native_query is not None:
        objective["frozen_query_manifest"] = native_query
    output.parent.mkdir(parents=True, exist_ok=True)
    objective_path.write_text(json.dumps(objective, ensure_ascii=False, indent=2), encoding="utf-8")
    add(objective_path, "objective_manifest")
    provenance = []
    for key in SEMANTIC_PROVENANCE_KEYS:
        for pointer, ref in _file_refs(semantic.get(key), "/" + key):
            provenance.append({"pointer": pointer, **ref,
                               "classification": "historical construction provenance, not a runtime numerical input"})
    result = {"schema": "rinkfit.s01_dependency_manifest.v1", "closure_complete": True,
        "source_record_assets_verified": source_record_assets,
        "current_native_material_readout_records_verified": current_material_records,
        "native_source_exposure_proofs": exposure_proofs,
        "example_i20_not_latest": example_i20_not_latest,
        "source_checkpoint_manifest_sha256": inspection["checkpoint_manifest_sha256"],
        "source_model_sha256": inspection["model_sha256"],
        "closure_policy": "all supported direct runtime semantic/finite-material refs and all marginal W/proof refs",
        "history_provenance_refs_not_required_for_numeric_replay": provenance,
        "files": entries}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        _, _, _, verified = _dependencies(output, inspection, model)
    except Exception:
        result["closure_complete"] = False
        result["export_failed_validation"] = True
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    result["numeric_closure_verification"] = verified
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def snapshot_checkpoint(checkpoint: str | Path, destination: str | Path,
                        *, dependency_manifest: str | Path | None = None,
                        graph: str | Path | None = None) -> dict[str, Any]:
    checkpoint, selected_manifest_path = _checkpoint_root(checkpoint)
    destination = Path(destination).resolve()
    if destination.exists():
        raise GraphError("destination must be new; neither repair checkpoint nor previous joint state is overwritten")
    if checkpoint == destination or checkpoint in destination.parents or destination in checkpoint.parents:
        raise GraphError("snapshot must be outside the live checkpoint tree")
    inspection = inspect_checkpoint(selected_manifest_path)
    model = _read(checkpoint / "model.json")
    dependency_data = None
    if dependency_manifest is not None:
        dependency_data = _dependencies(Path(dependency_manifest).resolve(), inspection, model)
    if graph is not None and dependency_data is None:
        raise GraphError("activating s01 in a joint graph requires complete objective/source dependencies")
    source_files = sorted(p for p in checkpoint.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in checkpoint.rglob("*")):
        raise GraphError("checkpoint symlinks must be materialised in its explicit export first")
    before = {p.relative_to(checkpoint).as_posix(): sha256(p) for p in source_files}
    destination.mkdir(parents=True)
    try:
        for relative, digest in before.items():
            target = destination / "checkpoint" / _relative(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkpoint / _relative(relative), target)
            if sha256(target) != digest:
                raise GraphError(f"checkpoint changed during import: {relative}")
        if sha256(selected_manifest_path) != inspection["checkpoint_manifest_sha256"]:
            raise GraphError("selected source manifest changed while copying; select a committed immutable checkpoint")
        copied_inspection = {"model_sha256": sha256(destination / "checkpoint" / "model.json")}
        if copied_inspection["model_sha256"] != inspection["model_sha256"]:
            raise GraphError("physical model changed between inspection and snapshot")
        replay_ready = dependency_data is not None and dependency_data[3]["full_numeric_W_E_closure_verified"]
        copied_dependencies = []
        if dependency_data is not None:
            dependency, entries, clock, closure = dependency_data
            for entry in entries:
                relative = Path("dependencies") / _relative(entry["snapshot_path"])
                target = destination / relative
                if entry.get("snapshot_policy") == "verified_external_evidence":
                    if sha256(Path(entry["resolved_source_path"])) != entry["sha256"]:
                        raise GraphError("external evidence changed during import")
                    copied_dependencies.append({"role":entry["role"],"path":entry["resolved_source_path"],
                        "sha256":entry["sha256"],"snapshot_policy":"verified_external_evidence"})
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(entry["resolved_source_path"], target)
                    if sha256(target) != entry["sha256"]:
                        raise GraphError(f"dependency changed during import: {entry['role']}")
                    copied_dependencies.append({"role": entry["role"], "path": relative.as_posix(),
                                                "sha256": entry["sha256"]})
            (destination / "dependency_source_manifest.json").write_text(
                json.dumps(dependency, indent=2, ensure_ascii=False), encoding="utf-8")
        receipt = {**inspection, "schema": "rinkfit.s01_joint_import_receipt.v1",
            "source_checkpoint": str(checkpoint), "snapshot": str(destination),
            "checkpoint_files": before, "dependencies": copied_dependencies,
            "dependency_paths_by_sha256": {e["sha256"]: e["path"] for e in copied_dependencies},
            "full_objective_dependencies_checked": replay_ready,
            "dependency_check_scope": "all supported runtime semantic refs, actual group W slices/coverage, marginal W, E and native source clock",
            "numeric_closure_verification": None if dependency_data is None else dependency_data[3],
            "objective_cost_recomputed": False,
            "ready_for_full_residual_replay": replay_ready,
            "solver_resume_authorised_by_this_receipt": False,
            "invalidate": ["predictions", "stations_after_geometry_change", "RS_roots",
                "Jacobians", "factorisations", "duals", "trust_region_model", "feasibility_certificates"],
            "scene_parameters_frozen": False}
        (destination / "material_registry.json").write_text(
            json.dumps(material_registry(model), ensure_ascii=False, indent=2), encoding="utf-8")
        if graph is not None:
            candidate = _read(Path(graph))
            if candidate["source_clock"]["source_video_sha256"] != clock["source_video_sha256"]:
                raise GraphError("s01 repair source and joint programme source video differ")
            scope = candidate["shots"]["s01"]["scope"]
            if (scope["first_pts"], scope["last_pts"]) != (clock["first_pts"], clock["last_pts"]):
                raise GraphError("s01 repair and joint programme scope differ")
            candidate = with_s01_event(candidate, model, model_ref="checkpoint/model.json",
                model_sha256=inspection["model_sha256"], import_receipt="import_receipt.json")
            (destination / "candidate_joint_graph.json").write_text(
                json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
            receipt["candidate_graph"] = "candidate_joint_graph.json"
        (destination / "import_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        return receipt
    except Exception as error:
        # Keep the failed destination visibly incomplete; never erase source data.
        (destination / "IMPORT_FAILED.json").write_text(
            json.dumps({"complete": False, "error": str(error)}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect"); inspect.add_argument("checkpoint")
    take = sub.add_parser("snapshot")
    take.add_argument("checkpoint"); take.add_argument("destination")
    take.add_argument("--dependencies"); take.add_argument("--graph")
    export = sub.add_parser("export-dependencies")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--observations", required=True)
    export.add_argument("--semantic-manifest", required=True)
    export.add_argument("--engineering-budget", required=True)
    export.add_argument("--source-clock", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--asset-root", action="append", default=[])
    export.add_argument("--example-i20-not-latest", action="store_true")
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_checkpoint(args.checkpoint)
    elif args.command == "snapshot":
        result = snapshot_checkpoint(args.checkpoint, args.destination,
            dependency_manifest=args.dependencies, graph=args.graph)
    else:
        result = export_dependencies(args.checkpoint, args.observations, args.semantic_manifest,
            args.engineering_budget, args.source_clock, args.out,
            asset_roots=args.asset_root, example_i20_not_latest=args.example_i20_not_latest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
