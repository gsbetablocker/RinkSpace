"""One dense native-source append, with compact immutable parent/ID references.

Only this joint-packet ABI is supported. The source store is read-only; sampled
curve projection initializes numerical branch/station and never source truth.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np

REGISTRY = "rinkfit.dense_source_suffix_registry.v1"
APPEND_PATHS = {"groups", "queries", "source_manifests", "recapture_rows",
                "engineering_pixel_dependency_groups.groups", "allowed_paths"}
IDS = ("source_row_ids", "source_dataset_ids", "source_family_ids")
AUX = {"station", "path", "row", "alpha"}


def _sha(path):
    from .io_utils import file_sha256
    return file_sha256(path)


def _ref(path):
    path = Path(path).resolve()
    return {"file": str(path), "sha256": _sha(path)}


def _checked(ref, base):
    path = Path(ref.get("file", ref.get("path")))
    path = (path if path.is_absolute() else Path(base) / path).resolve()
    if not path.is_file() or _sha(path) != ref["sha256"]:
        raise ValueError("Changed dense packet dependency: " + str(path))
    return path


def _json_equal(a, b):
    return json.dumps(a, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(b, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _array(ref, base, dtype=None, shape=None):
    path = _checked(ref, base)
    dt = np.dtype(ref["dtype"])
    if dtype is not None and dt != np.dtype(dtype):
        raise ValueError("Dense packet binary dtype differs")
    dims = tuple(ref["shape"])
    if shape is not None and dims != tuple(shape):
        raise ValueError("Dense packet binary shape differs")
    offset = int(ref.get("offset_bytes", 0))
    if offset < 0 or path.stat().st_size < offset + int(np.prod(dims)) * dt.itemsize:
        raise ValueError("Truncated dense packet array")
    return np.memmap(path, dtype=dt, mode="r", offset=offset, shape=dims)


def _get(value, dotted):
    for key in dotted.split("."):
        value = value[key]
    return value


def _put(value, dotted, item):
    keys = dotted.split(".")
    for key in keys[:-1]:
        value = value[key]
    value[keys[-1]] = item


def _read_parent(ref, base, stack, registry_cache=None):
    path = _checked(ref, base)
    key = str(path).casefold()
    if key in stack:
        raise ValueError("Cyclic immutable dense packet parent")
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    return expand_dense_packet(raw, path, _stack=stack + (key,), _registry_cache=registry_cache)


def _registry_columns(registry, base):
    if registry.get("schema") != REGISTRY:
        raise ValueError("Unsupported dense source ID registry")
    n = int(registry["count"])
    if n <= 0 or int(registry["row_start"]) <= 0:
        raise ValueError("Invalid dense source suffix range")
    data = _array(registry["source_id_utf8"], base, "|u1")
    offsets = _array(registry["source_id_offsets"], base, "<u8", [n + 1])
    if offsets[0] != 0 or offsets[-1] != len(data) or np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("Invalid dense UTF8 source ID offsets")
    blob = bytes(data)
    sourceids = [blob[int(offsets[i]):int(offsets[i + 1])].decode("utf8", errors="strict") for i in range(n)]
    if any(not x or "\x00" in x for x in sourceids) or len(set(sourceids)) != n:
        raise ValueError("Empty or ambiguous dense original selection IDs")
    family = registry["source_family_ids"]
    if not family or len(family) != len(set(family)) or any(not isinstance(x, str) or not x for x in family):
        raise ValueError("Invalid dense source family dictionary")
    fi = _array(registry["source_family_index"], base, "<u4", [n])
    if np.any(fi >= len(family)):
        raise ValueError("Dense source family dictionary index overflow")
    ordinal = _array(registry["source_store_ordinal"], base, "<u8", [n])
    if np.any(ordinal[1:] <= ordinal[:-1]):
        raise ValueError("Dense source store ordinal must retain actual original stream order")
    dataset = registry["source_dataset_id"]
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("Missing dense source dataset identity")
    return sourceids, [dataset] * n, [family[int(i)] for i in fi]


def expand_dense_packet(value, packet_path, *, _stack=None, _registry_cache=None):
    """Hydrate exact parent metadata and columnar IDs for old Python consumers."""
    if not isinstance(value, dict) or value.get("schema") != "rinkfit.joint_packet.v1":
        return value
    if "dense_source_binding" not in value and "dense_parent_packet" not in value:
        return value
    path = Path(packet_path).resolve()
    stack = _stack or (str(path).casefold(),)
    # Each immutable ID suffix is decoded once within this recursive read.
    # A new top-level read always rechecks the original pinned source bytes.
    registry_cache = {} if _registry_cache is None else _registry_cache
    if "dense_parent_packet" in value:
        parent = _read_parent(value["dense_parent_packet"], path.parent, stack, registry_cache)
        if parent.get("schema") != value["schema"]:
            raise ValueError("Dense packet parent ABI differs")
        result = dict(parent)
        appends = value.get("dense_parent_append", {})
        if not isinstance(appends, dict) or set(appends) - APPEND_PATHS:
            raise ValueError("Unsupported dense packet parent append")
        for dotted in appends:
            if dotted.split(".")[0] in value:
                raise ValueError("Conflicting full override and parent append: " + dotted)
        result.update({k: v for k, v in value.items() if k != "dense_parent_append"})
        for dotted, tail in appends.items():
            prior = _get(parent, dotted)
            if isinstance(prior, list) and isinstance(tail, list):
                _put(result, dotted, prior + tail)
            elif dotted == "allowed_paths" and isinstance(prior, dict) and isinstance(tail, dict):
                if set(prior) & set(tail):
                    raise ValueError("Dense append overwrites inherited allowed source path")
                _put(result, dotted, dict(prior, **tail))
            else:
                raise ValueError("Dense packet parent append type differs")
        result.pop("dense_parent_append", None)
    else:
        result = dict(value)
    if "dense_source_binding" not in result:
        raise ValueError("Dense parent reference lacks its source suffix registry")
    registry = result["dense_source_binding"]
    n0, n = int(registry["row_start"]), int(registry["count"])
    extras = result.get("native_ad_source_bindings", [result["native_ad_source_binding"]] if "native_ad_source_binding" in result else [])
    if extras and extras[-1] != result.get("native_ad_source_binding"):
        raise ValueError("Latest native source registry differs from its append chain")
    if int(extras[0]["row_start"] if extras else result["row_count"]) != n0 + n:
        raise ValueError("Dense suffix range differs from packet denominator")
    cache_registry = dict(registry)
    for key in ("source_id_utf8", "source_id_offsets", "source_family_index", "source_store_ordinal"):
        ref = dict(registry[key]); dep = Path(ref.get("file", ref.get("path")))
        ref["file"] = str((dep if dep.is_absolute() else path.parent / dep).resolve())
        cache_registry[key] = ref
    cache_key = json.dumps(cache_registry, sort_keys=True, separators=(",", ":"))
    if cache_key not in registry_cache:
        registry_cache[cache_key] = _registry_columns(registry, path.parent)
    suffix = registry_cache[cache_key]
    for key, tail in zip(IDS, suffix):
        current = result.get(key)
        if not isinstance(current, list):
            raise ValueError("Dense source prefix ID list missing")
        if len(current) == n0:
            result[key] = current + tail
        elif len(current) >= n0 + n:
            if current[n0:n0+n] != tail:
                raise ValueError("Expanded dense source IDs differ from immutable registry")
        else:
            raise ValueError("Dense source prefix/full ID count differs")
    expected = n0 + n
    for extra in extras:
        start, count = int(extra["row_start"]), int(extra["count"])
        if start != expected: raise ValueError("Native source append chain has a gap or overlap")
        cache_key = json.dumps(extra, sort_keys=True, separators=(",", ":"))
        if cache_key not in registry_cache: registry_cache[cache_key] = _registry_columns(extra, path.parent)
        for key, tail in zip(IDS, registry_cache[cache_key]):
            current = result[key]
            if len(current) == start: result[key] = current + tail
            elif len(current) < start + count or current[start:start+count] != tail: raise ValueError("Native source IDs differ")
        expected += count
    if expected != int(result["row_count"]): raise ValueError("Native source append chain does not cover every raw row")
    return result


def compact_source_registry(value, packet_path):
    """Inverse hydration; source IDs and inherited Query/W metadata stay single."""
    if not isinstance(value, dict) or "dense_source_binding" not in value:
        return value
    path = Path(packet_path).resolve()
    result = dict(value)
    registry = result["dense_source_binding"]
    n0, n = int(registry["row_start"]), int(registry["count"])
    if int(result["row_count"]) != n0 + n:
        raise ValueError("Compact dense packet full denominator differs")
    suffix = _registry_columns(registry, path.parent)
    for key, tail in zip(IDS, suffix):
        current = result[key]
        if len(current) == n0 + n:
            if current[n0:] != tail:
                raise ValueError("Refuse writing changed dense source identity suffix")
            result[key] = current[:n0]
        elif len(current) != n0:
            raise ValueError("Refuse truncating dense source prefix IDs")
    if "dense_parent_packet" not in result:
        return result
    parent = _read_parent(result["dense_parent_packet"], path.parent, (str(path).casefold(),))
    if parent["row_count"] != n0:
        raise ValueError("Dense canonical parent source prefix size differs")
    for key in IDS:
        if result[key] != parent[key]:
            raise ValueError("Dense append changed inherited source ID prefix")
    raw = {"schema": result["schema"], "dense_parent_packet": result["dense_parent_packet"]}
    appends = {}
    for key, val in result.items():
        if key in {"schema", "dense_parent_packet", "dense_parent_append"}:
            continue
        if key in {"groups", "queries", "source_manifests"}:
            prior = parent[key]
            if not isinstance(val, list) or val[:len(prior)] != prior:
                raise ValueError("Dense packet changed immutable parent prefix: " + key)
            if len(val) > len(prior):
                appends[key] = val[len(prior):]
        elif key == "allowed_paths":
            prior = parent.get(key, {})
            if any(val.get(k) != v for k, v in prior.items()):
                raise ValueError("Dense packet changed inherited path authority")
            extra = {k: v for k, v in val.items() if k not in prior}
            if extra:
                appends[key] = extra
        elif key == "engineering_pixel_dependency_groups":
            prior = parent[key]
            before, after = dict(prior), dict(val)
            pg, ng = before.pop("groups"), after.pop("groups")
            if before == after and ng[:len(pg)] == pg:
                if len(ng) > len(pg):
                    appends[key + ".groups"] = ng[len(pg):]
            else:
                raise ValueError("Dense packet changed inherited engineering dependence policy")
        elif key == "recapture_rows" and isinstance(parent.get(key), list) and val[:len(parent[key])] == parent[key]:
            if len(val) > len(parent[key]):
                appends[key] = val[len(parent[key]):]
        elif key not in parent or not _json_equal(val, parent[key]):
            raw[key] = val
    if appends:
        raw["dense_parent_append"] = appends
    return raw


def _write_array(path, values):
    a = np.asarray(values)
    a.tofile(path)
    return dict(**_ref(path), dtype=a.dtype.str, shape=list(a.shape), layout="C", byte_order="little")


def _seed_frame(scene, camera, time0, rows, *, tolerance_px):
    """Batched finite-curve station seed; no source UV or qualification change."""
    from scipy.spatial import cKDTree
    from collections import defaultdict
    uv = np.asarray([r["source_uv"] for r in rows], dtype=float)
    mi = np.asarray([r["material_index"] for r in rows], dtype=np.uint32)
    paths = np.asarray([r["path_index"] for r in rows], dtype=np.uint32)
    station = np.full(len(rows), .5)
    selected = np.zeros(len(rows), bool)
    # Sampling only locates a basin; original measured row fixes source readout time.
    stations = np.linspace(0., 1., 65)
    middle_time = time0 + camera.tau * (float(np.median(uv[:, 1])) / 1079. - .5)
    for material in np.unique(mi):
        ii = np.flatnonzero(mi == material)
        allowed = sorted(set(p for i in ii for p in rows[i]["allowed_paths"]))
        ss, pp, projected = [], [], []
        for branch in allowed:
            w = scene.world(int(material), int(branch), stations)
            out = camera.at(w, middle_time)
            good = out["valid"] & np.isfinite(out["uv"]).all(1)
            ss.extend(stations[good]); pp.extend([branch] * int(good.sum())); projected.extend(out["uv"][good])
        if not ss:
            continue
        tree = cKDTree(np.asarray(projected))
        # Retain each row's source-supported branch set even when it differs locally.
        for branchset in {tuple(rows[i]["allowed_paths"]) for i in ii}:
            jj = np.asarray([i for i in ii if tuple(rows[i]["allowed_paths"]) == branchset], dtype=int)
            valid = np.isin(pp, branchset)
            idx = np.flatnonzero(valid)
            if not len(idx):
                continue
            local = tree if len(idx) == len(ss) else cKDTree(np.asarray(projected)[idx])
            _, near = local.query(uv[jj], k=1)
            near = idx[near] if len(idx) != len(ss) else near
            station[jj] = np.asarray(ss)[near]
            paths[jj] = np.asarray(pp, dtype=np.uint32)[near]
            selected[jj] = True
    readout_time = time0 + camera.tau * (uv[:, 1] / 1079. - .5)
    def project(s):
        w, legal, _ = scene.world_rows(mi, paths, s)
        result = camera.at(w, readout_time)
        return result["uv"], legal & result["valid"] & np.isfinite(result["uv"]).all(1)
    predicted, legal = project(station)
    finite = selected & legal
    status = np.zeros(len(rows), dtype=np.uint8)
    status[finite] = 1
    active = finite.copy()
    eps = np.cbrt(np.finfo(float).eps)
    while np.any(active):
        plus = np.minimum(1., station + eps)
        minus = np.maximum(0., station - eps)
        pu, pv = project(plus); mu, mv = project(minus)
        tangent = (pu - mu) / (plus - minus)[:, None]
        norm2 = np.sum(tangent*tangent, axis=1)
        residual = predicted - uv
        dot = np.sum(residual*tangent, axis=1)
        tangent_ok = pv & mv & np.isfinite(norm2) & (norm2 > np.finfo(float).tiny)
        gradient = np.abs(dot) / np.sqrt(np.maximum(norm2, np.finfo(float).tiny))
        bound = ((station == 0.) & (dot >= 0.)) | ((station == 1.) & (dot <= 0.))
        done = active & ((gradient <= tolerance_px) | bound)
        active &= ~(done | ~tangent_ok)
        if not np.any(active):
            break
        step = np.zeros(len(rows))
        step[active] = -dot[active] / norm2[active]
        oldcost = np.sum(residual*residual, axis=1)
        pending = active.copy()
        accepted = np.zeros(len(rows), bool)
        while np.any(pending):
            trial = station.copy()
            trial[pending] = np.clip(station[pending] + step[pending], 0., 1.)
            newuv, newvalid = project(trial)
            newcost = np.sum((newuv-uv)**2, axis=1)
            roundoff = np.finfo(float).eps * 8. * np.maximum(1., oldcost)
            improved = pending & newvalid & (newcost < oldcost - roundoff)
            image_step = np.linalg.norm(newuv - predicted, axis=1)
            tiny_accepted = improved & (image_step <= tolerance_px)
            station[improved] = trial[improved]
            predicted[improved] = newuv[improved]
            accepted |= improved
            active &= ~tiny_accepted
            pending &= ~improved
            too_small = pending & ((np.abs(step) * np.sqrt(np.maximum(norm2, 0.)) <= tolerance_px) | (np.abs(step) <= np.finfo(float).eps * np.maximum(1., np.abs(station))))
            pending &= ~too_small
            step[pending] *= .5
        stopped = active & ~accepted
        status[stopped] = 2
        active &= accepted
    prediction, valid = project(station)
    valid &= selected
    status[~valid] = 0
    error = np.linalg.norm(prediction-uv, axis=1)
    error[~valid] = np.inf
    return paths, station, status, error


def build_dense_joint_packet(parent_packet, input_binding, model_path, output, *,
                             engineering_E=2., station_tolerance_px=1e-4, dependency_aliases=None, resume_staging=None, fast_seed=False):
    from .io_utils import read_json, write_json, object_sha256
    from .packet_storage import write_packet
    from .dense_source_factory import open_dense_source_input
    from .joint_model import as_event_view
    from .screen_native_motion_v2 import _camera
    from rinkfit_start.native_scene import NativeScene
    pp, ip, mp, out = map(lambda x: Path(x).resolve(), (parent_packet, input_binding, model_path, output))
    resume = None if resume_staging is None else read_json(resume_staging)
    if out.exists() and resume is None:
        raise FileExistsError("Immutable dense packet output already exists")
    if resume is not None and (resume.get("schema")!="rinkfit.dense_packet_unpublished_staging_resume.v1" or Path(resume["output"]).resolve()!=out or Path(resume["parent_packet"]["file"]).resolve()!=pp or resume["parent_packet"]["sha256"]!=_sha(pp) or Path(resume["input_binding"]["file"]).resolve()!=ip or resume["input_binding"]["sha256"]!=_sha(ip) or (out/"packet.json").exists()):
        raise ValueError("Only the exact unpublished owned staging may resume")
    reuse_rows=0 if resume is None else int(resume["reuse_rows"])
    if not np.isfinite(engineering_E) or engineering_E <= 0 or not np.isfinite(station_tolerance_px) or station_tolerance_px <= 0:
        raise ValueError("Finite positive engineering and numerical budgets required")
    p, model = read_json(pp), read_json(mp)
    q = open_dense_source_input(ip, purpose="new_packet")
    q.verify_model_materials(mp)
    q.assert_dependencies()
    if q.binding.get("parent_packet") != _ref(pp):
        raise ValueError("Dense source eligibility does not belong to the exact retained parent")
    n0, count = int(p["row_count"]), q.row_count
    N = n0 + count
    shot_ids = ["s01", "s02", "s03", "s04", "s05", "s06"]
    if p["shot_ids"] != shot_ids[:4] or set(model["shots"]) != set(shot_ids):
        raise ValueError("Dense six-shot model/source scope differs")
    out.mkdir(parents=True,exist_ok=resume is not None)
    start = time.perf_counter()
    arrays, buffers, prefix = {}, {}, {}
    for name, desc in p["arrays"].items():
        a = _array(desc, pp.parent)
        if len(a) != n0:
            raise ValueError("Parent numerical row denominator differs")
        shape = (N,) + a.shape[1:]
        file = out / (name + ".bin")
        if resume is not None and file.stat().st_size!=int(np.prod(shape))*a.dtype.itemsize:
            raise ValueError("Partial numerical staging size differs")
        b = np.memmap(file, dtype=a.dtype, mode="w+" if resume is None else "r+", shape=shape)
        if resume is not None and b[:n0].tobytes()!=a.tobytes():
            raise ValueError("Partial numerical staging altered parent prefix")
        b[:n0] = a
        buffers[name] = b
        prefix[name] = hashlib.sha256(a.tobytes()).hexdigest()
    expected = {"uv", "native_pts", "shot_index", "kind", "material_index", "path", "admission",
        "fit_role", "E", "station", "row", "alpha", "source_sigma", "profile_normal", "query_uv",
        "alpha_bounds", "stroke_pair_index", "stroke_source_side"}
    if set(buffers) != expected:
        raise ValueError("Dense append needs an explicit implementation of changed numerical ABI")
    ordinal = np.memmap(out/"source_store_ordinal.u64", dtype="<u8", mode="w+" if resume is None else "r+", shape=(count,))
    family_index = np.memmap(out/"source_family_index.u32", dtype="<u4", mode="w+", shape=(count,))
    path_mask = np.memmap(out/"allowed_path_mask.u8", dtype="u1", mode="w+", shape=(count,))
    initial_status = np.memmap(out/"initial_projection_status.u8", dtype="u1", mode="w+", shape=(count,))
    initial_error = np.memmap(out/"initial_source_error.f32", dtype="<f4", mode="w+", shape=(count,))
    offsets = np.memmap(out/"source_id_offsets.u64", dtype="<u8", mode="w+" if resume is None else "r+", shape=(count+1,))
    offsets[0] = 0
    resume_aux_prefix={k:hashlib.sha256(buffers[k][:n0+reuse_rows].tobytes()).hexdigest() for k in AUX} if resume is not None else {}
    names, name_to_index, unknown, engineering = [], {}, defaultdict(list), defaultdict(list)
    by_shot, by_surface, branch_counts = Counter(), Counter(), Counter()
    status_counts = Counter()
    scene = NativeScene(as_event_view(model, "s01"))
    cameras = {sid: _camera(model, sid) for sid in shot_ids[1:]}
    consumed, batches, proof = 0, 0, hashlib.sha256()
    current, block = None, []
    def frame(rows):
        nonlocal consumed, batches
        sid, pts = rows[0]["shot_id"], int(rows[0]["native_pts"])
        uv = np.asarray([r["source_uv"] for r in rows], dtype="<f8")
        t = (pts - int(model["shots"][sid]["time_origin_pts"])) / 12800.
        lo, hi = n0+consumed, n0+consumed+len(rows)
        if consumed < reuse_rows:
            if consumed+len(rows)>reuse_rows or buffers["uv"][lo:hi].tobytes()!=uv.tobytes() or not np.all(buffers["native_pts"][lo:hi]==pts):
                raise ValueError("Resume boundary is not an unchanged complete source frame")
            branches=np.array(buffers["path"][lo:hi]);stations=np.array(buffers["station"][lo:hi])
            statuses=np.array(initial_status[consumed:consumed+len(rows)]);errors=np.array(initial_error[consumed:consumed+len(rows)],dtype=float)
        elif fast_seed:
            branches,stations,statuses,errors=_fast_seed_frame(scene,cameras[sid],t,rows)
        else:
            branches,stations,statuses,errors=_seed_frame(scene,cameras[sid],t,rows,tolerance_px=station_tolerance_px)
        vals = dict(uv=uv, native_pts=np.full(len(rows), pts), shot_index=np.full(len(rows), shot_ids.index(sid)),
            kind=np.zeros(len(rows)), material_index=[r["material_index"] for r in rows], path=branches,
            admission=np.ones(len(rows)), fit_role=np.ones(len(rows)), E=np.full(len(rows), engineering_E),
            station=stations, row=uv[:,1], alpha=np.zeros(len(rows)), source_sigma=np.ones(len(rows)),
            profile_normal=np.zeros((len(rows),2)), query_uv=uv, alpha_bounds=np.zeros((len(rows),2)),
            stroke_pair_index=np.full(len(rows),-1), stroke_source_side=np.full(len(rows),-1))
        for name in buffers:
            buffers[name][lo:hi] = vals[name]
        for j, r in enumerate(rows):
            at = consumed+j
            rid = r["source_row_id"]
            if r.get("point_correspondence") or r.get("query_available") or r.get("physical_point") or r["current_uv"] != r["source_uv"]:
                raise ValueError("Dense candidate became a fabricated fixed point/Query or changed source UV")
            allowed = list(map(int,r["allowed_paths"]))
            if not allowed or any(x<0 or x>=8 for x in allowed) or int(branches[j]) not in allowed:
                raise ValueError("Numerical dense branch outside actual source material candidates")
            raw = rid.encode("utf8")
            id_stream.write(raw); offsets[at+1] = id_stream.tell()
            ordinal[at] = int(r["source_global_ordinal"])
            family = r["source_family_id"]
            if family not in name_to_index:
                name_to_index[family] = len(names); names.append(family)
            family_index[at] = name_to_index[family]
            path_mask[at] = sum(1 << x for x in allowed)
            initial_status[at], initial_error[at] = statuses[j], errors[j]
            index = n0+at
            unknown[family].append(index)
            engineering[family+":native_exposure:"+str(pts)].append(index)
            by_shot[sid] += 1; by_surface[r["physical_surface"]] += 1
            branch_counts[(sid,r["joint_material_id"],int(branches[j]))] += 1
            status_counts[int(statuses[j])] += 1
            proof.update(struct.pack("<QqHddIII",int(ordinal[at]),pts,shot_ids.index(sid),
                float(uv[j,0]),float(uv[j,1]),int(r["material_index"]),int(branches[j]),len(raw)))
            proof.update(raw)
        consumed += len(rows); batches += 1
        if batches % 100 == 0:
            write_json(out/"progress.json", dict(frames=batches,appended_rows=consumed,total_rows=count,
                elapsed_seconds=time.perf_counter()-start,last_shot=sid,last_native_pts=pts))
    with (out/"source_id_utf8.u8").open("xb" if resume is None else "wb") as id_stream:
        for row in q.rows():
            key = row["shot_id"], int(row["native_pts"])
            if current is not None and key != current:
                frame(block); block = []
            current = key; block.append(row)
        if block:
            frame(block)
    if consumed != count:
        raise ValueError("Dense source input not fully consumed")
    q.assert_dependencies()
    for name, b in buffers.items():
        b.flush()
        if hashlib.sha256(b[:n0].tobytes()).hexdigest() != prefix[name]:
            raise ValueError("Dense append changed exact parent array prefix: "+name)
        arrays[name] = dict(**_ref(out/(name+".bin")),dtype=b.dtype.str,shape=list(b.shape),layout="C",byte_order="little")
    for b in (ordinal,family_index,path_mask,initial_status,initial_error,offsets):b.flush()
    def desc(name,dtype,shape):return dict(**_ref(out/name),dtype=dtype,shape=shape,layout="C",byte_order="little")
    registry = dict(schema=REGISTRY,row_start=n0,count=count,
        source_id_utf8=desc("source_id_utf8.u8","|u1",[(out/"source_id_utf8.u8").stat().st_size]),
        source_id_offsets=desc("source_id_offsets.u64","<u8",[count+1]),
        source_dataset_id="dense_native_source_20261021",source_family_ids=names,
        source_family_index=desc("source_family_index.u32","<u4",[count]),
        source_store_ordinal=desc("source_store_ordinal.u64","<u8",[count]),
        allowed_path_mask=desc("allowed_path_mask.u8","|u1",[count]),
        input_binding=_ref(ip),source_parts=q.binding["source_parts"],
        original_source_selection_IDs_retained=True,source_store_is_read_only=True)
    refs = [_ref(ip),_ref(mp),q.binding["material_map"],q.binding["eligibility_index"],
            q.binding["admission"],q.binding["reader"],_ref(__file__)]
    newgroups = [dict(id="dense_unknown_W:"+family,source_group_id=family,rows=ii,rank=0,W=[],
        statistic_kind="constraint_only_unknown_noise",unknown_noise_source_refs=refs[:1],
        source_noise_covariance_claimed=False,engineering_E_separate=True,
        time_connected_named_material_component=True) for family,ii in sorted(unknown.items())]
    dependency = deepcopy(p["engineering_pixel_dependency_groups"])
    dependency["groups"] += [dict(id=family,rows=ii)for family,ii in sorted(engineering.items())]
    material_binding = deepcopy(p["material_binding"])
    material_binding["source_material_map"] = q.binding["material_map"]
    material_binding["model_topology_ref"] = _ref(mp)
    oldactive = int(np.sum((buffers["admission"][:n0]==1)&(buffers["fit_role"][:n0]==1)))
    value = dict(p)
    value.update(row_count=N,shot_ids=shot_ids,arrays=arrays,
        dense_parent_packet=_ref(pp),dense_source_binding=registry,
        groups=p["groups"]+newgroups,active_rows=oldactive+count,admitted_rows=oldactive+count,
        recapture_rows=p.get("recapture_rows",[])+list(range(n0,N)),
        material_binding=material_binding,engineering_pixel_dependency_groups=dependency,
        source_manifests=p["source_manifests"]+refs,newly_appended_source_start=n0,
        constraint_only_unknown_noise_rows=int(p.get("constraint_only_unknown_noise_rows",0))+count,
        constraint_only_unknown_noise_groups=int(p.get("constraint_only_unknown_noise_groups",0))+len(newgroups),
        engineering_scope=dict(schema="rinkfit.engineering_active_shots.v1",shot_ids=shot_ids,three_rigs_one_scene=True),
        dense_joint_source_append=dict(schema="rinkfit.dense6_native_source_append.v1",parent_packet=_ref(pp),
            source_binding=_ref(ip),model_ref=_ref(mp),row_start=n0,count=count,
            actual_column_identity_sha256=proof.hexdigest(),source_noise_W_or_sigma_issued=False,
            new_source_E_is_engineering_not_sigma=True,source_sigma_suffix_is_unused_ABI_storage=True,
            new_Query_rows=0,new_fixed_world_point_rows=0,all_unselected_rows_retained_in_columnar_eligibility=True,
            source_UV_from_model=False,numerical_auxiliary_initialization_is_source_identity_truth=False,
            root_or_domain_invalid_rows_retained=True,actual_RS_roots_not_yet_certified=True))
    if dependency_aliases is not None:
        _checked(dependency_aliases,out)
        value["exact_dependency_aliases"]=dependency_aliases
        value["source_manifests"].append(dependency_aliases)
    # Historical scalar semantics remain inherited; new_start is in dedicated append.
    partition = np.zeros(N,dtype=np.int16)
    partition[np.asarray(value["independent_rows"],dtype=int)] += 1
    for g in value["groups"]:partition[np.asarray(g["rows"],dtype=int)] += 1
    if not np.array_equal(partition,((buffers["admission"]==1)&(buffers["fit_role"]==1)).astype(np.int16)):
        raise ValueError("Full old-W plus new-rank-zero source partition differs")
    write_packet(out/"packet.json",value)
    report = dict(schema="rinkfit.dense6_native_source_packet_build.v1",complete=True,packet=_ref(out/"packet.json"),
        parent_packet=_ref(pp),source_binding=_ref(ip),model_ref=_ref(mp),parent_rows=n0,new_rows=count,raw_rows=N,
        old_active_rows=oldactive,new_active_rows=count,active_rows=oldactive+count,
        parent_all_array_prefix_bytes_exact=True,parent_W_Query_IDs_exact=True,
        actual_column_identity_sha256=proof.hexdigest(),source_UV_ID_clock_preserved=True,
        new_source_statistics_rank=0,new_Query_rows=0,new_fixed_world_point_rows=0,
        new_unknown_W_components=len(newgroups),engineering_components=len(engineering),
        new_rows_by_shot=dict(by_shot),new_rows_by_surface=dict(by_surface),
        initialized_branch_counts=[dict(shot_id=k[0],feature_id=k[1],path_index=k[2],rows=v)for k,v in sorted(branch_counts.items())],
        initial_projection_status_counts=dict(status_counts),
        initial_projection_status_definition={"0":"no_finite_numerical_projection_kept_for_capture","1":"finite_local_station_initialization","2":"finite_station_stagnation_kept_for_capture"},
        initial_projection_status=desc("initial_projection_status.u8","|u1",[count]),
        initial_source_error=desc("initial_source_error.f32","<f4",[count]),
        initial_station_is_not_measured=True,initial_RS_root_certificate_issued=False,
        engineering_E=float(engineering_E),engineering_E_is_sigma=False,
        original_source_readings_regenerated=False,source_store_written=False,full_source_table_copy_written=False,
        solver_started=False,quality_accepted=False,elapsed_seconds=time.perf_counter()-start,
        resumed_unpublished_staging=resume is not None,resumed_completed_source_rows=reuse_rows,
        original_frame_progress=None if resume is None else resume["completed_frame_progress"],
        retained_completed_auxiliary_prefix_sha256=resume_aux_prefix,
        new_station_initialization="finite_projected_polyline_neighbor_interpolation" if fast_seed else "local_original_row_footpoint",
        original_source_pixel_decoding_repeated=False,
        source_reader_qualification=q.reader_qualification_status)
    write_json(out/"receipt.json",report)
    return report


def _assert_dense_source_identity_current(packet, *, return_verified_packet=False):
    """Verify the actual parent issuer and every appended actual source column."""
    from .io_utils import read_json
    from .dense_source_factory import open_dense_source_input
    path = Path(packet).resolve()
    doc = read_json(path)
    registry = doc.get("dense_source_binding")
    if not registry:
        raise ValueError("No dense native source suffix registry")
    parent_path = _checked(doc["dense_parent_packet"], path.parent)
    for folder in {path.parent,parent_path.parent}:
        for name in ("SOURCE_IDENTITY_CERTIFICATE_RECALLED.json","SOURCE_IDENTITY_EVIDENCE_RECALL.json"):
            if (folder/name).exists():
                raise ValueError("Dense/parent source identity is recalled: "+str(folder/name))
    # The selected parent is a capture-v4 wrapper; its real printed/board source
    # chain is verified by its actual versioned issuer, never a four-shot proxy.
    from .capture_primal_prefix_v4 import assert_capture_prefix_current
    parent_guard = assert_capture_prefix_current(parent_path)
    parent = read_json(parent_path)
    n0, count = int(registry["row_start"]),int(registry["count"])
    if parent["row_count"] != n0 or doc["row_count"] != n0+count:
        raise ValueError("Dense parent/source prefix denominator differs")
    for key in IDS:
        if doc[key][:n0] != parent[key]:
            raise ValueError("Dense source original parent IDs changed")
    for key in ("groups","queries"):
        if doc[key][:len(parent[key])] != parent[key]:
            raise ValueError("Dense source original W/Query prefix changed")
    if doc["independent_rows"] != parent["independent_rows"]:
        raise ValueError("Dense source changed original independent W members")
    arrays = {k:_array(v,path.parent)for k,v in doc["arrays"].items()}
    for k,ref in parent["arrays"].items():
        if k not in AUX and arrays[k][:n0].tobytes() != _array(ref,parent_path.parent).tobytes():
            raise ValueError("Dense source changed inherited raw source array: "+k)
    q = open_dense_source_input(_checked(registry["input_binding"],path.parent), purpose="historical_audit")
    q.assert_dependencies()
    if q.row_count != count or q.binding["parent_packet"] != _ref(parent_path):
        raise ValueError("Dense source actual eligibility denominator/parent differs")
    ordinals = _array(registry["source_store_ordinal"],path.parent,"<u8",[count])
    masks = _array(registry["allowed_path_mask"],path.parent,"|u1",[count])
    proof = hashlib.sha256()
    consumed = 0
    for j,r in enumerate(q.rows()):
        i=n0+j;uv=np.asarray(r["source_uv"],dtype="<f8");branch=int(arrays["path"][i])
        if (int(ordinals[j])!=r["source_global_ordinal"] or doc["source_row_ids"][i]!=r["source_row_id"]
            or doc["source_family_ids"][i]!=r["source_family_id"] or arrays["uv"][i].tobytes()!=uv.tobytes()
            or int(arrays["native_pts"][i])!=r["native_pts"] or doc["shot_ids"][int(arrays["shot_index"][i])]!=r["shot_id"]
            or int(arrays["material_index"][i])!=r["material_index"] or branch not in r["allowed_paths"]
            or int(masks[j])!=sum(1<<int(v)for v in r["allowed_paths"])
            or int(arrays["admission"][i])!=1 or int(arrays["fit_role"][i])!=1 or int(arrays["kind"][i])!=0):
            raise ValueError("Dense packet differs from actual eligible source at ordinal "+str(r["source_global_ordinal"]))
        raw=r["source_row_id"].encode("utf8")
        proof.update(struct.pack("<QqHddIII",int(ordinals[j]),r["native_pts"],int(arrays["shot_index"][i]),
            float(uv[0]),float(uv[1]),r["material_index"],branch,len(raw)));proof.update(raw)
        consumed += 1
    q.assert_dependencies()
    if consumed!=count or any(int(v["row_index"])>=n0 for v in doc["queries"]):
        raise ValueError("Dense source full denominator or absent Query changed")
    partition=np.zeros(count,dtype="u1")
    for g in doc["groups"][len(parent["groups"]):]:
        if g["rank"]!=0 or g["W"]!=[] or g["statistic_kind"]!="constraint_only_unknown_noise":
            raise ValueError("Dense source was assigned invented source noise precision")
        ii=np.asarray(g["rows"],dtype=int)-n0
        if np.any(ii<0)|np.any(ii>=count):
            raise ValueError("Dense statistical component crosses the parent prefix")
        partition[ii]+=1
    if not np.all(partition==1):
        raise ValueError("Dense unknown-noise dependency full partition differs")
    report=dict(schema="rinkfit.dense6_actual_source_identity_check.v1",passed=True,
        packet=_ref(path),parent_packet=_ref(parent_path),parent_guard=parent_guard,
        actual_input_binding=_ref(q.path),actual_rows=consumed,
        actual_column_identity_sha256=proof.hexdigest(),source_UV_ID_native_clock_exact=True,
        actual_full_stream_semantics_consumed=True,unknown_source_qualification_granted=False,
        source_reader_qualification=q.reader_qualification_status)
    return doc if return_verified_packet else report


def assert_dense_source_identity_current(packet, *, return_verified_packet=False):
    from .exact_dependency_aliases import dependency_alias_context
    path=Path(packet).resolve()
    raw=json.loads(path.read_text(encoding="utf-8-sig"))
    alias=raw.get("exact_dependency_aliases")
    with dependency_alias_context(alias) as context:
        result=_assert_dense_source_identity_current(path,return_verified_packet=return_verified_packet)
        if not return_verified_packet:
            result["exact_dependency_aliases"]=alias
            result["exact_archive_dependencies_used"]=list(context["used"])
        return result


def dense_packet_source_bindings(packet_path, row_indices):
    """Resolve only requested suffix rows to the original store stream/selection.

    The ordinal includes excluded source records, so feedback never treats a
    compact eligible-row index as an original per-stream row number.
    """
    from .io_utils import read_json
    path=Path(packet_path).resolve()
    doc=read_json(path)
    registry=doc["dense_source_binding"]
    start,count=int(registry["row_start"]),int(registry["count"])
    ordinals=_array(registry["source_store_ordinal"],path.parent,"<u8",[count])
    pts=_array(doc["arrays"]["native_pts"],path.parent)
    shots=_array(doc["arrays"]["shot_index"],path.parent)
    parts=registry["source_parts"]
    result={}
    for index in sorted(set(map(int,row_indices))):
        if index<start or index>=start+count:
            continue
        ordinal=int(ordinals[index-start])
        part=next((p for p in parts if int(p["ordinal_start"])<=ordinal<int(p["ordinal_start"])+int(p["row_count"])),None)
        if part is None:
            raise ValueError("Dense source ordinal is outside every original sealed stream")
        result[str(index)]=dict(shot_id=doc["shot_ids"][int(shots[index])],native_pts=int(pts[index]),
            selected_row_id=doc["source_row_ids"][index],source_row_id=doc["source_row_ids"][index],
            source_global_ordinal=ordinal,source_stream_ordinal=ordinal-int(part["ordinal_start"]),
            rows_ref=part["reference"])
    return result


def _fast_seed_frame(scene,camera,time0,rows):
    """Finite polyline-neighbor numerical seed; native capture solves true feet."""
    from scipy.spatial import cKDTree
    uv=np.asarray([r["source_uv"] for r in rows],dtype=float)
    mats=np.asarray([r["material_index"] for r in rows],dtype=np.uint32)
    branches=np.asarray([r["path_index"] for r in rows],dtype=np.uint32)
    station=np.full(len(rows),.5)
    status=np.zeros(len(rows),dtype=np.uint8)
    error=np.full(len(rows),np.inf)
    sampling=np.linspace(0.,1.,65)
    middle=time0+camera.tau*(float(np.median(uv[:,1]))/1079.-.5)
    for material in np.unique(mats):
        ii=np.flatnonzero(mats==material)
        sets={tuple(rows[i]["allowed_paths"]) for i in ii}
        for allowed in sets:
            jj=np.asarray([i for i in ii if tuple(rows[i]["allowed_paths"])==allowed],dtype=int)
            sample_uv=[];sample_s=[];sample_p=[];segment_a=[];segment_b=[];segment_s0=[];segment_s1=[];segment_p=[]
            for p in allowed:
                projection=camera.at(scene.world(int(material),int(p),sampling),middle)
                legal=projection["valid"]&np.isfinite(projection["uv"]).all(1)
                for k in np.flatnonzero(legal):
                    sample_uv.append(projection["uv"][k]);sample_s.append(sampling[k]);sample_p.append(p)
                for k in np.flatnonzero(legal[:-1]&legal[1:]):
                    segment_a.append(projection["uv"][k]);segment_b.append(projection["uv"][k+1])
                    segment_s0.append(sampling[k]);segment_s1.append(sampling[k+1]);segment_p.append(p)
            if not segment_a:
                continue
            a,b=np.asarray(segment_a),np.asarray(segment_b)
            # Nearest short segment midpoints locate only a numerical basin.
            # Four nearest candidates cover the immediate sampled neighbors.
            tree=cKDTree((a+b)*.5)
            _,near=tree.query(uv[jj],k=min(4,len(a)))
            near=np.asarray(near)
            if near.ndim==1:near=near[:,None]
            aa,bb=a[near],b[near]
            d=bb-aa;length=np.sum(d*d,axis=2)
            along=np.clip(np.sum((uv[jj,None,:]-aa)*d,axis=2)/np.maximum(length,np.finfo(float).tiny),0.,1.)
            candidate_s=np.asarray(segment_s0)[near]*(1.-along)+np.asarray(segment_s1)[near]*along
            candidate_p=np.asarray(segment_p,dtype=np.uint32)[near]
            flat_s=candidate_s.ravel();flat_p=candidate_p.ravel()
            world,legal,_=scene.world_rows(np.full(len(flat_s),material),flat_p,flat_s)
            times=np.broadcast_to((time0+camera.tau*(uv[jj,1]/1079.-.5))[:,None],candidate_s.shape).ravel()
            actual=camera.at(world,times)
            candidate_error=np.linalg.norm(actual["uv"].reshape(candidate_s.shape+(2,))-uv[jj,None,:],axis=2)
            valid=(legal&actual["valid"]).reshape(candidate_s.shape)&np.isfinite(candidate_error)
            candidate_error[~valid]=np.inf
            best=np.argmin(candidate_error,axis=1);good=np.isfinite(candidate_error[np.arange(len(jj)),best])
            hit=jj[good];choice=best[good]
            branches[hit]=candidate_p[good,choice];station[hit]=candidate_s[good,choice]
            error[hit]=candidate_error[good,choice];status[hit]=1
    return branches,station,status,error
