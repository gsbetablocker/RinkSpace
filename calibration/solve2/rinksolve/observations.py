"""Current S01 source observations; no legacy solver imports or execution.

Raw UV, native PTS and original source IDs remain immutable. Current delivered
qualification and statistical support are retained separately from new semantic
crosswalks. Loading a delivered generation never opens the old packet.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from . import schema as S
from .arrays import ArrayStore

FIRST_PTS, LAST_PTS, STEP = 256256, 575488, 256
SOURCE_ID, CLOCK_ID = "O6j8L8TmfKo", "O6j8L8TmfKo.native_clock"
IMPORT_SCHEMA = "rinksolve.s01_observations.v1"


def observation_id(dataset: str, pts: int, source_id: str) -> str:
    """Reversible acquisition identity, retaining every original ID character."""
    if not dataset or not source_id or type(pts) is not int:
        raise ValueError("nonempty original dataset/source ID and integer native PTS required")
    return f"raw:{len(dataset)}:{dataset}:{pts}:{source_id}"


def split_observation_id(value: str) -> tuple[str, int, str]:
    if not value.startswith("raw:"):
        raise ValueError("unknown observation identity")
    length, body = value[4:].split(":", 1)
    n = int(length)
    dataset = body[:n]
    if n < 1 or body[n:n+1] != ":":
        raise ValueError("invalid dataset length")
    pts, source_id = body[n+1:].split(":", 1)
    if not source_id:
        raise ValueError("missing original source ID")
    return dataset, int(pts), source_id


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path, value, *, compact=False):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=None if compact else 2, separators=(",", ":") if compact else None), encoding="utf-8")
    os.replace(temp, path)


def _strings(values):
    values = [v.encode("utf-8") if isinstance(v, str) else bytes(v) for v in values]
    return np.asarray(values, dtype=f"S{max((len(v) for v in values), default=1)}")


def _str(value):
    return str(value) if isinstance(value, (str, np.str_)) else bytes(value).decode("utf-8", "strict")


def _array(desc, base):
    path = Path(desc.get("path") or base / desc["file"])
    shape, dtype = tuple(desc["shape"]), np.dtype(desc["dtype"])
    offset = int(desc.get("offset_bytes", 0))
    if dtype.hasobject or dtype.kind not in "biufSU" or offset < 0:
        raise ValueError(f"invalid typed array {path}")
    end = offset + int(np.prod(shape, dtype=object)) * dtype.itemsize
    if end > path.stat().st_size:
        raise ValueError(f"array range exceeds current file: {path}")
    return np.memmap(path, mode="r", dtype=dtype, shape=shape, offset=offset)


def _absolute(value, base):
    if isinstance(value, dict):
        result = {k: _absolute(v, base) for k, v in value.items()}
        if "file" in result:
            result["path"] = str((base / result.pop("file")).resolve())
        return result
    if isinstance(value, list):
        return [_absolute(v, base) for v in value]
    return value


def _export_prefix(packet, output):
    result = {"row_start": 0, "count": len(packet["source_row_ids"]), "dataset_is_column": True}
    for name, key in (("source_id", "source_row_ids"), ("source_family", "source_family_ids"),
                      ("source_dataset", "source_dataset_ids")):
        values = packet[key]
        if len(values) != result["count"]:
            raise ValueError("prefix identity columns have unequal lengths")
        # Null is a genuine missing source family; its bitmap remains explicit.
        encoded = [b"" if v is None else str(v).encode("utf-8") for v in values]
        offsets = np.zeros(len(values) + 1, dtype="<u8")
        offsets[1:] = np.cumsum(np.asarray([len(v) for v in encoded], dtype="<u8"))
        blob = output / (name + "_prefix_utf8.bin")
        off = output / (name + "_prefix_offsets.npy")
        nulls = output / (name + "_prefix_null.npy")
        blob.write_bytes(b"".join(encoded))
        np.save(off, offsets, allow_pickle=False)
        np.save(nulls, np.asarray([v is None for v in values], dtype="|u1"), allow_pickle=False)
        result[name] = {"utf8_path": str(blob), "offsets_path": str(off), "null_path": str(nulls),
                        "utf8_sha256": _sha(blob), "offsets_sha256": _sha(off), "null_sha256": _sha(nulls)}
    return result


def _segment_values(segment, rows, base):
    local = rows - int(segment["row_start"])
    if local.size and (local.min() < 0 or local.max() >= segment["count"]):
        raise ValueError("source rows outside registry segment")
    def decode(blob, offsets):
        if len(offsets) != segment["count"] + 1 or int(offsets[0]) != 0 or int(offsets[-1]) != len(blob) or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("corrupt source ID UTF-8 offsets")
        return [blob[int(offsets[i]):int(offsets[i+1])].tobytes().decode("utf-8", "strict") for i in local]
    if segment.get("dataset_is_column"):
        columns = []
        for name in ("source_id", "source_family", "source_dataset"):
            d = segment[name]
            blob = np.memmap(d["utf8_path"], dtype="u1", mode="r")
            offsets = np.load(d["offsets_path"], mmap_mode="r", allow_pickle=False)
            values = decode(blob, offsets)
            nulls = np.load(d["null_path"], mmap_mode="r", allow_pickle=False)[local]
            columns.append([None if missing else value for value, missing in zip(values, nulls)])
        return tuple(columns)
    ids = decode(_array(segment["source_id_utf8"], base), _array(segment["source_id_offsets"], base))
    indices = _array(segment["source_family_index"], base)[local]
    if indices.size and indices.max() >= len(segment["source_family_ids"]):
        raise ValueError("source family foreign key outside current registry")
    return ids, [segment["source_family_ids"][int(i)] for i in indices], [segment["source_dataset_id"]] * len(ids)


@dataclass(frozen=True)
class ObservationImport:
    bundle: S.DataBundle
    selections: tuple[S.SelectionRef, ...]
    report: dict
    manifest_path: Path


def load_import(path):
    """Open only self-contained current delivery; no historical chain replay."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("schema") != IMPORT_SCHEMA:
        raise ValueError("unknown observation import schema")
    bundle = S.loads(Path(d["bundle_path"]).read_text(encoding="utf-8"), S.DataBundle)
    return ObservationImport(bundle, tuple(S.from_json_value(v, S.SelectionRef) for v in d["selections"]), d["report"], path)


def _support(packet, rows, base):
    """Preserve current W/Query qualification and dependence, not their old code."""
    selected = set(map(int, rows))
    groups = []
    checked_w = 0
    partial = []
    for group in packet["groups"]:
        overlap = selected.intersection(group["rows"])
        if not overlap:
            continue
        groups.append(group)
        if len(overlap) != len(group["rows"]):
            partial.append(group["id"])
        w = _array(group["W"], base) if isinstance(group["W"], dict) else np.asarray(group["W"], dtype="<f8")
        if w.size == 0 and group["rank"] == 0:
            # Explicit unknown-noise constraint-only group, not identity W.
            w = w.reshape(0, 2 * len(group["rows"]))
        if not np.all(np.isfinite(w)) or w.ndim != 2 or w.shape[1] != 2 * len(group["rows"]) or w.shape[0] != group["rank"]:
            raise ValueError(f"current W/rank/rows inconsistent in {group['id']}")
        checked_w += w.size
    queries = [q for q in packet["queries"] if int(q["row_index"]) in selected]
    for q in queries:
        if np.asarray(q["native_uv"]).shape != (len(q["alpha"]), 2):
            raise ValueError("current Query alpha/native_uv mapping differs")
    # Keep full small descriptors and only row-keyed current selections where possible.
    row_map = {str(int(row)): int(i) for i, row in enumerate(rows)}
    allowed = {k: v for k, v in packet.get("allowed_paths", {}).items() if k in row_map}
    return _absolute({"schema": "rinksolve.current_support_snapshot.v1",
        "source_row_mapping": {"table_id":"s01.raw","column":"original_row","lookup":"searchsorted on persisted ascending original_row; value is current SoA row, never semantic identity"}, "groups": groups, "queries": queries,
        "independent_rows": [r for r in packet["independent_rows"] if r in selected],
        "independent_statistic_kind": packet.get("independent_statistic_kind"),
        "allowed_paths": allowed, "source_station_domains": packet.get("source_station_domains"),
        "engineering_pixel_dependency_groups": {**packet.get("engineering_pixel_dependency_groups", {}), "groups":[g for g in packet.get("engineering_pixel_dependency_groups", {}).get("groups", []) if selected.intersection(g["rows"])]},
        "current_print_path_overrides": {k:v for k,v in packet.get("current_print_path_overrides", {}).items() if k in row_map},
        "current_withheld_rows": [r for r in packet.get("current_withheld_rows", []) if r in selected],
        "current_model_subset_v1": packet.get("current_model_subset_v1"),
        "qualification_semantics": "继承当前交付的来源/人工确认和同源统计；不是旧labels/path/数值邻近自动变成新实体确认。",
        "partial_group_ids": partial, "checked_W_values": checked_w}, base)


def import_current(descriptors_path, intersections_path, output, *, data_root):
    start_time = time.perf_counter()
    output, data_root = Path(output).resolve(), Path(data_root).resolve()
    output.relative_to(data_root)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("delivery already exists; use a new generation directory")
    d = json.loads(Path(descriptors_path).read_text(encoding="utf-8"))
    base = Path(d["packet_path"]).parent
    current = {k: _array(v, base) for k, v in d["arrays"].items()}
    n = len(current["native_pts"])
    if any(a.shape[0] != n for a in current.values()):
        raise ValueError("current source columns differ in length")
    pts = current["native_pts"]
    in_time = (pts >= FIRST_PTS) & (pts <= LAST_PTS)
    in_shot = current["shot_index"] == d["shot_ids"].index("s01")
    if np.any(in_time != in_shot):
        raise ValueError("S01 native PTS/shot_index actual values disagree")
    rows = np.flatnonzero(in_time).astype("<u8")
    selected_pts = pts[rows]
    frame_pts = np.arange(FIRST_PTS, LAST_PTS + STEP, STEP, dtype="<i8")
    if np.any((selected_pts-FIRST_PTS) % STEP) or not np.array_equal(np.unique(selected_pts), frame_pts):
        raise ValueError("S01 exact native 1248-exposure clock not covered")
    # One explicit data migration, not a normal startup requirement.
    packet = json.loads(Path(d["packet_path"]).read_text(encoding="utf-8"))
    prefix = _export_prefix(packet, output)
    support = _support(packet, rows, base)
    del packet
    segments = [prefix, d["dense_source_binding"], *d["native_ad_source_bindings"]]
    original_ids, family_ids, dataset_ids = ([None] * len(rows) for _ in range(3))
    cover = np.zeros(len(rows), dtype="u1")
    segment_counts = []
    for segment in segments:
        lower, upper = segment["row_start"], segment["row_start"] + segment["count"]
        positions = np.flatnonzero((rows >= lower) & (rows < upper))
        if not len(positions):
            continue
        ids, families, datasets = _segment_values(segment, rows[positions].astype(np.int64), base)
        for at, sid, family, dataset in zip(positions, ids, families, datasets):
            original_ids[at], family_ids[at], dataset_ids[at] = sid, family, dataset
        cover[positions] += 1
        segment_counts.append({"original_row_start": lower, "original_segment_count": segment["count"],
                               "selected_s01": len(positions), "source_dataset": segment.get("source_dataset_id", "per_row_original")})
    if np.any(cover != 1):
        raise ValueError("each selected original source ID must have exactly one current registry segment")
    stable_ids = _strings([observation_id(ds, int(p), sid) for ds, p, sid in zip(dataset_ids, selected_pts, original_ids)])
    if np.unique(stable_ids).size != len(stable_ids):
        raise ValueError("duplicate original dataset/native-PTS/source ID; explicit source disambiguation required")
    review = json.loads(Path(intersections_path).read_text(encoding="utf-8"))
    if review["source"]["sha256"] != d["source_clock"]["source_video_sha256"]:
        raise ValueError("intersection source differs from current native video")
    evidence = (
        S.Evidence("source.video", S.EvidenceKind.NATIVE_VIDEO, review["source"]["path"], "官方ISU 50fps单节目原片，native PTS时钟。", review["source"]["sha256"]),
        S.Evidence("input.current", S.EvidenceKind.COMPUTATION, str(Path(descriptors_path).resolve()), "当前交付原列与继承来源资格；不重放未变来源链。", _sha(descriptors_path)),
        S.Evidence("review.intersections", S.EvidenceKind.HUMAN_ANNOTATION, str(Path(intersections_path).resolve()), "静止材料及交点的原片审核；全局lane/house身份候选不据旧投影自动绑定。", _sha(intersections_path)))
    frame_ids = _strings([f"{SOURCE_ID}:{int(p)}" for p in frame_pts])
    frames = tuple(S.SourceFrame(_str(fid), CLOCK_ID, int(p), 1920, 1080, "source.video") for fid, p in zip(frame_ids, frame_pts))
    arrays, tables, indices = [], [], []
    unitless, pixel, native = S.UnitRef(S.Unit.DIMENSIONLESS), S.UnitRef(S.Unit.PIXEL, SOURCE_ID), S.UnitRef(S.Unit.NATIVE_PTS, CLOCK_ID)
    def save(name, values, unit, role):
        values = np.ascontiguousarray(values)
        if values.dtype.hasobject:
            raise ValueError("object/pickle arrays forbidden")
        path = output / (name.replace(".", "_") + ".npy")
        np.save(path, values, allow_pickle=False)
        ref = S.TypedArrayRef("array."+name, str(path), "npy", values.dtype.str, values.shape, unit, role, True, 1,
                              clock_id=CLOCK_ID if unit.unit is S.Unit.NATIVE_PTS else None, sha256=_sha(path))
        arrays.append(ref)
        return ref.id
    def add_table(name, data, primary, role, fkeys=()):
        columns = tuple(S.ColumnRef(key, save(name+"."+key, value, unit, role), meaning, text)
                        for key, value, meaning, text, unit in data)
        key = next(v for k, v, *_ in data if k == primary)
        order = np.argsort(key, kind="stable").astype("<u8")
        ixid = "index."+name
        indices.append(S.IdIndexRef(ixid, name, save(name+".sorted_ids", key[order], unitless, S.ArrayRole.SEMANTIC_FOREIGN_KEYS),
                        save(name+".sorted_rows", order, unitless, S.ArrayRole.SEMANTIC_FOREIGN_KEYS), 1,
                        "UTF-8 byte sort; exact searchsorted; duplicate IDs rejected."))
        tables.append(S.ColumnTableRef(name, len(key), 1, role, columns, primary, ixid, tuple(fkeys), ("input.current",)))
    add_table("s01.frames", [("frame_id",frame_ids,S.ColumnMeaning.STABLE_ID,"原片ID与native PTS构成的曝光身份。",unitless),
              ("native_pts",frame_pts,S.ColumnMeaning.NATIVE_PTS,"原生曝光时钟，不是导出序号。",native)], "frame_id", S.ArrayRole.IMMUTABLE_EVIDENCE)
    uv = current["uv"][rows]
    raw_data = [
        ("observation_id",stable_ids,S.ColumnMeaning.STABLE_ID,"可逆dataset/nativePTS/原sourceID；不使用row/hash作为观测身份。",unitless),
        ("source_id",_strings(original_ids),S.ColumnMeaning.OTHER,"原source ID UTF-8原文。",unitless),
        ("source_dataset_id",_strings(dataset_ids),S.ColumnMeaning.OTHER,"原dataset身份。",unitless),
        ("source_family_id",_strings(["" if f is None else f for f in family_ids]),S.ColumnMeaning.OTHER,"原source族，缺失由family_missing标记。",unitless),
        ("family_missing",np.asarray([f is None for f in family_ids],dtype="u1"),S.ColumnMeaning.OTHER,"原记录null保持缺失，不改写为字符串None。",unitless),
        ("original_row",rows,S.ColumnMeaning.OTHER,"当前交付原地址用于追查，不作为实体/观测身份。",unitless),
        ("native_pts",selected_pts,S.ColumnMeaning.NATIVE_PTS,"不变整数native PTS。",native),
        ("frame_id",frame_ids[((selected_pts-FIRST_PTS)//STEP).astype(np.int64)],S.ColumnMeaning.SOURCE_FRAME_ID,"原曝光外键。",unitless),
        ("u",uv[:,0],S.ColumnMeaning.RAW_U,"当前原始uv，prefix query_uv缺失零不当原像素。",pixel),
        ("v",uv[:,1],S.ColumnMeaning.RAW_V,"当前原始uv，只读。",pixel)]
    add_table("s01.raw",raw_data,"observation_id",S.ArrayRole.IMMUTABLE_EVIDENCE,
              (S.ColumnForeignKey("frame_id","s01.frames","frame_id"),))
    context = [("context_id",_strings(["context:"+_str(x) for x in stable_ids]),S.ColumnMeaning.STABLE_ID,"保留当前交付历史上下文的稳定记录。",unitless),
               ("observation_id",stable_ids,S.ColumnMeaning.OBSERVATION_ID,"原像素观测稳定外键。",unitless),
               ("use",_strings(["inactive"]*len(rows)),S.ColumnMeaning.ASSOCIATION_USE,"新跨表关联未组装；不撤销当前原来源确认，也不自动启用旧label/path。",unitless)]
    for key in current:
        if key in ("uv","native_pts","shot_index"):
            continue
        unit = pixel if key in ("source_sigma","row","query_uv") else unitless
        context.append(("legacy_"+key,current[key][rows],S.ColumnMeaning.OTHER,
                       "当前交付原值保留；旧数值导航/资格/工程权重与新语义实体crosswalk分开。",unit))
    add_table("s01.current_context",context,"context_id",S.ArrayRole.SEMANTIC_FOREIGN_KEYS,
              (S.ColumnForeignKey("observation_id","s01.raw","observation_id"),))
    selection_array = save("s01.selection_all",np.arange(len(rows),dtype="<u8"),unitless,S.ArrayRole.SEMANTIC_FOREIGN_KEYS)
    selections = (S.SelectionRef("selection.s01.raw_all","s01.raw",1,S.SelectionMode.ROW_INDEX,
                                 row_index_array_id=selection_array,selected_count=len(rows)),)
    raw_small, midpoint, grouped, parent_exclusions = [], [], {}, []
    for group in review["evidence_groups"]:
        refs = []
        if group.get("raw_edge_pairs"):
            for i, pair in enumerate(group["raw_edge_pairs"]):
                left, right, mid = (group["id"]+f".{i}.{part}" for part in ("left","right","midpoint"))
                for identity, key in ((left,"uv_left"),(right,"uv_right")):
                    raw_small.append(S.PixelObservation(identity,group["source_frame_id"],tuple(map(float,pair[key])),
                        ("source.video","review.intersections"),"manual_search_material_chroma_half_level"))
                midpoint.append(S.DerivedPixelObservation(mid,group["source_frame_id"],tuple(map(float,pair["uv_mid"])),
                    "material_width_midpoint.v1",(left,right),("review.intersections",)))
                refs.append(mid)
        else:
            for i, sample in enumerate(group["samples"]):
                identity = group["id"]+f".sample{i}"
                raw_small.append(S.PixelObservation(identity,group["source_frame_id"],tuple(map(float,sample["uv_mid"])),
                    ("source.video","review.intersections"),"manual_search_paint_trough_parabolic_subpixel"))
                if sample.get("hit_search_boundary"):
                    parent_exclusions.append({"observation_id":identity,"reason":"原材料搜索带命中边界，既有测量计算已排除；保留原像素，不参与交点支持。"})
                else:
                    refs.append(identity)
        grouped[group["id"]] = tuple(refs)
    intersections, associations = [], []
    for obs in review["observations"]:
        parents = tuple(p for group in obs["parent_evidence_group_ids"] for p in grouped[group])
        intersections.append(S.DerivedPixelObservation(obs["id"],obs["source_frame_id"],tuple(map(float,obs["uv_native_pixels"])),
            "image_intersection_of_measured_material_curves.v1",parents,("source.video","review.intersections")))
        associations.append(S.ObservationAssociation("association."+obs["id"],obs["id"],obs["fixed_feature_id"],S.AssociationUse.INACTIVE,
            ("review.intersections",),reason="固定材料交点保留；待新数值几何和共享原始支持联合统计模型准备，非每帧可滑点。"))
    world_id, entity_id, verify_id = "ankara_2026.world.pending", "ice.static_material", "verification.s01.ice"
    scene_unit = S.UnitRef(S.Unit.SCENE_UNIT,world_id)
    coordinates = S.CoordinateSystem(world_id,S.CoordinateKind.WORLD,3,("x","y","z"),(scene_unit,scene_unit,scene_unit),
        "当前只是待绑定的共同场景坐标登记；未提供世界XY、尺度或新数值几何。")
    surface_id = "geometry.s01.ice_surface.uninstantiated"
    primitives = [S.GeometryPrimitive(surface_id,entity_id,world_id,S.RepresentationKind.MODEL,"pending.new_ice_surface",())]
    curves = []
    for curve in review["source_curve_entities"]:
        pid = "geometry."+curve["id"]
        primitives.append(S.GeometryPrimitive(pid,entity_id,world_id,S.RepresentationKind.MODEL,"pending.new_shared_material_line",()))
        curves.append(S.CurveFeature(curve["id"],entity_id,pid,surface_id,S.CurveMeaning.MATERIAL_CENTERLINE,
                      verify_id,S.Interval(None,None,scene_unit),("review.intersections",)))
    landmarks = tuple(S.LandmarkFeature(f["id"],entity_id,verify_id,f["id"],
        S.IntersectionPosition(tuple(f["parent_curve_ids"]),f["id"]+".unique_crossing",f["intersection_branch"]["description"],
        ("review.intersections",))) for f in review["fixed_features"])
    bundle = S.DataBundle("s01.current_observations.g1",S.RevisionStamp(1,0,0,0),"s01",
        array_refs=tuple(arrays),column_tables=tuple(tables),id_indices=tuple(indices),evidence=evidence,
        clocks=(S.SourceClock(CLOCK_ID,1,12800,"source.video"),),frames=frames,raw_observations=tuple(raw_small),
        derived_observations=tuple(midpoint+intersections),coordinate_systems=(coordinates,),scenes=(S.Scene("ankara_2026.scene",world_id),),
        rigs=tuple(S.CameraRig(r,"ankara_2026.scene","用户登记三独立物理机位；本包不虚构安装坐标。",("input.current",)) for r in ("main","secondary","end")),
        shots=(S.Shot("s01","main",CLOCK_ID,tuple(f.id for f in frames)),),
        entities=(S.GeometryEntity(entity_id,"ankara_2026.scene","本次已核实静止冰面材料",S.MaterialScope.ICE_BODY),),
        verifications=(S.EntityVerification(verify_id,entity_id,S.VerificationStatus.CONFIRMED,S.MaterialScope.ICE_BODY,True,
            ("source.video","review.intersections"),"本次六条材料线及交叉的原片确认；其余既有确认保留在当前交付support，需stable crosswalk继承。"),),
        primitives=tuple(primitives),curves=tuple(curves),landmarks=landmarks,associations=tuple(associations))
    S.validate_bundle(bundle)
    issues = validate_batches(None,bundle,data_root=data_root,selections=selections)
    if issues:
        raise S.ContractError(issues)
    report = {"status":"raw_current_input_delivered_not_prepared_problem","prepared_problem":False,
        "original_raw_row_count":n,"s01_raw_rows":len(rows),"s01_native_exposures":len(frames),
        "source_registry_segments":segment_counts,"native_pts_shot_index_mismatches":0,
        "original_source_family_nulls":sum(f is None for f in family_ids),
        "preserved_current_groups":len(support["groups"]),"preserved_current_queries":len(support["queries"]),
        "preserved_independent_rows":len(support["independent_rows"]),"checked_current_W_values":support["checked_W_values"],
        "preserved_engineering_dependency_groups":len(support["engineering_pixel_dependency_groups"]["groups"]),
        "preserved_print_path_overrides":len(support["current_print_path_overrides"]),
        "preserved_withheld_rows":len(support["current_withheld_rows"]),
        "cross_shot_partial_groups":support["partial_group_ids"],"retained_parent_raw_pixels":len(raw_small),
        "excluded_parent_raw_pixels":len(parent_exclusions),
        "derived_material_midpoints":len(midpoint),"excluded_parent_measurements":parent_exclusions,"fixed_intersection_entities":len(landmarks),
        "fixed_intersection_exposure_observations":len(intersections),"new_fit_associations_enabled":0,
        "legacy_numeric_unit_status":{"alpha":"opaque legacy scalar; resolve Query/kind parameter before factors","alpha_bounds":"opaque legacy range; not new physical station domain","E":"preserved current budget scalar; unit must be resolved by new kernel","station":"old numerical navigation only; not a semantic fixed point"},
        "covariance":"unknown remains None; review precision estimates are not standard deviations",
        "world_position":"unknown shared material intersection; no per-frame sliding station",
        "ordinary_load_requires_old_packet":False,"raw_source_arrays_modified":False,
        "pending":["继承当前已确认来源到新stable Entity/Feature的crosswalk；旧未知/错点单独修",
                   "当前W/Query的新的联合残差与导数核；原支持完整保留，禁止默认为独立同权",
                   "新场景几何数值/原生投影契约/三机位安装/初值绑定",
                   "固定交点几何分支存在唯一性与共用C父样本联合计数",
                   "新PreparedProblem、CPU/GPU布局及完整前向/导数验证"],
        "value_validation":"read actual columns, primary keys, sorted indices, FK values, selected rows and native frames",
        "import_seconds":time.perf_counter()-start_time}
    _json(output/"bundle.json",S.to_json_value(bundle))
    _json(output/"current_support.json",support,compact=True)
    _json(output/"source_registry.json",{"schema":"rinksolve.original_source_registry.v1","raw_row_count":n,
        "segments":_absolute(segments,base),"legacy_source_material_binding":d["source_material_binding"],
        "legacy_material_binding":d["material_binding"]})
    _json(output/"manifest.json",{"schema":IMPORT_SCHEMA,"bundle_path":str(output/"bundle.json"),
        "current_support_path":str(output/"current_support.json"),"source_registry_path":str(output/"source_registry.json"),
        "selections":[S.to_json_value(s) for s in selections],"report":report})
    return load_import(output)


def validate_batches(problem, bundle, *, data_root=None, store=None, selections=()):
    """Value-level validation hook for schema.validate_prepared, never a fake pass."""
    issues = list(S.validate_array_contracts(bundle,raise_on_error=False))
    if problem is not None:
        issues.extend(S.validate_batch_plan(problem,bundle))
    def bad(path,code,message):
        issues.append(S.Issue(path,code,message))
    if store is None:
        if data_root is None:
            bad("array_refs","data_root_missing","实际数组检查需要明确Data根目录或ArrayStore。")
            return tuple(issues)
        store = ArrayStore(data_root)
    refs = {a.id:a for a in bundle.array_refs}
    loaded = {}
    for aid, ref in refs.items():
        try:
            loaded[aid] = store.open(ref)
        except (OSError,ValueError) as exc:
            bad("array_refs/"+aid,"current_array_read",str(exc))
    tables = {t.id:t for t in bundle.column_tables}
    indices = {i.id:i for i in bundle.id_indices}
    columns = {}
    for tid, table in tables.items():
        cols = {c.name:loaded[c.array_id] for c in table.columns if c.array_id in loaded}
        columns[tid] = cols
        for col in table.columns:
            a = cols.get(col.name)
            if a is not None and col.meaning in (S.ColumnMeaning.RAW_U,S.ColumnMeaning.RAW_V):
                if a.ndim != 1 or not np.all(np.isfinite(a)):
                    bad("column_tables/"+tid+"/"+col.name,"raw_pixel_values","原始像素必须一维有限，不能用零补缺失。")
        key = cols.get(table.primary_key_column)
        if key is None:
            continue
        if key.ndim != 1 or key.dtype.kind not in "SU":
            bad("column_tables/"+tid,"stable_id_values","稳定主键须是一维非空字符串，row不是身份。")
            continue
        if np.any(key == (b"" if key.dtype.kind=="S" else "")) or np.unique(key).size != table.row_count:
            bad("column_tables/"+tid,"duplicate_or_empty_primary_key","实际主键为空或重复。")
        ix = indices.get(table.id_index_id)
        if ix is None or ix.key_array_id not in loaded or ix.row_array_id not in loaded:
            continue
        keys, rows = loaded[ix.key_array_id], loaded[ix.row_array_id]
        if rows.ndim != 1 or rows.dtype.kind not in "iu" or keys.ndim != 1:
            bad("id_indices/"+ix.id,"id_index_values","索引实际列须为一维字符串key及整数row。")
        elif rows.size != table.row_count or (rows.size and (rows.min()<0 or rows.max()>=table.row_count)) or np.unique(rows).size != table.row_count:
            bad("id_indices/"+ix.id,"id_index_row_permutation","实际索引row必须恰好覆盖当前表一次。")
        elif keys.size != rows.size or not np.array_equal(keys,key[rows]):
            bad("id_indices/"+ix.id,"id_index_key_mapping","实际索引key与对应row主键不同。")
        elif keys.size > 1 and np.any(keys[1:] <= keys[:-1]):
            bad("id_indices/"+ix.id,"id_index_sort","实际稳定ID索引必须严格有序。")
    for tid, table in tables.items():
        for fk in table.foreign_keys:
            src = columns.get(tid,{}).get(fk.column_name)
            target = columns.get(fk.target_table_id,{}).get(fk.target_column_name)
            if src is None or target is None:
                continue
            if src.ndim != 1 or target.ndim != 1 or src.dtype.kind != target.dtype.kind:
                bad("column_tables/"+tid+"/"+fk.column_name,"foreign_key_dtype","实际外键和目标稳定ID类型不一致。")
            elif not np.all(np.isin(src,target)):
                bad("column_tables/"+tid+"/"+fk.column_name,"foreign_key_values","实际外键含当前目标主键不存在的身份。")
    frame_lookup = {f.id.encode("utf-8"):(f.native_pts,f.clock_id) for f in bundle.frames}
    for tid, table in tables.items():
        for pcol in (c for c in table.columns if c.meaning is S.ColumnMeaning.NATIVE_PTS):
            pts = columns[tid].get(pcol.name)
            if pts is None:
                continue
            clock = refs[pcol.array_id].clock_id
            allowed = np.asarray([f.native_pts for f in bundle.frames if f.clock_id==clock],dtype="<i8")
            if pts.ndim != 1 or pts.dtype.kind not in "iu" or not np.all(np.isin(pts,allowed)):
                bad("column_tables/"+tid+"/"+pcol.name,"native_exposure_values","实际PTS须为该源时钟已登记的整数原生曝光。")
            for fcol in (c for c in table.columns if c.meaning is S.ColumnMeaning.SOURCE_FRAME_ID):
                fids = columns[tid].get(fcol.name)
                if fids is not None and any(frame_lookup.get(_str(fid).encode("utf-8")) != (int(p),clock) for fid,p in zip(fids,pts)):
                    bad("column_tables/"+tid,"frame_pts_values","原生PTS实际值与稳定frame外键不一致。")
    wanted = list(selections)
    if problem is not None:
        wanted += [v.owner_selection for v in bundle.batched_variables]
        wanted += [f.association_selection for f in problem.factors if f.association_selection]
        wanted += [n.observation_selection for n in problem.noise_families if n.observation_selection]
    for selection in wanted:
        path = "selections/"+selection.id
        table = tables.get(selection.table_id)
        if table is None or table.generation != selection.generation:
            bad(path,"selection_generation","选择不属于当前表generation。")
            continue
        if selection.mode is S.SelectionMode.SMALL_IDS:
            if len(set(selection.ids)) != len(selection.ids):
                bad(path,"selection_duplicate_ids","实际选择重复稳定ID会重复计权。")
            if selection.row_index_array_id is not None or selection.row_mask_array_id is not None:
                bad(path,"selection_union","小ID选择不能同时携带row index或mask。")
            key = columns.get(table.id,{}).get(table.primary_key_column)
            if key is not None and not np.all(np.isin(_strings(selection.ids) if key.dtype.kind=="S" else np.asarray(selection.ids,dtype=key.dtype),key)):
                bad(path,"selection_id_values","实际选择含不存在的稳定ID。")
            actual = len(selection.ids)
        else:
            aid = selection.row_index_array_id if selection.mode is S.SelectionMode.ROW_INDEX else selection.row_mask_array_id
            a = loaded.get(aid)
            if a is None:
                bad(path,"selection_array_missing","实际选择数组不能读取。")
                continue
            if refs[aid].generation != selection.generation:
                bad(path,"selection_array_generation","实际选择数组generation不同。")
            if selection.mode is S.SelectionMode.ROW_INDEX:
                if a.ndim != 1 or a.dtype.kind not in "iu" or (a.size and (a.min()<0 or a.max()>=table.row_count)):
                    bad(path,"selection_row_values","实际row选择须为当前范围内的一维整数。")
                elif np.unique(a).size != a.size:
                    bad(path,"selection_duplicate_rows","同一证据重复选择会重复计权。")
                actual = int(a.size)
            else:
                if a.shape != (table.row_count,) or a.dtype.kind not in "bu" or not np.all((a==0)|(a==1)):
                    bad(path,"selection_mask_values","实际选择mask须等长且只有0/1。")
                actual = int(np.count_nonzero(a))
        if selection.selected_count is not None and actual != selection.selected_count:
            bad(path,"selection_actual_count",f"实际选择{actual}，声明{selection.selected_count}。")
    # Explicit source identity values must survive the new ID envelope exactly.
    for tid, cols in columns.items():
        if {"observation_id","source_id","source_dataset_id","native_pts"} <= cols.keys():
            wrong = False
            try:
                for oid, sid, ds, pts in zip(cols["observation_id"],cols["source_id"],cols["source_dataset_id"],cols["native_pts"]):
                    if split_observation_id(_str(oid)) != (_str(ds),int(pts),_str(sid)):
                        wrong = True
                        break
            except (ValueError,UnicodeError):
                wrong = True
            if wrong:
                bad("column_tables/"+tid,"original_identity_values","可逆观测ID与原dataset/nativePTS/sourceID实际值不一致。")
    for layout in bundle.compute_layouts:
        for binding in layout.bindings:
            ref = refs.get(binding.array_id)
            if ref is not None and binding.array_id in loaded:
                try:
                    store.validate_binding(ref,binding)
                except (ValueError,OSError) as exc:
                    bad("compute_layouts/"+layout.id,"compute_binding_values",str(exc))
    if problem is not None:
        families = {n.id:n for n in problem.noise_families}
        known_features = {f.id for f in (*bundle.landmarks,*bundle.curve_loci,*bundle.region_loci,*bundle.installed_artwork_features)}
        for factor in problem.factors:
            selection = factor.association_selection
            if selection is None:
                continue
            chosen = _selected_rows(selection,tables,columns,loaded)
            if chosen is None:
                continue
            table, cols = tables[selection.table_id], columns[selection.table_id]
            def semantic(meaning):
                return next((cols[c.name][chosen] for c in table.columns if c.meaning is meaning and c.name in cols),None)
            use = semantic(S.ColumnMeaning.ASSOCIATION_USE)
            feature = semantic(S.ColumnMeaning.FEATURE_ID)
            obs = semantic(S.ColumnMeaning.OBSERVATION_ID)
            path = "factors/"+factor.id
            if use is not None and not np.all(use==b"fit" if use.dtype.kind=="S" else use=="fit"):
                bad(path,"factor_inactive_values","实际因素选择中混入inactive/rejected/holdout关联。")
            if feature is not None and any(_str(v) not in known_features for v in feature):
                bad(path,"factor_feature_values","因素实际特征ID未在当前语义模型登记，不能按旧材料row推断。")
            family = families.get(factor.noise_family_id)
            if family and family.observation_selection is not None and obs is not None:
                ns = family.observation_selection
                observed_rows = _selected_rows(ns,tables,columns,loaded)
                if observed_rows is not None:
                    nt = tables[ns.table_id]
                    keys = columns[nt.id].get(nt.primary_key_column)
                    if keys is not None and not np.all(np.isin(obs,keys[observed_rows])):
                        bad(path,"factor_noise_coverage_values","实际因素观测不全在所声明噪声族实际选择中。")
    return tuple(issues)



def _selected_rows(selection, tables, columns, loaded):
    """Only called after checks; unresolved selections remain unresolved."""
    table = tables.get(selection.table_id)
    if table is None or table.generation != selection.generation:
        return None
    if selection.mode is S.SelectionMode.SMALL_IDS:
        key = columns.get(table.id, {}).get(table.primary_key_column)
        if key is None:
            return None
        wanted = _strings(selection.ids) if key.dtype.kind=="S" else np.asarray(selection.ids,dtype=key.dtype)
        return np.flatnonzero(np.isin(key, wanted))
    aid = selection.row_index_array_id if selection.mode is S.SelectionMode.ROW_INDEX else selection.row_mask_array_id
    a = loaded.get(aid)
    if a is None or a.ndim != 1:
        return None
    if selection.mode is S.SelectionMode.ROW_MASK:
        if a.shape != (table.row_count,) or not np.all((a==0)|(a==1)):
            return None
        return np.flatnonzero(a)
    if a.dtype.kind not in "iu" or (a.size and (a.min()<0 or a.max()>=table.row_count)):
        return None
    return a.astype(np.int64, copy=False)


class BatchValidator:
    """Bound current Data store for schema.validate_prepared's callable hook."""
    def __init__(self, data_root):
        self.store = ArrayStore(data_root)

    def validate_batches(self, problem, bundle):
        return validate_batches(problem, bundle, store=self.store)

    __call__ = validate_batches


def load_current_support(delivery: ObservationImport):
    """Current W/Query descriptors and inherited qualification, independent of packet."""
    manifest = json.loads(delivery.manifest_path.read_text(encoding="utf-8"))
    support = json.loads(Path(manifest["current_support_path"]).read_text(encoding="utf-8"))
    if support.get("schema") != "rinksolve.current_support_snapshot.v1":
        raise ValueError("unknown current support schema")
    return support


def validate_current_support(delivery: ObservationImport, *, data_root):
    """Validate affected current numerical support without provenance-chain replay."""
    support = load_current_support(delivery)
    store = ArrayStore(data_root)
    refs = {a.id:a for a in delivery.bundle.array_refs}
    table = next(t for t in delivery.bundle.column_tables if t.id=="s01.raw")
    cols = {c.name:store.open(refs[c.array_id]) for c in table.columns}
    original = cols["original_row"]
    issues = []
    def bad(path,code,message):
        issues.append(S.Issue(path,code,message))
    if original.ndim!=1 or original.dtype.kind not in "iu" or (len(original)>1 and np.any(original[1:]<=original[:-1])):
        bad("support/source_row_mapping","original_row_mapping","原始row到当前SoA映射必须严格升序无重复。")
        return tuple(issues)
    selected=set(map(int,original))
    for group in support["groups"]:
        path="support/groups/"+group["id"]
        if not group["rows"] or not set(group["rows"])<=selected:
            bad(path,"group_source_rows","当前W组含当前S01观测集合外的源row；必须显式处理必要关联。")
        try:
            w = _array(group["W"],Path(".")) if isinstance(group["W"],dict) else np.asarray(group["W"],dtype="<f8")
            if w.size==0 and group["rank"]==0:
                if group.get("source_noise_covariance_claimed") is True:
                    bad(path,"unknown_noise_claim","空W不能宣称已有噪声协方差。")
            elif w.shape!=(group["rank"],2*len(group["rows"])) or not np.all(np.isfinite(w)):
                bad(path,"current_W_values","当前W实际shape/rank/数值不符。")
        except (ValueError,OSError,KeyError) as exc:
            bad(path,"current_W_read",str(exc))
    for q in support["queries"]:
        uv,alpha=np.asarray(q["native_uv"]),np.asarray(q["alpha"])
        if q["row_index"] not in selected or uv.shape!=(len(alpha),2) or not np.all(np.isfinite(uv)) or not np.all(np.isfinite(alpha)):
            bad("support/queries/"+str(q["row_index"]),"query_source_values","当前Query原源row、alpha或nativeUV映射不符。")
    if not set(support["independent_rows"])<=selected:
        bad("support/independent_rows","independent_source_rows","既有独立统计行中含非当前S01源row。")
    for group in support["engineering_pixel_dependency_groups"]["groups"]:
        if not set(group["rows"]).intersection(selected):
            bad("support/dependencies/"+group["id"],"unrelated_dependency_group","当前依赖组与S01无关联。")
    return tuple(issues)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    imp = sub.add_parser("import-current")
    for key in ("descriptors","intersections","out","data-root"):
        imp.add_argument("--"+key,required=True)
    check = sub.add_parser("check")
    check.add_argument("manifest")
    check.add_argument("--data-root",required=True)
    args = parser.parse_args(argv)
    if args.command == "import-current":
        result = import_current(args.descriptors,args.intersections,args.out,data_root=args.data_root)
    else:
        result = load_import(args.manifest)
        S.validate_bundle(result.bundle)
        issues = (*validate_batches(None,result.bundle,data_root=args.data_root,selections=result.selections), *validate_current_support(result,data_root=args.data_root))
        if issues:
            raise S.ContractError(issues)
    print(json.dumps(result.report,ensure_ascii=False,allow_nan=False))


if __name__ == "__main__":
    main()
