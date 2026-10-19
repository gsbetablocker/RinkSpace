"""Traceable image evidence for vision training, separate from fitted geometry.

Solver FIT/HELD collections are an audit inventory, not pixel ground truth.
Only original user drawings produce weak supervision in ``samples``. Neither
world coordinates, camera projections nor display smoothing become labels.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from datetime import datetime, timezone
import hashlib
import json
import time
from pathlib import Path
import pickle
import re
import sqlite3

import numpy as np

from .storage import AssetStore


SCHEMA = "rinkspace.vision-image-evidence.v1"
ORIGINAL_PACKET_SHA = "ca5179e55bb1fd92b48e232ab1367ba583e41b1181b9849fa92d3df795942f78"
VIDEO_SHA = "f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891"
PACKET_OPTIONS = ("--additional-ice-source", "--additional-board-source",
                  "--additional-board-material-point-source", "--additional-board-circle-source",
                  "--persistent-ice-long-baseline-source")
QUALITY_KEYS = ("method", "extraction_method", "extraction", "sigma_px", "sigma",
                "normal", "normal_interval", "normal_interval_px", "source_support",
                "point_support_bounds_xy_px", "curve_observation_kind", "point_correspondence",
                "source_uv_is_image_boundary_context_only", "visibility", "visibility_note",
                "physical_identity_status", "evidence_role", "source_identity_authority",
                "source_review_authority", "qualification_status", "endpoint_observation",
                "missing_source_support", "forward_backward_px", "unwarped_source_patch_ncc",
                "model_projection_used_to_generate_or_select_uv", "source_clock",
                "official_exposure_match", "patch_reference_ray_not_exact_paint_curve_intersection")


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _dumps(value):
    return json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class _DataOnlyUnpickler(pickle.Unpickler):
    """The immutable compact export contains arrays and ordinary containers."""
    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"), ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
            ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
            ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "slice"),
            ("builtins", "complex"),
        }
        if (module, name) not in allowed:
            raise ValueError(f"Non-data pickle global rejected: {module}.{name}")
        return super().find_class(module, name)


def load_original_packet(path):
    """Read the certified original pixels without importing the solver graph."""
    path = Path(path)
    if _sha(path) != ORIGINAL_PACKET_SHA:
        raise ValueError("Original pixel packet differs from its immutable source SHA")
    with path.open("rb") as stream:
        return _DataOnlyUnpickler(stream).load()


def _uv(value):
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return []
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("Source UV must be a finite N x 2 array")
    return array.tolist()


def _pts(value, fallback=None):
    if value is None:
        return None if fallback is None else int(fallback)
    array = np.asarray(value)
    if array.ndim:
        values = np.unique(array)
        if len(values) != 1:
            raise ValueError("One audit row cannot silently merge multiple source timestamps")
        return int(values[0])
    return int(value)


def _asset(store, path, base=None):
    if not path:
        return None
    return str(store.source(Path(base) / path if base and not Path(path).is_absolute() else path))


def _advertisement_numbers(row):
    curve = row.get("curve", {})
    values = [curve.get("advertisement_id"), curve.get("advertisement_number")]
    values.extend(int(m) for m in re.findall(r"(?i)(?:^|[_:.])ad(?:vertisement)?[_:]?(\d+)", str(row.get("id", ""))))
    return sorted({int(n) for n in values if n is not None and 1 <= int(n) <= 26})


def _typed(row):
    name = str(row.get("feature_id", row.get("id", "unknown")))
    curve = row.get("curve", {})
    kind = str(curve.get("kind", row.get("curve_observation_kind", "")))
    if _advertisement_numbers(row):
        surface = "board_surface"
    elif "top" in name.lower():
        surface = "board_top"
    elif "contact" in name.lower():
        surface = "ice_board_contact"
    else:
        surface = "ice_marking_or_unresolved"
    primitive = "circle" if ("circle" in kind or re.search(r"\b[FRAC]\d?[_:]", name)) else "free_curve"
    if "line" in kind or "lane" in name.lower() or name in ("C", "B1", "B2"):
        primitive = "line"
    return {"surface": surface, "primitive": primitive, "curve": curve,
            "advertisement_numbers": _advertisement_numbers(row),
            "world_fields_excluded_from_labels": True}


def _init_db(connection):
    connection.execute("""CREATE TABLE IF NOT EXISTS observations (
        id TEXT PRIMARY KEY, packet_path TEXT NOT NULL, packet_sha256 TEXT NOT NULL,
        packet_kind TEXT NOT NULL, row_index INTEGER NOT NULL, observation_id TEXT,
        feature_id TEXT, block_id TEXT, collection_role TEXT, source_split TEXT,
        source_pts INTEGER, actual_exposure_pts INTEGER, time_base TEXT, source_video_sha256 TEXT,
        frame_path TEXT, frame_sha256 TEXT, image_width INTEGER, image_height INTEGER,
        uv_json TEXT NOT NULL, uv_count INTEGER NOT NULL, reference_uv_json TEXT,
        reference_pts INTEGER, reference_frame_path TEXT, sigma_json TEXT,
        typed_geometry_json TEXT NOT NULL, provenance_json TEXT NOT NULL, quality_json TEXT NOT NULL,
        authority TEXT NOT NULL, user_confirmation TEXT NOT NULL, audit_state TEXT NOT NULL,
        supervision_enabled INTEGER NOT NULL DEFAULT 0, kind TEXT NOT NULL,
        source_frame TEXT, source_video_path TEXT NOT NULL, record_json TEXT NOT NULL)""")
    connection.execute("CREATE INDEX IF NOT EXISTS observation_source ON observations(source_video_sha256, source_pts)")
    connection.execute("CREATE INDEX IF NOT EXISTS observation_feature ON observations(feature_id, audit_state)")


def _index_row(connection, store, path, digest, family, index, row, *, block=None,
               collection=None, actual=None, source_hash=VIDEO_SHA, time_base="1/15360", dims=(1920, 1080)):
    pair = family == "persistent_ice"
    target = row.get("target", {}) if pair else {}
    uv = _uv(target.get("uv", row.get("uv", [])))
    reference = row.get("anchor", {}) if pair else {}
    ref_uv = _uv(reference.get("uv", [])) if pair else None
    native = _pts(target.get("source_native_pts", row.get("source_native_pts")), row.get("pts", 0))
    actual = row.get("actual_pts", actual)
    clock = row.get("official_exposure_match", row.get("source_clock", {}))
    frame = target.get("raw_png") if pair else row.get("source_raw_frame", row.get("source_frame"))
    frame_path = _asset(store, frame, path.parent) if frame else None
    quality = {k: row[k] for k in QUALITY_KEYS if k in row}
    quality["source_pixel_precision_is_not_semantic_confirmation"] = True
    source_split = str(row.get("split", "absent"))
    observed_id = str(row.get("id", "unknown"))
    confirmed = "not_established"
    if observed_id == "lower_long_side_board_top_confirmed_source_interval":
        confirmed = "user_identity_only_not_precise_pixels"
        quality["user_identity_reference"] = str(store.event / "evidence/main_nearboard_source_candidates_20260928/user_confirmed_main_lower_board_top.json")
    pending = "identity_pending" if "identity_pending" in source_split else "source_uv_review_pending"
    if observed_id == "cut6_source_230507776_F3_outer_fit":
        pending = "user_review_requested_blurred_F3_paint_or_shadow"
    provenance = {"origin": row.get("origin"), "source_native_pts": row.get("source_native_pts"),
                  "row_pts": row.get("pts"), "official_exposure_match": clock,
                  "original_packet_path": row.get("original_packet_path"),
                  "source_UV_is_inventory_not_ground_truth": True}
    if pair:
        provenance.update(reference_frame_sha256=reference.get("raw_png_sha256"),
                          target_frame_sha256=target.get("raw_png_sha256"),
                          shared_anchor_correlated=True, source_only=bool(row.get("source_only")),
                          no_world_xyz_as_label=True)
    typed = _typed(row)
    kind = {"board_top": "board_top", "ice_board_contact": "ice_contact",
            "board_surface": "ad_boundary"}.get(typed["surface"], "ice_paint")
    video_path = str(store.event / "media/originals/oxXu1avhrvs.mp4")
    record = {"observation_id": observed_id, "feature_id": row.get("feature_id", observed_id),
              "typed_geometry": typed, "source_split": source_split, "collection_role": collection,
              "quality": quality, "provenance": provenance, "audit_state": pending,
              "supervision_enabled": False, "world_or_camera_pixels_are_not_labels": True}
    values = (f"{family}:{digest}:{index}", str(path), digest, family, index, observed_id,
              str(row.get("feature_id", row.get("track_id", observed_id))), block, collection,
              source_split, native, _pts(actual), time_base, source_hash,
              frame_path, target.get("raw_png_sha256") if pair else row.get("source_file_sha256"),
              int(dims[0]), int(dims[1]), _dumps(uv), len(uv), _dumps(ref_uv) if pair else None,
              reference.get("source_native_pts") if pair else None,
              _asset(store, reference.get("raw_png")) if pair else None,
              _dumps(row.get("sigma_px", row.get("sigma", row.get("sigma_native_px")))),
              _dumps(typed), _dumps(provenance), _dumps(quality),
              "source_packet_agent_or_algorithm_not_pixel_GT", confirmed, pending, 0,
              kind, frame_path, video_path, _dumps(record))
    connection.execute("INSERT INTO observations VALUES (" + ",".join("?" for _ in values) + ")", values)
    return len(uv) + (len(ref_uv) if pair else 0)


def iter_observations(database_path, *, packet_kind=None, limit=None):
    """Yield normalized audit rows without building a solver or loading frames.

    ``uv`` belongs to ``source_frame``/``source_pts``. Pairwise material rows
    additionally expose ``reference_uv``/``reference_pts`` in the other frame.
    Derived pixel audits belong in a separate measurements table; this reader
    opens SQLite read-only and never replaces original observations.
    """
    path = Path(database_path).resolve()
    query = "SELECT * FROM observations"
    parameters = []
    if packet_kind is not None:
        query += " WHERE packet_kind=?"
        parameters.append(packet_kind)
    query += " ORDER BY packet_kind,row_index"
    if limit is not None:
        if isinstance(limit, bool) or int(limit) < 0:
            raise ValueError("Audit row limit must be nonnegative")
        query += " LIMIT ?"
        parameters.append(int(limit))
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        human_by_observation = {}
        if connection.execute("SELECT 1 FROM sqlite_master WHERE name='human_semantic_links'").fetchone():
            for link in connection.execute("SELECT l.obs_id,d.record_json FROM human_semantic_links l JOIN human_semantic_decisions d ON d.decision_id=l.decision_id"):
                human_by_observation.setdefault(link[0], []).append(json.loads(link[1]))
        for item in connection.execute(query, parameters):
            row = dict(item)
            for key, name in (("uv_json", "uv"), ("reference_uv_json", "reference_uv"),
                              ("sigma_json", "sigma"), ("typed_geometry_json", "typed_geometry"),
                              ("provenance_json", "provenance"), ("quality_json", "quality"),
                              ("record_json", "record")):
                value = row.pop(key)
                row[name] = json.loads(value) if value is not None else None
            row["source_hash"] = row["source_video_sha256"]
            row["frame_size_wh"] = [row["image_width"], row["image_height"]]
            row["supervision_enabled"] = bool(row["supervision_enabled"])
            if row["id"] in human_by_observation:
                row["human_semantics"] = human_by_observation[row["id"]]
                row["original_audit_state"] = row["audit_state"]
                row["user_confirmation"] = "direct_user_feature_identity_only"
                row["audit_state"] = "human_identity_confirmed_source_edge_quality_pending"
            yield row
    finally:
        connection.close()


def _sample(store, path, pts, time_base, source_hash, frame_hash=None):
    from PIL import Image
    frame = Path(path)
    if not frame.is_file():
        raise FileNotFoundError(f"Original training frame missing: {frame}")
    digest = _sha(frame)
    if frame_hash and frame_hash != digest:
        raise ValueError(f"Original training frame SHA differs: {frame}")
    with Image.open(frame) as image:
        dims = list(image.size)
    return {"frame_path": str(frame), "frame_sha256": digest, "source_pts": int(pts),
            "time_base": str(Fraction(time_base)), "source_hash": source_hash,
            "source_time_seconds": float(int(pts) * Fraction(time_base)),
            "frame_size_wh": dims, "annotations": [], "ignored_regions": [],
            "authority": "direct_user_drawing", "pixel_ground_truth": False}


def _annotation(kind, points, intervals, group, primitive="free_curve", feature=None, authority=None):
    points = _uv(points)
    if len(points) != len(intervals):
        raise ValueError("Drawing supports must correspond to actual drawn points")
    return {"kind": kind, "uv": points, "normal": [[0., 1.]] * len(points),
            "interval_px": _jsonable(intervals), "primitive": primitive,
            "authority": authority or "direct_user_drawing", "state": "user_trace_weak",
            "group_id": group, "feature_id": feature or group,
            "drawing_support_only": True, "pixel_ground_truth": False,
            "manual_error_unquantified": True, "source_blur_error_unquantified": True,
            "samples_correlated_within_group": True,
            "support_direction_definition": "vertical_per_column_raster_support_not_physical_curve_normal"}


def _user_samples(store, sources):
    samples = []
    wide_path = store.event / "evidence/user_contours_20261004/edge1_2e2bd471877d/edge1_source_uv.json"
    wide = _read(wide_path)
    sources.append({"path": str(wide_path), "sha256": _sha(wide_path), "role": "direct_user_contours"})
    source = wide["source"]
    sample = _sample(store, _asset(store, source["source_frame"]), source["source_pts"], source["time_base"],
                     source.get("same_event_source_video_sha256", VIDEO_SHA), source["source_frame_sha256"])
    for contour in wide["contours"]:
        points = contour["observed_centres_source_px"]
        support = contour["column_support_x_ymin_ymax_alpha_sum"]
        intervals = [[s[1] - p[1], s[2] - p[1]] for p, s in zip(points, support)]
        sample["annotations"].append(_annotation("board_top", points, intervals, contour["id"],
                                                  authority={"path": str(wide_path), **wide["authority"]}))
    samples.append(sample)
    base = store.event / "evidence/low_end_reference_review_20260923"
    # Conservative source-image visibility exclusions, not positive segmentation labels.
    occluded = {1: [(1220, 1450)], 2: [(1000, 1210)], 3: [],
                4: [(290, 435), (500, 625), (610, 880), (870, 935),
                    (1180, 1340), (1375, 1500), (1570, 1620)]}
    for number in range(1, 5):
        name = f"frame{number}_user_far_board_edges_trace.json" if number < 4 else "frame4_user_board_top_and_ice_edge_trace.json"
        path = base / name
        trace = _read(path)
        review_path = base / f"frame{number}_user_review.json"
        review = _read(review_path)
        sources.append({"path": str(path), "sha256": _sha(path), "role": "direct_user_rough_board_trace"})
        sample = _sample(store, _asset(store, trace["source_frame"], base), trace["source_pts"],
                         trace["source_time_base"], VIDEO_SHA, review.get("native_frame_sha256"))
        for a, b in occluded[number]:
            sample["ignored_regions"].append({"x_range_source_px": [a, b],
                "reason": "user_stroke_continues_through_source_foreground_occlusion", "positive_label": False})
        for edge, kind in (("upper", "board_top"), ("lower", "ice_contact")):
            if number < 4:
                values = trace[edge + "_y_by_x"]
                widths = (trace["drawn_width_px_by_x"][0 if edge == "upper" else 1]
                          if number == 1 else [v[0 if edge == "upper" else 1] for v in trace["purple_core_width_by_x"]])
                x_start = trace["x_range_inclusive"][0]
                scale, offset = 1., 0.
            else:
                obj = trace["board_top_trace" if edge == "upper" else "ice_edge_trace"]
                values, widths = obj["y_by_markup_x"], obj["stroke_width_px_by_markup_x"]
                x_start = trace["x_range_inclusive_markup"][0]
                scale, offset = 24 / 25, 2.
            points, intervals = [], []
            for i, (y, width) in enumerate(zip(values, widths)):
                if y is None or not width:
                    continue
                x = (x_start + i - offset) * scale
                if not (0 <= x < 1920 and 0 <= y * scale < 1080):
                    continue
                if any(a <= x <= b for a, b in occluded[number]):
                    continue
                points.append([x, y * scale])
                intervals.append([-width * scale / 2, width * scale / 2])
            sample["annotations"].append(_annotation(kind, points, intervals,
                f"low_end_{trace['source_pts']}_user_{edge}", authority={"path": str(path),
                "review_path": str(review_path), "scope": "approximate_user_drawn_locations_not_precise_edges"}))
        samples.append(sample)
    _append_F1_raster_support(base, samples[3], sources)
    return samples


def _append_F1_raster_support(base, sample, sources):
    """Read actual painted screenshot cells; exclude off-image and hidden arcs."""
    import cv2
    from PIL import Image
    path = base / "frame3_user_landmark_markup_original.png"
    review_path = base / "frame3_user_review.json"
    review = _read(review_path)
    if _sha(path) != review["user_landmark_markup_sha256"]:
        raise ValueError("F1 original user markup SHA differs")
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    exact = np.all(rgb == [4, 36, 51], axis=2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(exact, 8)
    boxes = {"outer": (810, 813, 1314, 250), "middle": (1024, 837, 879, 162),
             "inner": (1245, 876, 450, 78)}
    sources.append({"path": str(path), "sha256": _sha(path), "role": "direct_user_F1_raster_markup"})
    for role, (x0, y0, width, height) in boxes.items():
        candidates = []
        for index in range(1, count):
            x, y, w, h, area = map(int, stats[index])
            overlap = max(0, min(x + w, x0 + width) - max(x, x0)) * max(0, min(y + h, y0 + height) - max(y, y0))
            score = overlap / max(1, w * h) + overlap / (width * height) + min(area, 10000) / 1000000
            candidates.append((score, index))
        score, index = max(candidates)
        if score < .1:
            raise ValueError(f"Actual F1 {role} user stroke not found")
        component = labels == index
        branches = {"upper": ([], []), "lower": ([], [])}
        for x in range(max(83, x0), min(2083, x0 + width)):
            ys = np.flatnonzero(component[:, x])
            ys = ys[(ys >= y0) & (ys < min(1125, y0 + height))]
            if not len(ys):
                continue
            groups = np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1)
            # Two separated raster runs supply two actual drawn branches;
            # the thin tips or ambiguous connected runs are omitted, not filled.
            if len(groups) != 2:
                continue
            for branch, group in zip(("upper", "lower"), groups):
                if role == "outer" and branch == "upper" and 850 <= x <= 1100:
                    continue
                y = float(group.mean())
                native = [(x - 83) * 24 / 25, y * 24 / 25]
                if not (0 <= native[0] < 1920 and 0 <= native[1] < 1080):
                    continue
                points, intervals = branches[branch]
                points.append(native)
                intervals.append([(group[0] - .5 - y) * 24 / 25,
                                  (group[-1] + .5 - y) * 24 / 25])
        for branch, (points, intervals) in branches.items():
            if not points:
                continue
            annotation = _annotation("ice_paint", points, intervals,
                f"F1_user_{role}_{branch}_{sample['source_pts']}", "circle", "F1",
                {"path": str(path), "review_path": str(review_path), "ring_role": role,
                 "mapping": "x=(markup_x-83)*24/25,y=markup_y*24/25",
                 "scope": "approximate_user_contour_actual_inframe_raster_only"})
            annotation["ring_role"] = role
            annotation["presentation_mapping_not_precise_edge_calibration"] = True
            sample["annotations"].append(annotation)


def prepare_dataset(project_root, output_dir):
    """Refresh a reusable SQLite audit and the existing user-drawing manifest.

    ``samples`` are weak, source-image drawing supports. All original solver
    rows and current explicit packets are indexed with supervision disabled.
    This function decodes no video, starts no solver/training, copies no frames,
    and writes only manifest.json and observations.sqlite in the fixed folder.
    """
    project = Path(project_root).resolve()
    store = AssetStore(project)
    output = Path(output_dir).resolve()
    expected = (store.event / "observations/vision_training").resolve()
    if output != expected:
        raise ValueError(f"Vision data must reuse the fixed directory: {expected}")
    output.mkdir(parents=True, exist_ok=True)
    config_path = project / "configs/solver_runtime/okada_named66.json"
    config = _read(config_path)
    root = Path(config["resource_root"])
    original_path = root / "shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl"
    original = load_original_packet(original_path)
    sources = [{"path": str(config_path), "sha256": _sha(config_path), "role": "current_source_configuration"},
               {"path": str(original_path), "sha256": ORIGINAL_PACKET_SHA, "role": "original_real_source_inventory"}]
    database = output / "observations.sqlite"
    counts, endpoints, ads_counts = Counter(), Counter(), Counter()
    retirement = None
    retirement_path = config["source_options"].get("--original-source-role-retirement-source")
    if retirement_path:
        retirement = _read(retirement_path)
        sources.append({"path": retirement_path, "sha256": _sha(retirement_path),
                        "role": "current_source_identity_role_correction"})
    with sqlite3.connect(database) as connection:
        _init_db(connection)
        connection.execute("DELETE FROM observations")
        for index, item in enumerate(original["programme_absolute_rows"]):
            row = item["source_row"]
            n = _index_row(connection, store, original_path, ORIGINAL_PACKET_SHA, "original_absolute", index, row,
                block=item["block"], collection=item["collection_role"], actual=item["group_actual_pts"])
            counts["original_absolute"] += 1
            endpoints["original_absolute"] += n
            ads_counts.update(_advertisement_numbers(row))
            if retirement:
                locator = retirement["source_row_locator"]
                if (item["block"] == locator["block"] and item["collection_role"] == locator["collection_role"]
                        and item["group_order"] == locator["group_order"] and item["row_order"] == locator["row_order"]
                        and row["id"] == locator["id"] and _pts(row["pts"]) == locator["pts"]):
                    key = f"original_absolute:{ORIGINAL_PACKET_SHA}:{index}"
                    connection.execute("UPDATE observations SET audit_state=? WHERE id=?",
                        ("retired_print_corner_identity_is_board_rail", key))
        for option in PACKET_OPTIONS:
            if not config["source_options"].get(option):
                continue
            path = Path(config["source_options"][option]).resolve()
            data, digest = _read(path), _sha(path)
            family = "persistent_ice" if option == "--persistent-ice-long-baseline-source" else option[2:]
            sources.append({"path": str(path), "sha256": digest, "role": family})
            rows = data["new_rows"] if family == "persistent_ice" else data["rows"]
            for index, row in enumerate(rows):
                role = "FIT" if row.get("split") == "fit" else "HELD" if row.get("split") in ("holdout", "held") else "PENDING"
                n = _index_row(connection, store, path, digest, family, index, row, collection=role,
                    source_hash=data.get("source_video_sha256", data.get("source_sha256", VIDEO_SHA)),
                    time_base=data["time_base"], dims=data.get("image_size_wh", data.get("frame_size_wh", [1920, 1080])))
                counts[family] += 1
                endpoints[family] += n
                ads_counts.update(_advertisement_numbers(row))
        audit_counts = dict(connection.execute("SELECT audit_state, COUNT(*) FROM observations GROUP BY audit_state"))
        split_counts = dict(connection.execute("SELECT source_split, COUNT(*) FROM observations GROUP BY source_split"))
        total = connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    samples = _user_samples(store, sources)
    advertisement_path = project / "configs/rink_model/ankara_2026_advertisements_user_confirmed_20260924.json"
    advertisements = _read(advertisement_path)
    sources.append({"path": str(advertisement_path), "sha256": _sha(advertisement_path), "role": "user_confirmed_permanent_advertisement_identity"})
    content = {n: pair["content_id"] for pair in advertisements["same_content_pairs"] for n in pair["numbers"]}
    permanent_ids = [{"advertisement_number": p["number"], "physical_id": f"Ad{p['number']}",
        "instance_id": p["instance_id"], "content_id": content[p["number"]], "region": p["region"],
        "identity_authority": "user_confirmed_26_physical_instances", "authority_path": str(advertisement_path),
        "audit_observation_rows": ads_counts[p["number"]], "pixel_supervision_available": False,
        "pixel_supervision_state": "source_pixels_pending_review_no_world_projection_labels"}
        for p in advertisements["positions"]]
    manifest = {"schema": SCHEMA, "project_root": str(project), "event_id": "ankara_2026",
        "database_path": str(database), "sources": sources, "samples": samples,
        "advertisement_instances": permanent_ids,
        "inventory": {"observation_rows": total, "rows_by_packet_kind": dict(counts),
            "real_UV_endpoint_count": sum(endpoints.values()), "UV_endpoints_by_packet_kind": dict(endpoints),
            "source_split_counts": split_counts, "audit_state_counts": audit_counts,
            "user_frame_samples": len(samples), "weak_annotation_groups": sum(len(s["annotations"]) for s in samples),
            "weak_user_drawn_centres": sum(len(a["uv"]) for s in samples for a in s["annotations"]),
            "audit_rows_used_as_training_supervision": 0},
        "label_contract": {"supervision": "actual_user_drawings_with_weak_intervals_and_visibility_exclusions",
            "drawing_width_is_not_physical_edge_precision": True,
            "manual_and_blur_error_unquantified": True,
            "FIT_or_HELD_does_not_establish_pixel_truth": True,
            "world_camera_projection_or_smoothed_pixels_used_as_labels": False,
            "native60_and_official50_must_use_explicit_exposure_correspondence": True,
            "frame_and_physical_group_split_required_for_training_evaluation": True},
        "review_questions": [{"observation_id": "cut6_source_230507776_F3_outer_fit",
            "source_pts": 230507776, "time_base": "1/15360", "source_time": "04:10:07.017",
            "uv": [1330.3853659034214, 730.0], "status": "user_review_pending_paint_boundary_or_shadow",
            "supervision_enabled": False}]}
    manifest_path = output / "manifest.json"
    temporary = output / "manifest.json.tmp"
    temporary.write_text(_dumps(manifest), encoding="utf8")
    temporary.replace(manifest_path)
    return manifest_path


PIXEL_AUDITOR = "original-rgb-axis-profiles-v1"


def join_user_decisions(database_path, decisions_path=None):
    """Join human feature semantics to audit evidence without changing old UV.

    Original observation records and all pixel profiles remain intact. Human
    confirmation of F3's outer ring does not certify which precise edge each
    old FIT pixel measures, and does not promote those FIT pixels to labels.
    """
    database = Path(database_path).resolve()
    path = Path(decisions_path or database.parent / "user_decisions.json")
    if not path.is_file():
        return {"decisions": 0, "linked_observations": 0, "linked_measurements": 0}
    data, digest = _read(path), _sha(path)
    linked_observations, linked_measurements = 0, 0
    with sqlite3.connect(database, timeout=10) as connection:
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("""CREATE TABLE IF NOT EXISTS human_semantic_decisions (
            decision_id TEXT PRIMARY KEY, source_pts INTEGER NOT NULL, physical_identity TEXT NOT NULL,
            authority TEXT NOT NULL, decision_path TEXT NOT NULL, decision_file_sha256 TEXT NOT NULL,
            record_json TEXT NOT NULL)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS human_semantic_links (
            obs_id TEXT NOT NULL, decision_id TEXT NOT NULL, PRIMARY KEY(obs_id,decision_id))""")
        for decision in data.get("decisions", []):
            if not str(decision.get("identity_authority", "")).startswith("direct_user_confirmation"):
                continue
            image = Path(decision["image_path"])
            if _sha(image) != decision["image_sha256"]:
                raise ValueError("Original human annotation image SHA differs")
            identity = decision["physical_identity"]
            feature, _, role = identity.partition(":")
            record = {k: decision[k] for k in ("id", "source_pts", "time_base", "physical_identity",
                "identity_authority", "image_path", "image_sha256", "image_mapping", "annotation_role",
                "pixel_accuracy_confirmed", "exact_edge_side_confirmed", "existing_fit_uv_not_automatically_corrected",
                "user_statement", "target_geometry", "observation_ids", "old_source_role_status",
                "source_video_sha256") if k in decision}
            record.update(decision_path=str(path), decision_file_sha256=digest,
                          source_UV_or_solver_roles_changed=False, automatic_supervision_promotion=False)
            connection.execute("INSERT OR REPLACE INTO human_semantic_decisions VALUES (?,?,?,?,?,?,?)",
                (decision["id"], decision["source_pts"], identity, decision["identity_authority"], str(path), digest, _dumps(record)))
            # Rebuild this decision's derived links: old PTS-only joins can
            # include another video's exposure at the same integer timestamp.
            previous_ids = [r[0] for r in connection.execute(
                "SELECT obs_id FROM human_semantic_links WHERE decision_id=?", (decision['id'],))]
            has_measurements = connection.execute("SELECT 1 FROM sqlite_master WHERE name='measurements'").fetchone()
            if has_measurements:
                for prior in previous_ids:
                    for index, endpoint, auditor, summary_json, flags_json in connection.execute(
                            "SELECT point_index,endpoint,auditor,summary_json,flags_json FROM measurements WHERE obs_id=?", (prior,)).fetchall():
                        summary, flags = json.loads(summary_json), json.loads(flags_json)
                        if summary.get('human_semantics', {}).get('id') != decision['id']:continue
                        summary.pop('human_semantics')
                        summary['prior_semantic_link_rechecked_against_full_source_key'] = True
                        derived = {'human_target_definition_confirmed', 'source_role_misassigned',
                            'human_feature_identity_confirmed', 'precise_edge_quality_unverified'}
                        flags = [flag for flag in flags if flag not in derived]
                        if 'identity_unverified' not in flags:flags.append('identity_unverified')
                        connection.execute("UPDATE measurements SET summary_json=?,flags_json=?,audit_flag=? WHERE obs_id=? AND point_index=? AND endpoint=? AND auditor=?",
                            (_dumps(summary), _dumps(flags), '|'.join(flags), prior, index, endpoint, auditor))
            connection.execute("DELETE FROM human_semantic_links WHERE decision_id=?", (decision["id"],))
            source_sha = decision.get("source_video_sha256")
            source_clock = decision.get("time_base")
            if not source_sha or not source_clock or decision.get('coordinate_domain', 'native_pixels') != 'native_pixels':
                # Missing identity is unresolved; a nearby or unique-looking
                # PTS is not enough to transfer human authority.
                continue
            from fractions import Fraction
            candidates = connection.execute("SELECT id,observation_id,time_base FROM observations "
                "WHERE source_pts=? AND source_video_sha256=? AND (feature_id=? OR observation_id LIKE ?)",
                (decision["source_pts"], source_sha, feature, f"%{feature}_{role}%")).fetchall()
            candidates = [(oid, observed) for oid, observed, clock in candidates
                if clock and Fraction(clock) == Fraction(source_clock)]
            explicit_observations = set(decision.get("observation_ids", []))
            for oid, observation_id in candidates:
                if explicit_observations and observation_id not in explicit_observations:
                    continue
                if not explicit_observations and role and role not in observation_id.lower():
                    continue
                connection.execute("INSERT OR REPLACE INTO human_semantic_links VALUES (?,?)", (oid, decision["id"]))
                linked_observations += 1
                if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='measurements'").fetchone():
                    continue
                measured = connection.execute("SELECT point_index,endpoint,auditor,summary_json,flags_json FROM measurements WHERE obs_id=?", (oid,)).fetchall()
                for index, endpoint, auditor, summary_json, flags_json in measured:
                    summary, flags = json.loads(summary_json), json.loads(flags_json)
                    summary["human_semantics"] = record
                    summary["physical_identity_verified_by_pixel_statistics"] = False
                    role_mismatch = decision.get("old_source_role_status") == "misassigned"
                    if role_mismatch:
                        new_flags = ("human_target_definition_confirmed", "source_role_misassigned",
                                     "precise_edge_quality_unverified")
                    else:
                        flags = [flag for flag in flags if flag != "identity_unverified"]
                        new_flags = ("human_feature_identity_confirmed", "precise_edge_quality_unverified")
                    for flag in new_flags:
                        if flag not in flags:
                            flags.append(flag)
                    connection.execute("UPDATE measurements SET summary_json=?,flags_json=?,audit_flag=? WHERE obs_id=? AND point_index=? AND endpoint=? AND auditor=?",
                        (_dumps(summary), _dumps(flags), "|".join(flags), oid, index, endpoint, auditor))
                    linked_measurements += 1
    result = {"decisions_path": str(path), "decisions_sha256": digest,
            "linked_observations": linked_observations, "linked_measurements": linked_measurements,
            "source_UV_or_solver_roles_changed": False, "pixel_supervision_promoted": False}
    for status_path in database.parent.glob("pixel_audit_*.json"):
        status = _read(status_path)
        status["direct_human_semantic_join"] = result
        status["pixel_statistic_flags_precede_human_semantic_join"] = True
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(_dumps(status), encoding="utf8")
        temporary.replace(status_path)
    return result


def _measurements_schema(connection):
    connection.execute("PRAGMA busy_timeout=10000")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("""CREATE TABLE IF NOT EXISTS measurements (
        obs_id TEXT NOT NULL, point_index INTEGER NOT NULL, endpoint TEXT NOT NULL,
        auditor TEXT NOT NULL, native_pts INTEGER, time_base TEXT, native_uv_json TEXT NOT NULL,
        source_video_sha256 TEXT, decoded_frame_pts INTEGER, image_width INTEGER, image_height INTEGER,
        profile_x_rgb_json TEXT, profile_y_rgb_json TEXT, summary_json TEXT NOT NULL,
        flags_json TEXT NOT NULL, audit_flag TEXT NOT NULL, source_evidence_only INTEGER NOT NULL,
        device TEXT NOT NULL, measured_at_utc TEXT NOT NULL,
        PRIMARY KEY (obs_id, point_index, endpoint, auditor))""")
    connection.execute("CREATE INDEX IF NOT EXISTS measured_frame ON measurements(native_pts, auditor)")
    connection.execute("""CREATE TABLE IF NOT EXISTS audit_video_sources (
        video_path TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
        source_sha256 TEXT NOT NULL, verified_at_utc TEXT NOT NULL)""")
    connection.commit()


def _verified_video_hash(connection, path):
    stat = path.stat()
    known = connection.execute("SELECT size_bytes,mtime_ns,source_sha256 FROM audit_video_sources WHERE video_path=?", (str(path),)).fetchone()
    if known and known[:2] == (stat.st_size, stat.st_mtime_ns):
        return known[2], "previous_SHA_verified_and_file_stat_unchanged"
    digest = _sha(path)
    connection.execute("INSERT OR REPLACE INTO audit_video_sources VALUES (?,?,?,?,?)",
        (str(path), stat.st_size, stat.st_mtime_ns, digest, datetime.now(timezone.utc).isoformat()))
    connection.commit()
    return digest, "SHA256_verified_once_for_video"


def _gpu_rgb_profiles(rgb, coordinates, device):
    import torch
    import torch.nn.functional as functional
    height, width = rgb.shape[:2]
    points = torch.tensor(coordinates, dtype=torch.float32, device=device)
    offsets = torch.arange(-16, 17, device=device, dtype=torch.float32)
    horizontal = points[:, None, :].expand(-1, 33, -1).clone()
    vertical = horizontal.clone()
    horizontal[:, :, 0] += offsets
    vertical[:, :, 1] += offsets
    grid = torch.cat((horizontal, vertical), dim=0)
    valid = ((grid[:, :, 0] >= 0) & (grid[:, :, 0] <= width - 1)
             & (grid[:, :, 1] >= 0) & (grid[:, :, 1] <= height - 1))
    valid = valid.reshape(2, len(points), 33).permute(1, 0, 2)
    grid[:, :, 0] = 2 * grid[:, :, 0] / (width - 1) - 1
    grid[:, :, 1] = 2 * grid[:, :, 1] / (height - 1) - 1
    image = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).permute(2, 0, 1)[None]
    with torch.inference_mode():
        values = functional.grid_sample(image, grid[None], mode="bilinear",
                                        padding_mode="zeros", align_corners=True)
        values = values[0].permute(1, 2, 0).reshape(2, len(points), 33, 3).permute(1, 0, 2, 3)
        gray = (values * values.new_tensor([.299, .587, .114])).sum(-1)
        gradient = gray.diff(dim=-1).abs()
        gradient = torch.where(valid[:, :, :-1] & valid[:, :, 1:], gradient, 0.)
        peak, peak_index = gradient.max(-1)
        contrast = gray.masked_fill(~valid, -torch.inf).max(-1).values - gray.masked_fill(~valid, torch.inf).min(-1).values
        support = ((gradient >= .5 * peak[:, :, None]) & (gradient > 0)).sum(-1)
        count = valid.sum(-1)
        average = gray.masked_fill(~valid, 0).sum(-1) / count
        variance = ((gray - average[:, :, None]).square().masked_fill(~valid, 0).sum(-1)
                    / (count - 1).clamp(min=1))
        summaries = torch.stack((contrast, peak, peak_index - 15.5, support.float(),
                                 gradient.mean(-1), variance.sqrt()), dim=-1).cpu().numpy()
        profiles = values.cpu().numpy()
    return profiles, summaries


def audit_source_pixels(database_path, video_path, device="cuda", *, cohort="all", resume=True):
    """Measure original RGB evidence on CUDA; never promote or change a label.

    One exact-PTS decode pass covers the requested window. Every source UV,
    including both endpoints of persistent-material pairs, receives a separate
    measurement. Contrast and gradient peaks cannot establish paint identity;
    flags are review navigation, with explicit heuristic thresholds. No model,
    rendered grid, transformed frame, or world prediction enters this audit.
    """
    import av
    import torch
    database = Path(database_path).resolve()
    video = Path(video_path).resolve()
    if device != "cuda" and not str(device).startswith("cuda:"):
        raise ValueError("This source pixel audit requires the requested CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; source audit was not silently changed to CPU")
    if cohort not in ("all", "explicit"):
        raise ValueError("Pixel audit cohort must be all or explicit")
    started = time.perf_counter()
    connection = sqlite3.connect(database, timeout=10)
    _measurements_schema(connection)
    status_path = database.parent / f"pixel_audit_{cohort}.json"
    source_hash, hash_check = _verified_video_hash(connection, video)
    existing = set(connection.execute("SELECT obs_id,point_index,endpoint FROM measurements WHERE auditor=? AND source_video_sha256=? "
                                     "AND (audit_flag NOT LIKE '%profile_crosses_image_boundary%' "
                                     "OR summary_json LIKE '%\"padding_samples_excluded_from_statistics\":true%')",
                                     (PIXEL_AUDITOR, source_hash))) if resume else set()
    query = "SELECT id,source_pts,actual_exposure_pts,time_base,source_video_sha256,image_width,image_height,uv_json,reference_uv_json,reference_pts FROM observations"
    if cohort == "explicit":
        query += " WHERE packet_kind<>'original_absolute'"
    groups, rejected, expected_count, row_count = {}, [], 0, 0
    for row in connection.execute(query):
        oid, pts, actual, clock, digest, width, height, pixels, references, ref_pts = row
        row_count += 1
        sets = [("target", pts, json.loads(pixels))]
        if references:
            sets.append(("reference", ref_pts, json.loads(references)))
        for endpoint, native_pts, uv in sets:
            for index, point in enumerate(uv):
                expected_count += 1
                if (oid, index, endpoint) in existing:
                    continue
                item = {"obs_id": oid, "point_index": index, "endpoint": endpoint,
                        "pts": native_pts, "time_base": clock, "uv": point,
                        "width": width, "height": height, "expected_hash": digest}
                reason = None
                if native_pts is None or not clock:
                    reason = "missing_source_time"
                elif digest != source_hash:
                    reason = "source_video_SHA_mismatch"
                elif Fraction(clock) != Fraction(1, 15360):
                    reason = "unexpected_source_time_base_requires_explicit_clock_mapping"
                if reason:
                    rejected.append((item, reason))
                else:
                    groups.setdefault(int(native_pts), []).append(item)
    total_new = sum(map(len, groups.values())) + len(rejected)
    state = {"schema": "rinkspace.original-pixel-evidence-audit.v1", "status": "running",
        "database_path": str(database), "video_path": str(video), "source_video_sha256": source_hash,
        "video_hash_check": hash_check, "device": device, "GPU_name": torch.cuda.get_device_name(device),
        "auditor": PIXEL_AUDITOR, "cohort": cohort, "source_rows": row_count,
        "requested_real_UV_endpoints": expected_count, "already_measured_endpoints": expected_count - total_new,
        "new_requested_endpoints": total_new, "requested_frames": len(groups),
        "source_pts_interval": [min(groups), max(groups)] if groups else None,
        "profile_domain": "original_decoded_RGB_native_pixels_plus_minus_16px_X_and_Y",
        "profile_radius_native_px": 16, "profile_samples_per_axis": 33,
        "heuristic_review_thresholds": {"low_gray_contrast": 5., "peak_offset": 8., "half_peak_support": 10},
        "source_evidence_only": True, "physical_identity_verified_by_pixel_statistics": False,
        "automatic_supervision_promotion_or_retirement": False, "old_observations_modified": False,
        "frames_written": 0, "completed_new_endpoints": 0, "decoded_requested_frames": 0}
    pending, flag_counts = [], Counter()

    def flush():
        if pending:
            with connection:
                connection.executemany("INSERT OR REPLACE INTO measurements VALUES (" + ",".join("?" for _ in range(19)) + ")", pending)
            pending.clear()

    def report(force=False):
        now = time.perf_counter()
        if force or now - report.last >= 15:
            flush()
            state["elapsed_seconds"] = round(now - started, 3)
            state["new_measurement_flags"] = dict(flag_counts)
            temporary = status_path.with_suffix(".json.tmp")
            temporary.write_text(_dumps(state), encoding="utf8")
            temporary.replace(status_path)
            print(_dumps({k: state[k] for k in ("status", "cohort", "completed_new_endpoints", "new_requested_endpoints", "decoded_requested_frames", "requested_frames", "elapsed_seconds")}), flush=True)
            report.last = now
    report.last = started - 20

    def save(item, profile=None, summary=None, error=None, frame_pts=None, dimensions=None):
        flags = ["identity_unverified"]
        details = {"statistics_are_source_evidence_only": True,
                   "statistics_do_not_determine_paint_board_shadow_or_feature_identity": True,
                   "source_coordinate_interpolation": "bilinear_RGB_at_declared_native_UV"}
        if error:
            flags.append(error)
        else:
            details["axis_statistics"] = {axis: {name: round(float(value), 6) for name, value in zip(
                ("gray_range", "maximum_absolute_gray_gradient", "strongest_gradient_offset_px",
                 "half_peak_support_sample_count", "mean_absolute_gray_gradient", "gray_std"), data)}
                for axis, data in zip(("x", "y"), summary)}
            if max(summary[:, 0]) < 5 or min(summary[:, 3]) > 10:
                flags.append("blur_or_weak_contrast_review")
            if min(abs(summary[:, 2])) > 8:
                flags.append("local_peak_offset_review")
            if any(c < 16 or c > size - 17 for c, size in zip(item["uv"], dimensions)):
                flags.append("profile_crosses_image_boundary")
            details["padding_samples_excluded_from_statistics"] = True
        dimensions = dimensions or (item["width"], item["height"])
        def axis_profile(axis):
            if profile is None:
                return None
            values = []
            for offset, color in zip(range(-16, 17), profile[axis]):
                x = item["uv"][0] + (offset if axis == 0 else 0)
                y = item["uv"][1] + (offset if axis == 1 else 0)
                values.append([round(float(v), 3) for v in color]
                              if 0 <= x <= dimensions[0] - 1 and 0 <= y <= dimensions[1] - 1 else None)
            return _dumps(values)
        pending.append((item["obs_id"], item["point_index"], item["endpoint"], PIXEL_AUDITOR,
            item["pts"], item["time_base"], _dumps(item["uv"]), source_hash, frame_pts,
            dimensions[0], dimensions[1], axis_profile(0), axis_profile(1),
            _dumps(details), _dumps(flags), "|".join(flags), 1, device,
            datetime.now(timezone.utc).isoformat()))
        state["completed_new_endpoints"] += 1
        flag_counts.update(flags)
        if len(pending) >= 250:
            flush()

    report(True)
    for item, reason in rejected:
        save(item, error=reason)
    remaining = set(groups)
    try:
        if remaining:
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                if Fraction(stream.time_base) != Fraction(1, 15360):
                    raise ValueError("Actual source stream time_base differs from original UV source contract")
                stream.thread_type = "AUTO"
                stream.codec_context.thread_count = 4
                container.seek(min(remaining), stream=stream, backward=True, any_frame=False)
                last_pts = max(remaining)
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    if frame.pts > last_pts:
                        break
                    if frame.pts not in remaining:
                        continue
                    items = groups[frame.pts]
                    remaining.remove(frame.pts)
                    dimensions = (frame.width, frame.height)
                    valid = []
                    for item in items:
                        if dimensions != (item["width"], item["height"]):
                            save(item, error="source_frame_dimensions_mismatch", frame_pts=frame.pts, dimensions=dimensions)
                        elif not (0 <= item["uv"][0] < frame.width and 0 <= item["uv"][1] < frame.height):
                            save(item, error="source_UV_out_of_frame", frame_pts=frame.pts, dimensions=dimensions)
                        else:
                            valid.append(item)
                    if valid:
                        rgb = frame.to_ndarray(format="rgb24")
                        profiles, statistics = _gpu_rgb_profiles(rgb, [i["uv"] for i in valid], device)
                        for item, profile, summary in zip(valid, profiles, statistics):
                            save(item, profile, summary, frame_pts=frame.pts, dimensions=dimensions)
                    state["decoded_requested_frames"] += 1
                    report()
        for pts in sorted(remaining):
            for item in groups[pts]:
                save(item, error="exact_source_PTS_frame_unavailable")
        state["status"] = "complete_source_evidence_audit_identity_unverified"
    except Exception as error:
        state["status"] = "failed_partial_source_evidence_preserved"
        state["error"] = f"{type(error).__name__}: {error}"
        report(True)
        raise
    finally:
        flush()
        report(True)
        connection.close()
    decision_join = join_user_decisions(database)
    if decision_join.get("linked_measurements"):
        state["direct_human_semantic_join"] = decision_join
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(_dumps(state), encoding="utf8")
        temporary.replace(status_path)
    return status_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="CUDA original RGB source-evidence audit")
    audit.add_argument("--database", required=True, type=Path)
    audit.add_argument("--video", required=True, type=Path)
    audit.add_argument("--device", default="cuda")
    audit.add_argument("--cohort", choices=("all", "explicit"), default="all")
    args = parser.parse_args()
    if args.command == "audit":
        print(audit_source_pixels(args.database, args.video, args.device, cohort=args.cohort))


def derive_reviewed_circle_packet(database, user_decisions_path, output_path=None):
    """Retire four misassigned Ad24 source points in a separate circle packet.

    The immutable parent, observation database and active solver configuration
    remain untouched. Retained JSON element literals, the six other rows and
    the geometry contract are copied exactly, without offset-derived new FIT.
    Human authority defines the target identity; native supports are assistant
    evidence and do not become precise human pixel ground truth.
    """
    parent_sha = "15df41056ee2a7dac710283f6a4aa8b9bd2df0df76a85b8e50245a827d21a853"
    evidence_sha = "868b96ff052fb073968c7bb7c5a230737c70ad7a76a2c0eb5e68f3cd6d5a2dce"
    observation_id = "native_Ad24_candidate_229582592"
    retired, retained = [0, 1, 2, 5], [3, 4, 6, 7, 8, 9]
    database, user_decisions_path = Path(database), Path(user_decisions_path)
    output_path = Path(output_path) if output_path is not None else database.parent / "reviewed_circle_sources.json"
    if not output_path.parent.is_dir():
        raise FileNotFoundError("Use an existing Data directory for the reviewed circle packet")
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        found = connection.execute(
            "SELECT * FROM source_semantic_reviews WHERE observation_id=? "
            "ORDER BY review_revision DESC, revision_timestamp DESC, review_id DESC LIMIT 1",
            (observation_id,)).fetchone()
        if found is None:
            raise ValueError("Ad24 source-role review is missing")
        review = dict(found)
    if (review["packet_sha256"] != parent_sha
            or review["review_status"] != "human_semantic_target_confirmed_source_role_misassigned"
            or review["review_origin"] != "assistant_original_pixels"
            or review["pixel_ground_truth"] or review["supervision_enabled"]):
        raise ValueError("Expected identity-only Ad24 source-role review is required")
    supports = json.loads(review["source_support_json"])
    if sorted(item["point_index"] for item in supports) != retired:
        raise ValueError("Reviewed retired indices differ from the explicit four-point retirement")
    decisions_sha = _sha(user_decisions_path)
    decisions = _read(user_decisions_path).get("decisions", [])
    matching = [d for d in decisions if observation_id in d.get("observation_ids", [])]
    if not matching:
        raise ValueError("Direct user definition for this Ad24 source row is missing")
    decision = matching[-1]
    human_review = json.loads(review["human_semantic_authority_json"])
    if (decision.get("identity_authority") != "direct_user_confirmation_of_target_boundary_identity"
            or decision.get("physical_identity") != "Ad24:outer"
            or decision.get("target_geometry", {}).get("primitive") != "circle"
            or decision.get("target_geometry", {}).get("boundary") != "outermost_visible_circular_print_stroke"
            or decision.get("old_source_role_status") != "misassigned"
            or decision.get("pixel_accuracy_confirmed") is not False
            or decision.get("model_projection_used") is not False
            or decision.get("source_pts") != review["source_pts"]
            or decision.get("time_base") != review["time_base"]
            or decision.get("image_sha256") != review["frame_sha256"]
            or decision.get("user_statement") != human_review.get("quote")):
        raise ValueError("Latest user decision must define only the actual outer black circle identity")
    parent_path = Path(review["packet_path"])
    if output_path.resolve() in (parent_path.resolve(), database.resolve(), user_decisions_path.resolve()):
        raise ValueError("Derived output cannot overwrite original evidence")
    if _sha(parent_path) != parent_sha:
        raise ValueError("Immutable circle parent packet SHA changed")
    if _sha(review["frame_path"]) != review["frame_sha256"]:
        raise ValueError("Ad24 original PNG SHA changed")
    text = parent_path.read_bytes().decode("utf-8-sig")
    parent = json.loads(text)
    contract = parent["geometry_contract"]
    same_print = contract["same_print_source_evidence"]
    if (parent.get("schema") != "rinkspace.additional-native-board-circles.v1"
            or parent.get("status") != "explicit_new_fit_target_authorized"
            or parent.get("source_video_sha256") != VIDEO_SHA
            or parent.get("time_base") != "1/15360"
            or parent.get("image_size_wh") != [1920, 1080]
            or contract.get("parameter_count") != 6
            or contract.get("active_material_parameter_count") != 5
            or contract.get("radius_binding") != "same_print_shared_radius_source_v1"
            or contract.get("shared_radius_suffix_offset") != 2
            or contract.get("inactive_legacy_Ad24_radius_suffix_offset") != 5
            or same_print.get("sha256") != evidence_sha
            or _sha(same_print["path"]) != evidence_sha):
        raise ValueError("Original six-slot/shared-radius and same-print evidence contract changed")
    rows = parent["rows"]
    selected = [i for i, row in enumerate(rows) if row["id"] == observation_id]
    if (len(rows) != 7 or len(selected) != 1 or any(row["split"] != "fit" for row in rows)
            or sum(len(row["uv"]) for row in rows) != 52):
        raise ValueError("Expected seven original FIT circle rows with 52 UVs")
    row_index = selected[0]
    row = rows[row_index]
    if (row["pts"] != review["source_pts"]
            or row["source_file_sha256"] != review["frame_sha256"]
            or row["uv"] != json.loads(review["original_uv_json"])
            or row.get("curve", {}).get("advertisement_id") != 24
            or any(len(row[name]) != 10 for name in ("uv", "sigma_px", "source_support"))
            or any(support["uv"] != point for support, point in zip(row["source_support"], row["uv"]))):
        raise ValueError("Parallel original Ad24 point arrays or source identity changed")

    decoder = json.JSONDecoder()

    def whitespace(position):
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    def members(start):
        position, spans = whitespace(start+1), {}
        while text[position] != "}":
            key, position = decoder.raw_decode(text, position)
            position = whitespace(position)
            if text[position] != ":":
                raise ValueError("Invalid parent JSON object")
            start_value = whitespace(position+1)
            _, end_value = decoder.raw_decode(text, start_value)
            spans[key] = (start_value, end_value)
            position = whitespace(end_value)
            if text[position] == ",":
                position = whitespace(position+1)
            elif text[position] != "}":
                raise ValueError("Invalid parent JSON member separator")
        return spans

    def elements(start):
        position, spans = whitespace(start+1), []
        while text[position] != "]":
            _, end_value = decoder.raw_decode(text, position)
            spans.append((position, end_value))
            position = whitespace(end_value)
            if text[position] == ",":
                position = whitespace(position+1)
            elif text[position] != "]":
                raise ValueError("Invalid parent JSON array separator")
        return spans

    top_start = whitespace(0)
    _, top_end = decoder.raw_decode(text, top_start)
    top = members(top_start)
    row_spans = elements(top["rows"][0])
    row_members = members(row_spans[row_index][0])
    edits, kept_literals = [], {}
    for name in ("uv", "sigma_px", "source_support"):
        start, end = row_members[name]
        element_spans = elements(start)
        separator = text[element_spans[0][1]:element_spans[1][0]]
        literals = [text[element_spans[i][0]:element_spans[i][1]] for i in retained]
        replacement = (text[start:element_spans[0][0]] + separator.join(literals)
                       + text[element_spans[-1][1]:end])
        edits.append((start, end, replacement))
        kept_literals[name] = literals
    for name, value in (("fit_uv_count", 48), ("fit_scalar_count", 96)):
        edits.append((*top[name], str(value)))
    retirement = {"schema": "rinkspace.reviewed-circle-source-role-retirement.v1",
        "database_path": str(database.resolve()), "source_id": review["source_id"],
        "observation_id": observation_id, "review_revision": review["review_revision"],
        "review_revision_timestamp": review["revision_timestamp"],
        "review_json_sha256": hashlib.sha256(review["review_json"].encode("utf8")).hexdigest(),
        "user_decision_id": decision.get("id"), "user_statement": decision["user_statement"],
        "human_authority_scope": "outermost_thin_black_circle_identity_only_not_precise_native_pixels",
        "native_position_support_authority": "assistant_original_pixels_not_human_precision_GT",
        "retired_points": [{"parent_index": i, "original_uv": row["uv"][i],
                            "original_sigma_px": row["sigma_px"][i]} for i in retired],
        "offset_midpoints_used_as_new_FIT": False, "new_source_UV_created": False,
        "original_parent_and_database_modified": False, "active_configuration_modified": False,
        "solver_started": False, "HELD_rows_added": 0, "other_six_rows_byte_exact": True,
        "retained_uv_sigma_support_element_literals_byte_exact": True,
        "geometry_contract_byte_exact": True, "same_print_source_evidence_sha256": evidence_sha,
        "pixel_ground_truth": False}
    extra = {"parent_packet_path": str(parent_path), "parent_packet_sha256": parent_sha,
        "source_semantic_review_revision": review["review_revision"],
        "user_decision_path": str(user_decisions_path.resolve()), "user_decision_sha256": decisions_sha,
        "old_uv_index_mapping": {"observation_id": observation_id, "retired_old_indices": retired,
            "new_to_parent_indices": retained,
            "parent_to_new_index": {str(i): (retained.index(i) if i in retained else None) for i in range(10)}},
        "retired_role_reason": "User requires the outermost thin black circle; four old UVs identify nearby internal printed ink rather than that source role. Original UV/sigma remain traceable in the immutable parent; assistant finite pixel supports are not precise human GT.",
        "source_semantic_derivation": retirement}
    if any(key in top for key in extra):
        raise ValueError("Parent already contains the derived revision fields")
    addition = ",\n" + ",\n".join("  "+json.dumps(key)+": "+json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False) for key, value in extra.items()) + "\n"
    edits.append((top_end-1, top_end-1, addition))
    derived_text = text
    for start, end, replacement in sorted(edits, reverse=True):
        derived_text = derived_text[:start]+replacement+derived_text[end:]
    derived = json.loads(derived_text)
    for i, original_row in enumerate(rows):
        expected = dict(original_row)
        if i == row_index:
            for name in ("uv", "sigma_px", "source_support"):
                expected[name] = [original_row[name][j] for j in retained]
        if derived["rows"][i] != expected:
            raise AssertionError("Unexpected derived source-row mutation")
    if derived["geometry_contract"] != contract or sum(len(r["uv"]) for r in derived["rows"]) != 48:
        raise AssertionError("Original geometry or expected FIT counts changed")
    original_text = text
    text = derived_text
    new_top = members(whitespace(0))
    new_row_spans = elements(new_top["rows"][0])
    for i, (start, end) in enumerate(row_spans):
        if i != row_index and original_text[start:end] != text[slice(*new_row_spans[i])]:
            raise AssertionError("An untouched circle row lost byte-identical JSON")
    if original_text[slice(*top["geometry_contract"])] != text[slice(*new_top["geometry_contract"])]:
        raise AssertionError("Geometry contract lost byte-identical JSON")
    new_row_members = members(new_row_spans[row_index][0])
    for name, old_literals in kept_literals.items():
        spans = elements(new_row_members[name][0])
        if [text[a:b] for a, b in spans] != old_literals:
            raise AssertionError("Retained source point elements lost byte-identical JSON")
    output_path.write_bytes(derived_text.encode("utf8"))
    if _read(output_path) != derived or _sha(parent_path) != parent_sha or _sha(user_decisions_path) != decisions_sha:
        raise AssertionError("Derived write or immutable parent/user-decision checksum changed")
    return {"path": str(output_path.resolve()), "sha256": _sha(output_path),
            "parent_packet_sha256": parent_sha, "user_decision_sha256": decisions_sha,
            "review_revision": review["review_revision"], "FIT_rows": 7, "HELD_rows": 0,
            "fit_uv_count": 48, "fit_scalar_count": 96, "retired_old_indices": retired,
            "retained_old_indices": retained, "other_six_rows_byte_exact": True,
            "retained_uv_sigma_support_literals_byte_exact": True, "geometry_contract_byte_exact": True,
            "original_parent_unchanged": True, "active_configuration_modified": False, "solver_started": False}


def decode_native_rgb(video_path, source_pts, *, expected_sha256, device="cuda"):
    """Decode one exact native PTS, without producing an image file.

    The caller can reuse the returned SHA-bound RGB for several appearance
    tests. Repeated frame work should verify the video once and decode a single
    sequence instead of calling this convenience function for every exposure.
    """
    import av
    import torch
    video_path = Path(video_path).resolve()
    actual_sha = _sha(video_path)
    if actual_sha != expected_sha256:
        raise ValueError("Native video SHA differs from the requested evidence source")
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        container.seek(int(source_pts), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < int(source_pts):
                continue
            if frame.pts != int(source_pts):
                raise ValueError(f"Requested native PTS {source_pts} is not a decoded exposure")
            original = frame.to_ndarray(format="rgb24")
            binding = {"source_video_path": str(video_path), "source_sha256": actual_sha,
                "source_pts": int(frame.pts), "time_base": str(frame.time_base),
                "shot_id": f"native-frame:{actual_sha}:{frame.pts}",
                "shot_identity_authority": "native_presentation_identity_only_camera_and_shot_unclassified",
                "source_pts_kind": "native_presentation_PTS",
                "actual_exposure_pts": None,
                "actual_exposure_authority": "not_joined_to_official_exposure_alias_table",
                "native_hw": list(original.shape[:2]), "image_domain": "native_brown_pixels",
                "decoded_native_rgb_sha256": hashlib.sha256(original.tobytes()).hexdigest()}
            rgb = torch.from_numpy(original.copy()).permute(2, 0, 1)[None].to(device).float()/255.
            return rgb, binding
    raise ValueError(f"Source exposure {source_pts} was not found")


def _appearance_features(rgb):
    """Photometric channels; these have no learned or physical identity."""
    import torch
    luminance = .299*rgb[:, 0:1] + .587*rgb[:, 1:2] + .114*rgb[:, 2:3]
    red = rgb[:, 0:1] - .5*(rgb[:, 1:2]+rgb[:, 2:3])
    blue = rgb[:, 2:3] - .5*(rgb[:, 0:1]+rgb[:, 1:2])
    return torch.cat((luminance, red, blue), 1)


def _actual_rgb(rgb, binding):
    from .vision_geometry import _cuda_tensor, _source_binding
    import torch
    rgb = _cuda_tensor(rgb, dimensions=4)
    binding = _source_binding(binding)
    if rgb.shape[:2] != (1, 3) or tuple(binding.get("native_hw", ())) != tuple(rgb.shape[-2:]):
        raise ValueError("Appearance source must be the full bound native [1,3,H,W] RGB")
    if binding.get("image_domain") != "native_brown_pixels" or binding.get("crop_xyxy") is not None:
        raise ValueError("Appearance evidence must retain original native pixel coordinates")
    if not rgb.is_floating_point() or not bool(torch.isfinite(rgb).all()) or bool(((rgb < 0)|(rgb > 1)).any()):
        raise ValueError("Actual RGB must be finite, floating point and in [0,1]")
    pixels=(rgb.detach().cpu().numpy().transpose(0,2,3,1)[0]*255).round().astype(np.uint8)
    digest=hashlib.sha256(pixels.tobytes()).hexdigest()
    expected=binding.get("decoded_native_rgb_sha256")
    if expected is not None and expected != digest:
        raise ValueError("RGB differs from the bound decoded native exposure; augmentation/projection is not source evidence")
    binding.update(decoded_native_rgb_sha256=digest,
        RGB_hash_authority="matches_caller_decoded_exposure_hash" if expected is not None
                           else "actual_tensor_hash_only_native_video_decode_claim_requires_verification")
    return rgb, binding


def make_advertisement_appearance_template(rgb, roi_xyxy, *, source_binding,
                                         content_family, instance_candidates):
    """Copy an original image ROI as an appearance reference, never a label.

    Instance candidates express authoritative same-content aliases, for
    example [10,23]. A match cannot distinguish those physical installations.
    The ROI is only a search crop; its four corners are not ad boundaries.
    """
    rgb, binding = _actual_rgb(rgb, source_binding)
    x0, y0, x1, y1 = map(int, roi_xyxy)
    h, w = rgb.shape[-2:]
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Appearance reference crop must lie inside its native frame")
    patch = rgb[:, :, y0:y1, x0:x1].clone()
    pixels = (patch.detach().cpu().numpy().transpose(0, 2, 3, 1)[0]*255).round().astype(np.uint8)
    return {"rgb": patch, "content_family": str(content_family),
        "instance_candidates": sorted(set(map(int, instance_candidates))),
        "reference": {"source_binding": binding, "crop_xyxy": [x0,y0,x1,y1],
            "rgb_patch_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "purpose": "appearance_reference_only", "crop_is_ad_boundary": False},
        "physical_instance_verified_by_template": False, "pixel_ground_truth": False}


def load_advertisement_appearance_template(path, *, expected_sha256, content_family,
                                         instance_candidates, package_part=None,
                                         expected_part_sha256=None, device="cuda"):
    """Read original photographic content, including a photo embedded in PPT.

    Only decoded image bytes are read from a package. Slide position, dimensions,
    transforms and schematic coordinates are deliberately unavailable here.
    """
    import cv2
    import torch
    from zipfile import ZipFile
    path = Path(path).resolve()
    if _sha(path) != expected_sha256:
        raise ValueError("Appearance reference file differs from its recorded SHA")
    if package_part is None:
        data = path.read_bytes()
    else:
        if not package_part.startswith("ppt/media/"):
            raise ValueError("Only embedded image content may be used as an appearance reference")
        with ZipFile(path) as package:
            data = package.read(package_part)
    part_sha = hashlib.sha256(data).hexdigest()
    if expected_part_sha256 is not None and part_sha != expected_part_sha256:
        raise ValueError("Appearance image bytes differ from their recorded SHA")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Appearance reference is not a readable photographic image")
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return {"rgb": torch.from_numpy(original.copy()).permute(2,0,1)[None].to(device).float()/255.,
        "content_family": str(content_family),
        "instance_candidates": sorted(set(map(int, instance_candidates))),
        "reference": {"path": str(path), "sha256": expected_sha256,
            "package_part": package_part, "part_sha256": part_sha,
            "purpose": "photographic_appearance_reference_only",
            "PPT_layout_or_dimensions_used": False},
        "physical_instance_verified_by_template": False, "pixel_ground_truth": False}


def match_advertisement_appearance(rgb, template, *, source_binding, widths_native_px,
                                  aspect_factors=(.8, 1., 1.2), search_roi=None,
                                  downsample=4, candidates_per_scale=3, maximum_candidates=12):
    """CUDA multi-scale photometric NCC in an actual source image.

    Returns search ROIs and continuous appearance scores, not accepted IDs or
    boundaries. White backgrounds, repeated content, occlusion and perspective
    can defeat NCC; each retained location requires actual-image review. No
    camera, rink, projection, schematic position or geometric prior is accepted.
    """
    import torch
    import torch.nn.functional as F
    rgb, binding = _actual_rgb(rgb, source_binding)
    reference = template["rgb"]
    if reference.device != rgb.device or reference.shape[:2] != (1,3):
        raise ValueError("Photographic template must be [1,3,h,w] on the same CUDA device")
    if downsample < 1 or int(downsample) != downsample:
        raise ValueError("Downsample must be a positive integer")
    for name,value in (("candidates_per_scale",candidates_per_scale),("maximum_candidates",maximum_candidates)):
        if isinstance(value,bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    h, w = rgb.shape[-2:]
    x0,y0,x1,y1 = map(int, search_roi or (0,0,w,h))
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Search crop must be within the bound actual source frame")
    size = (max(2,(y1-y0)//downsample), max(2,(x1-x0)//downsample))
    image = _appearance_features(F.interpolate(rgb[:,:,y0:y1,x0:x1], size=size, mode="bilinear", align_corners=False))
    # Constant offsets do not change NCC. Remove them before convolution so a
    # low-texture white board cannot magnify cancellation in the numerator.
    image=image-image.mean((-2,-1),keepdim=True)
    # Local variance by FP32 E[x²]-E[x]² can catastrophically cancel on smooth
    # boards. Double-precision integral moments are cheap, reused at every scale
    # and keep such windows from acquiring an artificial perfect NCC score.
    integral=F.pad(image.double().cumsum(-1).cumsum(-2),(1,0,1,0))
    integral_squared=F.pad(image.double().square().cumsum(-1).cumsum(-2),(1,0,1,0))
    def window_sum(values,kh,kw):
        return values[:,:,kh:,kw:]-values[:,:,:-kh,kw:]-values[:,:,kh:,:-kw]+values[:,:,:-kh,:-kw]
    ratio = reference.shape[-2]/reference.shape[-1]
    found=[]
    with torch.inference_mode():
        for native_width in widths_native_px:
            if not np.isfinite(native_width) or native_width <= 0:
                raise ValueError("Every search width must be finite and positive native pixels")
            for factor in aspect_factors:
                if not np.isfinite(factor) or factor <= 0:
                    raise ValueError("Aspect factors must be finite and positive")
                kw=max(3,int(round(native_width/downsample)))
                kh=max(3,int(round(native_width*ratio*factor/downsample)))
                if kw > image.shape[-1] or kh > image.shape[-2]:
                    continue
                feature = _appearance_features(F.interpolate(reference, size=(kh,kw), mode="bilinear", align_corners=False))
                centered=feature-feature.mean((-2,-1),keepdim=True)
                energy=centered.square().sum()
                if float(energy) < 1e-8:
                    continue
                # Per-channel mean removal prevents plain white board/background
                # from supplying the score. Relative channel contrast is retained.
                with torch.backends.cudnn.flags(allow_tf32=False):
                    numerator=F.conv2d(image,centered).double()
                total=window_sum(integral,kh,kw)
                squares=window_sum(integral_squared,kh,kw)
                local_energy=(squares-total.square()/(kh*kw)).sum(1,keepdim=True).clamp_min(0)
                score=(numerator/(energy*local_energy).clamp_min(1e-10).sqrt()).clamp(-1,1)
                score=torch.where(local_energy/(3*kh*kw) > 1e-5,score,torch.full_like(score,-1)).float()
                # Independent local maxima; no repeated top-k of one logo pixel.
                pooled=F.max_pool2d(score, max(3,(min(kh,kw)//2)*2+1),stride=1,
                                    padding=max(3,(min(kh,kw)//2)*2+1)//2)
                peaks=score.masked_fill(score < pooled,-2).flatten()
                values, indices=peaks.topk(min(int(candidates_per_scale),peaks.numel()))
                score_w=score.shape[-1]
                sx,sy=(x1-x0)/size[1],(y1-y0)/size[0]
                for value,index in zip(values.cpu().tolist(),indices.cpu().tolist()):
                    if value <= -1:
                        continue
                    yy,xx=divmod(index,score_w)
                    found.append({"roi_xyxy_native_px": [x0+xx*sx,y0+yy*sy,
                                                          x0+(xx+kw)*sx,y0+(yy+kh)*sy],
                        "appearance_ncc": float(value), "template_width_native_px": float(kw*sx),
                        "aspect_factor": float(factor)})
    chosen=[]
    for item in sorted(found,key=lambda p:p["appearance_ncc"],reverse=True):
        box=item["roi_xyxy_native_px"]
        duplicate=False
        for old in chosen:
            other=old["roi_xyxy_native_px"]
            intersection=max(0,min(box[2],other[2])-max(box[0],other[0]))*max(0,min(box[3],other[3])-max(box[1],other[1]))
            union=(box[2]-box[0])*(box[3]-box[1])+(other[2]-other[0])*(other[3]-other[1])-intersection
            duplicate |= intersection/max(union,1e-6) > .35
        if duplicate:
            continue
        item.update(content_family=template["content_family"],
                    physical_instance_candidates=template["instance_candidates"],
                    state="appearance_candidate_requires_original_image_identity_review",
                    physical_instance_verified=False, contour_verified=False,
                    pixel_ground_truth=False, supervision_enabled=False, solver_admitted=False)
        chosen.append(item)
        if len(chosen) >= maximum_candidates:
            break
    return {"schema":"rinkspace.image-only-advertisement-appearance-candidates.v1",
        "source_binding":binding, "appearance_reference":template["reference"],
        "search_roi_native_px":[x0,y0,x1,y1], "downsample":int(downsample),
        "candidates":chosen, "score_is_identity_confirmation":False,
        "roi_is_ad_boundary":False, "world_or_camera_projection_used":False,
        "numerical_method":"FP64_integral_local_moments_FP32_no_TF32_crosscorrelation",
        "minimum_patch_channel_variance":1e-5}


def extract_advertisement_ink_contours(rgb, roi_xyxy, *, source_binding,
                                      content_family, instance_candidates,
                                      ignored_regions=(), dark_delta=.14,
                                      red_r_minus_g=.065, red_r_minus_b=.02,
                                      minimum_channel_range=.12, minimum_contour_points=8,
                                      background_colour_family=None,
                                      colour_component_seed_uv=None,
                                      background_colour_thresholds=None):
    """Actual native ink isophotes, with gaps and search-crop cuts preserved.

    The mask is computed on CUDA. OpenCV traces its topology on CPU without
    morphology. No rectangle or missing outer stroke is fabricated. Isophotes
    need semantic and threshold/blur review before they can become measured ad
    boundaries; nothing returned here is a training label or solver input.
    """
    import cv2
    import torch
    from .vision_geometry import measure_boundary_profiles
    rgb,binding=_actual_rgb(rgb,source_binding)
    h,w=rgb.shape[-2:]
    x0,y0,x1,y1=map(int,roi_xyxy)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Ink search crop must be within the actual source frame")
    patch=rgb[0,:,y0:y1,x0:x1]
    gray=.299*patch[0]+.587*patch[1]+.114*patch[2]
    background=torch.quantile(gray,.85)
    masks={"dark":gray < background-dark_delta,
        "red":((patch[0]-patch[1] > red_r_minus_g)&(patch[0]-patch[2] > red_r_minus_b)&
                (patch.amax(0)-patch.amin(0) > minimum_channel_range)&(patch[0] > .2))}
    colour_limits={"blue":{"B_minus_R":.04,"B_minus_G":.015},
        "magenta":{"R_minus_G":.045,"B_minus_G":.045},
        "red":{"R_minus_G":.045,"R_minus_B":.025}}
    if background_colour_thresholds is not None:
        if background_colour_family not in colour_limits:
            raise ValueError("Colour thresholds need their explicit print background family")
        for key,value in background_colour_thresholds.items():
            if key not in colour_limits[background_colour_family] or not np.isfinite(value) or not 0<value<1:
                raise ValueError("Invalid native RGB opponent threshold")
            colour_limits[background_colour_family][key]=float(value)
    colour_definitions={
        "blue":(patch[2]-patch[0] > colour_limits["blue"]["B_minus_R"])&(patch[2]-patch[1] > colour_limits["blue"]["B_minus_G"]),
        "magenta":(patch[0]-patch[1] > colour_limits["magenta"]["R_minus_G"])&(patch[2]-patch[1] > colour_limits["magenta"]["B_minus_G"]),
        "red":(patch[0]-patch[1] > colour_limits["red"]["R_minus_G"])&(patch[0]-patch[2] > colour_limits["red"]["R_minus_B"]),
    }
    if background_colour_family is not None:
        if background_colour_family not in colour_definitions:
            raise ValueError("Print background colour must be blue, magenta or red")
        # Only observed colour transitions define this mask. It does not use a
        # text bounding box, rectangle fit, morphology or a projected ad panel.
        masks={f"printed_background_{background_colour_family}":colour_definitions[background_colour_family]}
    results=[]
    for channel,mask in masks.items():
        binary=mask.to(torch.uint8).cpu().numpy()*255
        if colour_component_seed_uv is not None:
            if background_colour_family is None:
                raise ValueError("An original colour seed is only valid for a print background")
            sx,sy=map(int,colour_component_seed_uv)
            if not (x0<=sx<x1 and y0<=sy<y1):
                raise ValueError("Original colour seed lies outside its actual source crop")
            _,labels=cv2.connectedComponents((binary>0).astype(np.uint8),connectivity=8)
            selected=int(labels[sy-y0,sx-x0])
            if selected==0:
                raise ValueError("Original seed does not contain the specified print colour")
            binary=(labels==selected).astype(np.uint8)*255
        outside_support=None
        if background_colour_family is not None:
            # White letters inside a coloured print are not its panel border.
            # When text meets an image-clipped face, RETR_TREE alone incorrectly
            # makes those letters part of the exterior contour. Identify actual
            # complement regions connected to visible crop margins, excluding
            # the image-clipped faces. This classifies existing mask points; it
            # neither fills an ink hole nor creates a new boundary pixel.
            _,regions=cv2.connectedComponents((binary==0).astype(np.uint8),connectivity=8)
            margins=[]
            if y0>0:margins.extend(regions[0].tolist())
            if y1<h:margins.extend(regions[-1].tolist())
            if x0>0:margins.extend(regions[:,0].tolist())
            if x1<w:margins.extend(regions[:,-1].tolist())
            board_outside=np.isin(regions,list(set(margins)-{0}))
            padded=np.pad(board_outside,1,constant_values=False)
            outside_support=np.zeros_like(board_outside)
            for dy in range(3):
                for dx in range(3):
                    outside_support |= padded[dy:dy+board_outside.shape[0],dx:dx+board_outside.shape[1]]
        contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for index,contour in enumerate(contours):
            coordinates=contour[:,0,:].astype(np.float32)+[x0,y0]
            if len(coordinates) < minimum_contour_points:
                continue
            good=(coordinates[:,0] > x0+1)&(coordinates[:,0] < x1-2)&(coordinates[:,1] > y0+1)&(coordinates[:,1] < y1-2)
            if outside_support is not None:
                local=(coordinates-[x0,y0]).astype(int)
                good &= outside_support[local[:,1],local[:,0]]
            for left,top,right,bottom in ignored_regions:
                good &= ~((coordinates[:,0]>=left)&(coordinates[:,0]<=right)&(coordinates[:,1]>=top)&(coordinates[:,1]<=bottom))
            # Roll a closed contour to an omitted sample, then split at every
            # cut. This prevents false joins across people or off-crop ink.
            closed=bool(good.all())
            if not closed and good.any():
                start=int(np.flatnonzero(~good)[0])
                coordinates=np.roll(coordinates,-start,axis=0)
                good=np.roll(good,-start)
            runs=[]
            for indices in np.split(np.flatnonzero(good),np.flatnonzero(np.diff(np.flatnonzero(good))>1)+1):
                if len(indices)>=minimum_contour_points:
                    runs.append(coordinates[indices])
            for fragment,points in enumerate(runs):
                tangent=(np.roll(points,-1,axis=0)-np.roll(points,1,axis=0)) if closed else np.gradient(points,axis=0)
                normal=np.stack((-tangent[:,1],tangent[:,0]),axis=1)
                lengths=np.linalg.norm(normal,axis=1)
                normal=np.divide(normal,lengths[:,None],out=np.zeros_like(normal),where=lengths[:,None]>0)
                pts=torch.as_tensor(points,device=rgb.device,dtype=torch.float32)
                ns=torch.as_tensor(normal,device=rgb.device,dtype=torch.float32)
                profiles=measure_boundary_profiles(rgb,torch.zeros((1,1,h,w),device=rgb.device),pts,ns,
                    class_index=0,search_radius=5.,sample_step=.5,rgb_source_binding=binding,
                    probability_source_binding=binding,projection_source_binding=binding,
                    segment_ids=torch.full((len(points),),len(results),device=rgb.device,dtype=torch.int64))
                def summary(value):
                    values=value.detach().cpu().numpy().astype(float)
                    finite=values[np.isfinite(values)]
                    return {"count":len(finite),"minimum":float(finite.min()) if len(finite) else None,
                            "median":float(np.median(finite)) if len(finite) else None,
                            "maximum":float(finite.max()) if len(finite) else None}
                results.append({"id":f"{content_family}:{channel}:{index}:{fragment}",
                    "uv":points.astype(int).tolist(),"normal":normal.tolist(),"closed":closed,
                    "primitive":"free_curve","channel":channel,"hierarchy":hierarchy[0,index].tolist(),
                    "state":"native_ink_isophote_requires_identity_and_outer_stroke_review",
                    "source_profile_summary":{key:summary(profiles["rgb_peak"][value]) for key,value in
                        (("rgb_contrast","contrast"),("rgb_peak_gradient","gradient"),
                         ("rgb_transition_width_native_px","fwhm_px"),("rgb_peak_offset_native_px","offset_px"))},
                    "local_sharpness_summary":{key:summary(value) for key,value in
                        profiles["local_sharpness_at_projection"].items()},
                    "RGB_candidate_evidence_point_indices":torch.nonzero(
                        profiles["rgb_peak"]["candidate_evidence_usable"],as_tuple=True)[0].cpu().tolist(),
                    "review_flag_point_indices":{key:torch.nonzero(value,as_tuple=True)[0].cpu().tolist()
                        for key,value in profiles["flags"].items()},
                    "typed_semantic_probabilities_available":False,
                    "isophote_role":"printed_background_colour_boundary_candidate" if background_colour_family
                                     else "printed_ink_isophote_boundary_candidate",
                    "profile_resampling":{"search_radius_native_px":5.,"sample_step_native_px":.5},
                    "semantic_identity_verified":False,"pixel_ground_truth":False,
                    "supervision_enabled":False,"solver_admitted":False})
    return {"schema":"rinkspace.image-only-advertisement-native-ink-contours.v1",
        "source_binding":binding,"content_family":str(content_family),
        "physical_instance_candidates":sorted(set(map(int,instance_candidates))),
        "roi_xyxy_native_px":[x0,y0,x1,y1],"ignored_regions":_jsonable(ignored_regions),
        "isophote_parameters":{"background_luminance_85_quantile":float(background),
            "dark_luminance_delta":dark_delta,"red_r_minus_g":red_r_minus_g,
            "red_r_minus_b":red_r_minus_b,"minimum_channel_range":minimum_channel_range,
            "printed_background_colour_family":background_colour_family,
            "colour_component_seed_native_uv":_jsonable(colour_component_seed_uv),
            "background_colour_thresholds":colour_limits if background_colour_family else None},
        "contours":results,"mask_device":str(rgb.device),"topology_backend":"OpenCV_CPU",
        "morphological_edge_generation_used":False,"world_or_camera_projection_used":False,
        "background_border_scope":"existing_colour_isophote_adjacent_to_complement_connected_to_visible_board_margins; image_clipped_faces_excluded" if background_colour_family else None,
        "whole_ad_print_extent_verified":False,"pixel_ground_truth":False,
        "supervision_enabled":False,"solver_admitted":False}


def iter_advertisement_ink_candidates(packet_path):
    """Normalize current source contours; rejected/navigation history is skipped.

    IDs retain their explicit candidate status. The source authority and mask
    parameters survive loading; neither an old model initializer nor a search
    crop becomes part of this image observation interface.
    """
    packet_path=Path(packet_path).resolve()
    digest=_sha(packet_path)
    packet=_read(packet_path)
    for row in packet.get("sources",[]):
        if not row.get("contour_generation_no_PPT_or_world_projection_used"):
            raise ValueError("Contour packet lacks an explicit original-pixel provenance contract")
        requested=row.get("requested_advertisement_number",row.get("advertisement_number"))
        instances=row.get("physical_instance_candidates",[requested] if requested is not None else [])
        w,h=row["native_dimensions_wh"]
        binding=dict(row.get("image_only_packet",{}).get("source_binding",{}))
        binding.update(source_video_path=row["source_video_path"],source_sha256=row["source_video_sha256"],
            source_pts=int(row["source_pts"]),time_base=str(Fraction(row["time_base"])),
            native_hw=[h,w],image_domain="native_brown_pixels",
            decoded_native_rgb_sha256=row["decoded_native_RGB_sha256"])
        binding.setdefault("shot_id",f"native-frame:{row['source_pts']}")
        for contour in row.get("contours",[]):
            points=_uv(contour["uv"])
            normals=contour.get("normal")
            if normals is not None:
                normals=_uv(normals)
                if len(normals)!=len(points):
                    raise ValueError("Source contour normals differ from the actual UV count")
            yield {"id":f"{row['source_video_sha256']}:{row['source_pts']}:{contour['id']}",
                "kind":"ad_boundary","uv":points,"normal":normals,
                "primitive":contour.get("primitive","free_curve"),"closed":bool(contour.get("closed",False)),
                "source_binding":binding,"packet_path":str(packet_path),"packet_sha256":digest,
                "actual_exposure_pts":row.get("actual_exposure_pts"),
                "official_exposure":row.get("official_exposure"),
                "official_exposure_proof":row.get("official_exposure_proof"),
                "requested_advertisement_number":requested,"physical_instance_candidates":instances,
                "content_family":row.get("content_family"),"channel":contour.get("channel"),
                "hierarchy":contour.get("hierarchy"),"state":contour["state"],
                "authority":row.get("source_specific_identity_authority"),
                "physical_instance_human_confirmed":bool(row.get("source_instance_identity_human_confirmed",False)),
                "source_profile_summary":contour.get("source_profile_summary",{}),
                "review_flag_point_indices":contour.get("review_flag_point_indices",{}),
                "RGB_candidate_evidence_point_indices":contour.get("RGB_candidate_evidence_point_indices",
                    contour.get("source_evidence_usable_point_indices",[])),
                "whole_print_visible_in_original_review":row.get("appearance_full_print_visible_assistant"),
                "whole_ad_print_extent_verified":bool(row.get("full_print_extent_verified",False)),
                "isophote_parameters":row.get("raw_isophote_parameters",contour.get("exact_isophote_parameters",{})),
                "world_or_camera_projection_used":False,"pixel_ground_truth":False,
                "supervision_enabled":False,"solver_admitted":False}


def iter_complete_source_ad_boundaries(packet_path, *, include_context=False):
    """Read all observed ad boundaries in a reviewed original full source image.

    Permanent human identity stays authoritative. Pixel curves remain original
    image observations with explicit visibility gaps and edge-quality limits;
    this adapter never converts them to a solver or training admission.
    """
    packet_path=Path(packet_path).resolve()
    packet=_read(packet_path)
    if packet.get("schema")!="rinkspace.source-advertisements-complete-image-observations.v1":
        raise ValueError("Expected complete original-image advertisement observations")
    if packet.get("world_or_camera_projection_used") or packet.get("text_bbox_used_as_panel_geometry"):
        raise ValueError("Source ad boundary packet must contain actual image evidence only")
    digest=_sha(packet_path)
    for ad in packet["advertisements"]:
        for component in ad["components"]:
            for contour in component["contours"]:
                draw=bool(contour.get("source_review_draw_boundary",False))
                if not include_context and not draw:
                    continue
                points=_uv(contour["uv"])
                normals=_uv(contour["normal"])
                if len(points)!=len(normals):
                    raise ValueError("An observed boundary lost its per-point native normal")
                yield {"id":contour["id"],"kind":"ad_boundary","uv":points,"normal":normals,
                    "primitive":contour.get("primitive","free_curve"),"closed":bool(contour["closed"]),
                    "source_binding":packet["source"],"packet_path":str(packet_path),"packet_sha256":digest,
                    "advertisement_number":ad["advertisement_number"],"content_id":ad["content_id"],
                    "physical_identity_human_confirmed":ad["physical_identity_human_confirmed"],
                    "physical_identity_authority":ad["physical_identity_authority"],
                    "component_id":component["component_id"],"boundary_role":contour["boundary_role"],
                    "observed_kind":component["observed_kind"],"ignored_regions":component["ignored_regions"],
                    "source_profile_summary":contour.get("source_profile_summary",{}),
                    "review_flag_point_indices":contour.get("review_flag_point_indices",{}),
                    "source_review_draw_boundary":draw,"pixel_boundary_accuracy_verified":False,
                    "pixel_ground_truth":False,"supervision_enabled":False,"solver_admitted":False,
                    "world_or_camera_projection_used":False}


def audit_source_reliable_exposures(video_path, reference_sample, output_path, *,
                                    timeline_path, source_annotations_path,
                                    radius_seconds=2., device="cuda",
                                    flow_analysis_hw=(160,480), flow_updates=8):
    """Read every native frame around a human source, without crossing its cut.

    Frozen forward/backward RGB flow transports search regions only. Actual
    profiles are measured again in every original exposure; transported guides
    and edge maxima never become annotations. The whole requested native window
    is indexed, including the explicitly excluded other-shot tail. No frame,
    video, fitted geometry, training label or solver configuration is written.
    """
    import av
    import torch
    import torch.nn.functional as F
    from .vision_geometry import GPUFlow,measure_boundary_profiles
    video_path=Path(video_path).resolve()
    source_sha=_sha(video_path)
    if source_sha!=reference_sample["source_hash"]:
        raise ValueError("Continuous source differs from the human annotation video")
    timeline=_read(timeline_path)
    if timeline["source_sha256"]!=source_sha:
        raise ValueError("Shot timeline belongs to a different original video")
    clock=Fraction(reference_sample["time_base"])
    annotations_sha_start=_sha(source_annotations_path)
    target=int(reference_sample["source_pts"])
    shot=next((r for r in timeline["intervals"] if r["start_pts"]<=target<r["end_pts"]),None)
    if shot is None or shot["interval_type"]!="shot":
        raise ValueError("The human source is not inside an identified continuous shot interval")
    if not np.isfinite(radius_seconds) or radius_seconds<=0:
        raise ValueError("Source window radius must be finite and positive")
    margin=int(Fraction(str(radius_seconds))/clock)
    window_start,window_end=target-margin,target+margin
    groups=[a for a in reference_sample["annotations"]
            if a["feature_id"] in ("Lane1","F1:outer")]
    if not groups:
        raise ValueError("No authoritative F1/Lane1 original search guides exist")
    frames=[]
    with av.open(str(video_path)) as container:
        stream=container.streams.video[0]
        if Fraction(stream.time_base)!=clock:
            raise ValueError("Native source timebase differs from its human annotation")
        container.seek(max(0,window_start),stream=stream,backward=True,any_frame=False)
        previous_small=None
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts<window_start:continue
            if frame.pts>=window_end:break
            pixels=frame.to_ndarray(format="rgb24")
            rgb=torch.from_numpy(pixels.copy()).permute(2,0,1)[None].to(device).float()/255.
            small=F.interpolate(rgb,size=(36,64),mode="bilinear",align_corners=False)
            difference=None if previous_small is None else float((small-previous_small).abs().mean())
            frames.append({"pts":int(frame.pts),"pixels":pixels,
                "rgb_sha256":hashlib.sha256(pixels.tobytes()).hexdigest(),
                "adjacent_global_RGB_MAD":difference})
            previous_small=small
    target_index=next((i for i,f in enumerate(frames) if f["pts"]==target),None)
    if target_index is None:
        raise ValueError("The exact human reference PTS was not decoded")
    if reference_sample.get("frame_sha256") and _sha(reference_sample["frame_path"])!=reference_sample["frame_sha256"]:
        raise ValueError("The original human frame image checksum changed")
    from PIL import Image
    original=np.asarray(Image.open(reference_sample["frame_path"]).convert("RGB"))
    if not np.array_equal(original,frames[target_index]["pixels"]):
        raise ValueError("Human original image and native source exposure are not byte-exact")
    h,w=original.shape[:2]
    indices=[i for i,f in enumerate(frames) if shot["start_pts"]<=f["pts"]<shot["end_pts"]]
    if target_index not in indices:
        raise ValueError("Original reference left its own continuous shot")
    ordinary=[frames[i]["adjacent_global_RGB_MAD"] for i in indices[1:]
              if frames[i]["adjacent_global_RGB_MAD"] is not None]
    typical=float(np.median(ordinary)) if ordinary else 0.
    cut_candidates=[{"native_pts":f["pts"],"adjacent_global_RGB_MAD":f["adjacent_global_RGB_MAD"]}
        for f in frames[1:] if f["adjacent_global_RGB_MAD"] is not None and
        f["adjacent_global_RGB_MAD"]>max(.05,8*typical)]
    # An unexpected source discontinuity also limits propagation. Its scene/camera
    # identity is not inferred; the raw evidence is retained for human review.
    prior=[c["native_pts"] for c in cut_candidates if c["native_pts"]<=target]
    following=[c["native_pts"] for c in cut_candidates if c["native_pts"]>target]
    safe_start=max(shot["start_pts"],max(prior) if prior else shot["start_pts"])
    safe_end=min(shot["end_pts"],min(following) if following else shot["end_pts"])
    indices=[i for i in indices if safe_start<=frames[i]["pts"]<safe_end]
    flow=GPUFlow(device=device,num_updates=flow_updates)
    offsets=[0]
    for group in groups:offsets.append(offsets[-1]+len(group["uv"]))
    initial_points=torch.tensor([q for a in groups for q in a["uv"]],device=device,dtype=torch.float32)
    initial_normals=torch.tensor([q for a in groups for q in a["normal"]],device=device,dtype=torch.float32)
    outputs={}
    empty_prob=torch.zeros((1,1,h,w),device=device)
    crop=(0,max(0,int(initial_points[:,1].min())-120),w,h)
    def tensor_frame(index):
        return torch.from_numpy(frames[index]["pixels"].copy()).permute(2,0,1)[None].to(device).float()/255.
    def binding(index):
        return {"source_video_path":str(video_path),"source_sha256":source_sha,
            "source_pts":frames[index]["pts"],"time_base":str(clock),"shot_id":shot["id"],
            "native_hw":[h,w],"image_domain":"native_brown_pixels",
            "decoded_native_rgb_sha256":frames[index]["rgb_sha256"]}
    def values(tensor):
        array=tensor.detach().cpu().numpy().astype(float)
        return [float(x) if np.isfinite(x) else None for x in array]
    def summary(tensor,mask):
        array=tensor[mask].detach().cpu().numpy().astype(float)
        array=array[np.isfinite(array)]
        return {"count":len(array),"median":float(np.median(array)) if len(array) else None,
            "p25":float(np.quantile(array,.25)) if len(array) else None,
            "p75":float(np.quantile(array,.75)) if len(array) else None}
    def sample_field(field,points):
        grid=points.clone();grid[:,0]=grid[:,0]*2/(w-1)-1;grid[:,1]=grid[:,1]*2/(h-1)-1
        return F.grid_sample(field,grid[None,:,None],align_corners=True,
                             padding_mode="zeros")[0,:,:,0].T
    def inspect(index,rgb,points,normals,alive,fb_error):
        bind=binding(index)
        profile=measure_boundary_profiles(rgb,empty_prob,points,normals,class_index=0,
            search_radius=18.,sample_step=.5,rgb_source_binding=bind,
            probability_source_binding=bind,projection_source_binding=bind)
        result=[]
        for gi,group in enumerate(groups):
            section=slice(offsets[gi],offsets[gi+1]);valid=alive[section]&profile["valid_mask"][section].all(1)&~profile["flags"]["invalid_point_or_normal"][section]
            peak=profile["rgb_peak"];narrow=valid&(peak["contrast"][section]>=.04)&(peak["fwhm_px"][section]<=3.)&(peak["competing_peaks"][section]==0)
            region_rgb=profile["rgb_profile"][section];position=profile["offsets_native_px"]
            local_channels={"dark":-region_rgb.mean(-1),
                "blue_excess":region_rgb[:,:,2]-.5*(region_rgb[:,:,0]+region_rgb[:,:,1]),
                "red_excess":region_rgb[:,:,0]-.5*(region_rgb[:,:,1]+region_rgb[:,:,2])}
            colour_summary={}
            for name,channel in local_channels.items():
                flank=(position.abs()>=12)[None]&profile["valid_mask"][section]
                weights=flank.float();mass=weights.sum(1).clamp_min(1)
                mean_x=(weights*position).sum(1)/mass;mean_y=(weights*channel).sum(1)/mass
                slope=(weights*(position-mean_x[:,None])*(channel-mean_y[:,None])).sum(1)/(weights*(position-mean_x[:,None]).square()).sum(1).clamp_min(1e-8)
                residual=channel-mean_y[:,None]-slope[:,None]*(position-mean_x[:,None])
                central=(position.abs()<=12)[None]&profile["valid_mask"][section]
                amplitude=torch.where(central,residual,-torch.inf).amax(1)-torch.where(central,residual,torch.inf).amin(1)
                colour_summary[name]=summary(amplitude,valid)
            result.append({"group_id":group["group_id"],"feature_id":group["feature_id"],
                "human_feature_identity_preserved":True,"original_human_guide_point_count":offsets[gi+1]-offsets[gi],
                "forward_backward_valid_search_points":int(alive[section].sum()),
                "full_original_RGB_profiles":int(valid.sum()),"sharp_single_RGB_edge_candidate_count":int(narrow.sum()),
                "RGB_contrast":summary(peak["contrast"][section],valid),
                "RGB_gradient_native_px":summary(peak["gradient"][section],valid),
                "RGB_transition_FWHM_native_px":summary(peak["fwhm_px"][section],valid),
                "RGB_competing_peak_count":summary(peak["competing_peaks"][section],valid),
                "local_sharpness":{name:summary(v[section],valid) for name,v in profile["local_sharpness_at_projection"].items() if name!="patch_complete"},
                "dark_blue_red_detrended_peak_to_peak":colour_summary,
                "FB_error_native_px":summary(fb_error[section],alive[section]),
                "transported_search_uv_only":points[section].detach().cpu().tolist(),
                "transported_search_normal_only":normals[section].detach().cpu().tolist(),
                "pointwise":{"FB_valid":alive[section].detach().cpu().tolist(),
                    "profile_valid":valid.detach().cpu().tolist(),"RGB_contrast":values(peak["contrast"][section]),
                    "RGB_gradient":values(peak["gradient"][section]),"RGB_FWHM_native_px":values(peak["fwhm_px"][section])},
                "RGB_peak_does_not_prove_paint_identity":True,"is_training_annotation":False,"is_solver_point":False})
        outputs[index]={"source_pts":frames[index]["pts"],"time_base":str(clock),
            "decoded_native_RGB_sha256":frames[index]["rgb_sha256"],"source_sha256":source_sha,
            "actual_exposure_pts":None,"official_exposure_identity":"unknown_no_original_pixel_alias_proof_for_this_warmup_interval",
            "groups":result}
    reference_rgb=tensor_frame(target_index)
    alive=torch.ones(len(initial_points),device=device,dtype=torch.bool)
    inspect(target_index,reference_rgb,initial_points,initial_normals,alive,torch.zeros(len(initial_points),device=device))
    started=time.time();completed=1
    for direction in (-1,1):
        current_index=target_index;current_rgb=reference_rgb
        points=initial_points.clone();normals=initial_normals.clone();alive=torch.ones(len(points),device=device,dtype=torch.bool)
        requested=[i for i in indices if i<target_index][::-1] if direction<0 else [i for i in indices if i>target_index]
        for index in requested:
            next_rgb=tensor_frame(index)
            if direction<0:
                pair=flow.compute(next_rgb,current_rgb,analysis_hw=flow_analysis_hw,crop_xyxy=crop,source0=binding(index),source1=binding(current_index))
                displacement,pair_valid,pair_error=pair.backward,pair.valid_backward,pair.fb_error_backward
            else:
                pair=flow.compute(current_rgb,next_rgb,analysis_hw=flow_analysis_hw,crop_xyxy=crop,source0=binding(current_index),source1=binding(index))
                displacement,pair_valid,pair_error=pair.forward,pair.valid_forward,pair.fb_error_forward
            delta=sample_field(displacement,points);error=sample_field(pair_error,points)[:,0]
            good=sample_field(pair_valid.float(),points)[:,0]>.99
            plus=points+normals+sample_field(displacement,points+normals)
            minus=points-normals+sample_field(displacement,points-normals)
            normals=plus-minus;normal_lengths=torch.linalg.vector_norm(normals,dim=1)
            normals=normals/normal_lengths.clamp_min(1e-8)[:,None]
            points=points+delta;alive &=good&torch.isfinite(error)&(normal_lengths>1e-8)&(points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)
            inspect(index,next_rgb,points,normals,alive,error)
            current_index,current_rgb=index,next_rgb;completed+=1
            del pair,displacement,pair_valid,pair_error
            if completed%20==0:
                print(_dumps({"stage":"same_shot_original_RGB_FB_and_profiles","completed_native_frames":completed,
                    "total_same_shot_native_frames":len(indices),"source_pts":frames[index]["pts"],"elapsed_seconds":round(time.time()-started,2)}),flush=True)
    ranked={}
    for group in groups:
        candidates=[]
        for frame in outputs.values():
            quality=next(q for q in frame["groups"] if q["group_id"]==group["group_id"])
            candidates.append({"source_pts":frame["source_pts"],"time_base":str(clock),
                "RGB_transition_FWHM_native_px":quality["RGB_transition_FWHM_native_px"],
                "RGB_gradient_native_px":quality["RGB_gradient_native_px"],
                "sharp_single_RGB_edge_candidate_count":quality["sharp_single_RGB_edge_candidate_count"],
                "FB_valid_points":quality["forward_backward_valid_search_points"],
                "original_point_count":quality["original_human_guide_point_count"]})
        ranked[group["group_id"]]=sorted(candidates,key=lambda q:(-q["sharp_single_RGB_edge_candidate_count"],
            -q["FB_valid_points"],q["RGB_transition_FWHM_native_px"]["median"] or float("inf")))
    result={"schema":"rinkspace.source-reliable-exposure-audit.v1","source_video_path":str(video_path),
        "source_video_sha256":source_sha,"time_base":str(clock),"reference_native_pts":target,
        "requested_pts_interval_half_open":[window_start,window_end],
        "actual_same_shot_pts_interval_half_open":[max(window_start,safe_start),min(window_end,safe_end)],
        "native_decoded_frame_count":len(frames),"same_shot_measured_native_frame_count":len(outputs),
        "excluded_after_cut_or_outside_shot_native_frame_count":len(frames)-len(outputs),
        "classification":{"path":str(Path(timeline_path).resolve()),"sha256":_sha(timeline_path),"interval":shot},
        "source_annotations":{"path":str(Path(source_annotations_path).resolve()),"sha256":annotations_sha_start,
            "file_sha256_at_completion":_sha(source_annotations_path),
            "source_file_changed_during_audit":annotations_sha_start!=_sha(source_annotations_path),
            "sample_id":reference_sample.get("sample_id"),"group_point_counts":{a["group_id"]:len(a["uv"]) for a in groups},
            "human_authority_unchanged":True,"transported_guides_only_find_original_ROIs_never_GT":True},
        "cut_evidence":{"adjacent_global_RGB_MAD_median_same_shot":typical,"large_original_RGB_changes":cut_candidates,
            "no_flow_or_feature_transfer_across_cut":True,"global_MAD_only_cut_evidence_not_clarity_ranking":True},
        "all_decoded_native_frames":[{k:v for k,v in f.items() if k!="pixels"} for f in frames],
        "flow":{"weights_path":str(flow.weights_path),"weights_sha256":flow.weights_sha256,
            "updates":flow_updates,"analysis_hw":list(flow_analysis_hw),"crop_native_xyxy":list(crop),
            "native_pixel_units_restored":True,"FB_accuracy_not_feature_identity":True},
        "profile":{"search_radius_native_px":18.,"sample_step_native_px":.5,
            "typed_semantic_model_supplied":False,"constant_zero_probability_placeholder":True,
            "same_source_original_RGB_measured_each_frame":True,"source_RGB_peak_is_not_a_paint_label":True},
        "frames":[outputs[i] for i in sorted(outputs)],"feature_specific_candidate_ranking":ranked,
        "whole_shot_or_full_rink_accuracy_validated":False,"camera_or_world_fit_started":False,
        "source_annotations_modified":False,"training_labels_created":0,"solver_points_created":0}
    output_path=Path(output_path).resolve()
    output_path.write_text(_dumps(result),encoding="utf8")
    return {"path":str(output_path),"sha256":_sha(output_path),"decoded_native_frames":len(frames),
        "same_shot_native_frames":len(outputs),"cut_evidence":result["cut_evidence"],
        "top_by_feature":{k:v[:3] for k,v in ranked.items()}}
