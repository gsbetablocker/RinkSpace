"""Source-backed evidence for camera classification, without model mutations.

Legacy labels are deliberately namespaced; image similarity retrieves candidates
and never upgrades an old installation hypothesis into a physical camera fact.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def legacy_inventory(root: Path) -> dict:
    """Extract links and original status, not old geometry parameters or policies."""
    lib_path = root / "outputs/fixed_camera_v111/camera_library_frozen.json"
    lib = read_json(lib_path)
    registry_path = root / "configs/rink_model/source_registry.json"
    registry = read_json(registry_path)
    by_hash = {r.get("source_sha256"): r for r in registry["sources"] if r.get("source_sha256")}
    ref_fields = ("view_id", "reference_id", "source_id", "source_time_s", "pts", "source_pts", "time_base",
                  "source_sha256", "camera_group_id", "installation_id", "status", "image_path", "source_image_path")
    refs = []
    for i, ref in enumerate(lib["references"]):
        old = {k: ref[k] for k in ref_fields if k in ref}
        registry_entry = by_hash.get(ref.get("source_sha256"), {})
        refs.append({"evidence_id": f"legacy_library_reference_{i:02d}", "legacy": old,
                     "origin": {"path": lib_path.relative_to(root).as_posix(), "json_pointer": f"/references/{i}"},
                     "resolved_registry_source_id": registry_entry.get("source_id", ref.get("source_id")),
                     "current_interpretation": "historical_candidate_not_promoted",
                     "independent_classification_check": False})
    installations = []
    for i, item in enumerate(lib["installations"]):
        fields = ("installation_id", "camera_group_id", "session_id", "session_epoch_status", "view_ids",
                  "common_hardware_identity_with_main_camera", "center_independently_surveyed")
        installations.append({"legacy": {k: item[k] for k in fields if k in item},
                              "origin": {"path": lib_path.relative_to(root).as_posix(), "json_pointer": f"/installations/{i}"},
                              "current_interpretation": "historical_candidate_not_promoted"})
    observations = []
    for path in sorted((root / "outputs/event_survey_v11/all_source_review_v2").glob("*_review.json")):
        review = read_json(path)
        for i, row in enumerate(review["selected_views"]):
            fields = ("source_id", "session_id", "source_path", "source_sha256", "source_pts", "time_base",
                      "source_time_seconds", "image_path", "shot_id", "camera_group_id", "camera_group_status",
                      "candidate_library_group_ids", "orientation_sector_observed", "review_reason",
                      "installation_version_status", "panel_index", "visual_review_status", "reviewer_kind")
            observations.append({"evidence_id": f"legacy_review_{row['source_id']}_{row['source_pts']}",
                                 "legacy": {k: row[k] for k in fields if k in row},
                                 "origin": {"path": path.relative_to(root).as_posix(), "json_pointer": f"/selected_views/{i}"},
                                 "current_interpretation": "point_observation_not_whole_shot_assignment",
                                 "independent_classification_check": False})
    cut_path = root / "outputs/event_reuse_v11/men_15557_shot_review/shot_boundary_review.json"
    return {"schema_version": "rinkspace.camera-evidence.legacy.v1", "references": refs,
            "installations": installations, "observations": observations,
            "historical_cut_review": {"path": cut_path.relative_to(root).as_posix(), "sha256": sha256(cut_path),
                                      "record": read_json(cut_path)},
            "source_files": [{"path": p.relative_to(root).as_posix(), "sha256": sha256(p)}
                             for p in [lib_path, registry_path]],
            "geometry_modified": False,
            "women_fs_use": "classification_allowed; actual used images are recorded separately; geometry evaluation role remains unchanged"}


def thumbnail(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, (32, 18), interpolation=cv2.INTER_AREA).reshape(-1)


def media_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": path.resolve().as_posix(), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cached_samples(cache: Path, identity: dict):
    provenance = cache.with_suffix(".provenance.json")
    if cache.exists() and provenance.exists() and read_json(provenance).get("input_identity") == identity:
        with np.load(cache, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    return None


def sample_track(path: Path, cache: Path, *, step_s: float | None = None,
                 keyframes_only: bool = False, start_s: float = 0, end_s: float | None = None):
    """Cache observed original PTS; no synthetic frame/time interpolation."""
    identity = {"media": media_identity(path)}
    cached = cached_samples(cache, identity)
    if cached is not None:
        return cached
    pts, rgb = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        if keyframes_only:
            stream.codec_context.skip_frame = "NONKEY"
        if start_s:
            container.seek(int(start_s / stream.time_base), stream=stream)
        next_t = start_s
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start_s or (step_s is not None and t + 1e-8 < next_t):
                continue
            if end_s is not None and t >= end_s:
                break
            pts.append(frame.pts)
            rgb.append(thumbnail(frame.to_ndarray(format="rgb24")))
            if step_s is not None:
                next_t += step_s
        data = {"pts": np.asarray(pts, dtype=np.int64), "rgb": np.asarray(rgb, dtype=np.uint8),
                "time_base": np.asarray(str(stream.time_base))}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    write_json(cache.with_suffix(".provenance.json"), {"input_identity": identity,
               "sampling": {"step_s": step_s, "keyframes_only": keyframes_only, "start_s": start_s, "end_s": end_s},
               "hash_meaning": "Cache contains RGB thumbnails and decoded original PTS; input identity checks file path/size/mtime. Original source SHA256 is retained separately in duplicate mapping."})
    return data


def nearest_retrieval(query: dict, target: dict, count: int = 5):
    """Coarse image retrieval; returned hits alone are not duplicate evidence."""
    q = query["rgb"].astype(np.float32)
    x = target["rgb"].astype(np.float32)
    dist = ((q*q).sum(1)[:, None] + (x*x).sum(1)[None, :] - 2*q@x.T) / q.shape[1]
    inds = np.argsort(dist, axis=1)[:, :count]
    return inds, np.take_along_axis(dist, inds, axis=1)


def coarse_offset(query: dict, target: dict) -> dict:
    inds, distances = nearest_retrieval(query, target)
    qt = query["pts"] * float(Fraction(str(query["time_base"])))
    tt = target["pts"] * float(Fraction(str(target["time_base"])))
    bins = Counter(round(float(tt[j] - qt[i])/5)*5 for i, row in enumerate(inds) for j in row)
    return {"offset_s": bins.most_common(1)[0][0], "votes": bins.most_common(10),
            "retrieval_only": True, "distances": distances}


def align_near_offset(query: dict, target: dict, offset: float, radius: float = 10.0) -> list[dict]:
    q = query["rgb"].astype(np.float32)
    x = target["rgb"].astype(np.float32)
    qtb, ttb = Fraction(str(query["time_base"])), Fraction(str(target["time_base"]))
    qt = query["pts"] * float(qtb)
    tt = target["pts"] * float(ttb)
    anchors = []
    for i, t in enumerate(qt):
        ids = np.flatnonzero(abs(tt - (t + offset)) <= radius)
        if not len(ids):
            continue
        distances = np.mean((x[ids] - q[i])**2, axis=1)
        j = int(ids[np.argmin(distances)])
        anchors.append({"query_pts": int(query["pts"][i]), "query_time_base": str(qtb),
                        "query_time_s": float(t), "target_pts": int(target["pts"][j]),
                        "target_time_base": str(ttb), "target_time_s": float(tt[j]),
                        "offset_s": float(tt[j] - t), "thumbnail_rgb_mse": float(distances.min()),
                        "status": "pixel_duplicate_anchor" if distances.min() <= 50 else "unresolved_pixel_match"})
    return anchors


def mapping_segments(anchors: list[dict], max_gap_s: float = 1.5) -> list[dict]:
    """Preserve only locally supported mapping ranges; do not bridge failed hits."""
    segments, current = [], []
    for index, anchor in enumerate(anchors):
        good = anchor["status"] == "pixel_duplicate_anchor"
        if current and (not good or anchor["query_time_s"] - anchors[current[-1]]["query_time_s"] > max_gap_s
                        or abs(anchor["offset_s"] - anchors[current[-1]]["offset_s"]) > .25):
            segments.append(current)
            current = []
        if good:
            current.append(index)
    if current:
        segments.append(current)
    result = []
    for ids in segments:
        if len(ids) < 2:
            continue
        points = [anchors[i] for i in ids]
        q = np.array([r["query_time_s"] for r in points])
        t = np.array([r["target_time_s"] for r in points])
        a, b = np.polyfit(q-q[0], t-t[0], 1)
        b = float(t[0]+b-a*q[0])
        result.append({"query_interval_s": [float(q[0]), float(q[-1])],
                       "target_interval_s": [float(t[0]), float(t[-1])], "anchor_indices": ids,
                       "affine_time_ratio": float(a), "affine_offset_s": b,
                       "max_anchor_residual_s": float(abs(t-(a*q+b)).max()),
                       "mapping_type": "sampled_duplicate_mapping_between_observed_anchors",
                       "frame_exact_between_anchors": False})
    return result


def replay_segments(rows: list[dict]) -> list[dict]:
    """Keep temporally consistent exact-image chains; action similarity is excluded."""
    runs, run = [], []
    for i, row in enumerate(rows):
        good = row["masked_thumbnail_mse"] < 20
        if run:
            previous = rows[run[-1]]
            dt = row["replay_time_s"]-previous["replay_time_s"]
            du = row["original_time_s"]-previous["original_time_s"]
            if not good or dt > .25 or du < 0 or du > .3:
                runs.append(run)
                run = []
        if good:
            run.append(i)
    if run:
        runs.append(run)
    result = []
    for indices in runs:
        if len(indices) < 3:
            continue
        x = np.asarray([rows[i]["replay_time_s"] for i in indices])
        y = np.asarray([rows[i]["original_time_s"] for i in indices])
        a, b = np.polyfit(x-x[0], y-y[0], 1)
        b = y[0]+b-a*x[0]
        result.append({"replay_interval_s": [float(x[0]), float(x[-1])],
                       "original_interval_s": [float(y[0]), float(y[-1])],
                       "original_seconds_per_replay_second": float(a), "offset_s": float(b),
                       "max_fit_residual_s": float(abs(y-(a*x+b)).max()), "anchor_indices": indices,
                       "status": "same_image_content_sampled_temporal_chain",
                       "independent_camera_evidence": False,
                       "exact_boundary_not_claimed": True})
    return result


def frame_at_pts(path: Path, target_pts: int) -> tuple[np.ndarray, dict]:
    """Return an observed frame, explicitly reporting any requested-PTS miss."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        container.seek(target_pts, stream=stream)
        for frame in container.decode(stream):
            if frame.pts >= target_pts:
                return frame.to_ndarray(format="rgb24"), {"requested_pts": target_pts,
                    "source_pts": frame.pts, "time_base": str(stream.time_base),
                    "source_time_s": float(frame.pts*stream.time_base),
                    "exact_requested_pts_found": frame.pts == target_pts}
    raise ValueError(f"No frame at or after {target_pts}: {path}")


def sample_mapped_proxy(workspace: Path, cache: Path) -> dict:
    """Use existing playback pixels with their verified per-frame source PTS map."""
    identity = {"media": media_identity(workspace / "proxy.mp4"),
                "mapping": media_identity(workspace / "proxy_mapping.json")}
    cached = cached_samples(cache, identity)
    if cached is not None:
        return cached
    mapping = read_json(workspace / "proxy_mapping.json")["frames"]
    pts, pixels, next_time = [], [], 0.0
    with av.open(str(workspace / "proxy.mp4")) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        for index, frame in enumerate(container.decode(stream)):
            entry = mapping[index]
            if frame.pts != entry["proxy_pts"]:
                raise ValueError(f"Proxy map disagreement at frame {index}")
            if entry["source_time_s"] + 1e-8 < next_time:
                continue
            pts.append(entry["source_pts"])
            pixels.append(thumbnail(frame.to_ndarray(format="rgb24")))
            next_time += 1
    data = {"pts": np.asarray(pts), "rgb": np.asarray(pixels, dtype=np.uint8),
            "time_base": np.asarray(mapping[0]["source_time_base"])}
    np.savez_compressed(cache, **data)
    write_json(cache.with_suffix(".provenance.json"), {"input_identity": identity,
               "sampling": {"step_s": 1}, "mapping_validation": "Every decoded proxy PTS checked against source mapping before selection"})
    return data


PROGRAMME_PAIRS = [("cetin", "EJV9lyZ7D28", "Rd66CneD9io"),
                   ("jin", "PVXWd7nLt8Q", "Rd66CneD9io"),
                   ("ko", "gnskF9dQwJo", "Rd66CneD9io"),
                   ("okada", "HZQx6HK8zR8", "Rd66CneD9io"),
                   ("ebihara", "GY6bDwCGjKg", "2U7RhoKchXA")]


def build_cetin_replay_mapping(root: Path, output: Path):
    """Find identical-image replay chains, masking changed broadcast graphics."""
    data = sample_track(root / "data/input/EJV9lyZ7D28.398.mp4", output / "cetin_dense_descriptors.npz")
    times = data["pts"] * float(Fraction(str(data["time_base"])))
    rgb = data["rgb"].reshape(-1, 18, 32, 3)
    mask = np.ones((18, 32), bool)
    mask[:7, :13] = False  # changing score graphic
    mask[14:, 24:] = False  # clock / replay watermark
    replay_indices = np.flatnonzero((times >= 284) & (times <= 371))[::10]
    original_indices = np.flatnonzero(times < 250)
    q = rgb[replay_indices][:, mask].reshape(len(replay_indices), -1).astype(np.float32)
    x = rgb[original_indices][:, mask].reshape(len(original_indices), -1).astype(np.float32)
    distance = ((q*q).sum(1)[:, None] + (x*x).sum(1)[None, :] - 2*q@x.T) / q.shape[1]
    rows = []
    for i, j in enumerate(distance.argmin(axis=1)):
        rows.append({"replay_pts": int(data["pts"][replay_indices[i]]), "replay_time_s": float(times[replay_indices[i]]),
                     "original_pts": int(data["pts"][original_indices[j]]), "original_time_s": float(times[original_indices[j]]),
                     "masked_thumbnail_mse": float(distance[i, j]),
                     "status": "pixel_match_candidate" if distance[i, j] < 20 else "unresolved_not_assumed_same_camera"})
    document = {"schema_version": "rinkspace.replay-evidence.v1", "source_id": "EJV9lyZ7D28", "time_base": str(data["time_base"]),
                "rows": rows, "segments": replay_segments(rows),
                "mask_32x18_excluded_rectangles": [[0, 0, 13, 7], [24, 14, 32, 18]],
                "status": "same_image_replay_chains_verified_at_sampled_pts_not_independent_observations",
                "sampling_interval_s": .2, "geometry_model_use": False, "all_replay_frames_verified": False,
                "search_scope": {"replay_query_s": [284, 371], "original_match_s": [0, 250]},
                "scope_basis": "Existing contact sheet visually showed replay montage after the performance; these ranges are evidence-specific retrieval parameters, not a global task restriction.",
                "limitations": ["匹配覆盖人物和静态背景，不把同动作的其它视角当重复拍摄画面。", "起止为匹配采样点，不伪造剪辑边界；速度比是采样拟合，逐点PTS更精确。", "未匹配淡变、切镜与画面仍未决；搜索范围以外未声称无回放。"]}
    write_json(output / "cetin_replay_mapping.json", document)
    return document


def build_programme_mappings(root: Path, output: Path) -> dict:
    """Find local programme uploads by pixels, preserving unresolved samples."""
    index = []
    for key, programme_id, broadcast_id in PROGRAMME_PAIRS:
        workspace = root / "outputs/workspace" / key
        metadata = read_json(workspace / "metadata.json")
        source = Path(metadata["source_path"])
        source = source if source.is_absolute() else root / source
        target = root / f"outputs/event_survey_v11/{broadcast_id}/{broadcast_id}.f133.mp4"
        query_path = output / f"{key}_second_descriptors.npz"
        query = sample_track(source, query_path, step_s=1) if key == "cetin" else sample_mapped_proxy(workspace, query_path)
        target_name = "women_fs" if broadcast_id == "Rd66CneD9io" else "men_fs"
        coarse = sample_track(target, output / f"{target_name}_keyframe_descriptors.npz", keyframes_only=True)
        retrieval = coarse_offset(query, coarse)
        duration = float(query["pts"][-1] * Fraction(str(query["time_base"])))
        offset = retrieval["offset_s"]
        window = sample_track(target, output / f"{key}_broadcast_window.npz", step_s=None if key == "cetin" else .1,
                              start_s=max(0, offset-15), end_s=offset+duration+15)
        anchors = align_near_offset(query, window, offset)
        accepted = [a for a in anchors if a["status"] == "pixel_duplicate_anchor"]
        failed = [a for a in anchors if a["status"] != "pixel_duplicate_anchor"]
        summary = read_json(root / f"outputs/event_survey_v11/{broadcast_id}/summary.json")
        target_hash = summary.get("source_sha256")
        if not target_hash:
            manifest = next((root / f"outputs/event_survey_v11/{broadcast_id}/chunks").glob("*/manifest.json"))
            target_hash = read_json(manifest).get("source_sha256")
        group = f"{key}_upload_{target_name}_same_broadcast_images"
        document = {"schema_version": "rinkspace.duplicate-mapping.v1", "programme_source_id": programme_id,
                    "programme_source_path": source.relative_to(root).as_posix(),
                    "programme_source_sha256": metadata["source_sha256"],
                    "programme_time_base": str(query["time_base"]),
                    "query_representation": "original_track" if key == "cetin" else "mapped_playback_proxy",
                    "query_proxy_mapping": None if key == "cetin" else (workspace / "proxy_mapping.json").relative_to(root).as_posix(),
                    "broadcast_source_id": broadcast_id, "broadcast_source_path": target.relative_to(root).as_posix(),
                    "broadcast_source_sha256": target_hash, "broadcast_time_base": str(window["time_base"]),
                    "duplicate_group_id": group, "classification_reference_use": True, "geometry_model_use": False,
                    "anchors": anchors, "segments": mapping_segments(anchors), "sampling_seconds": 1,
                    "coarse_retrieval": {"offset_bins_seconds": retrieval["votes"], "query_title_not_used": True},
                    "precision_note": "Actual source PTS anchors, via verified proxy mapping where identified. Affine ranges are sampled evidence, not frame-exact mappings or exact edit boundaries.",
                    "unresolved_anchor_indices": [i for i, a in enumerate(anchors) if a["status"] != "pixel_duplicate_anchor"],
                    "validation": {"observed_anchor_count": len(anchors), "pixel_match_count": len(accepted),
                                   "unresolved_anchor_count": len(failed), "all_frames_verified": False,
                                   "independent_classification_validation": False}}
        write_json(output / f"{key}_duplicate_mapping.json", document)
        index.append({"path": (output / f"{key}_duplicate_mapping.json").relative_to(root).as_posix(),
                      "programme_source_id": programme_id, "broadcast_source_id": broadcast_id,
                      "group_id": group,
                      "programme_interval_s": [accepted[0]["query_time_s"], accepted[-1]["query_time_s"]] if accepted else None,
                      "broadcast_interval_s": [accepted[0]["target_time_s"], accepted[-1]["target_time_s"]] if accepted else None,
                      "status": "sampled_pixel_verified_with_explicit_gaps", "independent_evidence": False,
                      "anchors_verified": len(accepted), "anchors_unresolved": len(failed)})
    replay = build_cetin_replay_mapping(root, output)
    cetin = read_json(output / "cetin_duplicate_mapping.json")
    valid = [a for a in cetin["anchors"] if a["status"] == "pixel_duplicate_anchor"]
    qx = np.asarray([a["query_time_s"] for a in valid])
    tx = np.asarray([a["target_time_s"] for a in valid])
    replay_links = []
    for i, segment in enumerate(replay["segments"]):
        replay_links.append({"replay_chain_index": i,
            "programme_replay_interval_s": segment["replay_interval_s"],
            "programme_original_interval_s": segment["original_interval_s"],
            "estimated_broadcast_replay_interval_s": np.interp(segment["replay_interval_s"], qx, tx).tolist(),
            "estimated_broadcast_original_interval_s": np.interp(segment["original_interval_s"], qx, tx).tolist(),
            "broadcast_source_id": "Rd66CneD9io", "independent_evidence": False,
            "time_mapping_precision": "Interpolated between recorded pixel-matched integer-second anchors; these are sampled chain endpoints, not exact edit boundaries. Actual programme PTS and original anchors remain authoritative."})
    result = {"schema_version": "rinkspace.duplicate-index.v1", "mappings": index,
              "internal_replay_mappings": ["outputs/camera_classification/evidence/cetin_replay_mapping.json"],
              "broadcast_replay_links": replay_links,
              "remaining_scope": "Five existing local uploads matched; uploads without local pixels and replay material outside tested CETIN ranges remain unresolved. Different URLs or broadcast sessions alone never imply independent camera evidence."}
    write_json(output / "duplicate_mappings.json", result)
    return result


HIGH_CORNER_SEEDS = {"Rd66CneD9io": [3, 11], "2U7RhoKchXA": [9], "DXiFMHULDZ8": [14],
                     "oxXu1avhrvs": [15], "uaod0sM-9Hw": [5], "mEoZsjFK4r0": [4],
                     "M_2a8gDUyrQ": [11], "yS-iLdqoHtg": [6]}


def high_corner_correspondence(root: Path) -> dict:
    observations = []
    for sid, indices in HIGH_CORNER_SEEDS.items():
        review = read_json(root / f"outputs/event_survey_v11/all_source_review_v2/{sid}_review.json")
        observations.extend(review["selected_views"][i-1] for i in indices)
    reference = next(r for r in observations if r["source_id"] == "oxXu1avhrvs")
    detector = cv2.SIFT_create(nfeatures=4000, contrastThreshold=.015)
    mask = np.zeros((480, 852), np.uint8)
    mask[:200] = 255
    mask[200:, :70] = 255
    mask[200:, 810:] = 255
    def extract(row):
        im = cv2.resize(cv2.imread(str(root / row["image_path"])), (852, 480))
        return detector.detectAndCompute(im, mask)
    k1, d1 = extract(reference)
    pairs = []
    for row in observations:
        if row is reference:
            continue
        k2, d2 = extract(row)
        good = [a for a, b in cv2.BFMatcher().knnMatch(d1, d2, k=2) if a.distance < .7*b.distance]
        p = np.asarray([k1[m.queryIdx].pt for m in good])
        q = np.asarray([k2[m.trainIdx].pt for m in good])
        H, inliers = cv2.findHomography(p, q, cv2.RANSAC, 3)
        error = np.linalg.norm(cv2.perspectiveTransform(p[:, None], H)[:, 0]-q, axis=1)/2
        keep = inliers.ravel().astype(bool)
        zones = {"far_wall_and_stands": p[:, 1] < 100,
                 "boards_and_rink_context": (p[:, 1] >= 100) & (p[:, 1] < 200),
                 "near_side_structure": p[:, 1] >= 200}
        # The near-side set is not used to fit this second diagnostic warp.
        far = p[:, 1] < 200
        far_H, _ = cv2.findHomography(p[far], q[far], cv2.RANSAC, 3)
        near = (~far) & keep
        holdout = np.linalg.norm(cv2.perspectiveTransform(p[near, None], far_H)[:, 0]-q[near], axis=1)/2 if near.any() else np.array([])
        pairs.append({"reference_source_id": reference["source_id"], "reference_pts": reference["source_pts"],
                      "target_source_id": row["source_id"], "target_pts": row["source_pts"],
                      "target_time_s": row["source_time_seconds"], "time_base": row["time_base"],
                      "image_path": row["image_path"], "tentative_matches": len(good), "inlier_matches": int(keep.sum()),
                      "inliers_by_region": {name: int((keep & z).sum()) for name, z in zones.items()},
                      "reprojection_p95_at_426_width_px": float(np.percentile(error[keep], 95)),
                      "near_structure_not_used_for_fit": {"count": len(holdout), "p95_px_at_426_width": float(np.percentile(holdout, 95)) if len(holdout) else None},
                      "inlier_reference_xy_426": (p[keep]/2).round(3).tolist(),
                      "inlier_target_xy_426": (q[keep]/2).round(3).tolist(),
                      "classification_use": "reference_selection_not_independent_validation"})
    return {"schema_version": "rinkspace.camera-evidence.multidepth.v1", "reference": {
                k: reference[k] for k in ("source_id", "source_pts", "time_base", "source_time_seconds", "image_path")},
            "pairs": pairs, "observation_status": "visually_reviewed_multisession_position_family_candidate",
            "visual_support": ["远端窗格与看台红梯、白椅序列一致", "近侧蓝围挡、座椅与远端结构的遮挡关系一致", "各场可见不同观众/运动员及冰痕，不把相同场馆视角当重复视频"],
            "against_unqualified_merge": ["240p不能排除小平移或场次间重新安装", "自由单应仅检验图像关系，不能单独证明光心相同", "只用远端结构拟合后，男自近侧p95约2.75px、冰舞自约4.42px；明显高于全图单应残差，不应无条件合并精确安装中心", "mEoZsjFK4r0近侧结构仅1匹配，跨深度支持弱于其他场次", "第二次近侧检查未用于拟合，但已用于第一次全图匹配筛选，并非独立几何留出", "每场只选取历史代表观测，尚非完整镜头独立检查"],
            "cross_session_installation_status": "same_position_family_supported_at_image_resolution_exact_center_and_hardware_unresolved",
            "geometry_modified": False}
