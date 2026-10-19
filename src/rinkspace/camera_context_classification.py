"""Incremental evidence for the off-ice kiss-and-cry camera context.

Static backdrop matches only retrieve a context. Chair contours and foreground
floral structure are measured separately before a same-source position proposal
is eligible for temporal aggregation. Cross-source evidence remains a candidate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .camera_classification import SOURCES, descriptor_matches, read, sha256, write


VERSION = "kiss-and-cry-static-context-v2"
REFERENCE_PTS = {
    "Rd66CneD9io": [190464512, 190925312],
    "2U7RhoKchXA": [48799232, 49260032],
}
REFERENCE_CASES = {"Rd66CneD9io": "Rd66CneD9io_012447", "2U7RhoKchXA": "2U7RhoKchXA_003171"}
PARAMETERS = {
    "image_size": [426, 240], "sift_features": 700, "ratio_test": .72,
    "ransac_px": 2.5, "context_min_inliers": 20, "background_min_inliers": 8,
    "chair_min_inliers": 4, "near_object_min_inliers": 4,
    "multiregion_min_inliers": 25, "color_retrieval_fraction": .16,
    "color_retrieval_mean_saturation": 35,
    "context_query_min_span_pixels": [200, 90],
    "v2_correction": "v1弱匹配误中赛事logo/远处布景；去掉顶部logo条，内容判断也需近景静态物件且查询画面覆盖足够宽高",
    "meaning": "adjustable retrieval/evidence parameters; never a user quota or physical camera count",
}


def region_masks():
    """Hand-observed static regions in the actual 426 x 240 reference images."""
    masks = {key: np.zeros((240, 426), np.uint8) for key in ("backdrop", "chair_contour", "side_flowers", "foreground_flowers")}
    masks["backdrop"][41:88, 13:391] = 255
    # People occupy approximately x=35..252, y=88..240 and are excluded.
    # Keep the empty right chair's rim/back and legs, excluding its cushion/toys.
    masks["chair_contour"][165:188, 286:368] = 255
    masks["chair_contour"][182:233, 261:282] = 255
    masks["chair_contour"][180:235, 365:385] = 255
    masks["side_flowers"][43:236, 397:426] = 255
    masks["side_flowers"][70:238, 0:12] = 255
    # These near flowers are a separate depth from the backdrop. Any overlaid
    # score bar can remove support, but cannot satisfy this masked reference.
    masks["foreground_flowers"][210:240, 151:234] = 255
    return masks


def features(image, masks=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = None if masks is None else np.maximum.reduce(list(masks.values()))
    keypoints, descriptors = cv2.SIFT_create(nfeatures=PARAMETERS["sift_features"], contrastThreshold=.018).detectAndCompute(gray, mask)
    points = np.asarray([keypoint.pt for keypoint in keypoints], np.float32).reshape(-1, 2)
    regions = None
    if masks is not None:
        regions = []
        for point in points:
            x, y = np.clip(np.round(point).astype(int), [0, 0], [425, 239])
            regions.append(next((name for name, value in masks.items() if value[y, x]), "outside"))
    return {"points": points, "descriptors": descriptors, "regions": regions}


def color_retrieval(image):
    hsv = cv2.cvtColor(cv2.resize(image, (128, 72)), cv2.COLOR_BGR2HSV)
    h, s, v = hsv.transpose(2, 0, 1)
    colorful_backdrop = ((h >= 90) & (h <= 175) & (s >= 45) & (v >= 35))
    fraction = float(colorful_backdrop.mean())
    mean_saturation = float(s.mean())
    return fraction >= PARAMETERS["color_retrieval_fraction"] and mean_saturation >= PARAMETERS["color_retrieval_mean_saturation"], {
        "purple_blue_fraction": round(fraction, 4), "mean_saturation": round(mean_saturation, 2),
        "meaning": "retrieval only; rejection does not classify content or camera"}


def match_regions(query, reference):
    if query["descriptors"] is None or reference["features"]["descriptors"] is None:
        return None
    ref = reference["features"]
    raw = descriptor_matches(query["descriptors"], ref["descriptors"])
    good = [a for pair in raw if len(pair) == 2 for a, b in [pair] if a.distance < PARAMETERS["ratio_test"] * b.distance]
    if len(good) < PARAMETERS["context_min_inliers"]:
        return None
    qp = np.asarray([query["points"][m.queryIdx] for m in good], np.float32)
    rp = np.asarray([ref["points"][m.trainIdx] for m in good], np.float32)
    h, keep = cv2.findHomography(qp, rp, cv2.RANSAC, PARAMETERS["ransac_px"])
    if h is None or keep is None:
        return None
    keep = keep.ravel().astype(bool)
    qp, rp = qp[keep], rp[keep]
    if len(qp) < PARAMETERS["context_min_inliers"]:
        return None
    names = [ref["regions"][m.trainIdx] for m, included in zip(good, keep) if included]
    counts = {key: names.count(key) for key in region_masks()}
    if counts["backdrop"] < PARAMETERS["background_min_inliers"]:
        return None
    # The ISU/event logo is repeated in wipes, outros and rink boards. Even a
    # low-error match is not evidence that this shot shows the seating area.
    near_region = (counts["chair_contour"] >= 4 or counts["foreground_flowers"] >= 4 or counts["side_flowers"] >= 5)
    minimum_x, minimum_y = PARAMETERS["context_query_min_span_pixels"]
    if not near_region or np.ptp(qp[:, 0]) < minimum_x or np.ptp(qp[:, 1]) < minimum_y:
        return None
    projected = cv2.perspectiveTransform(qp[None], h)[0]
    errors = np.linalg.norm(projected-rp, axis=1)
    near_count = counts["chair_contour"] + counts["foreground_flowers"]
    strong = (len(qp) >= PARAMETERS["multiregion_min_inliers"]
              and counts["chair_contour"] >= PARAMETERS["chair_min_inliers"]
              and near_count >= PARAMETERS["near_object_min_inliers"]
              and (counts["side_flowers"] >= 3 or counts["foreground_flowers"] >= 3)
              and np.ptp(rp[:, 0]) >= 250 and np.ptp(rp[:, 1]) >= 130)
    return {
        "reference_id": reference["reference_id"], "reference_source_id": reference["observation"]["source_id"],
        "inliers": len(qp), "region_inliers": counts,
        "query_points_xy": qp.round(2).tolist(), "reference_points_xy": rp.round(2).tolist(),
        "reference_point_regions": names, "homography_query_to_reference": h.tolist(),
        "median_residual_px": round(float(np.median(errors)), 4),
        "context_supported": True, "multidepth_layout_support": bool(strong),
        "interpretation": "背景+椅背/椅腿+花饰分别可追溯；背景单平面只识别布景，不确认物理机位。",
    }


def extract_refs(root, timelines):
    refs = []
    masks = region_masks()
    prior_path = root / "outputs/camera_classification/evidence/context_references.json"
    prior_manifest = read(prior_path) if prior_path.exists() else {}
    prior_observations = {
        (row["observation"]["source_id"], row["observation"]["source_pts"]): row["observation"]
        for row in prior_manifest.get("references", [])
    }
    for sid, wanted in REFERENCE_PTS.items():
        all_obs = {obs["source_pts"]: obs for segment in timelines[sid]["segments"] for obs in segment["observations"]}
        for pts in wanted:
            # A revised boundary can change the current segment's representative
            # samples. Keep the already reviewed original-PTS frame as evidence.
            original = all_obs.get(pts) or prior_observations.get((sid, pts))
            if original is None:
                raise ValueError(f"Reviewed reference missing: {sid} PTS {pts}")
            obs = dict(original)
            obs["source_time_seconds"] = obs.get("source_seconds", obs.get("source_time_seconds"))
            obs["source_path"] = timelines[sid]["track"]["source_path"]
            obs.pop("spatial_yuv_descriptor", None)
            image_path = root / obs["image_path"]
            expected_hash = obs.get("image_sha256")
            if expected_hash and sha256(image_path) != expected_hash:
                raise ValueError(f"Reviewed reference hash changed: {image_path}")
            im = cv2.imread(str(image_path))
            if im is None:
                raise FileNotFoundError(obs["image_path"])
            im = cv2.resize(im, (426, 240), interpolation=cv2.INTER_AREA)
            refs.append({"reference_id": f"context:{sid}:{pts}", "observation": obs,
                         "features": features(im, masks), "case_id": REFERENCE_CASES[sid],
                         "new_usage": "classification_reference_and_development_not_independent_check",
                         "visual_basis": "完整预留区间采样序列已被主代理查看；本代理另查看两场实际参考帧。三把椅子（两侧浅金色、中央红粉色）/右侧花饰/前景花篮/紫蓝布景。",
                         "counter_evidence": "邻近架设位置和重新安装仍不能只靠相同布景排除；参考只提供该播出来源作用域内的位置证据。"})
    return refs


def inspect_observation(root, obs, refs):
    base = {key: obs[key] for key in ("source_id", "source_pts", "image_path", "time_base", "image_sha256") if key in obs}
    base["source_time_seconds"] = obs.get("source_seconds", obs.get("source_time_seconds"))
    im = cv2.imread(str(root / obs["image_path"]))
    if im is None:
        return dict(base, status="media_missing", matches=[])
    im = cv2.resize(im, (426, 240), interpolation=cv2.INTER_AREA)
    retrieve, screen = color_retrieval(im)
    if not retrieve:
        return dict(base, status="screened_no_candidate", retrieval=screen, matches=[])
    query = features(im)
    matches = [match for ref in refs if (match := match_regions(query, ref))]
    own = [m for m in matches if m["reference_source_id"] == obs["source_id"] and m["multidepth_layout_support"]]
    strong = [m for m in matches if m["multidepth_layout_support"]]
    return dict(base, status="context_supported" if matches else "no_static_match", retrieval=screen, matches=matches,
                content_kind="off_rink_arena_kiss_and_cry" if matches else None,
                candidate_installation_id=f'candidate:{obs["source_id"]}:kiss_and_cry' if matches else None,
                position_support_eligible=bool(own), multidepth_context_support=bool(strong),
                reason="同源多区域对应可用于当前完整镜头的多时刻核查；跨来源参考仅支持相似场景及候选对应。",
                propagation="observation evidence only; combine multiple source PTS within the current segment; do not propagate through an appearance group")


def run(root, workers=4):
    root = Path(root).resolve()
    out = root / "outputs/camera_classification"
    timelines, source_inputs = {}, {}
    for sid in SOURCES:
        path = out / "timelines" / f"{sid}.json"
        payload = path.read_bytes()
        timelines[sid] = json.loads(payload)
        source_inputs[sid] = {
            "timeline_path": path.relative_to(root).as_posix(),
            "timeline_sha256": hashlib.sha256(payload).hexdigest(),
            "meaning": "hash of exactly the source timeline bytes read for this run",
        }
    refs = extract_refs(root, timelines)
    manifest = {"version": VERSION, "parameters": PARAMETERS,
                "references": [{k: v for k, v in ref.items() if k != "features"} for ref in refs],
                "region_definitions": {name: {"pixels": int(np.count_nonzero(mask)), "meaning": name} for name, mask in region_masks().items()},
                "notes": ["增加分类参考用途，不修改几何模型。", "上述两预留区间已转为开发参考，不再声称为该分类版本独立检查。"]}
    write(out / "evidence/context_references.json", manifest)
    positive, counts = [], {}
    start = time.perf_counter()
    for sid in SOURCES:
        unique = {o["image_path"]: o for s in timelines[sid]["segments"] for o in s["observations"]}
        cache_path = out / "cache" / f"{sid}_context_matches.json"
        prior = read(cache_path) if cache_path.exists() else {}
        cached = {row["image_path"]: row for row in prior.get("rows", [])} if prior.get("version") == VERSION else {}
        pending = [o for path, o in unique.items()
                   if path not in cached or any(cached[path].get(key) != o.get(key)
                                               for key in ("source_pts", "time_base", "image_sha256"))]
        print(f"{sid}: {len(unique)} observations, {len(pending)} new context retrievals", flush=True)
        cv2.setNumThreads(1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, result in enumerate(pool.map(lambda obs: inspect_observation(root, obs, refs), pending)):
                cached[result["image_path"]] = result
                if (index+1) % 1000 == 0:
                    print(f"{sid}: {index+1}/{len(pending)}", flush=True)
        rows = [cached[path] for path in unique]
        write(cache_path, {"version": VERSION, "references_path": "outputs/camera_classification/evidence/context_references.json", "rows": rows})
        selected = [row for row in rows if row["status"] == "context_supported"]
        for row in selected:
            compact = dict(row)
            compact["matches"] = [{key: value for key, value in match.items()
                                   if key not in {"query_points_xy", "reference_points_xy", "reference_point_regions", "homography_query_to_reference"}}
                                  for match in row["matches"]]
            compact["detailed_match_cache"] = cache_path.relative_to(root).as_posix()
            compact["cache_lookup_image_path"] = row["image_path"]
            positive.append(compact)
        counts[sid] = {"observations_processed": len(rows), "context_supported_observations": len(selected),
                       "same_source_multidepth_eligible_observations": sum(row.get("position_support_eligible", False) for row in selected),
                       "media_missing": sum(row["status"] == "media_missing" for row in rows)}
        print(sid, counts[sid], flush=True)
    result = {"schema_version": "rinkspace.camera-context-evidence.v1", "version": VERSION,
              "implementation_sha256": sha256(Path(__file__)), "parameters": PARAMETERS,
              "references_path": "outputs/camera_classification/evidence/context_references.json", "source_counts": counts,
              "source_inputs": source_inputs,
              "observations": positive, "elapsed_seconds": round(time.perf_counter()-start, 3),
              "assignment_contract": {"unit": "source_id and source_pts observation", "content": "off_rink_arena_kiss_and_cry",
                   "supported_position_requires": "至少两个分离时刻同源position_support_eligible，且当前镜头观察没有其他画面/未决切镜冲突；跨来源仍为候选。",
                   "installation_id_scope": "broadcast_source_only_capture_session_unresolved", "not_geometry_evidence": True},
              "limitations": ["颜色预筛未通过仅表示没有检索，不能判断不是等分区。", "布景重复不能证明拍摄场次，未排除重播。",
                              "单独一帧的多区域匹配不完成整个镜头的物理归属。", "未通过此增量模块证明所有等分区画面已发现。"]}
    write(out / "evidence/context_assignments.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.root, args.workers)
