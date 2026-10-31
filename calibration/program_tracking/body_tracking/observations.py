"""Select the Ankara programme skater and run the frozen original ViTPose-H.

The two commands are deliberately separate: ``select`` records target identity
from original video, cached boxes, and actual SAM masks; ``pose`` only infers
COCO17 points for selected real detections. No box or pose is interpolated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import av
import cv2
import numpy as np
from rinkspace.storage import AssetStore, resolve_asset


HERE = Path(__file__).resolve().parent
PROGRAMME = HERE.parent
REPO = HERE.parents[2]
DEFAULT_SHOTS = PROGRAMME / "source_shots.json"
DEFAULT_PEOPLE = Path("D:/CodexData/RinkSpace_program_tracking_tmp/people/boxes.json")
DEFAULT_MASKS = DEFAULT_PEOPLE.parent / "masks"
DEFAULT_UPSTREAM = resolve_asset(REPO, REPO / "environments/body/GVHMR")
DEFAULT_H = DEFAULT_UPSTREAM / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
DEFAULT_H_CODE = HERE / "source_helpers/run_vitpose_h.py"
DEFAULT_YOLO = resolve_asset(REPO, REPO / "models/cache/yolo11n-pose.pt")
YOLO_SITE = resolve_asset(REPO, REPO / ".venv-gpu/Lib/site-packages")
FIRST_PTS = 229114112
END_PTS = 231705600
STEP = 256
TIME_BASE = "1/15360"
NATIVE_WH = (1920, 1080)
COCO_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
              "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
              "left_wrist", "right_wrist", "left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(shots_path: Path, people_path: Path) -> tuple[dict, dict, list[int]]:
    manifest = json.loads(resolve_asset(REPO, shots_path).read_text(encoding="utf-8-sig"))
    people = json.loads(resolve_asset(REPO, people_path).read_text(encoding="utf-8-sig"))
    storage = AssetStore(REPO)
    manifest['source_path_original'] = manifest.get('source_path_original', manifest['source_path'])
    manifest['source_path'] = str(storage.source(manifest['source_path']))
    if manifest["source_resolution"] != list(NATIVE_WH) or manifest["source_time_base"] != TIME_BASE:
        raise ValueError("Unexpected original source geometry/time base")
    if people["sample_size_wh"] != [960, 540] or people["time_base"] != TIME_BASE:
        raise ValueError("Unexpected cached detector geometry/time base")
    if storage.source(people["source"]) != storage.source(manifest["source_path"]):
        raise ValueError("People cache points to a different source")
    if manifest["window_pts"] != [FIRST_PTS - 17 * STEP, END_PTS]:
        raise ValueError("Original programme window changed")
    if manifest["shots"][0]["end_pts"] != FIRST_PTS:
        raise ValueError("First 17-frame shot boundary changed")
    expected = list(range(FIRST_PTS, END_PTS, STEP))
    if len(expected) != 10123:
        raise AssertionError("Frozen output denominator changed")
    by_pts = {int(row["pts"]): row["boxes"] for row in people["frames"]}
    if len(by_pts) != len(people["frames"]) or set(expected) - by_pts.keys():
        raise ValueError("Cached people boxes do not cover every valid source PTS")
    for i, shot in enumerate(manifest["shots"][1:], 1):
        if shot["start_pts"] != (FIRST_PTS if i == 1 else manifest["shots"][i - 1]["end_pts"]):
            raise ValueError("Noncontiguous source shots")
    if manifest["shots"][-1]["end_pts"] != END_PTS:
        raise ValueError("Last source shot ends at wrong PTS")
    return manifest, by_pts, expected


def box_to_native(box: list[float], sample_wh=(960, 540), native_wh=NATIVE_WH) -> np.ndarray:
    """Convert original cached xyxy coordinates explicitly, not just centres."""
    scale = np.array([native_wh[0] / sample_wh[0], native_wh[1] / sample_wh[1]] * 2,
                     dtype=np.float32)
    native = np.asarray(box[:4], dtype=np.float32) * scale
    if not np.isfinite(native).all() or native[2] <= native[0] or native[3] <= native[1]:
        raise ValueError(f"Invalid cached person box: {box}")
    return native


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = np.maximum(a[:2], b[:2])
    x1, y1 = np.minimum(a[2:], b[2:])
    inter = max(0., x1 - x0) * max(0., y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1.))


def iter_exact_frames(source: Path, pts: list[int]):
    """Sequential exact-PTS PyAV decode with one seek; no PNG staging."""
    if not pts or pts != sorted(set(pts)):
        raise ValueError("PTS must be nonempty, sorted and unique")
    wanted = iter(pts)
    target = next(wanted)
    with av.open(str(AssetStore(REPO).source(source))) as container:
        video = container.streams.video[0]
        if str(video.time_base) != TIME_BASE:
            raise ValueError(f"Original video time base changed: {video.time_base}")
        container.seek(max(0, target - 2 * 15360), stream=video, backward=True)
        for frame in container.decode(video):
            if frame.pts is None or frame.pts < target:
                continue
            if frame.pts != target:
                raise ValueError(f"Exact source PTS {target} absent; next decoded {frame.pts}")
            bgr = frame.to_ndarray(format="bgr24")
            if bgr.shape[:2] != (NATIVE_WH[1], NATIVE_WH[0]):
                raise ValueError(f"Native video resolution changed: {bgr.shape}")
            yield target, bgr
            try:
                target = next(wanted)
            except StopIteration:
                return
    raise ValueError(f"Original video ended before PTS {target}")


class PackedMasks:
    def __init__(self, directory: Path, first_cached_pts: int):
        self.directory = resolve_asset(REPO, directory)
        self.first_cached_pts = first_cached_pts
        self.chunk = -1
        self.pts_to_index: dict[int, int] = {}
        self.bits = None
        self.status = None

    def get(self, pts: int) -> tuple[np.ndarray, str]:
        chunk = (pts - self.first_cached_pts) // (120 * STEP)
        if chunk != self.chunk:
            path = self.directory / f"{chunk:05d}.npz"
            with np.load(path) as data:
                if data["shape"].tolist() != [540, 960]:
                    raise ValueError(f"Mask geometry changed: {path}")
                self.bits = data["masks"].copy()
                self.status = data["status"].copy()
                self.pts_to_index = {int(p): i for i, p in enumerate(data["pts"])}
            self.chunk = chunk
        index = self.pts_to_index.get(pts)
        if index is None:
            raise KeyError(f"No real cached mask at {pts}")
        return np.unpackbits(self.bits[index], axis=1, count=960).astype(bool), str(self.status[index])


def appearance_feature(small_bgr: np.ndarray, mask: np.ndarray, cached_box: list[float]) -> np.ndarray:
    """Coarse clothing histogram from actual source pixels inside the person mask."""
    x0, y0, x1, y1 = map(float, cached_box[:4])
    w, h = x1 - x0, y1 - y0
    left = max(0, int(x0 + .25 * w))
    right = min(960, int(np.ceil(x0 + .75 * w)))
    top = max(0, int(y0 + .23 * h))
    bottom = min(540, int(np.ceil(y0 + .69 * h)))
    if right <= left or bottom <= top:
        return np.zeros(16, dtype=np.float32)
    patch = small_bgr[top:bottom, left:right]
    present = mask[top:bottom, left:right]
    if present.sum() < 20:
        return np.zeros(16, dtype=np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = (hsv[:, :, 0] // 23).clip(0, 7)
    value = (hsv[:, :, 2] // 32).clip(0, 7)
    # The first eight bins preserve dark/bright costume structure; the second
    # eight preserve flower-colour information without retaining full images.
    hist = np.r_[np.bincount(value[present], minlength=8),
                 np.bincount(hue[present & (hsv[:, :, 1] > 55)], minlength=8)]
    hist = hist.astype(np.float32)
    return hist / max(float(hist.sum()), 1.)


def mask_fill(mask: np.ndarray, box: list[float]) -> float:
    x0, y0, x1, y1 = box[:4]
    xa, ya = max(0, int(x0)), max(0, int(y0))
    xb, yb = min(960, int(np.ceil(x1))), min(540, int(np.ceil(y1)))
    return float(mask[ya:yb, xa:xb].mean()) if xb > xa and yb > ya else 0.


def appearance_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all()) or a.sum() < .5 or b.sum() < .5:
        return 0.
    return float(1. - .5 * np.abs(a - b).sum())


def candidate_emission(candidate: dict, anchor_feature: np.ndarray) -> float:
    b = candidate["bbox_cache_xyxy"]
    area = max(1., (b[2] - b[0]) * (b[3] - b[1]))
    bottom = b[3] / 540.
    size = np.log1p(area / 5000.)
    appearance = appearance_similarity(np.asarray(candidate["appearance"]), anchor_feature)
    return float(1.1 * candidate["detector_score"] + .75 * size +
                 1.2 * bottom + .8 * appearance + .5 * candidate["mask_fill"] -
                 (2. if bottom < .38 else 0.))


def transition_score(a: dict | None, b: dict | None) -> float:
    if a is None and b is None:
        return -.65
    if a is None or b is None:
        return -1.35
    x = np.asarray(a["bbox_cache_xyxy"], np.float32)
    y = np.asarray(b["bbox_cache_xyxy"], np.float32)
    ac, bc = (x[:2] + x[2:]) / 2, (y[:2] + y[2:]) / 2
    ah, bh = max(1., x[3] - x[1]), max(1., y[3] - y[1])
    distance = float(np.linalg.norm(ac - bc) / max(35., .5 * (ah + bh)))
    scale = abs(float(np.log(bh / ah)))
    look = appearance_similarity(np.asarray(a["appearance"]), np.asarray(b["appearance"]))
    return float(1.4 * box_iou(x, y) + .65 * look - 2.1 * distance - .8 * scale -
                 (4. if distance > 1.5 else 0.))


def select_shot(rows: list[dict]) -> list[dict]:
    """Within-shot Viterbi association, with an explicit unknown state."""
    if not rows:
        return rows
    anchor = rows[0]["candidates"]
    if not anchor:
        raise ValueError("Shot-start source-reviewed target anchor has no detection")
    anchor_feature = np.asarray(anchor[0]["appearance"], np.float32)
    states = [row["candidates"] + [None] for row in rows]
    dp = np.full(len(states[0]), -np.inf)
    dp[0] = 0.  # nine source-reviewed shot starts have the skater in cache box 0
    back = []
    margins = []
    for i in range(1, len(states)):
        previous, current = states[i - 1], states[i]
        next_dp = np.full(len(current), -np.inf)
        trace = np.zeros(len(current), np.int16)
        for j, item in enumerate(current):
            scores = np.asarray([dp[k] + transition_score(prior, item)
                                 for k, prior in enumerate(previous)])
            best = int(np.argmax(scores))
            emission = -.7 if item is None else candidate_emission(item, anchor_feature)
            next_dp[j] = scores[best] + emission
            trace[j] = best
        back.append(trace)
        dp = next_dp
        ordered = np.sort(dp)
        margins.append(float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else float("inf"))
    path = [int(np.argmax(dp))]
    for trace in reversed(back):
        path.append(int(trace[path[-1]]))
    path.reverse()
    for i, (row, choice) in enumerate(zip(rows, path)):
        candidate = states[i][choice]
        row["selected_candidate_index"] = None if candidate is None else candidate["index"]
        row["target_status"] = "selected" if candidate is not None else "unknown"
        row["bbox_native_xyxy"] = None if candidate is None else candidate["bbox_native_xyxy"]
        row["observation_origin"] = "cached_yolo11n_pose_box" if candidate is not None else "none"
        row["association_margin"] = None if i == 0 or not np.isfinite(margins[i - 1]) else margins[i - 1]
        if (candidate is not None and i > 0 and row["association_margin"] is not None
                and row["association_margin"] < .16 and len(row["candidates"]) > 1):
            row["target_status"] = "unknown"
            row["selected_candidate_index"] = None
            row["bbox_native_xyxy"] = None
            row["observation_origin"] = "none"
            row["ambiguity_reason"] = "near_tie_in_shot_association"
        elif candidate is None:
            row["ambiguity_reason"] = "no_unique_same_shot_detection"
        else:
            row["ambiguity_reason"] = None
        row["track_id"] = f"shot_{row['shot_index']:02d}_target" if row["bbox_native_xyxy"] is not None else None
    return rows


def recover_missing(source: Path, rows: list[dict], yolo_path: Path) -> None:
    """Run a real local YOLO search only for cached frames with no person box."""
    missing = [i for i, row in enumerate(rows) if not row["candidates"]]
    if not missing:
        return
    # This is the already-used local detector checkpoint from the box cache.
    # Old Ultralytics checkpoints serialize model objects, while torch>=2.6
    # now defaults to weights_only=True. Restrict the compatibility override
    # to exactly this path and restore torch.load before inference.
    config = resolve_asset(REPO, REPO / ".cache/body_trajectory_v15_20260930/yolo_config")
    config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config)
    import torch
    import torchvision  # keep CUDA-compatible runtime version before Ultralytics site precedence
    if str(YOLO_SITE) not in sys.path:
        sys.path.insert(0, str(YOLO_SITE))
    from ultralytics import YOLO
    original_load = torch.load
    resolved = yolo_path.resolve()

    def load_local_yolo(file, *args, **kwargs):
        if Path(file).resolve() == resolved and "weights_only" not in kwargs:
            return original_load(file, *args, **kwargs, weights_only=False)
        return original_load(file, *args, **kwargs)

    with patch("torch.load", load_local_yolo):
        model = YOLO(str(yolo_path)).to("cuda")
    for i in missing:
        row = rows[i]
        near = [j for j in range(max(0, i - 6), min(len(rows), i + 7))
                if j != i and rows[j]["shot_index"] == row["shot_index"]
                and rows[j]["bbox_native_xyxy"] is not None]
        if not near:
            row["ambiguity_reason"] = "cached_no_detection_and_no_same_shot_anchor"
            continue
        nearest = min(near, key=lambda j: abs(j - i))
        anchor = np.asarray(rows[nearest]["bbox_native_xyxy"], np.float32)
        _, bgr = next(iter_exact_frames(source, [row["source_pts"]]))
        cx, cy = (anchor[:2] + anchor[2:]) / 2
        wh = (anchor[2:] - anchor[:2]) * 2.2
        crop_box = np.r_[np.maximum([0, 0], [cx, cy] - wh / 2),
                         np.minimum(NATIVE_WH, [cx, cy] + wh / 2)].astype(int)
        xa, ya, xb, yb = crop_box
        found = model.predict(bgr[ya:yb, xa:xb], device=0, imgsz=960,
                              conf=.08, classes=[0], verbose=False)[0].boxes
        choices = []
        for box, confidence in zip(found.xyxy.cpu().numpy(), found.conf.cpu().numpy()):
            box = box + np.array([xa, ya, xa, ya])
            centre = (box[:2] + box[2:]) / 2
            separation = np.linalg.norm(centre - (anchor[:2] + anchor[2:]) / 2)
            scale = max(30., np.linalg.norm(anchor[2:] - anchor[:2]))
            if separation / scale < .7:
                choices.append((float(separation / scale), float(confidence), box))
        choices.sort(key=lambda x: x[0])
        row["local_search"] = {"source_pts": row["source_pts"], "anchor_pts": rows[nearest]["source_pts"],
                               "crop_native_xyxy": crop_box.tolist(), "candidate_count": len(choices),
                               "detector_weights": str(yolo_path)}
        if choices and (len(choices) == 1 or choices[1][0] - choices[0][0] > .18):
            distance, confidence, box = choices[0]
            row["bbox_native_xyxy"] = box.tolist()
            row["target_status"] = "recovered_detection"
            row["observation_origin"] = "source_local_yolo11n_pose_detection"
            row["track_id"] = f"shot_{row['shot_index']:02d}_target"
            row["ambiguity_reason"] = None
            row["local_search"].update(selected_score=confidence, anchor_distance_ratio=distance,
                                       selected_bbox_native_xyxy=box.tolist())
        else:
            row["target_status"] = "unknown"
            row["ambiguity_reason"] = "local_detection_absent_or_conflicted"


def select(args: argparse.Namespace) -> None:
    manifest, by_pts, all_pts = load_inputs(args.shots, args.people)
    source = Path(manifest["source_path"])
    masks = PackedMasks(args.masks, manifest["window_pts"][0])
    shot_pts = []
    for shot in manifest["shots"][1:]:
        segment = list(range(shot["start_pts"], shot["end_pts"], STEP))
        if args.max_frames_per_shot:
            segment = segment[:args.max_frames_per_shot]
        shot_pts.extend(segment)
    if not args.max_frames_per_shot and shot_pts != all_pts:
        raise AssertionError("Selection did not cover 10123 PTS")
    rows = []
    start = time.perf_counter()
    current_shot = -1
    for pts, bgr in iter_exact_frames(source, shot_pts):
        shot_index = next(i for i, shot in enumerate(manifest["shots"])
                          if shot["start_pts"] <= pts < shot["end_pts"])
        if shot_index != current_shot:
            current_shot = shot_index
            print(f"select shot {shot_index} PTS {pts}", flush=True)
        mask, mask_status = masks.get(pts)
        small = cv2.resize(bgr, (960, 540), interpolation=cv2.INTER_AREA)
        candidates = []
        for index, box in enumerate(by_pts[pts]):
            native = box_to_native(box)
            candidates.append({"index": index, "bbox_cache_xyxy": list(map(float, box[:4])),
                               "bbox_native_xyxy": native.tolist(), "detector_score": float(box[4]),
                               "mask_fill": mask_fill(mask, box),
                               "appearance": appearance_feature(small, mask, box).tolist()})
        rows.append({"source_pts": pts, "time_base": TIME_BASE, "shot_index": shot_index,
                     "camera_id": manifest["shots"][shot_index]["camera_id"],
                     "subject_id": "programme_skater", "mask_status": mask_status,
                     "candidates": candidates})
    selected = []
    for shot_index in range(1, len(manifest["shots"])):
        group = [row for row in rows if row["shot_index"] == shot_index]
        selected.extend(select_shot(group))
    if args.recover_missing:
        recover_missing(source, selected, args.yolo)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "target_selection.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    report = {"schema": "rinkspace.body.selection.v1", "source": str(source),
              "source_sha256": manifest["source_sha256_from_timeline"],
              "source_pts_range": [FIRST_PTS, END_PTS], "time_base": TIME_BASE,
              "excluded_first_shot_frame_count": 17, "frame_count": len(selected),
              "status_counts": dict(Counter(row["target_status"] for row in selected)),
              "boxes_sha256": sha256(args.people), "shots_sha256": sha256(args.shots),
              "mask_directory": str(args.masks), "selection_seconds": time.perf_counter() - start,
              "mask_semantics": "SAM cache is a union of detected people; overlap supports a candidate but does not establish skater identity",
              "method": "nine visually reviewed shot-start skater anchors; per-shot box/mask/appearance Viterbi; near-tie unknown; exact-PTS source local YOLO recovery for cache misses",
              "accuracy": "automatic identity proposal; source-image review needed"}
    (args.output / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def load_h_code(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_original_h", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.NAMES) != COCO_NAMES:
        raise ValueError("Original H COCO keypoint order changed")
    return module


def pose(args: argparse.Namespace) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The original ViTPose-H path requires CUDA")
    rows = [json.loads(line) for line in args.selection.read_text(encoding="utf-8").splitlines() if line]
    pts = [int(row["source_pts"]) for row in rows]
    if pts != sorted(set(pts)) or any(p < FIRST_PTS or p >= END_PTS or (p - FIRST_PTS) % STEP for p in pts):
        raise ValueError("Selection contains invalid or duplicate source PTS")
    if not args.allow_partial and pts != list(range(FIRST_PTS, END_PTS, STEP)):
        raise ValueError("Full observations require exactly all 10123 valid PTS")
    shots = json.loads(args.shots.read_text(encoding="utf-8"))
    selection_report_path = args.selection.with_name("selection_report.json")
    if selection_report_path.exists():
        selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
        if selection_report["shots_sha256"] != sha256(args.shots):
            raise ValueError("Source shot manifest changed since identity selection")
        if selection_report["frame_count"] != len(rows):
            raise ValueError("Selection report frame count differs from JSONL")
    for row in rows:
        shot = shots["shots"][int(row["shot_index"])]
        if not shot["start_pts"] <= row["source_pts"] < shot["end_pts"]:
            raise ValueError(f"Selected source PTS {row['source_pts']} has wrong shot identity")
    source = Path(shots["source_path"])
    original = load_h_code(args.h_code)
    start = time.perf_counter()
    model, decode = original.load_h(args.upstream, args.checkpoint)
    torch.cuda.reset_peak_memory_stats()
    n = len(rows)
    coco17 = np.full((n, 17, 3), np.nan, np.float32)
    bbox = np.full((n, 4), np.nan, np.float32)
    affine = np.full((n, 2, 3), np.nan, np.float32)
    centers = np.full((n, 2), np.nan, np.float32)
    scales = np.full((n, 2), np.nan, np.float32)
    statuses = []
    origins = []
    track_ids = []
    pending = []
    forward_seconds = 0.

    def flush() -> None:
        nonlocal forward_seconds
        if not pending:
            return
        indices = [entry[0] for entry in pending]
        batch = torch.stack([entry[1] for entry in pending]).cuda()
        flipped = torch.cat([batch, batch.flip(3)], dim=0)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.inference_mode():
            heat = model(flipped)
            count = len(pending)
            merged = (heat[:count] + heat[count:, original.FLIP].flip(3)) * .5
        torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - tick
        xy, scores = decode(merged.cpu().numpy(), centers[indices], scales[indices], use_udp=True)
        coco17[indices] = np.concatenate([xy, scores], axis=-1).astype(np.float32)
        pending.clear()
        del batch, flipped, heat, merged

    for i, (pts_i, bgr) in enumerate(iter_exact_frames(source, pts)):
        row = rows[i]
        chosen = row.get("bbox_native_xyxy")
        statuses.append(row["target_status"])
        origins.append("original_vitpose_h" if chosen is not None else "none")
        track_ids.append(row.get("track_id") or "")
        if chosen is None:
            continue
        native = np.asarray(chosen, np.float32)
        if not np.isfinite(native).all() or native[2] <= native[0] or native[3] <= native[1]:
            raise ValueError(f"Invalid selected native bbox at {pts_i}")
        bbox[i] = native
        xywh = [native[0], native[1], native[2] - native[0], native[3] - native[1]]
        tensor, center, scale, mapping = original.udp_crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), xywh)
        centers[i], scales[i], affine[i] = center, scale, mapping
        pending.append((i, tensor))
        if len(pending) >= args.batch_size:
            flush()
        if i % 600 == 0:
            print(f"pose {i + 1}/{n} PTS {pts_i}", flush=True)
    flush()
    valid = np.isfinite(coco17).all(axis=(1, 2))
    keypoint_in_source = (np.isfinite(coco17[:, :, :2]).all(axis=2)
                          & (coco17[:, :, 0] >= 0) & (coco17[:, :, 0] < NATIVE_WH[0])
                          & (coco17[:, :, 1] >= 0) & (coco17[:, :, 1] < NATIVE_WH[1]))
    keypoint_valid = keypoint_in_source & np.isfinite(coco17[:, :, 2]) & (coco17[:, :, 2] > 0)
    source_clipped_target = (np.isfinite(bbox).all(axis=1)
                             & ((bbox[:, 0] <= 24) | (bbox[:, 1] <= 24)
                                | (bbox[:, 2] >= NATIVE_WH[0] - 24)
                                | (bbox[:, 3] >= NATIVE_WH[1] - 24)))
    hip = np.full((n, 2), np.nan, np.float32)
    both_hips = keypoint_valid[:, 11] & keypoint_valid[:, 12]
    hip[both_hips] = .5 * (coco17[both_hips, 11, :2] + coco17[both_hips, 12, :2])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "observations.npz", pts=np.asarray(pts, np.int64),
                        coco17=coco17, native_coco17_xy=coco17[:, :, :2],
                        keypoint_scores=coco17[:, :, 2], keypoint_valid=keypoint_valid,
                        keypoint_in_source=keypoint_in_source,
                        source_clipped_target=source_clipped_target,
                        visibility_status=np.asarray(["unreviewed" if v else "no_pose"
                                                      for v in valid], dtype="U16"),
                        bbox_native_xyxy=bbox, crop_affine=affine,
                        crop_center=centers, crop_scale_200=scales,
                        hip_midpoint_proxy_xy=hip, valid=valid,
                        shot_index=np.asarray([row["shot_index"] for row in rows], np.int16),
                        target_status=np.asarray(statuses, dtype="U32"),
                        track_id=np.asarray(track_ids, dtype="U32"),
                        observation_origin=np.asarray(origins, dtype="U32"),
                        source_time_base=np.asarray(TIME_BASE),
                        names=np.asarray(COCO_NAMES))
    report = {"schema": "rinkspace.body.observations.v1", "frame_count": n,
              "valid_h_frame_count": int(valid.sum()), "target_status_counts": dict(Counter(statuses)),
              "source": str(source), "source_sha256": shots["source_sha256_from_timeline"],
              "selection_sha256": sha256(args.selection), "h_code": str(args.h_code),
              "h_code_sha256": sha256(args.h_code), "checkpoint": str(args.checkpoint),
              "checkpoint_sha256": original.H_SHA256, "upstream": str(args.upstream),
              "python": sys.executable, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "batch_size": args.batch_size,
              "flip_test": True, "precision": "float32", "input_hw": [256, 192],
              "pose_forward_seconds": forward_seconds, "total_seconds": time.perf_counter() - start,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
              "coordinate_system": "original_native_pixels_1920x1080",
              "keypoint_score_semantics": "original H heatmap scores, not calibrated visibility probabilities",
              "keypoint_valid_semantics": "positive raw score, finite coordinates, and within original image; not independently reviewed visibility",
              "source_clipped_target_semantics": "selected detector box within 24 source pixels of an image edge; possible truncation requiring review",
              "hip_proxy_semantics": "diagnostic 2D midpoint, not native SMPL-X pelvis",
              "accuracy": "automatic keypoints; original-source visual audit still required"}
    (args.output / "observations_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    choose = sub.add_parser("select", help="Exact-PTS source identity association and evidence JSONL")
    choose.add_argument("--shots", type=Path, default=DEFAULT_SHOTS)
    choose.add_argument("--people", type=Path, default=DEFAULT_PEOPLE)
    choose.add_argument("--masks", type=Path, default=DEFAULT_MASKS)
    choose.add_argument("--yolo", type=Path, default=DEFAULT_YOLO)
    choose.add_argument("--output", type=Path, required=True)
    choose.add_argument("--max-frames-per-shot", type=int, default=0,
                        help="Development probe only; default 0 covers all 10123 frames")
    choose.add_argument("--recover-missing", action="store_true", default=True)
    infer = sub.add_parser("pose", help="Original frozen ViTPose-H on real selected boxes")
    infer.add_argument("--selection", type=Path, required=True)
    infer.add_argument("--shots", type=Path, default=DEFAULT_SHOTS)
    infer.add_argument("--h-code", type=Path, default=DEFAULT_H_CODE)
    infer.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    infer.add_argument("--checkpoint", type=Path, default=DEFAULT_H)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--batch-size", type=int, default=2)
    infer.add_argument("--allow-partial", action="store_true", help="Development probe only")
    args = parser.parse_args()
    if args.command == "select":
        select(args)
    else:
        pose(args)


if __name__ == "__main__":
    main()
