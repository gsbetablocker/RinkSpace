"""Extract masked static-image correspondences and reusable SIFT keyframes.

The pairs are source-image observations, not yet positions on the ice plane.
Every packet/frame is identified by its native PTS and pairs never cross a cut.
Decoded data and features live in --cache-dir, outside the project workspace.
"""

from __future__ import annotations
from calibration.solver_runtime.config import data_module_dir as _runtime_data_module_dir

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np


ROOT=Path('D:/OneDrive/codex_projs/RinkSpace')
DEFAULT_SHOTS = _runtime_data_module_dir(__file__) / 'source_shots.json'
DEFAULT_PEOPLE = Path("D:/CodexData/RinkSpace_program_tracking_tmp/people/boxes.json")
SCALE = 2.0  # 960x540 observations to native 1920x1080 pixels.
WORK_SIZE = (960, 540)
PTS_STEP = 256


def analysis_mask(pts: int, boxes: dict[int, list], margin: int = 17) -> np.ndarray:
    width, height = WORK_SIZE
    mask = np.full((height, width), 255, np.uint8)
    # Persistent broadcast overlays. Their text changes independently of the rink.
    mask[:135, :250] = 0
    mask[455:, 805:] = 0
    for box in boxes.get(pts, []):
        x1, y1, x2, y2 = box[:4]
        left, top = max(0, int(x1) - margin), max(0, int(y1) - margin)
        right, bottom = min(width, int(x2) + margin), min(height, int(y2) + margin)
        mask[top:bottom, left:right] = 0
    return mask


def region_hints(frame_bgr: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Conservative colour-line hints: 0 unknown, 1 ice, 2 boards.

    No plane label is assigned unless both wide blue and yellow dasher bands are
    visible. This is for prioritization only, never a geometry observation.
    """
    hints = np.zeros(len(xy), dtype=np.uint8)
    if len(xy) == 0:
        return hints
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (90, 65, 60), (135, 255, 255))
    yellow = cv2.inRange(hsv, (12, 48, 95), (43, 255, 255))
    blue_counts = np.count_nonzero(blue[:, 60:900], axis=1)
    yellow_counts = np.count_nonzero(yellow[:, 60:900], axis=1)
    blue_counts[:70] = 0
    yellow_counts[:140] = 0
    yellow_counts[510:] = 0
    blue_y = int(np.argmax(blue_counts))
    yellow_y = int(np.argmax(yellow_counts))
    if blue_counts[blue_y] < 155 or yellow_counts[yellow_y] < 150:
        return hints
    if not 35 < yellow_y - blue_y < 340:
        return hints
    y = xy[:, 1]
    hints[(y > blue_y + 12) & (y < yellow_y - 12)] = 2
    hints[y > yellow_y + 12] = 1
    return hints


def distributed_corners(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    candidates = cv2.goodFeaturesToTrack(
        gray, maxCorners=1100, qualityLevel=0.007, minDistance=7,
        mask=mask, blockSize=5,
    )
    if candidates is None:
        return np.empty((0, 1, 2), np.float32)
    candidates = candidates.reshape(-1, 2)
    bins = np.zeros((4, 6), dtype=np.int16)
    selected = []
    for x, y in candidates:
        col = min(5, int(x * 6 / WORK_SIZE[0]))
        row = min(3, int(y * 4 / WORK_SIZE[1]))
        if bins[row, col] >= 24:
            continue
        bins[row, col] += 1
        selected.append((x, y))
    return np.asarray(selected, dtype=np.float32).reshape(-1, 1, 2)


def pair_observation(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_bgr: np.ndarray,
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
) -> dict:
    p0 = distributed_corners(previous_gray, previous_mask)
    empty_xy = np.empty((0, 2), np.float32)
    if len(p0) < 4:
        return {"xy0": empty_xy, "xy1": empty_xy, "fb": np.empty(0, np.float32),
                "inlier": np.empty(0, np.uint8), "hint": np.empty(0, np.uint8),
                "corners": len(p0), "hull_fraction": 0.0}
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, forward, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, p0, None, **options)
    if p1 is None:
        return {"xy0": empty_xy, "xy1": empty_xy, "fb": np.empty(0, np.float32),
                "inlier": np.empty(0, np.uint8), "hint": np.empty(0, np.uint8),
                "corners": len(p0), "hull_fraction": 0.0}
    back, backward, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, p1, None, **options)
    if back is None:
        return {"xy0": empty_xy, "xy1": empty_xy, "fb": np.empty(0, np.float32),
                "inlier": np.empty(0, np.uint8), "hint": np.empty(0, np.uint8),
                "corners": len(p0), "hull_fraction": 0.0}
    q0, q1, qb = p0.reshape(-1, 2), p1.reshape(-1, 2), back.reshape(-1, 2)
    error = np.linalg.norm(qb - q0, axis=1)
    inside = (q1[:, 0] >= 0) & (q1[:, 0] < WORK_SIZE[0]) & (q1[:, 1] >= 0) & (q1[:, 1] < WORK_SIZE[1])
    q1safe = np.clip(np.rint(q1).astype(np.int32), (0, 0), (WORK_SIZE[0] - 1, WORK_SIZE[1] - 1))
    unmasked = current_mask[q1safe[:, 1], q1safe[:, 0]] > 0
    good = (forward.reshape(-1) > 0) & (backward.reshape(-1) > 0) & (error <= 1.0) & inside & unmasked
    xy0, xy1, fb = q0[good], q1[good], error[good]
    if len(xy0) >= 8:
        _, accepted = cv2.findHomography(xy0, xy1, cv2.RANSAC, 2.5, maxIters=1000, confidence=0.995)
        inlier = accepted.reshape(-1).astype(np.uint8) if accepted is not None else np.zeros(len(xy0), np.uint8)
    else:
        inlier = np.zeros(len(xy0), np.uint8)
    if int(inlier.sum()) >= 3:
        hull = cv2.convexHull(xy0[inlier > 0].astype(np.float32))
        hull_fraction = float(cv2.contourArea(hull) / (WORK_SIZE[0] * WORK_SIZE[1]))
    else:
        hull_fraction = 0.0
    return {
        "xy0": xy0 * SCALE, "xy1": xy1 * SCALE, "fb": fb * SCALE,
        "inlier": inlier, "hint": region_hints(previous_bgr, xy0),
        "corners": len(p0), "hull_fraction": hull_fraction,
    }


class ShotWriter:
    def __init__(self, shot: dict, index: int, cache_dir: Path):
        self.shot = shot
        self.index = index
        self.cache_dir = cache_dir
        self.pair_pts = []
        self.offsets = [0]
        self.xy0 = []
        self.xy1 = []
        self.fb = []
        self.inlier = []
        self.hint = []
        self.corners = []
        self.hull = []
        self.keyframe_pts = []
        self.keyframe_offsets = [0]
        self.keyframe_xy = []
        self.keyframe_desc = []

    def add_pair(self, prev_pts: int, curr_pts: int, data: dict) -> None:
        self.pair_pts.append((prev_pts, curr_pts))
        for field in ("xy0", "xy1", "fb", "inlier", "hint"):
            getattr(self, field).append(data[field])
        self.offsets.append(self.offsets[-1] + len(data["xy0"]))
        self.corners.append(data["corners"])
        self.hull.append(data["hull_fraction"])

    def add_keyframe(self, pts: int, keypoints, descriptors) -> None:
        self.keyframe_pts.append(pts)
        xy = np.array([kp.pt for kp in keypoints], np.float32).reshape(-1, 2) * SCALE
        desc = descriptors.astype(np.float16) if descriptors is not None else np.empty((0, 128), np.float16)
        self.keyframe_xy.append(xy)
        self.keyframe_desc.append(desc)
        self.keyframe_offsets.append(self.keyframe_offsets[-1] + len(xy))

    def write(self) -> dict:
        path = self.cache_dir / f"motion_shot_{self.index:02d}.npz"
        keypath = self.cache_dir / f"sift_shot_{self.index:02d}.npz"
        concat = lambda xs, shape, dtype: np.concatenate(xs).astype(dtype) if xs else np.empty(shape, dtype)
        np.savez_compressed(
            path,
            pair_pts=np.asarray(self.pair_pts, np.int64).reshape(-1, 2),
            offsets=np.asarray(self.offsets, np.int64),
            xy0=concat(self.xy0, (0, 2), np.float32),
            xy1=concat(self.xy1, (0, 2), np.float32),
            fb_error_px=concat(self.fb, (0,), np.float32),
            homography_inlier=concat(self.inlier, (0,), np.uint8),
            region_hint=concat(self.hint, (0,), np.uint8),
            detected_corner_count=np.asarray(self.corners, np.int32),
            inlier_hull_fraction=np.asarray(self.hull, np.float32),
        )
        np.savez_compressed(
            keypath,
            keyframe_pts=np.asarray(self.keyframe_pts, np.int64),
            offsets=np.asarray(self.keyframe_offsets, np.int64),
            xy_native=concat(self.keyframe_xy, (0, 2), np.float32),
            descriptors=concat(self.keyframe_desc, (0, 128), np.float16),
        )
        count = int(self.offsets[-1])
        accepted = sum(int(np.sum(a)) for a in self.inlier)
        return {
            "shot_index": self.index,
            "shot_id": self.shot["id"],
            "camera_id": self.shot["camera_id"],
            "start_pts": self.shot["start_pts"], "end_pts": self.shot["end_pts"],
            "motion_npz": str(path), "sift_npz": str(keypath),
            "pair_count": len(self.pair_pts), "correspondence_count": count,
            "homography_inlier_count": accepted,
            "sift_keyframe_count": len(self.keyframe_pts),
            "pairs_under_20_inliers": sum(int(np.sum(a) < 20) for a in self.inlier),
        }


def extract(source: Path, shot_manifest: Path, people_path: Path, cache_dir: Path, output: Path) -> dict:
    metadata = json.loads(shot_manifest.read_text(encoding="utf-8"))
    shots = metadata["shots"]
    people = json.loads(people_path.read_text(encoding="utf-8"))
    if people["frame_count"] != metadata["native_frame_count"]:
        raise ValueError("Person detections do not cover this native source window")
    boxes = {int(item["pts"]): item["boxes"] for item in people["frames"]}
    if len(boxes) != metadata["native_frame_count"]:
        raise ValueError("Person detections have missing or duplicate PTS")
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected = np.load(metadata["native_packet_pts_cache"])["pts"]
    if len(expected) != metadata["native_frame_count"]:
        raise ValueError("Native PTS index has inconsistent length")
    sift = cv2.SIFT_create(nfeatures=700, contrastThreshold=0.025)
    writer = ShotWriter(shots[0], 0, cache_dir)
    shot_index = 0
    completed = []
    previous = None
    for ordinal, frame in enumerate(decode_source(source, int(expected[0]), int(expected[-1] + PTS_STEP))):
        pts = int(frame.pts)
        if pts != int(expected[ordinal]):
            raise ValueError(f"Decoded native PTS mismatch at {ordinal}: {pts}")
        while pts >= shots[shot_index]["end_pts"]:
            completed.append(writer.write())
            print(f"shot {shot_index}: {completed[-1]['pair_count']} pairs / {completed[-1]['correspondence_count']} links", flush=True)
            shot_index += 1
            writer = ShotWriter(shots[shot_index], shot_index, cache_dir)
            previous = None
        if not shots[shot_index]["start_pts"] <= pts < shots[shot_index]["end_pts"]:
            raise ValueError(f"Decoded PTS {pts} outside shot {shot_index}")
        bgr = cv2.resize(frame.to_ndarray(format="bgr24"), WORK_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask = analysis_mask(pts, boxes)
        if previous is not None:
            last_pts, last_gray, last_bgr, last_mask = previous
            data = pair_observation(last_gray, gray, last_bgr, last_mask, mask)
            writer.add_pair(last_pts, pts, data)
        if previous is None or ordinal % 30 == 0:
            keypoints, desc = sift.detectAndCompute(gray, mask)
            writer.add_keyframe(pts, keypoints, desc)
        previous = (pts, gray, bgr, mask)
    completed.append(writer.write())
    print(f"shot {shot_index}: {completed[-1]['pair_count']} pairs / {completed[-1]['correspondence_count']} links", flush=True)
    if sum(item["pair_count"] for item in completed) != metadata["native_frame_count"] - len(shots):
        raise ValueError("A native adjacent pair is missing or crosses a cut")
    result = {
        "schema": "rinkspace.program.motion.v1",
        "source_shots": str(shot_manifest), "people_boxes": str(people_path),
        "native_frame_count": metadata["native_frame_count"],
        "pair_count": sum(item["pair_count"] for item in completed),
        "correspondence_count": sum(item["correspondence_count"] for item in completed),
        "sift_keyframes": sum(item["sift_keyframe_count"] for item in completed),
        "coordinate_system": "native_source_pixels_1920x1080",
        "point_semantics": "masked static-image correspondence; no ice-plane or 3D claim",
        "region_hint_codes": {"0": "unknown", "1": "ice_candidate", "2": "board_candidate"},
        "homography_inlier_semantics": "whole-image camera-motion-consistency filter; not proof of shared ice plane",
        "shots": completed,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def decode_source(source: Path, first_pts: int, end_pts: int):
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, first_pts - 15360), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < first_pts:
                continue
            if frame.pts >= end_pts:
                break
            yield frame


def load_sift_keyframe(sift_npz: str | Path, index: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Return (native PTS, native-pixel keypoints, SIFT descriptors)."""
    with np.load(sift_npz) as cache:
        start, end = int(cache["offsets"][index]), int(cache["offsets"][index + 1])
        return (int(cache["keyframe_pts"][index]), cache["xy_native"][start:end].copy(),
                cache["descriptors"][start:end].astype(np.float32))


def match_sift_descriptors(
    source_descriptors: np.ndarray,
    target_descriptors: np.ndarray,
    ratio: float = 0.76,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mutual-nearest SIFT matches; torch CUDA is used when available.

    Returns source indices, target indices, and Euclidean descriptor distances.
    This establishes visual identity only; geometric verification remains needed.
    """
    source = np.asarray(source_descriptors, dtype=np.float32)
    target = np.asarray(target_descriptors, dtype=np.float32)
    empty_i = np.empty(0, np.int32)
    empty_d = np.empty(0, np.float32)
    if len(source) == 0 or len(target) < 2:
        return empty_i, empty_i, empty_d
    try:
        import torch
    except ImportError:
        torch = None
    use_cuda = torch is not None and device in ("auto", "cuda") and torch.cuda.is_available()
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA descriptor matching was requested but is unavailable")
    if use_cuda:
        with torch.inference_mode():
            a = torch.from_numpy(source).to("cuda")
            b = torch.from_numpy(target).to("cuda")
            distance = torch.cdist(a, b)
            nearest, indices = torch.topk(distance, k=2, dim=1, largest=False)
            reverse = torch.argmin(distance, dim=0)
            valid = (nearest[:, 0] < ratio * nearest[:, 1]) & (
                reverse[indices[:, 0]] == torch.arange(len(a), device="cuda")
            )
            source_ids = torch.nonzero(valid).flatten()
            return (source_ids.cpu().numpy().astype(np.int32),
                    indices[source_ids, 0].cpu().numpy().astype(np.int32),
                    nearest[source_ids, 0].cpu().numpy().astype(np.float32))
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(source, target, k=2)
    reverse = matcher.match(target, source)
    best_reverse = {match.queryIdx: match.trainIdx for match in reverse}
    matches = [pair[0] for pair in forward if len(pair) == 2
               and pair[0].distance < ratio * pair[1].distance
               and best_reverse.get(pair[0].trainIdx) == pair[0].queryIdx]
    return (np.asarray([match.queryIdx for match in matches], np.int32),
            np.asarray([match.trainIdx for match in matches], np.int32),
            np.asarray([match.distance for match in matches], np.float32))


def backfill_cut_start_keyframes(source_shots: Path, people_path: Path, index_path: Path) -> int:
    """Add cut-start SIFT anchors to caches made before this feature was added."""
    source_data = json.loads(source_shots.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    people = json.loads(people_path.read_text(encoding="utf-8"))
    boxes = {int(item["pts"]): item["boxes"] for item in people["frames"]}
    sift = cv2.SIFT_create(nfeatures=700, contrastThreshold=0.025)
    added = 0
    for shot, entry in zip(source_data["shots"], index["shots"]):
        path = Path(entry["sift_npz"])
        with np.load(path) as old:
            pts = old["keyframe_pts"].copy()
            offsets = old["offsets"].copy()
            xy = old["xy_native"].copy()
            desc = old["descriptors"].copy()
        start = int(shot["start_pts"])
        if start in pts:
            if len(offsets) != len(pts) + 1:
                raise ValueError(f"Malformed SIFT offsets in {path}; repair before continuing")
            continue
        frames = list(decode_source(Path(source_data["source_path"]), start, start + PTS_STEP))
        if len(frames) != 1 or frames[0].pts != start:
            raise ValueError(f"Could not recover cut-start frame {start}")
        bgr = cv2.resize(frames[0].to_ndarray(format="bgr24"), WORK_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        points, descriptors = sift.detectAndCompute(gray, analysis_mask(start, boxes))
        new_xy = np.array([point.pt for point in points], np.float32).reshape(-1, 2) * SCALE
        new_desc = descriptors.astype(np.float16) if descriptors is not None else np.empty((0, 128), np.float16)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file,
                                keyframe_pts=np.r_[start, pts],
                                offsets=np.r_[0, len(new_xy), offsets[1:] + len(new_xy)],
                                xy_native=np.concatenate((new_xy, xy)),
                                descriptors=np.concatenate((new_desc, desc)))
        temporary.replace(path)
        entry["sift_keyframe_count"] += 1
        added += 1
    index["sift_keyframes"] = sum(item["sift_keyframe_count"] for item in index["shots"])
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shots", type=Path, default=DEFAULT_SHOTS)
    parser.add_argument("--people", type=Path, default=DEFAULT_PEOPLE)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=_runtime_data_module_dir(__file__) / 'motion_index.json')
    parser.add_argument("--backfill-cut-starts", action="store_true")
    args = parser.parse_args()
    if args.backfill_cut_starts:
        count = backfill_cut_start_keyframes(args.source_shots, args.people, args.output)
        print(f"added {count} cut-start SIFT keyframes")
        return
    shot_data = json.loads(args.source_shots.read_text(encoding="utf-8"))
    result = extract(Path(shot_data["source_path"]), args.source_shots, args.people, args.cache_dir, args.output)
    print(f"done: {result['pair_count']} pairs, {result['correspondence_count']} links, {result['sift_keyframes']} SIFT frames", flush=True)


if __name__ == "__main__":
    main()
