"""Decode an exact PTS interval and track static rink/board image evidence.

The output keeps native pixel coordinates and the original PTS. Tracks are only
image observations: a surface label is a candidate, not a proven 3-D match.
No skater, spectator, broadcast graphic or background feature is intentionally
seeded. Lost points remain NaN rather than being interpolated or resurrected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from fractions import Fraction

import av
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "data/input/oxXu1avhrvs.mp4"
DEFAULT_RUN = Path(__file__).resolve().parents[1] / "runs/low_end_pm1"
ANCHOR_PTS = 104693760
FIRST_PTS = 104678400
LAST_PTS = 104709120
PTS_STEP = 256
SMALL_WIDTH = 960
SMALL_HEIGHT = 540

# Native-frame polygons. These delimit static *candidate* material visible in
# this particular anchor. The broad central box protects against the skater.
# They are metadata, so a different shot can provide different regions.
DEFAULT_REGIONS = {
    "board": [
        [[25, 410], [1885, 402], [1885, 476], [285, 480], [75, 554], [25, 565]],
    ],
    "ice": [
        [[0, 690], [90, 540], [280, 487], [1880, 479], [1920, 540], [1920, 900],
         [1630, 900], [1600, 1070], [0, 1070]],
    ],
    "exclude": [
        [[925, 295], [1255, 295], [1255, 815], [925, 815]],  # skater + shadow
        [[0, 0], [565, 0], [565, 260], [0, 260]],  # television score
        [[1620, 918], [1915, 918], [1915, 1060], [1620, 1060]],  # clock
    ],
}


def decode_native(source: Path, output: Path, expected_pts: np.ndarray) -> np.ndarray:
    """Decode exactly the requested original PTS and write a same-rate clip."""
    grays: list[np.ndarray] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source)) as container, av.open(str(output), mode="w") as writer:
        stream = container.streams.video[0]
        if str(stream.time_base) != "1/15360":
            raise ValueError(f"Unexpected time base {stream.time_base}")
        out = writer.add_stream("libx264", rate=60, options={"crf": "18", "preset": "veryfast"})
        out.width = 1920
        out.height = 1080
        out.pix_fmt = "yuv420p"
        container.seek(int(expected_pts[0] - 2 * 15360), stream=stream, backward=True)
        index = 0
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < expected_pts[0]:
                continue
            if frame.pts > expected_pts[-1]:
                break
            if index >= len(expected_pts) or frame.pts != int(expected_pts[index]):
                raise ValueError(f"Source PTS discontinuity at output index {index}: {frame.pts}")
            bgr = frame.to_ndarray(format="bgr24")
            if bgr.shape != (1080, 1920, 3):
                raise ValueError(f"Unexpected frame dimensions {bgr.shape}")
            small = cv2.resize(bgr, (SMALL_WIDTH, SMALL_HEIGHT), interpolation=cv2.INTER_AREA)
            grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            item = av.VideoFrame.from_ndarray(bgr, format="bgr24")
            item.pts = index
            item.time_base = Fraction(1, 60)
            for packet in out.encode(item):
                writer.mux(packet)
            index += 1
        for packet in out.encode():
            writer.mux(packet)
    if index != len(expected_pts):
        raise ValueError(f"Expected {len(expected_pts)} source frames, decoded {index}")
    return np.stack(grays)


def _mask(polygons: list[list[list[int]]], exclusions: list[list[list[int]]]) -> np.ndarray:
    mask = np.zeros((SMALL_HEIGHT, SMALL_WIDTH), np.uint8)
    for poly in polygons:
        xy = np.asarray(poly, np.float32) * .5
        cv2.fillPoly(mask, [np.rint(xy).astype(np.int32)], 255)
    for poly in exclusions:
        xy = np.asarray(poly, np.float32) * .5
        cv2.fillPoly(mask, [np.rint(xy).astype(np.int32)], 0)
    return mask


def _seed_surface(gray: np.ndarray, mask: np.ndarray, per_tile: int = 12) -> np.ndarray:
    """Distributed Shi-Tomasi points keep one advertisement from dominating."""
    points: list[np.ndarray] = []
    tile_h, tile_w = 72, 96
    for y in range(0, SMALL_HEIGHT, tile_h):
        for x in range(0, SMALL_WIDTH, tile_w):
            patch = gray[y:y + tile_h, x:x + tile_w]
            roi = mask[y:y + tile_h, x:x + tile_w]
            if cv2.countNonZero(roi) < 80:
                continue
            corners = cv2.goodFeaturesToTrack(
                patch, maxCorners=per_tile, qualityLevel=.015,
                minDistance=7, mask=roi, blockSize=5,
            )
            if corners is None:
                continue
            p = corners[:, 0, :].astype(np.float64)
            p[:, 0] += x
            p[:, 1] += y
            points.extend(p)
    if not points:
        return np.empty((0, 2), np.float64)
    return np.asarray(points, np.float64)


def _propagate(gray: np.ndarray, anchor_index: int, seeded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(gray)
    n = len(seeded)
    tracks = np.full((frame_count, n, 2), np.nan, np.float64)
    valid = np.zeros((frame_count, n), np.bool_)
    tracks[anchor_index] = seeded
    valid[anchor_index] = True
    lk_args = dict(winSize=(25, 25), maxLevel=4,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .008),
                   flags=0, minEigThreshold=1e-5)
    for direction in (-1, 1):
        current = anchor_index
        stop = 0 if direction < 0 else frame_count - 1
        while current != stop:
            nxt = current + direction
            active = np.flatnonzero(valid[current])
            if len(active) < 1:
                current = nxt
                continue
            p0 = np.ascontiguousarray((tracks[current, active] * .5).astype(np.float32).reshape(-1, 1, 2))
            p1, status1, _ = cv2.calcOpticalFlowPyrLK(gray[current], gray[nxt], p0, None, **lk_args)
            p0_check, status0, _ = cv2.calcOpticalFlowPyrLK(gray[nxt], gray[current], p1, None, **lk_args)
            forward_back = np.linalg.norm(p0_check[:, 0] - p0[:, 0], axis=1)
            in_image = (np.isfinite(p1[:, 0]).all(axis=1)
                        & (p1[:, 0, 0] >= 5) & (p1[:, 0, 0] < SMALL_WIDTH - 5)
                        & (p1[:, 0, 1] >= 5) & (p1[:, 0, 1] < SMALL_HEIGHT - 5))
            good = ((status1[:, 0] == 1) & (status0[:, 0] == 1)
                    & (forward_back < 1.0) & in_image)
            # A single frame's camera motion has one dominant warp. This
            # rejection removes the occasional skater/shadow seed that slips
            # through the anchor mask while preserving ice and board support.
            ids = np.flatnonzero(good)
            if len(ids) >= 12:
                old = p0[ids, 0]
                new = p1[ids, 0]
                H, inlier = cv2.findHomography(old, new, cv2.RANSAC, 2.0,
                                               maxIters=1000, confidence=.995)
                if H is not None and inlier is not None and int(inlier.sum()) >= 8:
                    warped = cv2.perspectiveTransform(old.reshape(-1, 1, 2), H)[:, 0]
                    residual = np.linalg.norm(warped - new, axis=1)
                    good[ids] &= residual <= 2.3
            kept = active[good]
            tracks[nxt, kept] = p1[good, 0].astype(np.float64) * 2.0
            valid[nxt, kept] = True
            current = nxt
    return tracks, valid


def _draw_review(image: np.ndarray, xy: np.ndarray, surface: np.ndarray,
                 valid: np.ndarray, out: Path) -> None:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for p, s, alive in zip(xy, surface, valid):
        if not alive:
            continue
        color = (58, 206, 53) if s == "ice" else (44, 170, 253)
        cv2.circle(canvas, tuple(np.rint(p * .5).astype(int)), 2, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "green: ice candidate   orange: board candidate", (22, 525),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, "green: ice candidate   orange: board candidate", (22, 525),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(out)


# Tight body envelopes read from the source at the nine listed frames. Linear
# interpolation is only an exclusion aid: do not use it as a skater track.
_BODY_KEYS = np.asarray([
    [0, 740, 105, 1280, 770],
    [15, 830, 250, 1110, 795],
    [30, 800, 365, 1110, 815],
    [45, 855, 390, 1160, 790],
    [60, 975, 310, 1225, 785],
    [75, 1010, 365, 1240, 790],
    [90, 945, 465, 1130, 800],
    [105, 850, 470, 1035, 800],
    [120, 870, 370, 1050, 805],
], np.float64)


def athlete_boxes(count: int) -> np.ndarray:
    indices = np.arange(count)
    boxes = np.column_stack([
        np.interp(indices, _BODY_KEYS[:, 0], _BODY_KEYS[:, component])
        for component in range(1, 5)
    ])
    return np.rint(boxes).astype(np.int32)


_BOARD_Y_KEYS = np.asarray([
    [0, 178, 253], [15, 282, 359], [30, 347, 437],
    [45, 375, 481], [60, 395, 492], [75, 407, 520],
    [90, 420, 523], [105, 430, 535], [120, 435, 550],
], np.float64)


def _pair_masks(gray_index: int, boxes: np.ndarray) -> dict[str, np.ndarray]:
    top = int(np.interp(gray_index, _BOARD_Y_KEYS[:, 0], _BOARD_Y_KEYS[:, 1]))
    bottom = int(np.interp(gray_index, _BOARD_Y_KEYS[:, 0], _BOARD_Y_KEYS[:, 2]))
    masks = {}
    board = np.zeros((SMALL_HEIGHT, SMALL_WIDTH), np.uint8)
    ice = np.zeros_like(board)
    board[max(0, (top + 6) // 2):min(SMALL_HEIGHT, (bottom - 5) // 2), 8:950] = 255
    ice[max(0, (bottom + 12) // 2):520, 7:953] = 255
    # In the middle of this pan the left boards curve down into pixels that a
    # simple horizontal board band would incorrectly label as ice.
    if 25 <= gray_index <= 80:
        wedge = np.asarray([[0, bottom - 10], [280, bottom - 10],
                            [0, bottom + 205]], np.float32)
        cv2.fillPoly(ice, [np.rint(wedge * .5).astype(np.int32)], 0)
    # The fixed television graphics contain excellent but false "features".
    for mask in (board, ice):
        mask[0:132, 0:282] = 0
        mask[453:540, 823:960] = 0
        x0, y0, x1, y1 = boxes[gray_index]
        cv2.rectangle(mask, ((x0 - 15) // 2, (y0 - 15) // 2),
                      ((x1 + 15) // 2, (y1 + 15) // 2), 0, -1)
    masks["board"] = board
    masks["ice"] = ice
    return masks


def _pair_rows(gray: np.ndarray, boxes: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Independent local seed/FB-LK observations for every adjacent pair.

    Both i->i+1 and i+1->i are measured. These rows may not all be linked to
    the anchor and must be attached to the common scene by the later solver.
    """
    arrays: dict[str, list] = {name: [] for name in
                               ("from_index", "to_index", "xy_from", "xy_to",
                                "surface", "fb_error_native")}
    counts = np.zeros((len(gray) - 1, 2, 2), np.int32)
    mask_cache = [_pair_masks(i, boxes) for i in range(len(gray))]
    lk_args = dict(winSize=(31, 31), maxLevel=4,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, .008),
                   flags=0, minEigThreshold=1e-5)
    for edge in range(len(gray) - 1):
        for direction_index, (src, dst) in enumerate(((edge, edge + 1), (edge + 1, edge))):
            candidate_points: list[np.ndarray] = []
            candidate_surface: list[np.ndarray] = []
            for label in ("board", "ice"):
                p = _seed_surface(gray[src], mask_cache[src][label],
                                  per_tile=10 if label == "board" else 5)
                candidate_points.append(p)
                candidate_surface.append(np.full(len(p), label, dtype="<U5"))
            if not any(len(part) for part in candidate_points):
                continue
            p0 = np.concatenate(candidate_points).astype(np.float32)
            surface = np.concatenate(candidate_surface)
            lk_input = np.ascontiguousarray(p0.reshape(-1, 1, 2))
            p1, status_f, _ = cv2.calcOpticalFlowPyrLK(gray[src], gray[dst],
                                                        lk_input, None, **lk_args)
            p0_back, status_b, _ = cv2.calcOpticalFlowPyrLK(gray[dst], gray[src],
                                                             p1, None, **lk_args)
            fb = np.linalg.norm(p0_back[:, 0] - p0, axis=1)
            target = p1[:, 0]
            x = np.clip(np.rint(target[:, 0]).astype(np.int32), 0, SMALL_WIDTH - 1)
            y = np.clip(np.rint(target[:, 1]).astype(np.int32), 0, SMALL_HEIGHT - 1)
            inside = np.isfinite(target).all(axis=1) & (target[:, 0] >= 5)
            inside &= (target[:, 0] < SMALL_WIDTH - 5) & (target[:, 1] >= 5)
            inside &= target[:, 1] < SMALL_HEIGHT - 5
            target_region = np.zeros(len(p0), np.bool_)
            for label in ("board", "ice"):
                where = surface == label
                target_region[where] = mask_cache[dst][label][y[where], x[where]] > 0
            good = ((status_f[:, 0] == 1) & (status_b[:, 0] == 1)
                    & np.isfinite(fb) & (fb < 1.25) & inside & target_region)
            ids = np.flatnonzero(good)
            if len(ids) >= 12:
                H, inlier = cv2.findHomography(p0[ids], target[ids], cv2.RANSAC,
                                               2.5, maxIters=1000, confidence=.995)
                if H is not None and inlier is not None and int(inlier.sum()) >= 8:
                    prediction = cv2.perspectiveTransform(p0[ids, None], H)[:, 0]
                    residual = np.linalg.norm(prediction - target[ids], axis=1)
                    good[ids] &= residual <= 2.5
            kept = np.flatnonzero(good)
            counts[edge, direction_index, 0] = int((surface[kept] == "board").sum())
            counts[edge, direction_index, 1] = int((surface[kept] == "ice").sum())
            if not len(kept):
                continue
            arrays["from_index"].append(np.full(len(kept), src, np.int16))
            arrays["to_index"].append(np.full(len(kept), dst, np.int16))
            arrays["xy_from"].append(p0[kept].astype(np.float64) * 2.0)
            arrays["xy_to"].append(target[kept].astype(np.float64) * 2.0)
            arrays["surface"].append(surface[kept])
            arrays["fb_error_native"].append(fb[kept].astype(np.float32) * 2.0)
    dtypes = {"from_index": np.int16, "to_index": np.int16,
              "xy_from": np.float64, "xy_to": np.float64,
              "surface": "<U5", "fb_error_native": np.float32}
    result = {key: np.concatenate(parts).astype(dtypes[key]) if parts else
              np.empty((0, 2) if key.startswith("xy_") else 0, dtype=dtypes[key])
              for key, parts in arrays.items()}
    result["edge_counts"] = counts
    result["athlete_bbox"] = boxes
    # These records document pixel selection, not verified static 3-D identity.
    stats = {
        "directed_pair_rows": int(len(result["from_index"])),
        "minimum_forward_board_per_edge": int(counts[:, 0, 0].min()),
        "minimum_forward_ice_per_edge": int(counts[:, 0, 1].min()),
        "median_forward_board_per_edge": float(np.median(counts[:, 0, 0])),
        "median_forward_ice_per_edge": float(np.median(counts[:, 0, 1])),
        "zero_forward_edge_count": int((counts[:, 0].sum(axis=1) == 0).sum()),
        "athlete_bbox_provenance": "source images at indices 0,15,...,120 manually inspected; linearly interpolated coarse body envelopes",
        "board_band_provenance": "source images at indices 0,15,...,120 manually inspected; linear board-strip envelope",
    }
    return result, stats


def _patch_ncc(left: np.ndarray, right: np.ndarray, xy0: np.ndarray,
               xy1: np.ndarray, side: int = 17) -> float:
    a = cv2.getRectSubPix(left, (side, side), tuple(map(float, xy0))).astype(np.float32)
    b = cv2.getRectSubPix(right, (side, side), tuple(map(float, xy1))).astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.sum(a * b) / denom) if denom > 1e-5 else -1.0


def _anchor_direct(gray: np.ndarray, anchor_xy: np.ndarray, surface: np.ndarray,
                   tracks: np.ndarray, valid: np.ndarray,
                   boxes: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Recheck long chains directly against unchanged anchor image patches.

    Direct LK uses the long chain only as a location guess. The source patch
    always comes from the anchor. SIFT can establish additional links where
    descriptors are distinctive. Neither mode treats an unknown link as alive.
    """
    anchor = 60
    masks = [_pair_masks(i, boxes) for i in range(len(gray))]
    sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=.012)
    keypoints = [cv2.KeyPoint(float(x * .5), float(y * .5), 12, -1, 0, 0, i)
                 for i, (x, y) in enumerate(anchor_xy)]
    anchor_kp, anchor_desc = sift.compute(gray[anchor], keypoints)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    rows: dict[str, list] = {k: [] for k in
                             ("target_index", "anchor_id", "xy_target", "surface",
                              "method", "quality")}
    by_target = {}
    lk_args = dict(winSize=(31, 31), maxLevel=4,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, .008),
                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=1e-5)
    for target in range(0, len(gray), 5):
        if target == anchor:
            continue
        accepted: dict[int, tuple[np.ndarray, str, float]] = {}
        ids = np.flatnonzero(valid[target])
        if len(ids):
            p0 = np.ascontiguousarray((anchor_xy[ids] * .5).astype(np.float32).reshape(-1, 1, 2))
            initial = np.ascontiguousarray((tracks[target, ids] * .5).astype(np.float32).reshape(-1, 1, 2))
            new, status_f, _ = cv2.calcOpticalFlowPyrLK(gray[anchor], gray[target],
                                                         p0, initial, **lk_args)
            back, status_b, _ = cv2.calcOpticalFlowPyrLK(gray[target], gray[anchor],
                                                          new, p0.copy(), **lk_args)
            fb = np.linalg.norm(back[:, 0] - p0[:, 0], axis=1)
            for j, point_id in enumerate(ids):
                if status_f[j, 0] != 1 or status_b[j, 0] != 1 or not np.isfinite(fb[j]) or fb[j] > 1.1:
                    continue
                candidate = new[j, 0]
                x, y = np.rint(candidate).astype(int)
                if not (5 <= x < SMALL_WIDTH - 5 and 5 <= y < SMALL_HEIGHT - 5):
                    continue
                label = str(surface[point_id])
                if masks[target][label][y, x] == 0:
                    continue
                corr = _patch_ncc(gray[anchor], gray[target], p0[j, 0], candidate)
                if corr < .70:
                    continue
                accepted[int(point_id)] = (candidate.astype(np.float64) * 2.0,
                                           "lk", float(corr))
        if anchor_desc is not None and len(anchor_kp) >= 8:
            union = cv2.bitwise_or(masks[target]["board"], masks[target]["ice"])
            target_kp, target_desc = sift.detectAndCompute(gray[target], union)
            if target_desc is not None and len(target_kp) >= 8:
                forward = bf.knnMatch(anchor_desc, target_desc, k=2)
                backward = bf.knnMatch(target_desc, anchor_desc, k=2)
                reverse = {pair[0].queryIdx: pair[0].trainIdx for pair in backward
                           if len(pair) >= 2 and pair[0].distance < .72 * pair[1].distance}
                proposals = []
                for pair in forward:
                    if len(pair) < 2 or pair[0].distance >= .72 * pair[1].distance:
                        continue
                    m = pair[0]
                    if reverse.get(m.trainIdx) != m.queryIdx:
                        continue
                    point_id = int(anchor_kp[m.queryIdx].class_id)
                    if point_id < 0 or point_id >= len(anchor_xy):
                        continue
                    xy = np.asarray(target_kp[m.trainIdx].pt, np.float64)
                    x, y = np.rint(xy).astype(int)
                    label = str(surface[point_id])
                    if x < 0 or x >= SMALL_WIDTH or y < 0 or y >= SMALL_HEIGHT:
                        continue
                    if masks[target][label][y, x] == 0:
                        continue
                    proposals.append((point_id, xy, 1.0 - float(m.distance / pair[1].distance)))
                # Reject spatially inconsistent repeated text/line matches.
                if len(proposals) >= 8:
                    points = np.asarray([anchor_xy[q[0]] * .5 for q in proposals], np.float32)
                    targets = np.asarray([q[1] for q in proposals], np.float32)
                    H, inlier = cv2.findHomography(points, targets, cv2.RANSAC,
                                                   3.0, maxIters=2000, confidence=.997)
                    if H is not None and inlier is not None and int(inlier.sum()) >= 8:
                        predicted = cv2.perspectiveTransform(points[:, None], H)[:, 0]
                        residual = np.linalg.norm(predicted - targets, axis=1)
                        for j, (point_id, xy, confidence) in enumerate(proposals):
                            if inlier[j, 0] != 1 or residual[j] > 3.0:
                                continue
                            if point_id in accepted:
                                # Direct image-patch LK has better local pixel
                                # precision. A disagreement makes SIFT unused.
                                continue
                            accepted[point_id] = (xy * 2.0, "sift", confidence)
        by_target[target] = {"lk": 0, "sift": 0}
        for point_id, (xy, method, score) in accepted.items():
            rows["target_index"].append(target)
            rows["anchor_id"].append(point_id)
            rows["xy_target"].append(xy)
            rows["surface"].append(surface[point_id])
            rows["method"].append(method)
            rows["quality"].append(score)
            by_target[target][method] += 1
    array = {
        "target_index": np.asarray(rows["target_index"], np.int16),
        "anchor_id": np.asarray(rows["anchor_id"], np.int32),
        "xy_target": np.asarray(rows["xy_target"], np.float64).reshape(-1, 2),
        "surface": np.asarray(rows["surface"], dtype="<U5"),
        "method": np.asarray(rows["method"], dtype="<U4"),
        "quality": np.asarray(rows["quality"], np.float32),
    }
    stats = {"target_counts": by_target,
             "row_count": int(len(array["target_index"])),
             "note": "Direct matches to the unchanged anchor image at every fifth frame; not all targets or surfaces have enough matches"}
    return array, stats


def run(source: Path, run_dir: Path, regions: dict = DEFAULT_REGIONS) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    pts = np.arange(FIRST_PTS, LAST_PTS + PTS_STEP, PTS_STEP, dtype=np.int64)
    if len(pts) != 121 or int(pts[60]) != ANCHOR_PTS:
        raise AssertionError("Incorrect time interval")
    gray = decode_native(source, run_dir / "source.mp4", pts)
    anchor_index = 60
    surface_parts: list[np.ndarray] = []
    seed_parts: list[np.ndarray] = []
    for surface in ("board", "ice"):
        mask = _mask(regions[surface], regions["exclude"])
        points = _seed_surface(gray[anchor_index], mask)
        seed_parts.append(points * 2.0)
        surface_parts.append(np.full(len(points), surface, dtype="<U5"))
    if not seed_parts or sum(len(x) for x in seed_parts) < 20:
        raise ValueError("Too few static image candidates at anchor")
    anchor_xy = np.concatenate(seed_parts)
    surface = np.concatenate(surface_parts)
    tracks, valid = _propagate(gray, anchor_index, anchor_xy)
    if not np.array_equal(tracks[anchor_index], anchor_xy):
        raise AssertionError("Anchor was changed by tracker")
    np.savez_compressed(run_dir / "observations.npz", pts=pts,
                        anchor_index=np.int64(anchor_index), anchor_xy=anchor_xy,
                        tracks=tracks, valid=valid, surface=surface,
                        gray_small=gray)
    # Display only candidates spanning the anchor's immediate neighborhood.
    stable = valid[max(0, anchor_index - 8)] & valid[min(len(pts) - 1, anchor_index + 8)]
    _draw_review(gray[anchor_index], anchor_xy, surface, stable,
                 run_dir / "anchor_tracking.jpg")
    boxes = athlete_boxes(len(pts))
    pairs, pair_stats = _pair_rows(gray, boxes)
    np.savez_compressed(run_dir / "pair_tracks.npz", **pairs)
    direct, direct_stats = _anchor_direct(gray, anchor_xy, surface, tracks, valid, boxes)
    np.savez_compressed(run_dir / "anchor_direct.npz", **direct)
    source_record = {
        "original_source": str(source.resolve()),
        "clip": "source.mp4",
        "observation_file": "observations.npz",
        "anchor_review": "anchor_tracking.jpg",
        "original_time_base": "1/15360",
        "frame_rate": "60/1",
        "width": 1920, "height": 1080,
        "first_pts": int(pts[0]), "last_pts": int(pts[-1]),
        "anchor_pts": ANCHOR_PTS, "anchor_index": anchor_index,
        "frame_count": int(len(pts)), "native_pts_step": PTS_STEP,
        "first_seconds": float(pts[0]) / 15360,
        "last_seconds": float(pts[-1]) / 15360,
        "time_span_seconds_inclusive_endpoints": 2.0,
        "clip_encoding": "H.264 yuv420p 60 fps visual copy; exact original PTS in observations.npz",
        "gray_scale": {"width": SMALL_WIDTH, "height": SMALL_HEIGHT, "factor_from_native": .5},
        "anchor_regions_native_xy": regions,
        "surface_semantics": "candidate board/ice location at anchor; geometric identity not yet established",
        "track_semantics": "adjacent-frame bidirectional LK with forward-backward and one-frame warp checks; NaN/false after loss; no interpolation",
        "pair_file": "pair_tracks.npz",
        "pair_semantics": "120 adjacent source-frame edges measured in both directions; independent fresh local seeds, native xy; repeated physical points may appear in multiple rows",
        "athlete_bbox_location": "pair_tracks.npz:athlete_bbox, shape (121,4), native xyxy, coarse exclusion only",
        "pair_statistics": pair_stats,
        "anchor_direct_file": "anchor_direct.npz",
        "anchor_direct_semantics": "direct unchanged-anchor patch LK and distinctive SIFT at every fifth frame; unknown links omitted; target native xy",
        "anchor_direct_statistics": direct_stats,
        "points": int(len(anchor_xy)),
        "anchor_board_candidates": int(np.count_nonzero(surface == "board")),
        "anchor_ice_candidates": int(np.count_nonzero(surface == "ice")),
        "valid_counts": {"first": int(valid[0].sum()), "anchor": int(valid[60].sum()),
                         "last": int(valid[-1].sum())},
        "valid_surface_counts": {
            label: {"first": int((valid[0] & (surface == label)).sum()),
                    "anchor": int((valid[60] & (surface == label)).sum()),
                    "last": int((valid[-1] & (surface == label)).sum())}
            for label in ("board", "ice")},
        "limitation": "Two-dimensional static candidate tracks only; no 3-D scene match or camera accuracy assertion",
    }
    (run_dir / "source.json").write_text(json.dumps(source_record, ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
    return source_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--regions-json", type=Path,
                        help="Optional board/ice/exclude native-pixel polygon JSON")
    args = parser.parse_args()
    regions = json.loads(args.regions_json.read_text(encoding="utf-8")) if args.regions_json else DEFAULT_REGIONS
    report = run(args.source, args.run_dir, regions)
    print(json.dumps({k: report[k] for k in ("frame_count", "points",
                                               "anchor_board_candidates", "anchor_ice_candidates",
                                               "valid_counts", "valid_surface_counts")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
