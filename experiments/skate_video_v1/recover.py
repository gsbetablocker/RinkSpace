"""Reread missed source exposures from visible neighbouring boot tracks."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import time

import cv2
import numpy as np

from evidence import segments
from inputs import Inputs, atomic_npz, emit
from track import Navigation, VideoMasks, WEIGHTS, iou, measure


def common_crop(chips, bounds, j, k, poly):
    span = max(25., .7 * np.linalg.norm(np.ptp(poly, axis=0)))
    lo = np.maximum(bounds[[j, k], :2].max(axis=0),
                    np.floor(poly.min(axis=0) - span)).astype(int)
    hi = np.minimum(bounds[[j, k], 2:].min(axis=0),
                    np.ceil(poly.max(axis=0) + span)).astype(int)
    if np.min(hi-lo) < 16 or (poly.min(axis=0) < lo+1).any() or (poly.max(axis=0) > hi-2).any():
        return None
    result = []
    for i in (j, k):
        a, b = lo - bounds[i, :2]
        c, d = hi - bounds[i, :2]
        result.append(cv2.cvtColor(chips[i][b:d, a:c], cv2.COLOR_BGR2GRAY))
    return lo, result


def advance(chips, bounds, j, k, poly):
    pair = common_crop(chips, bounds, j, k, poly)
    if pair is None:
        return None
    origin, (g0, g1) = pair
    shape = np.zeros(g0.shape, np.uint8)
    cv2.fillPoly(shape, [np.rint(poly - origin).astype(np.int32)], 255)
    interior = cv2.goodFeaturesToTrack(g0, maxCorners=30, qualityLevel=.015, minDistance=3, mask=shape)
    boundary = (poly[::3] - origin).astype(np.float32)
    xy = boundary if interior is None else np.vstack((interior[:, 0], boundary))
    xy = xy.reshape(-1, 1, 2)
    dest, ok0, _ = cv2.calcOpticalFlowPyrLK(g0, g1, xy, None, winSize=(21, 21), maxLevel=3)
    if dest is None:
        return None
    back, ok1, _ = cv2.calcOpticalFlowPyrLK(g1, g0, dest, None, winSize=(21, 21), maxLevel=3)
    if back is None:
        return None
    error = np.linalg.norm((back - xy)[:, 0], axis=-1)
    keep = (error < 1.6) & ok0.ravel().astype(bool) & ok1.ravel().astype(bool)
    if keep.sum() < max(9, len(xy) * .45):
        return None
    T, good = cv2.estimateAffinePartial2D(xy[keep, 0], dest[keep, 0],
                                         method=cv2.RANSAC, ransacReprojThreshold=2.1)
    if T is None or good.mean() < .62:
        return None
    scale = np.linalg.norm(T[:, 0])
    if not .77 < scale < 1.28:
        return None
    moved = (poly - origin) @ T[:, :2].T + T[:, 2] + origin
    if np.linalg.norm(moved.mean(axis=0)-poly.mean(axis=0)) > .92 * np.linalg.norm(np.ptp(poly, axis=0)):
        return None
    return moved.astype(np.float32)


def reuse(chips, bounds, first, anchor, foot, packet, model, nav):
    """Propagate from a source mask without clicking on the evaluation exposure."""
    model.reset()
    order = list(range(anchor, first + (1 if first > anchor else -1), 1 if first > anchor else -1))
    mask = np.zeros(chips[anchor].shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(packet["boot"][anchor, foot] - bounds[anchor, :2]).astype(np.int32)], 1)
    for t, k in enumerate(order):
        result = model.step(chips[k], t, {}, seed_masks={foot: mask} if t == 0 else None)
    if foot not in result:
        return None
    current_mask, score = result[foot]
    item = measure(current_mask, score, bounds[first], nav.size[packet["first_index"] + first, foot])
    return item[0] if item is not None else None


def source_prompt_repair(inputs, nav, packet, model, anchors):
    """Reacquire specific boots in native crops with independently read prompts."""
    start = inputs.index[int(packet["source_pts"][0])]
    for row in anchors:
        first, seed, last = (inputs.index[row[name]] for name in ("start_pts", "pts", "end_pts"))
        foot = row["foot"]
        if not (start <= first <= seed <= last < start + len(packet["source_pts"])) or inputs.withheld[seed]:
            raise ValueError("Boot review anchor crosses a shot or the check protection")
        x, y, r, b = row["bounds"]
        origin = np.array([x, y])
        positive = np.asarray(row["boot_uv"], dtype=np.float32)
        negative = np.asarray(row["exclude_uv"], dtype=np.float32)
        points = np.vstack((positive, negative)) - origin
        labels = [1] * len(positive) + [0] * len(negative)
        if foot not in (0, 1) or ((points < 0) | (points >= [r-x, b-y])).any():
            raise ValueError("Invalid boot review prompts")
        prompt = {foot: (points.tolist(), labels, 2)}
        chips = [np.ascontiguousarray(frame[y:b, x:r]) for _, frame in inputs.frames(first, last + 1)]
        bounds = np.broadcast_to([x, y, r, b], (len(chips), 4))
        for direction, end in ((-1, first), (1, last)):
            model.reset()
            previous = None
            for t, i in enumerate(range(seed, end + direction, direction)):
                k = i - first
                prediction = model.step(chips[k], t, prompt if not t else {})
                if foot not in prediction:
                    break
                item = measure(*prediction[foot], bounds[k], nav.size[i, foot])
                if item is None:
                    break
                polygon, score, partial = item
                if previous is None:
                    if (any(cv2.pointPolygonTest(polygon, tuple(uv), False) < 0 for uv in positive)
                            or any(cv2.pointPolygonTest(polygon, tuple(uv), False) >= 0 for uv in negative)):
                        break
                else:
                    warped = advance(chips, bounds, k - direction, k, previous)
                    if warped is None or iou(polygon, warped) < .52:
                        break
                j = i - start
                packet["boot"][j, foot] = polygon
                packet["boot_status"][j, foot] = 1
                packet["boot_confidence"][j, foot] = score
                packet["roi_partial"][j, foot] = partial
                packet["repair_method"][j, foot] = 4
                packet["repair_anchor_pts"][j, foot] = [row["pts"], -1]
                previous = polygon
    return packet


def repair(inputs, nav, packet, model):
    a = inputs.index[int(packet["source_pts"][0])]
    b = inputs.index[int(packet["source_pts"][-1])] + 1
    bounds = packet["chip_bounds"]
    originals = packet["boot"].copy()
    status = packet["boot_status"].copy()
    packet["first_index"] = a
    anchors = (status == 1) & ~packet["roi_partial"] & ~inputs.withheld[a:b, None]
    recover = np.zeros_like(status)
    sources = np.full((*status.shape, 2), -1, np.int64)
    chips = [np.ascontiguousarray(frame[y0:y1, x0:x1])
             for (_, frame), (x0, y0, x1, y1) in zip(inputs.frames(a, b), bounds, strict=True)]
    replay = []
    for f in range(2):
        confirmed = np.flatnonzero(anchors[:, f])
        for begin, end in segments(status[:, f] == 0):
            j = np.searchsorted(confirmed, begin)
            k = np.searchsorted(confirmed, end, side="right")
            previous = int(confirmed[j-1]) if j else -1
            after = int(confirmed[k]) if k < len(confirmed) else b-a
            # Need local support on both sides. Each hop is validated by the
            # native round trip, and long unconstrained gaps remain explicit.
            available = defaultdict(list)
            for anchor, stop, direction in ((previous, end, 1), (after, begin, -1)):
                if not 0 <= anchor < b - a:
                    continue
                poly = originals[anchor, f]
                for t in range(anchor + direction, stop + direction, direction):
                    poly = advance(chips, bounds, t - direction, t, poly)
                    if poly is None:
                        break
                    if begin <= t <= end:
                        available[t].append((anchor, poly))
            for t in range(begin, end + 1):
                other = packet["boot"][t, 1-f]
                candidates = available[t]
                if len(candidates) == 2 and iou(candidates[0][1], candidates[1][1]) >= .54:
                    proposal = max(candidates, key=lambda c: -abs(c[0]-t))[1]
                    method = 1
                elif len(candidates) and abs(candidates[0][0] - t) <= 2:
                    proposal = candidates[0][1]
                    actual = [poly for poly, score in zip(packet["direction_boot"][t], packet["direction_scores"][t])
                              if np.isfinite(score)]
                    if actual:
                        overlap = [iou(proposal, poly) for poly in actual]
                        if max(overlap) >= .56:
                            proposal = actual[int(np.argmax(overlap))]
                            method = 2
                        else:
                            continue
                    else:
                        if not (0 <= previous < t < after < b-a and min(t-previous, after-t) <= 2):
                            continue
                        # Use an actual video prediction for a skipped network
                        # exposure when flow exists from only one side.
                        replay.append((t, f, candidates[0][0], proposal))
                        continue
                else:
                    continue
                if np.isfinite(other).all() and iou(proposal, other) > .37:
                    continue
                packet["boot"][t, f] = proposal
                packet["boot_status"][t, f] = 1
                packet["boot_confidence"][t, f] = 1.
                recover[t, f] = method
                sources[t, f, :len(candidates)] = packet["source_pts"][[c[0] for c in candidates]]
    for t, f, anchor, flow in replay:
        candidate = reuse(chips, bounds, t, anchor, f, packet, model, nav)
        if candidate is None or iou(candidate, flow) < .53:
            continue
        other = packet["boot"][t, 1-f]
        if np.isfinite(other).all() and iou(candidate, other) > .37:
            continue
        packet["boot"][t, f] = candidate
        packet["boot_status"][t, f] = 1
        packet["boot_confidence"][t, f] = 1.
        recover[t, f] = 3
        sources[t, f, 0] = packet["source_pts"][anchor]
    packet.pop("first_index")
    packet["repair_method"] = recover
    packet["repair_anchor_pts"] = sources
    return packet


def main(source, output):
    cv2.setNumThreads(1)
    inputs = Inputs()
    nav = Navigation(inputs)
    model = VideoMasks(WEIGHTS)
    (output / "tracked").mkdir(parents=True, exist_ok=True)
    for path in sorted((source / "tracked").glob("track_s*.npz")):
        dst = output / "tracked" / path.name
        if dst.exists():
            continue
        began = time.perf_counter()
        with np.load(path, allow_pickle=False) as f:
            packet = {k: f[k] for k in f.files}
        data = repair(inputs, nav, packet, model)
        atomic_npz(dst, **data)
        emit("source_recovery", shot=str(data["shot_ids"][0]), source_exposures=len(data["source_pts"]),
             added_feet=int((data["repair_method"] > 0).sum()),
             via_mask_memory=int((data["repair_method"] == 3).sum()),
             seconds=round(time.perf_counter() - began, 2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
