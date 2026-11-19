"""Measure fixed board and ice texture correspondences on the native exposures."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import cv2
import numpy as np

from common import (HEIGHT, OUTPUT, WIDTH, frozen_scene, observation_packet, save, scope, sha,
                    source_frames, visibility_packet, write)
from measure import DT, FIRST, load_camera, motion_clock
from review import restore_source_graphics
from stability_tracks import (BLOCK, SIZE, Tracklet, bind, board_mask, ice_corners, ji, measured_pair,
                              pick, rail_points, spatial_roles)

HALF = 7
SEED_STEP = 4
AGE_LIMIT = 8
LUMINANCE_SIZE = (2 * HALF + 1, 2 * HALF + 1)


def subpixel_peak(a, at, y, x):
    if not (0 <= y - at[0] and y + at[0] < a.shape[0] and
            0 <= x - at[1] and x + at[1] < a.shape[1]):
        return 0.
    v, left, right = a[y, x], a[y - at[0], x - at[1]], a[y + at[0], x + at[1]]
    k = float(left - 2 * v + right)
    return .5 * float(np.clip((left - right) / k, -1., 1.)) if k < -1.e-5 else 0.


def correlate(template_img, target, permitted, point, expected, radius, minimum):
    ox, oy = np.rint(expected).astype(int)
    sx, sy = radius
    a, b = max(HALF, ox - sx), max(HALF, oy - sy)
    c, d = min(SIZE[0] - HALF, ox + sx + 1), min(SIZE[1] - HALF, oy + sy + 1)
    if c <= a or d <= b:
        return None
    valid = permitted[b:d, a:c] != 0
    if not valid.any():
        return None
    patch = cv2.getRectSubPix(template_img, LUMINANCE_SIZE, tuple(map(float, point)))
    if np.std(patch) < 2.05:
        return None
    selected = target[b-HALF:d+HALF, a-HALF:c+HALF]
    score = cv2.matchTemplate(selected, patch, cv2.TM_CCOEFF_NORMED)
    score[~valid] = -1.
    _, best, _, (x,y) = cv2.minMaxLoc(score)
    if best < minimum:
        return None
    dx, dy = subpixel_peak(score, (0,1), y,x), subpixel_peak(score, (1,0), y,x)
    score[max(0, y - 4):y + 5, max(0, x - 5):x + 6] = -1.
    rival = float(score.max())
    if rival > best - .043 or rival > .965 * best:
        return None
    return np.array([a + x + dx, b + y + dy], np.float32), float(best)


def pair_point(old, image, point, family, hint, long=False):
    left, left_masks = old
    right, right_masks = image
    cutoff = .88 if family == 0 else .91
    limits = ((50, 18) if family == 0 else (45, 25)) if not long else (12, 10)
    initial = correlate(left, right, right_masks[family], point, hint, limits, cutoff)
    if initial is None:
        return None
    current, correlation = initial
    reverse = correlate(right, left, left_masks[family], current, point, (13, 10), cutoff - .020)
    if reverse is None:
        return None
    cycle = float(np.linalg.norm(reverse[0] - point))
    if cycle > (.53 if not long else .88):
        return None
    return current, float((1. - min(correlation, reverse[1])) * 9. + cycle)


def source_foot(data, n):
    use = data["eligible"] & (data["path"] == "board/continuous/bottom")
    coords = data["uv"][use]
    index = data["idx"][use]
    order = np.argsort(index, kind="stable")
    all_foot = np.full((n, SIZE[0]), np.nan, np.float32)
    grid = np.arange(SIZE[0]) * 2. + .5
    for take in np.split(order, np.flatnonzero(np.diff(index[order])) + 1):
        if len(take) < 4 or np.ptp(coords[take, 0]) < 350.:
            continue
        x, y = coords[take].T
        sort = np.argsort(x, kind="stable")
        x, y = x[sort], y[sort]
        knots = np.unique(np.clip((x // 75.).astype(int), 0, 25))
        station, height = [], []
        for k in knots:
            k_use = (x >= 75. * k) & (x < 75. * (k + 1))
            if k_use.any():
                station.append(np.median(x[k_use]))
                height.append(np.median(y[k_use]))
        if len(station) >= 4:
            curve = np.interp(grid, station, height, left=np.nan, right=np.nan)
            all_foot[index[take[0]]] = curve
    return all_foot


def surfaces(person, raw, p, rail, status, nrail, toes, cropped_board):
    danger = restore_source_graphics(raw, p, person)
    blocked = cv2.resize(danger, SIZE, interpolation=cv2.INTER_AREA) > 0.
    blocked = cv2.dilate(blocked.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
    board = board_mask(rail, status, nrail)
    from audit_tail_source import y_curve
    grid = np.arange(SIZE[0]) * 2. + .5
    projected = y_curve(rail[nrail:], status[nrail:], grid)
    feet = np.where(np.isfinite(toes), toes, projected)
    yy = np.arange(SIZE[1])[:, None] * 2. + .5
    if cropped_board:
        # The source-reviewed closing close-up shows board print to the top
        # sensor edge. Use the measured ice-side toe where the cap is cropped.
        inside = (yy > 15.) & (yy < toes[None, :] - 25.)
        board[inside] = 255
    ice = (yy > feet[None, :] + 20.) & (yy < HEIGHT - 13.) & (yy > 160.)
    # Unknown fore/aft scene regions stay unavailable. Actual texture can
    # lie between named paint features, anywhere on the visible ice side.
    board[blocked] = 0
    ice[blocked | (board != 0)] = False
    kernel = np.ones(LUMINANCE_SIZE, np.uint8)
    board_safe, ice_safe = (cv2.erode(q.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                                     borderValue=0) for q in (board, ice))
    # A bidirectional grayscale match can also follow skater shadows. The
    # ice cohort requires actual colored paint within the fixed-size patch.
    b, g, r = np.moveaxis(cv2.resize(raw, SIZE, interpolation=cv2.INTER_AREA).astype(np.float32), 2, 0)
    chroma = np.maximum(abs(r - .55 * g - .45 * b), abs(b - .60 * g - .40 * r))
    local_span = cv2.morphologyEx(chroma, cv2.MORPH_GRADIENT, np.ones((9, 9), np.uint8))
    ice_safe[(local_span < 7.) | (cv2.dilate(chroma, np.ones((9, 9), np.uint8)) < 8.5)] = 0
    return board_safe, ice_safe


def patched_seeds(gray, masks):
    by_surface = (pick(gray, masks[0], 44), ice_corners(gray, pick(gray, masks[1], 55)))
    return (np.vstack(by_surface),
            np.r_[np.zeros(len(by_surface[0]), np.uint8), np.ones(len(by_surface[1]), np.uint8)])


def flow_consistency(a, b, family):
    accepted = np.ones(len(a), bool)
    for f in (0, 1):
        selection = np.flatnonzero(family == f)
        if len(selection) < 7:
            continue
        u, v = a[selection], b[selection]
        matrix, mask = cv2.estimateAffine2D(u, v, method=cv2.RANSAC,
                ransacReprojThreshold=1.35 if f == 0 else 1.10, maxIters=250, refineIters=3)
        if matrix is None or mask.sum() < 5:
            continue
        predicted = np.c_[u, np.ones(len(u))] @ matrix.T
        d = np.linalg.norm(v - predicted, axis=1)
        accepted[selection] = d <= (2.9 if f == 0 else 2.)
    return accepted


def collect(native, record, shot, guide, named, visibility, destination):
    sid=shot["id"]
    pts, controls, rig = load_camera(shot, guide)
    times = motion_clock(shot, guide, pts)
    folds = (np.arange(len(pts)) // BLOCK + int(sid[1:])) % 4
    rails, foot = rail_points(native)
    height = float(native.scene["board"][5])
    positions = np.repeat(rig[None, :], len(pts), axis=0)
    motion = SimpleNamespace(times=times, first=FIRST, dt=DT)
    with np.load(visibility_packet(visibility, sid)) as f:
        np.testing.assert_array_equal(f["pts"], pts)
        assert f["source_sha256"].item() == record["source"]["sha256"]
        packed, mask_valid = f["packed"], f["valid"]
    if not mask_valid.all():
        raise ValueError(f"Incomplete source silhouette for {sid}")
    with np.load(observation_packet(named, sid)) as f:
        np.testing.assert_array_equal(f["pts"], pts[f["idx"]])
        toes = source_foot(f, len(pts))
    pieces = []
    recent = deque(maxlen=AGE_LIMIT+1)
    live = []
    src = source_frames(record["source"]["registered_path"], pts)
    for a in range(0, len(pts), 40):
        b = min(a + 40, len(pts))
        projected = ji.project_groups(native, positions, controls, FIRST, DT,
                   [(i, "rail", times[i]) for i in range(a,b)], {"rail": rails})
        for i, (rail, status) in enumerate(projected, a):
            p, frame = next(src)
            assert p == pts[i]
            native_gray = frame.to_ndarray(format="gray")
            gray = cv2.resize(native_gray, SIZE, interpolation=cv2.INTER_AREA)
            raw = frame.to_ndarray(format="bgr24")
            person = np.unpackbits(packed[i], axis=1, count=WIDTH)
            masks = surfaces(person, raw, p, rail, status, len(foot), toes[i], sid == "s20")
            current = (gray, masks)
            recent.append(current)
            if live:
                tracks = []
                observed_left, observed_right, family = [], [], []
                for g in live:
                    hit = [pair_point(recent[-2], current, q, int(f), q)
                           for q, f in zip(g.current, g.family, strict=True)]
                    valid = np.flatnonzero([x is not None for x in hit])
                    if not len(valid):
                        continue
                    xy = np.array([hit[j][0] for j in valid])
                    quality = np.array([hit[j][1] for j in valid], np.float32)
                    tracks.append((g, valid, xy, quality))
                    observed_left.append(g.current[valid])
                    observed_right.append(xy)
                    family.append(g.family[valid])
                updated = []
                if tracks:
                    keep = flow_consistency(np.vstack(observed_left), np.vstack(observed_right),
                                            np.concatenate(family))
                    offset = 0
                    for g, ix, xy, quality in tracks:
                        count = len(ix)
                        use = keep[offset:offset+count]
                        offset += count
                        if not use.any():
                            continue
                        ix, xy, quality = ix[use], xy[use], quality[use]
                        age = i - g.start
                        t = Tracklet(g.start, g.ids[ix], g.family[ix], g.world[ix], g.anchor[ix],
                                     xy, g.current[ix])
                        if age <= SEED_STEP + 1:
                            triple = g.previous[ix] if age >= 2 else np.full_like(xy, np.nan)
                            measurement = Tracklet(t.start, t.ids, t.family, t.world,
                                                  t.anchor, t.current, triple)
                            measured_pair(pieces, i, measurement, t.previous, 1, quality, folds[i])
                        if age == AGE_LIMIT:
                            pair = [pair_point(recent[0], current, pt, int(f), estimate, long=True)
                                    for pt, f, estimate in zip(t.anchor, t.family, xy, strict=True)]
                            pair_keep = np.flatnonzero([x is not None for x in pair])
                            if len(pair_keep):
                                v = np.array([pair[j][0] for j in pair_keep])
                                error = np.linalg.norm(v - xy[pair_keep], axis=1)
                                pair_keep = pair_keep[error <= 1.25]
                                if len(pair_keep):
                                    raw_xy = np.array([pair[j][0] for j in pair_keep])
                                    lag_quality = np.array([pair[j][1] for j in pair_keep], np.float32)
                                    long = Tracklet(t.start, t.ids[pair_keep], t.family[pair_keep],
                                            t.world[pair_keep], t.anchor[pair_keep], raw_xy, t.previous[pair_keep])
                                    measured_pair(pieces, i, long, long.anchor, AGE_LIMIT, lag_quality, folds[i])
                        else:
                            updated.append(t)
                live = updated
            if i % SEED_STEP == 0 and i + 1 < len(pts):
                q, family = patched_seeds(gray, masks)
                if len(q):
                    keep, xyz = bind(native.core, motion, controls, rig, i, q, family, foot, height)
                    if len(keep):
                        ids = 1_000_000 + 256 * i + keep
                        live.append(Tracklet(i, ids.astype(np.int32), family[keep], xyz, q[keep], q[keep], q[keep]))
            if i % 320 == 319:
                print(sid, i+1, "/", len(pts), "native patch frames", flush=True)
    if not pieces:
        raise ValueError("No qualified stationary matches in " + sid)
    rows = {k: np.concatenate([x[k] for x in pieces]) for k in pieces[0]}
    rows["role"] = spatial_roles(rows["world"], rows["family"], foot)
    save(destination / f"{sid}.npz", pts=pts, **rows, graphics_mask_per_exposure=True,
         source_sha256=record["source"]["sha256"], geometry_sha256=record["geometry_binding"]["value_sha256"])
    report = dict(shot=sid, frames=len(pts), rows=len(rows["index"]), methods=["bidirectional_local_NCC",
                  "local_raw_displacement_consistency", "direct_anchor_recheck_at_lag8",
                  "ice_chromatic_paint_guard"],
                  uses_geometry_for=["surface_search_mask", "relative_track_surface_lift"],
                  source_independent_match=True, by_family=dict(Counter(map(int, rows["family"]))))
    for number, name in enumerate(("board", "ice")):
        relevant = rows["family"] == number
        adj = relevant & (rows["lag"] == 1)
        lag = relevant & (rows["lag"] == AGE_LIMIT)
        report[name + "_adjacent_exposures"] = int(np.unique(rows["index"][adj]).size)
        report[name + "_lag8_exposures"] = int(np.unique(rows["index"][lag]).size)
    write(destination / f"{sid}.json", report)
    print(report, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--guide", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--visibility", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shots", nargs="+")
    a = p.parse_args()
    cv2.setNumThreads(1)
    record = scope()
    if sha(record["source"]["registered_path"]) != record["source"]["sha256"]:
        raise ValueError("Original file has changed")
    ji.enable_historical_source_readers()
    native = frozen_scene(record, verify=True)
    folder = OUTPUT / a.out
    folder.mkdir(exist_ok=True)
    shots = [s for s in record["shots"] if (a.shots is None or s["id"] in a.shots)
             and not (folder / f'{s["id"]}.json').is_file()]
    shots.sort(key=lambda s: s["id"] not in {"s02", "s04", "s14", "s18", "s20"})
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs = [pool.submit(collect, native, record, shot, OUTPUT/a.guide, OUTPUT/a.observations,
                            OUTPUT/a.visibility, folder) for shot in shots]
        for job in jobs:
            job.result()


if __name__ == "__main__":
    main()
