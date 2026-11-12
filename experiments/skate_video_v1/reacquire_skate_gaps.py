"""Reacquire the same whole skate using visible feature tracks and source masks."""

from argparse import ArgumentParser
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from evidence import segments
from follow_whole_skates import PRIOR_WHOLE, SOURCE_PROMPT, VIDEO_RECOVERY, VIDEO_WHOLE, WholeShape, boundary_support, write_navigation
from inputs import PROGRAMME, ROOT, SIZE, WEIGHTS, Inputs, atomic_json, atomic_npz, distribute, emit, geometry, read
from model import SkateModel
from recover import advance
from relink_whole_skates import RECOVER, REPLACE, SOURCE, Shot, diagonal, distance, report
from track import Navigation, VideoMasks, iou, mask_overlap, measure, unit
from whole_skate_review import filled


class ImageTracks:
    def __init__(self):
        import torch

        sys.path.insert(0, str(ROOT/"runtime_tools/body/CoTracker"))
        from cotracker.predictor import CoTrackerOnlinePredictor

        self.torch = torch
        torch.backends.cuda.matmul.allow_tf32 = True
        self.model = CoTrackerOnlinePredictor(
            checkpoint=str(ROOT/"cache/pretrained/cotracker/scaled_online.pth"), window_len=16,
        ).eval().cuda()

    def trace(self, chips, bounds, outline, start, stop):
        t = self.torch
        direction = 1 if stop > start else -1
        steps = np.arange(start, stop+direction, direction)
        # The moving crop only changes the image domain of the point tracker.
        # Both fitting and verification below use native source coordinates.
        h, w = self.model.interp_shape
        size = np.array([w, h])
        lo = bounds[steps, :2]
        ratios = size/(bounds[steps, 2:]-lo)
        first = chips[start]
        origin = lo[0]
        mask = filled(first.shape[:2], outline-origin)
        gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        sites = cv2.goodFeaturesToTrack(gray, 12, .013, 4., mask=mask)
        near_outline = (.92*outline[::6]+.08*outline.mean(axis=0)).astype(np.float32)
        points = near_outline if sites is None else np.vstack((near_outline, sites[:, 0]+origin))
        query = np.c_[np.zeros(len(points)), (points-origin+.5)*ratios[0]-.5]
        images = [cv2.resize(chips[j], (w, h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1] for j in steps]
        video = t.as_tensor(np.ascontiguousarray(images), device="cuda").permute(0, 3, 1, 2)[None].float()
        with t.inference_mode():
            self.model(video[:, :1], is_first_step=True,
                       queries=t.as_tensor(query[None], dtype=t.float32, device="cuda"), add_support_grid=True)
            stride = self.model.step
            for k in range(0, max(len(steps)-stride, 1), stride):
                uv, visible = self.model(video[:, k:k+stride*2], add_support_grid=True)
            native = (uv[0].float().cpu().numpy()+.5)/ratios[:, None]-.5+lo[:, None]
            visible = visible[0].cpu().numpy()
        results = {}
        for j, xy, good in zip(steps[1:], native[1:], visible[1:], strict=True):
            if good.sum() < 7:
                continue
            T, support = cv2.estimateAffinePartial2D(points[good], xy[good], method=cv2.RANSAC,
                                                     ransacReprojThreshold=2.6)
            if T is None or support.sum() < max(6, .60*good.sum()):
                continue
            A, c = T[:, :2], T[:, 2]
            factor = np.linalg.norm(A[:, 0])
            if not .60 < factor < 1.55:
                continue
            proposed = (outline @ A.T+c).astype(np.float32)
            results[int(j)] = proposed, xy[good][support.ravel() > 0].astype(np.float32)
        return results


def prompt_for_track(shot, f, j, footprint, xy, peer):
    """Use source-followed points to measure the assembly in this exposure."""
    inside = [p for p in xy if cv2.pointPolygonTest(footprint, tuple(p), True) >= .4]
    if len(inside) < 3:
        return None
    inside = np.asarray(inside)
    centre = footprint.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(inside-centre, axis=1)))
    chosen = [first]
    for _ in range(3):
        gap = np.linalg.norm(inside[:, None]-inside[chosen][None], axis=-1).min(axis=1)
        index = int(np.argmax(gap))
        if gap[index] < 4.:
            break
        chosen.append(index)
    i = shot.a+j
    nav = shot.nav
    size = shot.scale[j, f]
    outside = []
    if nav.direct[i, f].all():
        upper = nav.ankle[i, f]+unit(nav.shin[i, f])*.35*size
        if distance(footprint, upper) > .07*size:
            outside.append(upper)
    if peer is not None:
        q = peer.mean(axis=0)
        if distance(footprint, q) > .10*size:
            outside.append(q)
    p = np.vstack((inside[chosen], *outside))
    bounds = shot.bounds[j, f]
    selected = ((p >= bounds[:2]) & (p < bounds[2:])).all(axis=1)
    tags = np.r_[np.ones(len(chosen), int), np.zeros(len(outside), int)]
    if (tags[selected] == 1).sum() < 2:
        return None
    return (p[selected]-bounds[:2]).tolist(), tags[selected].tolist(), 2


def observed_outline(model, shot, f, j, p, xy, peer):
    prompt = prompt_for_track(shot, f, j, p, xy, peer)
    if prompt is None:
        return None
    model.reset()
    chip = shot.chips[f][j]
    bb = shot.bounds[j, f]
    item = model.step(chip, 0, {f: prompt}).get(f)
    item = None if item is None else measure(*item, bb, shot.scale[j, f])
    if item is None or item[2] or iou(p, item[0]) < .49:
        return None
    outline = item[0]
    q = shot.quality(j, f, outline)
    if q is None or q < .33:
        return None
    if peer is not None and mask_overlap(outline, peer)[1] > .63:
        return None
    return outline, q


def gaps_of_interest(shot, f, focus):
    status = shot.status[:, f]
    a, b = shot.a, shot.b
    direct = shot.nav.direct[a:b, f, 1]
    # Extend through two old partial outlines so a fragment cannot become the
    # only anchor immediately before or after an occlusion.
    want = (status == 0) & direct
    if focus:
        seconds = shot.inputs.times[a:b]
        want &= np.any(np.abs(seconds[:, None]-np.asarray(focus)[None, :]) < .43, axis=1)
    runs = list(segments(want))
    merged = []
    for x, y in runs:
        if (merged and x-merged[-1][1] <= 5 and y-merged[-1][0] <= 18
                and not shot.good[merged[-1][1]+1:x, f].any()):
            merged[-1] = merged[-1][0], y
        else:
            merged.append((x, y))
    for x, y in merged:
        strong = np.flatnonzero(shot.good[:, f])
        before = strong[(strong < x) & (strong >= x-15)]
        after = strong[(strong > y) & (strong <= y+15)]
        yield int(x), int(y), int(before[-1]) if len(before) else None, int(after[0]) if len(after) else None


def from_local_flow(shot, j, f):
    return [x.outline for x in shot.motion_options(f, j) if shot.good[x.anchor, f]]


def fill_gaps(shot, result, points, model, focus):
    log = []
    for f in range(2):
        for x, y, previous, after in gaps_of_interest(shot, f, focus):
            if previous is None and after is None:
                continue
            if (previous is None or after is None) and y-x > 12:
                continue
            if previous is not None and after is not None and after-previous > 46:
                continue
            first = previous if previous is not None else max(0, x-2)
            last = after if after is not None else min(len(shot.raw)-1, y+2)
            traces = {}
            for anchor, target in ((previous, last), (after, first)):
                if anchor is not None:
                    traces[anchor] = points.trace(shot.chips[f], shot.bounds[:, f],
                                                  shot.raw[anchor, f], anchor, target)
            proposals = {}
            for j in range(x, y+1):
                if shot.status[j, f] or not shot.nav.direct[shot.a+j, f, 1]:
                    continue
                candidates = [(anchor, *seq[j]) for anchor, seq in traces.items() if j in seq]
                if not candidates:
                    continue
                if len(candidates) == 2:
                    support = iou(candidates[0][1], candidates[1][1])
                    if support < .56:
                        continue
                else:
                    # With one surviving path, require independent local
                    # pixel motion. This rejects points that followed the
                    # occluding skate or an advertisement.
                    support = max((iou(candidates[0][1], p) for p in from_local_flow(shot, j, f)), default=0.)
                    if support < .60:
                        continue
                peer = result["region"][j, 1-f] if result["status"][j, 1-f] else None
                anchor, outline, xy = min(candidates, key=lambda row:abs(row[0]-j))
                observed = observed_outline(model, shot, f, j, outline, xy, peer)
                if observed is not None:
                    proposals[j] = anchor, observed[0], support
            for j, (anchor, p, support) in proposals.items():
                adjacent = []
                for k in shot.around(j):
                    if j == k:
                        continue
                    if k in proposals:
                        adjacent.append(shot.warp(f, k, j, proposals[k][1]))
                    elif shot.good[k, f]:
                        adjacent.append(shot.warp(f, k, j, shot.raw[k, f]))
                if max((iou(p, q) for q in adjacent if q is not None), default=0.) < .40 and support < .71:
                    continue
                result["region"][j, f], result["status"][j, f] = p, 1
                result["extent"][j, f], result["revision"][j, f] = VIDEO_RECOVERY, RECOVER
                result["tracking_anchor_pts"][j, f] = shot.inputs.pts[shot.a+anchor]
                result["mask_agreement"][j, f] = support
                log.append([int(shot.inputs.pts[shot.a+j]), f])
    return log


def source_prompt(shot, j, f):
    i = shot.a+j
    nav = shot.nav
    if not nav.direct[i, f, 1]:
        return None
    ankle = nav.ankle[i, f]
    shank = unit(nav.shin[i, f])
    scale = shot.scale[j, f]
    distal = ankle-.18*scale*shank
    guide = nav.centre[i, f]
    offset = guide-distal
    centre = distal+offset*min(.62, .31*scale/max(1., np.linalg.norm(offset)))
    if not np.isfinite(centre).all():
        return None
    peer = shot.raw[j, 1-f] if shot.status[j, 1-f] else None
    positives = [centre]
    toe = .5*(nav.guide[i, f, 1]+nav.guide[i, f, 8])
    second = .66*centre+.34*toe
    if np.linalg.norm(second-centre) > 3. and (peer is None or distance(peer, second) > 1.):
        positives.append(second)
    negatives = [ankle+.31*scale*shank]
    if nav.direct[i, 1-f, 1]:
        opposite = nav.ankle[i, 1-f]-.14*shot.scale[j, 1-f]*unit(nav.shin[i, 1-f])
        if np.linalg.norm(opposite-centre) >= .27*scale:
            negatives.append(opposite)
    uv = np.asarray(positives+negatives)
    labels = np.r_[np.ones(len(positives), int), np.zeros(len(negatives), int)]
    bounds = shot.bounds[j, f]
    keep = ((uv >= bounds[:2]) & (uv < bounds[2:])).all(axis=1)
    if not (keep & (labels == 1)).any():
        return None
    return (uv[keep]-bounds[:2]).tolist(), labels[keep].tolist(), 2


def follow_source_prompt(shot, model, f, seed, stop):
    first = source_prompt(shot, seed, f)
    if first is None:
        return {}
    model.reset()
    results = {}
    sign = 1 if stop > seed else -1
    for step, j in enumerate(range(seed, stop+sign, sign)):
        prompt = {f:first} if step == 0 else {}
        masks = model.step(shot.chips[f][j], step, prompt)
        item = measure(*masks[f], shot.bounds[j, f], shot.scale[j, f]) if f in masks else None
        if item is not None and not item[2]:
            results[j] = item[0]
    return results


def discover_gaps(shot, result, model, focus):
    """Start video memories on newly visible cuffs and test both continuations."""
    updates = []
    n = len(shot.raw)
    for f in range(2):
        for x, y, _, _ in gaps_of_interest(shot, f, focus):
            # Only look in the current shot. Start on separated ankles when
            # possible and keep opposite-foot pixels out of the prompt.
            left, right = max(0, x-2), min(n-1, y+2)
            bank = {}
            for lo in range(x, y+1, 7):
                hi = min(y, lo+6)
                steps = [j for j in range(max(left,lo-1), min(right,hi+1)+1)
                         if source_prompt(shot, j, f) is not None]
                if not steps:
                    continue
                a, b = steps[0], steps[-1]
                for seed, stop in ((a, min(right, hi+2)), (b, max(left, lo-2))):
                    memory = follow_source_prompt(shot, model, f, seed, stop)
                    if seed in bank:
                        bank[seed].update(memory)
                    elif memory:
                        bank[seed] = memory
            candidates = {}
            for j in range(x, y+1):
                if result["status"][j, f]:
                    continue
                i = shot.a+j
                scale = shot.scale[j, f]
                own = shot.nav.ankle[i, f]
                peer = result["region"][j, 1-f] if result["status"][j, 1-f] else None
                from_banks = [(seed, memory, memory[j]) for seed, memory in bank.items() if j in memory]
                old_motion = shot.motion_options(f, j)
                for seed, memory, p in from_banks:
                    if distance(p, own) > .34*scale:
                        continue
                    if peer is not None and mask_overlap(p, peer)[1] > .62:
                        continue
                    visible = shot.quality(j, f, p)
                    if visible is None or visible < .36:
                        continue
                    same = max((iou(p, q) for anchor, _, q in from_banks if seed != anchor), default=0.)
                    old_support = max((iou(p, q.outline) for q in old_motion), default=0.)
                    optical = []
                    for k in shot.around(j):
                        if k == j or k not in memory:
                            continue
                        moved = shot.warp(f, k, j, memory[k])
                        if moved is not None:
                            optical.append(iou(p, moved))
                    local = max(optical, default=0.)
                    if not (same >= .68 or (same >= .52 and max(local, old_support) >= .62)
                            or (old_support >= .63 and local >= .48)):
                        continue
                    score = .46*same+.29*local+.15*old_support+.10*visible
                    if j not in candidates or score > candidates[j][0]:
                        candidates[j] = score, p, seed, max(same, local, old_support)
            for j, (_, p, seed, support) in candidates.items():
                adjacent = []
                for k in shot.around(j):
                    if j == k:
                        continue
                    q = candidates[k][1] if k in candidates else result["region"][k, f] if result["status"][k, f] else None
                    if q is not None:
                        moved = shot.warp(f, k, j, q)
                        if moved is not None:
                            adjacent.append(iou(p, moved))
                if max(adjacent, default=0.) < .38 and support < .77:
                    continue
                result["region"][j, f], result["status"][j, f] = p, 1
                result["extent"][j, f], result["revision"][j, f] = VIDEO_RECOVERY, RECOVER
                result["tracking_anchor_pts"][j, f] = shot.inputs.pts[shot.a+seed]
                result["mask_agreement"][j, f] = support
                updates.append([int(shot.inputs.pts[shot.a+j]), f])
    return updates


def reviewed_source_tracks(inputs, nav, model, case):
    a, k, b = (inputs.index[case[field]] for field in ("first_pts", "pts", "last_pts"))
    if not a <= k <= b or inputs.shots[a] != inputs.shots[b]:
        raise ValueError("Reviewed skate tracking must stay inside the same shot")
    seeds = {row["foot"]: distribute(row["outline"], 80, closed=True) for row in case["tracks"]}
    centre = np.vstack(list(seeds.values())).mean(axis=0)
    guide = nav.ankle[a:b+1, list(seeds)].mean(axis=1)
    side = int(np.clip(2.6*nav.size[k, list(seeds)].max(), 200, 340))
    bounds = []
    chips = []
    for j, (_, frame) in enumerate(inputs.frames(a, b+1)):
        location = centre+.79*(guide[j]-guide[k-a])
        x, y = np.clip(np.rint(location-side*.5).astype(int), 0, np.asarray(SIZE)-side)
        bounds.append([x, y, x+side, y+side])
        chips.append(np.ascontiguousarray(frame[y:y+side, x:x+side]))
    bounds = np.asarray(bounds)
    paths = {k:seeds}
    for direction, end in ((-1, a), (1, b)):
        model.reset()
        for step, j in enumerate(range(k, end+direction, direction)):
            chip = chips[j-a]
            bb = bounds[j-a]
            seed = {f:filled(chip.shape[:2], p-bb[:2]) for f,p in seeds.items()} if step == 0 else {}
            result = model.step(chip, step, {}, seed_masks=seed)
            if j == k:
                continue
            paths[j] = {}
            for f, item in result.items():
                p = measure(*item, bb, nav.size[j, f])
                if p is not None and not p[2]:
                    paths[j][f] = p[0]
    return a, k, b, chips, bounds, paths


def apply_reviewed_tracks(inputs, nav, model, result):
    record = read(PROGRAMME/"whole_skate_overlap_review_1112.json")
    if record["source_video_sha256"] != inputs.media_sha or tuple(record["time_base"]) != tuple(inputs.tb):
        raise ValueError("The overlap review uses different source media")
    partial = result.setdefault("partial_occlusion", np.zeros(result["status"].shape, bool))
    revisions = []
    for case in record["cases"]:
        a, k, b, chips, bounds, paths = reviewed_source_tracks(inputs, nav, model, case)
        owner = case.get("merged_owner")
        feet = [row["foot"] for row in case["tracks"]]
        for j in range(a, b+1):
            p = paths[j]
            if not all(f in p for f in feet):
                continue
            if owner is not None:
                previous = result["region"][j, owner]
                hidden = 1-owner
                if not result["status"][j, owner] or mask_overlap(p[0], p[1])[1] > .34:
                    continue
                other = result["region"][j, hidden]
                if result["status"][j, hidden] and diagonal(other) > .57*nav.size[j, hidden] and mask_overlap(previous, other)[1] < .48:
                    continue
                bb, chip = bounds[j-a], chips[j-a]
                combined = filled(chip.shape[:2], p[0]-bb[:2]) | filled(chip.shape[:2], p[1]-bb[:2])
                old_mask = filled(chip.shape[:2], previous-bb[:2])
                retained = np.count_nonzero(old_mask & combined)/max(1,np.count_nonzero(old_mask))
                if j != k and (retained < .72 or mask_overlap(previous, p[hidden])[1] < .65):
                    continue
            supported = []
            for f in feet:
                if result["extent"][j, f] == SOURCE_PROMPT:
                    continue
                if owner is None and result["status"][j, f]:
                    continue
                if owner is not None and f != owner and result["extent"][j, f] == PRIOR_WHOLE:
                    continue
                if owner is None and result["status"][j, 1-f] and mask_overlap(p[f], result["region"][j, 1-f])[1] > .56:
                    continue
                agreement = []
                for neighbour in (j-1, j+1):
                    if neighbour not in paths or f not in paths[neighbour]:
                        continue
                    warped = advance(chips, bounds, neighbour-a, j-a, paths[neighbour][f])
                    if warped is not None:
                        agreement.append(iou(p[f], warped))
                support = max(agreement, default=0.)
                if j != k and (support < .47 or boundary_support(chips[j-a], bounds[j-a], p[f]) < .31):
                    continue
                supported.append((f, support))
            if owner is not None and len(supported) != 2:
                continue
            for f, support in supported:
                visible_part = next((row.get("visible_part", False) for row in case["tracks"] if row["foot"] == f), False)
                present = result["status"][j, f] > 0
                result["region"][j, f], result["status"][j, f] = p[f], 1
                result["extent"][j, f] = 0 if visible_part else SOURCE_PROMPT if j == k else VIDEO_WHOLE if present else VIDEO_RECOVERY
                result["revision"][j, f] = SOURCE if j == k else REPLACE if present else RECOVER
                result["tracking_anchor_pts"][j, f] = inputs.pts[k]
                result["mask_agreement"][j, f] = 1. if j == k else support
                partial[j, f] = visible_part
                revisions.append([int(inputs.pts[j]), f])
    return revisions


def main(args):
    cv2.setNumThreads(1)
    inputs = Inputs()
    history = args.history/"whole_follow_evidence.npz"
    target = args.output/"whole_follow_evidence.npz"
    with np.load(history, allow_pickle=False) as f:
        before = {k:f[k] for k in f.files}
    with np.load(target, allow_pickle=False) as f:
        np.testing.assert_array_equal(f["source_pts"], inputs.pts)
        result = {k:f[k] for k in f.files}
    nav = Navigation(inputs)
    shape = WholeShape(SkateModel(geometry(inputs)[0]))
    model = VideoMasks(WEIGHTS, image_size=512)
    locked = result["status"] > 0
    preserved = result["region"].copy()
    changes = []
    if args.reviewed:
        changes = apply_reviewed_tracks(inputs, nav, model, result)
        for pts, foot in changes:
            locked[inputs.index[pts], foot] = False
        emit("reviewed_skate_occlusion", revised=changes)
    tracks = ImageTracks() if args.features else None
    for a, b in (() if args.reviewed else inputs.ranges()):
        if args.focus and not any(inputs.times[a] <= s <= inputs.times[b-1] for s in args.focus):
            continue
        started = time.perf_counter()
        packet = {k:result[k][a:b] for k in ("region", "extent", "status", "mask_agreement", "tracking_anchor_pts", "revision")}
        source = {k:value.copy() for k,value in packet.items()}
        shot = Shot(inputs, nav, shape, int(a), int(b), source)
        gained = (fill_gaps(shot, packet, tracks, model, args.focus) if args.features
                  else discover_gaps(shot, packet, model, args.focus))
        changes.extend(gained)
        emit("visible_skate_recovery", shot=str(inputs.shots[a]), recovered=len(gained),
             seconds=round(time.perf_counter()-started,2))
    np.testing.assert_array_equal(result["region"][locked], preserved[locked])
    np.testing.assert_array_equal(result["status"] > 0, np.isfinite(result["region"]).all(axis=(2,3)))
    atomic_npz(target, **result)
    write_navigation(inputs, args.navigation, args.output, result)
    audit_path = args.output/"whole_relink_checks.json"
    audit = read(audit_path)
    audit.update(report(inputs, before, result))
    key = ("reviewed_overlap_revisions" if args.reviewed else
           "native_feature_track_recoveries" if args.features else "source_reinitialization_recoveries")
    audit[key] = sorted({tuple(x) for x in audit.get(key, [])+changes})
    original = (before["extent"] == PRIOR_WHOLE) | (before["extent"] == SOURCE_PROMPT)
    same = np.all(before["region"][original] == result["region"][original], axis=(1,2))
    audit["exact_preserved_source_and_whole"] = int(same.sum())
    audit["previously_merged_originals_corrected"] = int((~same).sum())
    atomic_json(audit_path, audit)
    emit("recovery_complete", recovered=len(changes), missing=audit["missing"])


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--navigation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus", nargs="+", type=float)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--features", action="store_true")
    mode.add_argument("--reviewed", action="store_true")
    main(parser.parse_args())
