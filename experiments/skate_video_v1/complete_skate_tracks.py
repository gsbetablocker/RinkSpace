"""Repair breaks in the whole-skate tracks using native motion from both sides."""

from argparse import ArgumentParser
from pathlib import Path
import time

import cv2
import numpy as np

from evidence import segments
from follow_whole_skates import (PRIOR_UNCERTAIN, PRIOR_WHOLE, SOURCE_PROMPT, VIDEO_RECOVERY, VIDEO_WHOLE,
                                 WholeShape, boundary_support, crops, extra_part, read_anchors, write_navigation)
from inputs import Inputs, WEIGHTS, atomic_json, atomic_npz, emit, geometry, read
from model import SkateModel
from recover import advance
from track import Navigation, VideoMasks, covered_body_components, covers_component, iou, mask_overlap, measure
from whole_skate_review import filled


# Source time offsets around the two views specifically selected for review.
FOCUS = {1143296: (-17, 20), 1518336: (-9, 18)}
REVIEWED_PRIOR = [(601856, 0), (1521664, 0), (1833984, 1), (2005504, 1)]


def actual_motion(chips, bounds, outline, start, stop):
    direction = 1 if stop > start else -1
    result = {}
    for j in range(start+direction, stop+direction, direction):
        outline = advance(chips, bounds, j-direction, j, outline)
        if outline is None:
            break
        result[j] = outline
    return result


def from_video(model, foot, chips, bounds, size, seed, start, stop, stride=1):
    model.reset()
    direction = 1 if stop > start else -1
    result = {}
    for t, j in enumerate(range(start, stop+direction, direction*stride)):
        prompt = {foot: filled(chips[j].shape[:2], seed-bounds[j, :2])} if not t else None
        found = model.step(chips[j], t, {}, seed_masks=prompt).get(foot)
        if found is not None:
            measured = measure(*found, bounds[j], max(18., size[j]))
            if measured is not None:
                result[j] = measured[0]
    return result


def motion_at(curves, j, chips, bounds):
    if j in curves:
        return curves[j]
    for k in (j-1, j+1):
        if k in curves:
            p = advance(chips, bounds, k, j, curves[k])
            if p is not None:
                return p
    return None


def apply_candidate(data, foot, k, p, chip, bounds, scale, inputs, nav, shape, anchor, agreement, *, reviewed=False):
    if data["extent"][k, foot] != PRIOR_UNCERTAIN:
        return False
    old = data["region"][k, foot] if data["status"][k, foot] else None
    peer = data["region"][k, 1-foot] if data["status"][k, 1-foot] else None
    if shape.error(p, foot) > .25:
        return False
    if peer is not None and mask_overlap(p, peer)[1] > .57 and (old is None or mask_overlap(old, peer)[1] < .48):
        return False
    if old is not None:
        if boundary_support(chip, bounds, p, old) < (.25 if reviewed else .34):
            return False
        shank = nav.joints[k, foot] if nav.direct[k, foot].all() else None
        p = extra_part(old, p, chip, bounds, shank, scale)
        if p is None:
            return False
    else:
        if boundary_support(chip, bounds, p) < .35:
            return False
        if not reviewed:
            ankle = nav.ankle[k, foot]
            if np.isfinite(ankle).all() and cv2.pointPolygonTest(p, tuple(ankle), True) < -.50*scale:
                return False
    data["region"][k, foot], data["status"][k, foot] = p, 1
    data["extent"][k, foot] = VIDEO_WHOLE if old is not None else VIDEO_RECOVERY
    data["tracking_anchor_pts"][k, foot] = inputs.pts[anchor]
    data["mask_agreement"][k, foot] = agreement
    return True


def complete_shot(inputs, nav, shape, model, a, b, data, anchors, *, long):
    t0 = time.perf_counter()
    chips, bounds, size = crops(inputs, data["region"][a:b], a, b, anchors)
    touched = 0
    for k, f in (() if long else anchors):
        if not a <= k < b or data["extent"][k, f] == 0:
            continue
        left, right = FOCUS[int(inputs.pts[k])]
        j = k-a
        interval = (max(0, j+left), min(b-a-1, j+right))
        source = data["region"][k, f]
        fixed = from_video(model, f, chips[f], bounds[:, f], size[:, f], source, j, interval[0])
        fixed.update(from_video(model, f, chips[f], bounds[:, f], size[:, f], source, j, interval[1]))
        for k1, p in fixed.items():
            actual = a+k1
            if data["extent"][actual, f] != PRIOR_UNCERTAIN:
                continue
            # Confirm that the output moves with its neighbouring object in
            # source pixels, allowing a true view change through a turn.
            follow = []
            for q in (k1-1, k1+1):
                if q in fixed:
                    advanced = advance(chips[f], bounds[:, f], q, k1, fixed[q])
                    if advanced is not None:
                        follow.append(iou(p, advanced))
            agreement = max(follow, default=0.)
            if agreement < .49:
                continue
            touched += apply_candidate(data, f, actual, p, chips[f][k1], bounds[k1, f], size[k1, f], inputs, nav, shape, k,
                                       agreement, reviewed=True)
    connected, from_both = 0, 0
    limbs, limb_span = covered_body_components(nav, np.arange(a, b))
    for f in range(2):
        extent = data["extent"][a:b, f]
        for x, y in (() if long else segments(extent == PRIOR_UNCERTAIN)):
            if x == 0 or y == len(extent)-1 or y-x > 11:
                continue
            previous = data["region"][a+x-1, f]
            following = data["region"][a+y+1, f]
            ltr = actual_motion(chips[f], bounds[:, f], previous, x-1, y)
            rtl = actual_motion(chips[f], bounds[:, f], following, y+1, x)
            for j in sorted(ltr.keys() & rtl.keys()):
                agreement = iou(ltr[j], rtl[j])
                if agreement < .63:
                    continue
                q = x-1 if j-x <= y-j else y+1
                p = ltr[j] if q == x-1 else rtl[j]
                connected += apply_candidate(data, f, a+j, p, chips[f][j], bounds[j, f], size[j, f], inputs, nav, shape, a+q,
                                             agreement)
        for x, y in segments(extent == PRIOR_UNCERTAIN):
            if x == 0 or y == len(extent)-1 or not (21 <= y-x <= 70 if long else y-x <= 20):
                continue
            # Opposite memories start from already localized whole skates
            # and follow the same object through each native exposure.
            stride = 2 if long else 1
            left = from_video(model, f, chips[f], bounds[:, f], size[:, f], data["region"][a+x-1, f], x-1, y, stride)
            right = from_video(model, f, chips[f], bounds[:, f], size[:, f], data["region"][a+y+1, f], y+1, x, stride)
            for j in range(x, y+1):
                p0 = motion_at(left, j, chips[f], bounds[:, f])
                p1 = motion_at(right, j, chips[f], bounds[:, f])
                if p0 is None or p1 is None:
                    continue
                agreement = iou(p0, p1)
                if agreement < .65:
                    continue
                q = x-1 if j-x <= y-j else y+1
                p = p0 if q == x-1 else p1
                if covers_component(p, limbs[j], limb_span[j], max(12., .33*size[j, f])):
                    continue
                from_both += apply_candidate(data, f, a+j, p, chips[f][j], bounds[j, f], size[j, f],
                                             inputs, nav, shape, a+q, agreement)
    emit("whole_gap_review", shot=str(inputs.shots[a]), native_video_additions=int(touched),
         two_way_motion_additions=int(connected), two_memory_additions=int(from_both), seconds=round(time.perf_counter()-t0, 1))
    return touched, connected, from_both


def reject_body_regions(inputs, nav, data):
    """Clear inherited partial masks covering a measured proximal limb."""
    limbs, spans = covered_body_components(nav, np.arange(len(inputs.pts)))
    removed = []
    check = (data["extent"] == PRIOR_UNCERTAIN) & (data["status"] > 0)
    for i, foot in zip(*np.where(check & nav.direct[:, :, 1])):
        p = data["region"][i, foot]
        size = float(np.linalg.norm(np.ptp(p, axis=0)))
        ankle = nav.ankle[i, nav.direct[i, :, 1]]
        if any(cv2.pointPolygonTest(p, tuple(point), True) >= -max(12., .24*size) for point in ankle):
            continue
        if not covers_component(p, limbs[i], spans[i], max(12., .29*size)):
            continue
        removed.append([int(inputs.pts[i]), int(foot)])
        data["region"][i, foot] = np.nan
        data["status"][i, foot] = 0
    return removed


def review_lifted_span(inputs, nav, shape, tracker, data):
    """Follow the lifted left skate across the hand occlusion and bright steel."""
    def index(seconds):
        return inputs.index[round(seconds*inputs.tb[1]/inputs.tb[0])]

    first, last = index(118.22), index(119.02)
    raw = data["region"][first:last+1].copy()
    raw[index(118.30)-first, 0] = np.nan  # Source review found the hand here.
    chips, bounds, size = crops(inputs, raw, first, last+1, {})
    for k, (_, frame) in enumerate(inputs.frames(first, index(118.40)+1), first):
        if not nav.direct[k, 0, 1]:
            continue
        j = k-first
        lo, hi = np.split(bounds[j, 0], 2)
        width = int(hi[0]-lo[0])
        centre = .32*(lo+hi)*.5+.68*nav.ankle[k, 0]
        x, y = np.clip(np.rint(centre-width*.5).astype(int), 0, np.array([1920, 1080])-width)
        bounds[j, 0] = [x, y, x+width, y+width]
        chips[0][j] = np.ascontiguousarray(frame[y:y+width, x:x+width])

    def follow(start, stop):
        a, b = index(start)-first, index(stop)-first
        return from_video(tracker, 0, chips[0], bounds[:, 0], size[:, 0], raw[a, 0], a, b)

    windows = (
        (118.30, 118.46, 118.50, follow(118.50, 118.22), follow(118.62, 118.22)),
        (118.64, 118.72, 118.62, follow(118.62, 118.82), follow(118.78, 118.58)),
    )
    revisions = []
    for start, stop, anchor, primary, check in windows:
        for k in range(index(start), index(stop)+1):
            extent = data["extent"][k, 0]
            if extent in (PRIOR_WHOLE, SOURCE_PROMPT):
                continue
            j = k-first
            p, q = primary.get(j), check.get(j)
            if p is None or q is None:
                continue
            agreement = iou(p, q)
            chip, bb, scale = chips[0][j], bounds[j, 0], size[j, 0]
            if agreement < .84 or shape.error(p, 0) > .22:
                continue
            old = data["region"][k, 0]
            present = bool(data["status"][k, 0])
            extent_update = extent
            if k <= index(118.38) and (not present or iou(old, p) < .55):
                if boundary_support(chip, bb, p) < .48:
                    continue
                # The blurred object is localized, but the full extent is unverified.
                extent_update = PRIOR_UNCERTAIN
            else:
                if not present or boundary_support(chip, bb, p, old) < .63:
                    continue
                shank = nav.joints[k, 0] if nav.direct[k, 0].all() else None
                p = extra_part(old, p, chip, bb, shank, scale)
                if p is None or shape.error(p, 0) > .24:
                    continue
                if k >= index(118.40) and extent != VIDEO_RECOVERY:
                    extent_update = VIDEO_WHOLE
            data["region"][k, 0], data["status"][k, 0], data["extent"][k, 0] = p, 1, extent_update
            data["tracking_anchor_pts"][k, 0] = inputs.pts[index(anchor)]
            data["mask_agreement"][k, 0] = agreement
            revisions.append(int(inputs.pts[k]))
    return revisions


def main(prior, output, *, long, source_review):
    cv2.setNumThreads(1)
    inputs = Inputs()
    target = output/"whole_follow_evidence.npz"
    with np.load(target, allow_pickle=False) as f:
        np.testing.assert_array_equal(f["source_pts"], inputs.pts)
        data = {key: f[key] for key in f.files}
    with np.load(prior/"whole_skate_evidence.npz", allow_pickle=False) as f:
        before, known = f["region"], f["extent"]
    initial = data["region"].copy()
    protected = data["extent"] > 0
    for stamp, foot in REVIEWED_PRIOR:
        k = inputs.index[stamp]
        if np.isfinite(before[k, foot]).all():
            data["region"][k, foot] = before[k, foot]
            data["extent"][k, foot] = PRIOR_WHOLE
            initial[k, foot] = before[k, foot]
            protected[k, foot] = True
    prompts = read_anchors(inputs)
    nav = Navigation(inputs)
    shape = WholeShape(SkateModel(geometry(inputs)[0]))
    tracker = VideoMasks(WEIGHTS, image_size=512)
    repaired = np.zeros(3, np.int64)
    if source_review:
        protected = (data["extent"] == PRIOR_WHOLE) | (data["extent"] == SOURCE_PROMPT)
        revisions = review_lifted_span(inputs, nav, shape, tracker, data)
        emit("reviewed_lifted_skate", revised_source_pts=revisions)
    else:
        for a, b in inputs.ranges():
            repaired += complete_shot(inputs, nav, shape, tracker, int(a), int(b), data, prompts, long=long)
    removed = reject_body_regions(inputs, nav, data) if long else []
    np.testing.assert_array_equal(data["region"][protected], initial[protected])
    np.testing.assert_array_equal(data["region"][known > 0], before[known > 0])
    atomic_npz(target, **data)
    write_navigation(inputs, prior, output, data)
    report = read(output/"whole_follow_checks.json")
    report.update(tracked=int((data["status"] > 0).sum()), missing=int((data["status"] == 0).sum()),
                  extent_uncertain=int(((data["status"] > 0) & (data["extent"] == PRIOR_UNCERTAIN)).sum()),
                  extended_by_video=int((data["extent"] == VIDEO_WHOLE).sum()),
                  recovered_by_video=int((data["extent"] == VIDEO_RECOVERY).sum()))
    if source_review:
        report["lifted_left_source_review_pts"] = sorted(set(report.get("lifted_left_source_review_pts", [])+revisions))
    elif long:
        report.update(repaired_long_video_spans=int(repaired[2]), proximal_body_rejections=removed)
    else:
        report.update(repaired_dense_native=int(repaired[0]), repaired_two_way_motion=int(repaired[1]),
                      repaired_two_video_memories=int(repaired[2]))
    atomic_json(output/"whole_follow_checks.json", report)
    emit("completed", **{key:report[key] for key in ("tracked", "missing", "extent_uncertain", "extended_by_video", "recovered_by_video")},
         body_rejected=len(report.get("proximal_body_rejections", [])))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--long-gaps", action="store_true")
    mode.add_argument("--source-review", action="store_true")
    args = parser.parse_args()
    main(args.prior, args.output, long=args.long_gaps, source_review=args.source_review)
