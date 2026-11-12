"""Reacquire whole skates and check their identity against native video motion."""

from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import time

import cv2
import numpy as np

from evidence import segments
from follow_whole_skates import (
    PRIOR_UNCERTAIN, PRIOR_WHOLE, SOURCE_PROMPT, VIDEO_RECOVERY, VIDEO_WHOLE,
    WholeShape, boundary_support, extra_part, prompt_row, review_frame, write_navigation,
)
from inputs import (
    PROGRAMME, SIZE, WEIGHTS, Inputs, atomic_json, atomic_npz, digest, distribute, emit, geometry, read,
)
from model import SkateModel
from recover import advance
from render_review import render
from track import Navigation, VideoMasks, covered_body_components, covers_component, iou, mask_overlap, measure
from whole_skate_review import filled


ANCHORS = PROGRAMME / "whole_skate_relink_anchors_1112.json"
KEEP, RECOVER, EXTEND, REPLACE, REJECT, SOURCE = range(6)


def source_anchors(inputs):
    record = read(ANCHORS)
    if record["source_video_sha256"] != inputs.media_sha or tuple(record["time_base"]) != tuple(inputs.tb):
        raise ValueError("The reviewed whole skates use a different source clock")
    return {(inputs.index[row["pts"]], row["foot"]): row for row in record["anchors"]}


def distance(polygon, point):
    return max(0., -cv2.pointPolygonTest(polygon, tuple(point), True)) if np.isfinite(point).all() else np.inf


def diagonal(polygon):
    return np.linalg.norm(np.ptp(polygon, axis=-2), axis=-1)


def changes(region):
    length = diagonal(region)
    typical = np.maximum(12., np.sqrt(length[1:]*length[:-1]))
    travel = np.linalg.norm(region[1:].mean(axis=2)-region[:-1].mean(axis=2), axis=-1)/typical
    scale = np.abs(np.log(np.maximum(1., length[1:])/np.maximum(1., length[:-1])))
    return (travel > .43) | (scale > .46)


def search_crops(inputs, nav, a, b, old, good):
    """Keep a stable pixel scale and use only trusted contours to steer the search."""
    n = b-a
    guide = nav.centre[a:b].copy()
    lengths = nav.size[a:b].copy()
    raw = old["region"]
    for f in range(2):
        rows = np.flatnonzero(good[:, f])
        if not len(rows):
            continue
        correction = raw[rows, f].mean(axis=1)-guide[rows, f]
        limit = lengths[rows, f]*.70
        correction *= np.minimum(1., limit/np.maximum(1., np.linalg.norm(correction, axis=1)))[:, None]
        for dim in range(2):
            guide[:, f, dim] += np.interp(np.arange(n), rows, correction[:, dim])
        ratio = diagonal(raw[rows, f])/np.maximum(lengths[rows, f], 1.)
        ratio = np.interp(np.arange(n), rows, np.clip(ratio, .74, 1.45))
        lengths[:, f] *= .32+.68*ratio
    bounds = np.empty((n, 2, 4), np.int32)
    for j in range(n):
        # Camera zoom changes gradually. A detector's one-frame size error
        # should never zoom the tracker into a cuff.
        scale = np.median(lengths[max(0, j-8):j+9], axis=0)
        side = np.clip(np.rint(2.50*scale+45), 156, 505).astype(int)
        start = np.clip(np.rint(guide[j]-side[:, None]/2).astype(int), 0, np.asarray(SIZE)-side[:, None])
        bounds[j] = np.concatenate((start, start+side[:, None]), axis=1)
    chips = [[], []]
    for j, (pts, frame) in enumerate(inputs.frames(a, b)):
        if pts != inputs.pts[a+j]:
            raise ValueError("Skate tracking did not decode the registered exposure")
        for f, (x, y, r, bottom) in enumerate(bounds[j]):
            chips[f].append(np.ascontiguousarray(frame[y:bottom, x:r]))
    return chips, bounds, lengths


@dataclass
class Candidate:
    outline: np.ndarray
    anchor: int
    score: float
    source: bool = False


@dataclass
class Decision:
    candidate: Candidate | None
    extent: int
    kind: int
    support: float
    score: float


def isolated_seeds(raw, status, extent, nav, a):
    """Locate reliable starting observations while screening one-frame fragments."""
    present = status > 0
    locked = (extent == PRIOR_WHOLE) | (extent == SOURCE_PROMPT)
    n = len(raw)
    length = diagonal(raw)
    expected = nav.size[a:a+n]
    jump = changes(raw)
    discontinuity = np.zeros_like(present)
    discontinuity[1:] |= jump
    discontinuity[:-1] |= jump
    fragment = present & ((length < .49*expected) | (length > 1.9*expected))
    duplicate = np.zeros(n, bool)
    for j in np.flatnonzero(present.all(axis=1)):
        p, q = raw[j]
        if np.all(p.max(axis=0) >= q.min(axis=0)) and np.all(q.max(axis=0) >= p.min(axis=0)):
            duplicate[j] = mask_overlap(p, q)[1] > .66
    reliable = (status == 1) & (extent > 0) & ~discontinuity & ~fragment & ~duplicate[:, None]
    reliable |= locked
    return locked, reliable, discontinuity, fragment, duplicate


def video_scan(model, foot, chips, bounds, scale, old, good, requested, forward):
    """Propagate known whole outlines through the actual source exposures."""
    result = {}
    n = len(chips)
    # No cuts are crossed. Each pass starts on a different observation.
    anchor = -1
    count = 0
    remaining = np.cumsum(requested[::-1])[::-1] if forward else np.cumsum(requested)
    order = range(n) if forward else range(n-1, -1, -1)
    model.reset()
    failed = 0
    for j in order:
        if not remaining[j]:
            break
        close = requested[max(0, j-4):j+5].any()
        refresh = good[j] and (anchor < 0 or abs(j-anchor) >= 22 or failed >= 3)
        if not close:
            if anchor >= 0:
                model.reset()
                anchor, count = -1, 0
            continue
        if not refresh and (anchor < 0 or j % 2 != int(not forward)):
            continue
        if refresh:
            if anchor >= 0 and (failed >= 3 or abs(j-anchor) > 56):
                model.reset()
                count = 0
            anchor, failed = j, 0
        seed = {foot: filled(chips[j].shape[:2], old[j]-bounds[j, :2])} if refresh else {}
        masks = model.step(chips[j], count, {}, seed_masks=seed)
        count += 1
        item = measure(*masks[foot], bounds[j], scale[j]) if foot in masks else None
        if item is None:
            failed += 1
            continue
        outline, score, clipped = item
        failed = 0
        if not clipped or refresh:
            result[j] = Candidate(outline, anchor, score)
    return result


def video_from_source(model, foot, chips, bounds, scale, seed, row):
    """Track the source-reviewed assembly through both sides of this exposure."""
    result = {}
    trace = row.get("outline")
    initial = distribute(trace, 80, closed=True) if trace else None
    for direction in (-1, 1):
        model.reset()
        final = int(np.clip(seed+direction*26, 0, len(chips)-1))
        for t, j in enumerate(range(seed, final+direction, direction)):
            prompt, mask = {}, {}
            if not t:
                if initial is not None:
                    if ((initial < bounds[j, :2]) | (initial >= bounds[j, 2:])).any():
                        raise ValueError("The search crop clipped a reviewed whole skate")
                    mask[foot] = filled(chips[j].shape[:2], initial-bounds[j, :2])
                else:
                    prompt[foot] = prompt_row(row, bounds[j])
            masks = model.step(chips[j], t, prompt, seed_masks=mask)
            item = measure(*masks[foot], bounds[j], scale[j]) if foot in masks else None
            if item is None:
                continue
            outline, score, clipped = item
            if not t and initial is not None:
                outline = initial
            if not t and initial is None and any(distance(outline, p) > 2.5 for p in row["inside"]):
                continue
            if not clipped:
                result[j] = Candidate(outline, seed, score, True)
    return result


class Shot:
    def __init__(self, inputs, nav, shape, a, b, old):
        self.inputs, self.nav, self.shape, self.a, self.b, self.old = inputs, nav, shape, a, b, old
        self.raw, self.status, self.extent = old["region"], old["status"], old["extent"]
        self.locked, self.good, self.jump, self.fragment, self.duplicate = isolated_seeds(
            self.raw, self.status, self.extent, nav, a)
        self.chips, self.bounds, self.scale = search_crops(inputs, nav, a, b, old, self.good)
        self.limbs, self.limb_span = covered_body_components(nav, np.arange(a, b))
        self.cache = {}

    def warp(self, foot, start, end, outline):
        key = (foot, start, end, id(outline))
        if key not in self.cache:
            self.cache[key] = (outline, advance(self.chips[foot], self.bounds[:, foot], start, end, outline))
        return self.cache[key][1]

    def around(self, j):
        return range(max(0, j-1), min(len(self.raw), j+2))

    def motion_options(self, f, j):
        result = []
        for k in self.around(j):
            if k == j or not self.status[k, f] or self.fragment[k, f]:
                continue
            p = self.warp(f, k, j, self.raw[k, f])
            if p is not None:
                result.append(Candidate(p, k, 1.1))
        return result

    def quality(self, j, f, p, source=False):
        size = self.scale[j, f]
        i = self.a+j
        length = float(diagonal(p))
        if not .47*size < length < 1.92*size:
            return None
        if self.shape.error(p, f) > .239:
            return None
        body = covers_component(p, self.limbs[j], self.limb_span[j], max(11., .27*size))
        own = distance(p, self.nav.ankle[i, f])
        other = distance(p, self.nav.ankle[i, 1-f])
        if body and own > max(8., .20*size):
            return None
        direct = self.nav.direct[i, :, 1]
        if direct[f] and own > size*(.85 if source else .67):
            return None
        if direct.all() and not source and own > .33*size and other+.22*size < own:
            return None
        return boundary_support(self.chips[f][j], self.bounds[j, f], p)

    def existing(self, j, f, motions):
        present = bool(self.status[j, f])
        if not present:
            return False, False
        p = self.raw[j, f]
        i = self.a+j
        size = self.scale[j, f]
        own = distance(p, self.nav.ankle[i, f])
        other = distance(p, self.nav.ankle[i, 1-f])
        directly_wrong_foot = self.nav.direct[i, :, 1].all() and own > .46*size and other+.32*size < own
        covers_limb = own > .25*size and covers_component(p, self.limbs[j], self.limb_span[j], max(10., .27*size))
        bad = covers_limb or directly_wrong_foot or bool(self.fragment[j, f])
        isolated = (len(motions) == 2 and iou(motions[0].outline, motions[1].outline) > .51
                    and max(iou(p, x.outline) for x in motions) < .38)
        return bad, isolated

    def options(self, j, f, banks):
        p = self.raw[j, f]
        present = bool(self.status[j, f])
        mode = int(self.extent[j, f])
        prior = Candidate(p, -1, 0.) if present else None
        if self.locked[j, f]:
            return [Decision(prior, mode, KEEP, 1., 6.)]
        requested = (not present or mode == PRIOR_UNCERTAIN or self.jump[j, f]
                     or self.duplicate[j] or self.fragment[j, f] or self.status[j, f] != 1)
        hits = [bank[j] for bank in banks if j in bank]
        if not requested and not any(x.source and x.anchor == j for x in hits):
            return [Decision(prior, mode, KEEP, 1., 2.15)]
        motions = self.motion_options(f, j)
        bad, isolated = self.existing(j, f, motions)
        out = [Decision(prior, mode, KEEP, 0., (.43 if bad or isolated else 1.40) if present else .20)]
        if present and bad and (isolated or self.duplicate[j] or self.jump[j, f]):
            out.append(Decision(None, PRIOR_UNCERTAIN, REJECT, 0., .46))
        # An optical-only candidate must come from both adjacent exposures.
        # A single flow track cannot assert an unobserved piece of steel.
        if len(motions) == 2 and iou(motions[0].outline, motions[1].outline) >= .59:
            hits += motions
        for bank in banks:
            if j in bank:
                continue
            for k in self.around(j):
                if k != j and k in bank:
                    m = bank[k]
                    warped = self.warp(f, k, j, m.outline)
                    if warped is not None:
                        hits.append(Candidate(warped, m.anchor, m.score, m.source))
        for h in hits:
            q = h.outline
            if h.source and h.anchor == j and any(x is h for bank in banks for x in (bank.get(j),)):
                out.append(Decision(h, SOURCE_PROMPT, SOURCE, 1., 5.))
                continue
            native = self.quality(j, f, q, h.source)
            if native is None or native < (.31 if h.source else .35):
                continue
            together = max((iou(q, other.outline) for other in hits if h.anchor != other.anchor), default=0.)
            local = max((iou(q, item.outline) for item in motions), default=0.)
            if h.source or together >= .53:
                for bank in banks:
                    for k in self.around(j):
                        if k == j or k not in bank:
                            continue
                        warp = self.warp(f, k, j, bank[k].outline)
                        if warp is not None:
                            local = max(local, iou(q, warp))
            if not (together >= .62 or (together >= .46 and local >= .60)
                    or (h.source and local >= .56 and native >= .37)):
                continue
            support = max(together, local)
            if not present:
                out.append(Decision(h, VIDEO_RECOVERY, RECOVER, support, 1.30+native+.52*support))
                continue
            overlap = iou(p, q)
            if (bad or isolated or self.status[j, f] == 3 or self.duplicate[j]) and overlap < .83 and support >= .63:
                quality_gain = diagonal(q) > 1.15*diagonal(p) or overlap < .53 or self.duplicate[j]
                if quality_gain:
                    out.append(Decision(h, VIDEO_WHOLE, REPLACE, support, 1.32+native+.46*support))
                    continue
            if mode != PRIOR_UNCERTAIN or overlap < .46:
                continue
            chip, bounds = self.chips[f][j], self.bounds[j, f]
            if boundary_support(chip, bounds, q, p) < .36:
                continue
            i = self.a+j
            shank = self.nav.joints[i, f] if self.nav.direct[i, f].all() else None
            extension = extra_part(p, q, chip, bounds, shank, self.scale[j, f])
            if extension is None or self.shape.error(extension, f) > .25:
                continue
            out.append(Decision(Candidate(extension, h.anchor, h.score, h.source), VIDEO_WHOLE,
                                EXTEND, support, 1.07+native+.48*support))
        baseline = [x for x in out if x.kind in (KEEP, REJECT)]
        alternatives = [x for x in out if x.kind not in (KEEP, REJECT)]
        return baseline + sorted(alternatives, key=lambda x: x.score, reverse=True)[:4]

    def repair(self, model, prompts):
        banks = []
        for f in range(2):
            request = (~self.locked[:, f] & ((self.extent[:, f] == PRIOR_UNCERTAIN) | self.jump[:, f]
                       | self.duplicate | self.fragment[:, f] | (self.status[:, f] != 1)))
            per_foot = [video_scan(model, f, self.chips[f], self.bounds[:, f], self.scale[:, f],
                                   self.raw[:, f], self.good[:, f], request, forward) for forward in (True, False)]
            for (i, foot), row in prompts.items():
                if f == foot and self.a <= i < self.b:
                    per_foot.append(video_from_source(model, f, self.chips[f], self.bounds[:, f],
                                                     self.scale[:, f], i-self.a, row))
            banks.append(per_foot)
        result = {k: v.copy() for k, v in self.old.items()}
        result["revision"] = np.zeros(self.status.shape, np.uint8)
        for j in range(len(self.raw)):
            options = [self.options(j, f, banks[f]) for f in range(2)]
            best = None
            for pair in product(*options):
                l, r = (x.candidate for x in pair)
                shared = 0.
                if l is not None and r is not None:
                    shared = mask_overlap(l.outline, r.outline)[1]
                old_pair = all(x.kind == KEEP for x in pair)
                if shared > .68 and not old_pair:
                    continue
                score = sum(x.score for x in pair) - (.85 if shared > .68 else 0.)
                if best is None or score > best[0]:
                    best = score, pair
            for f, choice in enumerate(best[1]):
                if choice.kind == KEEP:
                    continue
                h = choice.candidate
                result["revision"][j, f] = choice.kind
                result["region"][j, f] = np.nan if h is None else h.outline
                result["status"][j, f] = int(h is not None)
                result["extent"][j, f] = choice.extent
                result["mask_agreement"][j, f] = choice.support if h is not None else np.nan
                result["tracking_anchor_pts"][j, f] = -1 if h is None else self.inputs.pts[self.a+h.anchor]
        self.check_transitions(result)
        return result

    def check_transitions(self, result):
        # A substituted outline may not jump away from two mutually agreeing
        # native-flow predictions. Preserve real speed and fast rotations.
        polygons, status = result["region"].copy(), result["status"].copy()
        for j, f in zip(*np.where((result["revision"] != KEEP) & (status > 0))):
            if result["revision"][j, f] == SOURCE:
                continue
            supports = []
            for k in self.around(j):
                if k == j or not status[k, f]:
                    continue
                moved = self.warp(f, k, j, polygons[k, f])
                if moved is not None:
                    supports.append(moved)
            if len(supports) != 2 or iou(*supports) < .52:
                continue
            if max(iou(polygons[j, f], s) for s in supports) >= .41:
                continue
            prior = self.raw[j, f]
            if self.status[j, f] and max(iou(prior, s) for s in supports) >= .46:
                for key in ("region", "status", "extent", "tracking_anchor_pts", "mask_agreement"):
                    result[key][j, f] = self.old[key][j, f]
                result["revision"][j, f] = KEEP
            else:
                result["region"][j, f], result["status"][j, f], result["extent"][j, f] = np.nan, 0, 0
                result["tracking_anchor_pts"][j, f], result["mask_agreement"][j, f] = -1, np.nan
                result["revision"][j, f] = REJECT if self.status[j, f] else KEEP


def report(inputs, before, after):
    counts = np.bincount(after["revision"].ravel(), minlength=6)
    old = before["status"] > 0
    visible = after["status"] > 0
    old_large = changes(before["region"])
    new_large = changes(after["region"])
    inside = inputs.shots[1:] == inputs.shots[:-1]
    return dict(source_frames=len(inputs.pts), missing=int((~visible).sum()),
                extent_uncertain=int((visible & (after["extent"] == PRIOR_UNCERTAIN)).sum()),
                previously_missing_recovered=int((~old & visible).sum()),
                previous_outline_rejected=int((old & ~visible).sum()),
                revisions=dict(zip(("preserved", "recovered", "extended", "replaced", "rejected", "source"),
                                   map(int, counts))),
                large_shape_or_position_steps_before=int(old_large[inside].sum()),
                large_shape_or_position_steps_after=int(new_large[inside].sum()))


def main(args):
    cv2.setNumThreads(1)
    inputs = Inputs()
    output = args.output
    path = output/"whole_follow_evidence.npz"
    if args.video:
        with np.load(path, allow_pickle=False) as f:
            np.testing.assert_array_equal(f["source_pts"], inputs.pts)
            np.testing.assert_array_equal(f["source_video_sha256"], inputs.media_sha)
            data = {key:f[key] for key in ("region", "status", "extent", "partial_occlusion") if key in f}
        frames = (review_frame(inputs, src, data, i)
                  for i, (_, src) in enumerate(inputs.frames(0, len(inputs.pts))))
        render(inputs, output, "whole_skate_follow_review.mp4", frames, (2400, 1080), refresh=args.refresh)
        return
    history = args.history/"whole_follow_evidence.npz"
    with np.load(history, allow_pickle=False) as f:
        np.testing.assert_array_equal(f["source_pts"], inputs.pts)
        np.testing.assert_array_equal(f["source_video_sha256"], inputs.media_sha)
        before = {key:f[key] for key in f.files}
    identity = np.array(":".join((digest(history), digest(ANCHORS), digest(Path(__file__)))))
    prompts = source_anchors(inputs)
    nav = Navigation(inputs)
    shape = WholeShape(SkateModel(geometry(inputs)[0]))
    model = VideoMasks(WEIGHTS, image_size=512)
    (output/"shots").mkdir(parents=True, exist_ok=True)
    priority = ("s04", "s05", "s07", "s08", "s09", "s01", "s02", "s03", "s06")
    ranges = sorted(inputs.ranges(), key=lambda row:priority.index(inputs.shots[row[0]]))
    for a, b in ranges:
        name = str(inputs.shots[a])
        if args.shots and name not in args.shots:
            continue
        target = output/"shots"/f"{name}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as f:
                if np.array_equal(f["run_identity"], identity):
                    continue
        t = time.perf_counter()
        emit("shot_start", shot=name, pts=int(inputs.pts[a]), frames=int(b-a))
        packet = {key:before[key][a:b] for key in ("region", "status", "extent", "tracking_anchor_pts", "mask_agreement")}
        shot = Shot(inputs, nav, shape, int(a), int(b), packet)
        fixed = shot.repair(model, prompts)
        np.testing.assert_array_equal(packet["region"][shot.locked], fixed["region"][shot.locked])
        atomic_npz(target, source_pts=inputs.pts[a:b], run_identity=identity, **fixed)
        emit("shot_complete", shot=name, revisions=np.bincount(fixed["revision"].ravel(), minlength=6).tolist(),
             remaining=int((fixed["status"] == 0).sum()), seconds=round(time.perf_counter()-t, 2))
    if args.shots:
        return
    result = {key:val.copy() for key, val in before.items()}
    result["revision"] = np.zeros_like(before["status"])
    for a, b in inputs.ranges():
        with np.load(output/"shots"/f"{inputs.shots[a]}.npz", allow_pickle=False) as f:
            np.testing.assert_array_equal(f["source_pts"], inputs.pts[a:b])
            np.testing.assert_array_equal(f["run_identity"], identity)
            for key in ("region", "status", "extent", "tracking_anchor_pts", "mask_agreement", "revision"):
                result[key][a:b] = f[key]
    np.testing.assert_array_equal(result["status"] > 0, np.isfinite(result["region"]).all(axis=(2, 3)))
    locked = (before["extent"] == PRIOR_WHOLE) | (before["extent"] == SOURCE_PROMPT)
    np.testing.assert_array_equal(before["region"][locked], result["region"][locked])
    atomic_npz(path, **result)
    write_navigation(inputs, args.navigation, output, result)
    summary = report(inputs, before, result)
    summary["exact_preserved_source_and_whole"] = int(locked.sum())
    atomic_json(output/"whole_relink_checks.json", summary)
    emit("completed", **summary)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--navigation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", nargs="+")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if not args.video and not (args.history and args.navigation):
        parser.error("--history and --navigation are required for repair")
    main(args)
