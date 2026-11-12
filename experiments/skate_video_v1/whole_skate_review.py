"""Track and review a single source outline for each complete skate."""

from argparse import ArgumentParser
from pathlib import Path
import time

import cv2
import numpy as np

from inputs import Inputs, atomic_json, atomic_npz, digest, distribute, emit, read
from render_review import FOOT_COLOUR, lines, render, stamp, text, zoom
from track import mask_overlap


def source_packet(inputs, prior, cues):
    path = prior / "image_evidence.npz"
    with np.load(path, allow_pickle=False) as f:
        for key, value in (("source_pts", inputs.pts), ("shot_ids", inputs.shots), ("time_base", inputs.tb),
                           ("source_video_sha256", inputs.media_sha), ("camera_sha256", inputs.camera_sha)):
            np.testing.assert_array_equal(f[key], value)
        raw, status, line = f["boot"], f["boot_status"], f["steel"]
    signal = np.isfinite(line).all(axis=(2, 3))
    if cues is not None:
        source_digest = digest(path)
        for start, stop in inputs.ranges():
            with np.load(cues / "shots" / f"{inputs.shots[start]}.npz", allow_pickle=False) as f:
                np.testing.assert_array_equal(f["source_pts"], inputs.pts[start:stop])
                np.testing.assert_array_equal(f["tracked_region"], raw[start:stop])
                identity = str(f["run_identity"]).split(":")
                if identity[:3] != [inputs.media_sha, inputs.camera_sha, inputs.parent_sha] or identity[4] != source_digest:
                    raise ValueError("Different source region or camera in the cue data")
                present = ((f["component_motion_confirmed"] | (f["component_score"] >= 34.))
                           & ~signal[start:stop])
                line[start:stop][present] = f["component_rail"][present]
                signal[start:stop] |= present
    return raw, status, line, signal


def filled(shape, contour):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(contour*16.).astype(np.int32)], 1, shift=4)
    return mask


def extend_region(frame, raw, cue):
    """Retain the observed boundary and add only connected native foreground."""
    scale = float(np.linalg.norm(np.ptp(raw, axis=0)))
    if not 18. <= scale <= 360.:
        return raw, 0
    # An enclosed cue supports the original extent. Keep those pixels and
    # the precise original contour, even when another template can fit it.
    signed = np.array([cv2.pointPolygonTest(raw, tuple(p), True) for p in cue])
    if (signed >= -max(2., .018*scale)).mean() >= .82:
        return raw, 1
    points = np.vstack((raw, cue))
    lo = np.maximum(0, np.floor(points.min(axis=0)-.20*scale-5.)).astype(int)
    hi = np.minimum(frame.shape[1::-1], np.ceil(points.max(axis=0)+.20*scale+6.)).astype(int)
    x, y = lo
    r, b = hi
    chip = cv2.cvtColor(frame[y:b, x:r], cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(chip, (3, 3), .55)
    target = raw-lo
    cue = (cue-lo).astype(np.float32)
    mask = filled(chip.shape, target)
    area = cv2.countNonZero(mask)
    direction = cue[-1]-cue[0]
    direction /= max(np.linalg.norm(direction), 1e-6)
    outward = np.array([-direction[1], direction[0]])
    centre = target.mean(axis=0)
    if (cue.mean(axis=0)-centre) @ outward < 0.:
        outward *= -1.
    py, px = np.indices(chip.shape, dtype=np.float32)
    a = px*direction[0] + py*direction[1]
    d = px*outward[0] + py*outward[1]
    along = np.floor(a).astype(np.int32)
    bins = along-along.min()
    distal = np.full(bins.max()+1, -np.inf)
    np.maximum.at(distal, bins[mask > 0], d[mask > 0])
    available = np.flatnonzero(np.isfinite(distal))
    if not len(available):
        return raw, 0
    boundary = np.interp(bins, available, distal[available])
    rail_depth = cue @ outward
    far = rail_depth.max()+max(3., .033*scale)
    near = centre @ outward + max(.09*scale, .26*(rail_depth.mean()-centre @ outward))
    limits = target @ direction
    corridor = ((d >= np.maximum(boundary-2.4, near)) & (d <= far)
                & (a >= limits.min()-.13*scale) & (a <= limits.max()+.13*scale))
    # Refine the missing distal coverage while preserving the original
    # shaft/cuff and lateral source outline.
    envelope = (corridor | (mask > 0)).astype(np.uint8)
    adjacent = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    inner_envelope = cv2.erode(envelope, np.ones((3, 3), np.uint8))
    polarity = []
    for sign in (-1., 1.):
        q = (cue + sign*1.9*outward).astype(np.float32)
        polarity.append(cv2.remap(chip, q[:, :1], q[:, 1:], cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE).astype(np.float32))
    contrast = np.median(polarity[1]-polarity[0])
    if abs(contrast) < 2.:
        return raw, 0
    bright = contrast < 0.
    threshold, _ = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    candidates = []
    for delta in (-13., -6., 0., 6., 13.):
        native = ((smooth > threshold+delta) if bright else (smooth < threshold+delta)).astype(np.uint8)
        contours, _ = cv2.findContours(native & envelope, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for polygon in contours:
            if len(polygon) < 4 or not .03*area <= cv2.contourArea(polygon) <= 1.24*area:
                continue
            distance = np.array([cv2.pointPolygonTest(polygon, tuple(p), True) for p in cue])
            if (abs(distance) <= max(2.8, .026*scale)).mean() < .64 or np.median(distance) < -2.7:
                continue
            path = np.rint(distribute(polygon[:, 0], 48, closed=True)).astype(int)
            a, b = path.T
            appended = mask[b, a] == 0
            if not appended.any() or np.mean(inner_envelope[b[appended], a[appended]] == 0) > .28:
                continue
            other = filled(chip.shape, polygon[:, 0])
            if cv2.countNonZero(other & adjacent) < max(6, .035*cv2.countNonZero(other)):
                continue
            added = other & (1-mask)
            ratio = cv2.countNonZero(added)/max(area, 1)
            if not .029 <= ratio <= .60:
                continue
            cost = float(np.median(abs(distance)) + .028*abs(delta) + .50*ratio)
            candidates.append((cost, mask | other, added))
    candidates.sort(key=lambda item: item[0])
    for index, (_, proposal, added) in enumerate(candidates):
        match = False
        for j, (_, _, alternative) in enumerate(candidates):
            if index != j and cv2.countNonZero(added & alternative) >= .63*cv2.countNonZero(added | alternative):
                match = True
                break
        if not match:
            continue
        connected = cv2.morphologyEx(proposal, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) | proposal
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = max(contours, key=cv2.contourArea)[:, 0]
        if cv2.countNonZero(filled(mask.shape, outline) & mask) < .99*area:
            continue
        return distribute(outline+lo, len(raw), closed=True), 2
    return raw, 0


def track_whole(inputs, raw, status, line, signal):
    region = raw.copy()
    coverage = np.zeros(status.shape, np.uint8)
    start = time.perf_counter()
    for a, b in inputs.ranges():
        if signal[a:b].any():
            for k, (pts, frame) in enumerate(inputs.frames(a, b), start=a):
                if pts != inputs.pts[k]:
                    raise ValueError("Whole-skate region is off the source clock")
                for f in np.flatnonzero(signal[k] & (status[k] > 0)):
                    candidate, kind = extend_region(frame, raw[k, f], line[k, f])
                    if kind == 2 and status[k, 1-f] and mask_overlap(candidate, raw[k, 1-f])[1] >= .46:
                        continue
                    region[k, f], coverage[k, f] = candidate, kind
                if np.all(coverage[k] == 2) and mask_overlap(*region[k])[1] >= .46:
                    region[k], coverage[k] = raw[k], 0
        emit("whole_track", shot=str(inputs.shots[a]), exposures=int(b-a), additions=int((coverage[a:b] == 2).sum()),
             original_full=int((coverage[a:b] == 1).sum()), seconds=round(time.perf_counter()-start, 1))
    return region, coverage


def review_frame(inputs, source, region, status, coverage, i):
    main = source.copy()
    panel = np.full((1080, 480, 3), 23, np.uint8)
    text(panel, stamp(inputs, i), (18, 32), .58)
    for f, side in enumerate(("Left", "Right")):
        y = 78+f*474
        data = region[i, f]
        if status[i, f]:
            names = ("extent uncertain", "source outline", "source extension")
            label = names[coverage[i, f]]
            lines(main, data, FOOT_COLOUR[f], close=True, thick=2)
        else:
            label = "unassociated"
        if status[i, f] == 3:
            label += " / ambiguous track"
        text(panel, side+" skate  "+label, (18, y), .66, FOOT_COLOUR[f])
        crop = data if status[i, f] else inputs.old_gear[i, f, :14]
        panel[y+22:y+424] = zoom(main, crop, 480, 402)
    return np.hstack((main, panel))


def main(prior, cues, output, video):
    cv2.setNumThreads(1)
    inputs = Inputs(verify_media=not video)
    path = output / "whole_skate_evidence.npz"
    if video:
        with np.load(path, allow_pickle=False) as f:
            np.testing.assert_array_equal(f["source_pts"], inputs.pts)
            np.testing.assert_array_equal(f["source_video_sha256"], inputs.media_sha)
            region, status, coverage = f["region"], f["status"], f["extent"]
        frames = (review_frame(inputs, source, region, status, coverage, i)
                  for i, (_, source) in enumerate(inputs.frames(0, len(inputs.pts))))
        render(inputs, output, "whole_skate_review.mp4", frames, (2400, 1080))
        return
    raw, status, line, signal = source_packet(inputs, prior, cues)
    output.mkdir(parents=True, exist_ok=True)
    region, coverage = track_whole(inputs, raw, status, line, signal)
    np.testing.assert_array_equal(region[coverage != 2], raw[coverage != 2])
    atomic_npz(path, region=region, status=status, extent=coverage, source_pts=inputs.pts,
               shot_ids=inputs.shots, time_base=inputs.tb, source_video_sha256=np.array(inputs.media_sha))
    navigation = read(prior / "navigation.json")
    navigation["events"] = [row for row in navigation["events"]
                            if row.get("kind") == "source_action"]
    cases = [(1120000, 0), (1356544, 0), (1518336, 0), (2391552, 0), (2391552, 1)]
    for pts, f in cases:
        k = inputs.index.get(pts)
        if k is None:
            continue
        label = f"{inputs.times[k]:.3f}s {'左' if f == 0 else '右'}鞋刀整体轮廓"
        navigation["events"].append(dict(id=f"whole-{pts}-{f}", label=label, kind="whole_skate",
                                         start_pts=pts, end_pts=pts, worst_pts=pts, exposure_count=1))
    navigation["events"].sort(key=lambda row: (row["start_pts"], row["end_pts"]))
    atomic_json(output / "navigation.json", navigation)
    audit = dict(exposures=len(region), foot_exposures=int(status.size),
                 tracked=int((status > 0).sum()), original_outline_supported=int((coverage == 1).sum()),
                 connected_foreground_added=int((coverage == 2).sum()),
                 tracked_extent_uncertain=int(((status > 0) & (coverage == 0)).sum()),
                 source_extent_codes=["unverified extent", "original outline with source support", "native extension"],
                 source_pts=[int(inputs.pts[0]), int(inputs.pts[-1])], time_base=list(map(int, inputs.tb)))
    atomic_json(output / "whole_skate_checks.json", audit)
    emit("whole_result", **audit)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--cues", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if not args.video and args.prior is None:
        parser.error("--prior is required for a new whole-skate run")
    main(args.prior, args.cues, args.output, args.video)
