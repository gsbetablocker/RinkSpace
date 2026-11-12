"""Visible boot/steel evidence, source-only checks and exact-PTS failure navigation."""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path
import time

import cv2
import numpy as np

from attachment import BootAttachment, cutting_profiles
from edge_tracks import CUTTING_EDGE, OTHER_BOUNDARIES, SOURCE_REVIEW, lower_edge_seeds, reference_parts, trace_lower_edges
from inputs import (BOOT, STEEL, WEIGHTS, Inputs, atomic_json, atomic_npz, digest, direct_rows,
                    distribute, emit, geometry, masked_distance, read)
from track import Navigation, VideoMasks, reassociate, unit

STEEL_POINTS = 12
MAX_LINES = 6
INFERRED, PROPAGATED, OBSERVED, MULTIPLE = range(4)


def colour_status(mode):
    return ("inferred", "propagated", "observed", "multiple")[int(mode)]


def boot_axis(poly, shin):
    centred = poly - poly.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    along = vt[0]
    up = unit(shin)
    # In a toe-on view the long mask axis is often the shaft. Retain a second
    # direction estimated from the distal arc, instead of fixing a toe label.
    far = (centred @ up) <= np.quantile(centred @ up, .38)
    base = centred[far]
    _, _, bottom = np.linalg.svd(base - base.mean(axis=0), full_matrices=False)
    return along, bottom[0]


def opposed_edges(gray, points, inward, width):
    """Require a resolved steel strip on the inward side of the source edge."""
    at = np.stack((-1.6 * np.ones_like(width), np.minimum(1.1, .31 * width), .48 * width,
                   .66 * width, .84 * width, 1.03 * width,
                   1.22 * width, 1.40 * width, 1.59 * width), axis=-1)
    xy = points[:, None] + inward[:, None] * at[..., None]
    inside = ((xy >= .5) & (xy <= np.asarray(gray.shape[::-1]) - 1.5)).all(axis=(1, 2))
    values = cv2.remap(gray, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32),
                       cv2.INTER_LINEAR).astype(np.float32)
    outside, inner, middle = values[:, :3].T
    first = .60 * inner + .40 * middle - outside
    # The other boundary can be blurred or shifted slightly from the model.
    # Search only towards the steel web, at a width supported by this view.
    later = (values[:, 5:] - values[:, 3:7]) * -np.sign(first[:, None])
    opposite = later.max(axis=1)
    contrast = np.minimum(abs(first), opposite)
    return inside & (contrast >= 5.), contrast


def segment_options(frame, origin, detector, poly, other, ankle, shin, fitted, *, forbidden=()):
    if fitted is None:
        return []
    scale = np.linalg.norm(np.ptp(poly, axis=0))
    rails = np.concatenate((fitted.lower.reshape(-1, 2), fitted.upper.reshape(-1, 2)))
    margin = max(6., .15 * scale)
    lo = np.maximum(origin, np.floor(rails.min(axis=0) - margin)).astype(int)
    hi = np.minimum(origin + np.asarray(frame.shape[1::-1]), np.ceil(rails.max(axis=0) + margin)).astype(int)
    x, y = lo - origin
    r, b = hi - origin
    if r - x < 12 or b - y < 12:
        return []
    gray = frame[y:b, x:r]
    lines, _, _, _ = detector.detect(gray)
    if lines is None:
        return []
    contours = poly.astype(np.float32)
    outside = None if other is None else other.astype(np.float32)
    cost = []
    for line in lines.reshape(-1, 4):
        p, q = line.reshape(2, 2) + lo
        delta = q - p
        length = np.linalg.norm(delta)
        if not max(5., .10 * scale) <= length <= 1.12 * scale:
            continue
        sample = distribute([p, q], STEEL_POINTS)
        gap = -np.array([cv2.pointPolygonTest(contours, tuple(uv), True) for uv in sample])
        qualified = gap >= 1.25
        if forbidden:
            qualified &= np.min([masked_distance(sample, edge) for edge in forbidden], axis=0) >= 2.
        if outside is not None:
            qualified &= np.array([cv2.pointPolygonTest(outside, tuple(uv), False) < 0 for uv in sample])
        if shin is not None:
            relative = sample - ankle
            along = relative @ unit(shin)
            across = np.abs(relative @ unit(np.array([-shin[1], shin[0]])))
            qualified &= ~((along > .32 * fitted.radius) & (along < np.linalg.norm(shin)) & (across < .48 * fitted.radius))
        if qualified.sum() < 5:
            continue
        supported, normal, width, error = cutting_profiles(sample, fitted)
        supported &= qualified & (gap >= .62 * width)
        for view, side in zip(*np.where(supported.sum(axis=-1) >= 5)):
            good = supported[view, side].copy()
            ij = np.flatnonzero(good)
            web_inward = normal[view, side, ij]
            web_width = width[view, side, ij]
            at = sample[ij] + (.55 * web_width)[:, None] * web_inward
            valid, contrast = opposed_edges(gray, sample[ij] - lo, web_inward, web_width)
            valid &= np.array([cv2.pointPolygonTest(contours, tuple(uv), True) < .5 for uv in at])
            good[ij] = valid
            for first, last in segments(good):
                sub = sample[first:last+1]
                if len(sub) < 5 or np.linalg.norm(sub[0] - sub[-1]) < max(4., .075 * scale):
                    continue
                signal = np.median(contrast[(ij >= first) & (ij <= last)])
                value = (.22 + 1.4 * fitted.quality[view] + .09 * np.median(error[view, side, first:last+1])
                         + .23 * (1. - len(sub) / STEEL_POINTS) - .005 * min(signal, 30.))
                cost.append((float(value), distribute(sub, STEEL_POINTS)))
    cost.sort(key=lambda item: item[0])
    candidates = []
    for value, sample in cost:
        if any(np.median(masked_distance(sample, old)) < 1.6 for _, old in candidates):
            continue
        candidates.append((value, sample))
        if len(candidates) >= MAX_LINES - 1:
            break
    return candidates


def direction_regions(poly, steel, ankle, rows, shin):
    if not np.isfinite(poly).all():
        return np.full((2, 2), np.nan, np.float32), False
    if np.isfinite(steel).all():
        along = unit(steel[-1] - steel[0])
    else:
        _, along = boot_axis(poly, shin)
    support = poly @ along
    ends = np.stack([poly[support <= np.quantile(support, .15)].mean(axis=0),
                     poly[support >= np.quantile(support, .85)].mean(axis=0)])
    # These are silhouette regions, not material endpoints.
    toe = next((r for r in rows if r["semantic"] == "toe_shell"), None)
    heel = next((r for r in rows if r["semantic"] == "heel_shell"), None)
    if toe is not None:
        candidate = int(np.linalg.norm(ends - np.mean(toe["polyline_uv"], axis=0), axis=-1).argmin())
        return ends[[1 - candidate, candidate]], True
    if heel is not None:
        candidate = int(np.linalg.norm(ends - np.mean(heel["polyline_uv"], axis=0), axis=-1).argmin())
        return ends[[candidate, 1 - candidate]], True
    distances = np.linalg.norm(ends - ankle, axis=-1)
    # Ankle proximity only orders hypotheses. It cannot resolve a front/back
    # view reliably, particularly during plantarflexion.
    return ends if distances[0] <= distances[1] else ends[::-1], False


def nearby_shank(inputs, nav, i, foot, poly):
    """Use a local 2D leg cue when the boot can be associated to one."""
    scale = np.linalg.norm(np.ptp(poly, axis=0))
    predicted = inputs.old_coco[i, [13, 15, 14, 16]].reshape(2, 2, 2)
    choices = []
    for joints, direct in ((nav.joints[i], nav.direct[i].all(axis=1)), (predicted, [False, False])):
        for j, (knee, ankle) in enumerate(joints):
            shin = knee - ankle
            if not np.isfinite(shin).all() or np.linalg.norm(shin) < 3.:
                continue
            gap = max(0., -cv2.pointPolygonTest(poly, tuple(ankle), True))
            if gap <= max(5., .27 * scale):
                choices.append((gap + (.09 * scale if j != foot else 0.)
                                + (0. if direct[j] else .10 * scale), ankle, shin, direct[j]))
    if not choices:
        return None, None, False
    _, ankle, shin, direct = min(choices, key=lambda item: item[0])
    return ankle, shin, direct


def choose_lines(lines, unary, boot, scale, boot_flow):
    """Link attached pixel segments of both feet on the shot's source clock."""
    n, _, count = unary.shape
    hidden = count - 1
    states = np.array(list(product(range(count), repeat=2)))
    cost_by_frame = unary[:, 0, states[:, 0]] + unary[:, 1, states[:, 1]]
    for i in range(n):
        for left in np.flatnonzero(np.isfinite(unary[i, 0, :hidden])):
            for right in np.flatnonzero(np.isfinite(unary[i, 1, :hidden])):
                a, b = lines[i, 0, left], lines[i, 1, right]
                # The same visible edge cannot independently support two skates.
                shorter, longer = (a, b) if np.linalg.norm(a[-1]-a[0]) < np.linalg.norm(b[-1]-b[0]) else (b, a)
                if (masked_distance(shorter, longer) <= 2.4).mean() >= .72:
                    cost_by_frame[i, (states[:, 0] == left) & (states[:, 1] == right)] = np.inf
    best = cost_by_frame[0].copy()
    score = [best]
    centres = np.nan_to_num(lines.mean(axis=-2))
    boot_xy = np.nan_to_num(boot.mean(axis=-2))
    # Endpoints can move along one runner as the visible part changes.
    angles = lines[..., -1, :] - lines[..., 0, :]
    length = np.linalg.norm(angles, axis=-1)
    tangents = np.nan_to_num(angles / np.maximum(length[..., None], 1e-6))
    normals = np.stack((-tangents[..., 1], tangents[..., 0]), axis=-1)
    back = np.zeros((n, len(states)), np.int16)
    transitions = []
    for i in range(1, n):
        trans = np.zeros((len(states), len(states)))
        for foot in range(2):
            delta = boot_xy[i, foot] - boot_xy[i-1, foot]
            flow = boot_flow[i, foot]
            shift = flow if np.isfinite(flow).all() else delta
            mismatch = centres[i, foot][None] - centres[i-1, foot][:, None] - shift
            cross = np.maximum(abs((mismatch * normals[i-1, foot][:, None]).sum(-1)),
                               abs((mismatch * normals[i, foot][None]).sum(-1)))
            parallel = abs(tangents[i-1, foot] @ tangents[i, foot].T)
            limit = max(4., .11 * scale[i, foot])
            steps = .42 * np.minimum(2., cross / limit) + .38 * (1. - parallel)
            if np.isfinite(boot[i-1:i+1, foot]).all():
                long = (length[i-1, foot][:, None] > .24 * scale[i, foot]) & (length[i, foot][None] > .24 * scale[i, foot])
                steps[(cross > limit) | (long & (parallel < .55))] = np.inf
            steps[-1, :] = steps[:, -1] = .28
            steps[-1, -1] = 0.
            trans += steps[states[:, foot, None], states[None, :, foot]]
        cost = best[:, None] + trans
        back[i] = cost.argmin(axis=0)
        best = cost.min(axis=0) + cost_by_frame[i]
        score.append(best)
        transitions.append(trans)
    path = np.zeros(n, np.int16)
    path[-1] = best.argmin()
    remain = np.zeros(len(states))
    alternatives = np.full((n, 2), -1, np.int8)
    for i in range(n - 1, -1, -1):
        choice = path[i]
        total = score[i] + remain
        for foot in range(2):
            selected = states[choice, foot]
            if selected == hidden:
                continue
            available = np.flatnonzero((states[:, foot] != selected) & (states[:, foot] != hidden)
                                       & (total < total[choice] + .38))
            for other in available[np.argsort(total[available])]:
                alt = states[other, foot]
                if np.median(masked_distance(lines[i, foot, selected], lines[i, foot, alt])) > 2.3:
                    alternatives[i, foot] = alt
                    break
        if i:
            path[i - 1] = back[i, choice]
            remain = (transitions[i - 1] + cost_by_frame[i] + remain).min(axis=1)
    return states[path], alternatives


def process_shot(inputs, nav, data, attachment, seeds, references):
    a = inputs.index[int(data["source_pts"][0])]
    b = inputs.index[int(data["source_pts"][-1])] + 1
    np.testing.assert_array_equal(data["source_pts"], inputs.pts[a:b])
    boot = data["boot"]
    n = b - a
    lines = np.full((n, 2, MAX_LINES + 1, STEEL_POINTS, 2), np.nan, np.float32)
    unary = np.full((n, 2, MAX_LINES + 1), np.inf, np.float32)
    origin = np.zeros((n, 2, MAX_LINES + 1), np.uint8)
    feature = np.zeros((n, 2, MAX_LINES + 1), np.uint8)
    source_anchor = np.full((n, 2, MAX_LINES + 1), -1, np.int64)
    paired_web = np.zeros((n, 2, MAX_LINES + 1), bool)
    unary[:, :, -1] = .82
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    frames = []
    bounds = data["chip_bounds"]
    for k, (stamp, frame) in enumerate(inputs.frames(a, b)):
        if stamp != data["source_pts"][k]:
            raise ValueError("Steel source time differs from mask")
        x, y, r, bottom = bounds[k]
        frames.append(cv2.cvtColor(frame[y:bottom, x:r], cv2.COLOR_BGR2GRAY))
        for f in range(2):
            if (a+k, f) in seeds:
                lines[k, f, 0] = distribute(seeds[a+k, f], STEEL_POINTS)
                unary[k, f, 0] = -1.7
                origin[k, f, 0] = OBSERVED
                feature[k, f, 0] = 1
                source_anchor[k, f, 0] = stamp
    for (k, f), (confidence, anchor, curve) in trace_lower_edges(inputs, a, data, frames, seeds, references).items():
        if origin[k, f, 0] == OBSERVED:
            continue
        # The source trace establishes material identity. A standalone
        # contrast edge remains ambiguous among sole, post and steel web.
        lines[k, f, 1] = distribute(curve, STEEL_POINTS)
        unary[k, f, 1] = .57 - .36 * confidence
        origin[k, f, 1] = PROPAGATED
        feature[k, f, 1] = 1
        source_anchor[k, f, 1] = inputs.pts[anchor]
        other = boot[k, 1-f] if np.isfinite(boot[k, 1-f]).all() else None
        ankle, shin, direct_shank = nearby_shank(inputs, nav, a+k, f, boot[k, f])
        fitted = attachment.align(f, boot[k, f], ankle, shin)
        negative = [np.asarray(row["polyline_uv"]) for row in inputs.fit.get((a+k, f), [])
                    if row["semantic"] in OTHER_BOUNDARIES]
        candidates = segment_options(frames[k], bounds[k, :2], detector, boot[k, f], other,
                                     ankle if direct_shank else None, shin if direct_shank else None,
                                     fitted, forbidden=negative)
        paired_web[k, f, 1] = any(np.quantile(masked_distance(candidate, curve), .8) < 1.8
                                  for _, candidate in candidates)
    status = np.zeros((n, 2), np.uint8)
    semantic = np.zeros_like(status)
    steel = lines[:, :, -1].copy()
    alt = steel.copy()
    ends = np.full((n, 2, 2, 2), np.nan, np.float32)
    heel_toe_ordered = np.zeros((n, 2), bool)
    paths, alternatives = choose_lines(lines, unary, boot, nav.size[a:b], data["boot_flow_from_previous"])
    frame, foot = np.indices(paths.shape)
    for f in range(2):
        path, alternate = paths[:, f], alternatives[:, f]
        ix = np.arange(n)
        steel[:, f] = lines[ix, f, path]
        semantic[:, f] = feature[ix, f, path]
        status[:, f] = origin[ix, f, path]
        different = alternate >= 0
        alt[different, f] = lines[ix[different], f, alternate[different]]
        status[different & (status[:, f] != OBSERVED), f] = MULTIPLE
        for k in range(n):
            ends[k, f], heel_toe_ordered[k, f] = direction_regions(
                boot[k, f], steel[k, f], nav.ankle[a + k, f],
                direct_rows(inputs.fit.get((a + k, f), []), BOOT), nav.shin[a + k, f])
    return dict(steel=steel, steel_alt=alt, steel_candidates=lines[:, :, :-1], steel_candidate_cost=unary[:, :, :-1],
                steel_status=status, steel_semantic=semantic, steel_anchor_pts=source_anchor[frame, foot, paths],
                steel_web_confirmed=paired_web[frame, foot, paths], heel_toe=ends, heel_toe_ordered=heel_toe_ordered)


def load_parts(inputs, output, *, full=True, shots=None):
    packets = []
    for path in sorted((output / "tracked").glob("track_s*.npz")):
        if shots is not None and path.stem.removeprefix("track_") not in shots:
            continue
        with np.load(path, allow_pickle=False) as f:
            packets.append((path, {k: f[k] for k in f.files}))
    if not packets:
        raise ValueError("No tracked shots")
    if full:
        np.testing.assert_array_equal(np.concatenate([p["source_pts"] for _, p in packets]), inputs.pts)
        np.testing.assert_array_equal(np.concatenate([p["shot_ids"] for _, p in packets]), inputs.shots)
    return packets


def check_rows(inputs, data, *, all_rows=None):
    """Disjoint original source rows. No fabricated errors for missing outputs."""
    output = []
    steel_features = CUTTING_EDGE if "steel_anchor_pts" in data else STEEL
    for row in inputs.checks if all_rows is None else all_rows:
        i = inputs.index[row["pts"]]
        foot = row["foot"]
        feature = row["semantic"]
        if foot not in (0, 1) or feature not in BOOT | steel_features:
            continue
        target = np.asarray(row["polyline_uv"])
        is_boot = feature in BOOT
        curve = data["boot" if is_boot else "steel"][i, foot]
        old_curve = inputs.old_gear[i, foot]
        if is_boot:
            # This comparison uses the saved parent shell triangle outlines.
            # See pose stage for identical native visible-geometry scoring.
            old_error = None
        else:
            old_error = float(min(np.median(masked_distance(target, old_curve[14:31])),
                                  np.median(masked_distance(target, old_curve[31:48]))))
        present = bool(np.isfinite(curve).all())
        error = float(np.median(masked_distance(target, curve, closed=is_boot))) if present else None
        output.append(dict(pts=int(row["pts"]), foot=int(foot), semantic=feature, group="boot" if is_boot else "steel",
                           direct=row.get("evidence_status") == "observed", status=colour_status(data[
                               "boot_status" if is_boot else "steel_status"][i, foot]),
                           radius_px=float(row.get("radius_px") or 6.), error_px=error, old_steel_px=old_error))
    return output


def segments(mask):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return zip(starts, ends, strict=True)


def navigation(inputs, data, rows, *, pose_flags=None):
    events = []
    pts = inputs.pts
    for foot in range(2):
        side = "左" if foot == 0 else "右"
        for label, use, minimum in (
            ("靴体连续漏跟", data["boot_status"][:, foot] == INFERRED, 7),
            ("靴体身份多解", data["boot_status"][:, foot] == MULTIPLE, 4),
            ("靴体可见而刃口未确认", (data["boot_status"][:, foot] > 0)
             & (data["steel_status"][:, foot] == 0), 14),
        ):
            for a, b in segments(use):
                if b - a + 1 < minimum:
                    continue
                events.append(dict(id=f"{foot}-{label}-{a}", label=f"{side}{label} {inputs.times[a]:.2f}–{inputs.times[b]:.2f}s",
                                   start_pts=int(pts[a]), end_pts=int(pts[b]), worst_pts=int(pts[(a + b) // 2]),
                                   exposure_count=int(b - a + 1)))
    for j, row in enumerate(rows):
        if not row["direct"] or (row["error_px"] is not None and row["error_px"] <= row["radius_px"]):
            continue
        k = inputs.index[row["pts"]]
        label = "留出轮廓漏跟" if row["error_px"] is None else f"留出轮廓偏差 {row['error_px']:.1f}px"
        events.append(dict(id=f"check-{j}", label=f"{label} {row['semantic']} {inputs.times[k]:.2f}s",
                           start_pts=int(pts[max(0, k - 22)]), end_pts=int(pts[min(len(pts) - 1, k + 22)]),
                           worst_pts=int(pts[k]), exposure_count=1))
    if pose_flags is not None:
        for a, b in segments(pose_flags):
            events.append(dict(id=f"pose-{a}", label=f"三维分支不稳 {inputs.times[a]:.2f}–{inputs.times[b]:.2f}s",
                               start_pts=int(pts[max(0, a - 10)]), end_pts=int(pts[min(len(pts) - 1, b + 10)]),
                               worst_pts=int(pts[a]), exposure_count=int(b - a + 1)))
    for event in inputs.evidence["events"]:
        a = int(np.clip(np.searchsorted(pts, event["start_pts"]), 0, len(pts) - 1))
        b = int(np.clip(np.searchsorted(pts, event["end_pts"], side="right") - 1, a, len(pts) - 1))
        events.append(dict(id="action-" + event["id"], label=event.get("name_zh") or event.get("label_zh") or event["id"],
                           start_pts=int(pts[a]), end_pts=int(pts[b]), worst_pts=int(pts[a]),
                           exposure_count=b-a+1, kind="source_action"))
    cuts = np.flatnonzero(inputs.shots[1:] != inputs.shots[:-1]) + 1
    return dict(schema="rinkspace.viewer-source-navigation.v1", time_base=list(map(int, inputs.tb)),
                cuts=[dict(pts=int(pts[k]), previous_pts=int(pts[k - 1]),
                           label=f"{inputs.shots[k - 1]} → {inputs.shots[k]}  {inputs.times[k]:.2f}s") for k in cuts],
                events=sorted(events, key=lambda r: (r["start_pts"], r["end_pts"])),
                programme_actions=inputs.action)


def summarize(rows):
    out = {}
    for group in ("boot", "steel"):
        for role in ("all", "direct"):
            sample = [r for r in rows if r["group"] == group and (role == "all" or r["direct"])]
            found = [r for r in sample if r["error_px"] is not None]
            error = [r["error_px"] for r in found]
            ok = [r for r in found if r["error_px"] <= r["radius_px"]]
            out[f"{group}_{role}"] = dict(rows=len(sample), present=len(found), within_radius=len(ok),
                                           p50_px=float(np.median(error)) if error else None,
                                           p95_px=float(np.percentile(error, 95)) if error else None)
    return out


def cached_features(path, identity, build):
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["cache_identity"]) == identity:
                return {key: saved[key] for key in saved.files if key != "cache_identity"}
    result = build()
    atomic_npz(path, cache_identity=np.array(identity), **result)
    return result


def main(output, partial=False, replay_tracks=None, shots=None, boot_review=None):
    from model import SkateModel
    from recover import source_prompt_repair

    cv2.setNumThreads(1)
    inputs = Inputs()
    nav = Navigation(inputs)
    attachment = BootAttachment(SkateModel(geometry(inputs)[0]))
    seeds = lower_edge_seeds(inputs)
    references = reference_parts(inputs, seeds)
    prompts = {}
    if boot_review:
        document = read(boot_review)
        if (document["schema"] != "rinkspace.skate-boot-source-review.v1"
                or document["source_video_sha256"] != inputs.media_sha
                or tuple(document["time_base"]) != tuple(inputs.tb) or document["role"] != "fit"):
            raise ValueError("Boot review does not match the native source")
        for row in document["anchors"]:
            prompts.setdefault(str(inputs.shots[inputs.index[row["pts"]]]), []).append(row)
    track_root = output if replay_tracks is None else replay_tracks
    if replay_tracks is not None:
        if replay_tracks.resolve() == output.resolve():
            raise ValueError("Choose a separate output for revised tracking")
        with np.load(replay_tracks / "image_evidence.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["source_pts"], inputs.pts)
            if str(saved["source_video_sha256"]) != inputs.media_sha or str(saved["camera_sha256"]) != inputs.camera_sha:
                raise ValueError("Tracking replay source or camera changed")
    partial = partial or shots is not None
    packets = load_parts(inputs, track_root, full=not partial, shots=shots)
    code = Path(__file__).parent
    common = [inputs.media_sha, inputs.camera_sha, inputs.parent_sha, digest(inputs.document_path)]
    boot_code = [digest(code / file) for file in ("track.py", "inputs.py", "attachment.py")]
    steel_code = [digest(code / file) for file in ("evidence.py", "model.py", "edge_tracks.py")]
    review_sha = digest(SOURCE_REVIEW)
    steel_code.append(review_sha)
    from scripts import track_programme_skates
    steel_code.append(digest(Path(track_programme_skates.__file__)))
    (output / "tracked").mkdir(parents=True, exist_ok=True)
    all_features = []
    revised = []
    boot_model = None
    for raw_path, raw in packets:
        started = time.perf_counter()
        shot = str(raw["shot_ids"][0])
        identity = ":".join([digest(raw_path), *common, *boot_code])
        anchors = prompts.get(shot, [])
        if anchors:
            identity += ":" + ":".join((digest(boot_review), digest(code / "recover.py")))

        def link_boots():
            nonlocal boot_model
            if anchors:
                if boot_model is None:
                    boot_model = VideoMasks(WEIGHTS)
                source_prompt_repair(inputs, nav, raw, boot_model, anchors)
            return reassociate(inputs, nav, raw)

        emit("attachment_start", shot=shot, source_exposures=len(raw["source_pts"]))
        linked = output / "tracked" / f"attached_boot_{shot}.npz"
        packet = cached_features(linked, identity, link_boots)
        revised.append(packet)
        cached = output / "tracked" / f"attached_steel_{shot}.npz"
        features = cached_features(cached, ":".join([identity, *steel_code]),
                                   lambda: process_shot(inputs, nav, packet, attachment, seeds, references))
        all_features.append(features)
        emit("steel", shot=str(packet["shot_ids"][0]), exposures=len(packet["source_pts"]),
             detected=int((features["steel_status"] > 0).sum()), seconds=round(time.perf_counter() - started, 2))
    if partial:
        return
    metadata = ("sam2_sha256", "sam_stride")
    data = {key: revised[0][key] if key in metadata else np.concatenate([p[key] for p in revised])
            for key in revised[0]}
    data.update({key: np.concatenate([p[key] for p in all_features]) for key in all_features[0]})
    data.update(source_video_sha256=np.array(inputs.media_sha), camera_sha256=np.array(inputs.camera_sha),
                time_base=inputs.tb, check_protection=inputs.withheld, cutting_edge_review_sha256=np.array(review_sha),
                boot_review_sha256=np.array(digest(boot_review) if boot_review else ""))
    missing = data["boot_status"] == 0
    offscreen = ((nav.centre < [0, 0]) | (nav.centre >= [1920, 1080])).any(axis=-1)
    data["visibility_code"] = np.where(missing, np.where(offscreen, 3, 4),
                                      np.where(data["roi_partial"], 1, np.where(data["overlap"][:, None], 2, 0))).astype(np.uint8)
    data["visibility_meaning"] = np.array(["visible candidate", "partial or source border", "mutual overlap",
                                            "search hints outside source", "visibility unresolved"])
    rows = check_rows(inputs, data)
    atomic_npz(output / "image_evidence.npz", **data)
    atomic_json(output / "navigation.json", navigation(inputs, data, rows))
    summary = summarize(rows)
    atomic_json(output / "image_checks.json", dict(summary=summary, rows=rows))
    emit("image_checks", **summary)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--replay-tracks", type=Path)
    parser.add_argument("--boot-review", type=Path, help="Native source prompts for the affected boot tracks")
    parser.add_argument("--shots", nargs="+", help="Refresh selected shot caches before full publication")
    args = parser.parse_args()
    main(args.output, args.partial, args.replay_tracks, args.shots, args.boot_review)
