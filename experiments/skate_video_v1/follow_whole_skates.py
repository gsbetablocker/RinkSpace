"""Follow whole skates in both directions while retaining existing source outlines."""

from argparse import ArgumentParser
from pathlib import Path
import time

import cv2
import numpy as np

from attachment import normalized, silhouette, unit
from evidence import segments
from inputs import PROGRAMME, SIZE, WEIGHTS, Inputs, atomic_json, atomic_npz, digest, distribute, emit, geometry, read
from model import SkateModel
from recover import advance
from render_review import FOOT_COLOUR, lines, render, stamp, text, zoom
from track import POINTS, Navigation, VideoMasks, covered_body_components, covers_component, iou, mask_overlap, measure
from whole_skate_review import filled


ANCHORS = PROGRAMME / "whole_skate_source_anchors_1112.json"
PRIOR_UNCERTAIN, PRIOR_WHOLE, VIDEO_WHOLE, SOURCE_PROMPT, VIDEO_RECOVERY = range(5)


class WholeShape:
    """Silhouettes of the same rigid assembly under different viewing directions."""

    def __init__(self, model):
        angles = np.arange(20)*np.pi/10
        rolls = np.arange(20)*np.pi/10
        c, s = np.cos(rolls), np.sin(rolls)
        rotations = np.stack((c, s, -s, c), axis=-1).reshape(-1, 2, 2)
        self.directions = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
        faces = np.vstack((model.faces, model.geometry["lower_faces"]+model.boot.shape[1]))
        self.atlas = []
        for foot in range(2):
            mesh = np.vstack((model.boot[foot], model.geometry["lower"][foot]))
            views = []
            for pitch in np.deg2rad([-80, -64, -48, -32, -16, 0, 16, 32, 48, 64, 80]):
                for yaw in angles:
                    ray = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
                    down = -unit(np.array([0., 0., 1.])-ray[2]*ray)
                    axes = np.column_stack((np.cross(down, ray), down))
                    curve = silhouette(mesh @ axes, faces)
                    views.append(normalized(curve)[0] @ rotations)
            library = np.asarray(views, dtype=np.float32).reshape(-1, 40, 2)
            support = (library @ self.directions.T).max(axis=1)
            self.atlas.append((library, support))

    def error(self, curve, foot):
        target, _, radius = normalized(curve[::2])
        if radius < 2.:
            return 1.
        library, support = self.atlas[foot]
        target_support = (target @ self.directions.T).max(axis=0)
        coarse = ((support-target_support)**2).mean(axis=1)
        nearby = library[np.argpartition(coarse, 11)[:12]]
        dist = np.linalg.norm(nearby[:, :, None]-target[None, None], axis=-1)
        a, b = dist.min(axis=1), dist.min(axis=2)
        return float(np.min(.55*np.median(a, axis=-1)+.45*np.quantile(b, .75, axis=-1)))


def read_anchors(inputs):
    doc = read(ANCHORS)
    if doc["source_video_sha256"] != inputs.media_sha or tuple(doc["time_base"]) != tuple(inputs.tb):
        raise ValueError("Source prompts do not match the registered native video")
    return {(inputs.index[row["pts"]], row["foot"]): row for row in doc["anchors"]}


def crops(inputs, raw, start, stop, prompts):
    # The fitted ankle helps search for the skate only when the source track
    # is absent. All candidate outlines come from source-video pixels.
    guide = inputs.old_gear[start:stop, :, :14]
    centre = guide.mean(axis=2).copy()
    source = np.isfinite(raw).all(axis=(2, 3))
    size = np.linalg.norm(np.ptp(guide, axis=2), axis=-1)
    for foot in range(2):
        seen = np.flatnonzero(source[:, foot])
        if len(seen):
            known = raw[seen, foot].mean(axis=1)
            correction = known-centre[seen, foot]
            x = np.arange(len(centre))
            centre[:, foot] += np.column_stack([np.interp(x, seen, correction[:, a]) for a in (0, 1)])
            centre[seen, foot] = known
            measured = np.linalg.norm(np.ptp(raw[seen, foot], axis=1), axis=-1)
            size[:, foot] = np.interp(x, seen, measured)
    side = np.clip(np.rint(2.47*size+24), 140, 520).astype(np.int32)
    for i, f in prompts:
        if start <= i < stop:
            side[i-start, f] = int(np.ceil(max(158., 2.25*size[i-start, f])))
    lo = np.clip(np.rint(centre-side[..., None]*.5), 0, np.asarray(SIZE)-side[..., None]).astype(np.int32)
    bounds = np.concatenate((lo, lo+side[..., None]), axis=-1)
    chips = [[], []]
    for j, (pts, frame) in enumerate(inputs.frames(start, stop)):
        if pts != inputs.pts[start+j]:
            raise ValueError("Whole-skate crops left the native clock")
        for f in range(2):
            x, y, r, b = bounds[j, f]
            chips[f].append(np.ascontiguousarray(frame[y:b, x:r]))
    return chips, bounds, size


def prompt_row(row, bounds):
    inside, outside = np.asarray(row["inside"]), np.asarray(row["outside"])
    uv = np.vstack((inside, outside))
    valid = ((uv >= bounds[:2]) & (uv < bounds[2:])).all(axis=1)
    if not valid[:len(inside)].all():
        raise ValueError("The crop clipped a reviewed whole-skate prompt")
    labels = np.r_[np.ones(len(inside), dtype=int), np.zeros(len(outside), dtype=int)]
    return (uv[valid]-bounds[:2]).tolist(), labels[valid].tolist(), 2


def scan(model, inputs, start, foot, chips, bounds, size, raw, trust, weak, prompts, reverse):
    """One video memory per physical foot, on interleaved native exposures."""
    n = len(chips)
    shape = np.full((n, POINTS, 2), np.nan, np.float32)
    scores = np.full(n, -np.inf, np.float32)
    source = np.full(n, -1, np.int32)
    whole = np.zeros(n, bool)
    model.reset()
    scan_steps = np.arange(1 if reverse else 0, n, 2)
    # Source-read anchors are available to both passes on their exact frame.
    anchors = np.asarray([k-start for k, f in prompts if f == foot and start <= k < start+n], dtype=np.int32)
    order = np.union1d(scan_steps, anchors)
    if reverse:
        order = order[::-1]
    seed = -1
    reviewed = False
    lost = 0
    count = 0
    for j in order:
        clicks, masks = {}, {}
        special = prompts.get((start+j, foot))
        if special:
            clicks[foot] = prompt_row(special, bounds[j])
            seed, reviewed, lost = j, True, 0
        elif trust[j] and (seed < 0 or not reviewed or abs(j-seed) >= 32 or lost >= 4):
            masks[foot] = filled(chips[j].shape[:2], raw[j]-bounds[j, :2])
            seed, reviewed, lost = j, True, 0
        elif weak[j] and (seed < 0 or lost >= 8):
            # A new sighting can reinitialize the tracker after occlusion.
            # Its coverage stays provisional until source observations agree.
            if seed >= 0:
                model.reset()
                count = 0
            masks[foot] = filled(chips[j].shape[:2], raw[j]-bounds[j, :2])
            seed, reviewed, lost = j, False, 0
        if seed < 0:
            continue
        result = model.step(chips[j], count, clicks, seed_masks=masks).get(foot)
        count += 1
        item = None if result is None else measure(*result, bounds[j], max(18., size[j]))
        if item is None or result[1] < -.15:
            lost += 1
            continue
        polygon, score, _ = item
        if special:
            for point in special["inside"]:
                if cv2.pointPolygonTest(polygon, tuple(point), True) < -3.:
                    raise ValueError(f"Whole-skate prompt was missed at PTS {inputs.pts[start+j]}")
        shape[j], scores[j], source[j], whole[j] = polygon, score, seed, reviewed
        lost = 0
        if count % 180 == 0:
            emit("video_memory", shot=str(inputs.shots[start]), foot=foot, reverse=reverse,
                 pts=int(inputs.pts[start+j]), accepted=int(np.isfinite(scores).sum()))
    return shape, scores, source, whole


def boundary_support(image, bounds, outline, old=None):
    p = outline-bounds[:2]
    use = np.ones(len(outline), bool)
    if old is not None:
        use = np.array([cv2.pointPolygonTest(old, tuple(x), True) < -2.2 for x in outline])
    if use.sum() < 5:
        return 0.
    tangent = np.roll(p, -1, axis=0)-np.roll(p, 1, axis=0)
    n = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-4)
    p, n = p[use], n[use]
    values = []
    for sign in (-1., 1.):
        q = (p+sign*2.1*n).astype(np.float32)
        values.append(cv2.remap(image, q[:, :1], q[:, 1:], cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE).astype(np.float32))
    return float((np.mean(abs(values[1]-values[0]), axis=-1) > 6.).mean())


def nearby_prediction(chips, bounds, curves, j):
    """Carry an independent prediction into this exposure using native optical motion."""
    if np.isfinite(curves[j]).all():
        return curves[j]
    warps = []
    for k in (j-1, j+1):
        if 0 <= k < len(chips) and np.isfinite(curves[k]).all():
            shifted = advance(chips, bounds, k, j, curves[k])
            if shifted is not None:
                warps.append(shifted)
    return warps


def extra_part(base, proposed, chip, bounds, shank, scale):
    """Complete an existing outline while retaining its native cuff and sides."""
    lo = bounds[:2]
    a = filled(chip.shape[:2], base-lo)
    b = filled(chip.shape[:2], proposed-lo)
    added = b & (1-a)
    original_area = cv2.countNonZero(a)
    change = cv2.countNonZero(added)
    if change < max(9, .052*original_area) or change > .82*original_area:
        return None
    if cv2.countNonZero(a & b) < .69*original_area:
        return None
    if shank is not None:
        knee, ankle = shank
        up = unit(knee-ankle)
        d = (base-ankle) @ up
        y, x = np.where(added)
        uv = np.column_stack((x, y))+lo
        above = (uv-ankle) @ up > np.quantile(d, .86)+max(2., .035*scale)
        added[y[above], x[above]] = 0
    combined = a | added
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = max(contours, key=cv2.contourArea)[:, 0]
    outer = distribute(outline+lo, len(base), closed=True)
    if cv2.contourArea(outer) < 1.040*cv2.contourArea(base):
        return None
    return outer


def reconcile(inputs, nav, model_shape, start, raw, status, original_extent, chips, bounds, size, banks, prompts):
    result, state = raw.copy(), status.copy()
    reason = np.where(original_extent > 0, PRIOR_WHOLE, PRIOR_UNCERTAIN).astype(np.uint8)
    proof = np.full((len(raw), 2), -1, np.int64)
    motion_support = np.full((len(raw), 2), np.nan, np.float32)
    limbs, limb_span = covered_body_components(nav, np.arange(start, start+len(raw)))
    for j in range(len(raw)):
        i = start+j
        for f in range(2):
            if reason[j, f] == PRIOR_WHOLE:
                continue
            manual = (i, f) in prompts
            old = raw[j, f] if status[j, f] else None
            other = raw[j, 1-f] if status[j, 1-f] else None
            candidates = []
            for d, (polygons, quality, source, whole) in enumerate(banks[f]):
                p = polygons[j]
                if not np.isfinite(p).all():
                    continue
                if manual and source[j] != j:
                    continue
                if old is not None and mask_overlap(p, old)[0] < .34:
                    continue
                if other is not None and mask_overlap(p, other)[1] >= .60 and (
                        old is None or mask_overlap(old, other)[1] < .43):
                    continue
                if covers_component(p, limbs[j], limb_span[j], max(11., .32*size[j, f])):
                    continue
                errors = model_shape.error(p, f)
                if errors > .26 and not manual:
                    continue
                secondary = nearby_prediction(chips[f], bounds[:, f], banks[f][1-d][0], j)
                if isinstance(secondary, np.ndarray):
                    secondary = [secondary]
                agreement = max((iou(p, q) for q in secondary), default=0.)
                reviewed = whole[j]
                boundary = boundary_support(chips[f][j], bounds[j, f], p, old)
                same_motion = agreement >= (.54 if reviewed else .68)
                near_prompt = reviewed and (start+source[j], f) in prompts and abs(j-source[j]) <= 28
                if not manual and not same_motion and not (near_prompt and boundary >= .36):
                    continue
                if old is None and not manual:
                    if not (reviewed and (agreement >= .61 or near_prompt) and boundary >= .35):
                        continue
                    ankle = nav.ankle[i, f] if nav.direct[i, f, 1] else inputs.old_coco[i, 15+f]
                    if np.isfinite(ankle).all() and cv2.pointPolygonTest(p, tuple(ankle), True) < -.52*size[j, f]:
                        continue
                if old is not None and not manual:
                    if boundary < (.27 if reviewed else .39):
                        continue
                    shank = nav.joints[i, f] if nav.direct[i, f].all() else None
                    p = extra_part(old, p, chips[f][j], bounds[j, f], shank, size[j, f])
                    if p is None:
                        continue
                    if model_shape.error(p, f) > .27:
                        continue
                candidates.append((.65*agreement+.25*boundary+.15*quality[j]-.46*errors, p, source[j], agreement))
            if not candidates:
                continue
            _, chosen, anchor, agreement = max(candidates, key=lambda x: x[0])
            result[j, f], state[j, f] = chosen, 1
            reason[j, f] = SOURCE_PROMPT if manual else (VIDEO_RECOVERY if old is None else VIDEO_WHOLE)
            proof[j, f], motion_support[j, f] = inputs.pts[start+anchor], agreement
        # Reject two results that claim substantially the same source object.
        if np.all(state[j] > 0) and np.any(reason[j] >= VIDEO_WHOLE):
            if mask_overlap(*result[j])[1] > .67 and not (
                    np.all(status[j] > 0) and mask_overlap(*raw[j])[1] > .58):
                for f in np.flatnonzero(reason[j] >= VIDEO_WHOLE):
                    if reason[j, f] != SOURCE_PROMPT:
                        result[j, f], state[j, f], reason[j, f] = raw[j, f], status[j, f], PRIOR_UNCERTAIN
                        proof[j, f], motion_support[j, f] = -1, np.nan
    return dict(region=result, status=state, extent=reason, tracking_anchor_pts=proof, mask_agreement=motion_support)


def shot(model, model_shape, inputs, nav, start, stop, region, status, extent, prompts):
    a = time.perf_counter()
    raw, state, known = region[start:stop], status[start:stop], extent[start:stop]
    chips, bounds, size = crops(inputs, raw, start, stop, prompts)
    banks = []
    for f in range(2):
        available = state[:, f] == 1
        plausible = np.zeros(len(raw), bool)
        for k in np.flatnonzero(available):
            plausible[k] = model_shape.error(raw[k, f], f) < .22
        trust = available & plausible & (known[:, f] > 0)
        weak = available & plausible
        banks.append([scan(model, inputs, start, f, chips[f], bounds[:, f], size[:, f], raw[:, f],
                           trust, weak, prompts, reverse) for reverse in (False, True)])
    result = reconcile(inputs, nav, model_shape, start, raw, state, known, chips, bounds, size, banks, prompts)
    result.update(source_pts=inputs.pts[start:stop], shot_ids=inputs.shots[start:stop])
    counts = np.bincount(result["extent"].ravel(), minlength=5)
    emit("shot_complete", shot=str(inputs.shots[start]), counts=counts.tolist(),
         visible=int((result["status"] > 0).sum()), seconds=round(time.perf_counter()-a, 2))
    return result


def write_navigation(inputs, prior, output, result):
    navigation = read(prior/"navigation.json")
    for (i, f), _ in read_anchors(inputs).items():
        start, stop = max(0, i-12), min(len(inputs.pts)-1, i+16)
        navigation["events"].append(dict(id=f"whole-follow-{i}-{f}", label=f"{inputs.times[i]:.3f}s 整件鞋刀复核",
                                          start_pts=int(inputs.pts[start]), end_pts=int(inputs.pts[stop]),
                                          worst_pts=int(inputs.pts[i]), exposure_count=stop-start+1))
    for f in range(2):
        for mask, label in ((result["status"][:, f] == 0, "鞋刀暂失"),
                            ((result["status"][:, f] > 0) & (result["extent"][:, f] == 0), "整件范围待确认")):
            for a, b in inputs.ranges():
                for x, y in segments(mask[a:b]):
                    if y-x >= 18:
                        name = ("左" if f == 0 else "右")+label+f" {inputs.times[a+x]:.3f}s"
                        navigation["events"].append(dict(id=f"{label}-{f}-{a+x}", label=name, start_pts=int(inputs.pts[a+x]),
                                                          end_pts=int(inputs.pts[a+y]), worst_pts=int(inputs.pts[a+(x+y)//2]),
                                                          exposure_count=int(y-x+1)))
    navigation["events"].sort(key=lambda row: (row["start_pts"], row["end_pts"]))
    atomic_json(output/"navigation.json", navigation)


def publish(inputs, output, prior, region, status, known, identity):
    records = []
    for a, b in inputs.ranges():
        with np.load(output / "shots" / f"{inputs.shots[a]}.npz", allow_pickle=False) as f:
            np.testing.assert_array_equal(f["source_pts"], inputs.pts[a:b])
            np.testing.assert_array_equal(f["run_identity"], identity)
            records.append({key: f[key] for key in f.files if key != "run_identity"})
    result = {key: np.concatenate([part[key] for part in records]) for key in records[0]}
    preserved = known > 0
    np.testing.assert_array_equal(region[preserved], result["region"][preserved])
    np.testing.assert_array_equal(result["status"] > 0, np.isfinite(result["region"]).all(axis=(2, 3)))
    np.testing.assert_array_equal(result["source_pts"], inputs.pts)
    result.update(time_base=inputs.tb, source_video_sha256=np.array(inputs.media_sha))
    atomic_npz(output/"whole_follow_evidence.npz", **result)
    write_navigation(inputs, prior, output, result)
    cases = []
    for (i, f), _ in read_anchors(inputs).items():
        cases.append(dict(pts=int(inputs.pts[i]), seconds=float(inputs.times[i]), foot=f, extent=int(result["extent"][i, f]),
                          before_px2=float(cv2.contourArea(region[i, f])), after_px2=float(cv2.contourArea(result["region"][i, f]))))
    report = dict(source_frames=len(inputs.pts), foot_exposures=int(status.size), exact_prior_whole=int(preserved.sum()),
                  tracked=int((result["status"] > 0).sum()), missing=int((result["status"] == 0).sum()),
                  extent_uncertain=int(((result["status"] > 0) & (result["extent"] == PRIOR_UNCERTAIN)).sum()),
                  extended_by_video=int((result["extent"] == VIDEO_WHOLE).sum()),
                  recovered_by_video=int((result["extent"] == VIDEO_RECOVERY).sum()), source_prompt_cases=cases)
    atomic_json(output/"whole_follow_checks.json", report)
    emit("published", **report)


def review_frame(inputs, source, data, i):
    frame = source.copy()
    panel = np.full((1080, 480, 3), 23, np.uint8)
    text(panel, stamp(inputs, i), (18, 32), .60)
    names = ("extent uncertain", "original outline", "video tracked", "source anchor", "reacquired")
    uncertain_pair = "identity_unresolved" in data and bool(data["identity_unresolved"][i].any())
    for f, side in enumerate(("Left", "Right")):
        y = 78+f*474
        visible = bool(data["status"][i, f])
        p = data["region"][i, f]
        label = names[data["extent"][i, f]] if visible else "unresolved"
        if data["status"][i, f] == 3:
            label = "identity uncertain"
        if visible and "partial_occlusion" in data and data["partial_occlusion"][i, f]:
            label = "visible part only"
        if not visible and "skate_occluded" in data and data["skate_occluded"][i, f]:
            label = "occluded"
        if uncertain_pair:
            label = "identity unresolved"
        if visible:
            colour = (220,220,220) if uncertain_pair else FOOT_COLOUR[f]
            lines(frame, p, colour, close=True, thick=2)
        text(panel, side+" skate  "+label, (18, y), .69, FOOT_COLOUR[f])
        region = p if visible and not uncertain_pair else inputs.old_gear[i, f, :14]
        if "skate_occluded" in data:
            outline = (p, FOOT_COLOUR[f]) if visible and not uncertain_pair else None
            panel[y+22:y+424] = zoom(source, region, 480, 402, outline=outline)
        else:
            panel[y+22:y+424] = zoom(frame, region, 480, 402)
    return np.hstack((frame, panel))


def main(args):
    cv2.setNumThreads(1)
    inputs = Inputs(verify_media=not args.video and not args.merge)
    output = args.output
    if args.video:
        with np.load(output/"whole_follow_evidence.npz", allow_pickle=False) as f:
            np.testing.assert_array_equal(f["source_pts"], inputs.pts)
            np.testing.assert_array_equal(f["source_video_sha256"], inputs.media_sha)
            data = {key: f[key] for key in ("region", "status", "extent")}
        frames = (review_frame(inputs, source, data, i) for i, (_, source) in enumerate(inputs.frames(0, len(inputs.pts))))
        render(inputs, output, "whole_skate_follow_review.mp4", frames, (2400, 1080))
        return
    prior = args.prior/"whole_skate_evidence.npz"
    with np.load(prior, allow_pickle=False) as f:
        for key, target in (("source_pts", inputs.pts), ("shot_ids", inputs.shots), ("time_base", inputs.tb),
                            ("source_video_sha256", inputs.media_sha)):
            np.testing.assert_array_equal(f[key], target)
        region, status, extent = f["region"], f["status"], f["extent"]
    here = Path(__file__).parent
    identity = ":".join((inputs.media_sha, inputs.camera_sha, digest(prior), digest(ANCHORS),
                         str(args.image_size), digest(Path(__file__)), digest(here/"track.py"), digest(here/"recover.py")))
    if not args.merge:
        (output/"shots").mkdir(parents=True, exist_ok=True)
        model = VideoMasks(WEIGHTS, image_size=args.image_size)
        shape = WholeShape(SkateModel(geometry(inputs)[0]))
        nav = Navigation(inputs)
        prompts = read_anchors(inputs)
        priority = {str(inputs.shots[i]) for i, _ in prompts}
        ranges = sorted(inputs.ranges(), key=lambda bounds: str(inputs.shots[bounds[0]]) not in priority)
        for start, stop in ranges:
            name = str(inputs.shots[start])
            if args.shots and name not in args.shots:
                continue
            emit("shot_start", shot=name, first_pts=int(inputs.pts[start]), exposures=int(stop-start))
            result = shot(model, shape, inputs, nav, int(start), int(stop), region, status, extent, prompts)
            atomic_npz(output/"shots"/f"{name}.npz", run_identity=np.array(identity), **result)
    if args.merge or not args.shots:
        publish(inputs, output, args.prior, region, status, extent, identity)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--shots", nargs="+")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if not args.video and args.prior is None:
        parser.error("--prior is required for a tracking run")
    main(args)
