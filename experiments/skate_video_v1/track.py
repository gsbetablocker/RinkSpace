"""Two streaming SAM2 passes on exact native chips, followed by joint association."""

from argparse import ArgumentParser
from itertools import product
import time

import cv2
import numpy as np

from attachment import unit
from inputs import (BOOT, SIZE, WEIGHTS, Inputs, atomic_npz, digest,
                    direct_rows, distribute, emit, masked_distance, tracked_rows)

POINTS = 80


class Navigation:
    """Search coordinates and prompt hypotheses, with no crop-based position veto."""

    def __init__(self, inputs):
        self.inputs = inputs
        self.source_joints, self.source_direct = inputs.source_keypoints()
        order = [13, 15, 14, 16]
        self.joints = self.source_joints[:, order].reshape(-1, 2, 2, 2)
        self.direct = self.source_direct[:, order].reshape(-1, 2, 2)
        self.shin = self.joints[:, :, 0] - self.joints[:, :, 1]
        self.ankle = self.joints[:, :, 1]
        self.guide = inputs.old_gear[:, :, :14].copy()
        self.size = np.clip(np.linalg.norm(np.ptp(self.guide, axis=-2), axis=-1), 12, 180)
        correction = self.ankle - inputs.old_coco[:, [15, 16]]
        amount = np.linalg.norm(correction, axis=-1)
        delta = correction * np.minimum(1, self.size * 1.6 / np.maximum(amount, 1e-6))[..., None]
        self.guide += delta[:, :, None]
        self.centre = self.guide.mean(axis=-2)
        self.boot_guide = {}
        for key, rows in inputs.fit.items():
            direct = direct_rows(rows, BOOT)
            tracked = tracked_rows(rows, BOOT)
            if direct or tracked:
                self.boot_guide[key] = (direct or tracked, bool(direct))

    def crop(self, i):
        references = [self.guide[i].reshape(-1, 2), self.ankle[i]]
        for foot in range(2):
            row = self.boot_guide.get((i, foot))
            if row:
                references.extend(np.asarray(item["polyline_uv"]) for item in row[0])
        locations = np.vstack(references)
        locations = locations[np.isfinite(locations).all(axis=1)]
        inside = ((locations >= [-40, -40]) & (locations <= np.asarray(SIZE) + 40)).all(axis=1)
        locations = locations[inside] if inside.any() else np.array([[960., 880.]])
        margin = max(36., 1.22 * float(self.size[i].max()))
        bounds = np.stack((locations.min(axis=0) - margin, locations.max(axis=0) + margin))
        # Keep uniform pixel scale within the chip. A cropped frame is never treated
        # as proof that an out-of-chip skate is absent.
        centre = bounds.mean(axis=0)
        side = int(np.ceil(max(np.ptp(bounds, axis=0).max(), 128)))
        dims = np.minimum(SIZE, side).astype(int)
        start = np.clip(np.rint(centre - dims * .5).astype(int), 0, np.asarray(SIZE) - dims)
        return np.r_[start, start + dims]

    def prompt(self, i, foot, bounds):
        boot = self.guide[i, foot]
        ankle = self.ankle[i, foot]
        scale = self.size[i, foot]
        centre = boot.mean(axis=0)
        source, observed = self.boot_guide.get((i, foot), ([], False))
        off = bounds[:2]
        opposite = self.centre[i, 1 - foot]
        shank = unit(self.shin[i, foot])
        if source:
            points = np.concatenate([np.asarray(r["polyline_uv"]) for r in source])
            measured = (points.min(axis=0) + points.max(axis=0)) * .5
            if np.linalg.norm(measured - centre) > scale * .48:
                centre = measured
            pos = [centre]
            for row in source[:3]:
                arc = np.asarray(row["polyline_uv"])
                mid = arc[len(arc) // 2]
                # The input is a partial arc. Click just inside the boundary,
                # with the rest of the outline left for the video model.
                pos.append(mid * .78 + centre * .22)
            # Outsole-only anchors are useful boundaries but poor positive mask clicks.
            if all(r["semantic"] == "outsole" for r in source):
                pos = [p + shank * max(2., .12 * scale) for p in pos]
        else:
            if not self.direct[i, foot, 1]:
                return None
            toe = (boot[1] + boot[8]) * .5
            pos = [centre, centre * .66 + toe * .34]
        neg = [ankle + shank * max(8., scale * .30)]
        if np.linalg.norm(centre - opposite) > .7 * scale:
            neg.append(opposite)
        boundary = np.asarray(pos + neg)
        labels = np.r_[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
        take = ((boundary >= bounds[:2]) & (boundary < bounds[2:])).all(axis=1)
        if not np.any(take & (labels == 1)):
            return None
        return (boundary[take] - off).tolist(), labels[take].tolist(), 2 if observed else 1


class VideoMasks:
    def __init__(self, weights, *, image_size=None):
        import torch
        from transformers import Sam2VideoConfig, Sam2VideoModel, Sam2VideoProcessor

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("The installed vision_cuda CUDA runtime is required")
        self.processor = Sam2VideoProcessor.from_pretrained(weights, local_files_only=True)
        options = {}
        if image_size is not None:
            if image_size <= 0 or image_size % 64:
                raise ValueError("The video feature pyramid requires a multiple of 64")
            config = Sam2VideoConfig.from_pretrained(weights, local_files_only=True)
            config.image_size = config.prompt_encoder_config.image_size = image_size
            config.vision_config.backbone_feature_sizes = [[image_size // d] * 2 for d in (4, 8, 16)]
            config.memory_attention_rope_feat_sizes = [image_size // 16] * 2
            self.processor.target_size = image_size
            options["config"] = config
        self.model, report = Sam2VideoModel.from_pretrained(
            weights, local_files_only=True, output_loading_info=True, **options)
        failures = {k: report[k] for k in
                    ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if report.get(k)}
        if failures:
            raise RuntimeError(f"Local SAM2 load failed: {failures}")
        self.model.eval().requires_grad_(False).to("cuda", dtype=torch.bfloat16)
        self.mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
        self.horizon = max(self.model.config.num_maskmem,
                           self.model.config.max_object_pointers_in_encoder) + 2

    def reset(self):
        self.session = self.processor.init_video_session(
            inference_device="cuda", inference_state_device="cuda", video_storage_device="cpu",
            max_vision_features_cache_size=1, dtype=self.torch.bfloat16)

    def _prune(self, stamp):
        self.session.processed_frames = {stamp: self.session.processed_frames[stamp]}
        before = stamp - self.horizon
        for index, outputs in self.session.output_dict_per_obj.items():
            non_cond = outputs["non_cond_frame_outputs"]
            for k in list(non_cond):
                if k < before:
                    del non_cond[k]
            tracked = self.session.frames_tracked_per_obj[index]
            for k in list(tracked):
                if k < before:
                    del tracked[k]
            cond = outputs["cond_frame_outputs"]
            # Keep the initial visual signature and the recent conditioning frames.
            for k in sorted(cond)[1:-6]:
                del cond[k]
            for k in list(self.session.point_inputs_per_obj[index]):
                if k not in cond and k not in non_cond:
                    del self.session.point_inputs_per_obj[index][k]

    def step(self, chip, stamp, prompts, *, seed_masks=None):
        for foot, (xy, tags, _) in prompts.items():
            self.processor.add_inputs_to_inference_session(
                inference_session=self.session, frame_idx=stamp, obj_ids=foot,
                input_points=[[xy]], input_labels=[[tags]], original_size=chip.shape[:2])
        for foot, mask in (seed_masks or {}).items():
            self.processor.add_inputs_to_inference_session(
                inference_session=self.session, frame_idx=stamp, obj_ids=foot, input_masks=mask)
        self.session.obj_with_new_inputs = list(prompts) + list(seed_masks or {})
        if not self.session.get_obj_num():
            return {}
        t = self.torch
        with t.inference_mode(), t.autocast("cuda", dtype=t.bfloat16):
            rgb = t.from_numpy(np.ascontiguousarray(chip[:, :, ::-1])).to("cuda")
            rgb = rgb.permute(2, 0, 1).unsqueeze(0).float()
            rgb = t.nn.functional.interpolate(rgb, (self.model.config.image_size,) * 2,
                                              mode="bilinear", align_corners=False)
            frame = ((rgb / 255. - self.mean) / self.std).to(t.bfloat16)
            result = self.model(inference_session=self.session, frame=frame, frame_idx=stamp)
            masks = t.nn.functional.interpolate(result.pred_masks.float(), chip.shape[:2],
                                                mode="bilinear", align_corners=False)
            masks = (masks[:, 0] > 0).to(t.uint8).cpu().numpy()
            logits = result.object_score_logits.float().cpu().numpy().ravel()
        self._prune(stamp)
        return dict(zip(result.object_ids, zip(masks, logits)))


def measure(mask, logit, bounds, expected_size):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or logit < -.8:
        return None
    main = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main)
    rect = cv2.boundingRect(main)
    length = np.hypot(rect[2], rect[3])
    if len(main) < 4 or area < 13 or length < 7 or length > expected_size * 2.7:
        return None
    if area / (rect[2] * rect[3]) < .17:
        return None
    # Frame boundaries may represent physical crop. Keep the visible component
    # and flag it, but refuse masks cut at the experiment's own ROI border.
    touches = np.array([rect[0] <= 1, rect[1] <= 1,
                        rect[0] + rect[2] >= mask.shape[1] - 1,
                        rect[1] + rect[3] >= mask.shape[0] - 1])
    camera_border = bounds == [0, 0, *SIZE]
    if np.any(touches & ~camera_border):
        return None
    curve = distribute(main[:, 0] + bounds[:2], POINTS, closed=True)
    n_components = sum(cv2.contourArea(c) > area * .13 for c in contours)
    score = 1.2 + .18 * np.clip(logit, 0, 6)
    score -= .65 * abs(np.log(max(length / expected_size, .03)))
    score -= .36 * (n_components > 1) + .32 * touches.any()
    return curve, float(score), bool(touches.any() or n_components > 1)


def direction(model, inputs, nav, chips, bounds, indices, *, reverse=False, stride=1):
    model.reset()
    curves = np.full((len(chips), 2, POINTS, 2), np.nan, np.float32)
    scores = np.full((len(chips), 2), -np.inf, np.float32)
    prompt_source = np.zeros((len(chips), 2), np.uint8)
    clipped = np.zeros_like(prompt_source, dtype=bool)
    last = np.full(2, -100)
    last_good = np.full(2, -100)
    order = range(len(chips) - 1, -1, -1) if reverse else range(len(chips))
    counter = 0
    for step, offset in enumerate(order):
        i = indices[offset]
        direct = {f for f in range(2) if nav.boot_guide.get((i, f), (None, False))[1]}
        if step % stride and step < len(chips) - 1 and not direct:
            continue
        prompts = {}
        if not inputs.withheld[i]:
            for f in range(2):
                needs_refresh = (step - last[f] >= 36 or
                                 (step - last_good[f] >= 4 and step - last[f] >= 4))
                source = nav.boot_guide.get((i, f))
                if f in direct or needs_refresh or (source and step - last[f] >= 12):
                    prompt = nav.prompt(i, f, bounds[offset])
                    if prompt is not None:
                        prompts[f] = prompt
                        last[f] = step
        results = model.step(chips[offset], counter, prompts)
        counter += 1
        for f, (mask, logit) in results.items():
            result = measure(mask, logit, bounds[offset], nav.size[i, f])
            if result is None:
                continue
            polygon, score, partial = result
            curves[offset, f], scores[offset, f], clipped[offset, f] = polygon, score, partial
            if f in prompts:
                prompt_source[offset, f] = prompts[f][2]
            last_good[f] = step
        if counter % 120 == 0:
            emit("sam2", direction="reverse" if reverse else "forward",
                 pts=int(inputs.pts[i]), decoded_steps=step + 1, network_exposures=counter,
                 masked_feet=int(np.isfinite(scores).sum()))
    return curves, scores, prompt_source, clipped


def bridge(chips, bounds, result):
    """Fill skipped network exposures only with consistent native optical flow."""
    curves, scores, prompts, clipped = result
    for i in range(1, len(chips) - 1):
        if np.isfinite(scores[i]).any():
            continue
        pair = [i - 1, i + 1]
        if not all(np.isfinite(scores[k]).any() for k in pair):
            continue
        a = np.min(bounds[[i - 1, i, i + 1], :2], axis=0)
        b = np.max(bounds[[i - 1, i, i + 1], 2:], axis=0)
        shape = tuple((b - a)[::-1])
        gray = []
        for j in [i - 1, i, i + 1]:
            canvas = np.zeros(shape, np.uint8)
            x, y, w, h = bounds[j] - np.tile(a, 2)
            canvas[y:h, x:w] = cv2.cvtColor(chips[j], cv2.COLOR_BGR2GRAY)
            gray.append(canvas)
        for f in range(2):
            if not np.isfinite(scores[pair, f]).all():
                continue
            ends = []
            for k, g in zip(pair, (gray[0], gray[2])):
                uv = (curves[k, f] - a).astype(np.float32)
                next_uv, ok, _ = cv2.calcOpticalFlowPyrLK(
                    g, gray[1], uv[:, None], None, winSize=(19, 19), maxLevel=2)
                if next_uv is None or ok.mean() < .6:
                    break
                transform, inliers = cv2.estimateAffinePartial2D(
                    uv, next_uv[:, 0], method=cv2.RANSAC, ransacReprojThreshold=2.5)
                if transform is None or inliers.mean() < .5:
                    break
                ends.append((uv @ transform[:, :2].T + transform[:, 2]) + a)
            if len(ends) != 2:
                continue
            error = np.median(masked_distance(ends[0], ends[1], closed=True))
            if error > 4.:
                continue
            curves[i, f] = ends[0]
            scores[i, f] = min(scores[pair, f]) - .16 - .12 * error
            clipped[i, f] = clipped[pair, f].any()


def reacquire(model, inputs, nav, chips, bounds, indices, passes):
    """Read lost but uncropped feet again with independent source navigation."""
    curves = np.full_like(passes[0][0], np.nan)
    scores = np.full_like(passes[0][1], -np.inf)
    source = np.zeros_like(passes[0][2])
    clipped = np.zeros_like(passes[0][3])
    lost = ~np.logical_or.reduce([np.isfinite(p[1]) for p in passes])
    for k, i in enumerate(indices):
        if inputs.withheld[i]:
            continue
        prompts = {}
        for f in np.flatnonzero(lost[k]):
            if not nav.direct[i, f, 1]:
                continue
            prompt = nav.prompt(i, f, bounds[k])
            if prompt is None:
                continue
            xy, tags, status = prompt
            if status == 1 and not nav.boot_guide.get((i, f)):
                # Use a single positive when a bad toe guess may conflict with it.
                keep = [0] + [j for j, tag in enumerate(tags) if tag == 0]
                xy, tags = [xy[j] for j in keep], [tags[j] for j in keep]
            prompts[int(f)] = (xy, tags, status)
        if not prompts:
            continue
        model.reset()
        for f, (mask, logit) in model.step(chips[k], 0, prompts).items():
            item = measure(mask, logit, bounds[k], nav.size[i, f])
            if item is None:
                continue
            curves[k, f], score, clipped[k, f] = item
            scores[k, f] = score - .23
            source[k, f] = prompts[f][2]
    return curves, scores, source, clipped


def mask_overlap(a, b):
    origin = np.minimum(a.min(axis=0), b.min(axis=0))
    top = np.maximum(a.max(axis=0), b.max(axis=0))
    scale = min(1., 128. / max(float(np.ptp([origin, top], axis=0).max()), 1.))
    dims = np.ceil((top - origin) * scale).astype(int) + 3
    layers = np.zeros((2, dims[1], dims[0]), np.uint8)
    for j, polygon in enumerate((a, b)):
        cv2.fillPoly(layers[j], [np.rint((polygon - origin) * scale).astype(np.int32)], 1)
    area = layers.sum(axis=(1, 2))
    common = (layers[0] & layers[1]).sum()
    return float(common / max(area.sum() - common, 1)), float(common / max(area.min(), 1))


def iou(a, b):
    return mask_overlap(a, b)[0]


def flow_candidates(previous, current, bounds, polygons, valid):
    """Measure native two-way pixel motion on the shoe candidates themselves."""
    result = np.full((len(polygons), 2), np.nan, np.float32)
    active = np.flatnonzero(valid)
    if not len(active):
        return result
    origin = bounds[:, :2].min(axis=0)
    extent = bounds[:, 2:].max(axis=0) - origin
    images = []
    for gray, bound in zip((previous, current), bounds, strict=True):
        canvas = np.zeros(tuple(extent[::-1]), np.uint8)
        x, y, r, b = bound - np.tile(origin, 2)
        canvas[y:b, x:r] = gray
        images.append(canvas)
    p = polygons[active]
    xy = (.78 * p[:, ::8] + .22 * p.mean(axis=1, keepdims=True) - origin).astype(np.float32)
    flat = xy.reshape(-1, 1, 2)
    forward, ok, _ = cv2.calcOpticalFlowPyrLK(*images, flat, None, winSize=(19, 19), maxLevel=2)
    if forward is None:
        return result
    backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(*images[::-1], forward, None, winSize=(19, 19), maxLevel=2)
    if backward is None:
        return result
    good = (ok.ravel() & back_ok.ravel()).astype(bool) & (np.linalg.norm(backward - flat, axis=-1).ravel() <= 1.5)
    new = forward.reshape(xy.shape)
    current_box = bounds[1] - np.tile(origin, 2)
    inside = ((new >= current_box[:2] + 1) & (new < current_box[2:] - 1)).all(axis=-1)
    for j, (flow, keep) in enumerate(zip(new - xy, good.reshape(xy.shape[:2]) & inside, strict=True)):
        if keep.sum() < 5:
            continue
        median = np.median(flow[keep], axis=0)
        if np.median(np.linalg.norm(flow[keep] - median, axis=-1)) <= 3.:
            result[active[j]] = median
    return result


def boot_near_leg(nav, i, poly):
    """Test foot location without trusting the detector's left/right labels."""
    scale = np.linalg.norm(np.ptp(poly, axis=0))
    legs = nav.ankle[i, nav.direct[i, :, 1]]
    size = min(scale, nav.size[i].max())
    if any(cv2.pointPolygonTest(poly, tuple(uv), True) >= -max(7., .60 * size) for uv in legs):
        return True
    # A detector can miss the entire raised leg. A close fitted image ankle
    # is a weak localization cue; it supplies neither depth nor identity.
    seeds = nav.inputs.old_coco[i, [15, 16]]
    return any(np.isfinite(uv).all() and cv2.pointPolygonTest(poly, tuple(uv), True) >= -max(4., .16 * size)
               for uv in seeds)


def covered_body_components(nav, indices):
    """Sample measured proximal limbs, away from ankle cuffs and hands."""
    links = np.array([(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (12, 14), (15, 13), (16, 14)])
    limits = np.array([[.14, .72]] * 6 + [[.52, .90]] * 2)
    weight = np.linspace(limits[:, 0], limits[:, 1], 5, axis=1)[None, :, :, None]
    ends = nav.source_joints[indices][:, links]
    measured = nav.source_direct[indices][:, links].all(axis=-1)
    points = ends[:, :, :1] * (1. - weight) + ends[:, :, 1:] * weight
    span = np.linalg.norm(points[:, :, -1] - points[:, :, 0], axis=-1)
    return points, np.where(measured, span, 0.)


def covers_component(poly, points, span, min_span):
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    possible = (span > min_span) & (((points >= lo) & (points <= hi)).all(axis=-1).sum(axis=-1) >= 4)
    return any(sum(cv2.pointPolygonTest(poly, tuple(uv), True) >= 1. for uv in points[j]) >= 4
               for j in np.flatnonzero(possible))


def supported_boot_candidates(nav, indices, curves, valid, motion):
    """Carry local source support through short keypoint failures."""
    anchor = np.zeros_like(valid)
    usable = valid.copy()
    rectangular = np.zeros_like(valid)
    torso = nav.inputs.old_coco[indices][:, [5, 6, 11, 12]].astype(np.float32)
    centre = torso.mean(axis=1, keepdims=True)
    torso_sites = np.concatenate((.65 * torso + .35 * centre, centre), axis=1)
    torso_span = np.linalg.norm(np.ptp(torso, axis=1), axis=-1)
    size = np.linalg.norm(np.ptp(curves, axis=2), axis=-1)
    limb_points, limb_span = covered_body_components(nav, indices)
    fitted = nav.inputs.old_coco[indices]
    fitted_legs = fitted[:, [(15, 13), (16, 14), (13, 11), (14, 12)]]
    fraction = np.stack((np.linspace(.28, .64, 5), np.linspace(.28, .64, 5),
                         np.linspace(.20, .56, 5), np.linspace(.20, .56, 5)))[None, :, :, None]
    fitted_points = (1. - fraction) * fitted_legs[:, :, :1] + fraction * fitted_legs[:, :, 1:]
    fitted_span = np.linalg.norm(fitted_points[:, :, -1] - fitted_points[:, :, 0], axis=-1)
    for k, c in zip(*np.where(valid)):
        i = indices[k]
        poly = curves[k, c]
        # A mask enclosing most of the projected trunk is a body component.
        # Small raised skates can still pass in front of the trunk.
        covers_torso = (np.isfinite(torso_span[k]) and size[k, c] > .70 * torso_span[k]
                        and sum(cv2.pointPolygonTest(poly, tuple(uv), False) >= 0 for uv in torso_sites[k]) >= 4)
        core_length = .35 * size[k, c]
        if size[k, c] > 1.4 * nav.size[i].max():
            core_length = min(core_length, .24 * nav.size[i].max())
        covers_limb = covers_component(poly, limb_points[k], limb_span[k], max(12., core_length))
        ankles = fitted[k, [15, 16]]
        fitted_gap = -np.array([cv2.pointPolygonTest(poly, tuple(uv), True) if np.isfinite(uv).all() else -np.inf
                                for uv in ankles])
        # When one measured foot is missing or both detections follow the
        # grounded foot, an oversized mask can swallow the raised boot and
        # calf. The image-space fit supplies a conservative exclusion cue.
        fused_calf = (fitted_gap <= 1.) & (size[k, c] >= 1.22 * nav.size[i])
        calf_span = np.where(fused_calf, fitted_span[k, :2], 0.)
        covers_limb |= covers_component(poly, fitted_points[k, :2], calf_span, 15.)
        observed = nav.ankle[i, nav.direct[i, :, 1]]
        observed_gap = min((-cv2.pointPolygonTest(poly, tuple(uv), True) for uv in observed), default=np.inf)
        ankle_gap = min(fitted_gap.min(), observed_gap)
        if ankle_gap > max(7., .13 * size[k, c]):
            covers_limb |= covers_component(poly, fitted_points[k, 2:], fitted_span[k, 2:], 16.)
        w, h = cv2.minAreaRect(poly)[1]
        area = cv2.contourArea(poly)
        rectangular[k, c] = area >= .945 * w * h
        if not rectangular[k, c] and area >= .88 * w * h and ankle_gap > max(9., .20 * size[k, c]):
            hull = cv2.convexHull(poly)
            rectangular[k, c] = len(cv2.approxPolyDP(hull, .025 * cv2.arcLength(hull, True), True)) == 4
        conflict = covers_torso or covers_limb or rectangular[k, c]
        # A shoe can overlap the hand at the grip. A verified source boot arc
        # can disambiguate an overlap, provided the mask follows that arc.
        confirmed = any(guide[1] and np.quantile(masked_distance(
            np.vstack([row["polyline_uv"] for row in guide[0]]), poly, closed=True), .75) <= 4.
            for f in range(2) if (guide := nav.boot_guide.get((i, f))) is not None)
        located = (not conflict and boot_near_leg(nav, i, poly)) or (confirmed and not covers_torso and not covers_limb)
        anchor[k, c] = located
        usable[k, c] = not conflict or located
    centres = curves.mean(axis=2)
    dist = np.linalg.norm(centres[1:, None] - (centres[:-1] + motion)[:, :, None], axis=-1)
    scale = np.minimum(size[:-1, :, None], size[1:, None])
    links = usable[:-1, :, None] & usable[1:, None] & (dist <= np.maximum(4., .17 * scale))
    links &= np.maximum(size[:-1, :, None], size[1:, None]) <= 1.47 * scale
    links &= (~rectangular[:-1, :, None] | anchor[:-1, :, None]) & (~rectangular[1:, None] | anchor[1:, None])
    # Full-mask agreement is needed only on links that transport support.
    links_needing_check = links & ~(anchor[:-1, :, None] & anchor[1:, None])
    for k, c, d in zip(*np.where(links_needing_check)):
        links[k, c, d] = iou(curves[k, c] + motion[k, c], curves[k+1, d]) >= .44
    past = np.full_like(valid, 7, dtype=np.int8)
    future = past.copy()
    past[anchor] = future[anchor] = 0
    for k in range(1, len(valid)):
        past[k] = np.minimum(past[k], np.where(links[k-1], past[k-1, :, None] + 1, 7).min(axis=0))
    for k in range(len(valid)-2, -1, -1):
        future[k] = np.minimum(future[k], np.where(links[k], future[k+1, None] + 1, 7).min(axis=1))
    return anchor | (usable & ((past <= 2) | (future <= 2) | ((past <= 6) & (future <= 6))))


def mask_motion_conflict(previous, current, flow, old_valid, new_valid):
    """Check that transported pixels still land on the next boot instance."""
    left = np.flatnonzero(old_valid & np.isfinite(flow).all(axis=-1))
    right = np.flatnonzero(new_valid)
    translated = previous[left] + flow[left, None]
    centre_error = np.linalg.norm(translated.mean(axis=1)[:, None] - current[right].mean(axis=1)[None], axis=-1)
    old_size = np.linalg.norm(np.ptp(previous[left], axis=1), axis=-1)
    new_size = np.linalg.norm(np.ptp(current[right], axis=1), axis=-1)
    conflict = np.zeros((len(previous), len(current)), bool)
    for a, b in zip(*np.where(centre_error > np.maximum(5., .18 * np.minimum(old_size[:, None], new_size[None])))):
        conflict[left[a], right[b]] = mask_overlap(translated[a], current[right[b]])[1] < .20
    return conflict


def associate(inputs, nav, indices, curves, scores, prompt_source, clipped, *, source_motion=None, repair_slots=None):
    """Choose among joint two-boot assignments with whole-shot backtracking."""
    n = len(indices)
    # A state can assign either direction's instance (even if its SAM ID is
    # swapped), or explicitly leave a foot without a visible polygon.
    miss = scores.shape[1]
    states = np.array([(a, b) for a, b in product(range(miss + 1), repeat=2) if a != b or a == miss])
    valid = np.isfinite(scores)
    if source_motion is not None:
        valid = supported_boot_candidates(nav, indices, curves, valid, source_motion)
    centres = np.zeros((n, miss + 1, 2), dtype=np.float32)
    centres[:, :miss] = np.nan_to_num(curves.mean(axis=-2), nan=0.)
    ids = states % 2
    missing = states == miss
    unary = np.zeros((n, len(states)))
    for k, i in enumerate(indices):
        individual = np.zeros((2, miss + 1))
        individual[:, miss] = .72
        individual[:, :miss] = np.inf
        for c in np.flatnonzero(valid[k]):
            other_pass = [d for d in range(min(miss, 4)) if d // 2 != c // 2 and valid[k, d]]
            agreement = max((iou(curves[k, c], curves[k, d]) for d in other_pass), default=0.)
            for f in range(2):
                if repair_slots is not None and c in repair_slots and c != repair_slots[f]:
                    continue
                size = nav.size[i, f]
                distance = np.linalg.norm(centres[k, c] - nav.centre[i, f]) / max(size, 12.)
                if source_motion is not None:
                    fitted_ankle = inputs.old_coco[i, 15 + f]
                    if np.isfinite(fitted_ankle).all():
                        seed_gap = max(0., -cv2.pointPolygonTest(curves[k, c], tuple(fitted_ankle), True))
                        distance = min(distance, .35 + seed_gap / max(size, 12.))
                cost = 1.47 - scores[k, c] + .51 * min(distance, 4.) + .27 * (c % 2 != f)
                cost -= .45 * agreement
                guide = nav.boot_guide.get((i, f))
                if guide:
                    pts = np.vstack([r["polyline_uv"] for r in guide[0]])
                    delta = np.median(masked_distance(pts, curves[k, c], closed=True))
                    cost += min(delta / 8., 2.5) * (.8 if guide[1] else .32)
                individual[f, c] = cost
        unary[k] = individual[0, states[:, 0]] + individual[1, states[:, 1]]
        for pair in {(int(a), int(b)) for a, b in states if a < miss and b < miss and valid[k, a] and valid[k, b]}:
            overlap, shared = mask_overlap(curves[k, pair[0]], curves[k, pair[1]])
            select = (states == pair).all(axis=1)
            if source_motion is not None and shared > .68:
                unary[k, select] = np.inf
            else:
                unary[k, select] += 8. if overlap > .57 else 1.15 * overlap
    back = np.empty((n, len(states)), np.int16)
    fw = np.empty_like(unary)
    fw[0] = unary[0]
    transitions = np.empty((n - 1, len(states), len(states)), np.float32)
    for k in range(1, n):
        i0, i1 = indices[k - 1:k + 1]
        trans = np.zeros((len(states), len(states)))
        forbidden = np.zeros(trans.shape, bool)
        if source_motion is not None:
            mismatch = mask_motion_conflict(curves[k-1], curves[k], source_motion[k-1], valid[k-1], valid[k])
            mismatch = np.pad(mismatch, ((0, 1), (0, 1)))
        for f in range(2):
            has = ~missing[:, f]
            u = centres[k - 1, states[:, f]]
            v = centres[k, states[:, f]]
            prediction = nav.ankle[i1, f] - nav.ankle[i0, f]
            drift = np.linalg.norm(v[None] - u[:, None] - prediction, axis=-1)
            scale = max(5., .21 * nav.size[i1, f])
            if source_motion is not None:
                flow = source_motion[k - 1, np.minimum(states[:, f], miss - 1)]
                reliable = np.isfinite(flow).all(axis=-1)
                if not (nav.direct[i0, f, 1] and nav.direct[i1, f, 1]) or np.linalg.norm(prediction) > 1.25 * nav.size[i1, f]:
                    # A dropped detector foot can reappear on the other leg.
                    # Use the fitted image motion as the weak fallback only.
                    prediction = inputs.old_coco[i1, 15+f] - inputs.old_coco[i0, 15+f]
                motion = np.where(reliable[:, None], flow, prediction)
                drift = np.linalg.norm(v[None] - u[:, None] - motion[:, None], axis=-1)
                # Pixel flow transports the shoe through real image motion.
                # Detector displacement is a weaker fallback during blur.
                limit = np.where(reliable, max(9., .41 * nav.size[i1, f]),
                                 max(15., .68 * nav.size[i1, f]))
                supported = reliable | np.isfinite(prediction).all()
                displacement = np.linalg.norm(v[None] - u[:, None], axis=-1)
                drift = np.where(supported[:, None], drift, displacement)
                unconfirmed_jump = (~reliable[:, None]) & (displacement > max(14., .55 * nav.size[i1, f]))
                forbidden |= has[:, None] & has[None] & ((supported[:, None] & (drift > limit[:, None])) | unconfirmed_jump)
                forbidden |= mismatch[states[:, f, None], states[None, :, f]]
            trans += np.where(has[:, None] & has[None], .44 * np.minimum(drift / scale, 5.), 0.)
            trans += .26 * (has[:, None] != has[None])
            trans += .09 * (has[:, None] & has[None] & (ids[:, f, None] != ids[None, :, f]))
        trans[forbidden] = np.inf
        transitions[k - 1] = trans
        cost = fw[k - 1, :, None] + trans
        back[k] = cost.argmin(axis=0)
        fw[k] = unary[k] + cost.min(axis=0)
    path = np.empty(n, int)
    path[-1] = fw[-1].argmin()
    for k in range(n - 2, -1, -1):
        path[k] = back[k + 1, path[k + 1]]
    remaining = np.zeros(len(states))
    status = np.zeros((n, 2), np.uint8)
    output = np.full((n, 2, POINTS, 2), np.nan, np.float32)
    alternatives = output.copy()
    confidence = np.zeros((n, 2), np.float32)
    trim = np.zeros((n, 2), bool)
    overlap = np.zeros(n, bool)
    for k in range(n - 1, -1, -1):
        total = fw[k] + remaining
        chosen = states[path[k]]
        if (chosen < miss).all():
            overlap[k] = iou(curves[k, chosen[0]], curves[k, chosen[1]]) > .12
        for f, c in enumerate(chosen):
            if c == miss:
                continue
            output[k, f], trim[k, f] = curves[k, c], clipped[k, c]
            status[k, f] = 1
            confidence[k, f] = scores[k, c]
            # Mark ambiguity only if the alternative refers to different pixels,
            # not merely the same boot from the other propagation direction.
            other = [a for a in range(miss) if valid[k, a] and a != c
                     and iou(curves[k, a], curves[k, c]) < .45]
            if other:
                allowed = np.flatnonzero(np.isin(states[:, f], other))
                best = allowed[total[allowed].argmin()]
                if total[best] - total[path[k]] < .56:
                    alternatives[k, f] = curves[k, states[best, f]]
                    status[k, f] = 3
            # Status 'observed' applies only to the direct source arc, never the mask.
        if k:
            remaining = (transitions[k - 1] + unary[k] + remaining).min(axis=1)
    return dict(boot=output, boot_alt=alternatives, boot_status=status, boot_confidence=confidence,
                roi_partial=trim, overlap=overlap, chosen=states[path], direction_boot=curves,
                direction_scores=scores, prompt_source=prompt_source)


def reassociate(inputs, nav, packet):
    """Reuse saved SAM and optical-recovery candidates on the native clock."""
    a = inputs.index[int(packet["source_pts"][0])]
    b = inputs.index[int(packet["source_pts"][-1])] + 1
    np.testing.assert_array_equal(packet["source_pts"], inputs.pts[a:b])
    boot = packet["boot"]
    curves = packet["direction_boot"]
    scores = packet["direction_scores"]
    prompts = packet["prompt_source"]
    original_count = curves.shape[1]
    repaired = packet.get("repair_method", np.zeros(boot.shape[:2], np.uint8)) > 0
    curves = np.concatenate((curves, np.where(repaired[..., None, None], boot, np.nan)), axis=1)
    scores = np.concatenate((scores, np.where(repaired, packet["boot_confidence"], -np.inf)), axis=1)
    prompts = np.concatenate((prompts, np.zeros(repaired.shape, np.uint8)), axis=1)
    valid = np.isfinite(scores)
    clipped = ((curves <= 1.) | (curves >= np.asarray(SIZE) - 2.)).any(axis=(2, 3))
    for k, f in zip(*np.where((packet["chosen"] < original_count) & packet["roi_partial"])):
        clipped[k, packet["chosen"][k, f]] = True
    bounds = packet["chip_bounds"]
    flow = np.full((len(boot) - 1, curves.shape[1], 2), np.nan, np.float32)
    previous = None
    for k, ((_, frame), (x, y, r, bottom)) in enumerate(zip(inputs.frames(a, b), bounds, strict=True)):
        gray = cv2.cvtColor(frame[y:bottom, x:r], cv2.COLOR_BGR2GRAY)
        if previous is not None:
            flow[k - 1] = flow_candidates(previous, gray, bounds[k-1:k+1], curves[k-1], valid[k-1])
        previous = gray
    slots = np.arange(original_count, original_count + 2)
    result = associate(inputs, nav, np.arange(a, b), curves, scores, prompts, clipped,
                       source_motion=flow, repair_slots=slots)
    keep_repair = result["chosen"] == slots[None]
    if "repair_method" in packet:
        packet["repair_method"] = np.where(keep_repair, packet["repair_method"], 0)
        packet["repair_anchor_pts"] = np.where(keep_repair[..., None], packet["repair_anchor_pts"], -1)
    selected = np.minimum(result["chosen"][:-1], curves.shape[1] - 1)
    selected_flow = np.full((*boot.shape[:2], 2), np.nan, np.float32)
    selected_flow[1:] = flow[np.arange(len(flow))[:, None], selected]
    selected_flow[1:] = np.where(result["boot_status"][:-1, :, None] > 0, selected_flow[1:], np.nan)
    result["boot_flow_from_previous"] = selected_flow
    packet.update(result)
    return packet


def run(args):
    cv2.setNumThreads(1)
    inputs = Inputs(verify_media=True)
    nav = Navigation(inputs)
    model = VideoMasks(WEIGHTS)
    model_hash = digest(WEIGHTS / "model.safetensors")
    emit("binding", frames=len(inputs.pts), fit_rows=sum(map(len, inputs.fit.values())),
         check_rows=len(inputs.checks), protected_exposures=int(inputs.withheld.sum()), model_sha256=model_hash)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for a, b in inputs.ranges():
        shot = str(inputs.shots[a])
        if args.shot and shot not in args.shot:
            continue
        start, stop = a, b
        if args.pilot:
            moments = dict(s01=26.02, s02=48.88, s03=56.2, s04=66.8, s05=80.1, s06=110.6,
                           s07=128.6, s08=140.92, s09=161.38)
            centre = np.searchsorted(inputs.times, moments[shot])
            start = int(np.clip(centre - args.pilot // 2, a, max(a, b - args.pilot)))
            stop = min(b, start + args.pilot)
        part = out / f"track_{shot}.npz"
        if part.exists():
            with np.load(part, allow_pickle=False) as f:
                np.testing.assert_array_equal(f["source_pts"], inputs.pts[start:stop])
                if str(f["sam2_sha256"]) != model_hash or int(f["sam_stride"]) != args.stride:
                    raise ValueError("Use a new run directory for a different SAM2 configuration")
            parts.append(part)
            continue
        begin = time.perf_counter()
        frames = np.arange(start, stop)
        bounds = np.stack([nav.crop(i) for i in frames])
        chips = []
        for (_, frame), (x0, y0, x1, y1) in zip(inputs.frames(start, stop), bounds, strict=True):
            chips.append(np.ascontiguousarray(frame[y0:y1, x0:x1]))
        emit("decoded", shot=shot, start_pts=int(inputs.pts[start]), stop_pts=int(inputs.pts[stop - 1]),
             frames=len(chips), cpu_chip_mib=round(sum(c.nbytes for c in chips) / 2**20, 1))
        predictions = []
        for reverse in (False, True):
            result = direction(model, inputs, nav, chips, bounds, frames, reverse=reverse, stride=args.stride)
            if args.stride == 2:
                bridge(chips, bounds, result)
            predictions.append(result)
        predictions.append(reacquire(model, inputs, nav, chips, bounds, frames, predictions))
        curves, scores, prompt_source, clip = [
            np.concatenate([r[k] for r in predictions], axis=1) for k in range(4)]
        data = associate(inputs, nav, frames, curves, scores, prompt_source, clip)
        # Original image to chip is exactly the saved translation. SAM internally
        # scales this chip to its configured image domain and back.
        affine = np.broadcast_to(np.eye(3), (len(frames), 3, 3)).copy()
        affine[:, :2, 2] = -bounds[:, :2]
        inverse = affine.copy()
        inverse[:, :2, 2] *= -1.
        atomic_npz(part, source_pts=inputs.pts[frames], shot_ids=inputs.shots[frames],
                   source_to_chip=affine, chip_to_source=inverse, chip_bounds=bounds,
                   sam2_sha256=np.array(model_hash), sam_stride=np.array(args.stride),
                   **data)
        parts.append(part)
        emit("associated", shot=shot, seconds=round(time.perf_counter() - begin, 2),
             feet=int((data["boot_status"] > 0).sum()), ambiguous=int((data["boot_status"] == 3).sum()),
             native_exposures=len(frames), file=str(part))
    if not parts:
        raise ValueError("No requested programme shots")


if __name__ == "__main__":
    from pathlib import Path

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--stride", type=int, default=2, choices=(1, 2))
    parser.add_argument("--shot", nargs="+")
    run(parser.parse_args())
