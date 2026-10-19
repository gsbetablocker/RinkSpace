"""Typed image observations; board texture is never relabelled as ice points.

Tracking is image-only pyramidal LK with a backward check. No old similarity
transform, rink pose, or proposed camera model selects these correspondences.
All exported coordinates are original decoded image pixels. Mask provenance,
PTS and the resampling used by LK are explicit in the extraction manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class SceneTrack:
    id: int
    xy: np.ndarray
    birth_xy: np.ndarray
    birth_index: int
    age: int = 0


def foreground_mask(shape, polygons, sample_size, image=None, diagnostics=None, boundary_exclusion=None, context_boxes=None):
    result = np.zeros(shape[:2], np.uint8)
    sx, sy = shape[1] / sample_size[0], shape[0] / sample_size[1]
    candidates = []
    for p in polygons:
        q = np.asarray(p, float) * [sx, sy]
        if len(q) >= 3:
            cv2.fillPoly(result, [np.rint(q).astype(np.int32)], 255)
            lo, hi = q.min(axis=0), q.max(axis=0)
            # Screen timer/score are not the athlete seed.
            if (hi[1] - lo[1] > shape[0] * .06 and not
                    (lo[0] > shape[1] * .80 and lo[1] > shape[0] * .80)):
                candidates.append((cv2.contourArea(q.astype(np.float32)), lo, hi))
    if diagnostics is not None:
        diagnostics.update({'old_person_seed_count': len(candidates), 'source': 'current_image_and_existing_foreground', 'full_body_box_is_uncertainty_exclusion': True})
    if context_boxes:
        for context in context_boxes:
            x0, y0, x1, y1 = np.rint(context['exclusion_box_source_px']).astype(int)
            result[max(0, y0):min(shape[0], y1), max(0, x0):min(shape[1], x1)] = 255
        if diagnostics is not None:
            diagnostics['same_shot_temporal_foreground_exclusions'] = context_boxes
            diagnostics['temporal_exclusion_is_observation'] = False
    if image is not None and not candidates and not context_boxes:
        # The old plane mask may have no athlete at all. OpenCV's bundled HOG
        # detector supplies current-image proposals; no external model or
        # unrelated-project person annotations are loaded. Include every
        # proposal conservatively, so false positives remove evidence rather
        # than manufacture stationary geometry.
        scale = min(1., 960. / shape[1])
        small = cv2.resize(image, None, fx=scale, fy=scale)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, scores = hog.detectMultiScale(small, winStride=(4, 4), padding=(8, 8), scale=1.04)
        if diagnostics is not None:
            diagnostics['current_image_HOG_candidates'] = [{'box_source_px': (np.asarray(box, float) / scale).tolist(), 'score': float(score)} for box, score in zip(boxes, scores)]
            diagnostics['HOG_model'] = 'OpenCV_builtin_default_people_detector'
        for x, y, width, height in boxes:
            x, y, width, height = np.array([x, y, width, height], float) / scale
            # Raised arms and crouches differ from pedestrian training poses.
            # The deliberately enlarged region is not a recovered silhouette.
            x0, x1 = max(0, int(x - width * .65)), min(shape[1], int(x + width * 1.65))
            y0, y1 = max(0, int(y - height * .50)), min(shape[0], int(y + height * 1.20))
            result[y0:y1, x0:x1] = 255
    if image is not None and candidates:
        _, lo, hi = max(candidates, key=lambda item: item[0])
        height = hi[1] - lo[1]
        # Old on-ice masks can start at the waist. Extend their actual person
        # seed to a full-body uncertainty region; do not classify its upper
        # body as a board simply because it lies above the ice boundary.
        rise = max(shape[0] * .12, min(shape[0] * .45, height * 1.5))
        pad = min(shape[1] * .18, max(shape[1] * .04, height * .30))
        x0, x1 = max(0, int(lo[0] - pad)), min(shape[1], int(hi[0] + pad + 1))
        y0, y1 = max(0, int(lo[1] - rise)), min(shape[0], int(hi[1] + .04 * shape[0] + 1))
        # Actual dark/skin components linked to old foreground can extend the
        # bounds further. This is an exclusion candidate, never a silhouette
        # truth, foot measurement, or positive geometric observation.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, sat, value = cv2.split(hsv)
        coloured = ((value < 135) | ((hue <= 35) & (sat > 30) & (value < 240))).astype(np.uint8)
        coloured[:y0] = 0; coloured[y1:] = 0
        coloured[:, :x0] = 0; coloured[:, x1:] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(coloured)
        overlaps = np.bincount(labels[result > 0].ravel(), minlength=count)
        for k in range(1, count):
            if stats[k, cv2.CC_STAT_AREA] < 60:
                continue
            if overlaps[k] >= 30:
                result[labels == k] = 255
        # The conservative full-body box also excludes pale clothing/arms
        # not separable by colour. Its assumed expansion is explicitly logged.
        result[y0:y1, x0:x1] = 255
        if diagnostics is not None:
            diagnostics['expanded_old_seed_box_source_px'] = [x0, y0, x1, y1]
    if boundary_exclusion is not None:
        # Colour-only skin candidates can include a yellow board strip. Do
        # not let this extra feature exclusion move or erase its measurement.
        boundary_exclusion['mask'] = cv2.dilate(result, np.ones((15, 15), np.uint8))
    if image is not None:
        # Close-ups with arms outside every pedestrian/leg box need an
        # image-supported exclusion too. Broad connected warm skin candidates
        # only REMOVE evidence. Reject very slender colour lines; never use
        # this colour test to assert that remaining pixels are static.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        b, g, r = [a.astype(np.int16) for a in cv2.split(image)]
        skin = ((hue <= 16) & (saturation >= 18) & (saturation <= 175) &
                (value < 245) & (r - g > 3) & (g - b >= -5)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(skin)
        keep = np.zeros(count, np.uint8)
        for k in range(1, count):
            x, y, width, height, area = stats[k]
            if area >= max(80, shape[0] * shape[1] * .00018) and area / (width * height) >= .14:
                keep[k] = 255
        skin_exclusion = cv2.dilate(keep[labels], np.ones((25, 25), np.uint8))
        result[skin_exclusion > 0] = 255
        if diagnostics is not None:
            diagnostics['warm_skin_component_count'] = int(np.count_nonzero(keep))
            diagnostics['warm_skin_excluded_pixel_count'] = int(np.count_nonzero(skin_exclusion))
            diagnostics['skin_candidates_are_conservative_not_confirmed_people'] = True
    return cv2.dilate(result, np.ones((15, 15), np.uint8))


def scene_masks(image, approximate_contact_y, foreground=None, contact_foreground=None):
    """Classify surfaces from observed rails/contact, never an old ice ROI.

    The old boundary is retained only as provenance. A missing contact cannot
    create Z=0 pixels. Only the observed horizontal span of a real lower yellow
    strip supplies an ice/board separation; corners outside it remain unknown.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    foreground = np.zeros((h, w), np.uint8) if foreground is None else foreground.copy()
    foreground[:int(.14 * h), :int(.32 * w)] = 255
    foreground[int(.84 * h):, int(.83 * w):] = 255
    boundary_foreground = foreground if contact_foreground is None else contact_foreground.copy()
    boundary_foreground[:int(.14 * h), :int(.32 * w)] = 255
    boundary_foreground[int(.84 * h):, int(.83 * w):] = 255
    blue = ((hue >= 92) & (hue <= 127) & (sat > 40) & (val > 40))
    blue[boundary_foreground > 0] = False
    density = blue.mean(axis=1)
    smooth = np.convolve(density, np.ones(5) / 5, mode='same')
    # This is a direct image rail measurement. The preceding model's guessed
    # contact position no longer truncates its search or the yellow-strip ROI.
    end_rail_search = max(1, int(h * .58))
    rail_y = int(np.argmax(smooth[:end_rail_search]))
    rail_supported = bool(smooth[rail_y] >= .10 and density[rail_y] >= .10)
    rail_lower = rail_y
    if rail_supported:
        while rail_lower + 1 < min(h, rail_y + int(.05 * h)) and density[rail_lower + 1] >= max(.07, density[rail_y] * .30):
            rail_lower += 1
    yellow = ((hue >= 13) & (hue <= 43) & (sat >= 12) & (val >= 85))
    yellow[boundary_foreground > 0] = False
    search_top = rail_lower + max(5, int(.012 * h)) if rail_supported else int(.08 * h)
    yellow[:search_top] = False
    yellow[int(.93 * h):] = False
    points, bins = [], []
    bin_width = max(1, w // 48)
    for column, x0 in enumerate(range(0, w, bin_width)):
        x1 = min(w, x0 + bin_width)
        support = yellow[:, x0:x1].mean(axis=1)
        ys = np.flatnonzero(support > .42)
        if not len(ys):
            continue
        groups = np.split(ys, np.where(np.diff(ys) > 2)[0] + 1)
        for group in groups:
            if 1 <= len(group) <= .035 * h:
                points.append([(x0 + x1 - 1) / 2, float(group[-1] + .5)])
                bins.append(column)
    points = np.asarray(points, float).reshape(-1, 2)
    bins = np.asarray(bins, int)
    contact = None
    best = np.zeros(len(points), bool)
    best_score = (-1, -1., -1.)
    if len(points) >= 6:
        rng = np.random.default_rng(1909)
        for _ in range(260):
            a, b = points[rng.choice(len(points), 2, replace=False)]
            if abs(a[0] - b[0]) < .25 * w:
                continue
            slope = (a[1] - b[1]) / (a[0] - b[0])
            if abs(slope) > .22:
                continue
            residual = abs(points[:, 1] - (a[1] + slope * (points[:, 0] - a[0])))
            tentative = residual < max(2.5, h / 180)
            # One observed lower edge per x-bin; multiple yellow fragments in
            # a bin cannot inflate independent spatial support.
            chosen = np.zeros(len(points), bool)
            for k in np.unique(bins[tentative]):
                ids = np.flatnonzero(tentative & (bins == k))
                chosen[ids[np.argmin(residual[ids])]] = True
            if chosen.sum() < 6:
                continue
            span = float(np.ptp(points[chosen, 0]))
            score = (int(chosen.sum()), span, -float(np.median(points[chosen, 1])))
            if score > best_score:
                best, best_score = chosen, score
        if best.sum() >= 6 and np.ptp(points[best, 0]) >= .25 * w:
            m, b = np.polyfit(points[best, 0], points[best, 1], 1)
            # A physical lower edge must lie below the measured upper rail.
            if not rail_supported or np.min(points[best, 1]) > rail_lower + 5:
                contact = {'observed_xy': points.tolist(), 'inlier_flags': best.tolist(),
                    'line_y_equals_mx_plus_b': [float(m), float(b)],
                    'span_x': [float(points[best, 0].min()), float(points[best, 0].max())],
                    'observed_span_fraction': float(np.ptp(points[best, 0]) / w),
                    'median_normal_residual_px': float(np.median(abs(points[best, 1] - m * points[best, 0] - b))),
                    'uncertainty_source_px': max(3., h / 180),
                    'status': 'candidate_lower_yellow_strip_edge_contact_not_independently_confirmed',
                    'physical_group_count': 1, 'old_ice_roi_used_as_ground_boundary': False}
    yy, xx = np.indices((h, w))
    board, ice = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    if contact:
        m, b = contact['line_y_equals_mx_plus_b']
        ground = m * xx + b
        # Do not extrapolate an observed straight part across an unseen corner.
        left, right = contact['span_x']
        supported = (xx >= max(0, left - bin_width / 2)) & (xx <= min(w - 1, right + bin_width / 2))
        ice[(yy > ground + max(12, h / 90)) & supported] = 255
        if rail_supported:
            board[(yy >= rail_lower + 3) & (yy < ground - 8) & supported] = 255
    elif rail_supported:
        # Only the actually measured rail is known elevated surface. Below
        # it is unknown, not an invented floor or a presumed full board band.
        board[max(0, rail_y - 2):min(h, rail_lower + 2)] = 255
    for mask in (board, ice):
        mask[foreground > 0] = 0
    paint = ((((hue < 4) | (hue > 165)) & (sat > 24)) |
             (((hue >= 90) & (hue < 140)) & (sat > 18))) & (ice > 0)
    ice_features = cv2.dilate(paint.astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    ice_features[ice == 0] = 0
    return {'board_surface': board, 'ice_marking': ice_features, 'ice_texture': ice, 'ice_plane_roi': ice}, contact, {
        'board_upper_rail_y_source_px': rail_y if rail_supported else None,
        'board_upper_rail_lower_y_source_px': rail_lower if rail_supported else None,
        'upper_rail_image_supported': rail_supported,
        'contact_search_prior_y_source_px': float(approximate_contact_y),
        'contact_prior_is_observation': False, 'contact_search_bounded_by_old_ice_roi': False,
        'ground_pixels_require_observed_lower_edge': True,
        'missing_contact_policy': 'unknown_below_observed_rail_no_Z0_pixels',
        'ground_classification_observed_span_only': True,
        'foreground_provenance': 'old_foreground_seeds_plus_current_image_components_full_body_uncertainty_exclusion_or_current_image_OpenCV_HOG_if_no_person_seed',
        'precise_person_segmentation': False,
        'surface_mask_pixel_counts': {'board': int(np.count_nonzero(board)), 'ice': int(np.count_nonzero(ice))}}


def paint_segments(image, mask):
    """Actual paint-boundary segments, deliberately not claimed line centres."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 18, 55)
    edges[mask == 0] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=70,
                            minLineLength=max(50, image.shape[1] // 16), maxLineGap=20)
    if lines is None:
        return []
    selected = sorted(lines[:, 0].tolist(), key=lambda p: -(p[0] - p[2])**2 - (p[1] - p[3])**2)[:24]
    return [{'endpoints_source_px': [[x1, y1], [x2, y2]],
             'identity_status': 'unassigned_paint_boundary_not_confirmed_centerline',
             'uncertainty_source_px': 4., 'independent_two_dimensional_landmark': False}
            for x1, y1, x2, y2 in selected]


def paint_centerlines(image, mask):
    """Measure coloured band centres from real transverse pixel runs.

    Hough only proposes directions. Each retained point is the centre of a
    contiguous coloured run sampled across that direction, not an edge moved
    by an assumed paint width. Curves and parallel marks remain unassigned;
    the camera solver must gate any global model identity association.
    """
    # Faded wide paint can split into two saturated edge fragments. Use weak
    # channel contrast across the entire ice ROI for structure extraction,
    # retaining the saturated threshold only for local texture tracking.
    blue_channel, green_channel, red_channel = [a.astype(float) for a in cv2.split(image)]
    output = []
    for colour, colour_mask in (
        ('red', ((red_channel - green_channel > 5) & (red_channel - blue_channel > 5))),
        ('blue', ((blue_channel - red_channel > 5) & (blue_channel - green_channel > 2))),
    ):
        binary = (colour_mask & (mask > 0)).astype(np.uint8) * 255
        proposed = cv2.HoughLinesP(binary, 1, np.pi / 720, 80,
                                  minLineLength=image.shape[1] // 12, maxLineGap=25)
        if proposed is None:
            continue
        sorted_lines = sorted(proposed[:, 0], key=lambda p: -np.linalg.norm(p[:2] - p[2:]))
        kept_lines = []
        for segment in sorted_lines:
            a, b = segment[:2].astype(float), segment[2:].astype(float)
            direction = (b - a) / np.linalg.norm(b - a)
            normal = np.array([-direction[1], direction[0]])
            midpoint = (a + b) / 2
            if any(abs(np.dot(n, midpoint - m)) < 18 and abs(np.dot(direction, d)) > .992
                   for m, n, d in kept_lines):
                continue
            centres, widths = [], []
            for u in np.linspace(0, 1, 24):
                seed = a * (1 - u) + b * u
                offsets = np.arange(-160, 161)
                samples = np.rint(seed + offsets[:, None] * normal).astype(int)
                valid = ((samples[:, 0] >= 0) & (samples[:, 0] < image.shape[1]) &
                         (samples[:, 1] >= 0) & (samples[:, 1] < image.shape[0]))
                on = np.zeros(len(offsets), bool)
                on[valid] = binary[samples[valid, 1], samples[valid, 0]] > 0
                ids = np.flatnonzero(on)
                groups = np.split(ids, np.where(np.diff(ids) > 12)[0] + 1)
                groups = [g for g in groups if len(g) >= 3 and 2 <= offsets[g[-1]] - offsets[g[0]] <= 220
                          and g[0] > 0 and g[-1] < len(offsets) - 1]
                if not groups:
                    continue
                group = min(groups, key=lambda g: min(abs(offsets[g])))
                if min(abs(offsets[group])) > 18:
                    continue
                # An occluder or image boundary is not a paint edge. Require
                # both measured sides plus a small exterior margin to remain
                # inside the unoccluded ice ROI.
                support = samples[max(0, group[0] - 5):min(len(samples), group[-1] + 6)]
                if ((support[:, 0] < 0).any() or (support[:, 0] >= image.shape[1]).any()
                        or (support[:, 1] < 0).any() or (support[:, 1] >= image.shape[0]).any()
                        or (mask[support[:, 1], support[:, 0]] == 0).any()):
                    continue
                centres.append(seed + normal * (offsets[group[0]] + offsets[group[-1]]) / 2)
                widths.append(float(offsets[group[-1]] - offsets[group[0]] + 1))
            if len(centres) < 10:
                continue
            centres = np.asarray(centres)
            vx, vy, x0, y0 = cv2.fitLine(centres.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
            fitted_direction = np.array([vx, vy])
            fitted_normal = np.array([-vy, vx])
            origin = np.array([x0, y0])
            residuals = abs((centres - origin) @ fitted_normal)
            good = residuals < 6
            if good.sum() < 10:
                continue
            span = (centres[good] - origin) @ fitted_direction
            if np.ptp(span) < image.shape[1] / 12:
                continue
            if any(abs(np.dot(n, origin - m)) < 18 and abs(np.dot(fitted_direction, d)) > .992
                   for m, n, d in kept_lines):
                continue
            kept_lines.append((origin, fitted_normal, fitted_direction))
            endpoints = np.stack([origin + fitted_direction * span.min(), origin + fitted_direction * span.max()])
            output.append({'colour': colour, 'endpoints_source_px': endpoints.tolist(),
                           'observed_centres_source_px': centres[good].tolist(),
                           'normal_residual_p95_source_px': float(np.percentile(residuals[good], 95)),
                           'median_paint_width_source_px': float(np.median(np.asarray(widths)[good])),
                           'uncertainty_source_px': max(4., float(np.median(np.asarray(widths)[good])) * .07),
                           'physical_group_id': 'unassigned_' + colour + '_paint_structure',
                           'identity_status': 'colour_band_centerline_candidate_global_identity_not_assigned',
                           'independent_two_dimensional_landmark': False,
                           'algorithm': 'HD_colour_runs_normal_to_Hough_direction_robust_fit'})
            if len(kept_lines) >= 5:
                break
    return output


class TypedLK:
    """Persistent track IDs and genuine pixel pairs, without pose prefiltering."""
    def __init__(self, channel, scale=.5, maximum=360):
        self.channel, self.scale, self.maximum = channel, scale, maximum
        self.tracks = []
        self.next_id = 0
        self.previous = None
        self.previous_index = None
        self.history = {}

    def update(self, image, mask, index):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, gray.shape[::-1], interpolation=cv2.INTER_NEAREST)
        kept, arrays = [], {}
        if self.previous is not None and self.tracks:
            a = np.asarray([t.xy for t in self.tracks], np.float32).reshape(-1, 1, 2)
            b, st, err = cv2.calcOpticalFlowPyrLK(self.previous, gray, a, None,
                winSize=(25, 25), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, .015))
            back, rst, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, b, None,
                winSize=(25, 25), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, .015))
            fb = np.linalg.norm(back[:, 0] - a[:, 0], axis=1)
            bxy = b[:, 0]
            inside = (bxy[:, 0] >= 1) & (bxy[:, 1] >= 1) & (bxy[:, 0] < gray.shape[1] - 1) & (bxy[:, 1] < gray.shape[0] - 1)
            ii = np.flatnonzero(inside)
            inside[ii] &= small_mask[bxy[ii, 1].astype(int), bxy[ii, 0].astype(int)] > 0
            good = st[:, 0].astype(bool) & rst[:, 0].astype(bool) & inside & (fb < 1.1) & (err[:, 0] < 26)
            # No RANSAC/similarity homography filtering is applied here.
            chosen = np.flatnonzero(good)
            arrays = {'previous_xy': a[chosen, 0] / self.scale, 'current_xy': bxy[chosen] / self.scale,
                      'track_ids': np.asarray([self.tracks[k].id for k in chosen], np.int64),
                      'birth_xy': np.asarray([self.tracks[k].birth_xy for k in chosen], np.float32).reshape(-1, 2) / self.scale,
                      'birth_frame_index': np.asarray([self.tracks[k].birth_index for k in chosen], np.int32),
                      'fb_error_source_px': fb[chosen] / self.scale,
                      'lk_patch_error': err[chosen, 0]}
            for k in chosen:
                t = self.tracks[k]
                kept.append(SceneTrack(t.id, bxy[k], t.birth_xy, t.birth_index, t.age + 1))
        if len(kept) < self.maximum * .8:
            available = small_mask.copy()
            for t in kept:
                cv2.circle(available, tuple(np.rint(t.xy).astype(int)), 7, 0, -1)
            # Tile quotas maintain spatial support, without allowing a single
            # repeated word to dominate all tracks. IDs are local temporal IDs.
            for col in range(8):
                a, b = col * gray.shape[1] // 8, (col + 1) * gray.shape[1] // 8
                submask = available[:, a:b]
                count = max(0, self.maximum // 8 - sum(a <= t.xy[0] < b for t in kept))
                if count:
                    corners = cv2.goodFeaturesToTrack(gray[:, a:b], count, .012, 6, mask=submask, blockSize=5)
                    if corners is not None:
                        for xy in corners[:, 0]:
                            xy = xy + [a, 0]
                            kept.append(SceneTrack(self.next_id, xy, xy.copy(), index))
                            self.next_id += 1
        self.tracks, self.previous, self.previous_index = kept, gray, index
        return arrays, len(kept)
