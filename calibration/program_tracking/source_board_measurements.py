"""Read visible board boundary candidates on one exact source exposure.

The caller owns decoding, person exclusions, and exact-pixel physical review.
No camera, scene projection, fitted residual, temporal interpolation, or file
output enters measurement. Colour components remain unassigned until reviewed.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .far_rail_source_components import blue_strip_candidates, source_exclusion


SCHEMA = 'rinkspace.single-exposure-source-board-measurements.v1'


def _main_yellow_fall(image, score, exclusion, x, top):
    """Read the principal local stripe, not weak yellow reflections below it.

    The old diagnostic chose the lowest qualifying peak, so small reflected
    yellow fluctuations underneath a visible kickplate displaced its readout.
    Here the source colour prominence selects a stripe, and its own falling
    transition is measured. This colour readout still carries no material ID.
    """
    # Native perspective/zoom changes the board's pixel height. The tail used
    # to assess neutral ice must not shorten an arbitrary 300px search bound.
    lo, hi = max(151, int(np.ceil(top+35))), 1030
    if hi <= lo:
        return None
    profile = np.median(score[lo:hi, x-2:x+3], axis=1)
    maxima = np.flatnonzero((profile[1:-1] >= profile[:-2])
                           & (profile[1:-1] >= profile[2:])
                           & (profile[1:-1] >= 11.))+1
    candidates = []
    for p in maxima:
        y = lo+int(p)
        if p+37 >= len(profile) or exclusion[y-4:y+38, x-3:x+4].any():
            continue
        # Red/brown advertisement ink can give a stronger (R+G)/2-B peak
        # than a pale yellow foot. Require both observed red and green to
        # exceed blue; a red-only CAN logo is not a yellow board boundary.
        blue, green, red = np.median(
            image[y-1:y+2, x-2:x+3].reshape(-1, 3), axis=0).astype(float)
        if green-blue < 6. or red-blue < 8.:
            continue
        # The ice immediately below a kickplate can carry its yellow reflection.
        # A local neutral-only test incorrectly rejected the visible main foot
        # and then selected a weaker reflection farther down. Read against the
        # weaker observed source tail, retaining its real colour and blur.
        tail = profile[p+14:min(len(profile), p+70)]
        baseline = float(np.percentile(tail, 20.))
        amplitude = float(profile[p]-baseline)
        if amplitude < 10.:
            continue
        below = image[y+20:y+40, x-2:x+3].astype(np.float32)
        if np.median(below) < 80.:
            continue
        crossings = []
        for fraction in (.75, .5, .25):
            threshold = baseline+fraction*amplitude
            falls = np.flatnonzero((profile[p:p+32] >= threshold)
                                  & (profile[p+1:p+33] < threshold))
            if not len(falls):
                break
            k = p+int(falls[0])
            crossings.append(float(lo+k+(profile[k]-threshold)/(profile[k]-profile[k+1])))
        if len(crossings) == 3 and crossings[0] <= crossings[1] <= crossings[2]:
            candidates.append((crossings[1], amplitude,
                               (crossings[0]-crossings[1], crossings[2]-crossings[1])))
    return max(candidates, key=lambda value: value[1]) if candidates else None


def _native_inputs(raw_bgr, exclusion_mask, source_binding):
    image = np.asarray(raw_bgr)
    excluded = np.asarray(exclusion_mask)
    if image.shape != (1080, 1920, 3) or image.dtype != np.uint8:
        raise ValueError('Exact native 1920x1080 uint8 BGR source required')
    if excluded.shape != image.shape[:2] or excluded.dtype != np.bool_:
        raise ValueError('Union of actual person/overlay exclusions required')
    source_binding = dict(source_binding)
    source_binding.setdefault('video_sha256', source_binding.get('source_sha256'))
    source_binding.setdefault('native_pts', source_binding.get('source_pts'))
    source_binding.setdefault('decoded_BGR_sha256', source_binding.get('decoded_native_BGR_sha256'))
    required = ('video_sha256', 'time_base', 'actual_pts', 'native_pts')
    if any(source_binding.get(k) is None for k in required):
        raise ValueError('Exact native and representative source PTS required')
    if (len(source_binding['video_sha256']) != 64
            or not isinstance(source_binding['actual_pts'], (int, np.integer))
            or not isinstance(source_binding['native_pts'], (int, np.integer))):
        raise ValueError('Explicit original video hash and integer PTS required')
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    expected = source_binding.get('decoded_BGR_sha256')
    if expected is not None and digest != expected:
        raise ValueError('Shared raw source bytes differ from source binding')
    binding = dict(source_binding, decoded_BGR_sha256=digest,
                   pixel_domain='native1920x1080')
    return image, source_exclusion(image, excluded), binding


def _paired_boundary_runs(image, exclusion, components):
    """Pair each measured blue upper sample with a local yellow lower sample.

    A component colour and its paired stripe do not certify board identity.
    Every lower query uses this exposure's pixels, and gaps are never filled.
    """
    smooth = cv2.GaussianBlur(image, (5, 5), .9).astype(np.float32)
    b, g, r = cv2.split(smooth)
    yellow = .5 * (r + g) - b
    output = []
    for component in components:
        source_upper, source_intervals, source_methods = _read_missing_blue_columns(image, exclusion, component)
        measured = []
        transition_bands = []
        yellow_transition_bands = []
        actual_upper_methods = []
        for upper, interval, method in zip(source_upper, source_intervals, source_methods):
            x, top = upper
            lower = _main_yellow_fall(image, yellow, exclusion, int(x), float(top))
            if lower is None:
                continue
            bottom, _, _ = lower
            measured.append((float(x), float(top), float(bottom)))
            transition_bands.append((float(interval[0]), float(interval[1])))
            yellow_transition_bands.append(lower[2])
            actual_upper_methods.append(method)
        measured = np.asarray(measured, np.float64).reshape(-1, 3)
        transition_bands = np.asarray(transition_bands, np.float64).reshape(-1, 2)
        yellow_transition_bands = np.asarray(yellow_transition_bands, np.float64).reshape(-1, 2)
        if not len(measured):
            continue
        # Do not bridge occluders, unsupported columns, or stripe switches.
        d = np.diff(measured, axis=0)
        slopes = d[:, 1:]/np.maximum(d[:, :1], 1.)
        # A real side-board can slope considerably in native source pixels.
        # Its locally smooth slope is not an occlusion or a material switch.
        jumps = np.zeros(len(d), bool)
        if len(d) > 1:
            jumps[1:] = (np.abs(np.diff(slopes, axis=0)) > .35).any(1)
        cuts = np.flatnonzero((d[:, 0] > 29.)
            | (np.abs(slopes) > 1.5).any(1) | jumps) + 1
        for indices in np.split(np.arange(len(measured)), cuts):
            if len(indices) < 4:
                continue
            values = measured[indices]
            output.append(dict(
                candidate_id=f"blue{int(component['component_id'])}.run{len(output)}",
                component_bbox_xywh=list(component['component_bbox_xywh']),
                blue_upper_uv=values[:, :2],
                yellow_lower_uv=values[:, [0, 2]],
                blue_transition_interval_y_px=transition_bands[indices],
                yellow_transition_interval_y_px=yellow_transition_bands[indices],
                measurement_origin='same_exposure_native_colour_transitions',
                blue_upper_readout='local_current_colour_material_transition',
                blue_upper_sample_readout_methods=[actual_upper_methods[i] for i in indices],
                yellow_lower_readout='principal_local_stripe_half_chroma_yellow_to_neutral_fall',
                readout_not_exact_physical_interface_certificate=True,
                status='pending_exact_source_physical_identity_review',
                static_material_confirmed=False,
                pixel_location_confirmed=False,
                allowed_for_fit=False))
    return output


def _read_missing_blue_columns(image, exclusion, component):
    """Reread a connected rail where a blue wall display joins its colour mask.

    Two actually measured strip samples navigate the missing column only. Its
    emitted ordinate is the current RGB material transition, not the line
    between those samples. A bright blue clock against a dark blue rail may
    have no positive blue-chroma rise, but its native RGB interface is visible.
    The physical identity of every such bracket remains unapproved.
    """
    original = np.asarray(component['observed_uv'], float)
    intervals = np.asarray(component['vertical_transition_interval_px'], float)
    values = {int(p[0]): (p, interval, 'positive_blue_chroma_half_transition')
              for p, interval in zip(original, intervals)}
    for x in range(16, 1904, 28):
        if x in values:
            continue
        right = int(np.searchsorted(original[:, 0], x))
        if right == 0 or right == len(original):
            continue
        left = right-1
        fraction = (x-original[left, 0])/(original[right, 0]-original[left, 0])
        navigation = original[left, 1]*(1-fraction)+original[right, 1]*fraction
        centre = int(round(navigation)); lo, hi = centre-12, centre+9
        if lo < 40 or hi >= 1030 or exclusion[lo:hi+1, x-3:x+4].any():
            continue
        profile = np.median(image[lo:hi+1, x-2:x+3].astype(float), axis=1)
        before, after = np.median(profile[2:6], axis=0), np.median(profile[14:19], axis=0)
        blue, green, red = after
        if blue-.5*(red+green) < 13. or blue-green < 8. or blue-red < 10.:
            continue
        contrast = after-before; strength = np.linalg.norm(contrast)
        if strength < 20.:
            continue
        direction = contrast/strength
        signal = profile@direction
        low, high = float(before@direction), float(after@direction)
        crossings = []
        for amount in (.1, .5, .9):
            level = low+amount*(high-low)
            rises = np.flatnonzero((signal[:-1] <= level)&(signal[1:] > level))
            rises = rises[np.abs(lo+rises-navigation) <= 9.]
            if len(rises) != 1:
                break
            k = int(rises[0])
            crossings.append(float(lo+k+(level-signal[k])/(signal[k+1]-signal[k])))
        if (len(crossings) == 3 and crossings[0] <= crossings[1] <= crossings[2]
                and abs(crossings[1]-navigation) <= 6.):
            values[x] = (np.array([float(x), crossings[1]]),
                         np.array([crossings[0]-crossings[1], crossings[2]-crossings[1]]),
                         'current_RGB_interface_to_blue_strip_neighbour_pixels_navigation_only')
    ordered = [values[x] for x in sorted(values)]
    return np.asarray([v[0] for v in ordered]), np.asarray([v[1] for v in ordered]), [v[2] for v in ordered]


def _independent_yellow_runs(image, exclusion):
    """Read a visible foot even where people occlude the upper rail.

    These supply only conservative ice exclusion. No missing upper rail UV is
    manufactured, and a yellow-only run can never admit a board observation.
    """
    smooth = cv2.GaussianBlur(image, (5, 5), .9).astype(np.float32)
    b, g, r = cv2.split(smooth)
    score = .5*(r+g)-b
    readings = []
    for x in range(16, 1904, 28):
        result = _main_yellow_fall(image, score, exclusion, x, 115.)
        if result is not None:
            readings.append((float(x), result[0], *result[2]))
    values = np.asarray(readings, np.float64).reshape(-1, 4)
    if not len(values):
        return []
    delta = np.diff(values[:, :2], axis=0)
    slopes = delta[:, 1]/np.maximum(delta[:, 0], 1.)
    jumps = np.zeros(len(slopes), bool)
    if len(slopes) > 1:
        jumps[1:] = np.abs(np.diff(slopes)) > .35
    cuts = np.flatnonzero((delta[:, 0] > 29.) | (np.abs(slopes) > 1.5) | jumps)+1
    output = []
    for take in np.split(np.arange(len(values)), cuts):
        if len(take) < 8:
            continue
        output.append(dict(candidate_id=f'independent_visible_yellow_foot_{len(output)}',
            yellow_lower_uv=values[take, :2],
            yellow_transition_interval_y_px=values[take, 2:],
            measurement_origin='same_exposure_native_principal_yellow_to_neutral_fall',
            upper_rail_visible_and_measured=False,
            allowed_for_board_bracket=False, allowed_for_fit=False,
            status='source_yellow_foot_scope_only_physical_identity_not_automatically_approved'))
    return output


def board_bracket_candidates(uv, bands, exclusion_mask, *, patch_radius=4):
    """Return candidate bounds only; this never grants physical admission.

    Each query must have visible bracketing samples no farther than one native
    28-pixel sampling interval. The full local patch must be unoccluded.
    """
    points = np.asarray(uv, np.float64).reshape(-1, 2)
    mask = np.asarray(exclusion_mask, bool)
    found = np.full(len(points), -1, np.int32)
    for i, (x, y) in enumerate(points):
        if (not np.isfinite([x, y]).all() or x < patch_radius
                or x >= 1920-patch_radius or y < patch_radius
                or y >= 1080-patch_radius):
            continue
        ix, iy = np.rint([x, y]).astype(int)
        if mask[iy-patch_radius:iy+patch_radius+1,
                ix-patch_radius:ix+patch_radius+1].any():
            continue
        for j, band in enumerate(bands):
            upper = np.asarray(band['blue_upper_uv'])
            lower = np.asarray(band['yellow_lower_uv'])
            k = int(np.searchsorted(upper[:, 0], x))
            if k < len(upper) and abs(upper[k, 0]-x) < 1e-9:
                top, bottom = upper[k, 1], lower[k, 1]
            elif k == 0 or k == len(upper) or upper[k, 0]-upper[k-1, 0] > 29.:
                continue
            else:
                a = (x-upper[k-1, 0])/(upper[k, 0]-upper[k-1, 0])
                top = (1-a)*upper[k-1, 1] + a*upper[k, 1]
                bottom = (1-a)*lower[k-1, 1] + a*lower[k, 1]
            if top + patch_radius < y < bottom-patch_radius:
                found[i] = j
                break
    return found


def measure_source_advertisement_corners(image, exclusion, bands, proposals):
    """Refine source-image proposals; retain instance identity as unconfirmed.

    A proposal must come from a previously reviewed source annotation or
    source-image feature match, never a model projection. Its origin is kept
    separately from the newly measured current-exposure corner pixels.
    """
    if not proposals:
        return []
    from .ad_corners import (_color_quad_from_proposal, _edge_quad,
                             _source_quality_color_corners, ROLES)
    results = []
    for proposal in proposals:
        if (proposal.get('proposal_origin') not in
                ('reviewed_same_exposure_source_annotation', 'source_image_feature_match')
                or proposal.get('model_projection_used') is not False):
            raise ValueError('Advertisement search proposal must be source-image-only')
        quad = np.asarray(proposal['quad'], np.float64)
        if quad.shape != (4, 2) or not np.isfinite(quad).all():
            raise ValueError('Four exact source proposal pixels required')
        if not str(proposal.get('physical_instance_id', '')).strip():
            raise ValueError('Source instance identity must remain traceable')
        colour = proposal.get('color')
        if colour in ('purple', 'navy', 'blue', 'red'):
            measured = _color_quad_from_proposal(image, exclusion, quad, colour) or {}
            measured = _source_quality_color_corners(image, exclusion, measured, colour)
        else:
            measured = _edge_quad(image, exclusion, quad)
        for role in ROLES:
            if role not in measured:
                continue
            value = measured[role]
            uv = np.asarray(value['uv'], np.float64)
            band = board_bracket_candidates(uv[None], bands, exclusion)[0]
            if band < 0:
                continue
            results.append(dict(
                physical_material_id=f"{proposal['physical_instance_id']}.{role}",
                uv=uv, sigma_px=float(value['sigma_px']),
                board_band_candidate_id=bands[band]['candidate_id'],
                measurement_origin='actual_source_two_visible_edge_intersection',
                source_identity_reference=proposal.get('identity_reference'),
                proposal_origin=proposal['proposal_origin'],
                status='pending_exact_source_pixel_and_instance_review',
                static_material_confirmed=False, pixel_location_confirmed=False,
                allowed_for_fit=False))
    return results


def measure_board_source_frame(raw_bgr, exclusion_mask, source_binding,
                               *, advertisement_proposals=()):
    """Single-exposure API for the programme's shared, sequential RGB reader."""
    image, exclusion, binding = _native_inputs(raw_bgr, exclusion_mask, source_binding)
    components = blue_strip_candidates(image, exclusion, split_connected_columns=True)
    bands = _paired_boundary_runs(image, exclusion, components)
    yellow_only = _independent_yellow_runs(image, exclusion)
    corners = measure_source_advertisement_corners(
        image, exclusion, bands, advertisement_proposals)
    return dict(schema=SCHEMA, source=binding, boundary_candidates=bands,
        yellow_lower_candidates=yellow_only,
        ice_scope_boundary_candidates=bands+yellow_only,
        advertisement_corner_candidates=corners,
        source_exposure_processed=True,
        physical_scope_or_precision_automatically_approved=False,
        camera_model_or_fitted_residual_used=False,
        status=('pending_exact_source_review' if bands else
                'unknown_no_visible_paired_blue_and_yellow_boundaries'))
