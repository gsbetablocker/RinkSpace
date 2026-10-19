"""Source-pixel arc organization, independent of event/world/camera geometry.

Image ellipses locate RGB search neighborhoods. Every retained support pixel
remains its original observation; ellipse samples are never output positions,
paint labels, physical marker identities or a verified world circle.
"""
from __future__ import annotations

import hashlib
import cv2
import numpy as np

from .vision_instances import source_binding, same_frame


def _runs_uv(runs):
    return np.asarray([(x+i, y) for y, x, length in runs for i in range(length)], np.float32).reshape(-1, 2)


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class _MaterialProfileFrame:
    """One authenticated immutable RGB snapshot for a compound profile call."""
    __slots__ = ("rgb", "binding", "digest", "luminance", "material_signals")

    def __init__(self, rgb, binding, expected_digest=None):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Material reading requires original native uint8 RGB")
        self.binding = source_binding(binding, rgb.shape[:2])
        # The caller's mutable array must not change the proof/luminance while
        # local candidates are read. Cache only this private same-call copy;
        # a later public call authenticates its source pixels afresh.
        self.rgb = np.array(rgb, dtype=np.uint8, order="C", copy=True)
        self.rgb.flags.writeable = False
        self.digest = hashlib.sha256(self.rgb.tobytes()).hexdigest()
        expected = expected_digest or self.binding.get("decoded_native_rgb_sha256")
        if expected is not None and expected != self.digest:
            raise ValueError("Material reading and bound original RGB differ")
        self.luminance = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
        self.luminance.flags.writeable = False
        self.material_signals = {}


def _inherit_material_search_context(reading, context):
    """A narrower search cannot erase unresolved source material competitors."""
    reading['local_source_flags'] = dict(reading['source_flags'])
    reading['source_context_profile'] = {key: context[key] for key in (
        'source_search_point_uv', 'source_search_normal', 'source_search_normal_interval_px',
        'material_transition', 'source_signal', 'source_flags', 'observed_material_intervals',
        'actual_source_transition_candidate_uv', 'source_q10_q50_q90_normal_offsets_px',
        'actual_source_signed_luminance_mass', 'actual_source_sided_material_delta', 'disposition')}
    reading['source_flags'] = {key: bool(reading['source_flags'].get(key, False)
        or context['source_flags'].get(key, False))
        for key in reading['source_flags'].keys() | context['source_flags'].keys()}
    reading['source_context_resolution_required'] = any(context['source_flags'].values())
    reading['source_context_ambiguity_not_resolved_by_narrowing'] = True
    if any(reading['source_flags'].values()):
        reading['actual_source_uv'] = np.empty((0, 2), np.float64)
        reading['disposition'] = 'source_material_role_unknown'
    return reading


def read_source_material_transition_profiles(rgb, *, binding, search_points_uv, normals,
        search_intervals_px, transition, expected_RGB_sha256=None,
        tangent_half_width_px=2, smoothing_sigma_px=1., engineering_support_margin_px=2.,
        keep_profiles=False, material_signal_rgb_weights=None, context_search_intervals_px=None,
        _validated_frame=None):
    """Measure a declared material transition in current native RGB.

    Search points, normals and finite intervals locate source pixels only. A
    gray-disc boundary uses signed luminance, never a blue-minus-red proxy.
    The q10/50/90 integrate the measured signed transition, retaining its blur
    support. Reversals, multiple separated transitions and truncated searches
    remain candidates with an unknown role; they do not become output UV.
    Identity and world correspondence must be supplied by a separate authority.
    Refining a prior wider search must retain it in ``context_search_intervals_px``;
    its unresolved material evidence survives even if the local flags clear.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Material reading requires original native uint8 RGB")
    if _validated_frame is None:
        frame = _MaterialProfileFrame(rgb, binding, expected_RGB_sha256)
        rgb, binding, digest = frame.rgb, frame.binding, frame.digest
    else:
        frame = _validated_frame
        if not isinstance(frame, _MaterialProfileFrame) or rgb is not frame.rgb:
            raise ValueError("Material cache requires its exact immutable source RGB")
        binding = source_binding(binding, rgb.shape[:2])
        same_frame(binding, frame.binding)
        digest = frame.digest
    expected = expected_RGB_sha256 or binding.get("decoded_native_rgb_sha256")
    if expected is not None and expected != digest:
        raise ValueError("Material reading and bound original RGB differ")
    if transition not in ("dark_to_light", "light_to_dark"):
        raise ValueError("An explicit signed material transition is required")
    points = np.asarray(search_points_uv, np.float64).reshape(-1, 2)
    directions = np.asarray(normals, np.float64).reshape(-1, 2)
    intervals = np.asarray(search_intervals_px, np.float64).reshape(-1, 2)
    if (len(points) != len(directions) or len(points) != len(intervals)
            or not all(np.isfinite(a).all() for a in (points, directions, intervals))
            or (intervals[:, 1]-intervals[:, 0] < 6).any()):
        raise ValueError("Finite matching source normal searches of at least six pixels are required")
    lengths = np.linalg.norm(directions, axis=1)
    if (lengths <= 1e-9).any():
        raise ValueError("Source normals must have a nonzero direction")
    directions = directions/lengths[:, None]
    if (not isinstance(tangent_half_width_px, (int, np.integer)) or tangent_half_width_px < 0
            or not np.isfinite(smoothing_sigma_px) or smoothing_sigma_px <= 0
            or not np.isfinite(engineering_support_margin_px) or engineering_support_margin_px < 0):
        raise ValueError("Invalid source profile support settings")
    # Match native OpenCV luminance, including its uint8 quantization. Unlike
    # colour-difference proxies this observes a neutral gray-to-white boundary.
    if material_signal_rgb_weights is None:
        luminance = frame.luminance
        signal_name = 'native_RGB_BT601_luminance'
    else:
        weights = np.asarray(material_signal_rgb_weights, np.float64)
        if weights.shape != (3,) or not np.isfinite(weights).all() or not np.any(weights):
            raise ValueError('An explicit finite nonzero source RGB material signal is required')
        token = weights.tobytes()
        if token not in frame.material_signals:
            value = np.einsum('ijk,k->ij', frame.rgb, weights)
            value.flags.writeable = False
            frame.material_signals[token] = value
        luminance = frame.material_signals[token]
        signal_name = 'explicit_source_RGB_linear_material_signal'
    sign = 1 if transition == "dark_to_light" else -1
    tangent_offsets = np.arange(-tangent_half_width_px, tangent_half_width_px+1)
    records = []
    for query, (point, normal, interval) in enumerate(zip(points, directions, intervals)):
        offsets = np.arange(interval[0], interval[1]+.25, 1., dtype=np.float64)
        tangent = np.array([-normal[1], normal[0]])
        # Read the smoothing halo from actual source pixels. Reflecting a
        # search bracket would invent a plateau at its endpoints and change
        # the signed blur mass simply because navigation was cropped.
        halo_offsets = np.r_[offsets[0]-2, offsets[0]-1, offsets, offsets[-1]+1, offsets[-1]+2]
        samples = (point[None, None]+normal[None, None]*halo_offsets[None, :, None]
                   +tangent[None, None]*tangent_offsets[:, None, None])
        valid = ((samples[..., 0] >= 0) & (samples[..., 0] <= rgb.shape[1]-1)
                 & (samples[..., 1] >= 0) & (samples[..., 1] <= rgb.shape[0]-1))
        # OpenCV 4 used this interpolation grid implicitly; OpenCV 5 float
        # remap is exact-coordinate. Preserve the actual legacy operator.
        samples = np.rint(samples.astype(np.float32)*32.)/32.
        values = cv2.remap(luminance, samples[..., 0].astype(np.float32),
                           samples[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        halo_profile = values.mean(0)
        profile = halo_profile[2:-2]
        smoothed = cv2.GaussianBlur(halo_profile[:, None], (1, 5), smoothing_sigma_px).ravel()[2:-2]
        signed = np.diff(smoothed)*sign
        normal_valid = valid.all(0)[2:-2]
        pair_valid = normal_valid[:-1] & normal_valid[1:]
        signed[~pair_valid] = np.nan
        mass = np.where(pair_valid, np.maximum(signed, 0), 0)
        total = float(mass.sum())
        midpoint = (offsets[:-1]+offsets[1:])*.5
        second = np.diff(smoothed, n=2)
        noise = float(1.4826*np.median(np.abs(second-np.median(second))))
        # One native luminance level is the measurement's quantization floor,
        # not a confidence or a veto on a wide/weak but coherent transition.
        reversal_floor = max(1., 3*noise)
        flags = dict(source_image_border_truncated=not bool(valid.all()),
                     no_signed_material_support=total <= 0,
                     competing_material_reversal=False,
                     separated_same_sign_transitions=False,
                     search_interval_truncated=False,
                     sided_material_direction_unverified=False)
        q = np.full(3, np.nan)
        components = []
        if total > 0:
            # A signed increment spans offsets[i]..offsets[i+1]. Integrate
            # on those actual interval endpoints; placing cumulative mass at
            # midpoints instead shifts every coherent interface by half a px.
            q = np.interp([.1, .5, .9], np.r_[0., np.cumsum(mass)]/total, offsets)
            central = (midpoint >= q[0]) & (midpoint <= q[2]) & pair_valid
            flags["competing_material_reversal"] = bool((signed[central] < -reversal_floor).any())
            # Quantized stair steps of one blurred interface can have several
            # peaks. A substantial flat gap separating two observed rises is
            # different: preserve both material intervals, do not pick nearest.
            rising = mass > max(noise, .25)
            starts = np.flatnonzero(rising & ~np.r_[False, rising[:-1]])
            ends = np.flatnonzero(rising & ~np.r_[rising[1:], False])
            for a, b in zip(starts, ends):
                component_mass = float(mass[a:b+1].sum())
                if component_mass >= reversal_floor:
                    components.append(dict(normal_interval_px=[float(midpoint[a]), float(midpoint[b])],
                                           signed_luminance_mass=component_mass))
            substantial = [row for row in components if row["signed_luminance_mass"] >= total*.1]
            flags["separated_same_sign_transitions"] = any(
                b["normal_interval_px"][0]-a["normal_interval_px"][1] >= 5
                for a, b in zip(substantial[:-1], substantial[1:]))
            flags["search_interval_truncated"] = bool(
                (pair_valid[0] and mass[0] >= reversal_floor)
                or (pair_valid[-1] and mass[-1] >= reversal_floor))
        side_count = min(5, max(1, len(profile)//4))
        sided_delta = float(sign*(profile[-side_count:].mean()-profile[:side_count].mean()))
        flags["sided_material_direction_unverified"] = not (sided_delta >= reversal_floor)
        accepted = not any(flags.values())
        candidate = point+normal*q[1]
        support = [float(q[0]-q[1]-engineering_support_margin_px),
                   float(q[2]-q[1]+engineering_support_margin_px)]
        record = dict(source_query_index=query, source_search_point_uv=point.copy(),
            source_search_normal=normal.copy(), source_search_normal_interval_px=interval.copy(),
            material_transition=transition, source_signal=signal_name,
            actual_source_transition_candidate_uv=candidate,
            actual_source_uv=candidate.copy() if accepted else np.empty((0, 2), np.float64),
            source_q10_q50_q90_normal_offsets_px=q,
            observed_normal_interval_px=np.asarray(support, np.float64),
            observed_material_intervals=components, source_luminance_noise_scale=noise,
            actual_source_signed_luminance_mass=total,
            actual_source_sided_material_delta=sided_delta, source_flags=flags,
            disposition="measured_material_transition" if accepted else "source_material_role_unknown",
            peak_position_rule="integrated_actual_signed_luminance_transition_q50",
            support_is_empirical_blur_plus_declared_engineering_margin=True,
            support_is_calibrated_covariance=False, weak_or_wide_used_as_hard_veto=False)
        if keep_profiles:
            record.update(actual_source_normal_offsets_px=offsets,
                actual_source_luminance_profile=profile,
                actual_source_smoothed_luminance_profile=smoothed,
                actual_source_signed_luminance_gradient=signed,
                actual_source_profile_valid=normal_valid)
        records.append(record)
    if context_search_intervals_px is not None:
        context = np.asarray(context_search_intervals_px, np.float64).reshape(-1, 2)
        if (context.shape != intervals.shape or not np.isfinite(context).all()
                or np.any(context[:, 0] > intervals[:, 0]) or np.any(context[:, 1] < intervals[:, 1])):
            raise ValueError('Original material context must contain each refined source search')
        contextual = read_source_material_transition_profiles(frame.rgb, binding=frame.binding,
            search_points_uv=points, normals=directions, search_intervals_px=context,
            transition=transition, expected_RGB_sha256=digest, tangent_half_width_px=tangent_half_width_px,
            smoothing_sigma_px=smoothing_sigma_px, engineering_support_margin_px=engineering_support_margin_px,
            material_signal_rgb_weights=material_signal_rgb_weights, _validated_frame=frame)
        for row, original in zip(records, contextual['readings']):
            _inherit_material_search_context(row, original)
    accepted = [row for row in records if row["disposition"] == "measured_material_transition"]
    return dict(schema="rinkspace.source-signed-material-profile-reading.v1",
        source_binding=binding, RGB_sha256=digest, readings=records,
        material_signal_rgb_weights=None if material_signal_rgb_weights is None else weights.tolist(),
        actual_source_uv=np.asarray([row["actual_source_uv"] for row in accepted], np.float64).reshape(-1, 2),
        source_query_indices=np.asarray([row["source_query_index"] for row in accepted], np.int64),
        physical_id=None, geometry_role="source_material_transition",
        source_identity_authority_required=True, physical_edge_role_verified=False,
        world_camera_projection_input_used=False,
        search_points_used_as_measurements=False, colour_difference_proxy_used=False,
        original_raw_source_preserved=True)


def read_source_stroke_midlines(rgb, *, binding, search_points_uv, normals,
        entering_search_intervals_px, leaving_search_intervals_px,
        material_signal_rgb_weights=None, paint_signal_increases=False,
        expected_RGB_sha256=None, tangent_half_width_px=2, keep_profiles=False,
        entering_context_search_intervals_px=None, leaving_context_search_intervals_px=None,
        _validated_frame=None):
    """Locate the geometric midline of two separately observed paint sides.

    Brackets are source search aids for the declared same stroke. This call
    measures both material interfaces independently and averages those two
    observed locations, never the darkest pixel or a paint intensity median.
    Interior texture cannot select a different centre. Missing/competing sides
    remain unknown; the full footprint never becomes a free-location interval.
    The caller separately authenticates that the two sides are the same paint.
    """
    if not isinstance(paint_signal_increases, (bool, np.bool_)):
        raise ValueError('Declare the actual source material signal polarity')
    frame = _MaterialProfileFrame(rgb, binding, expected_RGB_sha256) if _validated_frame is None else _validated_frame
    if not isinstance(frame, _MaterialProfileFrame) or (_validated_frame is not None and rgb is not frame.rgb):
        raise ValueError('Stroke-midline cache requires its exact original RGB')
    same_frame(source_binding(binding, frame.rgb.shape[:2]), frame.binding)
    first = np.asarray(entering_search_intervals_px, float).reshape(-1, 2)
    last = np.asarray(leaving_search_intervals_px, float).reshape(-1, 2)
    if (first.shape != last.shape or not np.isfinite(first).all() or not np.isfinite(last).all()
            or np.any(first[:, 0] >= first[:, 1]) or np.any(last[:, 0] >= last[:, 1])):
        raise ValueError('Two finite actual material-side search brackets required')
    common = dict(binding=frame.binding, search_points_uv=search_points_uv, normals=normals,
        material_signal_rgb_weights=material_signal_rgb_weights, expected_RGB_sha256=frame.digest,
        tangent_half_width_px=tangent_half_width_px, keep_profiles=keep_profiles,
        engineering_support_margin_px=0., _validated_frame=frame)
    entering = read_source_material_transition_profiles(frame.rgb,
        search_intervals_px=first, context_search_intervals_px=entering_context_search_intervals_px,
        transition='dark_to_light' if paint_signal_increases else 'light_to_dark', **common)
    leaving = read_source_material_transition_profiles(frame.rgb,
        search_intervals_px=last, context_search_intervals_px=leaving_context_search_intervals_px,
        transition='light_to_dark' if paint_signal_increases else 'dark_to_light', **common)
    rows = []
    for left, right in zip(entering['readings'], leaving['readings']):
        measured = (left['disposition'] == right['disposition'] == 'measured_material_transition')
        center = .5 * (left['actual_source_transition_candidate_uv'] + right['actual_source_transition_candidate_uv'])
        left_q = left['source_q10_q50_q90_normal_offsets_px']
        right_q = right['source_q10_q50_q90_normal_offsets_px']
        normal_center = .5 * (left_q[1] + right_q[1])
        support = .5 * np.array([left_q[0] + right_q[0], left_q[2] + right_q[2]]) - normal_center
        flags = dict(entering_material_side_unresolved=not (left['disposition'] == 'measured_material_transition'),
            leaving_material_side_unresolved=not (right['disposition'] == 'measured_material_transition'),
            material_sides_reversed=not bool(left_q[1] < right_q[1]))
        accepted = measured and not any(flags.values())
        rows.append(dict(source_query_index=left['source_query_index'],
            source_search_point_uv=left['source_search_point_uv'], source_search_normal=left['source_search_normal'],
            measurement_operator='geometric_midpoint_of_two_actual_material_interfaces',
            actual_source_uv=center if accepted else np.empty((0, 2), float),
            actual_source_center_candidate_uv=center, actual_RGB_entering_interface=left,
            actual_RGB_leaving_interface=right, source_flags=flags,
            observed_center_normal_interval_px=support,
            observed_paint_side_q50_normal_offsets_px=np.array([left_q[1], right_q[1]]),
            paint_footprint_is_not_centerline_uncertainty=True,
            intensity_median_or_darkest_peak_used_as_center=False, solver_sigma_changed=False,
            source_same_stroke_identity_confirmation_required=True,
            disposition='measured_paired_stroke_midline' if accepted else 'source_material_role_unknown'))
    accepted = [row for row in rows if row['disposition'] == 'measured_paired_stroke_midline']
    return dict(schema='rinkspace.source-paired-material-stroke-midline.v1',
        source_binding=frame.binding, RGB_sha256=frame.digest, readings=rows,
        actual_source_uv=np.asarray([row['actual_source_uv'] for row in accepted]).reshape(-1, 2),
        source_query_indices=np.asarray([row['source_query_index'] for row in accepted], np.int64),
        source_same_stroke_identity_confirmation_required=True,
        paint_footprint_is_not_centerline_uncertainty=True, solver_sigma_changed=False,
        world_camera_projection_input_used=False, search_points_used_as_measurements=False)


def read_connected_source_material_profiles(rgb, *, binding, search_points_uv, normals,
        search_intervals_px, transition, expected_RGB_sha256=None,
        tangent_half_width_px=2, smoothing_sigma_px=1., engineering_support_margin_px=2.,
        source_anchor_dominance_ratio=2., source_anchor_max_gap_px=32.,
        source_connection_tolerance_px=3., keep_profiles=False, _validated_frame=None):
    """Separate broad search profiles into actual connected material readings.

    A broad navigation bracket can contain the requested gray/white border,
    another paint band and codec terraces. Integrating their combined mass
    invents a point between borders. Keep all observed transitions, re-read
    each locally from RGB, then connect an ambiguous profile only when two
    unambiguous neighboring *source* readings support exactly one transition.
    Neither a navigation coordinate nor interpolation becomes output UV.
    Callers supply dense same-branch searches and separate physical identity.
    """
    if (source_anchor_dominance_ratio <= 1 or source_anchor_max_gap_px <= 0
            or source_connection_tolerance_px <= 0):
        raise ValueError("Explicit source continuity settings required")
    frame = (_MaterialProfileFrame(rgb, binding, expected_RGB_sha256)
             if _validated_frame is None else _validated_frame)
    if not isinstance(frame, _MaterialProfileFrame) or (_validated_frame is not None and rgb is not frame.rgb):
        raise ValueError('Material cache requires its exact immutable source RGB')
    same_frame(source_binding(binding, rgb.shape[:2]), frame.binding)
    if expected_RGB_sha256 is not None and expected_RGB_sha256 != frame.digest:
        raise ValueError('Material reading and bound original RGB differ')
    rgb, binding = frame.rgb, frame.binding
    broad = read_source_material_transition_profiles(rgb, binding=binding,
        search_points_uv=search_points_uv, normals=normals,
        search_intervals_px=search_intervals_px, transition=transition,
        expected_RGB_sha256=expected_RGB_sha256,
        tangent_half_width_px=tangent_half_width_px,
        smoothing_sigma_px=smoothing_sigma_px,
        engineering_support_margin_px=engineering_support_margin_px, keep_profiles=True,
        _validated_frame=frame)
    choices, selected, reasons = [], {}, {}
    for index, row in enumerate(broad["readings"]):
        # Close quantized stair steps belong to one blurred material rise.
        # A source-flat gap of five pixels preserves separate candidates.
        intervals = []
        for component in row["observed_material_intervals"]:
            start, end = component["normal_interval_px"]
            if intervals and start-intervals[-1][1] < 5:
                intervals[-1][1] = end
            else:
                intervals.append([start, end])
        local = []
        search_lo, search_hi = row["source_search_normal_interval_px"]
        for component_index, (start, end) in enumerate(intervals):
            # Bound at actual intervening source plateaus, not at the nearest
            # model peak. Four pixels include the measured blur tails/halo.
            lo = max(search_lo, start-4.)
            hi = min(search_hi, end+4.)
            if component_index:
                lo = max(lo, (intervals[component_index-1][1]+start)*.5)
            if component_index+1 < len(intervals):
                hi = min(hi, (end+intervals[component_index+1][0])*.5)
            if hi-lo < 6:
                continue
            measured = read_source_material_transition_profiles(rgb, binding=binding,
                search_points_uv=[row["source_search_point_uv"]],
                normals=[row["source_search_normal"]], search_intervals_px=[[lo, hi]],
                transition=transition, expected_RGB_sha256=expected_RGB_sha256,
                tangent_half_width_px=tangent_half_width_px,
                smoothing_sigma_px=smoothing_sigma_px,
                engineering_support_margin_px=engineering_support_margin_px,
                keep_profiles=keep_profiles, _validated_frame=frame)["readings"][0]
            measured["source_query_index"] = index
            measured["source_broad_profile_component_index"] = component_index
            measured["source_broad_navigation_interval_px"] = row["source_search_normal_interval_px"].copy()
            # A failed local reading remains in the candidate ledger.
            local.append(measured)
        choices.append(local)
        # A truncated/occluded strong edge is still a real competitor. Do not
        # erase it then promote a weaker codec terrace as the "only" border.
        ranked = sorted(local, key=lambda v: v["actual_source_signed_luminance_mass"], reverse=True)
        if (ranked and ranked[0]["disposition"] == "measured_material_transition"
                and (len(ranked) == 1 or
                ranked[0]["actual_source_signed_luminance_mass"] >=
                source_anchor_dominance_ratio*ranked[1]["actual_source_signed_luminance_mass"])):
            selected[index] = ranked[0]
            reasons[index] = "unique_or_dominant_actual_source_material_transition"
    points = np.asarray(search_points_uv, np.float64).reshape(-1, 2)
    # Source adjacency comes from the search tangent. Normal displacement is
    # measured independently; guidance does not enter the connection score.
    normal = np.asarray(normals, np.float64).reshape(-1, 2)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    tangent = np.array([-normal[0, 1], normal[0, 0]])
    station = points @ tangent
    order = np.argsort(station, kind="stable")
    anchors = [int(i) for i in order if int(i) in selected]
    for index in order:
        index = int(index)
        if index in selected:
            continue
        before = [a for a in anchors if 0 < station[index]-station[a] <= source_anchor_max_gap_px]
        after = [a for a in anchors if 0 < station[a]-station[index] <= source_anchor_max_gap_px]
        if not before or not after:
            continue
        left, right = before[-1], after[0]
        weight = (station[index]-station[left])/(station[right]-station[left])
        observed_neighbor = ((1-weight)*selected[left]["actual_source_uv"]
                             +weight*selected[right]["actual_source_uv"])
        connected = []
        for candidate in choices[index]:
            if candidate["disposition"] != "measured_material_transition":
                continue
            delta = float((candidate["actual_source_uv"]-observed_neighbor) @ normal[index])
            if abs(delta) <= source_connection_tolerance_px:
                connected.append(candidate)
        if len(connected) == 1:
            selected[index] = connected[0]
            reasons[index] = "one_actual_transition_connected_to_two_unambiguous_RGB_neighbors"
            selected[index]["source_connection_anchor_indices"] = [left, right]
            selected[index]["source_connection_interpolated_UV_is_not_measurement"] = True
    readings = []
    for index, original in enumerate(broad["readings"]):
        if index in selected:
            row = selected[index].copy()
            row["source_branch_selection_rule"] = reasons[index]
        else:
            row = original.copy()
            row["actual_source_uv"] = np.empty((0, 2), np.float64)
            row["disposition"] = "source_material_role_unknown"
            row["source_branch_selection_rule"] = "no_unique_connected_actual_source_material_transition"
        # Continuity/dominance locates a candidate; it does not authenticate
        # which physical boundary a mixed broad search originally contained.
        # Keep that evidence even when one narrowed component reads cleanly.
        if index in selected:
            _inherit_material_search_context(row, original)
        for candidate in choices[index]:
            _inherit_material_search_context(candidate, original)
        row["all_actual_source_material_candidates"] = choices[index]
        row["broad_mixed_signed_mass_not_used_as_output_UV"] = True
        readings.append(row)
    accepted = [r for r in readings if r["disposition"] == "measured_material_transition"]
    return dict(schema="rinkspace.connected-source-material-profile-reading.v1",
        source_binding=broad["source_binding"], RGB_sha256=broad["RGB_sha256"],
        readings=readings,
        actual_source_uv=np.asarray([r["actual_source_uv"] for r in accepted], np.float64).reshape(-1, 2),
        source_query_indices=np.asarray([r["source_query_index"] for r in accepted], np.int64),
        source_anchor_dominance_ratio=source_anchor_dominance_ratio,
        source_anchor_max_gap_px=source_anchor_max_gap_px,
        source_connection_tolerance_px=source_connection_tolerance_px,
        physical_id=None, physical_edge_role_verified=False,
        source_identity_authority_required=True, search_points_used_as_measurements=False,
        world_camera_projection_input_used=False, colour_difference_proxy_used=False,
        source_interpolation_used_as_measurement=False, original_raw_source_preserved=True)


def read_source_dark_stroke_profiles(rgb, *, binding, search_points_uv, normals,
        search_intervals_px, expected_RGB_sha256=None, tangent_half_width_px=2,
        smoothing_sigma_px=1., engineering_support_margin_px=2., keep_profiles=False,
        _validated_frame=None):
    """Observe dark paint bands and their centers, distinct from one border.

    Each candidate retains two actual signed RGB interfaces. The stroke
    center averages the two observed material interfaces. Darkness quantiles
    characterize the occupied paint, never the geometric centre itself.
    All competing bands remain candidates. A caller must resolve physical
    role and connectivity before assigning any candidate to a world curve.
    """
    frame = (_MaterialProfileFrame(rgb, binding, expected_RGB_sha256)
             if _validated_frame is None else _validated_frame)
    if not isinstance(frame, _MaterialProfileFrame) or (_validated_frame is not None and rgb is not frame.rgb):
        raise ValueError('Material cache requires its exact immutable source RGB')
    same_frame(source_binding(binding, rgb.shape[:2]), frame.binding)
    if expected_RGB_sha256 is not None and expected_RGB_sha256 != frame.digest:
        raise ValueError('Material reading and bound original RGB differ')
    rgb, binding = frame.rgb, frame.binding
    common = dict(binding=binding, search_points_uv=search_points_uv, normals=normals,
        search_intervals_px=search_intervals_px, expected_RGB_sha256=expected_RGB_sha256,
        tangent_half_width_px=tangent_half_width_px, smoothing_sigma_px=smoothing_sigma_px,
        engineering_support_margin_px=engineering_support_margin_px, keep_profiles=True,
        _validated_frame=frame)
    entering = read_source_material_transition_profiles(rgb, transition="light_to_dark", **common)
    leaving = read_source_material_transition_profiles(rgb, transition="dark_to_light", **common)
    records = []
    for index, (negative, positive) in enumerate(zip(entering["readings"], leaving["readings"])):
        events = []
        for sign, row in ((-1, negative), (1, positive)):
            parts = []
            for component in row["observed_material_intervals"]:
                start, end = component["normal_interval_px"]
                if parts and start-parts[-1][1] < 5:
                    parts[-1][1] = end
                else:
                    parts.append([start, end])
            events.extend((start, end, sign) for start, end in parts)
        events.sort()
        candidates = []
        offsets = positive["actual_source_normal_offsets_px"]
        intensity = positive["actual_source_luminance_profile"]
        search_lo, search_hi = positive["source_search_normal_interval_px"]
        for event_index in range(len(events)-1):
            start, fall_end, fall_sign = events[event_index]
            rise_start, end, rise_sign = events[event_index+1]
            if fall_sign != -1 or rise_sign != 1 or fall_end >= rise_start:
                continue
            border_readings = []
            for a, b, transition in ((start, fall_end, "light_to_dark"),
                                      (rise_start, end, "dark_to_light")):
                lo, hi = max(search_lo, a-4.), min(search_hi, b+4.)
                if hi-lo < 6:
                    break
                border_readings.append(read_source_material_transition_profiles(rgb,
                    binding=binding, search_points_uv=[positive["source_search_point_uv"]],
                    normals=[positive["source_search_normal"]], search_intervals_px=[[lo, hi]],
                    transition=transition, expected_RGB_sha256=expected_RGB_sha256,
                    tangent_half_width_px=tangent_half_width_px, smoothing_sigma_px=smoothing_sigma_px,
                    engineering_support_margin_px=engineering_support_margin_px,
                    keep_profiles=keep_profiles, _validated_frame=frame)["readings"][0])
            if len(border_readings) != 2:
                continue
            lo, hi = max(search_lo, start-4.), min(search_hi, end+4.)
            inside = (offsets >= lo) & (offsets <= hi)
            observed_offsets, values = offsets[inside], intensity[inside]
            if len(values) < 3:
                continue
            # Local background comes from real source pixels on the two ice
            # sides. This removes their observed illumination slope; it does
            # not impose a symmetric band, width or projected center.
            flank = min(3, len(values)//3)
            left, right = float(values[:flank].mean()), float(values[-flank:].mean())
            background = np.linspace(left, right, len(values))
            darkness = np.maximum(background-values, 0.)
            total = float(darkness.sum())
            q = np.full(3, np.nan)
            if total > 0:
                q = np.interp([.1, .5, .9], (np.cumsum(darkness)-darkness*.5)/total,
                              observed_offsets)
            flags = dict(no_observed_dark_stroke=total <= 0,
                entering_interface_unresolved=border_readings[0]["disposition"] != "measured_material_transition",
                leaving_interface_unresolved=border_readings[1]["disposition"] != "measured_material_transition",
                source_image_border_truncated=bool(negative["source_flags"]["source_image_border_truncated"]),
                source_background_flank_truncated=lo == search_lo or hi == search_hi)
            center = .5 * (border_readings[0]['actual_source_transition_candidate_uv']
                           + border_readings[1]['actual_source_transition_candidate_uv'])
            center_offset = float((center-positive['source_search_point_uv']) @ positive['source_search_normal'])
            candidate = dict(source_query_index=index, source_candidate_index=len(candidates),
                measurement_operator="geometric_midpoint_of_two_actual_material_interfaces",
                actual_source_dark_stroke_center_candidate_uv=center,
                actual_source_uv=center.copy() if not any(flags.values()) else np.empty((0, 2), np.float64),
                actual_RGB_entering_interface=border_readings[0], actual_RGB_leaving_interface=border_readings[1],
                source_normal_q10_q50_q90_darkness_offsets_px=q,
                observed_dark_stroke_normal_footprint_px=np.array([q[0]-center_offset, q[2]-center_offset]),
                footprint_is_not_centerline_covariance_or_solver_sigma=True,
                actual_source_darkness_mass=total,
                source_background_luminance_endpoints=np.array([left, right]), source_flags=flags,
                disposition="measured_dark_stroke_center_candidate" if not any(flags.values()) else "source_material_role_unknown",
                source_physical_role_verified=False, physical_id=None,
                actual_source_center_is_not_border_midpoint=False,
                darkness_mass_quantile_used_as_geometric_center=False,
                model_or_interpolated_UV_used_as_measurement=False)
            if keep_profiles:
                candidate.update(actual_source_normal_offsets_px=observed_offsets,
                    actual_source_luminance_profile=values, actual_source_background_profile=background,
                    actual_source_darkness_profile=darkness)
            candidates.append(candidate)
        records.append(dict(source_query_index=index, source_search_point_uv=positive["source_search_point_uv"],
            source_search_normal=positive["source_search_normal"],
            source_search_normal_interval_px=positive["source_search_normal_interval_px"],
            all_actual_source_dark_stroke_candidates=candidates, physical_id=None,
            physical_edge_role_verified=False, physical_role_resolution_required=True,
            source_candidate_selected_automatically=False))
    return dict(schema="rinkspace.source-dark-stroke-center-profile-candidates.v1",
        source_binding=entering["source_binding"], RGB_sha256=entering["RGB_sha256"], readings=records,
        measurement_operator="geometric_midpoint_of_two_actual_material_interfaces",
        physical_role_resolution_required=True, actual_source_uv_not_assigned_to_world=True,
        source_center_and_material_interface_operators_are_distinct=True,
        model_or_interpolated_UV_used_as_measurement=False, solver_sigma_changed=False)


def read_source_chromatic_stroke_centers(rgb, *, binding, search_points_uv, normals,
        search_intervals_px, material_signal="red_minus_green", expected_RGB_sha256=None,
        tangent_half_width_px=2, smoothing_sigma_px=1., keep_profiles=False,
        _validated_frame=None):
    """Read a colored paint center from its complete actual RGB band.

    The two ice-side transitions authenticate that the full band was read.
    Its geometric center averages the two actual material interfaces. Color
    mass only characterizes the occupied paint. Paint width/blur footprint
    are separate evidence and never a centerline zero-loss interval or sigma.
    Every disconnected color component remains a candidate with no physical
    role assigned. A clipped band never supplies an accepted center.
    """
    frame = (_MaterialProfileFrame(rgb, binding, expected_RGB_sha256)
             if _validated_frame is None else _validated_frame)
    if not isinstance(frame, _MaterialProfileFrame) or (_validated_frame is not None and rgb is not frame.rgb):
        raise ValueError('Material cache requires its exact immutable source RGB')
    same_frame(source_binding(binding, rgb.shape[:2]), frame.binding)
    if expected_RGB_sha256 is not None and expected_RGB_sha256 != frame.digest:
        raise ValueError('Material reading and bound original RGB differ')
    channels = {"red_minus_green": (0, 1), "blue_minus_red": (2, 0)}
    if material_signal not in channels:
        raise ValueError("Declare the actual colored paint material signal")
    positive_channel, negative_channel = channels[material_signal]
    signal = frame.rgb[..., positive_channel].astype(np.float64)-frame.rgb[..., negative_channel]
    points = np.asarray(search_points_uv, np.float64).reshape(-1, 2)
    directions = np.asarray(normals, np.float64).reshape(-1, 2)
    intervals = np.asarray(search_intervals_px, np.float64).reshape(-1, 2)
    if (len(points) != len(directions) or len(points) != len(intervals)
            or not all(np.isfinite(a).all() for a in (points, directions, intervals))
            or (intervals[:, 1]-intervals[:, 0] < 20).any()
            or not isinstance(tangent_half_width_px, (int, np.integer)) or tangent_half_width_px < 0
            or not np.isfinite(smoothing_sigma_px) or smoothing_sigma_px <= 0):
        raise ValueError("Matching finite complete-band searches and actual tangent pixels required")
    lengths = np.linalg.norm(directions, axis=1)
    if (lengths <= 1e-9).any():raise ValueError("Source normals must be nonzero")
    directions = directions/lengths[:, None]
    records = []
    for query, (point, normal, interval) in enumerate(zip(points, directions, intervals)):
        offsets = np.arange(interval[0], interval[1]+.25, 1., dtype=np.float64)
        tangent = np.array([-normal[1], normal[0]])
        samples = point[None, None]+offsets[None, :, None]*normal+tangent[None, None]*np.arange(-tangent_half_width_px, tangent_half_width_px+1)[:, None, None]
        valid = ((samples[..., 0] >= 0) & (samples[..., 0] <= rgb.shape[1]-1)
                 & (samples[..., 1] >= 0) & (samples[..., 1] <= rgb.shape[0]-1))
        samples = np.rint(samples.astype(np.float32)*32.)/32.
        profile = cv2.remap(signal, samples[..., 0].astype(np.float32), samples[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).mean(0)
        flank = min(10, max(3, len(profile)//10))
        endpoints = np.array([np.median(profile[:flank]), np.median(profile[-flank:])])
        background = np.linspace(*endpoints, len(profile))
        paint = np.maximum(profile-background, 0.)
        smoothed = cv2.GaussianBlur(paint[:, None], (1, 5), smoothing_sigma_px).ravel()
        second = np.diff(smoothed, n=2)
        noise = float(1.4826*np.median(np.abs(second-np.median(second))))
        # A color component is observed above local ice/codec fluctuations;
        # this threshold selects material, it is not its location uncertainty.
        floor = max(2., 3*noise, .1*float(smoothed.max()))
        supported = smoothed > floor
        starts = np.flatnonzero(supported & ~np.r_[False, supported[:-1]])
        ends = np.flatnonzero(supported & ~np.r_[supported[1:], False])
        components = []
        for a, b in zip(starts, ends):
            if b-a < 3:continue
            if components and a-components[-1][1]-1 <= 3:components[-1][1] = int(b)
            else:components.append([int(a), int(b)])
        candidates = []
        for a, b in components:
            lo, hi = max(0, a-10), min(len(profile)-1, b+10)
            mass = paint[lo:hi+1]
            total = float(mass.sum())
            if total <= 0:continue
            quantiles = np.interp([.1, .5, .9], (np.cumsum(mass)-mass*.5)/total, offsets[lo:hi+1])
            center_offset = float(quantiles[1])
            midpoint = .5*(offsets[:-1]+offsets[1:])
            gradient = np.diff(smoothed)
            borders = []
            for sign, region in ((1, (midpoint >= offsets[lo]) & (midpoint <= center_offset)),
                                 (-1, (midpoint >= center_offset) & (midpoint <= offsets[hi]))):
                gradient_mass = np.where(region, np.maximum(sign*gradient, 0.), 0.)
                amount = float(gradient_mass.sum())
                q = np.interp([.1, .5, .9], np.r_[0., np.cumsum(gradient_mass)]/amount, offsets) if amount > 0 else np.full(3, np.nan)
                borders.append(dict(material_transition="ice_to_colored_paint" if sign == 1 else "colored_paint_to_ice",
                    actual_source_uv=point+normal*q[1], actual_signed_color_transition_q10_q50_q90_offsets_px=q,
                    actual_signed_color_transition_mass=amount))
            flags = dict(source_image_border_truncated=not bool(valid[:, lo:hi+1].all()),
                complete_band_search_truncated=lo == 0 or hi == len(profile)-1,
                two_ice_side_transitions_unresolved=any(border["actual_signed_color_transition_mass"] <= floor for border in borders),
                competing_disconnected_colored_paint=len(components) > 1)
            center = .5 * (borders[0]['actual_source_uv']+borders[1]['actual_source_uv'])
            center_offset = float((center-point) @ normal)
            candidate = dict(source_candidate_index=len(candidates), measurement_operator="geometric_midpoint_of_two_actual_material_interfaces",
                actual_source_center_candidate_uv=center, actual_source_uv=center.copy() if not any(flags.values()) else np.empty((0, 2), np.float64),
                actual_RGB_band_interfaces=borders, source_color_mass_q10_q50_q90_offsets_px=quantiles,
                observed_paint_normal_footprint_px=quantiles-center_offset, actual_source_color_mass=total,
                paint_width_is_not_centerline_allowance=True, centerline_zero_loss_interval_px=None,
                solver_sigma_changed=False, source_flags=flags,
                color_mass_quantile_used_as_geometric_center=False,
                disposition="measured_colored_stroke_center_candidate" if not any(flags.values()) else "source_material_role_unknown",
                source_physical_role_verified=False, physical_id=None, model_or_interpolated_UV_used_as_measurement=False)
            if keep_profiles:candidate.update(actual_source_normal_offsets_px=offsets[lo:hi+1], actual_source_color_mass_profile=mass)
            candidates.append(candidate)
        reading = dict(source_query_index=query, source_search_point_uv=point, source_search_normal=normal,
            source_search_normal_interval_px=interval, all_actual_source_chromatic_stroke_candidates=candidates,
            source_material_signal=material_signal, local_ice_color_background_endpoints=endpoints,
            source_signal_noise_scale=noise, physical_role_resolution_required=True, source_candidate_selected_automatically=False)
        if keep_profiles:reading.update(actual_source_normal_offsets_px=offsets, actual_source_color_profile=profile, actual_source_color_background_profile=background)
        records.append(reading)
    return dict(schema="rinkspace.source-chromatic-stroke-center-profile-candidates.v1", source_binding=frame.binding,
        RGB_sha256=frame.digest, material_signal=material_signal, readings=records,
        paint_footprint_is_not_centerline_uncertainty_or_zero_loss_interval=True,
        model_or_interpolated_UV_used_as_measurement=False, solver_sigma_changed=False, physical_role_resolution_required=True)


def _ellipse_model(ellipse):
    center, size, degrees = ellipse
    center, radii = np.asarray(center, float), np.asarray(size, float)*.5
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    if not np.isfinite(center).all() or not np.isfinite(radii).all() or radii.min() <= 0:
        return None
    Q = rotation @ np.diag(1/radii**2) @ rotation.T
    return center, radii, rotation, Q


def _ellipse_residual(uv, model):
    center, _, _, Q = model
    delta = uv-center
    gradient = 2*delta @ Q
    value = np.einsum("ni,ij,nj->n", delta, Q, delta)-1
    return np.abs(value)/np.maximum(np.linalg.norm(gradient, axis=1), 1e-9)


def _reread_conic_source_profiles(rgb, search, normals, model, *, radius=12., material_orientation=None):
    """Full original RGB profiles; wide edges retained and no closest-peak rule."""
    offsets = np.arange(-radius, radius+.25, .5, dtype=np.float32)
    uv = search[:, None]+normals[:, None]*offsets[None, :, None]
    height, width = rgb.shape[:2]
    valid = (uv[:, :, 0] >= 0) & (uv[:, :, 0] < width-1) & (uv[:, :, 1] >= 0) & (uv[:, :, 1] < height-1)
    profile = cv2.remap(rgb.astype(np.float32)/255., uv[:, :, 0].astype(np.float32),
                        uv[:, :, 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    diff = profile[:, 1:]-profile[:, :-1]
    gradient = np.sqrt((diff*diff).mean(2))/.5
    usable = valid[:, 1:] & valid[:, :-1]
    score = np.where(usable, gradient, -np.inf)
    index = score.argmax(1)
    chosen = score[np.arange(len(score)), index]
    contrast = np.sqrt((profile.max(1)-profile.min(1))**2).mean(1)
    luminance = profile.mean(2)
    second = np.diff(luminance, n=2, axis=1)
    second_median = np.median(second, axis=1)
    noise = 1.4826*np.median(np.abs(second-second_median[:, None]), axis=1)
    contrast_threshold = np.maximum(2/255., 6*noise) if model is None else np.full(len(search), .025)
    local_gradient_threshold = np.maximum(1/1020., 3*noise) if model is None else np.full(len(search), .025/4.)
    peak_offsets = (offsets[:-1]+offsets[1:])*.5
    actual = search+normals*peak_offsets[index, None]
    source_valid = np.isfinite(chosen) & (contrast >= contrast_threshold)
    residual = _ellipse_residual(actual, model) if model is not None else np.abs(peak_offsets[index])
    coherent = source_valid & (residual <= 2.5)
    previous = np.pad(score[:, :-1], ((0, 0), (1, 0)), constant_values=-np.inf)
    following = np.pad(score[:, 1:], ((0, 0), (0, 1)), constant_values=-np.inf)
    # Adjacent concentric material borders are genuine separate peaks. Retain
    # every locally measurable gradient; the source conic organizes a group
    # without promoting the guide center/nearest peak into a measurement.
    local = usable & (score >= previous) & (score > following) & (score >= local_gradient_threshold[:, None])
    local &= (contrast >= contrast_threshold)[:, None]
    rows, cols = np.nonzero(local)
    pool = search[rows]+normals[rows]*peak_offsets[cols, None]
    pool_residual = _ellipse_residual(pool, model) if model is not None else np.abs(peak_offsets[cols])
    matching = pool_residual <= (2.5 if model is not None else radius)
    signed_gradient = (luminance[:, 1:]-luminance[:, :-1])/.5
    before_indices = (cols[:, None]+np.arange(-12, -5)[None]).clip(0, luminance.shape[1]-1)
    after_indices = (cols[:, None]+np.arange(6, 13)[None]).clip(0, luminance.shape[1]-1)
    inside = luminance[rows[:, None], before_indices].mean(1)
    outside = luminance[rows[:, None], after_indices].mean(1)
    material_delta = outside-inside
    material_significant = np.abs(material_delta) > 3*noise[rows]+1/255.
    material_significant &= (valid[rows[:, None], before_indices].all(1) &
                             valid[rows[:, None], after_indices].all(1))
    orientations = []
    for sign in ((material_orientation,) if material_orientation is not None else (-1, 1)):
        supported = matching & material_significant & (np.sign(material_delta) == sign)
        ids = np.unique(rows[supported])
        orientations.append((len(ids), sign, supported))
    _, material_sign, supported_material = max(orientations, key=lambda row: row[0])
    organized = np.zeros(len(search), bool)
    organized[rows[supported_material]] = True
    half = usable & (score >= chosen[:, None]*.5)
    position = np.arange(score.shape[1])[None]
    lo = np.where((position < index[:, None]) & ~half, position, -1).max(1)
    hi = np.where((position > index[:, None]) & ~half, position, score.shape[1]).min(1)
    width = (hi-lo-1)*.5
    pool_half = score[rows] >= gradient[rows, cols, None]*.5
    pool_lo = np.where((position < cols[:, None]) & ~pool_half, position, -1).max(1)
    pool_hi = np.where((position > cols[:, None]) & ~pool_half, position, score.shape[1]).min(1)
    pool_width = (pool_hi-pool_lo-1)*.5
    return dict(actual_source_peak_uv=actual, source_valid=source_valid, source_conic_coherence=coherent,
                all_actual_source_peak_uv=pool, all_actual_source_peak_profile_indices=rows,
                all_actual_source_peak_gradient=gradient[rows, cols],
                all_actual_source_peak_fwhm_px=pool_width,
                all_actual_source_peak_wide_relative_to_old_extractor=pool_width > 4,
                all_actual_source_peak_weak_relative_to_old_extractor=gradient[rows, cols] < .02,
                source_image_conic_group_membership=matching,
                actual_source_signed_luminance_gradient=signed_gradient[rows, cols],
                actual_source_inside_luminance=inside, actual_source_outside_luminance=outside,
                actual_source_inside_outside_material_delta=material_delta,
                source_material_noise_scale=noise[rows],
                full_normal_source_contrast=contrast,
                source_profile_contrast_threshold=contrast_threshold,
                source_profile_gradient_threshold=local_gradient_threshold,
                significant_single_material_transition=material_significant,
                source_material_transition_orientation="lightening_outward_candidate" if material_sign > 0 else "darkening_outward_candidate",
                source_material_conic_group_membership=supported_material,
                full_normal_source_luminance_profile=luminance,
                full_normal_source_signed_gradient=signed_gradient,
                source_image_conic_supported_profiles=organized,
                source_gradient=chosen, source_peak_fwhm_px=width, wide_source_peak=width > 4,
                weak_relative_to_old_extractor=chosen < .02,
                peak_position_rule="strongest_actual_RGB_gradient_over_entire_normal_search_range",
                width_or_weakness_used_as_hard_veto=False, profile_radius_native_px=radius)


def _smooth_source_contour_runs(run, spacing):
    """Split source connectivity at sharp junctions, retaining native ordering."""
    if len(run)<24:
        return []
    poly=cv2.approxPolyDP(run.astype(np.float32),spacing*.5,False)[:,0]
    if len(poly)<3:
        return [run]
    directions=np.diff(poly,axis=0)
    directions/=np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-9)
    corners=np.flatnonzero((directions[:-1]*directions[1:]).sum(1)<np.cos(.5))+1
    cuts=[0]+[int(((run-poly[index])**2).sum(1).argmin()) for index in corners]+[len(run)-1]
    cuts=sorted(set(cuts))
    return [run[a:b+1] for a,b in zip(cuts[:-1],cuts[1:]) if b-a>=24]


def partial_source_material_readings(rgb, *, binding, source_ice_search_mask,
                                     source_ice_search_binding, sample_spacing_px=6.,
                                     source_paint_probabilities=None, probability_source_binding=None):
    """Read curved gray-material contours including partial image-border arcs.

    The ice mask only locates candidate RGB regions. Its artificial outline and
    the native frame outline cannot become measurements. RGB material profiles
    are reread at every retained contour sample; no ellipse closes the arc.
    Multiple source contrast scales compete, with straight shadows retained as
    unresolved candidates. This branch assigns neither paint roles nor names.
    """
    rgb = np.asarray(rgb)
    binding = source_binding(binding, rgb.shape[:2])
    same_frame(binding, source_ice_search_binding)
    scope = _numpy(source_ice_search_mask).astype(bool).squeeze()
    height, width = rgb.shape[:2]
    if scope.shape != (height, width) or not scope.any():
        raise ValueError("Partial material search needs a same-source native ice-region candidate")
    paint_probability = None
    if source_paint_probabilities is not None:
        if probability_source_binding is None:
            raise ValueError("Source paint probabilities need their actual source binding")
        same_frame(binding, probability_source_binding)
        paint_probability = _numpy(source_paint_probabilities).squeeze()
        if (paint_probability.shape != (height, width) or not np.isfinite(paint_probability).all()
                or (paint_probability < 0).any() or (paint_probability > 1).any()):
            raise ValueError("Source paint probabilities must be a finite native [H,W] search field")
    luminance = rgb.astype(np.float32).mean(2)/255.
    chroma = (rgb.max(2).astype(float)-rgb.min(2))/255.
    # Near-achromatic material is a candidate, not a paint label. The complete
    # RGB profiles below retain signs, material levels and conflicting peaks.
    chroma_reference = chroma[scope]
    neutral_limit = float(np.quantile(chroma_reference, .75))
    neutral = chroma <= max(2/255., neutral_limit)
    sy, sx = np.nonzero(scope)
    extent = min(np.ptp(sy)+1, np.ptp(sx)+1)
    scales = [2**power+1 for power in range(3, int(np.ceil(np.log2(max(9, extent)))))]
    candidates, rejected = [], []
    scope_distance = cv2.distanceTransform(scope.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    yy,xx=np.indices((height,width))
    native_border_distance=np.minimum.reduce([yy,xx,height-1-yy,width-1-xx])
    for diameter in scales:
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(diameter,diameter))
        background = cv2.morphologyEx(luminance, cv2.MORPH_CLOSE,kernel)
        # Closing assumes an observed bright side beyond both ends. A band
        # clipped by the native frame has no such samples. Use a plateau from
        # the *observed* neighborhood only as an illumination search baseline;
        # the current RGB normal profiles still supply every output point.
        observed_plateau=cv2.dilate(luminance,kernel)
        border_search=native_border_distance<=diameter//2
        background=np.where(border_search,np.maximum(background,observed_plateau),background)
        contrast = np.maximum(background-luminance, 0)
        values = contrast[scope & neutral]
        positive = values[values > 1/255.]
        if len(positive) < 24:
            continue
        cap = float(np.quantile(positive, .95))
        histogram = np.rint(np.minimum(values, cap)/max(cap, 1/255.)*255).astype(np.uint8)
        threshold, _ = cv2.threshold(histogram, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        threshold = threshold/255.*cap
        # The broadcast's gray ink can share a blue illumination cast. Neutral
        # pixels estimate the background histogram; chroma cannot sever a real
        # connected gray contour into tiny islands and substitute texture loops.
        mask = ((contrast > threshold) & scope).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        for index, contour in enumerate(contours):
            uv = contour[:, 0].astype(np.float32)
            if len(uv) < 24:
                continue
            # Remove only artificial source-image/scope closure, independently
            # of paint confidence; never turn a mask/frame edge into RGB UV.
            actual_domain = ((uv[:, 0] > 1) & (uv[:, 0] < width-2) &
                             (uv[:, 1] > 1) & (uv[:, 1] < height-2))
            ix = uv.astype(int)
            actual_domain &= scope_distance[ix[:, 1], ix[:, 0]] > 2
            if actual_domain.sum() < 24:
                continue
            if not actual_domain.all():
                cut = int(np.flatnonzero(~actual_domain)[0])
                uv = np.roll(uv, -cut, axis=0)
                actual_domain = np.roll(actual_domain, -cut)
            # Native contour order is preserved. Perimeter border runs are
            # separate, including where a foreground material leaves the frame.
            run_values = _visible_search_runs(actual_domain)
            for run_id in np.unique(run_values):
                run = uv[actual_domain][run_values == run_id]
                source_loop_has_light_hole = bool(hierarchy[0,index,2]>=0 or hierarchy[0,index,3]>=0)
                native_image_clipped = bool(
                    (contour[:,0,0]<=1).any() or (contour[:,0,0]>=width-2).any() or
                    (contour[:,0,1]<=1).any() or (contour[:,0,1]>=height-2).any())
                for run in _smooth_source_contour_runs(run,sample_spacing_px):
                    if len(run) < 24:
                        continue
                    if np.linalg.norm(run[0]-run[-1]) < 2 and actual_domain.all():
                        closed = True
                    else:
                        closed = False
                    deltas = np.diff(run, axis=0)
                    arc_distance = np.r_[0, np.cumsum(np.linalg.norm(deltas, axis=1))]
                    locations = np.arange(0, arc_distance[-1], sample_spacing_px)
                    if len(locations) < 12:
                        continue
                    seeds = np.stack([np.interp(locations, arc_distance, run[:, channel]) for channel in (0, 1)], 1)
                    step = max(2., sample_spacing_px*2)
                    before = np.stack([np.interp(np.maximum(0, locations-step), arc_distance, run[:, channel]) for channel in (0, 1)], 1)
                    after = np.stack([np.interp(np.minimum(arc_distance[-1], locations+step), arc_distance, run[:, channel]) for channel in (0, 1)], 1)
                    tangent = after-before
                    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
                    normals = np.stack([tangent[:, 1], -tangent[:, 0]], 1)
                    probes = seeds+normals*3
                    opposite = seeds-normals*3
                    positive_side = cv2.remap(mask, probes[:, 0].astype(np.float32)[:, None],probes[:, 1].astype(np.float32)[:, None],cv2.INTER_NEAREST).ravel()
                    negative_side = cv2.remap(mask, opposite[:, 0].astype(np.float32)[:, None],opposite[:, 1].astype(np.float32)[:, None],cv2.INTER_NEAREST).ravel()
                    reverse = positive_side > negative_side
                    normals[reverse] *= -1
                    # A line/shadow band does not become an open circle by merely
                    # spanning the image. Source tangent curvature must be present.
                    center = seeds.mean(0)
                    _, singular, _ = np.linalg.svd(seeds-center, full_matrices=False)
                    line_width = singular[-1]/np.sqrt(len(seeds))
                    turning = 1-np.abs((tangent[0]*tangent[-1]).sum())
                    typed_support = (cv2.remap(paint_probability.astype(np.float32), seeds[:,0,None].astype(np.float32),
                        seeds[:,1,None].astype(np.float32), cv2.INTER_LINEAR).ravel() if paint_probability is not None else
                        np.zeros(len(seeds), np.float32))
                    typed_search_support = bool((typed_support >= .2).mean() >= .5)
                    insufficient_turning = bool(line_width < 2 or (not closed and turning < .015))
                    # A distant physical ring can be nearly straight in native
                    # perspective. Geometry does not veto actual signed RGB
                    # borders when source-only paint evidence independently
                    # supports the local material search. That evidence is a
                    # navigation hypothesis, never a physical circle/name.
                    if insufficient_turning and not typed_search_support:
                        rejected.append(dict(source_search_scale_px=diameter, native_source_contour_index=index,
                            reason="gray_source_material_is_straight_or_has_insufficient_turning",
                            source_search_contour_uv=run, physical_role_unresolved=True))
                        continue
                    try:
                        image_model = _ellipse_model(cv2.fitEllipseDirect(seeds.astype(np.float32)))
                    except cv2.error:
                        image_model = None
                    image_residual = (_ellipse_residual(seeds,image_model) if image_model is not None else
                                      np.full(len(seeds),np.inf))
                    conic_consistent = bool(image_model is not None and
                        np.quantile(image_residual,.9) <= sample_spacing_px and
                        image_model[1].max() <= max(height,width)*2 and
                        image_model[1].min()/image_model[1].max() >= .01)
                    if not conic_consistent and not typed_search_support:
                        rejected.append(dict(source_search_scale_px=diameter, native_source_contour_index=index,
                            reason="source_material_contour_is_irregular_not_one_smooth_visible_curved_boundary",
                            source_search_contour_uv=run,source_curve_image_conic_residual_px=image_residual,
                            fitted_pixels_used_as_measurements=False))
                        continue
                    profiles = _reread_conic_source_profiles(rgb, seeds, normals, None, material_orientation=1)
                    supported = profiles["source_image_conic_supported_profiles"]
                    if supported.sum() < 12 or supported.mean() < .5:
                        rejected.append(dict(source_search_scale_px=diameter, native_source_contour_index=index,
                            reason="gray_contour_has_insufficient_signed_RGB_material_transition",
                            source_search_contour_uv=run, full_curve_source_RGB_profiles=profiles))
                        continue
                    candidates.append(dict(id=f"source_local_material_curve_{len(candidates)}",kind="ice_paint",
                        primitive="free_curve",physical_role="source_paint_role_unresolved", physical_id=None,
                        identity_confirmed=False, origin="source_annotation_search_hint", source_binding=binding,
                        points_uv=seeds,normals=normals,segment_ids=np.zeros(len(seeds),np.int32),closed=False,
                        visible_original_material_contour_closed=closed,
                        material_topological_role_candidate="gray_material_visible_boundary_role_unresolved",
                        material_contour_depth=int(hierarchy[0,index,3]>=0),
                        source_gray_search_scale_px=diameter,source_gray_contrast_threshold=threshold,
                        source_gray_achromatic_candidate_limit=neutral_limit,
                        source_actual_image_border_intersection=native_image_clipped,
                        source_material_light_hole_parent_or_child=source_loop_has_light_hole,
                        native_frame_boundary_pixels_removed_as_artificial_closure=True,
                        source_ice_mask_is_only_search_navigation=True,
                    clipped_material_illumination_baseline="observed_neighbor_bright_plateau_search_only",
                        source_curve_image_conic_residual_px=image_residual,
                        source_image_shape_used_only_for_curvature_diagnostic=True,
                        source_local_paint_probability=typed_support,
                        source_probability_used_only_as_material_search_support=True,
                        local_perspective_fragment_insufficient_turning=insufficient_turning,
                        local_source_conic_consistent=conic_consistent,
                        geometry_did_not_veto_typed_signed_RGB_fragment=typed_search_support,
                        full_curve_source_RGB_profiles=profiles, source_ice_domain_verified=False,
                        status="partial_or_single_gray_material_boundary_with_actual_RGB_peaks",
                        fitted_points_used_as_measurements=False,localization_accuracy_accepted=False))
    # Distinct scales are alternative source material searches. Do not count
    # the same current RGB peak cloud repeatedly or force all gray curves into
    # one object. Keep alternative candidates in the unresolved ledger.
    selected, alternatives = [], []
    for candidate in sorted(candidates,key=lambda item: int(item["full_curve_source_RGB_profiles"]["source_image_conic_supported_profiles"].sum()),reverse=True):
        pool = candidate["full_curve_source_RGB_profiles"]
        actual = pool["all_actual_source_peak_uv"][pool["source_material_conic_group_membership"]]
        duplicate=False
        for previous in selected:
            p=previous["full_curve_source_RGB_profiles"]
            other=p["all_actual_source_peak_uv"][p["source_material_conic_group_membership"]]
            # Image-local source peak neighborhoods only, no conic projection.
            near=np.sqrt(((actual[:,None]-other[None])**2).sum(2)).min(1) <= sample_spacing_px
            if near.mean() > .8:
                duplicate=True;break
        (alternatives if duplicate else selected).append(candidate)
    return dict(source_binding=binding,source_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        guides=selected,source_shape_readings=_direct_source_shape_readings(selected),
        alternative_source_material_candidates=alternatives,rejected_source_material_candidates=rejected,
        original_source_contours_not_erased=True, source_ice_mask_used_as_measurements=False,
        native_frame_boundary_used_as_measurements=False, physical_marker_names_inferred=False,
        source_ice_domain_verified=False, is_training_label=False,accuracy_accepted=False)


def _source_material_topology(rgb, nodes, groups, spacing, *, minimum_nested_layers=3,
                              source_navigation_regions=None):
    """Local RGB dark-material hierarchy; mask thresholds are search only.

    A fitted upper cap can cross two annulus borders with the same gradient
    sign. It is not the boundary of a closed source material component. This
    source-derived hierarchy supplies full outer/hole/inner contours including
    low-gradient turning ends that the old extractor omitted.
    """
    luminance = rgb.astype(np.float32).mean(2)/255.
    height, width = luminance.shape
    candidates, evidence = [], []
    for group_index, group in enumerate(groups):
        support = np.concatenate([nodes[index]["uv"] for index in group])
        spread = np.ptp(support, axis=0)
        if not group or spread.min() < 8:
            continue
        # Scale comes from actual distributed source support, not world radius.
        if source_navigation_regions is not None:
            navigation = source_navigation_regions[group_index]
            lo = np.asarray(navigation["source_RGB_search_bbox_xyxy"][:2], int)
            hi = np.asarray(navigation["source_RGB_search_bbox_xyxy"][2:], int)
            diameter = int(navigation["source_illumination_search_kernel_diameter_px"]) | 1
        else:
            diameter = int(np.ceil(spread.min()+2*spacing)) | 1
            margin = np.maximum(spread*.5, diameter)
            lo = np.floor(support.min(0)-margin).astype(int).clip([0, 0], [width-1, height-1])
            hi = np.ceil(support.max(0)+margin).astype(int).clip([1, 1], [width, height])
        crop = luminance[lo[1]:hi[1], lo[0]:hi[0]]
        if min(crop.shape) < diameter:
            continue
        background = cv2.morphologyEx(crop, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
        contrast = np.maximum(background-crop, 0)
        source_histogram = np.rint(contrast*255).clip(0, 255).astype(np.uint8)
        threshold, mask = cv2.threshold(source_histogram, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        record = dict(source_component_indices=[nodes[index]["index"] for index in group],
                      source_navigation_bbox_xyxy=[*lo.tolist(), *hi.tolist()],
                      source_contrast_otsu_threshold=threshold/255.,
                      source_illumination_search_kernel_diameter_px=diameter,
                      material_mask_is_measurement=False, material_hierarchies=[],
                      all_source_material_contour_candidates=[])
        if hierarchy is None:
            evidence.append(record); continue
        hierarchy = hierarchy[0]
        for index, contour in enumerate(contours):
            record["all_source_material_contour_candidates"].append(dict(
                candidate_contour_index=index, original_RGB_search_contour_uv=contour[:, 0]+lo,
                hierarchy_next_previous_child_parent=hierarchy[index],
                source_material_search_area_px=float(cv2.contourArea(contour)),
                candidate_is_actual_boundary_measurement=False))
        for outer_index, outer in enumerate(contours):
            hole_index = int(hierarchy[outer_index, 2])
            if hierarchy[outer_index, 3] != -1 or hole_index < 0:
                continue
            # Tree order has no semantic authority: the first child can be a
            # tiny ice-texture hole. Search the largest observed light hole;
            # all other native contour hypotheses remain in the evidence.
            children = []
            while hole_index >= 0:
                children.append(hole_index)
                hole_index = int(hierarchy[hole_index, 0])
            hole_index = max(children, key=lambda index: cv2.contourArea(contours[index]))
            hole = contours[hole_index]
            inner_index = int(hierarchy[hole_index, 2])
            if inner_index >= 0:
                children = []
                while inner_index >= 0:
                    children.append(inner_index)
                    inner_index = int(hierarchy[inner_index, 0])
                inner_index = max(children, key=lambda index: cv2.contourArea(contours[index]))
                if minimum_nested_layers < 3 and cv2.contourArea(contours[inner_index]) <= diameter:
                    inner_index = -1
            if inner_index < 0 and minimum_nested_layers >= 3:
                continue
            inner = contours[inner_index] if inner_index >= 0 else None
            loop = [outer, hole]+([inner] if inner is not None else [])
            # Original RGB components must be nested closed regions, not
            # a morphological closure of missing arcs or a navigation border.
            if any(len(contour) < 12 or cv2.contourArea(contour) <= diameter for contour in loop):
                continue
            if any((contour[:, 0, 0] <= 0).any() or (contour[:, 0, 1] <= 0).any() or
                   (contour[:, 0, 0] >= crop.shape[1]-1).any() or
                   (contour[:, 0, 1] >= crop.shape[0]-1).any() for contour in loop):
                continue
            local_models = []
            for contour in loop:
                model = _ellipse_model(cv2.fitEllipseDirect((contour[:, 0]+lo).astype(np.float32)))
                if model is None:
                    break
                residual = _ellipse_residual(contour[:, 0]+lo, model)
                if np.quantile(residual, .9) > max(3., spacing):
                    break
                local_models.append(model)
            if len(local_models) != len(loop):
                continue
            outer_area = cv2.contourArea(outer)
            # Reject tiny texture holes. All three source loops should have
            # distributed image area and a common neighborhood.
            if min(cv2.contourArea(contour) for contour in loop[1:])/outer_area < .025:
                continue
            if max(np.linalg.norm(model[0]-local_models[0][0]) for model in local_models) > diameter:
                continue
            outer_uv = outer[:, 0]+lo
            support_near = np.array([abs(cv2.pointPolygonTest(outer, tuple(map(float, point-lo)), True))
                                     < diameter for point in support])
            if support_near.sum() < 24:
                continue
            topology_id = f"source_material_topology_{group_index}_{outer_index}"
            record["material_hierarchies"].append(dict(id=topology_id,
                closed_source_material_outer_contour_uv=outer_uv,
                closed_source_light_hole_contour_uv=hole[:, 0]+lo,
                closed_source_inner_material_contour_uv=inner[:, 0]+lo if inner is not None else None,
                actual_source_support_count=int(support_near.sum()),
                observed_layout=("dark_annulus_light_hole_dark_inner_component" if inner is not None else
                                 "dark_annulus_light_hole_without_confirmed_inner_component"),
                pixel_masks_are_search_only=True, physical_identity_verified=False))
            for depth, (contour, model) in enumerate(zip(loop, local_models)):
                candidates.append(dict(model=model, source_material_contour_uv=contour[:, 0]+lo,
                    topology_id=topology_id, material_contour_depth=depth,
                    material_orientation=1 if depth % 2 == 0 else -1,
                    source_component_indices=record["source_component_indices"],
                    source_material_topology_verified_as_search=True))
        evidence.append(record)
    return candidates, evidence


def _visible_search_runs(visible):
    """Clipped searches stay separate; never join across unseen image pixels."""
    transitions = visible & ~np.roll(visible, 1)
    runs = np.cumsum(transitions).astype(np.int32)
    # Keep first/last phase fragments separate. Merging their IDs without
    # rotating contour order would join across the intervening missing arc.
    return runs[visible]


def _direct_source_shape_readings(guides):
    """Return original RGB peaks separately from every flow/mask/ellipse hint."""
    readings = []
    for guide in guides:
        profiles = dict(guide["full_curve_source_RGB_profiles"])
        # Legacy inline JSON preserves numerical values, but not ndarray
        # dtype/shape. Normalize only for vector indexing; never claim that
        # an inferred computational dtype restores the former array bytes.
        fields=("temporal_material_signature_group_membership","source_material_conic_group_membership",
                "source_image_conic_supported_profiles","all_actual_source_peak_profile_indices",
                "all_actual_source_peak_gradient","all_actual_source_peak_uv","all_actual_source_peak_fwhm_px",
                "all_actual_source_peak_wide_relative_to_old_extractor",
                "all_actual_source_peak_weak_relative_to_old_extractor",
                "actual_source_signed_luminance_gradient","actual_source_inside_outside_material_delta")
        legacy=[]
        for key in fields:
            if key in profiles and not isinstance(profiles[key],np.ndarray):
                legacy.append(key)
                profiles[key]=np.asarray(profiles[key])
        profiles['all_actual_source_peak_uv']=profiles['all_actual_source_peak_uv'].reshape(-1,2)
        for key in ('temporal_material_signature_group_membership','source_material_conic_group_membership'):
            if key in profiles and key in legacy:profiles[key]=profiles[key].astype(bool)
        if 'all_actual_source_peak_profile_indices' in legacy:
            profiles['all_actual_source_peak_profile_indices']=profiles['all_actual_source_peak_profile_indices'].astype(np.int64)
        membership = profiles.get("temporal_material_signature_group_membership",
                                  profiles["source_material_conic_group_membership"])
        profile_multiple_transition = np.zeros(len(profiles["source_image_conic_supported_profiles"]),bool)
        if guide.get("primitive")=="free_curve":
            # The complete peak pool remains below. A boundary point set uses
            # one actually sampled RGB-gradient maximum per source profile,
            # never all quantization wiggles across the whole gray band.
            eligible=np.flatnonzero(membership)
            source_rows=profiles["all_actual_source_peak_profile_indices"]
            selected=np.zeros(len(membership),bool)
            for row in np.unique(source_rows[eligible]):
                indices=eligible[source_rows[eligible]==row]
                gradients=profiles["all_actual_source_peak_gradient"][indices]
                best=indices[int(gradients.argmax())]
                selected[best]=True
                distances=np.linalg.norm(profiles["all_actual_source_peak_uv"][indices]-profiles["all_actual_source_peak_uv"][best],axis=1)
                nearby_band=max(2.,float(profiles["all_actual_source_peak_fwhm_px"][best])*.5)
                profile_multiple_transition[row]=bool(((gradients>=gradients.max()*.8)&(distances>nearby_band)).any())
            membership=selected
        points = profiles["all_actual_source_peak_uv"][membership]
        rows = profiles["all_actual_source_peak_profile_indices"][membership]
        available = np.zeros(len(profiles["source_image_conic_supported_profiles"]), bool)
        available[rows] = True
        run_ids = _visible_search_runs(available)
        row_run = np.full(len(available), -1, int)
        row_run[available] = run_ids
        runs = [dict(actual_source_uv=points[row_run[rows] == number],
                     original_source_profile_indices=rows[row_run[rows] == number],
                     contains_interpolated_or_fitted_pixels=False)
                for number in np.unique(run_ids)]
        initialized = guide.get("human_confirmed_instance_identity_initialized") is True
        readings.append(dict(id=guide["id"], source_local_id=guide["id"],
            physical_id=guide.get("physical_id") if initialized else None,
            source_binding=guide["source_binding"], physical_role=guide["physical_role"],
            material_topological_role_candidate=guide["material_topological_role_candidate"],
            physical_identity_verified=initialized, actual_source_uv=points,
            physical_curve_id=guide.get("physical_curve_id") if initialized else None,
            identity_authority=guide.get("identity_authority") if initialized else None,
            identity_scope="human_confirmed_source_physical_instance; RGB boundary quality is separate" if initialized else
                           "unassigned_source_local_observation",
            human_confirmed_identity_source_binding=guide.get("human_confirmed_identity_source_binding") if initialized else None,
            original_source_profile_indices=rows, measured_runs=runs,
            full_curve_missing_or_conflicting_profiles=~available,
            actual_source_gradient=profiles["all_actual_source_peak_gradient"][membership],
            actual_source_peak_fwhm_px=profiles["all_actual_source_peak_fwhm_px"][membership],
            source_quality_flags=dict(
                wide_relative_to_previous_extractor=profiles["all_actual_source_peak_wide_relative_to_old_extractor"][membership],
                weak_relative_to_previous_extractor=profiles["all_actual_source_peak_weak_relative_to_old_extractor"][membership],
                precise_localization_not_validated=np.ones(len(points), bool),
                competing_same_sign_material_transition_peaks=profile_multiple_transition[rows]),
            actual_source_signed_luminance_gradient=profiles["actual_source_signed_luminance_gradient"][membership],
            actual_source_inside_outside_material_delta=profiles["actual_source_inside_outside_material_delta"][membership],
            fitted_or_flow_pixels_used_as_measurements=False, source_masks_used_as_measurements=False,
            source_peak_pool_retained=True, boundary_accuracy_verified=False,
            all_original_source_peak_candidates=profiles["all_actual_source_peak_uv"],
            all_original_source_peak_material_membership=profiles["source_material_conic_group_membership"],
            physical_paint_edge_verified=False,
            source_array_input_normalization=dict(legacy_JSON_vector_indexing_normalized=bool(legacy),
                legacy_field_names=legacy,original_dtype_known=not bool(legacy),
                original_dtype_restored=False,numerical_source_UV_unchanged=True,
                normalization_is_not_byte_exact_historical_restore=bool(legacy)),
            measurement_geometry="actual_RGB_gradient_peak_point_set", accuracy_accepted=False))
    return readings


def bind_confirmed_source_instances(result, confirmed_source_guide_packet, *, role_definitions=None):
    """Initialize known instance identities without replacing any measured UV.

    This consumes the existing source_annotations_to_guides adapter only at its
    exact reference frame. Human points associate already read RGB paths with
    known objects; they never become automatic measurements. RGB topology can
    connect child layers to that object. Nearby unlinked material stays local.
    """
    binding = result["source_binding"]
    same_frame(binding, confirmed_source_guide_packet["source_binding"])
    if result.get("event_or_world_geometry_used") or result.get("fitted_or_mask_pixels_used_as_measurements"):
        raise ValueError("Known identity cannot turn projected or mask UV into source RGB measurements")
    references = confirmed_source_guide_packet.get("guides", [])
    families, reference_rows = {}, {}
    for reference in references:
        same_frame(binding, reference["source_binding"])
        if (reference.get("origin") != "user_confirmed_visual_supervision"
                or reference.get("identity_confirmed") is not True or not reference.get("identity_authority")):
            raise ValueError("Instance initialization requires recorded human source identity authority")
        if reference.get("kind") != "ice_paint":
            continue
        curve_id = str(reference["id"])
        parent = curve_id.split(":", 1)[0]
        entry = families.setdefault(parent, dict(physical_id=parent, identity_confirmed=True,
            identity_source_binding=reference["source_binding"], identity_authority=[], confirmed_curve_ids=[],
            confirmed_source_group_ids=[], human_identity_has_priority_over_model_probability=True,
            source_pixel_boundary_accuracy_is_separate=True))
        if curve_id not in entry["confirmed_curve_ids"]:
            entry["confirmed_curve_ids"].append(curve_id)
        entry["confirmed_source_group_ids"].append(reference["source_group_id"])
        entry["identity_authority"].append(reference["identity_authority"])
        if reference.get("physical_role") != "painted_stroke_center_axis":
            reference_rows.setdefault(parent, []).append(reference)
    readings_by_id = {item["id"]: item for item in result.get("source_shape_readings", [])}
    associations, layer_parents = {}, {}

    def distance_to_reference(points, reference):
        trace = np.asarray(reference["points_uv"], float)
        points = np.asarray(points, float)
        if len(trace) == 1:
            return np.linalg.norm(points-trace[0], axis=1)
        start, vector = trace[:-1], np.diff(trace, axis=0)
        delta = points[:, None]-start[None]
        fraction = (delta*vector[None]).sum(2)/np.maximum((vector*vector).sum(1), 1e-12)[None]
        near = start[None]+fraction.clip(0, 1)[..., None]*vector[None]
        return np.linalg.norm(points[:, None]-near, axis=2).min(1)

    # Anchor an observed outer path, rather than naming every gray path inside
    # a broad circle-search ROI. The recorded search radius is an engineering
    # association neighborhood, never a pixel-accuracy bound.
    for guide in result.get("guides", []):
        reading = readings_by_id.get(guide["id"])
        if reading is None or not len(reading["actual_source_uv"]):
            continue
        if guide.get("material_contour_depth", 0) != 0:
            continue
        points = np.asarray(reading["actual_source_uv"])
        options = []
        for parent, members in reference_rows.items():
            distances = [distance_to_reference(points, member) for member in members]
            distance = np.min(distances, axis=0)
            radius = max(float(member.get("search_radius_px", 18.)) for member in members)
            coverage = float((distance <= radius).mean())
            if coverage >= .5:
                best = members[int(np.argmin([np.median(row) for row in distances]))]
                options.append((coverage, float(np.median(distance)), parent, best, distance, radius))
        options.sort(key=lambda option: (-option[0], option[1]))
        if not options:
            continue
        if len(options) > 1 and options[0][0]-options[1][0] < .15:
            associations[guide["id"]] = dict(status="multiple_confirmed_parent_correspondence_candidates",
                candidate_physical_ids=[option[2] for option in options], boundary_points_unchanged=True)
            continue
        coverage, median, parent, reference, distance, radius = options[0]
        associations[guide["id"]] = dict(status="human_reference_initialized_source_instance_correspondence",
            physical_id=parent, physical_curve_id=reference["id"], source_reference_group_id=reference["source_group_id"],
            expected_physical_role=reference["physical_role"], actual_RGB_source_association_fraction=coverage,
            source_reference_median_distance_px=median, engineering_search_radius_px=radius,
            source_reference_search_radius_is_accuracy_bound=False, boundary_points_unchanged=True)
        topology = guide.get("source_material_topology_id")
        if topology and guide.get("material_contour_depth") == 0:
            layer_parents[topology] = parent
    role_map = dict(role_definitions or {})
    layer_names = {0: "outer", 1: "middle", 2: "inner", 3: "center"}
    for guide in result.get("guides", []):
        association = associations.get(guide["id"], {})
        parent = association.get("physical_id")
        topology = guide.get("source_material_topology_id")
        if parent is None and topology in layer_parents:
            parent = layer_parents[topology]
            subrole = layer_names.get(guide.get("material_contour_depth"))
            curve_id = f"{parent}:{subrole}" if subrole else None
            definition = role_map.get(curve_id, {})
            expected_role = guide["physical_role"]
            if isinstance(definition, dict):
                expected_role = definition.get("physical_role", expected_role)
                if definition.get("semantic_definition") == "named_filled_disc_color_transition_boundary":
                    expected_role = "filled_disc_boundary"
            elif isinstance(definition, str):
                expected_role = definition
            association = dict(status="observed_RGB_material_layer_under_human_named_parent",
                physical_id=parent, physical_curve_id=curve_id,
                expected_physical_role=expected_role,
                child_role_authority="observed_source_material_hierarchy_and_recorded_event_role_definition",
                child_is_new_human_pixel_annotation=False, boundary_points_unchanged=True)
            associations[guide["id"]] = association
        if parent is None:
            continue
        guide.update(physical_id=parent, physical_curve_id=association.get("physical_curve_id"),
            physical_role=association.get("expected_physical_role", guide["physical_role"]), identity_confirmed=True,
            human_confirmed_instance_identity_initialized=True,
            human_confirmed_identity_source_binding=confirmed_source_guide_packet["source_binding"],
            identity_authority=families[parent]["identity_authority"], source_identity_association=association,
            source_boundary_accuracy_verified=False, human_reference_UV_used_as_measured_UV=False)
    original_uv = {key: np.asarray(item["actual_source_uv"]).copy() for key, item in readings_by_id.items()}
    result["source_shape_readings"] = _direct_source_shape_readings(result.get("guides", []))
    for reading in result["source_shape_readings"]:
        if not np.array_equal(original_uv[reading["id"]], reading["actual_source_uv"]):
            raise RuntimeError("Identity initialization changed source RGB measurements")
    result["confirmed_source_instances"] = list(families.values())
    for entry in result["confirmed_source_instances"]:
        measured = [reading for reading in result["source_shape_readings"] if reading["physical_id"] == entry["physical_id"]]
        entry.update(current_source_binding=binding, actual_RGB_point_count=sum(len(reading["actual_source_uv"]) for reading in measured),
            source_reading_ids=[reading["id"] for reading in measured],
            visibility_status="actual_source_readings_present_with_quality_pending" if measured else "source_missing_identity_retained")
    result["source_instance_identity_associations"] = associations
    result["human_reference_UV_used_as_actual_boundary_output"] = False
    result["identity_initialization_is_independent_measurement_validation"] = False
    return result


def continue_confirmed_source_instances(result, previous):
    """Carry initialized instance identities over the recorded source motion."""
    known = previous.get("confirmed_source_instances", [])
    if not known:
        return result
    binding, old = result["source_binding"], previous["source_binding"]
    if any(binding.get(key) != old.get(key) for key in ("source_sha256", "time_base", "shot_id", "installation_id", "native_hw")):
        raise ValueError("Named source correspondence cannot cross a source or shot reset")
    if result.get("previous_source_binding") is not None:
        same_frame(old, result["previous_source_binding"])
    old_guides = {guide["id"]: guide for guide in previous.get("guides", [])}
    before = {reading["id"]: np.asarray(reading["actual_source_uv"]).copy()
              for reading in result.get("source_shape_readings", [])}
    fields = ("physical_id", "physical_curve_id", "physical_role", "identity_confirmed",
              "human_confirmed_instance_identity_initialized", "human_confirmed_identity_source_binding",
              "identity_authority", "source_identity_association", "source_boundary_accuracy_verified",
              "human_reference_UV_used_as_measured_UV")
    for guide in result.get("guides", []):
        reference = old_guides.get(guide["id"], {})
        if reference.get("human_confirmed_instance_identity_initialized") is not True:
            continue
        # The runtime's existing native flow/search identity supplies the
        # correspondence. No reference-frame human UV is read at this point.
        if guide.get("hint_previous_source_binding") is not None:
            same_frame(old, guide["hint_previous_source_binding"])
        motion = guide.get("temporal_flow_metadata")
        if motion is not None:
            pairs = motion.get("source_bindings")
            if not motion.get("source_alignment_verified") or pairs is None or len(pairs) != 2:
                raise ValueError("Named temporal correspondence requires the actual source motion proof")
            same_frame(old, pairs[0]);same_frame(binding, pairs[1])
        elif not (result.get("memory_update") == "verified_identical_original_exposure_source_reread"
                  and result.get("source_RGB_sha256") == previous.get("source_RGB_sha256")
                  and binding.get("exposure_proof_sha256")
                  and binding.get("canonical_exposure_pts") == old.get("canonical_exposure_pts")):
            raise ValueError("Named source identity cannot propagate without source correspondence")
        for field in fields:
            if field in reference:
                guide[field] = reference[field]
    result["source_shape_readings"] = _direct_source_shape_readings(result.get("guides", []))
    for reading in result["source_shape_readings"]:
        if not np.array_equal(before[reading["id"]], reading["actual_source_uv"]):
            raise RuntimeError("Named temporal correspondence changed actual source RGB measurements")
    ledger = []
    for parent in known:
        entry = dict(parent)
        measured = [reading for reading in result["source_shape_readings"] if reading["physical_id"] == entry["physical_id"]]
        entry.update(current_source_binding=binding, actual_RGB_point_count=sum(len(reading["actual_source_uv"]) for reading in measured),
            source_reading_ids=[reading["id"] for reading in measured],
            visibility_status="actual_source_readings_present_with_quality_pending" if measured else "source_missing_identity_retained")
        ledger.append(entry)
    result["confirmed_source_instances"] = ledger
    result["human_reference_UV_used_as_actual_boundary_output"] = False
    result["human_reference_UV_reused_as_current_frame_guide"] = False
    result["identity_initialization_is_independent_measurement_validation"] = False
    return result


def _material_loop_correspondence(candidate, topology_candidates, native_hw, tolerance):
    """A conic seed must follow one whole observed material loop, not two arcs."""
    height, width = native_hw
    profiles = candidate["full_curve_source_RGB_profiles"]
    peaks = profiles["all_actual_source_peak_uv"]
    pool = profiles["source_material_conic_group_membership"]
    rows = profiles["all_actual_source_peak_profile_indices"]
    details = []
    for item in topology_candidates:
        contour = np.rint(item["source_material_contour_uv"]).astype(int)
        mask = np.ones((height, width), np.uint8)
        mask[contour[:, 1], contour[:, 0]] = 0
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        uv = np.rint(peaks).astype(int)
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        near = np.zeros(len(peaks), bool)
        near[valid] = distance[uv[valid, 1], uv[valid, 0]] <= tolerance
        supported = np.unique(rows[pool & near])
        details.append(dict(topology_id=item["topology_id"], material_contour_depth=item["material_contour_depth"],
            actual_source_profile_support_count=len(supported),
            actual_source_profile_support_fraction=len(supported)/max(1, len(profiles["source_image_conic_supported_profiles"])) ))
    return details


def source_paint_fragment_readings(rgb, packet, *, source_ice_search_mask=None,
                                   source_ice_search_binding=None):
    """Retain native paint-search RGB peaks before blur/component vetoes.

    Semantic support only chooses where to inspect original RGB. Actual UVs
    are native nonmaximum source-gradient pixels; broad/weak peaks retain a
    measured normal interval and quality flags. Connected source-pixel paths
    organize visible fragments without fitting or completing any ring. These
    readings supply no physical identity or automatically accepted paint role.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Use unchanged native uint8 RGB")
    source = packet.get("source", packet)
    binding = source_binding(packet["source_binding"], rgb.shape[:2])
    same_frame(binding, source.get("source_binding", binding))
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    if packet.get("RGB_sha256", packet.get("source_RGB_sha256", digest)) != digest:
        raise ValueError("Source fragment ledger and original RGB differ")
    metadata = source.get("metadata", {})
    if metadata.get("world_camera_projection_input_used") or metadata.get("probabilities_conditioned_on_geometry"):
        raise ValueError("Paint fragment search needs no-prior source probabilities")
    names = tuple(source.get("classes", packet.get("classes", ())))
    probability = source.get("semantic_probabilities", packet.get("class_probabilities"))
    if probability is None or "ice_paint" not in names:
        return dict(source_binding=binding, source_RGB_sha256=digest, source_fragment_readings=[],
                    status="missing_source_only_native_paint_search_probabilities", actual_RGB_evidence_erased=False)
    probability = _numpy(probability)
    if probability.ndim == 4 and probability.shape[0] == 1:
        probability = probability[0]
    height, width = rgb.shape[:2]
    if (probability.shape != (len(names), height, width) or not np.isfinite(probability).all()
            or (probability < 0).any() or (probability > 1).any()
            or not np.allclose(probability.sum(0), 1., atol=1e-3)):
        raise ValueError("Fragment search needs normalized native source probabilities")
    paint = probability[names.index("ice_paint")]
    # This matches the existing extractor's local semantic search support;
    # semantic probability is never a source boundary measurement.
    type_support = cv2.dilate(paint, np.ones((3, 3), np.uint8))
    scope = None
    if source_ice_search_mask is not None:
        if source_ice_search_binding is None:
            raise ValueError("Ice search mask needs the same original source binding")
        same_frame(binding, source_ice_search_binding)
        scope = _numpy(source_ice_search_mask).squeeze().astype(bool)
        if scope.shape != (height, width):
            raise ValueError("Ice search mask must retain native dimensions")
    values = rgb.astype(np.float32)/255.
    gx = np.stack([cv2.Sobel(values[:, :, channel], cv2.CV_32F, 1, 0, ksize=3,
                           borderType=cv2.BORDER_REPLICATE)/8 for channel in range(3)], 2)
    gy = np.stack([cv2.Sobel(values[:, :, channel], cv2.CV_32F, 0, 1, ksize=3,
                           borderType=cv2.BORDER_REPLICATE)/8 for channel in range(3)], 2)
    magnitude = np.sqrt((gx*gx+gy*gy).mean(2))
    strongest = (gx*gx+gy*gy).argmax(2)
    yy, xx = np.indices((height, width))
    normal = np.stack((gx[yy, xx, strongest], gy[yy, xx, strongest]), 2)
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-12)
    minus = cv2.remap(magnitude, (xx-normal[:, :, 0]).astype(np.float32),
                     (yy-normal[:, :, 1]).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    plus = cv2.remap(magnitude, (xx+normal[:, :, 0]).astype(np.float32),
                    (yy+normal[:, :, 1]).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Quantized native evidence is retained. Weakness/noise and width are
    # explicit diagnostics, rather than automatic paint acceptance or vetoes.
    peaks = (magnitude >= minus) & (magnitude >= plus) & (magnitude >= .5/255.) & (type_support >= .2)
    peaks[[0, -1], :] = False
    peaks[:, [0, -1]] = False
    count, labels, stats, _ = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), connectivity=8)
    y, x = np.nonzero(peaks)
    uv = np.column_stack((x, y)).astype(np.float32)
    if not len(uv):
        return dict(source_binding=binding, source_RGB_sha256=digest, source_fragment_readings=[],
            actual_source_peak_count=0, source_fragments_count=0,
            status="no_native_RGB_peak_with_source_paint_search_support", actual_RGB_evidence_erased=False,
            physical_marker_names_inferred=False, is_training_label=False, accuracy_accepted=False)
    normals = normal[y, x]
    offsets = np.arange(-16, 17, dtype=np.float32)
    search = uv[:, None]+normals[:, None]*offsets[None, :, None]
    # OpenCV remap internally uses signed-short destination indices. Stream
    # its native work matrices without limiting the number of source peaks.
    def profiles_from(field):
        return np.concatenate([cv2.remap(field, part[..., 0], part[..., 1], cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE) for part in np.array_split(search,
            max(1, int(np.ceil(len(search)/32766.))))])
    prof = profiles_from(magnitude)
    luma = profiles_from(values.mean(2))
    half = prof >= magnitude[y, x, None]*.5
    position = np.arange(len(offsets))[None]
    left = np.where((position < 16) & ~half, position, -1).max(1)
    right = np.where((position > 16) & ~half, position, len(offsets)).min(1)
    widths = (right-left-1).astype(np.float32)
    second = np.diff(luma, n=2, axis=1)
    noise = np.maximum(.5/255., np.median(np.abs(second-np.median(second, axis=1, keepdims=True)), axis=1)*1.4826)
    profile_clipped = ((search[..., 0] < 0) | (search[..., 0] > width-1) |
                       (search[..., 1] < 0) | (search[..., 1] > height-1)).any(1)
    owners = labels[y, x]
    # Every contour vertex indexes an already selected actual source pixel;
    # no simplify/ellipse/spline samples are introduced between fragments.
    contours, _ = cv2.findContours(peaks.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    runs = {}
    for contour in contours:
        points = contour[:, 0]
        owner = int(labels[points[0, 1], points[0, 0]])
        runs.setdefault(owner, []).append(dict(actual_source_uv=points.astype(np.float32),
            contains_interpolated_or_fitted_pixels=False, closed=False,
            source_neighbor_distance_max_px=float(np.linalg.norm(np.diff(points, axis=0), axis=1).max(initial=0))))
    readings = []
    for component in range(1, count):
        selected = np.flatnonzero(owners == component)
        points = uv[selected]
        material_delta = luma[selected, -4:].mean(1)-luma[selected, :4].mean(1)
        readings.append(dict(id=f"source_paint_fragment_{component}", source_local_id=f"source_paint_fragment_{component}",
            source_binding=binding, physical_id=None, physical_identity_verified=False,
            physical_role="source_paint_role_unresolved", primitive="source_RGB_edge_fragment",
            actual_source_uv=points, actual_source_gradient=magnitude[y[selected], x[selected]],
            actual_source_normals=normals[selected], actual_source_peak_fwhm_px=widths[selected],
            measured_normal_interval_px=np.stack((-widths[selected]*.5, widths[selected]*.5), 1),
            actual_source_inside_outside_material_delta=material_delta,
            measured_runs=runs.get(component, []), source_component_bbox_xyxy=np.r_[
                stats[component, :2], stats[component, :2]+stats[component, 2:4]],
            native_source_pixel_count=len(selected), source_paint_probability=type_support[y[selected], x[selected]],
            source_quality_flags=dict(wide_relative_to_previous_extractor=widths[selected] > 4,
                weak_relative_to_previous_extractor=magnitude[y[selected], x[selected]] < .02,
                small_relative_to_previous_component_gate=np.full(len(selected), len(selected) < 12),
                source_profile_image_clipped=profile_clipped[selected],
                source_profile_width_interval_clipped=(left[selected] < 0) | (right[selected] >= len(offsets)),
                low_gradient_to_local_source_noise=magnitude[y[selected], x[selected]] < noise[selected]*3,
                outside_unverified_source_ice_search_mask=(~scope[y[selected], x[selected]] if scope is not None else
                                                          np.ones(len(selected), bool)),
                precise_localization_not_validated=np.ones(len(selected), bool)),
            source_type_is_search_support_only=True, source_ice_domain_verified=False,
            physical_paint_edge_verified=False, fitted_or_flow_pixels_used_as_measurements=False,
            source_masks_used_as_measurements=False, width_or_small_component_used_as_hard_veto=False,
            topology_status="contiguous_actual_native_source_pixel_chain_without_circle_completion",
            is_training_label=False, accuracy_accepted=False))
    return dict(source_binding=binding, source_RGB_sha256=digest, source_fragment_readings=readings,
        actual_source_peak_count=len(uv), source_fragments_count=len(readings),
        source_type_probability_used_as_search_only=True, primitive_probability_used_as_physical_identity=False,
        original_quality_rejected_evidence_retained=True, actual_RGB_evidence_erased=False,
        source_ice_mask_used_as_hard_veto=False, image_boundary_used_as_measurements=False,
        physical_marker_names_inferred=False, is_training_label=False, accuracy_accepted=False)


def _visible_material_fragment_guides(rgb, record, binding, paint_probability, *, prefix):
    """Reread actual open mask paths when complete nested topology is missing."""
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = record["source_navigation_bbox_xyxy"]
    diameter = record["source_illumination_search_kernel_diameter_px"]
    gray = rgb[y0:y1, x0:x1].astype(np.float32).mean(2)/255.
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)))
    mask = ((background-gray) > record["source_contrast_otsu_threshold"]).astype(np.uint8)
    guides = []
    for item in record["all_source_material_contour_candidates"]:
        if item["source_material_search_area_px"] <= diameter:
            continue
        contour = np.asarray(item["original_RGB_search_contour_uv"], np.float32)
        actual = ((contour[:,0] > max(1,x0+1)) & (contour[:,0] < min(width-2,x1-2)) &
                  (contour[:,1] > max(1,y0+1)) & (contour[:,1] < min(height-2,y1-2)))
        if actual.sum() < 24:
            continue
        if not actual.all():
            shift = int(np.flatnonzero(~actual)[0])
            contour = np.roll(contour, -shift, axis=0);actual = np.roll(actual, -shift)
        run_ids = _visible_search_runs(actual)
        for number in np.unique(run_ids):
            for run in _smooth_source_contour_runs(contour[actual][run_ids==number], 6.):
                arc = np.r_[0,np.cumsum(np.linalg.norm(np.diff(run,axis=0),axis=1))]
                positions = np.arange(0,arc[-1],6.)
                if len(positions) < 6:
                    continue
                seeds = np.stack([np.interp(positions,arc,run[:,channel]) for channel in (0,1)],1)
                before = np.stack([np.interp(np.maximum(0,positions-12),arc,run[:,channel]) for channel in (0,1)],1)
                after = np.stack([np.interp(np.minimum(arc[-1],positions+12),arc,run[:,channel]) for channel in (0,1)],1)
                tangent = after-before
                tangent /= np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-9)
                normals = np.stack((tangent[:,1],-tangent[:,0]),1)
                positive = seeds+normals*3-[x0,y0];negative = seeds-normals*3-[x0,y0]
                p = cv2.remap(mask,positive[:,0,None].astype(np.float32),positive[:,1,None].astype(np.float32),cv2.INTER_NEAREST).ravel()
                n = cv2.remap(mask,negative[:,0,None].astype(np.float32),negative[:,1,None].astype(np.float32),cv2.INTER_NEAREST).ravel()
                normals[p>n] *= -1
                typed = cv2.remap(paint_probability,seeds[:,0,None].astype(np.float32),
                                  seeds[:,1,None].astype(np.float32),cv2.INTER_LINEAR).ravel()
                if (typed>=.2).mean() < .5:
                    continue
                profiles = _reread_conic_source_profiles(rgb,seeds,normals,None,material_orientation=1)
                supported = profiles["source_image_conic_supported_profiles"]
                if supported.sum() < 6:
                    continue
                guides.append(dict(id=f"{prefix}_{len(guides)}",kind="ice_paint",primitive="free_curve",
                    source_binding=binding,physical_id=None,physical_role="source_paint_role_unresolved",identity_confirmed=False,
                    origin="source_annotation_search_hint",points_uv=seeds,normals=normals,
                    segment_ids=np.zeros(len(seeds),np.int32),closed=False,
                    full_curve_source_RGB_profiles=profiles,
                    source_native_material_component_index=item["candidate_contour_index"],
                    source_native_material_hierarchy=item["hierarchy_next_previous_child_parent"],
                    source_material_topology_id=f"{prefix}_candidate_{item['candidate_contour_index']}",
                    material_topological_role_candidate="visible_gray_material_border_without_verified_complete_hierarchy",
                    source_paint_probability_is_only_search_support=True, source_primitive_head_used_only_for_neighborhood_search=True,
                    complete_closed_hierarchy_missing=True, missing_layer_or_disk_not_completed=True,
                    source_native_search_boundary_removed_as_artificial_closure=True,
                    fitted_points_used_as_measurements=False,localization_accuracy_accepted=False))
    return guides


def source_primitive_material_readings(rgb, packet, *, primitive_probabilities,
        primitive_source_binding, source_ice_search_mask, source_ice_search_binding):
    """Primitive-head regions navigate RGB material searches, never name rings.

    A locally straight lower arc remains in the same RGB neighborhood as its
    curved upper arc. Two observed material layers suffice; a third disk or a
    complete physical circle is not invented. Every reported point is reread
    from current source RGB normal profiles, separately from ellipse search UV.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Use unchanged native uint8 RGB")
    binding = source_binding(packet["source_binding"], rgb.shape[:2])
    same_frame(binding, primitive_source_binding)
    same_frame(binding, source_ice_search_binding)
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    if packet.get("RGB_sha256", packet.get("source_RGB_sha256", digest)) != digest:
        raise ValueError("Source material RGB differs from source-head binding")
    source = packet.get("source", packet)
    same_frame(binding, source.get("source_binding", binding))
    metadata = source.get("metadata", {})
    if metadata.get("probabilities_conditioned_on_geometry") or metadata.get("world_camera_projection_input_used"):
        raise ValueError("Source-head search cannot use world-conditioned probabilities")
    names = tuple(source.get("classes", packet.get("classes", ())))
    semantics = _numpy(source.get("semantic_probabilities", packet.get("class_probabilities")))
    primitive = _numpy(primitive_probabilities)
    if semantics.ndim == 4:
        semantics = semantics[0]
    if primitive.ndim == 4:
        primitive = primitive[0]
    height, width = rgb.shape[:2]
    scope = _numpy(source_ice_search_mask).squeeze().astype(bool)
    for values, expected in ((semantics, (len(names), height, width)), (primitive, (3, height, width))):
        if (values.shape != expected or not np.isfinite(values).all() or (values < 0).any()
                or (values > 1).any() or not np.allclose(values.sum(0), 1., atol=1e-3)):
            raise ValueError("Source-head searches need normalized native probabilities")
    if scope.shape != (height, width) or "ice_paint" not in names:
        raise ValueError("Source primitive search needs native ice navigation and paint head")
    # Local primitive classification provides SEARCH regions, not object
    # identity. Lower line-like pixels are included by spatial/RGB adjacency.
    search_mask = (primitive.argmax(0) == 1) & (semantics[names.index("ice_paint")] >= .2) & scope
    count, _, stats, _ = cv2.connectedComponentsWithStats(search_mask.astype(np.uint8), 8)
    fragments = source_paint_fragment_readings(rgb, packet,
        source_ice_search_mask=scope, source_ice_search_binding=binding)
    pixels = (np.concatenate([item["actual_source_uv"] for item in fragments["source_fragment_readings"]], 0)
              if fragments["source_fragment_readings"] else np.empty((0, 2), np.float32))
    nodes, groups, search_regions = [], [], []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 12 or min(w, h) < 3:
            continue
        margin = max(6., float(min(w, h))*.5)
        lo = np.maximum([0, 0], np.array([x, y])-margin)
        hi = np.minimum([width, height], np.array([x+w, y+h])+margin)
        selected = pixels[((pixels >= lo) & (pixels < hi)).all(1)]
        if len(selected) < 24:
            continue
        nodes.append(dict(index=index, uv=selected, lo=selected.min(0), hi=selected.max(0)))
        groups.append([len(nodes)-1])
        search_regions.append(dict(source_primitive_search_region_index=index,
            actual_source_head_bbox_xyxy=[int(x), int(y), int(x+w), int(y+h)],
            source_RGB_search_bbox_xyxy=np.r_[lo, hi],
            source_illumination_search_kernel_diameter_px=int(min(w,h)+12) | 1,
            lower_line_like_pixels_excluded=False, head_mask_is_pixel_measurement=False,
            physical_marker_name_inferred=False))
    material, evidence = _source_material_topology(rgb, nodes, groups, 6., minimum_nested_layers=2,
                                                  source_navigation_regions=search_regions)
    guides, rejected = [], []
    for candidate in material:
        center, radii, rotation, quadratic = candidate["model"]
        phases = np.linspace(0, 2*np.pi, max(24, int(np.ceil(2*np.pi*np.sqrt(np.mean(radii*radii))/6.))), endpoint=False)
        search = center+np.stack((np.cos(phases), np.sin(phases)), 1)*radii @ rotation.T
        normals = (search-center) @ quadratic
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        visible = (search[:,0]>=0)&(search[:,0]<width-1)&(search[:,1]>=0)&(search[:,1]<height-1)
        profiles = _reread_conic_source_profiles(rgb, search, normals, candidate["model"],
            material_orientation=candidate["material_orientation"])
        supported = profiles["source_image_conic_supported_profiles"] & visible
        if supported.sum() < 6:
            rejected.append(dict(topology_id=candidate["topology_id"],
                reason="insufficient_actual_signed_source_RGB_profiles", actual_source_supported_profiles=int(supported.sum())))
            continue
        guides.append(dict(id=f"source_head_material_curve_{len(guides)}", kind="ice_paint", primitive="conic",
            source_binding=binding, physical_id=None, physical_role="source_paint_role_unresolved", identity_confirmed=False,
            origin="source_annotation_search_hint", points_uv=search, normals=normals,
            segment_ids=np.zeros(len(search), np.int32), closed=False,
            full_curve_source_RGB_profiles=profiles, material_contour_depth=candidate["material_contour_depth"],
            source_material_topology_id=candidate["topology_id"],
            material_topological_role_candidate=("source_light_hole_border" if candidate["material_contour_depth"]%2 else
                                                "source_dark_material_outer_border"),
            image_conic_is_world_circle=False, fitted_points_used_as_measurements=False,
            source_primitive_head_used_only_for_neighborhood_search=True,
            minimum_nested_material_layers_required=2,
            actual_source_supported_profile_count=int(supported.sum()), full_curve_missing_profile_count=int((~supported).sum()),
            localization_accuracy_accepted=False))
    for index, record in enumerate(evidence):
        if not record["material_hierarchies"]:
            guides.extend(_visible_material_fragment_guides(rgb,record,binding,
                semantics[names.index("ice_paint")].astype(np.float32),prefix=f"source_head_open_material_{index}"))
    return dict(source_binding=binding, source_RGB_sha256=digest, guides=guides,
        source_shape_readings=_direct_source_shape_readings(guides), source_primitive_search_regions=search_regions,
        source_material_topology_evidence=evidence, rejected_source_material_candidates=rejected,
        source_component_indices_are_search_partitions=True, source_ice_domain_verified=False,
        source_fragment_readings=fragments["source_fragment_readings"],
        raw_RGB_fragments_not_erased_by_material_search_failure=True,
        primitive_prediction_is_physical_identity=False, physical_marker_names_inferred=False,
        fitted_or_mask_pixels_used_as_measurements=False, is_training_label=False, accuracy_accepted=False)


def source_circle_search_guides(rgb, packet, *, binding=None, hypothesis_count=512,
                                maximum_source_residual_px=2., sample_spacing_px=6.,
                                role_definitions=None, random_seed=17):
    """Organize disconnected actual RGB arcs into explicitly local hypotheses.

    Accepts a source-reader packet or its compact source ledger. Local component
    neighborhoods generate hypotheses with OpenCV's existing fitEllipseDirect.
    Distributed upper AND lower source pixels, multiple components, measured
    curvature and source-normal agreement are required. Straight/reflective
    evidence is retained in the unresolved ledger. Names are source-local;
    external role definitions must be explicitly supplied, never inferred from
    the primitive or from an event/world projection.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Use unchanged native uint8 RGB")
    source = packet.get("source", packet)
    actual_binding = source_binding(binding or packet["source_binding"], rgb.shape[:2])
    same_frame(actual_binding, source.get("source_binding", packet["source_binding"]))
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    if packet.get("RGB_sha256", digest) != digest:
        raise ValueError("Source component ledger and actual original RGB differ")
    metadata = source.get("metadata", {})
    if metadata.get("world_camera_projection_input_used") or metadata.get("probabilities_conditioned_on_geometry"):
        raise ValueError("Organize actual no-prior RGB evidence, not projected observations")
    if hypothesis_count < 1 or maximum_source_residual_px <= 0 or sample_spacing_px <= 0:
        raise ValueError("Use positive source-image hypothesis/search parameters")
    height, width = rgb.shape[:2]
    components = source.get("components", packet.get("source_components", ()))
    nodes = []
    for index, component in enumerate(components):
        if component.get("typed_class_candidate", component.get("kind")) != "ice_paint":
            continue
        pixels = (_numpy(component["source_support_uv"]) if "source_support_uv" in component else
                  _runs_uv(component["source_pixel_runs_y_x_length"]))
        if (pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all() or
                not np.allclose(pixels, np.rint(pixels)) or (pixels < 0).any() or
                (pixels[:, 0] >= width).any() or (pixels[:, 1] >= height).any()):
            raise ValueError("Only actual native source pixel indices may organize an image curve")
        if len(pixels) >= 6:
            nodes.append(dict(index=index, uv=pixels.astype(np.float32), lo=pixels.min(0), hi=pixels.max(0)))
    normalized = rgb.astype(np.float32)/255
    gx = np.stack([cv2.Sobel(normalized[:, :, channel], cv2.CV_32F, 1, 0, ksize=3)/8 for channel in range(3)], 2)
    gy = np.stack([cv2.Sobel(normalized[:, :, channel], cv2.CV_32F, 0, 1, ksize=3)/8 for channel in range(3)], 2)
    strongest = (gx*gx+gy*gy).argmax(2)
    yy, xx = np.indices((height, width))
    normal_field = np.stack((gx[yy, xx, strongest], gy[yy, xx, strongest]), 2)
    normal_field /= np.maximum(np.linalg.norm(normal_field, axis=2, keepdims=True), 1e-9)
    # Local graph membership is a search hypothesis, not a marker grouping.
    neighbors = [[] for _ in nodes]
    reach = np.array([width*.08, height*.06])
    for a, first in enumerate(nodes):
        for b in range(a+1, len(nodes)):
            second = nodes[b]
            gap = np.maximum(np.maximum(first["lo"]-second["hi"], second["lo"]-first["hi"]), 0)
            if (gap <= reach).all():
                neighbors[a].append(b); neighbors[b].append(a)
    groups, seen = [], set()
    for start in range(len(nodes)):
        if start in seen:
            continue
        stack, group = [start], []
        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index); group.append(index); stack.extend(neighbors[index])
        groups.append(group)
    material_candidates, material_evidence = _source_material_topology(rgb, nodes, groups, sample_spacing_px)
    rng = np.random.default_rng(random_seed)
    hypotheses, competing, rejections, all_support = [], [], [], []
    for group in groups:
        pixels = np.concatenate([nodes[index]["uv"] for index in group])
        owners = np.concatenate([np.full(len(nodes[index]["uv"]), nodes[index]["index"]) for index in group])
        all_support.append(pixels)
        if len(group) < 3 or np.ptp(pixels[:, 1]) < 8:
            rejections.append(dict(component_indices=owners.tolist(), reason="source_components_do_not_supply_multiple_sides_or_curvature"))
            continue
        ix = pixels.astype(int)
        observed_normals = normal_field[ix[:, 1], ix[:, 0]]
        group_models = []
        for _ in range(hypothesis_count):
            chosen = rng.choice(len(pixels), 6, replace=False)
            if len(set(owners[chosen])) < 3:
                continue
            try:
                model = _ellipse_model(cv2.fitEllipseDirect(pixels[chosen]))
            except cv2.error:
                continue
            if model is None:
                continue
            center, radii, rotation, Q = model
            if radii.min() < 4 or radii.max() > max(height, width)*2 or radii.min()/radii.max() < .01:
                continue
            residual = _ellipse_residual(pixels, model)
            radial_normals = (pixels-center) @ Q
            radial_normals /= np.maximum(np.linalg.norm(radial_normals, axis=1, keepdims=True), 1e-9)
            align = np.abs((radial_normals*observed_normals).sum(1))
            inlier = (residual <= maximum_source_residual_px) & (align >= .7)
            supported = pixels[inlier]
            if len(supported) < 24 or len(set(owners[inlier])) < 3:
                continue
            local = (supported-center) @ rotation / radii
            angle = np.arctan2(local[:, 1], local[:, 0])
            occupied = np.unique(((angle+np.pi)/(2*np.pi)*24).astype(int).clip(0, 23))
            upper, lower = int((supported[:, 1] < center[1]).sum()), int((supported[:, 1] > center[1]).sum())
            if len(occupied) < 8 or min(upper, lower) < 12:
                continue
            # A shallow straight/reflection band can masquerade as a huge
            # ellipse. Source normals must actually turn on both source sides.
            turn = float(np.ptp(np.arctan2(radial_normals[inlier, 1], radial_normals[inlier, 0])))
            if turn < .4:
                continue
            score = float(len(supported)*(len(occupied)/24)/(1+np.median(residual[inlier])))
            group_models.append((score, model, inlier, residual, align, occupied, upper, lower))
        group_models.sort(key=lambda row: row[0], reverse=True)
        selected = []
        for row in group_models:
            score, model, inlier, residual, align, occupied, upper, lower = row
            if any((inlier & previous[2]).sum()/max(1, min(inlier.sum(), previous[2].sum())) > .8 for previous in selected):
                continue
            selected.append(row)
            center, radii, rotation, Q = model
            perimeter = 2*np.pi*np.sqrt(np.mean(radii*radii))
            phases = np.linspace(0, 2*np.pi, max(16, int(np.ceil(perimeter/sample_spacing_px))), endpoint=False)
            search = center+np.stack((np.cos(phases), np.sin(phases)), 1)*radii @ rotation.T
            normals = (search-center) @ Q
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
            inside = (search[:, 0] >= 0) & (search[:, 0] < width) & (search[:, 1] >= 0) & (search[:, 1] < height)
            source_profiles = _reread_conic_source_profiles(rgb, search, normals, model)
            coherent = source_profiles["source_image_conic_supported_profiles"] & inside
            visible_count = int(inside.sum())
            phase_bins = (np.arange(len(search))/len(search)*24).astype(int)
            reread_bins = np.unique(phase_bins[coherent])
            quarters = [int((coherent & (phase_bins//6 == quarter)).sum()) for quarter in range(4)]
            major_coordinate = np.cos(phases) if radii[0] >= radii[1] else np.sin(phases)
            ends_supported = [int((coherent & (major_coordinate*sign > .97)).sum()) for sign in (-1, 1)]
            if visible_count and (coherent.sum()/visible_count < .45 or len(reread_bins) < 18 or
                                  min(quarters) < 3 or min(ends_supported) < 2):
                rejections.append(dict(component_indices=sorted(set(owners[inlier])),
                    reason="source_arc_fit_not_supported_by_full_curve_actual_RGB_profiles",
                    candidate_center_uv=center.tolist(), reread_phase_bin_count=len(reread_bins),
                    actual_RGB_conic_supported_samples=int(coherent.sum()), visible_search_samples=visible_count,
                    major_axis_endpoint_source_support=ends_supported,
                    width_or_weakness_was_not_a_veto=True))
                continue
            identity = f"source_arc_conic_candidate_{len(competing)}"
            role = (role_definitions or {}).get(identity, "source_paint_role_unresolved")
            candidate = dict(id=identity, kind="ice_paint", primitive="conic", physical_role=role,
                origin="source_annotation_search_hint", identity_confirmed=False, physical_id=None,
                source_binding=actual_binding, points_uv=search[inside], normals=normals[inside],
                segment_ids=_visible_search_runs(inside), closed=bool(inside.all()),
                search_radius_px=max(8., maximum_source_residual_px*6),
                actual_source_support_uv=pixels[inlier], original_component_indices=owners[inlier],
                upper_actual_source_support_count=upper, lower_actual_source_support_count=lower,
                occupied_source_phase_bins=occupied, source_normal_agreement=align[inlier],
                source_sampson_residual_px=residual[inlier], source_score=score,
                full_curve_source_RGB_profiles=source_profiles,
                source_revisited_occupied_phase_bins=reread_bins,
                image_conic_center_uv=center, image_conic_radii_px=radii, image_conic_rotation=rotation,
                image_conic_is_world_circle=False, fitted_points_used_as_measurements=False,
                typed_source_role_unresolved=role == "source_paint_role_unresolved",
                wide_weak_original_RGB_peaks_must_be_reread=True, localization_accuracy_accepted=False)
            candidate["source_material_loop_correspondence"] = _material_loop_correspondence(
                candidate, material_candidates, (height, width), sample_spacing_px)
            candidate["status"] = "competing_arc_pairing_without_whole_source_material_loop_verification"
            candidate["rejection_reason"] = "signed_transition_alone_can_join_different_borders_of_one_annulus"
            competing.append(candidate)
        if not selected:
            rejections.append(dict(component_indices=sorted(set(owners)), reason="no_distributed_upper_lower_RGB_normal_supported_conic"))
    for item in material_candidates:
        center, radii, rotation, Q = item["model"]
        phases = np.linspace(0, 2*np.pi, max(24, int(np.ceil(2*np.pi*np.sqrt(np.mean(radii*radii))/sample_spacing_px))), endpoint=False)
        search = center+np.stack((np.cos(phases), np.sin(phases)), 1)*radii @ rotation.T
        normals = (search-center) @ Q
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        visible = (search[:, 0] >= 0) & (search[:, 0] < width-1) & (search[:, 1] >= 0) & (search[:, 1] < height-1)
        profiles = _reread_conic_source_profiles(rgb, search, normals, item["model"],
                                                material_orientation=item["material_orientation"])
        observed = profiles["source_image_conic_supported_profiles"] & visible
        bins = (np.arange(len(search))/len(search)*24).astype(int)
        occupied = np.unique(bins[observed])
        if observed.sum()/max(1, visible.sum()) < .7 or len(occupied) < 20:
            rejections.append(dict(topology_id=item["topology_id"], material_contour_depth=item["material_contour_depth"],
                reason="closed_material_search_contour_missing_distributed_actual_signed_RGB_peaks",
                actual_source_supported_profile_count=int(observed.sum()), visible_search_count=int(visible.sum())))
            continue
        identity = f"source_local_circle_{len(hypotheses)}"
        role = (role_definitions or {}).get(identity, "source_paint_role_unresolved")
        source_pixels = np.concatenate([node["uv"] for node in nodes if node["index"] in item["source_component_indices"]])
        source_owners = np.concatenate([np.full(len(node["uv"]), node["index"]) for node in nodes if node["index"] in item["source_component_indices"]])
        residue = _ellipse_residual(source_pixels, item["model"])
        near = residue <= max(3., maximum_source_residual_px)
        hypotheses.append(dict(id=identity, kind="ice_paint", primitive="conic", physical_role=role,
            origin="source_annotation_search_hint", identity_confirmed=False, physical_id=None,
            source_binding=actual_binding, points_uv=search[visible], normals=normals[visible],
            segment_ids=_visible_search_runs(visible), closed=bool(visible.all()), search_radius_px=12.,
            actual_source_support_uv=source_pixels[near], original_component_indices=source_owners[near],
            source_material_topology_id=item["topology_id"], material_contour_depth=item["material_contour_depth"],
            material_topological_role_candidate="source_light_hole_border" if item["material_contour_depth"] == 1 else "source_dark_material_outer_border",
            material_topology_is_physical_paint_identity=False,
            status="closed_source_material_topology_and_signed_RGB_peak_search_candidate",
            full_curve_source_RGB_profiles=profiles, source_revisited_occupied_phase_bins=occupied,
            image_conic_center_uv=center, image_conic_radii_px=radii, image_conic_rotation=rotation,
            image_conic_is_world_circle=False, fitted_points_used_as_measurements=False,
            typed_source_role_unresolved=role == "source_paint_role_unresolved",
            wide_weak_original_RGB_peaks_must_be_reread=True, localization_accuracy_accepted=False))
    supported_components = {int(index) for row in hypotheses for index in row["original_component_indices"]}
    fragments = source_paint_fragment_readings(rgb, packet)
    return dict(source_binding=actual_binding, source_RGB_sha256=digest, guides=hypotheses,
                source_shape_readings=_direct_source_shape_readings(hypotheses),
                source_fragment_readings=fragments["source_fragment_readings"],
                source_fragment_status=fragments.get("status", "actual_RGB_fragments_retained_before_shape_gate"),
                competing_arc_hypotheses=competing, source_material_topology_evidence=material_evidence,
                rejected_groups=rejections,
                unresolved_source_components=[node["index"] for node in nodes],
                unorganized_source_components=[node["index"] for node in nodes if node["index"] not in supported_components],
                original_source_components_not_erased=True,
                source_quality_rejections=packet.get("source_quality_rejections", {}),
                actual_source_support_count=sum(len(node["uv"]) for node in nodes),
                source_image_hypotheses_only=True, physical_marker_names_inferred=False,
                event_or_world_geometry_used=False, camera_optimized=False, is_training_label=False,
                source_ice_domain_verified=False,
                source_domain_status=("unverified_source_ice_search_region_available" if
                    packet.get("source_region_constraints", {}).get("source_role_status", {}).get("ice_paint") ==
                    "source_region_search_scope_available" else "source_material_domain_unresolved_no_whole_image_ice_assumption"),
                accuracy_accepted=False)


def _sample_native_field(field, uv, *, nearest=False):
    if hasattr(field, "detach"):
        import torch
        from torch.nn import functional as F
        height, width = field.shape[-2:]
        points = torch.as_tensor(uv, device=field.device, dtype=torch.float32)
        grid = points*points.new_tensor([2/(width-1), 2/(height-1)])-1
        return F.grid_sample(field.float(), grid[None, :, None], mode="nearest" if nearest else "bilinear",
                             align_corners=True)[0, :, :, 0].T.detach().cpu().numpy()
    value = np.asarray(field)
    if value.ndim == 4:
        value = value[0].transpose(1, 2, 0)
    if value.ndim == 2:
        value = value[:, :, None]
    result = cv2.remap(value.astype(np.float32), uv[:, 0].astype(np.float32)[:, None],
                       uv[:, 1].astype(np.float32)[:, None], cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)
    return result.reshape(len(uv), -1)


class SourceMaterialShapeReader:
    """Same-shot source shape search memory with fresh RGB evidence every frame.

    SourceReader's existing RAFT transports search locations, never output UV.
    Current full-normal RGB signatures reverify retained material transitions.
    Missing motion/cuts reset correspondence; any unsupported run stays absent.
    Neither an ice-domain claim nor a physical circle name is inferred.
    """
    def __init__(self, **search_parameters):
        self.search_parameters = search_parameters
        self.previous = None

    def reset(self):
        self.previous = None

    def read_material_transition_profiles(self, rgb, packet, **search):
        """Read explicitly declared source-material normals without conic gating.

        This finite-source path shares the reader's original RGB/clock binding,
        but does not bootstrap a new object or change its temporal memory. The
        returned optical measurements still need source identity/role authority.
        """
        if not packet.get("RGB_sha256"):
            raise ValueError("Material profile reading needs a bound source RGB ledger")
        return read_source_material_transition_profiles(rgb,
            binding=packet["source_binding"], expected_RGB_sha256=packet["RGB_sha256"], **search)

    def read_connected_material_transition_profiles(self, rgb, packet, **search):
        """Read one connected actual RGB branch within a finite search guide."""
        if not packet.get("RGB_sha256"):
            raise ValueError("Connected material reading needs a bound source RGB ledger")
        return read_connected_source_material_profiles(rgb,
            binding=packet["source_binding"], expected_RGB_sha256=packet["RGB_sha256"], **search)

    def read_dark_stroke_center_candidates(self, rgb, packet, **search):
        """Keep every measured RGB band; physical centerline role stays open."""
        if not packet.get("RGB_sha256"):
            raise ValueError("Dark stroke reading needs a bound source RGB ledger")
        return read_source_dark_stroke_profiles(rgb, binding=packet["source_binding"],
            expected_RGB_sha256=packet["RGB_sha256"], **search)

    def read_chromatic_stroke_center_candidates(self, rgb, packet, **search):
        """Colored paint centers retain width separately from localization."""
        if not packet.get("RGB_sha256"):
            raise ValueError("Colored stroke reading needs a bound source RGB ledger")
        return read_source_chromatic_stroke_centers(rgb, binding=packet["source_binding"],
            expected_RGB_sha256=packet["RGB_sha256"], **search)

    def read(self, rgb, packet, *, flow_pair=None, reset=False,
             primitive_probabilities=None, primitive_source_binding=None,
             source_ice_search_mask=None, source_ice_search_binding=None,
             confirmed_source_guide_packet=None, role_definitions=None):
        binding = source_binding(packet["source_binding"], np.asarray(rgb).shape[:2])
        digest = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        if packet.get("RGB_sha256", digest) != digest:
            raise ValueError("Current source shape RGB differs from its reader ledger")
        previous = self.previous
        cut = bool(previous and any(previous["source_binding"].get(key) != binding.get(key)
                   for key in ("source_sha256", "time_base", "shot_id", "installation_id", "native_hw")))
        if previous and not cut and binding["source_pts"] < previous["source_binding"]["source_pts"]:
            raise ValueError("Source shape memory cannot run backwards without reset")
        if reset or cut or previous is None:
            if primitive_probabilities is not None:
                result = source_primitive_material_readings(rgb, packet,
                    primitive_probabilities=primitive_probabilities, primitive_source_binding=primitive_source_binding,
                    source_ice_search_mask=source_ice_search_mask, source_ice_search_binding=source_ice_search_binding)
            else:
                result = source_circle_search_guides(rgb, packet, **self.search_parameters)
            if confirmed_source_guide_packet is not None:
                result = bind_confirmed_source_instances(result, confirmed_source_guide_packet,
                                                         role_definitions=role_definitions)
            result["memory_update"] = "source_shot_material_bootstrap"
            result["source_local_identity_is_physical_marker_identity"] = False
            self.previous = result
            return result
        same_pts = binding["source_pts"] == previous["source_binding"]["source_pts"]
        if confirmed_source_guide_packet is not None:
            raise ValueError("Confirmed reference guide points initialize a source instance once, not every frame")
        if same_pts:
            if digest != previous["source_RGB_sha256"]:
                raise ValueError("Same original PTS has changed RGB in source shape memory")
            return {**previous, "memory_update": "same_original_source_frame_reread"}
        current_fragments = source_paint_fragment_readings(rgb, packet,
            source_ice_search_mask=source_ice_search_mask, source_ice_search_binding=source_ice_search_binding)
        if flow_pair is None:
            confirmed_repeat = (digest == previous["source_RGB_sha256"] and
                binding.get("exposure_proof_sha256") and
                binding.get("exposure_proof_sha256") == previous["source_binding"].get("exposure_proof_sha256") and
                binding.get("canonical_exposure_pts") == previous["source_binding"].get("canonical_exposure_pts"))
            if confirmed_repeat:
                guides = [{**{key:value for key,value in row.items() if key != "temporal_flow_metadata"},
                           "source_binding": binding, "hint_previous_source_binding": previous["source_binding"]}
                          for row in previous["guides"]]
                result = {**previous, "source_binding": binding, "guides": guides,
                          "source_shape_readings": _direct_source_shape_readings(guides),
                          "source_fragment_readings": current_fragments["source_fragment_readings"],
                          "memory_update": "verified_identical_original_exposure_source_reread"}
                result = continue_confirmed_source_instances(result, previous)
                self.previous = result
                return result
            result = dict(source_binding=binding, source_RGB_sha256=digest, guides=[], source_shape_readings=[],
                        source_fragment_readings=current_fragments["source_fragment_readings"],
                        memory_update="missing_source_bound_temporal_correspondence", accuracy_accepted=False,
                        missing_source_local_ids=[row["id"] for row in previous["guides"]],
                        raw_source_packet_preserved=True, interpolated_boundary_uv_generated=False)
            result = continue_confirmed_source_instances(result, previous)
            # A missing motion correspondence invalidates the current search
            # paths, not the recorded human identity of the source objects.
            # Retain that catalog through consecutive missing frames without
            # reusing reference UV or filling any missing boundary points.
            self.previous = result
            return result
        metadata = flow_pair.metadata
        declared = metadata.get("source_bindings")
        if not metadata.get("source_alignment_verified") or declared is None or len(declared) != 2:
            raise ValueError("Temporal source shape search requires both native flow bindings")
        same_frame(previous["source_binding"], declared[0])
        same_frame(binding, declared[1])
        if metadata.get("native_hw", binding["native_hw"]) != binding["native_hw"]:
            raise ValueError("Temporal shape flow resolution differs from the native source")
        if tuple(flow_pair.forward.shape[-2:]) != tuple(binding["native_hw"]):
            raise ValueError("Temporal shape flow is not a native source displacement field")
        guides, ledger = [], []
        height, width = np.asarray(rgb).shape[:2]
        for old in previous["guides"]:
            seed = np.asarray(old["points_uv"], np.float32)
            movement = _sample_native_field(flow_pair.forward, seed)
            supported = _sample_native_field(flow_pair.valid_forward, seed, nearest=True)[:, 0] > .5
            predicted = seed+movement
            supported &= np.isfinite(predicted).all(1)
            supported &= (predicted[:, 0] >= 0) & (predicted[:, 0] < width-1) & (predicted[:, 1] >= 0) & (predicted[:, 1] < height-1)
            old_profiles = old["full_curve_source_RGB_profiles"]
            if supported.sum() < 6:
                ledger.append(dict(id=old["id"], status="missing_distributed_source_flow_support", actual_boundary_filled=False))
                continue
            model = None
            center = radii = rotation = None
            if old.get("primitive") == "free_curve":
                # Flow moves a visible open search path. It neither closes the
                # path nor forces its current RGB peaks into an ellipse.
                tangent = np.gradient(predicted, axis=0)
                tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
                normals = np.stack((tangent[:,1], -tangent[:,0]), 1)
                normals[(normals*np.asarray(old["normals"])).sum(1) < 0] *= -1
            else:
                model = _ellipse_model(cv2.fitEllipseDirect(predicted[supported].astype(np.float32)))
                if model is None:
                    ledger.append(dict(id=old["id"], status="invalid_source_image_search_geometry", actual_boundary_filled=False))
                    continue
                center, radii, rotation, Q = model
                normals = (predicted-center) @ Q
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
            depth = old.get("material_contour_depth", 0)
            profiles = _reread_conic_source_profiles(rgb, predicted, normals, model,
                                                    material_orientation=1 if depth % 2 == 0 else -1)
            old_luma = old_profiles["full_normal_source_luminance_profile"]
            new_luma = profiles["full_normal_source_luminance_profile"]
            a, b = old_luma-old_luma.mean(1, keepdims=True), new_luma-new_luma.mean(1, keepdims=True)
            correlation = (a*b).sum(1)/np.maximum(np.sqrt((a*a).sum(1)*(b*b).sum(1)), 1e-9)
            comparable = supported & profiles["source_image_conic_supported_profiles"]
            observed = correlation[comparable]
            median = float(np.median(observed)) if len(observed) else 0.
            mad = float(np.median(np.abs(observed-median))) if len(observed) else 0.
            threshold = max(.25, median-max(.25, 3*1.4826*mad))
            signature_reference = comparable & (correlation >= threshold)
            level_delta = new_luma.mean(1)-old_luma.mean(1)
            contrast_log_ratio = np.log(np.maximum(b.std(1), 1e-6)/np.maximum(a.std(1), 1e-6))
            def material_consistency(value, floor):
                visible_values = value[signature_reference]
                center = float(np.median(visible_values)) if len(visible_values) else 0.
                deviation = float(np.median(np.abs(visible_values-center))) if len(visible_values) else 0.
                tolerance = max(floor, 3*1.4826*deviation)
                return np.abs(value-center) <= tolerance, dict(median=center, robust_tolerance=tolerance)
            level_supported, level_reference = material_consistency(level_delta, .025)
            contrast_supported, contrast_reference = material_consistency(contrast_log_ratio, .25)
            accepted = signature_reference & level_supported & contrast_supported
            raw_membership = profiles["source_material_conic_group_membership"]
            rows = profiles["all_actual_source_peak_profile_indices"]
            profiles["temporal_material_signature_group_membership"] = raw_membership & accepted[rows]
            profiles["source_material_signature_conflict"] = ~accepted
            profiles["source_material_normal_profile_correlation"] = correlation
            profiles["source_material_signature_threshold_from_actual_visible_profiles"] = threshold
            profiles["source_material_mean_luminance_change"] = level_delta
            profiles["source_material_contrast_log_ratio"] = contrast_log_ratio
            profiles["source_material_actual_visible_level_reference"] = level_reference
            profiles["source_material_actual_visible_contrast_reference"] = contrast_reference
            # Search coordinates survive for recovery; no rejected actual peak
            # is replaced by an ellipse/flow value, and rejected gaps stay gaps.
            guide = {**old, "points_uv": predicted, "normals": normals, "source_binding": binding,
                     "origin": "temporal_source_search_hint", "full_curve_source_RGB_profiles": profiles,
                     "actual_source_support_uv": np.empty((0, 2), np.float32),
                     "original_component_indices": np.empty(0, np.int32),
                     "image_conic_center_uv": center, "image_conic_radii_px": radii, "image_conic_rotation": rotation,
                     "image_conic_used_as_temporal_search_model": model is not None,
                     "temporal_flow_supported": supported, "source_RGB_material_supported": accepted,
                     "source_RGB_measured_run_ids": _visible_search_runs(accepted),
                     "hint_previous_source_binding": previous["source_binding"],
                     "temporal_flow_metadata": metadata,
                     "closed": bool(accepted.all()), "missing_source_sample_count": int((~accepted).sum()),
                     "status": "same_shot_actual_RGB_material_reread_with_missing_runs"}
            guides.append(guide)
            ledger.append(dict(id=old["id"], accepted_original_RGB_profile_count=int(accepted.sum()),
                missing_or_conflicting_profile_count=int((~accepted).sum()),
                source_flow_supported_count=int(supported.sum()), actual_boundary_filled=False))
        result = dict(source_binding=binding, source_RGB_sha256=digest, guides=guides,
                      source_shape_readings=_direct_source_shape_readings(guides),
                      source_fragment_readings=current_fragments["source_fragment_readings"],
                      memory_update="source_bound_flow_search_and_current_RGB_material_reread",
                      temporal_material_reading_ledger=ledger, previous_source_binding=previous["source_binding"],
                      source_quality_rejections=packet.get("source_quality_rejections", {}),
                      raw_source_packet_preserved=True, interpolated_boundary_uv_generated=False,
                      source_ice_domain_verified=False, physical_marker_names_inferred=False,
                      source_domain_status=previous.get("source_domain_status", "source_material_domain_unresolved"),
                      source_local_identity_is_physical_marker_identity=False,
                      is_training_label=False, accuracy_accepted=False)
        result = continue_confirmed_source_instances(result, previous)
        self.previous = result
        return result
