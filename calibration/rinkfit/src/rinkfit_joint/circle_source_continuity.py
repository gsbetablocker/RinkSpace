"""Carry a confirmed ice-paint arc through actual adjacent source exposures.

Source-to-source ice registration only locates a finite profile bracket. Every
returned UV is independently reread from both current RGB material interfaces.
Ambiguous profiles, occlusion and lost registration remain failures. A new
source anchor is required after a gap; no world/model ellipse fills missing UV.
"""
from __future__ import annotations
import hashlib
import numpy as np


def _same_source_step(previous, current):
    if any(previous.get(k) != current.get(k) for k in ('source_sha256', 'time_base', 'shot_id')):
        raise ValueError('Circle source track crosses a source/clock/camera cut')
    if previous.get('time_base') != '1/12800' or abs(int(current['source_pts'])-int(previous['source_pts'])) != 256:
        raise ValueError('A source circle chain requires adjacent official 50fps exposures')


def _warp(points, H):
    z = np.c_[points, np.ones(len(points))] @ H.T
    if np.any(abs(z[:, 2]) < 1e-12):
        raise ValueError('Source registration has a pole in this arc')
    return z[:, :2] / z[:, 2, None]


def _footprint_clear(point, normal, mask, half_width, tangent_half_width):
    tangent = np.array([-normal[1], normal[0]])
    q = (point + np.arange(-half_width, half_width+1)[:, None, None]*normal
         + np.arange(-tangent_half_width, tangent_half_width+1)[None, :, None]*tangent)
    q = np.rint(q.reshape(-1, 2)).astype(int)
    return bool(np.all((q[:, 0] >= 0) & (q[:, 0] < mask.shape[1])
                       & (q[:, 1] >= 0) & (q[:, 1] < mask.shape[0]))
                and np.all(mask[q[:, 1], q[:, 0]]))


def read_adjacent_source_circle_arcs(previous_rgb, current_rgb, *, previous_binding,
        current_binding, previous_arc_uv, previous_arc_normals, previous_static_ice_mask,
        current_static_ice_mask, profile_half_width_px=8, tangent_half_width_px=2,
        forward_backward_tolerance_px=1., registration_tolerance_px=1.5):
    """Return exact current material candidates plus an auditable source chain.

    Static ice masks must exclude complete persons/context, graphics, boards and
    other non-ice pixels. They are not inferred by this function. Distributed
    static ice keypoints and forward/backward flow verify the planar source
    registration. One unique observed dark stroke is required per query.
    """
    import cv2
    from rinkspace.vision_source_shapes import read_source_dark_stroke_profiles
    from rinkspace.vision_instances import source_binding
    _same_source_step(previous_binding, current_binding)
    before = np.asarray(previous_rgb); now = np.asarray(current_rgb)
    if before.shape != now.shape or before.ndim != 3 or before.dtype != np.uint8 or now.dtype != np.uint8:
        raise ValueError('Exact same-size native uint8 RGB exposures required')
    previous_binding = source_binding(previous_binding, before.shape[:2])
    current_binding = source_binding(current_binding, now.shape[:2])
    for rgb, binding in ((before, previous_binding), (now, current_binding)):
        if hashlib.sha256(rgb.tobytes()).hexdigest() != binding.get('decoded_native_rgb_sha256'):
            raise ValueError('Source-circle RGB authentication failed')
    pm = np.asarray(previous_static_ice_mask, bool); cm = np.asarray(current_static_ice_mask, bool)
    if pm.shape != before.shape[:2] or cm.shape != now.shape[:2]:
        raise ValueError('Static ice source mask shape differs')
    points = np.asarray(previous_arc_uv, float).reshape(-1, 2)
    normals = np.asarray(previous_arc_normals, float).reshape(-1, 2)
    if len(points) != len(normals) or not len(points) or not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError('Actual previous source arc Queries and normals required')
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    gray0 = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY); gray1 = cv2.cvtColor(now, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray0, 600, .01, 12, mask=pm.astype('uint8')*255)
    if corners is None or len(corners) < 12:
        raise ValueError('Insufficient static ice landmarks; preserve the source gap')
    opts = dict(winSize=(31, 31), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))
    nxt, ok1, err = cv2.calcOpticalFlowPyrLK(gray0, gray1, corners, None, **opts)
    rev, ok2, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, nxt, None, **opts)
    a = corners[:, 0].astype(float); b = nxt[:, 0].astype(float)
    fb = np.linalg.norm(rev[:, 0]-a, axis=1)
    ij = np.rint(b).astype(int)
    inside = ((ij[:, 0] >= 0) & (ij[:, 0] < cm.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < cm.shape[0]))
    keep = (ok1[:, 0] != 0) & (ok2[:, 0] != 0) & (fb <= forward_backward_tolerance_px) & inside
    valid = np.flatnonzero(inside); keep[valid] &= cm[ij[valid, 1], ij[valid, 0]]
    if keep.sum() < 12:
        raise ValueError('Static source forward/backward landmarks lost')
    H, inliers = cv2.findHomography(a[keep], b[keep], cv2.RANSAC, registration_tolerance_px)
    if H is None or inliers is None or inliers.sum() < 12:
        raise ValueError('Static ice source registration is unresolved')
    inlier = inliers[:, 0].astype(bool); aa = a[keep][inlier]; bb = b[keep][inlier]
    if np.ptp(aa[:, 0]) < 600 or np.ptp(aa[:, 1]) < 80:
        raise ValueError('Source landmarks do not constrain a distributed ice region: count=%d span=%s' % (len(aa), np.ptp(aa, axis=0).tolist()))
    error = np.linalg.norm(_warp(aa, H)-bb, axis=1)
    seeds = _warp(points, H)
    tangents = np.c_[-normals[:, 1], normals[:, 0]]
    t1 = _warp(points+tangents, H)-seeds
    nn = np.c_[t1[:, 1], -t1[:, 0]]; nn /= np.linalg.norm(nn, axis=1)[:, None]
    clear = np.array([_footprint_clear(p, n, pm, profile_half_width_px, tangent_half_width_px)
                      and _footprint_clear(q, n1, cm, profile_half_width_px, tangent_half_width_px)
                      for p, n, q, n1 in zip(points, normals, seeds, nn)])
    result = [dict(query_index=i, disposition='source_ice_or_occlusion_unknown',
                   actual_source_uv=[], source_UV_invented=False) for i in range(len(points))]
    indices = np.flatnonzero(clear)
    if len(indices):
        profiles = read_source_dark_stroke_profiles(now, binding=current_binding,
            search_points_uv=seeds[indices], normals=nn[indices],
            search_intervals_px=np.tile([-float(profile_half_width_px), float(profile_half_width_px)], (len(indices), 1)),
            tangent_half_width_px=tangent_half_width_px, engineering_support_margin_px=0.)
        for i, reading in zip(indices, profiles['readings']):
            candidates = reading['all_actual_source_dark_stroke_candidates']
            measured = [c for c in candidates if c['disposition'] == 'measured_dark_stroke_center_candidate']
            # An unresolved competitor is not erased to leave an artificial winner.
            unique = len(candidates) == len(measured) == 1
            result[i] = dict(query_index=int(i), disposition='source_arc_midline_reread' if unique else 'source_material_competition_or_blur_unknown',
                actual_source_uv=measured[0]['actual_source_uv'].tolist() if unique else [],
                actual_paired_material_profile=measured[0] if unique else None,
                search_seed_uv=seeds[i].tolist(), current_source_normal=nn[i].tolist(),
                source_warp_is_search_only=True, source_UV_invented=False,
                all_actual_candidates_retained=candidates)
    return dict(schema='rinkfit.adjacent_static_source_circle_arc_readings.v1',
                previous_source_binding=previous_binding, current_source_binding=current_binding,
                source_homography=H.tolist(), source_landmarks=int(len(aa)),
                landmark_reprojection_max_px=float(error.max()),
                source_forward_backward_max_px=float(fb[keep].max()),
                previous_static_ice_mask_sha256=hashlib.sha256(pm.tobytes()).hexdigest(),
                current_static_ice_mask_sha256=hashlib.sha256(cm.tobytes()).hexdigest(),
                source_warp_is_observation=False, physical_circle_identity_granted=False,
                readings=result, unknown_rows_filled=False)


def read_adjacent_source_arc_contours(previous_rgb, current_rgb, *, previous_binding,
        current_binding, previous_arc_uv, previous_arc_normals, previous_static_ice_mask,
        current_static_ice_mask, profile_half_width_px=8, tangent_half_width_px=2,
        forward_backward_tolerance_px=1.):
    """Track confirmed local contour neighborhoods when whole-ice H is absent.

    This is contour identity, not a persistent material-point correspondence.
    Tangential aperture drift cannot issue static-motion latents or station
    correspondence. Failed local tracks cannot be filled by a global model.
    """
    import cv2
    from rinkspace.vision_source_shapes import read_source_dark_stroke_profiles
    from rinkspace.vision_instances import source_binding
    _same_source_step(previous_binding, current_binding)
    a = np.asarray(previous_rgb); b = np.asarray(current_rgb)
    if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError('Exact native source RGB pair required')
    previous_binding = source_binding(previous_binding, a.shape[:2])
    current_binding = source_binding(current_binding, b.shape[:2])
    for rgb, binding in ((a, previous_binding), (b, current_binding)):
        if hashlib.sha256(rgb.tobytes()).hexdigest() != binding.get('decoded_native_rgb_sha256'):
            raise ValueError('Source contour RGB authentication failed')
    pm = np.asarray(previous_static_ice_mask, bool); cm = np.asarray(current_static_ice_mask, bool)
    if pm.shape != a.shape[:2] or cm.shape != b.shape[:2]:
        raise ValueError('Static ice source mask shape differs')
    points = np.asarray(previous_arc_uv, float).reshape(-1, 2)
    normals = np.asarray(previous_arc_normals, float).reshape(-1, 2)
    if len(points) != len(normals) or not len(points) or not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError('Actual previous source contour Queries required')
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0):raise ValueError('Nonzero source contour normals required')
    normals = normals / lengths[:, None]
    gray0 = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY); gray1 = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    opts = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01))
    query = points.astype('float32').reshape(-1, 1, 2)
    next_uv, valid1, errors = cv2.calcOpticalFlowPyrLK(gray0, gray1, query, None, **opts)
    back_uv, valid2, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, next_uv, None, **opts)
    seeds = next_uv[:, 0].astype(float); fb = np.linalg.norm(back_uv[:, 0]-points, axis=1)
    valid = (valid1[:, 0] != 0) & (valid2[:, 0] != 0) & (fb <= forward_backward_tolerance_px)
    # Read enough context around the whole queried paint stroke, not only the
    # measured center. A person anywhere in the local tracking window vetoes it.
    for i, (p, q, n) in enumerate(zip(points, seeds, normals)):
        valid[i] &= (_footprint_clear(p, n, pm, 15, 15) and _footprint_clear(q, n, cm, 15, 15))
    result = [dict(query_index=i, disposition='source_contour_track_or_static_ice_unknown',
                   actual_source_uv=[], source_UV_invented=False) for i in range(len(points))]
    ii = np.flatnonzero(valid)
    if len(ii):
        profiles = read_source_dark_stroke_profiles(b, binding=current_binding,
            search_points_uv=seeds[ii], normals=normals[ii],
            search_intervals_px=np.tile([-float(profile_half_width_px), float(profile_half_width_px)], (len(ii), 1)),
            tangent_half_width_px=tangent_half_width_px, engineering_support_margin_px=0.)
        for i, reading in zip(ii, profiles['readings']):
            candidates = reading['all_actual_source_dark_stroke_candidates']
            measured = [c for c in candidates if c['disposition'] == 'measured_dark_stroke_center_candidate']
            unique = len(candidates) == len(measured) == 1
            result[i] = dict(query_index=int(i),
                disposition='source_arc_midline_reread' if unique else 'source_material_competition_or_blur_unknown',
                actual_source_uv=measured[0]['actual_source_uv'].tolist() if unique else [],
                actual_paired_material_profile=measured[0] if unique else None,
                source_forward_backward_error_px=float(fb[i]), source_LK_patch_error=float(errors[i, 0]),
                source_search_seed_uv=seeds[i].tolist(), current_source_normal=normals[i].tolist(),
                flow_coordinate_is_observation=False, all_actual_candidates_retained=candidates)
    return dict(schema='rinkfit.adjacent_source_circle_contour_readings.v1',
                previous_source_binding=previous_binding, current_source_binding=current_binding,
                readings=result, source_forward_backward_errors_px=fb.tolist(),
                source_local_track_pass_count=int(valid.sum()), global_source_registration_issued=False,
                source_contour_identity_only=True, persistent_material_point_correspondence=False,
                station_correspondence_issued=False, static_motion_latents_issued=False,
                unknown_rows_filled=False, physical_circle_identity_granted=False)


def read_adjacent_source_profile_contours(previous_rgb, current_rgb, *, previous_binding,
        current_binding, previous_arc_uv, previous_arc_normals, previous_static_ice_mask,
        current_static_ice_mask, profile_half_width_px=20, tangent_half_width_px=2,
        minimum_profile_correlation=.8):
    """Carry only an isolated contour through its measured normal profile.

    An untextured paint edge has no observable tangential point coordinate.
    This branch explicitly leaves that station unknown. Both source exposures
    must contain exactly one independent paired paint band in the source-only
    bracket, with matching measured normal appearance. No missing center or
    flow coordinate is output as a source observation.
    """
    import cv2
    from rinkspace.vision_source_shapes import read_source_dark_stroke_profiles
    from rinkspace.vision_instances import source_binding
    _same_source_step(previous_binding, current_binding)
    before = np.asarray(previous_rgb); now = np.asarray(current_rgb)
    if before.shape != now.shape or before.dtype != np.uint8 or now.dtype != np.uint8:
        raise ValueError('Exact same-size native source RGB pair required')
    previous_binding = source_binding(previous_binding, before.shape[:2])
    current_binding = source_binding(current_binding, now.shape[:2])
    for rgb, binding in ((before, previous_binding), (now, current_binding)):
        if hashlib.sha256(rgb.tobytes()).hexdigest() != binding.get('decoded_native_rgb_sha256'):
            raise ValueError('Source normal-profile RGB authentication failed')
    pm = np.asarray(previous_static_ice_mask, bool); cm = np.asarray(current_static_ice_mask, bool)
    if pm.shape != before.shape[:2] or cm.shape != now.shape[:2]:
        raise ValueError('Static source ice masks differ')
    points = np.asarray(previous_arc_uv, float).reshape(-1, 2)
    normals = np.asarray(previous_arc_normals, float).reshape(-1, 2)
    if len(points) != len(normals) or not len(points) or not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError('Actual previous source contour Queries required')
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 0):raise ValueError('Nonzero source normal required')
    normals /= lengths[:, None]
    clear = np.array([_footprint_clear(p, n, pm, max(15, profile_half_width_px+4), 15) and _footprint_clear(p, n, cm, max(15, profile_half_width_px+4), 15)
                      for p, n in zip(points, normals)])
    ii = np.flatnonzero(clear)
    result = [dict(query_index=i, disposition='source_ice_or_occlusion_unknown',
                   actual_source_uv=[], source_UV_invented=False) for i in range(len(points))]
    def actual_profile(rgb, center, normal):
        tangent = np.array([-normal[1], normal[0]])
        q = (center + np.arange(-8., 9.)[:, None, None]*normal
             + np.arange(-2., 3.)[None, :, None]*tangent)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return cv2.remap(gray, q[:, :, 0].astype('float32'), q[:, :, 1].astype('float32'), cv2.INTER_LINEAR).mean(1)
    if len(ii):
        common = dict(search_points_uv=points[ii], normals=normals[ii],
            search_intervals_px=np.tile([-float(profile_half_width_px), float(profile_half_width_px)], (len(ii), 1)),
            tangent_half_width_px=tangent_half_width_px, engineering_support_margin_px=0.)
        prior = read_source_dark_stroke_profiles(before, binding=previous_binding, **common)
        current = read_source_dark_stroke_profiles(now, binding=current_binding, **common)
        for i, old, new in zip(ii, prior['readings'], current['readings']):
            a = old['all_actual_source_dark_stroke_candidates']; b = new['all_actual_source_dark_stroke_candidates']
            unique = (len(a) == len(b) == 1 and a[0]['disposition'] == b[0]['disposition'] == 'measured_dark_stroke_center_candidate')
            result[i] = dict(query_index=int(i), disposition='source_material_competition_or_blur_unknown',
                actual_source_uv=[], old_actual_candidates=a, current_actual_candidates=b)
            if not unique:continue
            pa = actual_profile(before, a[0]['actual_source_uv'], normals[i]); pb = actual_profile(now, b[0]['actual_source_uv'], normals[i])
            pa = pa-pa.mean(); pb = pb-pb.mean(); denominator = np.linalg.norm(pa)*np.linalg.norm(pb)
            correlation = float(pa@pb/denominator) if denominator > 0 else -1.
            if correlation < minimum_profile_correlation:
                result[i]['source_profile_correlation'] = correlation;continue
            result[i] = dict(query_index=int(i), disposition='source_arc_midline_reread',
                actual_source_uv=b[0]['actual_source_uv'].tolist(), current_source_normal=normals[i].tolist(),
                actual_paired_material_profile=b[0], previous_actual_paired_material_profile=a[0],
                source_profile_correlation=correlation, actual_source_normal_matching_only=True,
                source_search_is_observation=False, persistent_material_point_correspondence=False)
    return dict(schema='rinkfit.adjacent_source_circle_normal_contour_readings.v1',
        previous_source_binding=previous_binding, current_source_binding=current_binding, readings=result,
        source_local_track_pass_count=sum(r['disposition']=='source_arc_midline_reread' for r in result),
        source_contour_identity_only=True, physical_circle_identity_granted=False,
        source_tangential_station_unknown=True, station_correspondence_issued=False,
        static_motion_latents_issued=False, model_navigation_used=False, unknown_rows_filled=False)
