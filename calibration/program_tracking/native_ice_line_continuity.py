"""Use real image motion to retain the identity of static ice lines.

Transported lines are search guides only. Every admitted measurement must still
be read from the current native image. Wide bands use their own two-side reader;
this helper also carries their line identity. Circles are handled separately.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np


def is_thin_line(path):
    return (not path.get('_circle') and
            path['feature_id'] not in ('B1', 'B2', 'C') and
            path['operator'] == 'paint_center')


def straight_guide(points, extent_hint):
    """Retain the named finite extent; do not create current observations."""
    points = np.asarray(points, float)
    points = points[np.isfinite(points).all(1)]
    if len(points) < 2:
        return np.empty((0, 2))
    center = points.mean(0)
    direction = np.linalg.svd(points - center, full_matrices=False)[2][0]
    hint = np.asarray(extent_hint, float)
    hint = hint[np.isfinite(hint).all(1)]
    if len(hint) < 2:
        hint = points
    along = (hint - center) @ direction
    lo, hi = float(along.min()), float(along.max())
    for axis, bounds in enumerate(((-120, 2039), (-120, 1199))):
        if abs(direction[axis]) > 1e-9:
            edge = sorted((np.asarray(bounds) - center[axis]) / direction[axis])
            lo, hi = max(lo, edge[0]), min(hi, edge[1])
    if hi - lo < 10:
        return np.empty((0, 2))
    return center + np.linspace(lo, hi, max(2, int((hi-lo)/8)+1))[:, None] * direction


def identity_filter(rows, guide, policy):
    """Reject competing material branches, not localize pixels by projection."""
    if not policy or not policy.get('current_identity_constrained') or len(guide) < 2:
        return rows
    origin, end = np.asarray(guide)[[0, -1]]
    tangent = end - origin
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    kept = []
    hull = np.asarray(policy.get('current_source_motion_support_hull_uv', []), np.float32)
    for row in rows:
        distance = float(abs((np.asarray(row['uv'])-origin) @ normal))
        supported = (len(hull) < 3 or cv2.pointPolygonTest(hull, tuple(map(float,row['uv'])), True) >= -2.)
        tolerance = policy['normal_identity_tolerance_px'] if supported else policy['search_radius_px']-4.
        # A bound is only a source identity check; it never replaces row['uv'].
        if distance <= tolerance:
            row['source_evidence']['temporal_material_identity'] = dict(
                guide_only=True, current_material_reread=True,
                identity_source=policy['identity_source'],
                reference_frame=policy['reference_frame'],
                within_measured_motion_support=supported,
                normal_distance_px=distance,
                normal_identity_tolerance_px=tolerance)
            kept.append(row)
    return kept


def thin_material_width_limit(path, native):
    """Use measured narrow paint in this frame to reject filled house bands.

    Contrast alone cannot distinguish a longitudinal stripe from the dark fill
    of a target. The visible lane boundaries provide a local material-width
    reference; scene drawing stroke widths are not physical measurements.
    """
    if not is_thin_line(path):
        return None
    measured = getattr(native, '_thin_paint_width_reference', [])
    if len(measured) < 6:
        return None
    reference_axis = getattr(native, '_thin_paint_width_reference_axis_xy', None)
    if reference_axis is not None and path.get('xyz') is not None:
        xyz=np.asarray(path['xyz'],float)
        axis=xyz[-1,:2]-xyz[0,:2]
        length=float(np.linalg.norm(axis))
        if length and abs(float(axis@reference_axis)/length)<1.-1e-6:
            # Apparent width and motion blur depend on transverse direction.
            # A longitudinal stripe's image width cannot reject an end line.
            return None
    width = float(np.median(measured))
    return dict(maximum_width_px=max(12., 1.75*width),
                current_boundary_width_median_px=width,
                current_boundary_readout_count=len(measured),
                purpose='separate a narrow painted stripe from a filled house band')


def record_thin_material_width(path, native, rows):
    if path['curve_id'] not in ('lane4:edge2','lane3:edge0','lane3:edge2',
                                'lane2:edge0','lane2:edge2','lane1:edge0'):
        return
    if len(rows) < 6:
        return
    uv = np.asarray([r['uv'] for r in rows], float)
    if np.linalg.norm(np.ptp(uv,axis=0)) < 350:
        return
    widths = [float(np.linalg.norm(np.diff(np.asarray(
        r['source_evidence']['actual_source_sides_uv'],float),axis=0))) for r in rows]
    native._thin_paint_width_reference = getattr(native, '_thin_paint_width_reference', []) + widths
    if path.get('xyz') is not None:
        xyz=np.asarray(path['xyz'],float);axis=xyz[-1,:2]-xyz[0,:2]
        if np.linalg.norm(axis)>0:native._thin_paint_width_reference_axis_xy=axis/np.linalg.norm(axis)


def current_material_intervals(stations, readout_stations, *, maximum_gap=260.):
    """Represent supported runs without expanding an isolated edge response.

    Neighbouring source measurements may jointly locate a faint straight
    segment. A lone response cannot grow into a visible line on either side.
    The model and temporal guide still determine which physical line to read.
    """
    values = np.unique(np.asarray(readout_stations,float))
    mask = np.zeros(len(stations),bool)
    if len(values) < 2:
        return mask
    for group in np.split(values,np.flatnonzero(np.diff(values)>maximum_gap)+1):
        if len(group)>=2 and group[-1]-group[0]>=20:
            mask |= (stations>=group[0]) & (stations<=group[-1])
    return mask


class NativeLineIdentity:
    def __init__(self, motion_quality_path, guides):
        self.motion_quality_path = Path(motion_quality_path)
        record = json.loads(self.motion_quality_path.read_text(encoding='utf8'))
        self.motion = {r['index']: r for r in record['registration']}
        support_path = self.motion_quality_path.with_name('all_source_motion_support.json')
        self.support = {}
        if support_path.exists():
            support = json.loads(support_path.read_text(encoding='utf8'))
            self.support = {r['index']:r for r in support['records']}
        initial_path = self.motion_quality_path.with_name('initial_source_motion_200_201.json')
        if initial_path.exists():
            initial = json.loads(initial_path.read_text(encoding='utf8'))
            initial = initial.get('registration', initial)
            if isinstance(initial, dict) and initial.get('index') == 201:
                self.motion[201] = initial
                self.support[201] = initial
        self.guides = [g for g in guides if g.get('kind') == 'line']
        # A native visual identity review supplies a search location, never
        # output vertices. This corrects the known F3 query on the broad band.
        review_path=self.motion_quality_path.parent/'thin_source_identity/source_identity_000201.json'
        if review_path.exists():
            review=json.loads(review_path.read_text(encoding='utf8'))
            source=Path(review['source_png_path'])
            if hashlib.sha256(source.read_bytes()).hexdigest()!=review['source_png_sha256']:
                raise ValueError('Native line identity review source changed')
            self.guides.append(dict(kind='line',frame=review['programme_index'],
                curve_id='house_connection_F3_R3:segment0',
                uv=review['F3_R3']['approximate_visible_source_interval_uv'],
                authority='native_visual_identity_review',review_path=str(review_path)))
        self.states = {}
        self.last_policy = {}

    def seed_accepted_frame(self, packet_path, acceptance_path):
        packet_path, acceptance_path = Path(packet_path), Path(acceptance_path)
        receipt = json.loads(acceptance_path.read_text(encoding='utf8'))
        index = int(packet_path.stem)
        accepted = next((r for r in receipt['frames'] if r['index'] == index), None)
        if not receipt.get('accepted') or not accepted or accepted['packet_sha256'] != hashlib.sha256(packet_path.read_bytes()).hexdigest():
            raise ValueError('Thin-line source initialization is not the accepted packet')
        packet = json.loads(packet_path.read_text(encoding='utf8'))
        for feature in packet['features']:
            cid = feature.get('target_id', '').removeprefix('ice.')
            if not cid.startswith(('lane', 'house_connection_', 'curling_centerline')):
                continue
            uv = np.asarray([p for s in feature['segments'] for p in s['uv']],float)
            if len(uv) < 2:
                continue
            guide = straight_guide(uv,uv)
            if len(guide) >= 2:
                self.states[cid] = dict(frame=index,guide=guide,
                    user_accepted_source_packet=str(packet_path),
                    user_acceptance_receipt=str(acceptance_path))

    def _transport(self, a, b, points):
        if a >= b:
            return None
        uv = np.asarray(points, float).copy()
        variance = 0.
        maximum_error = 0.
        current_hull = None
        for index in range(a+1, b+1):
            r = self.motion.get(index)
            if not r or not r.get('usable'):
                return None
            h = np.asarray(r['homography'], float)
            if h.shape != (3, 3) or not np.isfinite(h).all():
                return None
            error = float(r['p95_reprojection_px'])
            variance += error**2
            maximum_error = max(maximum_error, error)
            # Missing intervals may not accumulate unbounded registration drift.
            if np.sqrt(variance) > 6.:
                return None
            supported = self.support.get(index, {})
            hull = np.asarray(supported.get('inlier_convex_hull_uv', []), np.float32).reshape(-1,2)
            if len(hull) >= 3:
                within = np.array([cv2.pointPolygonTest(hull, tuple(map(float,p)), True) >= -2. for p in uv])
                if within.sum() < 2 or np.linalg.norm(np.ptp(uv[within],axis=0)) < 35:
                    return None
                # Use the observed part of the old line for transport. The
                # source's finite model extent is restored only as a guide.
                uv = uv[within]
                current_hull = cv2.perspectiveTransform(hull[None].astype(np.float64),h)[0]
            uv = cv2.perspectiveTransform(uv[None].astype(np.float64), h)[0]
            if not np.isfinite(uv).all():
                return None
        return uv, float(np.sqrt(variance)), maximum_error, current_hull

    def select_guide(self, path, index, fallback, origin):
        cid = path['curve_id']
        policy = dict(current_identity_constrained=False,
                      identity_source='existing_model_or_human_navigation',
                      native_image_coordinates_not_generated=True)
        if path.get('_circle') or path['operator'] != 'paint_center':
            return fallback, origin, policy
        exact = [g for g in self.guides if g['curve_id'] == cid and g['frame'] == index]
        if exact:
            # Prefer the current programme's explicit user drawing over a
            # recovered trace on another source representation.
            exact.sort(key=lambda g: str(g.get('authority', '')).startswith('recovered_'))
            guide = straight_guide(exact[0]['uv'], fallback)
            if len(guide) >= 2:
                identity_source=('current_frame_native_visual_identity_review' if exact[0].get('authority')=='native_visual_identity_review' else 'current_frame_user_labelled_line')
                policy.update(current_identity_constrained=True,
                              identity_source=identity_source,
                              reference_frame=index, normal_identity_tolerance_px=9.,
                              search_radius_px=18., registration_error_px=0.)
                if exact[0].get('review_path'):policy['native_identity_review_path']=exact[0]['review_path']
                self.last_policy[cid] = policy
                return guide, policy['identity_source'], policy
        state = self.states.get(cid)
        if state:
            moved = self._transport(state['frame'], index, state['guide'])
            if moved is not None:
                uv, error, maximum_error, hull = moved
                guide = straight_guide(uv, fallback)
                if len(guide) >= 2:
                    policy.update(current_identity_constrained=True,
                                  identity_source='previous_native_material_with_current_source_image_motion',
                                  reference_frame=state['frame'],
                                  normal_identity_tolerance_px=min(14., 6.+2.*error),
                                  search_radius_px=min(24., 14.+2.*error),
                                  registration_error_px=error,
                                  maximum_registration_p95_px=maximum_error,
                                  motion_quality_path=str(self.motion_quality_path))
                    if hull is not None:
                        policy['current_source_motion_support_hull_uv']=hull.tolist()
                        policy['motion_outside_support_is_not_validation']=True
                    self.last_policy[cid] = policy
                    return guide, policy['identity_source'], policy
        self.last_policy[cid] = policy
        return fallback, origin, policy

    def record(self, index, features):
        """Call after physical family checks; rejected candidates cannot seed it."""
        for feature in features:
            fit = feature.get('source_line_fit')
            if not fit or feature.get('point_count', 0) < 2:
                continue
            cid = feature['target_id'].removeprefix('ice.')
            rows = fit.get('direct_readouts', [])
            if len(rows) < 2:
                continue
            origin = np.asarray(fit['point_origin'], float)
            tangent = np.asarray(fit['unit_tangent'], float)
            normal = np.asarray(fit['unit_normal'], float)
            coefficients = np.asarray(fit['normal_offset_coefficients'], float)
            native_uv = np.asarray([r['uv'] for r in rows], float)
            stations = (native_uv-origin) @ tangent
            span = float(np.ptp(stations))
            if span < 35:
                continue
            stations = np.linspace(stations.min(), stations.max(), 33)
            guide = origin + stations[:,None]*tangent + np.polyval(coefficients,stations)[:,None]*normal
            self.states[cid] = dict(frame=index, guide=guide,
                                    actual_current_native_readouts=len(rows),
                                    source_span_px=span)
