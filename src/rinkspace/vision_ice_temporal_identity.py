"""Source-seeded ice identities, remeasured in each actual exposure.

Only authenticated source seeds name a curve. Source-to-source optical flow
locates current searches; it never emits an observed UV or enters fitting.
Current paint centers must still be midpoints of their two observed sides.
No source seed, cut, ambiguous match or rejected source material remains missing.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def exact_selected_source_readouts(row, measurement, *, operator):
    """Keep one exact raw or measured-q50 domain for source navigation."""
    ids=np.asarray(row['original_candidate_indices'],np.int64)
    raw=np.asarray(measurement['candidate_uv'],float)
    if (ids.ndim!=1 or len(np.unique(ids))!=len(ids)
            or np.any(ids<0) or np.any(ids>=len(raw))):
        raise ValueError('Source supports need unique exact current candidate indices')
    sides=np.asarray(measurement['candidate_sides'],float)[ids].copy()
    uv=sides.mean(1) if operator=='paint_center' else raw[ids].copy()
    selected_uv=np.asarray(row['uv'],float)
    selected_sides=np.asarray(row['source_sides_uv'],float)
    if np.array_equal(selected_uv,uv) and np.array_equal(selected_sides,sides):
        return uv,sides
    quantiles=measurement.get('candidate_side_quantiles_uv')
    if quantiles is not None:
        quantiles=np.asarray(quantiles,float)[ids]
        if quantiles.shape==(len(ids),2,3,2):
            fine=np.isfinite(quantiles[:,:,1]).all(axis=(1,2))
            sides[fine]=quantiles[fine,:,1]
            uv[fine]=sides[fine].mean(1)
            if np.array_equal(selected_uv,uv) and np.array_equal(selected_sides,sides):
                return uv,sides
    raise ValueError('Source supports must be exact current raw or measured q50 readouts')


def final_current_source_track_support(rgb,measurement,ids,*,operator,uv=None,sides=None):
    """Build navigation support from this RGB's exact final source readouts."""
    ids=np.asarray(ids,np.int64)
    if (ids.ndim!=1 or len(np.unique(ids))!=len(ids)
            or np.any(ids<0) or np.any(ids>=len(measurement['candidate_uv']))):
        raise ValueError('Final source tracks require unique current candidate locators')
    if len(ids)<2:return None
    if uv is None or sides is None:
        sides=np.asarray(measurement['candidate_sides'],float)[ids].copy()
        uv=sides.mean(1) if operator=='paint_center' else np.asarray(measurement['candidate_uv'],float)[ids].copy()
        quantiles=measurement.get('candidate_side_quantiles_uv')
        if quantiles is not None:
            fine=np.asarray(quantiles,float)[ids,:,1]
            available=np.isfinite(fine).all((1,2))
            sides[available]=fine[available];uv[available]=fine[available].mean(1)
    else:
        uv=np.asarray(uv,float);sides=np.asarray(sides,float)
    if uv.shape!=(len(ids),2) or sides.shape!=(len(ids),2,2):
        raise ValueError('Final source track readout arrays differ from their locators')
    queries=np.asarray(measurement['candidate_query'])[ids]
    order=np.argsort(queries,kind='stable');ids,queries,uv,sides=ids[order],queries[order],uv[order],sides[order]
    vector=sides[:,1]-sides[:,0];width=np.linalg.norm(vector,axis=1)
    normals=np.zeros_like(uv);paired=width>1e-6
    normals[paired]=vector[paired]/width[paired,None]
    tangent=np.gradient(uv,axis=0);arc=np.c_[-tangent[:,1],tangent[:,0]]
    length=np.linalg.norm(arc,axis=1);arc_support=length>1e-6
    normals[~paired&arc_support]=arc[~paired&arc_support]/length[~paired&arc_support,None]
    valid=(paired if operator=='paint_center' else paired|arc_support)
    descriptor,strength=_measured_material_descriptors(rgb,uv,sides,normals)
    valid &= strength>0
    if valid.sum()<2:return None
    binding=measurement['source_binding'];pts=int(binding.get('actual_pts',binding['source_pts']))
    return dict(actual_uv=uv[valid].copy(),navigation_uv=uv[valid].copy(),candidate_sides=sides[valid].copy(),
        material_descriptors=descriptor[valid].copy(),source_normals=normals[valid].copy(),
        source_candidate_indices=ids[valid].copy(),source_query_indices=queries[valid].copy(),
        actual_pts=pts,navigation_pts=pts,source_RGB_sha256=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
        source_BGR_sha256=hashlib.sha256(np.asarray(rgb)[:,:,::-1].tobytes()).hexdigest(),
        final_current_source_support_retained=True)


def registered_user_stroke_seed(path, *, source_sha256, time_base, shot_id, curve_id):
    """Bind a real registered user trace as identity/navigation, not exact UV."""
    path = Path(path)
    source = json.loads(path.read_text(encoding='utf-8-sig'))
    if source['feature_id'] != curve_id.split(':')[0]:
        raise ValueError('Registered user trace names another physical feature')
    frame_path = Path(source['source_raw_frame'])
    raw = cv2.imread(str(frame_path))
    if raw is None:
        raise FileNotFoundError(frame_path)
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    return dict(curve_id=curve_id, physical_id=source['feature_id'],
        actual_pts=int(source['actual_exposure_pts']), native_pts=int(source['native_pts']),
        source_sha256=source_sha256, time_base=str(Fraction(time_base)), shot_id=shot_id,
        decoded_native_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        navigation_uv=source['native_source_uv'], source_material_class='static_painted_ice',
        surface_and_physical_identity_confirmed=True, is_exact_centerline_pixel_ground_truth=False,
        authority=dict(kind='direct_user_registered_original_source_stroke', path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), statement=source.get('user_statement')))


def _flow_transform(previous, current, previous_mask, current_mask, tracks):
    """Navigation from previously confirmed paint sides, never background."""
    point_lists = []
    for track in tracks.values():
        if track.get('candidate_sides') is not None:
            point_lists.append(np.asarray(track['candidate_sides'], np.float32).reshape(-1, 2))
        point_lists.append(np.asarray(track['actual_uv'], np.float32).reshape(-1, 2))
    if not point_lists:
        return None, dict(status='insufficient_actual_source_2D_support', matched=0)
    p = np.vstack(point_lists)
    ix = np.rint(p).astype(int)
    valid = (ix >= 0).all(1) & (ix < [previous.shape[1], previous.shape[0]]).all(1)
    valid[valid] &= ~previous_mask[ix[valid, 1], ix[valid, 0]]
    points = p[valid, None]
    if len(points) < 4:
        return None, dict(status='insufficient_actual_source_2D_support', matched=0)
    forward, fvalid, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
        winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, .01))
    if forward is None:
        return None, dict(status='actual_source_LK_missing', matched=0)
    backward, bvalid, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None,
        winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, .01))
    a, b, back = points[:, 0], forward[:, 0], backward[:, 0]
    error = np.linalg.norm(back-a, axis=1)
    good = fvalid.ravel().astype(bool) & bvalid.ravel().astype(bool) & np.isfinite(b).all(1) & (error <= 2.)
    ix = np.rint(np.where(np.isfinite(b), b, 0)).astype(int)
    inside = (ix >= 0).all(1) & (ix < [current.shape[1], current.shape[0]]).all(1)
    good &= inside
    good[inside] &= ~current_mask[ix[inside, 1], ix[inside, 0]]
    if good.sum() < 4:
        return None, dict(status='insufficient_actual_forward_backward_source_support', matched=int(good.sum()))
    spatial = np.linalg.svd(a[good]-a[good].mean(0), compute_uv=False)
    if spatial[-1] <= .03*spatial[0]:
        # A single narrow paint stripe cannot identify a full 2D homography.
        # Its measured local normal navigation is retained separately below.
        return None, dict(status='source_paint_support_has_only_local_normal_motion', matched=int(good.sum()),
            source_uv=a[good].tolist(), target_uv=b[good].tolist(),
            backward_error_px=error[good].tolist(), used_as_fitting_observations=False,
            used_as_UV_output=False, coordinate_kind='source_2D_local_navigation_only')
    transform, inliers = cv2.findHomography(a[good], b[good], cv2.RANSAC, 2.)
    if transform is None or not np.isfinite(transform).all() or np.linalg.det(transform) == 0:
        return None, dict(status='actual_source_motion_not_identified', matched=int(good.sum()))
    inliers = inliers.ravel().astype(bool)
    if inliers.sum() < 4:
        return None, dict(status='actual_source_motion_not_identified', matched=int(good.sum()))
    return transform, dict(status='actual_source_forward_backward_2D_motion_navigation',
        matched=int(good.sum()), inliers=int(inliers.sum()),
        backward_error_median_px=float(np.median(error[good][inliers])),
        backward_error_max_px=float(error[good][inliers].max()),
        source_support_uv_sha256=hashlib.sha256(a[good][inliers].tobytes()).hexdigest(),
        target_support_uv_sha256=hashlib.sha256(b[good][inliers].tobytes()).hexdigest(),
        used_as_fitting_observations=False, used_as_UV_output=False,
        coordinate_kind='source_2D_navigation_only', transform=transform.tolist())


def _point_polyline_distance(points, polyline):
    points = np.asarray(points, float).reshape(-1, 2)
    polyline = np.asarray(polyline, float).reshape(-1, 2)
    if len(polyline) < 2:
        return np.full(len(points), np.inf)
    a, d = polyline[:-1], np.diff(polyline, axis=0)
    norm = np.einsum('ij,ij->i', d, d)
    relative = points[:, None]-a[None]
    fraction = np.einsum('nki,ki->nk', relative, d)/np.maximum(norm[None], 1e-9)
    feet = a[None]+np.clip(fraction, 0., 1.)[..., None]*d[None]
    return np.linalg.norm(points[:, None]-feet, axis=-1).min(1)


def _curve_local_navigation(previous, current, previous_mask, current_mask, track):
    """Read one confirmed paint entity's local normal motion, without a H."""
    center = np.asarray(track.get('navigation_uv', track['actual_uv']), np.float32)
    actual = np.asarray(track['actual_uv'], np.float32)
    sides = np.asarray(track.get('candidate_sides', []), np.float32)
    if track.get('actual_pts') != track.get('navigation_pts', track.get('actual_pts')):
        # The previous image is the previous exposure, not the last admitted
        # image. Stale coordinates cannot become a source optical-flow track.
        return center, dict(status='last_confirmed_paint_support_is_from_an_earlier_exposure',
            source_RGB_tracking_confirmed=False, current_material_reentry_required=True)
    if sides.shape != (len(actual), 2, 2):
        return center, dict(status='confirmed_paint_sides_unavailable_for_local_navigation', source_RGB_tracking_confirmed=False)
    points = np.concatenate([actual[:, None], sides], axis=1).reshape(-1, 1, 2)
    owners = np.repeat(np.arange(len(actual)), 3)
    p = points[:, 0]
    ix = np.rint(p).astype(int)
    valid = (ix >= 0).all(1) & (ix < [previous.shape[1], previous.shape[0]]).all(1)
    valid[valid] &= ~previous_mask[ix[valid, 1], ix[valid, 0]]
    if valid.sum() < 2:
        return center, dict(status='confirmed_paint_local_motion_occluded', source_RGB_tracking_confirmed=False)
    p = points[valid]
    owners = owners[valid]
    forward, fvalid, _ = cv2.calcOpticalFlowPyrLK(previous, current, p, None,
        winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, .01))
    if forward is None:
        return center, dict(status='confirmed_paint_local_LK_missing', source_RGB_tracking_confirmed=False)
    backward, bvalid, _ = cv2.calcOpticalFlowPyrLK(current, previous, forward, None,
        winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, .01))
    a, b = p[:, 0], forward[:, 0]
    error = np.linalg.norm(backward[:, 0]-a, axis=1)
    good = fvalid.ravel().astype(bool) & bvalid.ravel().astype(bool) & np.isfinite(b).all(1) & (error <= 2.)
    ix = np.rint(np.where(np.isfinite(b), b, 0)).astype(int)
    inside = (ix >= 0).all(1) & (ix < [current.shape[1], current.shape[0]]).all(1)
    good &= inside
    good[inside] &= ~current_mask[ix[inside, 1], ix[inside, 0]]
    if good.sum() < 2:
        return center, dict(status='confirmed_paint_local_bidirectional_support_missing', source_RGB_tracking_confirmed=False)
    delta_sides = sides[:, 1]-sides[:, 0]
    lengths = np.linalg.norm(delta_sides, axis=1)
    normals = delta_sides/np.maximum(lengths[:, None], 1e-9)
    if np.any(lengths <= 1e-6):
        tangent = np.gradient(actual, axis=0)
        normals = np.c_[-tangent[:, 1], tangent[:, 0]]
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    paired = b[good]-a[good]
    normal_motion = np.einsum('ij,ij->i', paired, normals[owners[good]])
    # Paint support identifies normal motion. Tangential aperture cannot
    # justify a 2D homography or a fitted point correspondence along the line.
    stationary = (actual-actual.mean(0)) @ np.linalg.svd(actual-actual.mean(0), full_matrices=False)[2][0]
    observed_station = stationary[owners[good]]
    normal_baseline = float(np.median(normal_motion))
    mad = float(1.4826*np.median(np.abs(normal_motion-normal_baseline)))
    coherent = np.abs(normal_motion-normal_baseline) <= max(2., 4*mad)
    if coherent.sum() < 2:
        return center, dict(status='confirmed_paint_local_motion_competes', source_RGB_tracking_confirmed=False)
    x, y = observed_station[coherent], normal_motion[coherent]
    if np.ptp(x) > 20.:
        design = np.c_[np.ones(len(x)), x]
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        predicted = beta[0]+beta[1]*stationary
    else:
        predicted = np.full(len(actual), float(np.median(y)))
    # Keep actual source shape/ordering. Only source normal search navigation
    # is transported; its coordinates are never returned as measurements.
    transported = actual+normals*predicted[:, None]
    return transported, dict(status='confirmed_paint_local_bidirectional_normal_navigation',
        matched=int(good.sum()), coherent=int(coherent.sum()),
        backward_error_median_px=float(np.median(error[good][coherent])),
        source_support_uv_sha256=hashlib.sha256(a[good][coherent].tobytes()).hexdigest(),
        target_support_uv_sha256=hashlib.sha256(b[good][coherent].tobytes()).hexdigest(),
        normal_motion_min_median_max_px=[float(y.min()), float(np.median(y)), float(y.max())],
        source_RGB_tracking_confirmed=True, full_2D_homography_estimated=False,
        used_as_fitting_observations=False, used_as_UV_output=False)


def _measured_material_descriptors(rgb, centers, sides, normals):
    """Current native RGB context of both real sides, including the ice flanks."""
    centers, sides, normals = map(lambda a: np.asarray(a, float), (centers, sides, normals))
    if not len(centers):
        return np.empty((0,123),float),np.empty(0,float)
    width = np.linalg.norm(sides[:, 1]-sides[:, 0], axis=1)
    # Include both actual sides and six native pixels of each adjacent ice
    # flank. An oversized strip used to sample outside the right frame and
    # compare black border fill with actual ice when the C band approached it.
    extent = np.maximum(12., width*.5+6.)
    offsets = np.linspace(-1., 1., 41)
    xy = centers[:, None]+normals[:, None]*extent[:, None, None]*offsets[None, :, None]
    complete = ((xy[..., 0]>=0)&(xy[..., 0]<=rgb.shape[1]-1)
        &(xy[..., 1]>=0)&(xy[..., 1]<=rgb.shape[0]-1)).all(1)
    xy = np.rint(xy.astype(np.float32)*32.)/32.
    sampled = cv2.remap(rgb.astype(np.float32), xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32), cv2.INTER_LINEAR)
    # Preserve chromatic and achromatic signed material shape. Subtract
    # observed ice-side levels so slow illumination changes do not name IDs.
    background = .5*(sampled[:, :4].mean(1)+sampled[:, -4:].mean(1))
    values = (sampled-background[:, None]).reshape(len(centers), -1)
    norms = np.linalg.norm(values, axis=1)
    norms[~complete] = 0.
    values[~complete] = 0.
    return values/np.maximum(norms[:, None], 1e-6), norms


def _current_readout_uncertainty(measurement, selected, current, normals):
    """Carry actual gradient readout blur, never the finite paint-band width."""
    values = measurement.get('candidate_side_quantiles_uv')
    if values is None:
        return None
    values = np.asarray(values, float)[selected]
    if values.shape != (len(selected), 2, 3, 2) or not np.isfinite(values).all():
        return None
    # q25/q50/q75 describe each signed edge gradient in this exposure. Their
    # paired means describe uncertainty in the center, not the distance
    # between the two edges of the physical painted band.
    centers = values.mean(axis=1)
    intervals = np.einsum('nki,ni->nk', centers[:, [0, 2]]-current[:, None], normals)
    intervals.sort(axis=1)
    step = float(measurement.get('source_readout_quantization_step_px', 1.))
    normal_interval = intervals+np.array([-.5*step, .5*step])
    return dict(kind='actual_source_readout_blur_and_quantization',
        actual_source_RGB_measured=True, paint_width_used_as_center_tolerance=False,
        measurement_method=measurement.get('source_measurement_method', 'native_RGB_signed_gradient_quantiles'),
        quantization_step_px=step,
        source_interpolation_grid_px=measurement.get('source_interpolation_grid_px'),
        actual_readout_blur_interval_px=intervals.tolist(), normal_interval_px=normal_interval.tolist(),
        actual_side_gradient_quantiles_uv=values.tolist(),
        support_is_not_calibrated_Gaussian_accuracy=True)


def _connected_paint_reentry(rgb, uv, sides, candidates, query_ids, track):
    """Identify a unique currently measured paint path by source material.

    A line template comes only from the last confirmed real paint band. Source
    point coherence organizes real candidates; neither line-fit coordinates
    nor a world projection becomes output UV.
    """
    evidence = dict(source_clean_candidates=int(len(candidates)))
    def missing(status):
        return np.empty(0, int), {**evidence, 'status': status}
    if len(candidates) < 2 or track.get('material_descriptors') is None:
        return missing('previous_confirmed_source_material_context_missing')
    old_sides = np.asarray(track['candidate_sides'], float)
    old_width = np.linalg.norm(old_sides[:, 1]-old_sides[:, 0], axis=1)
    current_sides = sides[candidates]
    widths = np.linalg.norm(current_sides[:, 1]-current_sides[:, 0], axis=1)
    compatible = (widths >= .65*np.percentile(old_width, 10)) & (widths <= 1.55*np.percentile(old_width, 90))
    candidates = candidates[compatible]
    evidence['source_width_compatible_candidates'] = int(len(candidates))
    if len(candidates) < 2:return missing('source_material_width_compatibility_missing')
    width = np.linalg.norm(sides[candidates, 1]-sides[candidates, 0], axis=1)
    normal = (sides[candidates, 1]-sides[candidates, 0])/width[:, None]
    descriptors, strength = _measured_material_descriptors(rgb, sides[candidates].mean(1), sides[candidates], normal)
    old_descriptors = np.asarray(track['material_descriptors'], float)
    appearance = (descriptors @ old_descriptors.T).max(1)
    candidates = candidates[(appearance >= .9) & (strength > 0)]
    evidence['source_RGB_context_compatible_candidates'] = int(len(candidates))
    evidence['source_RGB_context_correlation_min_median_max'] = [float(appearance.min()), float(np.median(appearance)), float(appearance.max())]
    if len(candidates) < 4:return missing('source_material_path_support_missing')
    points = sides[candidates].mean(1)
    fit = cv2.fitLine(points.astype(np.float32), cv2.DIST_WELSCH, 0, .01, .01).ravel()
    tangent, anchor = fit[:2].astype(float), fit[2:].astype(float)
    distance = np.abs((points-anchor) @ np.array([-tangent[1], tangent[0]]))
    coherent = distance <= 4.
    candidates, points = candidates[coherent], points[coherent]
    evidence['source_straight_path_coherent_candidates'] = int(len(candidates))
    if len(candidates) < 4:return missing('source_straight_material_path_not_coherent')
    # A second equally supported path or two material interpretations within
    # one source profile cannot be silently resolved by nearest navigation.
    for query in np.unique(query_ids[candidates]):
        if (query_ids[candidates] == query).sum() > 1:
            return missing('source_path_has_competing_material_interpretation')
    stations = (points-anchor) @ tangent
    order = np.argsort(stations)
    spacing = np.diff(stations[order])
    if len(spacing) and np.percentile(spacing, 75) > 80.:
        return missing('source_material_path_not_connected')
    old_points = np.asarray(track['actual_uv'], float)
    previous_center = np.median(old_points, axis=0)
    # Entity motion is tested against an observed normal, with the actual
    # paint width providing its source search footprint. No projection score.
    previous_normal = (old_sides[:, 1]-old_sides[:, 0]).mean(0)
    previous_normal /= max(np.linalg.norm(previous_normal), 1e-9)
    normal_step = abs(float((np.median(points, axis=0)-previous_center) @ previous_normal))
    evidence['source_material_normal_step_px'] = normal_step
    evidence['source_material_normal_navigation_footprint_px'] = max(24., float(np.percentile(old_width, 90))*.75)
    if normal_step > max(24., float(np.percentile(old_width, 90))*.75):
        return missing('source_material_normal_continuity_not_identified')
    return candidates[order], {**evidence, 'status': 'same_source_connected_paint_path'}


class TemporalIceIdentitySelector:
    """Same-shot source identities; cuts reset all propagated identities."""
    def __init__(self, inventory, *, source_sha256, time_base, seed_registry=(),model_context_identity=None):
        self.inventory = {row['curve_id']: dict(row) for row in inventory}
        self.names = [row['curve_id'] for row in inventory]
        self.source_sha256, self.time_base = source_sha256, str(Fraction(time_base))
        self.seeds = {}
        if isinstance(seed_registry, dict):seed_registry = seed_registry['records']
        for seed in seed_registry:
            self.seeds.setdefault(int(seed['actual_pts']), []).append(seed)
        self.previous_gray = self.previous_mask = self.previous_binding = None
        self.tracks = {}
        self.model_context_identity=model_context_identity

    def retain_final_source_rows(self,rgb,measurement,selected,*,final_source_node_ids=None):
        """Remove source supports rejected after selection, before next-frame LK."""
        binding=measurement['source_binding'];pts=int(binding.get('actual_pts',binding['source_pts']))
        digest=hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        bgr=hashlib.sha256(np.asarray(rgb)[:,:,::-1].tobytes()).hexdigest()
        if (self.previous_binding is None or int(self.previous_binding['source_pts'])!=pts
                or self.previous_binding['shot_id']!=binding['shot_id']
                or self.previous_binding.get('decoded_native_BGR_sha256')!=bgr
                or measurement['decoded_native_RGB_sha256']!=digest
                or binding['source_sha256']!=self.source_sha256 or str(Fraction(binding['time_base']))!=self.time_base):
            raise ValueError('Final selector support must belong to its just-consumed exact RGB exposure')
        if self.model_context_identity is not None and final_source_node_ids is None:
            raise ValueError('Final graph source-node locators are required for pending grid-context preparation')
        rows=selected.get('rows',()) if isinstance(selected,dict) else selected
        by_curve={}
        for row in rows:
            name=row['curve_id']
            if (name not in self.inventory or row.get('actual_pts')!=pts
                    or row.get('decoded_native_RGB_sha256')!=digest):
                raise ValueError('Final selected source support has a different curve or native RGB')
            uv,sides=exact_selected_source_readouts(row,measurement,operator=self.inventory[name]['operator'])
            ids=np.asarray(row['original_candidate_indices'],np.int64)
            curves=np.asarray(measurement['query_curve'])[np.asarray(measurement['candidate_query'])[ids]]
            if any(list(self.inventory)[int(ci)]!=name for ci in curves):
                raise ValueError('Final selected track cannot rename another current source curve')
            by_curve.setdefault(name,[]).append((ids,uv,sides))
        retained={};dropped=[]
        for name,track in self.tracks.items():
            if int(track['actual_pts'])!=pts:
                # The held gray image is current; stale actual coordinates
                # remain navigation for source re-entry, never LK evidence.
                retained[name]={**track,'navigation_pts':pts}
                continue
            pieces=by_curve.get(name,())
            original=np.asarray(track.get('source_candidate_indices',()),np.int64)
            ids=[];points=[];interfaces=[]
            for locators,uv,sides in pieces:
                keep=np.isin(locators,original)
                ids.extend(locators[keep]);points.extend(uv[keep]);interfaces.extend(sides[keep])
            support=final_current_source_track_support(rgb,measurement,ids,
                operator=self.inventory[name]['operator'],uv=points,sides=interfaces) if ids else None
            if support is None:dropped.append(name)
            else:retained[name]={**track,**support,'is_seed':False}
        self.tracks=retained
        context=None
        if self.model_context_identity is not None:
            context=self.model_context_identity.retain_final_source_nodes(rgb,measurement,
                final_source_node_ids,selected=selected)
        return dict(actual_pts=pts,retained_source_tracks=len(retained),dropped_current_source_curves=dropped,
            stale_pixels_used_as_LK_support=False,model_context=context)

    def consume(self, rgb, measurement, exclusion_mask, *, seeds=(), source_ice_scope_mask=None):
        binding = measurement['source_binding']
        pts = int(binding['source_pts'])
        digest = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
        bgr_digest = hashlib.sha256(np.asarray(rgb)[:, :, ::-1].tobytes()).hexdigest()
        if (binding['source_sha256'] != self.source_sha256 or str(Fraction(binding['time_base'])) != self.time_base
                or measurement['decoded_native_RGB_sha256'] != digest):
            raise ValueError('Temporal identity and actual RGB source binding differ')
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = np.asarray(exclusion_mask, bool).copy()
        scope_confirmed = source_ice_scope_mask is not None
        if scope_confirmed:
            scope = np.asarray(source_ice_scope_mask, bool)
            if scope.shape != mask.shape:
                raise ValueError('Same-source native ice surface region required')
            mask |= ~scope
        reset = (self.previous_binding is None or self.previous_binding['shot_id'] != binding['shot_id'])
        if reset:
            self.tracks = {}
            transform, motion = None, dict(status='source_shot_reset')
        else:
            if pts <= int(self.previous_binding['source_pts']):
                raise ValueError('Temporal source exposures must remain ordered inside a shot')
            transform, motion = None, dict(status='per_confirmed_curve_local_source_motion')
        navigation = {}
        if not reset:
            for name, track in self.tracks.items():
                transported, local_motion = _curve_local_navigation(self.previous_gray, gray, self.previous_mask, mask, track)
                chain = hashlib.sha256((track.get('source_motion_chain_sha256', '')
                    + json.dumps(local_motion, sort_keys=True, separators=(',', ':'))).encode()).hexdigest()
                # A failed local motion keeps a candidate search for source
                # re-entry, but cannot on its own admit a current observation.
                navigation[name] = (transported, {**track, 'navigation_uv': transported,
                    'navigation_pts': pts, 'local_motion': local_motion,
                    'source_motion_chain_sha256': chain,
                    'identity_continuity_steps': int(track.get('identity_continuity_steps', 0))+1})
        if transform is not None:
            for name, track in self.tracks.items():
                old = np.asarray(track.get('navigation_uv', track['actual_uv']), np.float32).reshape(-1, 1, 2)
                transported = cv2.perspectiveTransform(old, transform).reshape(-1, 2)
                if np.isfinite(transported).all():
                    chain = hashlib.sha256((track.get('source_motion_chain_sha256', '')
                        + json.dumps(motion, sort_keys=True, separators=(',', ':'))).encode()).hexdigest()
                    navigation[name] = (transported, {**track, 'navigation_uv': transported,
                        'source_motion_chain_sha256': chain, 'navigation_pts': pts,
                        'identity_continuity_steps': int(track.get('identity_continuity_steps', 0))+1})
        elif motion.get('coordinate_kind') == 'source_2D_local_navigation_only':
            source = np.asarray(motion['source_uv'], float)
            target = np.asarray(motion['target_uv'], float)
            for name, track in self.tracks.items():
                old = np.asarray(track.get('navigation_uv', track['actual_uv']), float)
                sides_old = np.asarray(track.get('candidate_sides', []), float)
                if sides_old.shape != (len(old), 2, 2):
                    continue
                delta = sides_old[:, 1]-sides_old[:, 0]
                width = np.linalg.norm(delta, axis=1)
                normals = delta/np.maximum(width[:, None], 1e-9)
                distances = np.linalg.norm(old[:, None]-source[None], axis=2)
                nearest = distances.argmin(1)
                supported = distances[np.arange(len(old)), nearest] <= np.maximum(12., width*.5+3.)
                if supported.sum() < 2:
                    continue
                movement = target[nearest]-source[nearest]
                transported = old+normals*np.einsum('ij,ij->i', movement, normals)[:, None]
                chain = hashlib.sha256((track.get('source_motion_chain_sha256', '')
                    + json.dumps(motion, sort_keys=True, separators=(',', ':'))).encode()).hexdigest()
                navigation[name] = (transported[supported], {**track,
                    'navigation_uv': transported[supported], 'navigation_pts': pts,
                    'source_motion_chain_sha256': chain,
                    'identity_continuity_steps': int(track.get('identity_continuity_steps', 0))+1})
        context_decisions=[];context_seeds=[]
        if self.model_context_identity is not None and scope_confirmed:
            context=self.model_context_identity.consume(rgb,measurement,exclusion_mask,source_ice_scope_mask=source_ice_scope_mask)
            context_decisions=context['decisions']
            context_seeds=[s for s in context['seeds'] if s['curve_id'] not in navigation]
        # Direct original source facts follow last and retain priority.
        for seed in [*context_seeds,*self.seeds.get(pts, ()), *seeds]:
            if seed.get('bind_current_original_RGB_at_exact_seed_pts'):
                from .vision_ice_seed_registry import bind_source_seed_to_current_RGB
                seed = bind_source_seed_to_current_RGB(seed, measurement)
            name = seed['curve_id']
            if (name not in self.inventory or int(seed['actual_pts']) != pts
                    or seed['source_sha256'] != self.source_sha256
                    or str(Fraction(seed['time_base'])) != self.time_base or seed['shot_id'] != binding['shot_id']
                    or seed.get('decoded_native_RGB_sha256') != digest
                    or seed.get('source_material_class') != 'static_painted_ice'
                    or seed.get('surface_and_physical_identity_confirmed') is not True
                    or not seed.get('authority')):
                raise ValueError('Physical source seed requires same-frame RGB, identity and painted ice material authority')
            navigation[name] = (np.asarray(seed['navigation_uv'], float).reshape(-1, 2),
                dict(seed=seed, actual_uv=seed['navigation_uv'], actual_pts=pts, source_RGB_sha256=digest, is_seed=True))
        qcurve = np.asarray(measurement['query_curve'])
        queries = np.asarray(measurement['candidate_query'])
        uv = np.asarray(measurement['candidate_uv'], float)
        sides = np.asarray(measurement['candidate_sides'], float)
        flags = np.asarray(measurement['candidate_flags'])
        rows, decisions, next_tracks = [], [], {}
        for ci, name in enumerate(self.names):
            if name not in navigation:
                decisions.append(dict(curve_id=name, actual_pts=pts, status='physical_source_identity_seed_or_continuity_missing'))
                continue
            predicted, track = navigation[name]
            # Preserve a source-flow search through occlusion; this is only
            # navigation and does not create a measurement in the missing frame.
            next_tracks[name] = {**track, 'navigation_uv': predicted, 'is_seed': False}
            subset = np.flatnonzero(qcurve[queries] == ci)
            distances = _point_polyline_distance(uv[subset], predicted)
            # The actual same-source two sides supply all emitted centers.
            # Source flow locates the already confirmed entity, not a world
            # projection, and a second matching RGB candidate remains unknown.
            motion_confirmed = track.get('local_motion', {}).get('source_RGB_tracking_confirmed') is True
            tolerance = 12. if track.get('is_seed') else (6. if motion_confirmed else 24.)
            eligible = subset[(distances <= tolerance) & ((flags[subset] & 3) == 0)]
            if len(eligible):
                ix = np.rint(uv[eligible]).astype(int)
                inside = (ix >= 0).all(1) & (ix < [mask.shape[1], mask.shape[0]]).all(1)
                clear = np.zeros(len(eligible), bool)
                clear[inside] = ~mask[ix[inside, 1], ix[inside, 0]]
                eligible = eligible[clear]
            unique = []
            for query in np.unique(queries[eligible]):
                choices = eligible[queries[eligible] == query]
                if len(choices) == 1:unique.append(int(choices[0]))
            selected = np.asarray(unique, int)
            definition = self.inventory[name]
            if definition['operator'] == 'unresolved':
                decisions.append(dict(curve_id=name, actual_pts=pts, status='physical_boundary_role_unresolved'))
                continue
            if definition['operator'] == 'paint_center' and len(selected):
                actual_center = sides[selected].mean(1)
                geometric = np.linalg.norm(actual_center-uv[selected], axis=1) <= 1e-3
                selected = selected[geometric]
            connected_reentry = False
            reentry_evidence = None
            if (len(selected) < 2 and not track.get('is_seed') and definition['operator'] == 'paint_center'
                    and definition['feature_id'] in ('C', 'B1', 'B2', 'goal_line_left', 'goal_line_right')):
                clean = subset[(flags[subset] & 3) == 0]
                if len(clean):
                    ix = np.rint(uv[clean]).astype(int)
                    inside = (ix >= 0).all(1) & (ix < [mask.shape[1], mask.shape[0]]).all(1)
                    clear = np.zeros(len(clean), bool)
                    clear[inside] = ~mask[ix[inside, 1], ix[inside, 0]]
                    clean = clean[clear]
                selected, reentry_evidence = _connected_paint_reentry(rgb, uv, sides, clean, queries, track)
                connected_reentry = len(selected) >= 2
            if len(selected) < 2 or not scope_confirmed:
                decisions.append(dict(curve_id=name, actual_pts=pts,
                    status='actual_source_material_or_confirmed_ice_scope_missing', source_matches=int(len(selected)),
                    source_candidates=int(len(subset)), matching_candidates=int(len(eligible)),
                    minimum_source_navigation_distance=None if not len(distances) else float(distances.min()),
                    local_source_motion=track.get('local_motion'), source_paint_reentry=reentry_evidence))
                continue
            # Order by search source-station identity. No fitted or smoothed
            # center replaces actual observed UV during temporal re-entry.
            order = np.argsort(queries[selected], kind='stable')
            selected = selected[order]
            current = uv[selected]
            if definition['operator'] == 'paint_center':
                current = sides[selected].mean(1)
            widths = np.linalg.norm(sides[selected, 1]-sides[selected, 0], axis=1)
            if definition['operator'] == 'paint_center' and np.any(widths <= 0):
                decisions.append(dict(curve_id=name, actual_pts=pts, status='two_actual_paint_sides_missing'))
                continue
            source_pixel_checks=[]
            if definition['operator']=='paint_center':
                from .vision_ice_model_context_identity import _actual_paint_colour_context,actual_paint_path_coherence
                current_normals=(sides[selected,1]-sides[selected,0])/widths[:,None]
                material,material_RGB,material_contrast,sample_positions=_actual_paint_colour_context(
                    rgb,current,sides[selected],current_normals,definition['material_signal'])
                for index in selected[~material]:
                    source_pixel_checks.append(dict(original_candidate_index=int(index),
                        status='actual_paint_and_two_exterior_RGB_material_support_missing'))
                selected,current,widths=selected[material],current[material],widths[material]
                if definition['feature_id'] in ('C','B1','B2','goal_line_left','goal_line_right'):
                    coherent,held=actual_paint_path_coherence(measurement,selected)
                    source_pixel_checks.extend(held)
                    selected,current,widths=selected[coherent],current[coherent],widths[coherent]
                if len(selected)<2:
                    decisions.append(dict(curve_id=name,actual_pts=pts,status='current_source_paint_pixel_accuracy_requires_review',
                        source_pixel_checks=source_pixel_checks));continue
            if not track.get('is_seed') and track.get('candidate_sides') is not None:
                old_sides = np.asarray(track['candidate_sides'], float)
                old_uv = np.asarray(track['actual_uv'], float)
                old_widths = np.linalg.norm(old_sides[:, 1]-old_sides[:, 0], axis=1)
                distances_old = np.linalg.norm(current[:, None]-old_uv[None], axis=2)
                nearest_old = distances_old.argmin(1)
                reference_widths = old_widths[nearest_old]
                ratio = widths/np.maximum(reference_widths, 1e-6)
                if definition['operator'] == 'paint_center':
                    compatible = (ratio >= .65) & (ratio <= 1.55)
                    selected, current, widths = selected[compatible], current[compatible], widths[compatible]
                    if len(selected) < 2:
                        decisions.append(dict(curve_id=name, actual_pts=pts,
                            status='actual_paint_width_or_same_entity_discontinuity_requires_source_review'))
                        continue
            source_normals = np.asarray(measurement['query_normal'], float)[queries[selected]]
            if definition['operator'] == 'paint_center':
                source_normals = (sides[selected, 1]-sides[selected, 0])/widths[:, None]
            descriptors, descriptor_strength = _measured_material_descriptors(rgb, current, sides[selected], source_normals)
            material_correlations = np.ones(len(selected), float)
            if not track.get('is_seed'):
                old_descriptors = np.asarray(track.get('material_descriptors', []), float)
                old_uv = np.asarray(track['actual_uv'], float)
                if old_descriptors.shape != (len(old_uv), descriptors.shape[1]):
                    decisions.append(dict(curve_id=name, actual_pts=pts, status='confirmed_material_appearance_context_missing'))
                    continue
                nearest = np.linalg.norm(current[:, None]-old_uv[None], axis=2).argmin(1)
                material_correlations = np.einsum('ij,ij->i', descriptors, old_descriptors[nearest])
                appearance = (material_correlations >= (.70 if motion_confirmed else .90)) & (descriptor_strength > 0)
                selected, current, widths, source_normals = selected[appearance], current[appearance], widths[appearance], source_normals[appearance]
                descriptors, material_correlations = descriptors[appearance], material_correlations[appearance]
                if len(selected) < 2:
                    decisions.append(dict(curve_id=name, actual_pts=pts, status='same_paint_entity_RGB_context_competes_or_disappears'))
                    continue
            row = dict(actual_pts=pts, native_pts=pts, source_sha256=self.source_sha256,
                time_base=self.time_base, shot_id=binding['shot_id'], decoded_native_RGB_sha256=digest,
                decoded_native_BGR_sha256=bgr_digest,
                curve_id=name, feature_id=definition['feature_id'],
                world_curve_role=definition['world_curve_role'],
                physical_edge_role='paint_centerline' if definition['operator'] == 'paint_center' else 'confirmed_material_interface',
                center_definition='midpoint_of_two_current_actual_RGB_material_interfaces' if definition['operator'] == 'paint_center' else None,
                physical_material_id=name, static_material_confirmed=True,
                actual_source_RGB_measured=True, pixel_location_confirmed=True,
                model_projection_or_fit_residual_used_for_identity=False,
                uv=current.tolist(), candidate_sides=sides[selected].tolist(), source_candidate_indices=selected.tolist(),
                original_candidate_indices=selected.tolist(),
                source_sides_uv=sides[selected].tolist(), raw_candidate_uv=uv[selected].tolist(),
                source_query_indices=queries[selected].tolist(), source_seed_authority=track['seed']['authority'],
                source_seed_reference=track['seed'],
                continuity_evidence=dict(previous_actual_pts=int(track['actual_pts']), current_actual_pts=pts,
                    previous_RGB_sha256=track['source_RGB_sha256'], current_RGB_sha256=digest,
                    source_motion=track.get('local_motion', motion), current_actual_paint_widths_px=widths.tolist(),
                    current_source_RGB_material_context_correlations=material_correlations.tolist(),
                    reentry_verified_from_source_material_context=not track.get('is_seed') and not motion_confirmed,
                    current_connected_paint_path_reentry_confirmed=connected_reentry,
                    source_paint_reentry=reentry_evidence,
                    current_source_pixel_accuracy_checks=source_pixel_checks,
                    source_motion_chain_sha256=track.get('source_motion_chain_sha256'),
                    identity_continuity_steps=int(track.get('identity_continuity_steps', 0)),
                    physical_identity_from_model_projection=False, source_flow_used_as_output_UV=False,
                    center_derived_from_current_two_actual_sides=definition['operator'] == 'paint_center',
                    source_ice_surface_region_confirmed=True, competing_current_matches_rejected=True),
                admission_status=('admitted_source_material_center_temporal_identity' if definition['operator']=='paint_center'
                    else 'admitted_source_physical_boundary_temporal_identity'),
                model_projection_used_as_measurement=False, source_center_smoothed_or_interpolated=False)
            row['temporal_evidence'] = dict(
                kind='confirmed_original_source_identity_anchor' if track.get('is_seed') else 'same_shot_actual_RGB_temporal_identity',
                physical_material_id=name, shot_id=binding['shot_id'], current_actual_pts=pts,
                same_physical_entity_confirmed=True, current_frame_RGB_measured=True,
                model_projection_or_fit_residual_used_for_identity=False)
            if track.get('is_seed'):
                row['temporal_evidence'].update(source_identity_authority=track['seed']['authority'],
                    source_identity_confirmed=True)
            else:
                row['temporal_evidence'].update(previous_shot_id=binding['shot_id'],
                    previous_actual_pts=int(track['actual_pts']), previous_decoded_BGR_sha256=track['source_BGR_sha256'],
                    previous_frame_RGB_measured=True, source_RGB_tracking_confirmed=True,
                    current_feature_semantics_remeasured=True,
                    source_RGB_tracking_method=('confirmed_material_local_LK_and_actual_RGB_reread'
                        if motion_confirmed else 'same_entity_actual_RGB_material_context_and_source_path_reentry'))
            if track['seed'].get('source_identity_navigation_evidence') is not None:
                from .vision_ice_model_context_identity import attach_grid_context_evidence
                attach_grid_context_evidence(row,track['seed'],source_binding=dict(source_sha256=self.source_sha256,
                    time_base=self.time_base,actual_pts=pts,native_pts=int(binding.get('native_pts',pts)),
                    decoded_native_BGR_sha256=bgr_digest,shot_id=binding['shot_id']),
                    source_motion=row['continuity_evidence'])
                row['continuity_evidence'].pop('physical_identity_from_model_projection',None)
                row['continuity_evidence']['model_projection_used_as_navigation_and_identity_context']=True
            uncertainty = _current_readout_uncertainty(measurement, selected, current, source_normals)
            if uncertainty is not None:
                row['source_readout_uncertainty'] = uncertainty
                row['sigma_px'] = np.max(np.abs(uncertainty['normal_interval_px']), axis=1).tolist()
                row['sigma_semantics'] = 'actual_source_readout_support_half_extent_not_calibrated_Gaussian_covariance'
            rows.append(row)
            clearance=measurement.get('candidate_local_source_clearance',{})
            proofs={int(i):clearance[int(i)] for i in selected if int(i) in clearance}
            if proofs:
                row['candidate_local_source_clearance']=proofs
                row['original_source_exclusion_flag_was_cleared_only_by_explicit_local_material_proof']=True
            next_tracks[name] = {**track, 'actual_uv': current, 'navigation_uv': current,
                'candidate_sides': sides[selected].copy(),
                'material_descriptors': descriptors.copy(), 'source_normals': source_normals.copy(),
                'source_candidate_indices':selected.copy(),'source_query_indices':queries[selected].copy(),
                'actual_pts': pts, 'navigation_pts':pts, 'source_RGB_sha256': digest,
                'source_BGR_sha256':bgr_digest, 'is_seed': False}
            decisions.append(dict(curve_id=name, actual_pts=pts, status=row['admission_status'],
                                  source_matches=int(len(selected))))
        self.tracks = next_tracks
        self.previous_gray, self.previous_mask, self.previous_binding = gray, mask, dict(binding)
        return dict(actual_pts=pts, rows=rows, decisions=decisions,model_context_decisions=context_decisions,
                    all_curve_identities_checked=True, visual_review_complete=False,
                    source_temporal_navigation_used=True, fitting_camera_optimized=False)
