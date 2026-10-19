"""Bind whole-rink hypotheses to source frames using fresh ice observations.

The frozen whole-rink geometry is never fitted here. RootSIFT supplies only an
initial camera guess; acceptance requires mask-isolated LK, forward/backward
consistency and raw ice-patch correlation against a fixed model reference.
Observed local edges retain their model lineage beyond an anchor's age.
Radial reference coordinates remain an explicit nonlinear projection chain.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw

from .continuous_registration import ContinuousConfig, ContinuousRegistrar
from .foreground_rows import row_foreground_exclusion
from .geometry import project
from .io import dump_json, load_json, sha256
from .local_registration import RegistrationConfig, ice_mask, _fit
from .structure_render import (_scene, _distort, _distortion_parameters, _homogeneous,
                               _visible_segment, _stroke, _font, STYLES)
from .workspace import load_index, load_shots, source_path

ALGORITHM_VERSION = 'whole-rink-observed-continuity-8'
DRAWABLE_STATES = {'candidate', 'predicted', 'model_reobserved', 'image_tracked', 'motion_predicted'}


@dataclass(frozen=True)
class WholeRinkConfig:
    sample_width: int = 960
    sample_height: int = 540
    reobserve_every: int = 10
    max_prediction_s: float = .6
    near_reference_s: float = .8
    max_references: int = 8
    anchor_conflict_p95_px_at_960: float = 12.
    continuity_mode: str = 'observed'
    max_engineering_error_px_at_960: float = 12.

    def __post_init__(self):
        if not (self.sample_width >= 240 and self.sample_height >= 120 and
                self.sample_width % 2 == 0 and self.sample_height % 2 == 0):
            raise ValueError('Require an even processing size of at least 240x120')
        if (not 0 < self.max_prediction_s <= .6 or self.reobserve_every < 1 or self.max_references < 1
                or self.anchor_conflict_p95_px_at_960 <= 0):
            raise ValueError('Blind prediction is bounded to 0.6 seconds; counts must be positive')
        if self.continuity_mode not in {'legacy', 'observed'} or self.max_engineering_error_px_at_960 <= 0:
            raise ValueError('Invalid continuity mode or engineering error budget')


def _polygon_mask(shape, polygons):
    result = np.zeros(shape, np.uint8)
    for poly in polygons:
        p = np.asarray(poly, float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError('Expected finite image polygon')
        cv2.fillPoly(result, [np.rint(p).astype(np.int32)], 255)
    return result


def _source_timestamp(record):
    """Keep canonical source identity and UI aliases from diverging silently."""
    pts = record['pts']
    if isinstance(pts, bool) or not isinstance(pts, (int, np.integer)):
        raise ValueError('Source PTS must be integer ticks, never rounded')
    base = Fraction(record['time_base'])
    if base <= 0:
        raise ValueError('Source PTS time_base must be positive')
    if 'source_pts' in record and record['source_pts'] != pts:
        raise ValueError('source_pts alias disagrees with canonical source PTS')
    if 'source_time_base' in record and Fraction(record['source_time_base']) != base:
        raise ValueError('source_time_base alias disagrees with canonical source PTS time_base')
    time = float(int(pts)*base)
    if 'source_time_s' in record:
        declared = record['source_time_s']
        if not isinstance(declared, (float, int)) or not np.isfinite(declared) or abs(time-declared) > 1e-7:
            raise ValueError('Source time must agree with exact PTS')
    digest = record['source_sha256']
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdefABCDEF' for c in digest)):
        raise ValueError('Source identity requires a complete SHA-256')
    return time


def _polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, 1., True).reshape(-1, 2).tolist()
            for c in contours if cv2.contourArea(c) >= 4]


def frame_masks(image, explicit_ice=None, explicit_people=()):
    base = ice_mask(image)
    domain = base['ice'] if explicit_ice is None else explicit_ice
    foreground = row_foreground_exclusion(image, domain)
    person = _polygon_mask(domain.shape, explicit_people)
    excluded = cv2.bitwise_or(foreground['foreground'], person)
    usable = cv2.bitwise_and(domain, cv2.bitwise_not(excluded))
    return {**base, 'ice': domain, 'foreground': excluded, 'usable': usable,
            'ice_fraction': float(np.mean(domain > 0)), 'usable_fraction': float(np.mean(usable > 0)),
            'foreground_method': foreground['method'], 'precise_person_segmentation': False}


def _rootsift(image, mask):
    """An ice-only proposal descriptor; never sufficient for a pose output."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    isolated = np.where(mask > 0, gray, 0).astype(np.uint8)
    safe = cv2.erode(mask, np.ones((17, 17), np.uint8))
    keys, desc = cv2.SIFT_create(nfeatures=2500, contrastThreshold=.005, edgeThreshold=14).detectAndCompute(isolated, safe)
    if desc is None:
        return np.empty((0, 2), np.float32), None
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    selected = [i for i, k in enumerate(keys)
                if distance[min(mask.shape[0]-1, round(k.pt[1])), min(mask.shape[1]-1, round(k.pt[0]))] > 3*k.size]
    if not selected:
        return np.empty((0, 2), np.float32), None
    desc = np.sqrt(desc[selected] / np.maximum(desc[selected].sum(axis=1, keepdims=True), 1e-9))
    return np.asarray([keys[i].pt for i in selected], np.float32), desc.astype(np.float32)


def _sift_proposal(reference, current):
    rp, rd = reference; cp, cd = current
    if rd is None or cd is None or min(len(rd), len(cd)) < 4:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward, reverse = matcher.knnMatch(rd, cd, k=2), matcher.knnMatch(cd, rd, k=2)
    backward = {m.queryIdx: m.trainIdx for pair in reverse if len(pair) == 2
                for m, n in [pair] if m.distance < .8*n.distance}
    good = [m for pair in forward if len(pair) == 2 for m, n in [pair]
            if m.distance < .8*n.distance and backward.get(m.trainIdx) == m.queryIdx]
    if len(good) < 6:
        return None
    x = np.asarray([rp[m.queryIdx] for m in good], np.float32)
    y = np.asarray([cp[m.trainIdx] for m in good], np.float32)
    cv2.setRNGSeed(20260920)
    affine, inliers = cv2.estimateAffinePartial2D(x, y, method=cv2.RANSAC,
        ransacReprojThreshold=4, maxIters=2500, confidence=.999)
    if affine is None or inliers is None or int(inliers.sum()) < 6:
        return None
    matrix = np.vstack([affine, [0., 0., 1.]])
    if not .15 < abs(np.linalg.det(matrix)) < 8:
        return None
    return {'H': matrix, 'kind': 'ice_rootsift_similarity_proposal_only',
            'proposal_matches': len(good), 'proposal_inliers': int(inliers.sum())}


def _registrar():
    return ContinuousRegistrar(ContinuousConfig(keyframe_interval_s=.2, max_keyframes=5,
        max_recovery_gap_s=.6, patch_radius_px=8, pyramid_levels=1,
        min_patch_correlation=.78), RegistrationConfig())


def _compact_fit(fit, *, points=True):
    excluded = {'inlier_flags'}
    if not points:
        excluded |= {'fit_current_xy', 'fit_reference_xy', 'fit_keyframe_xy'}
    return {k: v for k, v in fit.items() if k not in excluded}


def _similarity_fit(current_xy, target_xy, shape, config):
    """Four-DOF relative raw-image approximation, using actual ice pairs only.

    The fixed world-to-reference pose stays projective. Lower complexity is an
    engineering restriction on its short camera transfer, not physical camera
    calibration. All count, residual and distribution gates remain unchanged.
    """
    current_xy, target_xy = np.asarray(current_xy, np.float32), np.asarray(target_xy, np.float32)
    base = {'status': 'unavailable', 'matches': len(current_xy),
            'relative_geometry': 'relative_similarity_approximation', 'relative_degrees_of_freedom': 4}
    if len(current_xy) < config.min_tracks:
        return {**base, 'reason': 'insufficient_consistent_ice_tracks'}
    cv2.setRNGSeed(20260919)
    affine, flags = cv2.estimateAffinePartial2D(current_xy, target_xy, method=cv2.RANSAC,
        ransacReprojThreshold=config.ransac_threshold_px, maxIters=3000, confidence=.999, refineIters=10)
    if affine is None or flags is None or not np.isfinite(affine).all():
        return {**base, 'reason': 'similarity_fit_failed'}
    h = np.vstack([affine, [0., 0., 1.]])
    inliers = flags.ravel() > 0
    points = current_xy[inliers]
    base['train_inliers'] = len(points)
    if len(points) < config.min_inliers or np.mean(inliers) < .45:
        return {**base, 'reason': 'too_few_ice_plane_inliers'}
    height, width = shape
    hull = cv2.convexHull(points).reshape(-1, 2)
    area = float(cv2.contourArea(hull)/(height*width))
    span = np.ptp(points, axis=0)
    base.update(support_fraction=area, support_width_fraction=float(span[0]/width),
                support_height_fraction=float(span[1]/height))
    if (area < config.min_support_fraction or span[0] < width*config.min_support_width_fraction
            or span[1] < height*config.min_support_height_fraction):
        return {**base, 'reason': 'insufficient_spatial_support'}
    error = np.linalg.norm(project(h, points)-target_xy[inliers], axis=1)
    base.update(inlier_median_reference_px=float(np.median(error)),
                inlier_p95_reference_px=float(np.percentile(error, 95)))
    if not .12 < np.linalg.det(h[:2, :2]) < 8:
        return {**base, 'reason': 'implausible_frame_area_or_orientation'}
    if base['inlier_p95_reference_px'] > config.max_inlier_p95_px:
        return {**base, 'reason': 'large_fitting_residual'}
    return {**base, 'status': 'candidate', 'reason': 'independent_landmark_validation_pending',
        'H_reference_from_current': h.tolist(), 'support_polygon': hull.tolist(),
        'fit_current_xy': points.tolist(), 'fit_reference_xy': target_xy[inliers].tolist(),
        'inlier_flags': inliers.tolist()}


def _ice_edge(ref, image, masks, initial, *, prewarp=False):
    """Expose each old registrar acceptance gate, optionally in a scale-stable
    reference pixel image. The input mask is isolated before any warping.
    Thresholds and RANSAC support requirements match ContinuousRegistrar.
    """
    tracker, key = ref['tracker'], ref['key']
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    usable = masks['usable']
    if prewarp:
        height, width = gray.shape
        inverse = np.linalg.inv(initial)
        gray = cv2.warpPerspective(np.where(usable > 0, gray, 0).astype(np.uint8), inverse, (width, height))
        # Every bilinear contributor must belong to actual current-frame ice.
        support = cv2.warpPerspective((usable > 0).astype(np.float32), inverse, (width, height))
        usable = (support > .9999).astype(np.uint8)*255
        guess = key['points'].copy()
    else:
        guess = project(initial, key['points']).astype(np.float32)
    isolated, safe = tracker._safe(gray, usable)
    visible = tracker._inside(guess, safe)
    x, guess = key['points'][visible], guess[visible]
    stages = {'reference_ice_features': len(key['points']), 'initial_visible_in_safe_current_ice': len(x),
              'flow_both_directions': 0, 'forward_backward_pass': 0,
              'textured_patch_pass': 0, 'photometric_patch_pass': 0}
    base = {'stage_counts': stages, 'prewarped_to_reference_pixels': prewarp,
            'initial_current_from_reference': initial.tolist(), 'thresholds': {
                'forward_backward_px': tracker.config.forward_backward_px,
                'min_patch_correlation': tracker.config.min_patch_correlation,
                'min_patch_std': tracker.config.min_patch_std, 'ransac': asdict(tracker.fit_config)}}
    if len(x) < tracker.fit_config.min_tracks:
        return {**base, 'status': 'unavailable', 'reason': 'no_visible_keyframe_support'}
    radius = tracker.config.patch_radius_px
    params = dict(winSize=(radius*2+1, radius*2+1), maxLevel=tracker.config.pyramid_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .005), minEigThreshold=1e-6)
    y, sf, _ = cv2.calcOpticalFlowPyrLK(key['gray'], isolated, x[:, None], guess[:, None].copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params)
    z, sb, _ = cv2.calcOpticalFlowPyrLK(isolated, key['gray'], y, x[:, None].copy(),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params)
    y, z = y[:, 0], z[:, 0]
    ok = (sf[:, 0] > 0) & (sb[:, 0] > 0) & tracker._inside(y, safe)
    stages['flow_both_directions'] = int(ok.sum())
    ok &= np.linalg.norm(z-x, axis=1) < tracker.config.forward_backward_px
    stages['forward_backward_pass'] = int(ok.sum())
    correlations = np.zeros(len(x), np.float32)
    for i in np.flatnonzero(ok):
        a = cv2.getRectSubPix(key['raw'], (15, 15), tuple(map(float, x[i]))).astype(np.float32)
        b = cv2.getRectSubPix(gray, (15, 15), tuple(map(float, y[i]))).astype(np.float32)
        if min(float(a.std()), float(b.std())) < tracker.config.min_patch_std:
            ok[i] = False
            continue
        a -= a.mean(); b -= b.mean()
        correlations[i] = np.sum(a*b)/max(float(np.linalg.norm(a)*np.linalg.norm(b)), 1e-8)
    stages['textured_patch_pass'] = int(ok.sum())
    ok &= correlations >= tracker.config.min_patch_correlation
    stages['photometric_patch_pass'] = int(ok.sum())
    # Prewarp is only an LK initialization/capture aid. Return its measured
    # pixels to the actual image BEFORE fitting, so a background projective
    # proposal can never smuggle its eight DOF into the selected ice mapping.
    current_points = project(initial, y[ok]) if prewarp else y[ok]
    fit = _similarity_fit(current_points, x[ok], gray.shape, tracker.fit_config)
    old_fit = _fit(current_points, x[ok], gray.shape, tracker.fit_config)
    fit['unconstrained_homography_diagnostic'] = {k: old_fit[k] for k in (
        'status', 'reason', 'H_reference_from_current', 'train_inliers', 'support_fraction',
        'inlier_p95_reference_px') if k in old_fit}
    fit['unconstrained_homography_diagnostic']['used_for_selected_pose'] = False
    stages['ransac_inliers'] = fit.get('train_inliers')
    stages['spatial_support_fraction'] = fit.get('support_fraction')
    stages['support_width_fraction'] = fit.get('support_width_fraction')
    stages['support_height_fraction'] = fit.get('support_height_fraction')
    fit.update(base, fitting_residual_coordinate_space='observed_fixed_reference_pixels',
        pair_algorithm='fixed_reference_mask_isolated_LK+FB+patch_correlation+similarity_RANSAC',
        photometric_patch_correlation_median=float(np.median(correlations[ok])) if np.any(ok) else None)
    return fit


def _projection_chain(ref, h, probe, sample_size, source_size):
    view = ref['view']
    native = h @ ref['transform']
    distortion = _distortion_parameters(view.get('radial_distortion'))
    nonlinear = distortion is not None and distortion['k1'] != 0
    return {'H_reference_undistorted_from_model': view['H_image_from_model'],
        'reference_geometry_domain': 'undistorted_reference_sample_px' if nonlinear else 'raw_reference_sample_px_distortion_unchecked',
        'reference_image_domain': view.get('image_domain', 'raw_px_distortion_unchecked'),
        'reference_sample_size': list(view['sample_size']),
        'reference_visible_branch_sign': view.get('visible_branch_sign', 1),
        'reference_radial_distortion': distortion, 'reference_radial_domain': 'central_monotonic_branch_only',
        'H_current_sample_from_reference_raw': native.tolist(),
        'raw_transfer_visible_branch_sign': 1 if float(native[2] @ np.r_[probe, 1.]) > 0 else -1,
        'reference_branch_probe_raw': np.asarray(probe).tolist(),
        'H_source_image_from_sample': np.diag([source_size[0]/sample_size[0], source_size[1]/sample_size[1], 1.]).tolist(),
        'relative_geometry': 'relative_similarity_approximation_at_model_reobservation; local_projective_propagation_between',
        'approximation': 'local_raw_image_transfer_after_candidate_reference_radial_distortion' if nonlinear
                         else 'unverified_raw_plane_homography'}


def _model_probe_disagreement(model, first, second, usable, threshold):
    """Numerical stability probes, NOT additional ice measurements or accuracy.

    Test supplied structure samples and a uniform model-domain lattice wherever
    either hypothesis projects onto usable current ice. This also detects a
    hypothesis flinging previously visible model areas out of the frame.
    """
    _, primitives, bounds = _scene(model)
    xmin, ymin, xmax, ymax = bounds
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 41), np.linspace(ymin, ymax, 25))
    points = [np.c_[xx.ravel(), yy.ravel()]]
    for primitive in primitives:
        p = np.asarray(primitive['points'])
        if len(p) >= 2:
            points.append((p[:-1, None]+(p[1:]-p[:-1])[:, None]*np.linspace(0, 1, 9)[None, :, None]).reshape(-1, 2))
    points = np.unique(np.round(np.concatenate(points), 8), axis=0)
    height, width = usable.shape
    distances, compared = [], []
    def inside(xy):
        return (0 <= xy[0] < width and 0 <= xy[1] < height and usable[int(xy[1]), int(xy[0])] > 0)
    for p in points:
        a, b = _chain_point(p, first), _chain_point(p, second)
        if a is None or b is None:
            continue
        a, b = a[1][:2]/a[1][2], b[1][:2]/b[1][2]
        if inside(a) or inside(b):
            distances.append(float(np.linalg.norm(a-b)))
            compared.append(p.tolist())
    p95 = float(np.percentile(distances, 95)) if distances else None
    return {'probe_count': len(distances), 'median_sample_px': float(np.median(distances)) if distances else None,
        'p95_sample_px': p95, 'threshold_sample_px': threshold,
        'conflict': len(distances) < 8 or p95 > threshold,
        'reason': 'insufficient_comparable_model_probes' if len(distances) < 8 else 'model_projection_disagreement',
        'probe_definition': 'given_structure_samples_plus_model_bounds_lattice; union_of_current_usable_ice',
        'engineering_rejection_only': True, 'independent_accuracy_measurement': False}


class WholeRinkLocalizer:
    def __init__(self, model_path, root='.', config=None):
        self.model_path, self.root = Path(model_path).resolve(), Path(root).resolve()
        self.model = load_json(self.model_path)
        self.model_sha256 = sha256(self.model_path)
        # Capture before processing starts, rather than hashing potentially
        # edited working files only after a long-running sequence finishes.
        self.code_provenance = {name: sha256(Path(__file__).with_name(name)) for name in (
            'whole_rink_localization.py', 'continuous_registration.py', 'structure_render.py',
            'local_registration.py', 'foreground_rows.py', 'observed_localization.py')}
        self.config = config or WholeRinkConfig()
        self.refs = []
        reference_ids = set()
        _scene(self.model)  # Reject malformed geometry before processing video.
        for view in self.model.get('view_poses', []):
            for field in ('id', 'image', 'image_sha256', 'source_sha256', 'pts', 'time_base',
                          'sample_size', 'H_image_from_model'):
                if field not in view:
                    raise ValueError(f'Model reference {view.get("id")} missing {field}')
            _source_timestamp(view)
            if view['id'] in reference_ids:
                raise ValueError('Model reference IDs must be unique')
            reference_ids.add(view['id'])
            # Stored reference JPEGs may be full source resolution; sample_size
            # declares the H/mask/radial domain after full-frame resizing.
            # A pre-undistorted image cannot enter this raw-image route, which
            # would otherwise apply its radial warp a second time.
            if view.get('image_domain', 'raw_px_distortion_unchecked') != 'raw_px_distortion_unchecked':
                raise ValueError('Whole-rink reference image must use raw_px_distortion_unchecked; pre-undistorted references require a separate route')
            reference_h = np.asarray(view['H_image_from_model'], float)
            if (reference_h.shape != (3, 3) or not np.isfinite(reference_h).all()
                    or np.linalg.matrix_rank(reference_h) != 3):
                raise ValueError('Model reference H must be finite and nonsingular')
            _distortion_parameters(view.get('radial_distortion'))
            if (not isinstance(view['sample_size'], list) or len(view['sample_size']) != 2
                    or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in view['sample_size'])):
                raise ValueError('Reference sample_size requires positive integer pixel dimensions')
            path = (self.root / view['image']).resolve()
            if not path.is_file() or sha256(path) != view['image_sha256']:
                raise ValueError(f'Model reference image identity changed: {view["id"]}')
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'Cannot read model reference {view["id"]}')
            if view.get('visible_branch_sign', 1) not in (-1, 1):
                raise ValueError('Reference visible_branch_sign must be +1 or -1')
            rw, rh = map(int, view['sample_size'])
            cw, ch = self.config.sample_width, self.config.sample_height
            transform = np.diag([cw/rw, ch/rh, 1.])
            image = cv2.resize(image, (cw, ch), interpolation=cv2.INTER_AREA)
            ice_poly = view.get('ice_polygon_sample', view.get('ice_polygon_image'))
            if ice_poly is None:
                raise ValueError('Model reference needs an explicit whole visible ice polygon')
            plane = _polygon_mask((ch, cw), [project(transform, np.asarray(ice_poly))])
            people = [project(transform, np.asarray(p)).tolist() for p in
                      view.get('person_polygons_sample', view.get('person_polygons_image', []))]
            masks = frame_masks(image, plane, people)
            tracker = _registrar()
            sample = {'sample_index': -1, 'source_time_s': float(int(view['pts'])*Fraction(view['time_base']))}
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            key = tracker._key(gray, masks['usable'], sample, np.eye(3))
            self.refs.append({'view': view, 'image': image, 'masks': masks, 'tracker': tracker,
                'key': key, 'transform': transform, 'features': _rootsift(image, masks['usable'])})
        self.reset()

    def reset(self):
        self.propagation_segment = getattr(self, 'propagation_segment', -1)+1
        self.local = _registrar()
        self.bridge = None
        self.last_anchor = None
        self.last_anchor_time = None
        self.reference_branch_probe = None
        self.last_time = None
        self.last_shot = None
        self.last_source_sha256 = None
        self.observed_edges = {}
        self.anchor_sample_index = None
        self.pending_conflict = None
        self.anchor_lineage = []

    def _reobserve(self, image, masks, sample, prior):
        attempts, candidates = [], []
        current_features = None
        current_camera_features = None
        from .model_localization import _features as camera_features, _proposal as camera_proposal
        # Source time only chooses a numerical initial guess. No identity pose
        # is returned without observing and fitting real ice correspondences.
        ordered = sorted(self.refs, key=lambda r: (r['view']['source_sha256'] != sample['source_sha256'],
            abs(float(int(r['view']['pts'])*Fraction(r['view']['time_base']))-sample['source_time_s'])))
        for ref in ordered[:self.config.max_references]:
            view = ref['view']; proposals = []
            if ref['key'] is None:
                attempts.append({'reference_id': view['id'], 'status': 'unavailable', 'reason': 'reference_has_insufficient_ice_texture'})
                continue
            if prior is not None and self.last_anchor is ref:
                proposals.append({'H': prior, 'kind': 'bounded_local_motion_initial_guess'})
            near = (view['source_sha256'] == sample['source_sha256'] and
                    abs(float(int(view['pts'])*Fraction(view['time_base']))-sample['source_time_s']) <= self.config.near_reference_s)
            if near:
                proposals.append({'H': np.eye(3), 'kind': 'same_source_near_reference_initial_guess_only'})
            if not proposals:
                if current_features is None:
                    current_features = _rootsift(image, masks['usable'])
                proposal = _sift_proposal(ref['features'], current_features)
                if proposal is not None:
                    proposals.append(proposal)
            if not proposals:
                attempts.append({'reference_id': view['id'], 'status': 'unavailable', 'reason': 'no_ice_camera_proposal',
                    'reference_ice_descriptors': len(ref['features'][0]),
                    'current_ice_descriptors': len(current_features[0]) if current_features is not None else None})
            found = False
            for phase in range(2):
                if phase == 1:
                    # Reuse the old camera proposal only as an initial guess.
                    # Board points never enter the following ice fit or grid.
                    if 'camera_features' not in ref:
                        ref['camera_features'] = camera_features(ref['image'])
                    if current_camera_features is None:
                        current_camera_features = camera_features(image)
                    camera = camera_proposal(*ref['camera_features'], *current_camera_features)
                    proposals = ([{'H': c['H'], 'kind': 'background_camera_proposal_only',
                        'proposal_matches': c['matches'], 'proposal_scale': c.get('scale'),
                        'proposal_reference_blur_px': c.get('reference_blur_px')}
                        for c in [camera, *camera.get('alternatives', [])]] if camera is not None else [])
                    if not proposals:
                        attempts.append({'reference_id': view['id'], 'status': 'unavailable',
                            'reason': 'no_background_camera_proposal', 'plane_geometry_supported': False})
                for proposal in proposals:
                    for prewarp in ([False] if np.allclose(proposal['H'], np.eye(3)) else [False, True]):
                        fit = _ice_edge(ref, image, masks, proposal['H'], prewarp=prewarp)
                        attempts.append({'reference_id': view['id'], 'proposal': proposal['kind'],
                            **{k: v for k, v in proposal.items() if k.startswith('proposal_')}, **_compact_fit(fit)})
                        if fit['status'] == 'candidate':
                            candidates.append((ref, np.linalg.inv(np.asarray(fit['H_reference_from_current'])), fit))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        if not candidates:
            return None, attempts
        # A fit residual is not independent validation; all accepted states
        # remain candidate. Spatially broader real correspondences rank first.
        best = max(candidates, key=lambda v: v[2]['train_inliers']*v[2]['support_fraction']**.5)
        return best, attempts

    def push(self, image, sample, force_break=False):
        if self.config.continuity_mode == 'legacy':
            return self._push_legacy(image, sample, force_break)
        from .observed_localization import push_observed
        return push_observed(self, image, sample, force_break)

    def _push_legacy(self, image, sample, force_break=False):
        time = _source_timestamp(sample)
        expected = (self.config.sample_height, self.config.sample_width)
        if image.shape[:2] != expected:
            raise ValueError(f'Expected processing image shape {expected}')
        if any(isinstance(sample.get(k), bool) or not isinstance(sample.get(k), int) or sample[k] <= 0
               for k in ('width', 'height')):
            raise ValueError('Source pixel dimensions must be positive integers')
        changed_source = self.last_source_sha256 is not None and sample['source_sha256'] != self.last_source_sha256
        broken = (self.last_time is None or force_break or changed_source or (self.last_shot is not None and sample.get('shot_id') != self.last_shot)
                  or (self.last_time is not None and not 0 < time-self.last_time <= .6))
        if broken:
            self.reset()
        self.last_source_sha256 = sample['source_sha256']
        masks = frame_masks(image)
        local = self.local.push(image, sample, masks, force_break=broken)
        row = {**sample, 'model_id': self.model['model_id'], 'model_revision': self.model['revision'],
            'model_sha256': self.model_sha256, 'status': 'unavailable', 'reason': 'no_fixed_model_ice_reobservation',
            'geometry_level': self.model['geometry_level'], 'unit': self.model['unit'],
            'independent_validation': False, 'map_usable_for_metric_output': False,
            'H_image_from_model': None, 'H_model_from_image': None, 'projection_chain': None,
            'anchor_id': None, 'sample_size': [image.shape[1], image.shape[0]],
            'ice_polygons_sample': _polygons(masks['ice']), 'foreground_polygons_sample': _polygons(masks['foreground']),
            'mask_status': masks['mask_status'], 'foreground_method': masks['foreground_method'],
            'precise_person_segmentation': False, 'fit_support_polygon_sample': [],
            'whole_structure_overlay': 'conditional_extrapolation_on_full_visible_ice_not_independent_support',
            'local_registration': _compact_fit(local, points=False), 'break_applied': bool(broken),
            'propagation_segment_id': self.propagation_segment,
            'source_identity_break_applied': bool(changed_source)}
        prior = None
        local_h = local.get('H_reference_from_current')
        age = None if self.last_anchor_time is None else time-self.last_anchor_time
        if (self.bridge is not None and local_h is not None and local['status'] == 'candidate'
                and age is not None and age <= self.config.max_prediction_s+1e-9):
            prior = np.linalg.inv(np.asarray(local_h)) @ self.bridge
        attempt = broken or sample['sample_index'] % self.config.reobserve_every == 0
        candidate, attempts = self._reobserve(image, masks, sample, prior) if attempt else (None, [])
        row['reobservation_attempts'] = attempts
        if candidate is not None:
            ref, h, fit = candidate
            fresh_probe = project(np.linalg.inv(ref['transform']),
                np.mean(np.asarray(fit['fit_reference_xy']), axis=0)[None])[0]
            fresh_chain = _projection_chain(ref, h, fresh_probe,
                (image.shape[1], image.shape[0]), (sample['width'], sample['height']))
            if prior is not None:
                previous_chain = _projection_chain(self.last_anchor, prior, self.reference_branch_probe,
                    (image.shape[1], image.shape[0]), (sample['width'], sample['height']))
                consistency = _model_probe_disagreement(self.model, previous_chain, fresh_chain,
                    masks['usable'], self.config.anchor_conflict_p95_px_at_960*image.shape[1]/960)
                row['model_anchor_consistency'] = consistency
                if consistency['conflict']:
                    row.update(status='ambiguous', reason='model_anchor_conflict',
                        ice_reobservation=_compact_fit(fit), model_anchor_hypotheses=[
                            {'role': 'bounded_prior_not_ground_truth', 'reference_id': self.last_anchor['view']['id'],
                             'anchor_age_s': age, 'projection_chain': previous_chain},
                            {'role': 'fresh_ice_reobservation_candidate', 'reference_id': ref['view']['id'],
                             'projection_chain': fresh_chain}], propagation_barrier=True)
                    # Neither hypothesis wins. Clear the old propagation and
                    # do not let offline reverse fill cross this disagreement.
                    self.reset()
                    self.last_time, self.last_shot = time, sample.get('shot_id')
                    self.last_source_sha256 = sample['source_sha256']
                    return row
            row.update(status='candidate', reason='fixed_model_reference_ice_reobserved',
                ice_reobservation=_compact_fit(fit), fit_support_polygon_sample=fit['support_polygon'])
            self.last_anchor, self.last_anchor_time = ref, time
            # The mean is only a numerical branch probe inside the observed
            # correspondence hull, never a new material landmark/circle centre.
            self.reference_branch_probe = fresh_probe
            if local['status'] != 'candidate' or local_h is None:
                # A dead local component may restart only now: this frame has
                # just reobserved a real fixed-model reference. Its identity is
                # merely the new propagation gauge, explicitly bound by h.
                self.local = _registrar()
                local = self.local.push(image, sample, masks, force_break=True)
                local_h = local.get('H_reference_from_current')
                row['local_registration'] = _compact_fit(local, points=False)
                row['local_restart_reason'] = 'fresh_model_ice_reobservation_establishes_new_bound_component'
            # The local identity is only a coordinate gauge. This bridge is
            # created solely after an actual model-reference image match.
            if local_h is not None and local['status'] == 'candidate':
                self.bridge = np.asarray(local_h) @ h
            else:
                self.bridge = None
            row['anchor_age_s'] = 0.
        elif prior is not None:
            ref, h = self.last_anchor, prior
            row.update(status='predicted', reason='bounded_ice_motion_since_model_reobservation',
                       anchor_age_s=age, fit_support_polygon_sample=local.get('support_polygon', []))
        else:
            self.last_time, self.last_shot = time, sample.get('shot_id')
            return row
        view = ref['view']
        native_transfer = h @ ref['transform']
        raw_scale = np.diag([sample['width']/image.shape[1], sample['height']/image.shape[0], 1.])
        distortion = _distortion_parameters(view.get('radial_distortion'))
        nonlinear = distortion is not None and distortion['k1'] != 0
        row.update(anchor_id=view['id'], max_anchor_age_s=self.config.max_prediction_s,
            model_reference={'view_id': view['id'], 'image_sha256': view['image_sha256'],
                'source_sha256': view['source_sha256'], 'source_id': view.get('source_id'),
                'pts': view['pts'], 'time_base': view['time_base'], 'image': view['image'],
                'sample_size': view['sample_size'], 'image_domain': view.get('image_domain', 'raw_px_distortion_unchecked')},
            image_domain='raw_reference_transfer_approximation' if nonlinear else 'raw_px_distortion_unchecked',
            projection_chain=_projection_chain(ref, h, self.reference_branch_probe,
                (image.shape[1], image.shape[0]), (sample['width'], sample['height'])),
            cumulative_drift_independently_validated=False)
        if not nonlinear:
            raw_h = raw_scale @ native_transfer @ np.asarray(view['H_image_from_model'])
            row['H_image_from_model'], row['H_model_from_image'] = raw_h.tolist(), np.linalg.inv(raw_h).tolist()
        self.last_time, self.last_shot = time, sample.get('shot_id')
        return row


def _chain_point(point, chain):
    first = _homogeneous(np.asarray(chain['H_reference_undistorted_from_model']), point)
    if first is None or chain.get('reference_visible_branch_sign', 1)*first[2] <= 1e-10:
        return None
    raw = _distort(first[:2]/first[2], chain['reference_radial_distortion'])
    if raw is None:
        return None
    current = _homogeneous(np.asarray(chain['H_current_sample_from_reference_raw']), raw)
    if current is None or chain.get('raw_transfer_visible_branch_sign', 1)*current[2] <= 1e-10:
        return None
    return first, current


def projection_segments(points, chain, width, height, *, closed=False):
    """Sample in model coordinates; reject both stages' horizon crossings."""
    points = np.asarray(points, float)
    if closed and not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    # Vectorized sampling avoids tens of thousands of Python H applications
    # per video frame. Curves are simplified only after both projection stages
    # and only within a continuous finite branch, to subpixel image tolerance.
    samples = np.concatenate([points[:-1, None] + (points[1:]-points[:-1])[:, None]
                              * np.linspace(0, 1, 33)[None, :, None]], axis=0).reshape(-1, 2)
    first = np.c_[samples, np.ones(len(samples))] @ np.asarray(chain['H_reference_undistorted_from_model']).T
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        raw = first[:, :2]/first[:, 2:]
        distortion = chain['reference_radial_distortion']
        radial_valid = np.ones(len(raw), bool)
        if distortion is not None:
            delta = raw - np.asarray(distortion['centre_image'])
            radius_squared = np.sum((delta/distortion['scale_image'])**2, axis=1, keepdims=True)
            radial_valid = (1+3*distortion['k1']*radius_squared[:, 0]) > 0
            raw = np.asarray(distortion['centre_image'])+delta*(1+distortion['k1']*radius_squared)
        current = np.c_[raw, np.ones(len(raw))] @ np.asarray(chain['H_current_sample_from_reference_raw']).T
        xy = current[:, :2]/current[:, 2:]
    # The view declares which homogeneous half-plane was actually observed.
    # Radial candidates are usable only before their first folding derivative.
    valid = (np.isfinite(xy).all(axis=1) & (chain.get('reference_visible_branch_sign', 1)*first[:, 2] > 1e-10)
             & (chain.get('raw_transfer_visible_branch_sign', 1)*current[:, 2] > 1e-10) & radial_valid)
    joins = valid[:-1] & valid[1:] & (np.sign(first[:-1, 2]) == np.sign(first[1:, 2]))
    joins &= np.sign(current[:-1, 2]) == np.sign(current[1:, 2])
    starts = np.flatnonzero(joins & ~np.r_[False, joins[:-1]])
    ends = np.flatnonzero(joins & ~np.r_[joins[1:], False])+2
    for start, end in zip(starts, ends):
        branch = xy[start:end]
        if np.max(np.abs(branch)) < 1e8:
            branch = cv2.approxPolyDP(branch.astype(np.float32), .12, False).reshape(-1, 2)
        for a, b in zip(branch, branch[1:]):
            segment, reason = _visible_segment(np.eye(3), a, b, width, height)
            if reason is None:
                yield segment


def render_frame(image, model, row, *, captions=True, render_diagnostics=None):
    """Draw the entire supplied hypothesis, clipped only by ice/foreground."""
    output = image.copy(); height, width = image.shape[:2]
    if render_diagnostics is not None:
        render_diagnostics['visible_model_line_pixels'] = 0
    if row['status'] in DRAWABLE_STATES and row.get('projection_chain') is not None:
        allowed = _polygon_mask((height, width), row['ice_polygons_sample'])
        allowed[_polygon_mask((height, width), row['foreground_polygons_sample']) > 0] = 0
        layer, alpha = np.zeros_like(image), np.zeros((height, width), np.uint8)
        _, primitives, _ = _scene(model)
        for primitive in primitives:
            phase = 0.
            style=STYLES[primitive['status']]
            if 'landmark_observations' in row:
                style={**style,'rgb':(137,207,250) if primitive.get('grid') else (239,92,211)}
            for segment in projection_segments(primitive['points'], row['projection_chain'], width, height,
                                                closed=primitive['closed']):
                phase = _stroke(layer, alpha, segment, style, 1 if primitive.get('grid') else 2, phase)
        alpha[allowed == 0] = 0
        if render_diagnostics is not None:
            visible = alpha.copy()
            if captions:
                visible[:72] = 0
                visible[height-25:] = 0
            render_diagnostics['visible_model_line_pixels'] = int(np.count_nonzero(visible))
        weight = alpha.astype(float)[..., None]/255
        # OpenCV has already antialiased the colored line against black.
        output = np.clip(image*(1-weight)+layer*(allowed[..., None] > 0), 0, 255).astype(np.uint8)
    landmark_counts=None
    if 'landmark_observations' in row:
        from .landmark_render import draw_landmarks
        output,landmark_counts=draw_landmarks(output,row)
    if not captions:
        return output
    rgb = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)); draw = ImageDraw.Draw(rgb)
    font, _, _ = _font(max(13, width//48)); small, _, _ = _font(max(11, width//56))
    draw.rectangle((0, 0, width, 72), fill=(11, 20, 28))
    states = {'candidate': '冰面重观测候选', 'predicted': '短时传播', 'unavailable': '无法定位',
              'ambiguous': '存在冲突', 'model_reobserved': '认出场地',
              'image_tracked': '持续跟踪', 'motion_predicted': '暂时推算'}
    if landmark_counts is not None:
        states.update(model_reobserved='地标定位候选',image_tracked='沿图像连续推算')
    draw.text((9, 3), f"整场条件草模 · 比例与角部含假设 · 非实测尺寸 | {states[row['status']]}", font=font, fill='#ffcd86')
    draw.text((9, 28), f"{row.get('source_id', '')}  {row['source_time_s']:.3f}s  PTS {row['pts']} × {row['time_base']}", font=small, fill='white')
    nonlinear = row.get('projection_chain', {}).get('reference_radial_distortion') if row.get('projection_chain') else None
    relative = (row.get('projection_chain') or {}).get('relative_geometry', '')
    detail = ('候选畸变后的局部原始像素传递近似' if nonlinear and nonlinear['k1'] else
        '相对相似映射近似；无独立几何验收' if relative else '原始像素近似；无独立几何验收')
    if row.get('camera'):
        detail = '同一相机＋分类型图像观测；场地和镜头参数仍含条件'
    if row.get('future_anchor_used'):
        future = row.get('future_anchor_observation')
        detail = ((f"前后观测约束，未来锚 {future['source_time_s']:.3f}s；" if future else
                   '未来锚参与比较；') + detail)
    caption = (('使用前后原片观测；' if row.get('future_anchor_used') else '使用真实原片观测；') +
               '场地与相机仍含假设，连续显示不等于贴准') if row.get('camera') else f"参考 {row.get('anchor_id') or '无'} · {detail}"
    draw.text((9, 49), caption, font=small, fill='#d6e4f4')
    draw.rectangle((0, height-25, width, height), fill=(11, 20, 28))
    legend = ('模型投影：青为已观察结构，蓝为推断，橙为假设；尚未全场验准' if row.get('camera') else
              '青：观测线   蓝：推断线   橙虚线：假设   全冰面外推 ≠ 全冰面已验证')
    if landmark_counts is not None:
        legend=f"绿：参与求解的真实弧 | 粉：模型预测 | 检出 {landmark_counts['detected']}  对应 {landmark_counts['associated']}  使用 {landmark_counts['used']}（非验准）"
    draw.text((9, height-23), legend, font=small, fill='#ffcd86')
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def cut_signature(image):
    tiny = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(tiny, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hist /= max(float(hist.sum()), 1)
    return cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY), hist


def cut_diagnostic(previous, current):
    """Reuse the source-intake cut candidate criteria at full source cadence."""
    difference = float(np.mean(cv2.absdiff(previous[0], current[0]))/255)
    histogram = float(np.sum(np.abs(previous[1]-current[1]))/2)
    return {'candidate': bool((difference > .055 and histogram > .16) or difference > .31),
            'mean_difference': difference, 'histogram_distance': histogram,
            'status': 'automatic_cut_candidate_not_human_confirmed'}


def _source_images(root, source_id, rows, config):
    lookup = {(int(r['pts']), str(Fraction(r['time_base']))): r for r in rows}
    first = min(int(r['pts'])*Fraction(r['time_base']) for r in rows)
    last = max(int(r['pts'])*Fraction(r['time_base']) for r in rows)
    with av.open(str(source_path(root, source_id))) as source:
        stream = source.streams.video[0]
        source.seek(int(first/stream.time_base), stream=stream)
        count = 0
        for frame in source.decode(stream):
            if frame.pts is None: raise ValueError('Source frame missing PTS')
            if int(frame.pts)*frame.time_base > last: break
            row = lookup.get((int(frame.pts), str(frame.time_base)))
            if row is not None:
                count += 1
                yield row, cv2.resize(frame.to_ndarray(format='bgr24'),
                    (config.sample_width, config.sample_height), interpolation=cv2.INTER_AREA)
        if count != len(rows): raise RuntimeError('Reverse/render pass lost exact source frames')


def _source_shot_context(shots, source_time):
    """Pass through recorded roles/epochs without promoting an automatic label.

    This uses the recorded source-shot interval, which can differ by a frame
    from this run's full-cadence cut. Both identities stay explicit.
    """
    shot = next((s for s in shots if s['start_s'] <= source_time < s['end_s']), None)
    if shot is None:
        return {}
    import copy
    passthrough = ('video_role', 'role', 'epoch', 'epoch_id', 'background_epoch',
        'background_epoch_id', 'video_epoch', 'camera_epoch', 'camera_epoch_id', 'replay')
    result = {key: copy.deepcopy(shot[key]) for key in passthrough if key in shot}
    result['source_shot_metadata'] = {key: copy.deepcopy(shot[key]) for key in
        ('shot_id', 'start_s', 'end_s', 'review_status', 'boundary_status', 'role_candidates', *passthrough)
        if key in shot}
    if 'shot_id' in shot:
        result['source_shot_id'] = shot['shot_id']
    return result


def reverse_fill(root, source_id, frames, config, *, model=None):
    """Offline bounded propagation from future *real* model observations.

    Source timestamps never change. Only the local registrar sees a separately
    declared reverse-elapsed tracking clock; no inferred frame becomes an anchor.
    A future-anchor estimate must agree at its overlap with existing forward
    estimates. Disagreement leaves both hypotheses explicit and blank instead
    of silently joining independently bounded tracks at their expiry boundary.
    """
    filled, conflicts = 0, 0
    anchors = [(i, r) for i, r in enumerate(frames) if r['status'] == 'candidate' and r.get('projection_chain')]
    for anchor_index, anchor in anchors:
        if anchor['status'] != 'candidate': continue
        # This can change during the reverse pass. Never skip a barrier row
        # then track across the resulting timestamp gap on the same shot.
        last_barrier = max((i for i, r in enumerate(frames[:anchor_index])
                            if r.get('propagation_barrier')), default=-1)
        targets = [r for r in frames[last_barrier+1:anchor_index] if r.get('shot_id') == anchor.get('shot_id')
            and r.get('source_sha256') == anchor.get('source_sha256')
            and r.get('propagation_segment_id') == anchor.get('propagation_segment_id')
            and not r.get('propagation_barrier')
            and 0 < anchor['source_time_s']-r['source_time_s'] <= config.max_prediction_s+1e-9]
        if not any(r['status'] == 'unavailable' for r in targets): continue
        sequence = [*targets, anchor]
        pictures = {r['sample_index']: image for r, image in _source_images(root, source_id, sequence, config)}
        local = _registrar()
        anchor_time = int(anchor['pts'])*Fraction(anchor['time_base'])
        base = Fraction(anchor['time_base'])
        for ordinal, row in enumerate(reversed(sequence)):
            elapsed = anchor_time-int(row['pts'])*Fraction(row['time_base'])
            ticks = elapsed/base
            if ticks.denominator != 1: raise ValueError('Reverse tracking time base cannot represent source PTS')
            image = pictures[row['sample_index']]
            tracking_sample = {'sample_index': ordinal, 'pts': int(ticks), 'time_base': str(base),
                               'source_time_s': float(elapsed)}
            masks = frame_masks(image)
            relative = local.push(image, tracking_sample, masks, force_break=ordinal == 0)
            if ordinal == 0 or relative['status'] != 'candidate': continue
            if relative.get('reference_frame_only'): continue
            import copy
            chain = copy.deepcopy(anchor['projection_chain'])
            chain['H_current_sample_from_reference_raw'] = (np.linalg.inv(np.asarray(relative['H_reference_from_current']))
                @ np.asarray(chain['H_current_sample_from_reference_raw'])).tolist()
            probe = chain.get('reference_branch_probe_raw')
            if probe is not None:
                chain['raw_transfer_visible_branch_sign'] = 1 if float(np.asarray(chain['H_current_sample_from_reference_raw'])[2] @ np.r_[probe, 1.]) > 0 else -1
            if row['status'] in {'candidate', 'predicted'} and row.get('projection_chain'):
                if model is None:
                    raise ValueError('Reverse overlap consistency requires the fixed shared model')
                consistency = _model_probe_disagreement(model, row['projection_chain'], chain, masks['usable'],
                    config.anchor_conflict_p95_px_at_960*image.shape[1]/960)
                row.setdefault('reverse_anchor_consistency_checks', []).append({**consistency,
                    'future_anchor_pts': anchor['pts'], 'future_anchor_time_base': anchor['time_base']})
                if consistency['conflict']:
                    row['forward_status_before_reverse_conflict'] = row['status']
                    row['forward_reason_before_reverse_conflict'] = row['reason']
                    row.update(status='ambiguous', reason='model_anchor_conflict',
                        model_anchor_consistency=consistency, model_anchor_hypotheses=[
                            {'role': 'forward_estimate_not_ground_truth', 'reference_id': row['anchor_id'],
                             'projection_chain': copy.deepcopy(row['projection_chain'])},
                            {'role': 'future_anchor_reverse_estimate_not_ground_truth', 'reference_id': anchor['anchor_id'],
                             'future_anchor_pts': anchor['pts'], 'future_anchor_time_base': anchor['time_base'],
                             'projection_chain': chain}], propagation_barrier=True,
                        projection_chain=None, H_image_from_model=None, H_model_from_image=None, anchor_id=None,
                        conflict_discovered_by='offline_reverse_same_frame_overlap')
                    conflicts += 1
                    break
                continue
            if row['status'] != 'unavailable': continue
            diagnostic = _compact_fit(relative, points=False)
            for key in ('pts', 'time_base', 'source_time_s', 'sample_index'): diagnostic.pop(key, None)
            row['forward_reason_before_reverse'] = row['reason']
            row.update(status='predicted', reason='bounded_reverse_ice_motion_from_future_model_reobservation',
                projection_chain=chain, anchor_id=anchor['anchor_id'], anchor_age_s=float(elapsed),
                max_anchor_age_s=config.max_prediction_s, model_reference=copy.deepcopy(anchor['model_reference']),
                image_domain=anchor['image_domain'], future_anchor_used=True, propagation_direction='offline_reverse',
                future_anchor_observation={'pts': anchor['pts'], 'time_base': anchor['time_base'],
                    'source_time_s': anchor['source_time_s'], 'source_frame_index': anchor.get('source_frame_index'),
                    'status': anchor['status'], 'model_reference': copy.deepcopy(anchor['model_reference'])},
                reverse_local_registration={**diagnostic, 'tracking_clock': {'kind': 'reverse_elapsed_from_future_anchor',
                    'elapsed_s': float(elapsed), 'source_anchor_pts': anchor['pts'], 'source_anchor_time_base': anchor['time_base']}},
                fit_support_polygon_sample=relative.get('support_polygon', []),
                cumulative_drift_independently_validated=False)
            if anchor.get('H_image_from_model') is not None:
                h = (np.asarray(chain['H_source_image_from_sample']) @ np.asarray(chain['H_current_sample_from_reference_raw'])
                     @ np.asarray(chain['H_reference_undistorted_from_model']))
                row['H_image_from_model'], row['H_model_from_image'] = h.tolist(), np.linalg.inv(h).tolist()
            filled += 1
    return {'enabled': True, 'future_anchor_used': bool(filled), 'filled_frames': filled,
            'overlap_conflict_frames': conflicts, 'overlap_conflict_policy': 'same_frame_full_model_probes; no_hypothesis_promoted; reverse_stops_at_barrier',
            'max_prediction_s': config.max_prediction_s, 'inferred_frames_become_anchors': False}


def _rewrite_overlay(root, source_id, frames, model, config, output, rate, time_base):
    path = output/'whole_rink_overlay.reverse.tmp.mp4'
    first = int(frames[0]['pts'])*Fraction(frames[0]['time_base'])
    with av.open(str(path), 'w') as encoded:
        writer = encoded.add_stream('libx264', rate=rate)
        writer.width, writer.height, writer.pix_fmt = 640, 360, 'yuv420p'
        writer.time_base = time_base; writer.options = {'crf': '26', 'preset': 'fast'}
        for i, (row, image) in enumerate(_source_images(root, source_id, frames, config)):
            overlay = render_frame(image, model, row)
            frame = av.VideoFrame.from_ndarray(cv2.resize(overlay, (640, 360), interpolation=cv2.INTER_AREA), format='bgr24')
            pts = (int(row['pts'])*Fraction(row['time_base'])-first)/time_base
            if pts.denominator != 1: raise ValueError('Reverse overlay PTS rounding rejected')
            frame.pts, frame.time_base = int(pts), time_base
            for packet in writer.encode(frame): encoded.mux(packet)
            if i in {0, len(frames)//2, len(frames)-1}:
                cv2.imwrite(str(output/f'comparison_{i:05d}.jpg'), np.hstack([image, overlay]))
        for packet in writer.encode(): encoded.mux(packet)
    path.replace(output/'whole_rink_overlay.mp4')


def track_window(root, model_path, source_id, start_s, end_s, output, *,
                 selection_basis, cut_times=(), config=None, offline_reverse=False,
                 evaluation_window=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new empty run directory; previous evidence is not overwritten')
    output.mkdir(parents=True, exist_ok=True)
    config = config or WholeRinkConfig()
    localizer = WholeRinkLocalizer(model_path, root, config)
    index = load_index(root, source_id)
    expected_digest = index.get('source_sha256', index.get('video_id'))
    if index.get('source_sha256', expected_digest) != index.get('video_id', expected_digest):
        raise ValueError('Source index SHA-256 fields disagree')
    current_digest = sha256(source_path(root, source_id))
    if current_digest != expected_digest:
        raise ValueError('Source file SHA-256 differs from its decoded PTS index; rebuild the index')
    chosen = [r for r in index['frames'] if start_s <= r['source_time_s'] < end_s]
    if not chosen:
        raise ValueError('No decoded source frames in selected full interval')
    lookup = {(int(r['pts']), str(Fraction(r['time_base']))): r for r in chosen}
    evaluation_start, evaluation_end = evaluation_window or (start_s, end_s)
    if not start_s <= evaluation_start < evaluation_end <= end_s:
        raise ValueError('Evaluation interval must be inside solver context')
    evaluation_rows = [r for r in chosen if evaluation_start <= r['source_time_s'] < evaluation_end]
    if not evaluation_rows:
        raise ValueError('Empty evaluation interval')
    first_time = int(evaluation_rows[0]['pts'])*Fraction(evaluation_rows[0]['time_base'])
    solver_first_time = int(chosen[0]['pts'])*Fraction(chosen[0]['time_base'])
    # Reuse source shot boundaries in addition to explicit cuts. They remain
    # labelled automatic candidates, not newly manually confirmed boundaries.
    try:
        shots = load_shots(root, source_id)['shots']
    except FileNotFoundError:
        shots = []
    cuts = sorted(set(float(t) for t in cut_times) | {float(s['start_s']) for s in shots if start_s < s['start_s'] < end_s})
    frames, mappings, writers = [], [], []
    rate = Fraction(1, 1)/(int(chosen[1]['pts'])*Fraction(chosen[1]['time_base'])-solver_first_time) if len(chosen)>1 else Fraction(25)
    time_base = Fraction(chosen[0]['time_base'])
    for name in ('source.mp4', 'whole_rink_overlay.mp4'):
        container = av.open(str(output/name), 'w')
        writer = container.add_stream('libx264', rate=rate)
        writer.width, writer.height, writer.pix_fmt = 640, 360, 'yuv420p'
        writer.time_base = time_base; writer.options = {'crf': '26', 'preset': 'fast'}
        writers.append((container, writer))
    previous_time, shot_number = None, 0
    previous_signature, last_cut_time, detected_cuts = None, -float('inf'), []
    merged_cut_observations, applied_cuts = [], []
    try:
        with av.open(str(source_path(root, source_id))) as container:
            stream = container.streams.video[0]; stream.thread_type = 'AUTO'
            container.seek(int(solver_first_time/stream.time_base), stream=stream)
            for decoded in container.decode(stream):
                if decoded.pts is None:
                    raise ValueError('Decoded source frame missing PTS')
                ts = Fraction(decoded.pts)*decoded.time_base
                if float(ts) >= end_s:
                    break
                original = lookup.get((int(decoded.pts), str(decoded.time_base)))
                if original is None:
                    continue
                i = len(frames)
                image = cv2.resize(decoded.to_ndarray(format='bgr24'), (config.sample_width, config.sample_height), interpolation=cv2.INTER_AREA)
                signature = cut_signature(image)
                cut = cut_diagnostic(previous_signature, signature) if previous_signature is not None else None
                image_cut = cut is not None and cut['candidate'] and float(ts)-last_cut_time > .35
                declared = [t for t in cuts if previous_time is not None and previous_time < t <= float(ts)]
                duplicate = bool(declared and float(ts)-last_cut_time <= .08)
                if duplicate:
                    merged_cut_observations.append({'declared_times_s': declared,
                        'applied_transition_s': last_cut_time, 'reason': 'adjacent_duplicate_cut_candidate'})
                declared_cut = bool(declared and not duplicate)
                broken = i == 0 or image_cut or declared_cut
                if image_cut:
                    detected_cuts.append({**cut, 'source_time_s': float(ts), 'pts': int(decoded.pts),
                        'time_base': str(decoded.time_base), 'source_frame_index': original['source_frame_index'],
                        'transition_interval_s': [previous_time, float(ts)]})
                    last_cut_time = float(ts)
                if declared_cut:
                    last_cut_time = float(ts)
                if i and broken:
                    applied_cuts.append(float(ts))
                if i and broken:
                    shot_number += 1
                sample = {**original, 'sample_index': i, 'source_id': source_id,
                    'source_sha256': current_digest,
                    'source_pts': int(decoded.pts), 'source_time_base': str(decoded.time_base),
                    'width': decoded.width, 'height': decoded.height, 'shot_id': f'{source_id}_whole_{shot_number:03d}'}
                row = localizer.push(image, sample, force_break=broken)
                row.update(_source_shot_context(shots, float(ts)))
                if cut is not None:
                    row['source_frame_cut_diagnostic'] = {**cut, 'break_applied': bool(image_cut)}
                frames.append(row)
                if not evaluation_start <= float(ts) < evaluation_end:
                    previous_time, previous_signature = float(ts), signature
                    continue
                overlay = render_frame(image, localizer.model, row)
                for picture, (target, writer) in zip((image, overlay), writers):
                    frame = av.VideoFrame.from_ndarray(cv2.resize(picture, (640, 360), interpolation=cv2.INTER_AREA), format='bgr24')
                    pts = (ts-first_time)/time_base
                    if pts.denominator != 1:
                        raise ValueError('Encoder time base cannot preserve source relative timestamps')
                    frame.pts, frame.time_base = int(pts), time_base
                    for packet in writer.encode(frame): target.mux(packet)
                mappings.append({'pts': int(decoded.pts), 'time_base': str(decoded.time_base),
                                 'source_frame_index': original['source_frame_index']})
                if i in {0, len(chosen)//2, len(chosen)-1}:
                    cv2.imwrite(str(output/f'comparison_{i:05d}.jpg'), np.hstack([image, overlay]))
                if i % 100 == 0:
                    print(i, dict(Counter(x['status'] for x in frames)), flush=True)
                previous_time, previous_signature = float(ts), signature
        if len(frames) != len(chosen):
            raise RuntimeError('Full decoded source window lost frames')
        for target, writer in writers:
            for packet in writer.encode(): target.mux(packet)
    finally:
        for target, _ in writers: target.close()
    reverse_diagnostic = {'enabled': False}
    if offline_reverse:
        if config.continuity_mode != 'legacy':
            raise ValueError('Observed continuity uses the joint graph postprocessor, not legacy reverse fill')
        reverse_diagnostic = reverse_fill(root, source_id, frames, config, model=localizer.model)
        if reverse_diagnostic['filled_frames'] or reverse_diagnostic['overlap_conflict_frames']:
            _rewrite_overlay(root, source_id, frames, localizer.model, config, output, rate, time_base)
    for name, prefix in [('source.mp4', 'proxy'), ('whole_rink_overlay.mp4', 'derived')]:
        with av.open(str(output/name)) as check:
            count = 0
            for count, frame in enumerate(check.decode(video=0), start=1):
                if count > len(mappings): raise RuntimeError('Encoder added frames')
                mapping = mappings[count-1]
                expected = int(mapping['pts'])*Fraction(mapping['time_base'])-first_time
                if frame.pts is None or int(frame.pts)*frame.time_base != expected:
                    raise RuntimeError('Encoded proxy/source relative PTS mismatch')
                mapping.update({f'{prefix}_pts': int(frame.pts), f'{prefix}_time_base': str(frame.time_base)})
            if count != len(mappings): raise RuntimeError('Encoded proxy frame count mismatch')
    from .continuity_storage import save_observations
    storage = save_observations(frames, output/'local_observations.npz')
    solver_frames = frames
    frames = [r for r in solver_frames if evaluation_start <= r['source_time_s'] < evaluation_end]
    record = {'schema_version': 'rinkspace.whole-rink-localization.v1', 'algorithm_version': ALGORITHM_VERSION,
        'implementation_sha256_at_start': localizer.code_provenance,
        'anchor_conflict_policy': 'full_visible_ice_model_probe_P95_engineering_rejection_not_geometric_validation; neither_hypothesis_wins; propagation_and_reverse_barrier',
        'model_id': localizer.model['model_id'], 'model_revision': localizer.model['revision'],
        'model_sha256': localizer.model_sha256, 'source_id': source_id, 'source_sha256': frames[0]['source_sha256'],
        'source_identity_verified_before_decode': True,
        'config': asdict(config), 'selected_window': {'start_s': evaluation_start, 'end_s': evaluation_end,
            'first_pts': frames[0]['pts'], 'last_pts': frames[-1]['pts'], 'time_base': frames[0]['time_base'],
            'frame_count': len(frames), 'selection_basis': selection_basis},
        'cut_times_s': applied_cuts,
        'original_declared_cut_times_s': cuts, 'merged_cut_observations': merged_cut_observations,
        'detected_cut_candidates': detected_cuts,
        'cut_policy': 'source_candidate_boundaries_plus_declared_cuts_plus_full_frame_intake_detector',
        'source_role_policy': 'existing_source_shot_fields_only; unreviewed_values_preserved; source_shot_id_distinct_from_full_cadence_run_shot_id',
        'local_correspondence_storage': storage,
        'solver_context': {'start_s': start_s, 'end_s': end_s, 'frame_count': len(solver_frames),
            'source_pts_unchanged': True, 'file': 'solver_context.json', 'future_context_available': end_s > evaluation_end,
            'camera_cuts_keep_components_separate': True},
        'offline_reverse_propagation': reverse_diagnostic,
        'status_counts': dict(Counter(r['status'] for r in frames)), 'all_selected_frames_in_denominator': True,
        'independent_validation_passed': False, 'original_media_changed': False, 'frames': frames}
    dump_json(output/'localization.json', record)
    dump_json(output/'solver_context.json', {k:v for k,v in record.items() if k != 'frames'} | {'frames': solver_frames})
    dump_json(output/'source_pts_map.json', {'source_sha256': record['source_sha256'],
        'proxy_sha256': sha256(output/'source.mp4'), 'frames': mappings})
    dump_json(output/'overlay_pts_map.json', {'source_sha256': record['source_sha256'],
        'derived_sha256': sha256(output/'whole_rink_overlay.mp4'), 'model_id': record['model_id'],
        'model_revision': record['model_revision'], 'frames': mappings})
    return record
