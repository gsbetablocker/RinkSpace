"""Fit a continuous world root trajectory directly to full-programme sources.

One cubic B-spline world XYZ trajectory is shared across physical camera cuts.
Each cut has a smooth correction to the saved image-conditioned root direction;
every point of the saved body shares that R/T. No articulation, camera, shoe,
ice-contact or floor condition is optimized. Missing source evidence remains
missing and the resulting temporal inference is explicitly marked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation

from .camera_bridge import project_world
from .pelvis_ray_family_working import edge_clock
from .whole_body_source_envelope import envelope_pixel_factor, DIRECTIONS
from .body_prior import original_observation_mask

BODY_SCALE = .9136808725841283
UNIT = .31353729994266055
OPENING_RELIABLE_UPPER_BODY_SOURCE_PTS = 229175552
OPENING_FIRST_FULL_BODY_SOURCE_PTS = 229182208
SOURCE_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16])
# The existing proper COCO vertex regressor also supplies nose/eye/ear proxies.
# They constrain coarse root orientation in portraits; no face pose is fitted.
SOURCE_SIGMA = np.asarray([12.] * 11 + [24., 24.])


def _load(path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (z.files if keys is None else keys)}


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def packet_metres_per_scene_unit(packet):
    """Read the declared uniform model scale instead of the historical v8 scale."""
    value = np.asarray(packet['metres_per_scene_unit'])
    if value.shape != () or value.dtype.kind not in 'fiu':
        raise ValueError('Body input requires one explicit metres-per-scene-unit scalar')
    unit = float(value)
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError('Metres per scene unit must be finite and positive')
    return unit


def _slerp_pairs(a, b, alpha):
    qa, qb = Rotation.from_matrix(a).as_quat(), Rotation.from_matrix(b).as_quat()
    dot = np.sum(qa * qb, axis=1)
    qb[dot < 0] *= -1
    dot = np.clip(np.abs(dot), 0., 1.)
    angle = np.arccos(dot)
    small = angle < 1e-7
    denominator = np.sin(np.maximum(angle, 1e-7))
    wa = np.sin((1. - alpha) * angle) / denominator
    wb = np.sin(alpha * angle) / denominator
    wa[small], wb[small] = 1. - alpha[small], alpha[small]
    quaternion = qa * wa[:, None] + qb * wb[:, None]
    quaternion /= np.linalg.norm(quaternion, axis=1)[:, None]
    return Rotation.from_quat(quaternion).as_matrix()


def prepare_programme_input(base, camera_path, hip_path, mapping_path, output):
    """Directly bind saved source/local geometry to current physical cameras."""
    base, output = Path(base), Path(output)
    prior_path, obs_path = base/'prior25/body_prior.npz', base/'observations/observations.npz'
    # Read only local geometry/clock from the historical body file.
    prior = _load(prior_path, ['pts','prior_valid','prior_origin','model_slot_pts',
        'interpolation_left_source_pts','interpolation_right_source_pts','interpolation_alpha',
        'coco17_body_offset_m','native22_body_offset_m','shot_index'])
    obs, cameras, hip = map(_load, [obs_path, camera_path, hip_path])
    native = cameras['pts']
    for item in [prior, obs, hip]:
        if not np.array_equal(item['pts'], native):
            raise ValueError('Full-programme original native clock differs')
    representative = cameras['exposure_representative_pts']
    actual = np.unique(representative)
    canonical_rows = np.searchsorted(native, actual)
    if not np.array_equal(native[canonical_rows], actual):
        raise ValueError('Current physical camera lacks a canonical actual exposure')
    native_lookup = {int(p): i for i, p in enumerate(native)}
    source_rows = canonical_rows.copy()
    for i, row in enumerate(canonical_rows):
        wanted = int(hip['source_observation_pts'][row])
        candidate = native_lookup.get(wanted, int(row))
        if representative[candidate] != actual[i]:
            raise ValueError('Saved source observation is not in the same confirmed exposure')
        source_rows[i] = candidate
    native_mask_input = dict(coco17=np.concatenate((obs['native_coco17_xy'],
        obs['keypoint_scores'][:,:,None]),axis=2), keypoint_valid=obs['keypoint_valid'],
        keypoint_in_source=obs['keypoint_in_source'],
        source_clipped_target=obs['source_clipped_target'],bbox_native_xyxy=obs['bbox_native_xyxy'])
    original_mask, original_truncated = original_observation_mask(native_mask_input)
    boxes = obs['bbox_native_xyxy'][source_rows]
    finite_box = np.isfinite(boxes).all(axis=1)
    portrait = finite_box & (boxes[:,3] >= 1080.-24.) & ((boxes[:,3]-boxes[:,1]) > 1080./2.)
    body_rows = canonical_rows.copy()
    donor_actual_row = np.arange(len(actual))
    shot_actual = obs['shot_index'][canonical_rows]
    reliable_opening_row = int(np.searchsorted(actual,OPENING_RELIABLE_UPPER_BODY_SOURCE_PTS))
    if (reliable_opening_row>=len(actual) or actual[reliable_opening_row]!=OPENING_RELIABLE_UPPER_BODY_SOURCE_PTS
            or shot_actual[reliable_opening_row]!=1):
        raise ValueError('Reviewed reliable opening upper-body exposure must be present exactly')
    opening_inferred = (shot_actual==1) & (actual<OPENING_FIRST_FULL_BODY_SOURCE_PTS)
    # The reviewed upright upper-body source is useful despite cropped feet.
    # A later fully boxed source is turning/low-headed, a wrong portrait donor.
    portrait_pose_untrusted = portrait & ~((shot_actual==1)&~opening_inferred)
    inferred_pose = ~prior['prior_valid'][body_rows] | portrait_pose_untrusted | opening_inferred
    # A cropped image-conditioned body is not an independent full-body shape/
    # direction observation. Infer the entire fixed local pose and direction
    # from a nearby visible same-cut body, never generate missing source UV.
    reliable_body = (prior['prior_valid'][canonical_rows] & finite_box & ~portrait
                     & original_mask[source_rows][:,[5,6,11,12]].all(axis=1))
    for shot in np.unique(shot_actual):
        wanted = np.flatnonzero((shot_actual == shot) & inferred_pose)
        candidates = np.flatnonzero((shot_actual == shot) & reliable_body)
        if len(wanted) and not len(candidates):
            raise ValueError('No saved local pose for an entire cut')
        for i in wanted:
            donor = (reliable_opening_row if shot==1 else
                     candidates[np.argmin(np.abs(actual[candidates] - actual[i]))])
            donor_actual_row[i] = donor
            body_rows[i] = canonical_rows[donor]
    local = prior['coco17_body_offset_m'][body_rows].astype(float) * BODY_SCALE
    local_native = prior['native22_body_offset_m'][body_rows].astype(float) * BODY_SCALE
    if not np.isfinite(local).all() or not np.isfinite(local_native).all():
        raise ValueError('Saved local pose temporal inference did not produce finite geometry')
    if np.max(np.abs(local_native[:, 0])) > 1e-8:
        raise ValueError('Saved native joint0 is not the local origin')
    cam_all = cameras['camera25']
    raw_world = {}
    raw_bindings = []
    for shot in range(1, 10):
        path = base/f'prior25/raw_shot_{shot:02d}.npz'
        # Only image-conditioned root rotation is read, never network/global XYZ.
        with np.load(path, allow_pickle=False) as z:
            stamps, angles = z['source_pts'].copy(), z['global_orient'].copy()
        for p, angle in zip(stamps, angles):
            row = native_lookup[int(p)]
            r = cv2.Rodrigues(cam_all[row, 3:6])[0]
            raw_world[int(p)] = r.T @ cv2.Rodrigues(angle.astype(float))[0]
        raw_bindings.append(dict(path=str(path), sha256=_digest(path)))
    left = prior['interpolation_left_source_pts'][body_rows].astype(np.int64)
    right = prior['interpolation_right_source_pts'][body_rows].astype(np.int64)
    alpha = prior['interpolation_alpha'][body_rows].astype(float)
    a, b = [], []
    for i, (l, r) in enumerate(zip(left, right)):
        if int(l) not in raw_world or int(r) not in raw_world:
            # A direct saved model row may use its model slot rather than brackets.
            p = int(prior['model_slot_pts'][body_rows[i]])
            if p not in raw_world:
                raise ValueError('Saved local pose lacks its actual raw root rotation')
            left[i] = right[i] = p
            alpha[i] = 0.
        a.append(raw_world[int(left[i])]); b.append(raw_world[int(right[i])])
    Q_prior = _slerp_pairs(np.asarray(a), np.asarray(b), alpha)
    source_uv = obs['native_coco17_xy'][source_rows].copy()
    source_score = obs['keypoint_scores'][source_rows].copy()
    # Preserve the established original source-domain/truncation policy.
    # False lower-body heatmap predictions in bottom-truncated portraits never
    # become usable observations merely through finite, in-image UV/scores.
    source_qualified = original_mask[source_rows].copy()
    # Preserve previously reviewed hip failures and conditional opening proxies.
    source_uv[:, [11, 12]] = hip['source_H_hip_pair_uv'][canonical_rows]
    source_score[:, [11, 12]] = hip['source_H_hip_pair_score'][canonical_rows]
    reviewed_hip_eligible = (hip['source_H_hip_pair_eligible'][canonical_rows]
        & hip['source_hip_proxy_candidate_valid'][canonical_rows, None]
        & np.isfinite(source_uv[:, [11, 12]]).all(axis=2))
    opening_override = hip['source_review_conditional_proxy_override_valid'][canonical_rows].copy()
    if np.count_nonzero(opening_override) != 4 or np.any(obs['shot_index'][canonical_rows][opening_override] != 1):
        raise ValueError('Opening source override must contain only four reviewed actual hip exposures')
    source_qualified[:, [11,12]] &= reviewed_hip_eligible
    source_qualified[opening_override,11:13] = reviewed_hip_eligible[opening_override]
    dt, origins, clock_rows = edge_clock(actual, mapping_path)
    times = np.r_[0., np.cumsum(dt)]
    cam = cam_all[canonical_rows].copy()
    initial = np.full((len(actual), 3), np.nan)
    direct_init = np.zeros(len(actual), bool)
    for i in range(len(actual)):
        ids = SOURCE_INDICES[source_qualified[i, SOURCE_INDICES]]
        if len(ids) < 4:
            continue
        K = np.array([[np.exp(cam[i, 6]), 0., cam[i, 8]],
                      [0., np.exp(cam[i, 7]), cam[i, 9]], [0., 0., 1.]])
        try:
            ok, rvec, tvec = cv2.solvePnP(local[i, ids], source_uv[i, ids], K,
                                        cam[i, 11:25], flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            continue
        if ok:
            qi = cv2.Rodrigues(rvec)[0]
            camera_points = local[i] @ qi.T + tvec.ravel()
            if (camera_points[:, 2] > .05).all():
                R = cv2.Rodrigues(cam[i, 3:6])[0]
                initial[i] = cam[i, :3] * UNIT + R.T @ tvec.ravel()
                direct_init[i] = True
    if not direct_init.any():
        raise ValueError('No source body provides a positive-depth numerical initialization')
    # Missing roots receive a numerical temporal initialization only; the final
    # spline will be fitted to original observations, never these interpolants.
    for j in range(3):
        initial[:, j] = np.interp(times, times[direct_init], initial[direct_init, j])
    packet = dict(actual_pts=actual, source_native_pts=native[source_rows],
        camera25=cam, shot_index=obs['shot_index'][canonical_rows],
        local_COCO17_m=local, local_native22_m=local_native,
        body_prior_native_row=body_rows, body_prior_native_pts=native[body_rows],
        body_local_pose_donor_actual_row=donor_actual_row,
        body_local_pose_source_portrait=portrait,
        body_geometry_source_fit_enabled=~opening_inferred,
        body_opening_temporal_inference_only=opening_inferred,
        opening_reliable_upper_body_source_pts=np.asarray(OPENING_RELIABLE_UPPER_BODY_SOURCE_PTS),
        opening_near_static_user_source_end_pts=np.asarray(OPENING_FIRST_FULL_BODY_SOURCE_PTS),
        body_local_pose_inferred=inferred_pose,
        body_local_pose_origin=prior['prior_origin'][body_rows],
        raw_root_prior_Q_world=Q_prior, raw_root_left_source_pts=left,
        raw_root_right_source_pts=right, raw_root_alpha=alpha,
        source_COCO17_native_uv=source_uv, source_COCO17_score=source_score,
        source_COCO17_qualified=source_qualified,
        source_original_observation_mask=original_mask[source_rows],
        source_original_truncated=original_truncated[source_rows],
        source_opening_reviewed_hip_only_override=opening_override,
        source_bbox_native_xyxy=obs['bbox_native_xyxy'][source_rows],
        source_hip_proxy_valid=hip['source_hip_proxy_candidate_valid'][canonical_rows],
        source_hip_status=hip['source_hip_status'][canonical_rows],
        initial_P_m=initial, initial_source_pnp_valid=direct_init,
        fit_time_seconds=times, edge_dt_seconds=dt, edge_time_origin=origins,
        native_pts=native, native_exposure_representative_pts=representative,
        native_actual_row=np.searchsorted(actual, representative),
        metres_per_scene_unit=np.asarray(UNIT), body_scale=np.asarray(BODY_SCALE))
    output.mkdir(exist_ok=True, parents=True)
    path = output/'fullprogramme_body_source_input.npz'
    np.savez_compressed(path, **packet)
    _json(output/'original_source_clock_rows.json', clock_rows)
    metadata = dict(schema='rinkspace.fullprogramme.saved-body-source-input.v1',
        input_bindings=[dict(path=str(path0), sha256=_digest(path0))
                        for path0 in [prior_path, obs_path, camera_path, hip_path, mapping_path]],
        raw_root_rotation_bindings=raw_bindings, output=dict(path=str(path),sha256=_digest(path)),
        local_geometry_153cm_scaled_once=True, old_body_world_R_or_XYZ_read=False,
        source_UV_not_generated_or_filled=True, source_unknown_temporal_inference_allowed=True,
        initial_pnp_is_only_numerical_warm=True, original_body_camera_K_condition_retained=True,
        original_observation_mask_policy_reused=True,
        reviewed_opening_override_four_actual_hips_only=True,
        no_original_bottom_truncated_knee_or_ankle_revived=True,
        portrait_whole_body_and_raw_root_direction_inferred_from_same_cut_visible_body=True,
        portrait_pose_inferred_actual_count=int(portrait.sum()),
        all_body_local_samples_use_the_same_donor=True,
        opening_reliable_upper_body_source_pts=OPENING_RELIABLE_UPPER_BODY_SOURCE_PTS,
        opening_near_static_user_source_end_pts=OPENING_FIRST_FULL_BODY_SOURCE_PTS,
        reliable_opening_source_review='Head, shoulders, chest/abdomen and clothed hip visible; upright coarse torso; cropped soles are irrelevant',
        opening_unreliable_fixed_body_geometry_disabled_but_source_UV_and_original_masks_preserved=True,
        opening_position_temporally_inferred_from_later_reliable_source_in_same_world_spline=True,
        equipment_floor_contact_factors=False, accuracy_accepted=False)
    _json(output/'input_binding.json', metadata)
    return packet


def _basis(times, spacing):
    first, last = float(times[0]), float(times[-1])
    if last <= first:
        raise ValueError('A body trajectory requires increasing actual times')
    breaks = np.r_[np.arange(first, last, spacing), last]
    knots = np.r_[np.repeat(first, 3), breaks, np.repeat(last, 3)]
    return BSpline.design_matrix(times, knots, 3).tocsr(), knots


def _rotation_basis(times, spacing):
    """Spread cut-local breaks evenly so a clipped tail is not a tiny cell.

    The position basis remains global. A nominal 100 ms rotation interval
    previously could end in a 16.7 ms cell with an independently clamped last
    control. The index-space curvature prior did not account for that shorter
    time support. Equal cut-local intervals keep the requested upper spacing
    and remove that exceptional endpoint freedom without changing local pose.
    """
    times = np.asarray(times, float)
    first, last = float(times[0]), float(times[-1])
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('Rotation fit times must be finite and nondecreasing')
    if not np.isfinite(spacing) or spacing <= 0 or last <= first:
        raise ValueError('Rotation basis requires positive spacing and cut duration')
    intervals = max(1, int(np.ceil((last-first)/spacing)))
    breaks = np.linspace(first, last, intervals+1)
    knots = np.r_[np.repeat(first, 3), breaks, np.repeat(last, 3)]
    return BSpline.design_matrix(times, knots, 3).tocsr(), knots


def apply_body_fit_clock_evidence_overlay(packet, overlay_path, camera_path):
    """Change only root-fit elapsed times from explicitly bound source evidence.

    Camera/native aliases and all saved pose/source rows remain intact. Equal
    fit times represent a confirmed duplicate which the historical camera
    ledger retained as two canonical rows. Original fit-clock arrays are kept.
    """
    path = Path(overlay_path)
    spec = json.loads(path.read_text(encoding='utf-8-sig'))
    if spec.get('schema') != 'rinkspace.body-fit-clock-source-evidence-overlay.v1':
        raise ValueError('Unknown body fit-clock evidence schema')
    current_camera_sha = _digest(camera_path)
    camera_ledger_rebound = spec['camera_sha256'] != current_camera_sha
    if camera_ledger_rebound:
        reference = spec.get('camera_ledger_reference')
        if not reference or reference['sha256'] != spec['camera_sha256'] or _digest(reference['path']) != reference['sha256']:
            raise ValueError('Body fit-clock overlay belongs to a different camera without a bound source ledger')
        keys = ['pts', 'exposure_representative_pts', 'metadata_json']
        old, current = _load(reference['path'], keys), _load(camera_path, keys)
        for key in keys[:2]:
            if not np.array_equal(old[key], current[key]):
                raise ValueError('Replacement camera changes the pixel-evidence native exposure ledger')
        old_meta, current_meta = (json.loads(str(value['metadata_json'])) for value in (old, current))
        for key in ['source_sha256', 'source_time_base']:
            if old_meta[key] != current_meta[key]:
                raise ValueError('Replacement camera changes the pixel-evidence video or source clock')
        if not np.array_equal(packet['native_pts'], current['pts']) or not np.array_equal(
                packet['native_exposure_representative_pts'], current['exposure_representative_pts']):
            raise ValueError('Current body input changes the evidence native exposure ledger')
    for evidence in spec['evidence_bindings']:
        if _digest(evidence['path']) != evidence['sha256']:
            raise ValueError('Body fit-clock evidence changed: '+evidence['path'])
    if 'body_fit_clock_original_edge_dt_seconds' in packet:
        raise ValueError('Do not apply a body fit-clock overlay twice')
    dt = np.asarray(packet['edge_dt_seconds'], float).copy()
    origins = np.asarray(packet['edge_time_origin'], str).astype('U160')
    actual = packet['actual_pts']
    changed = []
    for update in spec['edge_updates']:
        left, right = map(int, update['camera_canonical_native_pts'])
        row = int(np.searchsorted(actual, left))
        if row+1 >= len(actual) or int(actual[row]) != left or int(actual[row+1]) != right:
            raise ValueError('Clock evidence edge is not a current adjacent canonical pair')
        if not np.isclose(dt[row], update['original_dt_seconds'], rtol=0., atol=1e-12):
            raise ValueError('Original body edge time differs from the bound evidence')
        official_left, official_right = map(int, update['source_official_pts'])
        replacement = (official_right-official_left)/12800.
        if replacement < 0 or official_left < 0 or official_right < 0:
            raise ValueError('Body evidence needs ordered nonnegative official times')
        if replacement == 0 and not update.get('same_source_exposure_verified', False):
            raise ValueError('Zero body edge time needs verified identical exposure evidence')
        if not update.get('source_pixel_correspondence_verified', False):
            raise ValueError('Body clock change requires actual source pixel correspondence')
        dt[row] = replacement
        origins[row] = ('source_verified_official50_same_exposure_body_clock_overlay' if replacement == 0
                        else 'source_verified_official50_distinct_exposure_body_clock_overlay')
        changed.append(row)
    if len(set(changed)) != len(changed) or not changed:
        raise ValueError('Body fit-clock edge updates must be unique and nonempty')
    result = dict(packet)
    result['body_fit_clock_original_fit_time_seconds'] = packet['fit_time_seconds'].copy()
    result['body_fit_clock_original_edge_dt_seconds'] = packet['edge_dt_seconds'].copy()
    result['body_fit_clock_original_edge_time_origin'] = packet['edge_time_origin'].copy()
    result['body_fit_clock_evidence_overlay_edge_rows'] = np.asarray(changed, np.int64)
    result['body_fit_clock_evidence_overlay_path'] = np.asarray(str(path))
    result['body_fit_clock_evidence_overlay_sha256'] = np.asarray(_digest(path))
    result['edge_dt_seconds'] = dt
    result['edge_time_origin'] = origins
    result['fit_time_seconds'] = np.r_[0., np.cumsum(dt)]
    binding = dict(path=str(path), sha256=_digest(path),
        current_camera_sha256=current_camera_sha,
        original_evidence_camera_sha256=spec['camera_sha256'],
        evidence_reused_only_with_identical_source_native_alias_ledger=camera_ledger_rebound,
        historical_camera_canonical_rows_retained=len(actual),
        native_presentations_retained=len(packet['native_pts']),
        distinct_body_fit_exposure_times=int(len(np.unique(result['fit_time_seconds']))),
        changed_edge_rows=changed,
        changed_edges=spec['edge_updates'], original_clock_arrays_retained=True,
        camera_aliases_source_UV_local_pose_and_official_pose_patch_unchanged=True)
    return result, binding


def _difference_matrix(count):
    if count < 3:
        return csr_matrix((0, count))
    rows = np.repeat(np.arange(count - 2), 3)
    columns = (np.arange(count - 2)[:, None] + np.arange(3)).ravel()
    return coo_matrix((np.tile([1., -2., 1.], count - 2), (rows, columns)),
                      shape=(count - 2, count)).tocsr()


def _second_derivative_basis(times, knots):
    """Exact cubic B-spline d²/dt² basis, using physical fit-clock seconds."""
    count = len(knots)-4
    first = np.arange(count-1)
    gain1 = 3./(knots[first+4]-knots[first+1])
    d1 = coo_matrix((np.c_[-gain1,gain1].ravel(),
        (np.repeat(first,2),np.c_[first,first+1].ravel())),shape=(count-1,count)).tocsr()
    second = np.arange(count-2)
    gain2 = 2./(knots[second+4]-knots[second+2])
    d2 = coo_matrix((np.c_[-gain2,gain2].ravel(),
        (np.repeat(second,2),np.c_[second,second+1].ravel())),shape=(count-2,count-1)).tocsr()
    return (BSpline.design_matrix(times,knots[2:-2],1).tocsr() @ d2 @ d1).tocsr()


def _first_derivative_basis(times, knots):
    count = len(knots)-4
    first = np.arange(count-1)
    gain = 3./(knots[first+4]-knots[first+1])
    d = coo_matrix((np.c_[-gain,gain].ravel(),
        (np.repeat(first,2),np.c_[first,first+1].ravel())),shape=(count-1,count)).tocsr()
    return (BSpline.design_matrix(times,knots[1:-1],2).tocsr() @ d).tocsr()


class FullProgrammeSplineProblem:
    """Source pixels/envelope directly act on shared XYZ/rotation controls."""

    def __init__(self, packet, *, spacing_s=.1, envelope=None,
                 envelope_local_points_m=None, envelope_local_valid=None,
                 source_factor_gain=None,
                 source_fit_rejections=None,
                 acceleration_sigma_m_s2=(15.,15.,25.),
                 motion_reference_interval_s=.02,
                 opening_planar_velocity_sigma_m_s=.5,
                 opening_vertical_velocity_sigma_m_s=.2,
                 opening_planar_binding='soft-velocity',
                 opening_source_support='conditional-upper-body',
                 rotation_control_curvature_rad=np.deg2rad(15.),
                 rotation_correction_prior_rad=np.deg2rad(30.)):
        self.p = packet
        self.n = len(packet['actual_pts'])
        self.spacing = float(spacing_s)
        self.local = packet['local_COCO17_m']
        self.source = packet['source_COCO17_native_uv']
        self.geometry_enabled = packet['body_geometry_source_fit_enabled']
        point_fit_enabled = packet.get('body_source_point_fit_enabled',
                                      np.ones((self.n,17),bool))
        if np.asarray(point_fit_enabled).shape != (self.n,17):
            raise ValueError('Per-point body geometry authority differs from source clock')
        if opening_source_support not in ('conditional-upper-body','temporal-only'):
            raise ValueError('Unknown opening source-support policy')
        self.opening_source_support = opening_source_support
        self.conditional_upper_body_rows = (
            (packet['shot_index']==1) & packet['body_opening_temporal_inference_only']
            & (packet['actual_pts']<packet['opening_near_static_user_source_end_pts'])
            if opening_source_support=='conditional-upper-body' else np.zeros(self.n,bool))
        # Local geometry remains an inferred donor. Preserve real source UV and
        # its qualification: the visible head/shoulders can condition a coarse
        # root even though that donor is not an independent pose measurement.
        # Hips/legs and the old whole-person envelope retain their original gate.
        geometry_authority=np.broadcast_to(self.geometry_enabled[:,None],(self.n,17)).copy()
        geometry_authority[self.conditional_upper_body_rows,:7]=True
        self.effective_point_fit_enabled=geometry_authority & point_fit_enabled
        if source_fit_rejections is not None:
            exclusions = np.asarray(source_fit_rejections, bool)
            if exclusions.shape != (self.n,17):
                raise ValueError('Semantic source fit mask differs from the source clock')
            self.effective_point_fit_enabled &= ~exclusions
        self.qualified = (packet['source_COCO17_qualified'][:, SOURCE_INDICES]
            & self.effective_point_fit_enabled[:,SOURCE_INDICES])
        # Optional, explicit source-family covariance condition. Original UV,
        # candidate qualification and pixel error scales remain unchanged.
        # Positive gains cannot silently suppress a qualified observation.
        self.source_factor_gain = (np.ones(self.qualified.shape, float)
            if source_factor_gain is None else np.asarray(source_factor_gain, float))
        if (self.source_factor_gain.shape != self.qualified.shape
                or not np.isfinite(self.source_factor_gain).all()
                or np.any(self.source_factor_gain <= 0)):
            raise ValueError('Source factor gains must be explicit positive N-by-source values')
        self.qprior = packet['raw_root_prior_Q_world']
        self.cameras = packet['camera25']
        self.metres_per_scene_unit = packet_metres_per_scene_unit(packet)
        self.R = np.asarray([cv2.Rodrigues(c[3:6])[0] for c in self.cameras])
        self.C = self.cameras[:, :3] * self.metres_per_scene_unit
        self.K = np.asarray([[[np.exp(c[6]), 0., c[8]],
                              [0., np.exp(c[7]), c[9]], [0., 0., 1.]] for c in self.cameras])
        self.B, self.position_knots = _basis(packet['fit_time_seconds'], self.spacing)
        self.m = self.B.shape[1]
        self.bcols = self.B.indices.reshape(self.n, 4)
        self.bvals = self.B.data.reshape(self.n, 4)
        self.position_initial_regularizer = _difference_matrix(self.m)
        self.acceleration_basis = _second_derivative_basis(packet['fit_time_seconds'],self.position_knots)
        dt = np.diff(packet['fit_time_seconds'])
        exposure_weight_s = np.r_[dt[0]/2.,(dt[:-1]+dt[1:])/2.,dt[-1]/2.]
        self.motion_reference_interval_s = float(motion_reference_interval_s)
        self.acceleration_sigma_m_s2 = np.asarray(acceleration_sigma_m_s2,float)
        if self.acceleration_sigma_m_s2.shape != (3,) or np.any(self.acceleration_sigma_m_s2<=0):
            raise ValueError('Three positive metre/second² engineering acceleration scales required')
        self.position_regularizer = self.acceleration_basis.multiply(
            np.sqrt(exposure_weight_s/self.motion_reference_interval_s)[:,None]).tocsr()
        opening = ((packet['shot_index']==1)
                   & (packet['actual_pts']<=packet['opening_near_static_user_source_end_pts']))
        self.opening_stationary_actual_rows = np.flatnonzero(opening)
        # Approximately in-place describes a weak motion condition. It does
        # not require the pelvis XY to be exactly constant while the body turns.
        # Retain the historical chart only for explicit replay of frozen runs.
        if opening_planar_binding not in ('soft-velocity','shared-xy'):
            raise ValueError('Unknown opening planar binding')
        self.opening_planar_binding = opening_planar_binding
        self.shared_opening_xy_control_indices = (
            np.unique(self.bcols[opening]) if opening_planar_binding=='shared-xy'
            else np.asarray([],dtype=int))
        self.shared_opening_xy_control_mask = np.zeros(self.m,bool)
        self.shared_opening_xy_control_mask[self.shared_opening_xy_control_indices] = True
        self.opening_planar_velocity_sigma_m_s = float(opening_planar_velocity_sigma_m_s)
        self.opening_velocity_sigma_m_s = np.asarray([opening_planar_velocity_sigma_m_s,
            opening_planar_velocity_sigma_m_s,opening_vertical_velocity_sigma_m_s],float)
        self.opening_velocity_derivative_basis = _first_derivative_basis(
            packet['fit_time_seconds'][opening],self.position_knots)
        self.opening_velocity_basis = self.opening_velocity_derivative_basis.multiply(
                np.sqrt(exposure_weight_s[opening]/self.motion_reference_interval_s)[:,None]).tocsr()
        rotation_rows, rotation_cols, rotation_vals = [], [], []
        rot_reg_rows, rot_reg_cols, rot_reg_vals = [], [], []
        self.rotation_knots, self.rotation_cut_control_offsets = [], [0]
        qcount, regcount = 0, 0
        for shot in np.unique(packet['shot_index']):
            frames = np.flatnonzero(packet['shot_index'] == shot)
            b, knots = _rotation_basis(packet['fit_time_seconds'][frames], self.spacing)
            coo = b.tocoo()
            rotation_rows.extend(frames[coo.row]); rotation_cols.extend(coo.col + qcount)
            rotation_vals.extend(coo.data)
            d = _difference_matrix(b.shape[1]).tocoo()
            rot_reg_rows.extend(d.row + regcount); rot_reg_cols.extend(d.col + qcount)
            rot_reg_vals.extend(d.data)
            regcount += d.shape[0]; qcount += b.shape[1]
            self.rotation_knots.append(knots)
            self.rotation_cut_control_offsets.append(qcount)
        self.Bq = coo_matrix((rotation_vals, (rotation_rows, rotation_cols)),shape=(self.n,qcount)).tocsr()
        self.qm = qcount
        self.qcols = self.Bq.indices.reshape(self.n, 4)
        self.qvals = self.Bq.data.reshape(self.n, 4)
        self.rotation_regularizer = coo_matrix((rot_reg_vals,(rot_reg_rows,rot_reg_cols)),
                                               shape=(regcount,qcount)).tocsr()
        self.rc = float(rotation_control_curvature_rad)
        self.rprior = float(rotation_correction_prior_rad)
        self.envelope = envelope
        self.env_local = envelope_local_points_m
        self.env_valid = (np.ones(self.n, bool) if envelope_local_valid is None
                          else np.asarray(envelope_local_valid, bool))
        self.envelope_on = envelope is not None and envelope_local_points_m is not None
        if self.envelope_on:
            if (len(self.env_local) != self.n or self.env_valid.shape != (self.n,)
                    or not np.isfinite(self.env_local[self.env_valid]).all()):
                raise ValueError('Same-body envelope samples differ from actual clock')
        init_matrix = vstack([self.B, self.position_initial_regularizer * .1]).tocsr()
        initial_controls = np.asarray([lsqr(init_matrix,
            np.r_[packet['initial_P_m'][:, j], np.zeros(self.position_initial_regularizer.shape[0])],
            atol=1e-7, btol=1e-7)[0] for j in range(3)]).T
        self.x0 = np.r_[initial_controls.ravel(), np.zeros(qcount * 3)]
        self._cached_x = None
        self._cached = None

    def evaluate(self, x):
        x = np.asarray(x, float)
        if self._cached_x is not None and np.array_equal(x, self._cached_x):
            return self._cached
        position_controls = self.position_controls_from_x(x)
        rotation_controls = x[self.m * 3:].reshape(self.qm, 3)
        P = self.B @ position_controls
        delta = self.Bq @ rotation_controls
        residuals = []
        rows, columns, values = [], [], []
        alluv = np.empty((self.n, 17, 2))
        Q = np.empty((self.n, 3, 3))
        source_residual = np.zeros((self.n, len(SOURCE_INDICES), 2))
        envelope_residual = np.zeros((self.n, 8))
        camera_min_depth = np.empty(self.n)
        cursor = 0

        def block_projection_jac(frame, local_points, dQ, q):
            world = local_points @ q.T + P[frame]
            campts = (world-self.C[frame]) @ self.R[frame].T
            uv, jp = cv2.projectPoints(campts,np.zeros(3),np.zeros(3),
                                      self.K[frame],self.cameras[frame,11:25])
            pixel_cam = jp[:,3:6].reshape(-1,2,3)
            pixel_world = np.einsum('npk,kj->npj',pixel_cam,self.R[frame])
            dw = np.einsum('rjk,nk->njr',dQ,local_points)
            rotation_jac = np.einsum('npj,njr->npr',pixel_world,dw)
            return uv.reshape(-1,2), pixel_world, rotation_jac, campts

        def add_root_block(frame, jp, jq, row0):
            # Matrices have k residual rows x3 metre/rotation derivatives.
            count = len(jp)
            rr = np.repeat(np.arange(row0,row0+count),12)
            cp = (self.bcols[frame,:,None]*3+np.arange(3)).ravel()
            cp = cp.reshape(4,3)
            cp[self.shared_opening_xy_control_mask[self.bcols[frame]],:2] = np.arange(2)
            cp = cp.ravel()
            cq = (self.qcols[frame,:,None]*3+np.arange(3)+self.m*3).ravel()
            rows.extend([rr,rr]); columns.extend([np.tile(cp,count),np.tile(cq,count)])
            values.extend([(jp[:,None,:]*self.bvals[frame,None,:,None]).ravel(),
                           (jq[:,None,:]*self.qvals[frame,None,:,None]).ravel()])

        for i in range(self.n):
            correction, dcorrection = cv2.Rodrigues(delta[i])
            Q[i] = correction @ self.qprior[i]
            dq = np.einsum('rjk,kl->rjl',dcorrection.reshape(3,3,3),self.qprior[i])
            uv, jp, jq, campts = block_projection_jac(i,self.local[i],dq,Q[i])
            alluv[i] = uv
            weight = self.qualified[i].astype(float) * self.source_factor_gain[i] / SOURCE_SIGMA
            observed = np.where(self.qualified[i,:,None], self.source[i,SOURCE_INDICES], uv[SOURCE_INDICES])
            source_residual[i] = (uv[SOURCE_INDICES]-observed)*weight[:,None]
            residuals.append(source_residual[i].ravel())
            source_dimension = 2*len(SOURCE_INDICES)
            add_root_block(i,(jp[SOURCE_INDICES]*weight[:,None,None]).reshape(source_dimension,3),
                           (jq[SOURCE_INDICES]*weight[:,None,None]).reshape(source_dimension,3),cursor)
            cursor += source_dimension
            if self.envelope_on and self.env_valid[i] and self.geometry_enabled[i]:
                points = self.env_local[i]
                env_world = points @ Q[i].T + P[i]
                env_cam = (env_world-self.C[i]) @ self.R[i].T
                env_uv = cv2.projectPoints(env_cam,np.zeros(3),np.zeros(3),self.K[i],
                                           self.cameras[i,11:25])[0].reshape(-1,2)
                ev = envelope_pixel_factor(env_uv,self.envelope['source_support_native_px'][i],
                    self.envelope['source_envelope_qualified'][i],
                    tolerance_px=self.envelope['source_tolerance_native_px'][i],
                    sigma_px=self.envelope['source_sigma_native_px'][i],
                    lower_bound_only=(self.envelope['source_envelope_lower_bound_only'][i]
                        if 'source_envelope_lower_bound_only' in self.envelope else False))
                selected = ev['support_vertex_index']
                _, ejp, ejq, _ = block_projection_jac(i,points[selected],dq,Q[i])
                gain = ev['residual_derivative_gain']
                ejp = np.einsum('kp,kpj->kj',DIRECTIONS,ejp)*gain[:,None]
                ejq = np.einsum('kp,kpj->kj',DIRECTIONS,ejq)*gain[:,None]
                envelope_residual[i] = ev['residual']
                residuals.append(ev['residual'])
                add_root_block(i,ejp,ejq,cursor); cursor += 8
                domain_points, domain_cam = points, env_cam
            else:
                domain_points, domain_cam = self.local[i], campts
            # Optical-domain condition, unrelated to modeled ice or contact.
            low = int(np.argmin(domain_cam[:,2])); camera_min_depth[i] = domain_cam[low,2]
            depth_residual = min(0., camera_min_depth[i]-.02)/.05
            residuals.append(np.asarray([depth_residual]))
            if depth_residual < 0:
                dp = self.R[i,2:3]/.05
                dr = (self.R[i,2] @ np.einsum('rjk,k->jr',dq,domain_points[low]))[None,:]/.05
            else:
                dp=dr=np.zeros((1,3))
            add_root_block(i,dp,dr,cursor); cursor += 1
        for controls, d, scale, offset in [(position_controls,self.position_regularizer,self.acceleration_sigma_m_s2,0),
            (rotation_controls,self.rotation_regularizer,self.rc,self.m*3)]:
            residuals.append((d @ controls/scale).ravel())
            coo = d.tocoo()
            rr = (coo.row[:,None]*3+np.arange(3)+cursor).ravel()
            cc = (coo.col[:,None]*3+np.arange(3)+offset).ravel()
            if offset==0:
                cc=cc.reshape(-1,3)
                cc[self.shared_opening_xy_control_mask[coo.col],:2]=np.arange(2)
                cc=cc.ravel()
            vv = (coo.data[:,None]/np.broadcast_to(scale,(3,))).ravel()
            rows.append(rr);columns.append(cc);values.append(vv)
            cursor += d.shape[0]*3
        # The user's source review says the entire opening before the body is
        # fully visible is approximately in place while the lens pulls back.
        # Vertical motion remains a soft condition;
        # never a fixed world point or a measured/locked ground trajectory.
        d = self.opening_velocity_basis
        residuals.append((d @ position_controls/self.opening_velocity_sigma_m_s).ravel())
        coo = d.tocoo()
        rows.append((coo.row[:,None]*3+np.arange(3)+cursor).ravel())
        cc=coo.col[:,None]*3+np.arange(3)
        cc[self.shared_opening_xy_control_mask[coo.col],:2]=np.arange(2)
        columns.append(cc.ravel())
        values.append((coo.data[:,None]/self.opening_velocity_sigma_m_s).ravel())
        cursor += d.shape[0]*3
        residuals.append((rotation_controls/self.rprior).ravel())
        rows.append(np.arange(cursor,cursor+self.qm*3));columns.append(np.arange(self.m*3,(self.m+self.qm)*3))
        values.append(np.full(self.qm*3,1./self.rprior));cursor += self.qm*3
        residual = np.concatenate(residuals)
        jacobian = coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),
                              shape=(cursor,len(x))).tocsr()
        self._cached_x=x.copy()
        self._cached=(residual,jacobian,P,Q,alluv,source_residual,envelope_residual,camera_min_depth)
        return self._cached

    def fun(self,x): return self.evaluate(x)[0]
    def jac(self,x): return self.evaluate(x)[1]

    def opening_supported_control_columns(self):
        """Free only spline controls whose actual support ends in the opening.

        Keep the same global position spline and per-cut rotation splines. A
        control touching any source node outside the existing opening interval
        stays fixed; there is no manually chosen control count or new window.
        """
        opening=np.zeros(self.n,bool)
        opening[self.opening_stationary_actual_rows]=True
        outside=~opening
        interval=self.p['fit_time_seconds'][opening][[0,-1]]
        qsupport=np.concatenate([np.column_stack((knots[:-4],knots[4:]))
            for knots in self.rotation_knots])
        psupport=np.column_stack((self.position_knots[:-4],self.position_knots[4:]))
        selected=[]
        for basis,support in ((self.B,psupport),(self.Bq,qsupport)):
            inside_basis=basis[opening].copy();inside_basis.eliminate_zeros()
            outside_basis=basis[outside].copy();outside_basis.eliminate_zeros()
            inside_support=np.diff(inside_basis.tocsc().indptr)>0
            outside_support=np.diff(outside_basis.tocsc().indptr)>0
            fully_inside=(support[:,0]>=interval[0]) & (support[:,1]<=interval[1])
            selected.append(np.flatnonzero(inside_support & ~outside_support & fully_inside))
        pcontrols,qcontrols=selected
        # The opening mask belongs to cut1, so no other cut's rotation control
        # can have nonzero inside support. Check that ownership explicitly.
        if len(qcontrols) and np.any(qcontrols>=self.rotation_cut_control_offsets[1]):
            raise ValueError('Opening rotation controls belong to another cut')
        columns=np.r_[(pcontrols[:,None]*3+np.arange(3)).ravel(),
            (qcontrols[:,None]*3+np.arange(3)+self.m*3).ravel()]
        if not len(columns):
            raise ValueError('No spline controls have support confined to the opening')
        return columns,dict(
            rule='Complete cubic knot support within existing opening interval; nonzero inside and zero at every outside-opening source node',
            opening_fit_time_interval_s=interval.tolist(),
            opening_actual_rows=self.opening_stationary_actual_rows.tolist(),
            opening_actual_pts=self.p['actual_pts'][opening].tolist(),
            position_control_indices=pcontrols.tolist(),
            rotation_control_indices=qcontrols.tolist(),
            position_control_support_intervals_s=psupport[pcontrols].tolist(),
            rotation_control_support_intervals_s=qsupport[qcontrols].tolist(),
            free_columns=columns.tolist(),
            fixed_columns=int((self.m+self.qm)*3-len(columns)),
            same_global_position_spline=True,
            outside_opening_basis_support_exactly_zero=True)

    def position_controls_from_x(self,x):
        controls=np.asarray(x[:self.m*3]).reshape(self.m,3).copy()
        controls[self.shared_opening_xy_control_indices,:2]=controls[0,:2]
        return controls

    def infer_unobserved_position_control_warm_start(self):
        """Conditionally minimize the same motion objective before joint fitting.

        These columns touch no usable body-source frame. An old failed opening
        trajectory is therefore not a valid numerical initializer: source-heavy
        global stopping criteria can leave its small motion cost unfinished.
        Solve only its quadratic motion block with later source-bearing controls
        as the numerical boundary, then allow all controls to move in the fit.
        No source pixels, fitted world positions, knots or output G are filtered.
        """
        observed = self.qualified.any(axis=1)
        if self.envelope_on:
            observed |= (self.geometry_enabled & self.env_valid
                         & self.envelope['source_envelope_qualified'].any(axis=1))
        free = np.flatnonzero(np.diff(self.B[observed].tocsc().indptr)==0)
        fixed = np.setdiff1d(np.arange(self.m),free)
        p = self.x0[:self.m*3].reshape(self.m,3).copy()
        before = p.copy()
        if self.opening_planar_binding=='shared-xy':
            boundary=self.B[[self.opening_stationary_actual_rows[-1]]] @ p
            p[self.shared_opening_xy_control_indices,:2]=np.asarray(boundary).ravel()[:2]
        # Source-bearing controls stay free in the joint fit. The unobserved
        # controls use the existing soft motion objective for initialization.
        temporal_axes=[2] if self.opening_planar_binding=='shared-xy' else [0,1,2]
        for j in temporal_axes:
            a = vstack([self.position_regularizer/self.acceleration_sigma_m_s2[j],
                self.opening_velocity_basis/self.opening_velocity_sigma_m_s[j]]).tocsr()
            if len(free):
                target = -(a[:,fixed] @ p[fixed,j])
                p[free,j] = lsqr(a[:,free],target,atol=1e-12,btol=1e-12,iter_lim=1000)[0]
        self.x0[:self.m*3] = p.ravel()
        return dict(vertical_control_indices=free.tolist(),
                    shared_unknown_opening_xy_control_indices=self.shared_opening_xy_control_indices.tolist(),
                    opening_planar_binding=self.opening_planar_binding,
                    shared_unknown_opening_xy_initialized_from_later_source_numerical_boundary=self.opening_planar_binding=='shared-xy',
                    source_bearing_vertical_controls_not_changed=True,
                    position_control_max_shift_m=float(np.linalg.norm(p-before,axis=1).max()),
                    purpose='Exact conditional minimization of the existing inside-fit temporal objective; numerical warm start only')


def apply_reviewed_whole_pose_inference(packet, inference_packet_path, body_samples_path):
    """Overlay reviewed local geometry after fresh current-camera preparation.

    Camera, raw-root direction, source UV/score/qualification and both clocks
    come only from the freshly prepared packet. The overlay carries fixed local
    whole-body inference and its explicit per-point geometry authority.
    """
    previous = _load(inference_packet_path)
    for key in ['actual_pts','native_pts','native_exposure_representative_pts',
                'source_native_pts','source_COCO17_native_uv','source_COCO17_score',
                'source_COCO17_qualified','source_original_observation_mask',
                'fit_time_seconds','edge_dt_seconds','edge_time_origin','shot_index']:
        equal = (np.array_equal(packet[key],previous[key],equal_nan=True)
            if np.asarray(packet[key]).dtype.kind in 'fc' else
            np.array_equal(packet[key],previous[key]))
        if not equal:
            raise ValueError('Reviewed whole-pose overlay changed original source/clock: '+key)
    inferred = np.asarray(previous['body_local_pose_inference_rows'],int)
    anchor_pts = previous['body_local_pose_inference_anchor_actual_pts']
    anchors = np.searchsorted(packet['actual_pts'],anchor_pts)
    if np.any(anchors>=len(packet['actual_pts'])) or not np.array_equal(packet['actual_pts'][anchors],anchor_pts):
        raise ValueError('Reviewed whole-pose anchors are absent from current camera/source clock')
    rows = np.unique(np.r_[inferred,anchors])
    if len(np.unique(packet['shot_index'][rows])) != 1 or np.any(packet['body_opening_temporal_inference_only'][rows]):
        raise ValueError('Whole-pose overlay may not cross a cut or revisit opening')
    samples = _load(body_samples_path,['actual_pts','local_pose_origin','body_scale'])
    if not np.array_equal(samples['actual_pts'],packet['actual_pts']) or not np.isclose(float(samples['body_scale']),BODY_SCALE,rtol=0,atol=1e-12):
        raise ValueError('Whole-body samples do not match source clock or single body scale')
    if not np.array_equal(samples['local_pose_origin'][inferred],previous['body_local_pose_origin'][inferred]):
        raise ValueError('Body-envelope sample provenance does not match reviewed fixed articulation')
    current_prior = packet['raw_root_prior_Q_world'].copy()
    for key in ['local_COCO17_m','local_native22_m','body_local_pose_inferred',
                'body_local_pose_origin','body_local_pose_donor_actual_row']:
        if packet[key].dtype.kind in 'US':
            packet[key] = packet[key].astype('<U96')
        packet[key][rows] = previous[key][rows]
    packet['body_source_point_fit_enabled'] = np.ones((len(packet['actual_pts']),17),bool)
    packet['body_source_point_fit_enabled'][rows] = previous['body_source_point_fit_enabled'][rows]
    packet['body_local_pose_inference_rows'] = inferred.copy()
    packet['body_local_pose_inference_anchor_actual_pts'] = anchor_pts.copy()
    if not np.array_equal(packet['raw_root_prior_Q_world'],current_prior):
        raise ValueError('Whole-pose overlay must not import old world/root directions')
    return dict(path=str(inference_packet_path),sha256=_digest(inference_packet_path),
        body_samples_path=str(body_samples_path),body_samples_sha256=_digest(body_samples_path),
        inferred_actual_exposures=int(len(inferred)),anchor_actual_pts=anchor_pts.tolist(),
        local_complete_body_geometry_only=True,
        raw_root_prior_recomputed_from_current_camera_not_overridden=True,
        source_clock_and_source_pixels_exact=True,
        source_face_pixels_kept_diagnostic_in_explicit_inferred_window=True)


def apply_reviewed_whole_pose_samples(samples, packet, body_samples_path):
    """Layer the reviewed whole-pose window over clock-corrected samples."""
    reviewed = _load(body_samples_path)
    for key in ['actual_pts','mesh_original_vertex_index','shared_betas','body_scale']:
        if not np.array_equal(samples[key],reviewed[key]):
            raise ValueError('Reviewed body samples differ from saved body identity/clock: '+key)
    rows=np.unique(np.r_[packet['body_local_pose_inference_rows'],
        np.searchsorted(packet['actual_pts'],packet['body_local_pose_inference_anchor_actual_pts'])])
    for key in ['mesh_envelope_rootlocal_scaled_m','local_pose_valid','local_pose_origin',
                'body_pose_axis_angle']:
        if samples[key].shape[0]!=len(packet['actual_pts']) or samples[key].shape!=reviewed[key].shape:
            raise ValueError('Reviewed body sample geometry differs from actual clock: '+key)
        if samples[key].dtype.kind in 'US':
            samples[key]=samples[key].astype('<U96')
        samples[key][rows]=reviewed[key][rows]
    return samples


def check_current_camera_binding(packet, camera_path):
    current = _load(camera_path,['pts','camera25','exposure_representative_pts'])
    if (not np.array_equal(packet['native_pts'],current['pts']) or not np.array_equal(
            packet['native_exposure_representative_pts'],current['exposure_representative_pts'])):
        raise ValueError('Prepared body/current physical camera native clock or aliases differ')
    rows=np.searchsorted(current['pts'],packet['actual_pts'])
    if np.any(rows>=len(current['pts'])) or not np.array_equal(current['pts'][rows],packet['actual_pts']):
        raise ValueError('Prepared body actual clock differs from current physical camera')
    if not np.array_equal(packet['camera25'],current['camera25'][rows]):
        raise ValueError('Prepared body camera is stale: prepare a new output against current camera')


def check_warm_control_basis(problem, packet, warm):
    """Warm only the same spline parameters; never import an old camera/Q."""
    expected = dict(
        actual_pts=packet['actual_pts'], fit_time_seconds=packet['fit_time_seconds'],
        shot_index=packet['shot_index'], position_knots=problem.position_knots,
        rotation_cut_control_offsets=np.asarray(problem.rotation_cut_control_offsets),
        rotation_knots_concatenated=np.concatenate(problem.rotation_knots),
        rotation_knot_offsets=np.r_[0,np.cumsum([len(k) for k in problem.rotation_knots])])
    for key, value in expected.items():
        if key not in warm or not np.array_equal(warm[key],value):
            raise ValueError('Warm controls differ from the exact current spline basis: '+key)
    if (warm['position_controls_world_m'].shape!=(problem.m,3)
            or warm['root_rotation_correction_controls'].shape!=(problem.qm,3)):
        raise ValueError('Warm controls differ from the current trajectory basis')


def load_source_fit_rejections(path, packet, packet_path):
    """Exclude source-reviewed semantic mistakes without rewriting observations."""
    mask = np.zeros((len(packet['actual_pts']), 17), bool)
    if not path:
        return mask, None
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    if spec.get('source_packet_sha256') != _digest(packet_path):
        raise ValueError('Semantic source review belongs to a different body packet')
    if not spec.get('source_only_reviewed', False):
        raise ValueError('Semantic source rejection requires an actual source-only review')
    records = spec.get('rejected_points', [])
    if not records:
        raise ValueError('Source fit rejection overlay must identify actual observations')
    for record in records:
        pts, joint = int(record['actual_pts']), int(record['COCO_index'])
        row = int(np.searchsorted(packet['actual_pts'], pts))
        if (row >= len(mask) or int(packet['actual_pts'][row]) != pts
                or joint not in SOURCE_INDICES or mask[row, joint]):
            raise ValueError('Semantic source rejection is not a unique fitted observation')
        if int(record['source_native_pts']) != int(packet['source_native_pts'][row]):
            raise ValueError('Semantic source review uses a different source exposure')
        if not np.array_equal(np.asarray(record['original_uv'], float),
                              packet['source_COCO17_native_uv'][row, joint]):
            raise ValueError('Semantic source review does not preserve the original source UV')
        if record.get('decision') not in ('wrong-semantic-point', 'not-source-verifiable'):
            raise ValueError('Only source semantic invalidity can exclude an observation')
        mask[row, joint] = True
    binding = dict(path=str(path), sha256=_digest(path),
        source_packet_sha256=_digest(packet_path), rejected_observations=len(records),
        original_UV_scores_qualification_sigma_and_clock_unchanged=True,
        source_only_reviewed=True, purpose='Source semantic fit qualification only')
    return mask, binding


def run(args):
    output=Path(args.output); output.mkdir(exist_ok=True,parents=True)
    packet_path=output/'fullprogramme_body_source_input.npz'
    if packet_path.exists():
        p=_load(packet_path)
    else:
        print('Binding fullprogramme existing local body/source/current camera',flush=True)
        p=prepare_programme_input(args.base,args.camera,args.hips,args.clock,output)
    check_current_camera_binding(p,args.camera)
    sample=_load(args.body_samples) if args.body_samples else None
    official_clock_binding=None
    if getattr(args,'official_pose_clock_patch',None):
        if sample is None:
            raise ValueError('Official saved-pose clock correction requires original whole-body samples')
        from .official_body_pose_clock_patch import apply_official_pose_clock_patch
        p,sample,official_clock_binding=apply_official_pose_clock_patch(
            p,sample,args.official_pose_clock_patch,camera_path=args.camera)
        check_current_camera_binding(p,args.camera)
        _json(output/'official_saved_pose_clock_application.json',official_clock_binding)
    pose_inference_binding=None
    if getattr(args,'pose_inference_packet',None):
        overlay_samples=(getattr(args,'pose_inference_body_samples',None) or args.body_samples)
        if sample is None or not overlay_samples:
            raise ValueError('Reviewed whole-pose overlay requires its same-body samples')
        pose_inference_binding=apply_reviewed_whole_pose_inference(p,args.pose_inference_packet,overlay_samples)
        sample=apply_reviewed_whole_pose_samples(sample,p,overlay_samples)
        _json(output/'reviewed_whole_pose_inference_binding.json',pose_inference_binding)
    effective_samples_path=args.body_samples
    if official_clock_binding is not None or pose_inference_binding is not None:
        np.savez_compressed(packet_path,**p)
        effective_samples_path=output/'prepared_body_envelope_samples.npz'
        np.savez_compressed(effective_samples_path,**sample)
    envelope=None; samples=None; sample_valid=None
    if args.envelope and sample is not None:
        envelope=_load(args.envelope)
        if not np.array_equal(p['actual_pts'],envelope['actual_pts']) or not np.array_equal(p['actual_pts'],sample['actual_pts']):
            raise ValueError('Body/envelope actual clock differs from bound programme')
        donor = p['body_local_pose_donor_actual_row']
        samples=sample['mesh_envelope_rootlocal_scaled_m'][donor]
        sample_valid=sample['local_pose_valid'][donor]
    source_rejections, source_rejection_binding = load_source_fit_rejections(
        getattr(args, 'source_fit_rejections', None), p, packet_path)
    problem=FullProgrammeSplineProblem(p,spacing_s=args.spacing,envelope=envelope,
        envelope_local_points_m=samples,envelope_local_valid=sample_valid,
        opening_planar_binding=getattr(args,'opening_planar_binding','soft-velocity'),
        opening_source_support=getattr(args,'opening_source_support','conditional-upper-body'),
        source_fit_rejections=source_rejections)
    warm_binding=None
    opening_support_changed=False
    exact_warm = bool(getattr(args, 'warm_controls_exact_state', False))
    if exact_warm and not args.warm_controls:
        raise ValueError('Exact-state continuation requires existing warm controls')
    if args.warm_controls:
        warm_keys=['position_controls_world_m','root_rotation_correction_controls',
            'actual_pts','fit_time_seconds','shot_index','position_knots',
            'rotation_cut_control_offsets','rotation_knots_concatenated','rotation_knot_offsets']
        warm=_load(args.warm_controls,warm_keys)
        check_warm_control_basis(problem,p,warm)
        exact_bindings = None
        if exact_warm:
            warm_path = Path(args.warm_controls)
            warm_report = json.loads((warm_path.parent/'report.json').read_text(encoding='utf-8'))
            previous_support=warm_report.get('opening_source_support','temporal-only')
            opening_support_changed=previous_support!=problem.opening_source_support
            if opening_support_changed and not (
                    getattr(args,'allow_opening_source_support_change',False)
                    and previous_support=='temporal-only'
                    and problem.opening_source_support=='conditional-upper-body'):
                raise ValueError('Opening support change needs an explicit controlled restoration request')
            if (_digest(warm_path) != warm_report['output']['sha256']
                    or _digest(packet_path) != warm_report['input']['sha256']):
                raise ValueError('Exact continuation requires the identical saved source packet and controls')
            old_state = _load(warm_path, ['camera25', 'metres_per_scene_unit', 'raw_root_prior_Q_world'])
            if any(not np.array_equal(old_state[k], p[k]) for k in old_state):
                raise ValueError('Exact continuation cannot change camera, units or raw direction')
            for filename, key in [(args.envelope, 'source_envelope_binding'),
                                  (args.body_samples, 'complete_body_samples_binding')]:
                if not filename or _digest(filename) != warm_report[key]['sha256']:
                    raise ValueError('Exact continuation cannot change body surface or source-envelope objective')
            exact_bindings = dict(source_packet_sha256=_digest(packet_path),
                camera_sha256=_digest(args.camera),
                source_envelope_sha256=_digest(args.envelope),
                local_samples_sha256=_digest(args.body_samples),
                previous_opening_planar_binding=warm_report.get('opening_planar_binding',
                    'shared-xy' if warm_report.get('opening_planar_position_is_one_shared_fitted_variable_not_preset_world_XY') else 'soft-velocity'),
                current_opening_planar_binding=problem.opening_planar_binding,
                previous_opening_source_support=previous_support,
                current_opening_source_support=problem.opening_source_support,
                source_objective_changed_only_by_restored_original_upper7=opening_support_changed,
                 source_semantic_fit_rejection_binding=source_rejection_binding,
                identical_physical_initial_state_checked_by_residual_replay=True)
        warm_rotation = warm['root_rotation_correction_controls']
        warm_delta = problem.Bq @ warm_rotation
        # Changing an unreliable portrait's entire local pose/direction must
        # also change its numerical correction warm start. Retaining the old
        # portrait correction would transplant its flipped orientation basin.
        if p['body_local_pose_inferred'].any() and not exact_warm:
            donor = p['body_local_pose_donor_actual_row']
            warm_delta[p['body_local_pose_inferred']] = warm_delta[donor[p['body_local_pose_inferred']]]
            qinit = vstack([problem.Bq,problem.rotation_regularizer*.05]).tocsr()
            warm_rotation = np.asarray([lsqr(qinit,np.r_[warm_delta[:,j],
                np.zeros(problem.rotation_regularizer.shape[0])],atol=1e-7,btol=1e-7)[0]
                for j in range(3)]).T
        problem.x0=np.r_[warm['position_controls_world_m'].ravel(),warm_rotation.ravel()]
        warm_binding=dict(path=args.warm_controls,sha256=_digest(args.warm_controls),
                          purpose='Numerical initialization, not source/world measurement',
                          exact_current_spline_basis_checked=True,
                          exact_same_state_requested=exact_warm,
                          exact_same_state_objective_bindings=exact_bindings,
                          imported_fields=warm_keys,
                          old_camera_or_world_root_direction_not_imported=True,
                          inferred_whole_body_rotation_correction_uses_same_visible_donor=not exact_warm)
    print('Fitting fullprogramme shared continuous world root; source and camera fixed',flush=True)
    temporal_control_warm_binding=(dict(skipped_for_exact_same_state_continuation=True)
        if exact_warm else problem.infer_unobserved_position_control_warm_start())
    started=time.perf_counter()
    initial_objective_replay = None
    removed_semantic_source_cost = 0.
    if exact_warm:
        warm_evaluation=problem.evaluate(problem.x0)
        warm_residual=warm_evaluation[0]
        initial_objective_replay=float(warm_residual@warm_residual)
        expected=float(warm_report['total_weighted_cost'])
        added_upper_cost=(float(np.sum(warm_evaluation[5][problem.conditional_upper_body_rows,:7]**2))
                          if opening_support_changed else 0.)
        if source_rejection_binding is not None:
            previous_source = _load(args.warm_controls,
                ['body_effective_source_point_fit_enabled', 'source_residual_weighted'])
            previous_mask = previous_source['body_effective_source_point_fit_enabled'][:, SOURCE_INDICES]
            current_mask = problem.effective_point_fit_enabled[:, SOURCE_INDICES]
            if np.any(current_mask & ~previous_mask) and not opening_support_changed:
                raise ValueError('Semantic source rejection cannot promote a previously gated point')
            removed_semantic_source_cost = float(np.sum(
                previous_source['source_residual_weighted'][previous_mask & ~current_mask]**2))
            added_upper_cost = float(np.sum(warm_evaluation[5][current_mask & ~previous_mask]**2))
        if not np.isclose(initial_objective_replay+removed_semantic_source_cost-added_upper_cost,
                          expected,rtol=1e-12,atol=1e-7):
            raise ValueError('Exact warm controls do not replay the saved full objective')
        print(f'Exact same-state objective replay: {initial_objective_replay:.12g}',flush=True)
        if opening_support_changed:
            print(f'Original objective replay after removing restored upper-source terms: {initial_objective_replay-added_upper_cost:.12g}',flush=True)
        del warm_evaluation,warm_residual
    tolerances=dict(ftol=float(getattr(args,'ftol',1e-4)),
        xtol=float(getattr(args,'xtol',1e-6)),gtol=float(getattr(args,'gtol',1e-3)))
    if any(not np.isfinite(v) or v <= 0 for v in tolerances.values()):
        raise ValueError('Solver stopping tolerances must be finite and positive')
    control_scope=getattr(args,'fit_control_scope','full-programme')
    if control_scope not in ('full-programme','opening-supported'):
        raise ValueError('Unknown body control-fit scope')
    control_scope_binding=None
    if control_scope=='opening-supported':
        if not exact_warm or problem.opening_planar_binding!='soft-velocity':
            raise ValueError('Opening-only control fitting requires exact physical warm controls and soft XY')
        free_columns,control_scope_binding=problem.opening_supported_control_columns()
        fixed_initial=problem.x0.copy()
        initial_P_Q=problem.evaluate(fixed_initial)[2:4]
        initial_outside_P=initial_P_Q[0][np.setdiff1d(np.arange(problem.n),problem.opening_stationary_actual_rows)].copy()
        initial_outside_Q=initial_P_Q[1][np.setdiff1d(np.arange(problem.n),problem.opening_stationary_actual_rows)].copy()
        def expand_controls(reduced):
            x=fixed_initial.copy();x[free_columns]=reduced
            return x
        fit_fun=lambda reduced:problem.fun(expand_controls(reduced))
        fit_jac=lambda reduced:problem.jac(expand_controls(reduced))[:,free_columns]
        fit_initial=fixed_initial[free_columns]
    else:
        fit_fun,fit_jac,fit_initial=problem.fun,problem.jac,problem.x0
    fit=least_squares(fit_fun,fit_initial,jac=fit_jac,method='trf',tr_solver='lsmr',
        tr_options=dict(atol=1e-3,btol=1e-3,maxiter=100),x_scale='jac',
        max_nfev=args.max_nfev,**tolerances,verbose=2)
    if control_scope=='opening-supported':
        fit.x=expand_controls(fit.x)
    residual,jac,P,Q,uv,source_residual,envelope_residual,min_depth=problem.evaluate(fit.x)
    if control_scope=='opening-supported':
        outside=np.setdiff1d(np.arange(problem.n),problem.opening_stationary_actual_rows)
        if (not np.array_equal(P[outside],initial_outside_P)
                or not np.array_equal(Q[outside],initial_outside_Q)
                or not np.array_equal(fit.x[np.setdiff1d(np.arange(len(fit.x)),free_columns)],
                    fixed_initial[np.setdiff1d(np.arange(len(fit.x)),free_columns)])):
            raise ValueError('Opening-only fit changed an outside-opening state or fixed control')
        control_scope_binding.update(outside_opening_P_Q_bitexact=True,
            fixed_controls_bitexact=True,initial_state_is_exact_warm_physical_trajectory=True,
            outside_opening_rows=len(outside))
    physical_position_controls=problem.position_controls_from_x(fit.x)
    G=P.copy();G[:,2]=0.
    unit=packet_metres_per_scene_unit(p)
    guv=np.asarray([project_world(G[i:i+1]/unit,p['camera25'][i])[0][0] for i in range(len(P))])
    source_count=p['source_COCO17_qualified'][:,SOURCE_INDICES].sum(axis=1)
    inferred=(p['body_local_pose_inferred'] | (source_count<4)
              | ~p['source_COCO17_qualified'][:,[11,12]].all(axis=1)
              | ~p['body_geometry_source_fit_enabled'])
    valid=np.isfinite(P).all(axis=1) & np.isfinite(Q).all(axis=(1,2)) & (min_depth>.005)
    output_arrays=dict(actual_pts=p['actual_pts'],source_native_pts=p['source_native_pts'],
        P_world_m=P,Q_world=Q,camera25=p['camera25'],valid=valid,inferred=inferred,
        connect_from_previous=np.r_[False,np.ones(len(P)-1,bool)],
        G_world_m=G,G_native_uv=guv,COCO17_native_uv=uv,
        COCO17_world_m=np.einsum('nij,nkj->nki',Q,p['local_COCO17_m'])+P[:,None,:],
        native22_world_m=np.einsum('nij,nkj->nki',Q,p['local_native22_m'])+P[:,None,:],
        source_residual_weighted=source_residual,source_envelope_residual=envelope_residual,
        source_COCO17_native_uv=p['source_COCO17_native_uv'],
        source_COCO17_qualified=p['source_COCO17_qualified'],source_COCO17_score=p['source_COCO17_score'],
        raw_root_prior_Q_world=p['raw_root_prior_Q_world'],
        raw_root_left_source_pts=p['raw_root_left_source_pts'],
        raw_root_right_source_pts=p['raw_root_right_source_pts'],raw_root_alpha=p['raw_root_alpha'],
        source_hip_proxy_valid=p['source_hip_proxy_valid'],source_hip_status=p['source_hip_status'],
        local_pose_origin=p['body_local_pose_origin'],body_local_pose_inferred=p['body_local_pose_inferred'],
        body_local_pose_donor_actual_row=p['body_local_pose_donor_actual_row'],
        body_local_pose_source_portrait=p['body_local_pose_source_portrait'],
        body_geometry_source_fit_enabled=p['body_geometry_source_fit_enabled'],
        body_opening_temporal_inference_only=p['body_opening_temporal_inference_only'],
        body_effective_source_point_fit_enabled=problem.effective_point_fit_enabled,
        body_source_semantic_fit_rejection_mask=source_rejections,
        body_conditional_upper_body_source_enabled=problem.conditional_upper_body_rows,
        body_root_position_source_mode=np.where(problem.conditional_upper_body_rows,
            'conditional-upper-body',np.where(p['body_opening_temporal_inference_only'],'temporal-only','original-body-source')),
        opening_reliable_upper_body_source_pts=p['opening_reliable_upper_body_source_pts'],
        opening_near_static_user_source_end_pts=p['opening_near_static_user_source_end_pts'],
        source_original_observation_mask=p['source_original_observation_mask'],
        source_original_truncated=p['source_original_truncated'],
        source_opening_reviewed_hip_only_override=p['source_opening_reviewed_hip_only_override'],
        position_controls_world_m=physical_position_controls,
        shared_opening_xy_control_indices=problem.shared_opening_xy_control_indices,
        opening_shared_estimated_xy_world_m=physical_position_controls[0,:2],
        conditional_acceleration_world_m_s2=problem.acceleration_basis @ physical_position_controls,
        opening_near_stationary_actual_rows=problem.opening_stationary_actual_rows,
        opening_planar_velocity_condition_m_s=problem.opening_velocity_basis @ physical_position_controls[:,:2],
        opening_conditional_velocity_world_m_s=problem.opening_velocity_derivative_basis @ physical_position_controls,
        source_fitted_COCO_indices=SOURCE_INDICES,
        source_fitted_COCO_sigma_native_px=SOURCE_SIGMA,
        root_rotation_correction_controls=fit.x[problem.m*3:].reshape(problem.qm,3),
        position_knots=problem.position_knots,rotation_cut_control_offsets=np.asarray(problem.rotation_cut_control_offsets),
        rotation_knots_concatenated=np.concatenate(problem.rotation_knots),
        rotation_knot_offsets=np.r_[0,np.cumsum([len(k) for k in problem.rotation_knots])],
        fit_time_seconds=p['fit_time_seconds'],edge_time_origin=p['edge_time_origin'],edge_dt_seconds=p['edge_dt_seconds'],
        shot_index=p['shot_index'],native_pts=p['native_pts'],
        native_exposure_representative_pts=p['native_exposure_representative_pts'],native_actual_row=p['native_actual_row'],
        unique_world_G_known=np.zeros(len(P),bool),unique_world_G_m=np.full((len(P),3),np.nan),
        minimum_body_camera_depth_m=min_depth,body_scale=p['body_scale'].copy(),
        metres_per_scene_unit=p['metres_per_scene_unit'].copy())
    output_arrays['body_source_point_fit_enabled']=p.get('body_source_point_fit_enabled',np.ones((len(P),17),bool))
    for key in ['body_local_pose_inference_rows','body_local_pose_inference_anchor_actual_pts']:
        if key in p:output_arrays[key]=p[key]
    for key in p:
        if key.startswith(('body_fit_clock_', 'body_same_exposure_cache_')):
            output_arrays[key]=p[key]
    native_rows=p['native_actual_row']
    output_arrays.update(native_P_world_m=P[native_rows],native_Q_world=Q[native_rows],
        native_G_world_m=G[native_rows],native_G_native_uv=guv[native_rows],
        native_valid=valid[native_rows],native_inferred=inferred[native_rows])
    path=output/'actual_body_root_fit.npz'
    np.savez_compressed(path,**output_arrays)
    errors=np.linalg.norm(uv[:,SOURCE_INDICES]-np.where(p['source_COCO17_qualified'][:,SOURCE_INDICES,None],
                                                    p['source_COCO17_native_uv'][:,SOURCE_INDICES],uv[:,SOURCE_INDICES]),axis=2)
    report=dict(schema='rinkspace.fullprogramme.continuous-world-body-root-candidate.v1',
        output=dict(path=str(path),sha256=_digest(path)),
        input=dict(path=str(packet_path),sha256=_digest(packet_path)),
        actual_exposures=len(P),native_presentations=len(p['native_pts']),shots=9,
        actual_exposures_field_semantics='Historical canonical camera rows; see distinct_body_fit_exposure_times for source-clock repair',
        historical_camera_canonical_rows=len(P),
        distinct_body_fit_exposure_times=int(len(np.unique(p['fit_time_seconds']))),
        spline_world_position_shared_across_camera_cuts=True,knot_spacing_initial_s=args.spacing,
        metres_per_scene_unit=unit,metric_scale_status='declared_uniform_working_scale_not_measured_calibration',
        pelvis_semantics='P is the fitted body pelvis; G is its world vertical ice projection; neither measures blade contact',
        position_controls=problem.m,rotation_controls=problem.qm,
        effective_position_parameters=problem.m*3-2*max(len(problem.shared_opening_xy_control_indices)-1,0),
        source_original_COCO_indices=SOURCE_INDICES.tolist(),source_original_sigma_px=SOURCE_SIGMA.tolist(),source_envelope_used=problem.envelope_on,
        body_geometry_point_fit_authority_is_separate_from_original_source_qualification=True,
        qualified_source_points_diagnostic_only_due_inferred_local_geometry=int(np.sum(
            p['source_COCO17_qualified'] & ~p.get('body_source_point_fit_enabled',np.ones((len(P),17),bool)))),
        source_envelope_disabled_missing_local_pose=int((~problem.env_valid).sum()) if problem.envelope_on else 0,
        rotation_correction_is_coarse_per_cut=True,camera_or_source_UV_changed=False,
        unreliable_portrait_local_pose_inferred_from_visible_same_cut_body=True,
        portrait_inferred_actual_frames=int(p['body_local_pose_source_portrait'].sum()),
        position_output_post_filtered=False,equipment_floor_contact_factors=False,
        rotation_cut_breaks_equal_time_intervals=True,
        rotation_cut_clipped_short_tail_interval_removed=True,
        weak_acceleration_engineering_sigma_m_s2=problem.acceleration_sigma_m_s2.tolist(),
        motion_residual_time_normalization_reference_seconds=problem.motion_reference_interval_s,
        acceleration_prior_is_inside_source_fit_not_a_measurement_or_hard_gate=True,
        source_user_approximate_in_place_opening_end_source_pts=int(p['opening_near_static_user_source_end_pts']),
        opening_planar_binding=problem.opening_planar_binding,
        opening_source_support=problem.opening_source_support,
        conditional_opening_upper_body_rows=int(problem.conditional_upper_body_rows.sum()),
        conditional_opening_upper_body_original_qualified_points=int(
            p['source_COCO17_qualified'][problem.conditional_upper_body_rows,:7].sum()),
        conditional_opening_upper_body_effectively_fitted_points=int(
            problem.qualified[problem.conditional_upper_body_rows,:7].sum()),
        conditional_upper_body_local_geometry_is_inferred_not_measured=True,
        original_opening_temporal_inference_only_array_retained_as_historical_policy=True,
        opening_envelope_original_master_gate_retained=True,
        opening_hips_and_legs_original_source_authority_retained=True,
        opening_planar_position_is_one_shared_fitted_variable_not_preset_world_XY=problem.opening_planar_binding=='shared-xy',
        opening_shared_XY_control_count=len(problem.shared_opening_xy_control_indices),
        approximate_in_place_opening_is_soft_motion_condition=problem.opening_planar_binding=='soft-velocity',
        opening_planar_velocity_engineering_sigma_m_s=problem.opening_planar_velocity_sigma_m_s,
        opening_whole_root_velocity_engineering_sigma_m_s=problem.opening_velocity_sigma_m_s.tolist(),
        opening_height_temporally_inferred_not_fixed_or_floor_constrained=True,
        unreliable_opening_body_geometry_not_used_to_fit_position=problem.opening_source_support=='temporal-only',
        opening_absolute_world_position_not_locked=True,
        weak_rotation_control_curvature_deg=float(np.rad2deg(problem.rc)),
        weak_rotation_correction_prior_deg=float(np.rad2deg(problem.rprior)),
        body153cm_approximate_prior_scaled_once=True,inferred_frames=int(inferred.sum()),valid_frames=int(valid.sum()),
        inferred_semantics='saved missing local pose or insufficient original source points or unknown source hip; same trajectory fits adjacent real sources',
        direct_source_not_filled=True,all_body_points_share_RT=True,
        solver_success=bool(fit.success),solver_status=int(fit.status),solver_message=str(fit.message),
        solver_nfev=int(fit.nfev),solver_optimality=float(fit.optimality),total_weighted_cost=float(2*fit.cost),
        solver_stopping_tolerances=tolerances,
        fit_control_scope=control_scope,
        control_fit_scope_binding=control_scope_binding,
        exact_warm_initial_objective_replay=initial_objective_replay,
        source_semantic_fit_rejection_binding=source_rejection_binding,
        exact_warm_source_semantic_rejection_removed_cost=removed_semantic_source_cost,
        exact_warm_opening_source_restoration=(dict(
            restored_upper7_initial_weighted_cost=added_upper_cost,
            historical_objective_replay=initial_objective_replay+removed_semantic_source_cost-added_upper_cost,
            original_source_packet_and_physical_controls_unchanged=True)
            if exact_warm and opening_support_changed else None),
        elapsed_seconds=time.perf_counter()-started,source_weighted_error_each_keypoint_median=np.median(errors,axis=0).tolist(),
        source_video_and_camera_accuracy_not_proven=True,accuracy_accepted=False,
        no_measured_velocity_or_blade_contact_claim=True,no_user_circle_fit_or_gate=True,
        original_observation_mask_policy_reused=True,
        reviewed_opening_four_exposures_override_only_hips=True,
        warm_controls_binding=warm_binding,
        unobserved_position_control_temporal_warm_binding=temporal_control_warm_binding,
        reviewed_whole_pose_inference_binding=pose_inference_binding,
        official_saved_pose_clock_application=official_clock_binding,
        body_fit_clock_evidence_overlay=(dict(
            path=str(p['body_fit_clock_evidence_overlay_path']),
            sha256=str(p['body_fit_clock_evidence_overlay_sha256']),
            historical_camera_canonical_rows_retained=len(P),
            native_presentations_retained=len(p['native_pts']),
            distinct_body_fit_exposure_times=int(len(np.unique(p['fit_time_seconds']))),
            changed_edge_rows=p['body_fit_clock_evidence_overlay_edge_rows'].tolist(),
            camera_aliases_and_source_UV_unchanged=True,
            camera_aliases_source_UV_and_local_pose_unchanged=('body_same_exposure_cache_rows' not in p),
            local_pose_changed_only_by_separately_bound_direct_cache=('body_same_exposure_cache_rows' in p))
            if 'body_fit_clock_evidence_overlay_path' in p else None),
        same_exposure_direct_pose_cache=(dict(
            path=str(p['body_same_exposure_cache_manifest_path']),
            sha256=str(p['body_same_exposure_cache_manifest_sha256']),
            changed_actual_rows=p['body_same_exposure_cache_rows'].tolist(),
            direct_donor_actual_pts=p['body_same_exposure_cache_donor_actual_pts'].tolist(),
            source_UV_scores_gates_and_camera_aliases_unchanged=True,
            cached_local_body_only_no_new_pose_or_FK=True,
            root_prior_recomputed_from_current_camera=True)
            if 'body_same_exposure_cache_rows' in p else None),
        code=dict(path=str(Path(__file__)),sha256=_digest(Path(__file__))))
    if args.envelope: report['source_envelope_binding']=dict(path=args.envelope,sha256=_digest(args.envelope))
    if effective_samples_path:
        report['complete_body_samples_binding']=dict(path=str(effective_samples_path),sha256=_digest(effective_samples_path))
        report['original_body_samples_binding']=dict(path=args.body_samples,sha256=_digest(args.body_samples))
    _json(output/'report.json',report)
    (output/'fullprogramme_body_sequence.used.py').write_bytes(Path(__file__).read_bytes())
    print('SAVED '+str(path),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True)
    parser.add_argument('--camera',required=True)
    parser.add_argument('--hips',required=True)
    parser.add_argument('--clock',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--envelope')
    parser.add_argument('--body-samples')
    parser.add_argument('--spacing',type=float,default=.1)
    parser.add_argument('--max-nfev',type=int,default=25)
    parser.add_argument('--warm-controls')
    parser.add_argument('--fit-control-scope',choices=['full-programme','opening-supported'],default='full-programme',
        help='Opening-supported frees only actual spline controls with zero support at all outside-opening source nodes')
    parser.add_argument('--warm-controls-exact-state',action='store_true',
        help='Require identical current source/camera/surface objective and replay warm controls without reseeding')
    parser.add_argument('--opening-planar-binding',choices=['soft-velocity','shared-xy'],default='soft-velocity',
        help='Approximate in-place opening uses soft velocity; shared-xy is only for replay of historical hard-bound runs')
    parser.add_argument('--opening-source-support',choices=['conditional-upper-body','temporal-only'],default='conditional-upper-body',
                        help='Retain original qualified upper-body source pixels for an inferred opening root; temporal-only replays the historical gate')
    parser.add_argument('--allow-opening-source-support-change',action='store_true',
                        help='Explicit controlled restoration of upper7 source factors during exact physical-state warm start')
    parser.add_argument('--ftol',type=float,default=1e-4)
    parser.add_argument('--xtol',type=float,default=1e-6)
    parser.add_argument('--gtol',type=float,default=1e-3)
    parser.add_argument('--pose-inference-packet',help='Reviewed whole-local-pose overlay; fresh camera/root/source binding is retained')
    parser.add_argument('--pose-inference-body-samples',help='Reviewed local whole-body samples to overlay after clock correction')
    parser.add_argument('--official-pose-clock-patch',help='Source-authorized official exposure clock correction for saved local pose/root priors')
    parser.add_argument('--source-fit-rejections',help='Source-only reviewed semantic exclusions; original UV, qualification, sigma and clocks are retained')
    args=parser.parse_args();run(args)


if __name__=='__main__': main()
