"""Read-only, same-PTS sensitivity receipts for saved whole-programme fits.

This module never restores FK, loads a body model, projects, aligns or fits.
The compared scale/depth experiments are finite-budget conditional sensitivity,
not confidence intervals, independent accuracy evidence or candidate selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np


BODY_SOURCE_GROUPS = {'face': (0, 1, 2, 3, 4), 'torso': (5, 6, 11, 12),
                      'lower': (13, 14, 15, 16), 'upper': (7, 8, 9, 10)}


LEG_JOINTS = {1: 'left_hip', 2: 'right_hip', 4: 'left_knee', 5: 'right_knee',
              7: 'left_ankle', 8: 'right_ankle', 10: 'left_foot', 11: 'right_foot'}
STATE_SHAPES = {'P_world_mid': (3,), 'G_world_mid': (3,), 'Q_world_mid': (3, 3),
                'skate_origin_world_mid': (2, 3), 'skate_R_world_mid': (2, 3, 3),
                'local_R_mid': (21, 3, 3)}
DIAGNOSTICS = ('skate_clearance_m_mid', 'skate_support_gap_m_mid',
               'foot_cavity_max_violation_m_mid', 'cavity_penetration_mid',
               'heel_drift_mid', 'forefoot_error_mid', 'skate_edge_world_mid')
META_FIELDS = ('shared_betas', 'shared_body_scale', 'body_scale', 'metres_per_scene_unit',
               'solver_metres_per_scene_unit', 'body_length_ratio', 'ice_plane_known',
               'camera_binding_json', 'footwear_binding_json', 'skate_dimensions_json',
               'parameter_provenance_json', 'position_uncertainty_status')
PARTS = {
    'body_image': {'body_pixels', 'body_source_regions'},
    'source_image': {'old_gear_pixels', 'old_shoe_support_regions', 'full_source_skate_structure',
               'source_pixel_edge_profile', 'complete_runner_extent',
               'complete_visible_structure_extent', 'unassigned_source_skate_sets',
               'visible_boot_edges', 'visible_blade_curves', 'boot_direction_endpoints'},
    'source_derived_scene': {'source_support_location_xy', 'native_source_support_contact'},
    'priors': {'shared_static_prior', 'neutral_foot_compatibility', 'source_motion',
               'finite_network_pose_prior', 'finite_parent_root_prior',
               'actual_so3_angular_acceleration', 'actual_root_translation_acceleration',
               'actual_local_acceleration', 'finite_relative_skate_wear_prior',
               'relative_skate_angular_velocity', 'actual_skate_so3_angular_acceleration'},
    'finite_physics': {'dynamic_foot_compatibility', 'finite_ice', 'episode_support', 'grip',
                'continuous_support', 'flight_integrated_com',
                'flight_specific_angular_impulse', 'temporal_contact_mode', 'cavity_AL'},
}
CATEGORY_MEANING = {
    'body_image': 'Saved anatomical/region image residuals only; not a 3D accuracy measurement.',
    'source_image': 'Explicit saved footwear/source image residual blocks only; excludes priors, scene-derived targets and finite physics.',
    'source_derived_scene': 'Source-selected support targets using reference ice/backprojection/support geometry. native_source_support_contact and source_support_location_xy are not pure image-only evidence.',
    'priors': 'Finite pose/root/shape/wear/smoothness priors. source_motion is the legacy source-reviewed quiet/relation prior; neutral compatibility is an anatomical template prior.',
    'finite_physics': 'Finite geometric/material/support/dynamics penalties, including inferred modes and AL; not direct contact or physical accuracy truth.',
    'unknown': 'Unknown saved block retained separately; never silently counted as source evidence.',
}
LIMITATION = ('有限预算、指定体型/尺度/深度初始化及固定相机条件下的敏感性；'
              '不是置信区间、可辨识性证明或独立准确性验证。不按总cost选择候选。'
              'P为骨盆，G为其垂直投影；鞋O为模型原点，均不等于实测接触点或重心。')


def file_identity(path):
    path = Path(path).resolve()
    before = path.stat()
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Input changed while being read: ' + str(path))
    return dict(path=str(path), sha256=sha.hexdigest(), bytes=after.st_size)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _inventory(path):
    """Read NPY headers without inflating all saved meshes/parameters."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                          else np.lib.format.read_array_header_2_0)
                shape, fortran, dtype = reader(stream)
            result[name[:-4]] = dict(shape=list(shape), dtype=str(dtype), fortran_order=fortran)
    return result


def _fields(path, names):
    with np.load(path, allow_pickle=False) as packet:
        return {name: packet[name] for name in names if name in packet}


def _plain(value):
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value.tolist()


def _metadata(packet):
    out = {}
    for name in META_FIELDS:
        if name not in packet:
            continue
        if name == 'position_uncertainty_status':
            out[name] = np.unique(packet[name]).tolist()
        elif name.endswith('_json'):
            out[name] = json.loads(str(packet[name]))
        else:
            out[name] = _plain(packet[name])
    return out


def _clock(pts, time_base):
    pts, tb = np.asarray(pts), np.asarray(time_base)
    if pts.ndim != 1 or not len(pts) or pts.dtype.kind not in 'iu' or np.any(np.diff(pts) <= 0):
        raise ValueError('Ordered unique integer source PTS required')
    if tb.shape != (2,) or tb.dtype.kind not in 'iu' or np.any(tb <= 0):
        raise ValueError('Positive integer source time_base required')
    return pts, tb


def _equal_clock(a, b, name):
    if np.asarray(a).shape != np.asarray(b).shape or not np.array_equal(a, b):
        raise ValueError('Exact same PTS/time_base/shot population required: ' + name)


def part_category(name):
    # A source-looking name must never swallow an explicitly named prior.
    if 'prior' in name.lower():
        return 'priors'
    for category, names in PARTS.items():
        if name in names:
            return category
    if name.startswith('additional_unassigned_source_'):
        return 'source_image'
    return 'unknown'


def classified_costs(metrics):
    raw = metrics.get('weighted_parts')
    if not isinstance(raw, dict):
        raise ValueError('Saved metrics must include the original weighted_parts dictionary')
    grouped = {category: {} for category in CATEGORY_MEANING}
    for name, value in raw.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError('Invalid saved weighted cost: ' + name)
        grouped[part_category(name)][name] = value
    return dict(weighted_parts=raw, categories=grouped,
                category_sums={key: float(sum(values.values())) for key, values in grouped.items()},
                total_checksum_only=float(sum(raw.values())), used_for_candidate_selection=False)


def distribution(values, pts=None):
    a = np.asarray(values, float)
    good = np.isfinite(a)
    finite = a[good]
    row = dict(total_count=int(a.size), finite_count=int(good.sum()),
               nonfinite_count=int((~good).sum()))
    if not len(finite):
        return row
    row.update(minimum=float(finite.min()), median=float(np.median(finite)),
               mean=float(finite.mean()), rms=float(np.sqrt(np.mean(finite * finite))),
               p95=float(np.percentile(finite, 95)), maximum=float(finite.max()))
    if pts is not None:
        for name, function in (('minimum', np.nanargmin), ('maximum', np.nanargmax)):
            index = np.unravel_index(function(np.where(good, a, np.nan)), a.shape)
            row[name + '_pts'] = int(np.asarray(pts)[index[0]])
            row[name + '_component_index'] = [int(value) for value in index[1:]]
    return row


def body_evidence_rescore(packet, observations, body_pixel_weights):
    """Review saved COCO UV against ONE nominal raw-source ledger and weights.

    Group membership follows raw source labels, while model_joint_ids selects
    the predicted COCO joint. Candidate-specific mapping fields are ignored.
    Only body image residuals enter this cost; no prior or other block is added.
    """
    pts, tb = _clock(packet['pts'], packet['time_base'])
    if len(pts) < 2 or np.any(np.diff(pts) != pts[1] - pts[0]):
        raise ValueError('Common body evidence requires an equal-step source PTS clock')
    _equal_clock(pts, observations['pts'], 'nominal body evidence ledger')
    n = len(pts)
    use = np.asarray(observations['use'])
    uv, sigma = np.asarray(observations['uv']), np.asarray(observations['sigma_px'])
    if use.shape != (n, 17) or use.dtype.kind != 'b' or uv.shape != (n, 17, 2) or sigma.shape != (n, 17):
        raise ValueError('Nominal body use/uv/sigma_px must have exact Nx17 source-ledger shapes')
    identity = np.broadcast_to(np.arange(17), (n, 17))
    mapping = np.asarray(observations.get('model_joint_ids', identity))
    if mapping.shape != (n, 17) or mapping.dtype.kind not in 'iu' or np.any((mapping < 0) | (mapping >= 17)):
        raise ValueError('Invalid nominal body model_joint_ids mapping')
    fi, j = np.nonzero(use)
    target, sig = np.asarray(uv[fi, j], float), np.asarray(sigma[fi, j], float)
    if not np.isfinite(target).all() or not np.isfinite(sig).all() or np.any(sig <= 0):
        raise ValueError('Used nominal source UV and sigma must be finite with positive sigma')
    weights = np.asarray(body_pixel_weights, float)
    if weights.shape != (len(fi) * 2,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Nominal body_pixels weights must be an explicit positive vector of exactly two coordinates per used observation')
    period = float(int(pts[1] - pts[0]) * int(tb[0]) / int(tb[1]))
    common = dict(review_only=True, used_for_fitting=False, original_candidate_fit_cost=False,
                  source_observations=int(len(fi)), sample_period_seconds=period,
                  native_pts_step=int(pts[1] - pts[0]),
                  nominal_mapping='model_joint_ids' if 'model_joint_ids' in observations else 'identity',
                  nominal_reassigned_source_labels=int(np.count_nonzero(mapping != identity)),
                  candidate_saved_mapping_ignored=True,
                  grouping='Raw source joint labels; mapping only selects model COCO prediction.',
                  objective_scope='body_pixels only; all other saved weighted_parts retain their own fitted objective',
                  formula='d=(COCO_uv[fi,nominal_map[fi,j]]-nominal_uv[fi,j])/nominal_sigma[fi,j]; u=sqrt(1+sum(d*d)/4); r=d*sqrt(2/(u+1)); cost=0.5*dt*(r_flat**2 @ nominal_body_pixels_weights)')
    if 'COCO_uv' not in packet:
        return dict(common, status='missing', reason='Saved result has no COCO_uv; no projection/FK replacement computed.')
    predicted = np.asarray(packet['COCO_uv'])
    if predicted.shape != (n, 17, 2):
        raise ValueError('Saved COCO_uv must use the exact full source clock and 17 model joints')
    prediction = np.asarray(predicted[fi, mapping[fi, j]], float)
    if not np.isfinite(prediction).all():
        raise ValueError('Saved COCO_uv is nonfinite at a used nominal body observation')
    pixel_distance = np.linalg.norm(prediction - target, axis=1)
    d = (prediction - target) / sig[:, None]
    u = np.sqrt(1 + np.sum(d * d, axis=1) / 4)
    r = d * np.sqrt(2 / (u + 1))[:, None]
    exact_cost = float(.5 * period * ((r.ravel() ** 2) @ weights))
    point_cost = .5 * period * np.sum(r * r * weights.reshape(-1, 2), axis=1)
    groups = {}
    for name, ids in BODY_SOURCE_GROUPS.items():
        selected = np.isin(j, ids)
        values = pixel_distance[selected]
        groups[name] = dict(source_joint_ids=list(ids), observations=int(selected.sum()),
                            pixel_median=float(np.median(values)) if len(values) else None,
                            pixel_p95=float(np.percentile(values, 95)) if len(values) else None,
                            exact_weighted_body_cost=float(point_cost[selected].sum()))
    return dict(common, status='scored', exact_weighted_body_cost=exact_cost, groups=groups,
                pixel_median=float(np.median(pixel_distance)) if len(fi) else None,
                pixel_p95=float(np.percentile(pixel_distance, 95)) if len(fi) else None)


def body_cost_equivalence(score, saved_body_pixels, *, absolute_tolerance=1e-10):
    """Expose nominal replay disagreement numerically; never manufacture a pass."""
    if score['status'] != 'scored':
        return dict(status='missing', reason=score.get('reason'), saved_body_pixels=saved_body_pixels,
                    absolute_tolerance=absolute_tolerance, relative_tolerance=0.)
    if saved_body_pixels is None:
        return dict(status='missing', reason='Nominal final_metrics weighted_parts has no body_pixels',
                    recomputed_body_pixels=score['exact_weighted_body_cost'])
    delta = score['exact_weighted_body_cost'] - saved_body_pixels
    match = abs(delta) <= absolute_tolerance
    return dict(status='match' if match else 'mismatch', equivalent=bool(match),
                recomputed_body_pixels=score['exact_weighted_body_cost'], saved_body_pixels=saved_body_pixels,
                difference_recomputed_minus_saved=float(delta), absolute_difference=float(abs(delta)),
                absolute_tolerance=absolute_tolerance, relative_tolerance=0.,
                meaning='Only nominal body_pixels equivalence is checked; no common-objective claim for any other block.')


def _common_body_inputs(stage, prepared, receipt):
    obs = _fields(Path(prepared) / 'observations.npz',
                  ('pts', 'use', 'uv', 'sigma_px', 'model_joint_ids', 'body_joint_assignment_provenance_json'))
    required = {'pts', 'use', 'uv', 'sigma_px'}
    if not required.issubset(obs):
        raise ValueError('Nominal body ledger is missing required fields: ' + ', '.join(sorted(required - set(obs))))
    weights = _fields(Path(stage) / 'objective_block_weights.npz', ('body_pixels',))
    if 'body_pixels' not in weights:
        raise ValueError('Nominal objective_block_weights.npz must explicitly save body_pixels weights')
    record = dict(observations=receipt['files']['prepared']['observations.npz'],
                  objective_block_weights=receipt['files']['stage']['objective_block_weights.npz'],
                  source_groups=BODY_SOURCE_GROUPS,
                  mapping_origin='nominal observations.npz model_joint_ids, identity only when absent',
                  all_candidates_review_only=True, other_objective_blocks_not_replayed=True)
    if 'body_joint_assignment_provenance_json' in obs:
        record['saved_mapping_provenance'] = json.loads(str(obs['body_joint_assignment_provenance_json']))
    return obs, weights['body_pixels'], record


def _rotation_degrees(a, b):
    rel = np.asarray(a).swapaxes(-1, -2) @ np.asarray(b)
    sine = np.linalg.norm(np.stack((rel[..., 2, 1] - rel[..., 1, 2],
                                   rel[..., 0, 2] - rel[..., 2, 0],
                                   rel[..., 1, 0] - rel[..., 0, 1]), axis=-1), axis=-1) / 2
    cosine = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arctan2(sine, cosine))


def compare_arrays(nominal, comparison):
    for key in ('pts', 'time_base', 'source_shot_ids'):
        _equal_clock(nominal[key], comparison[key], key)
    for key in STATE_SHAPES:
        if nominal[key].shape != comparison[key].shape:
            raise ValueError('Saved state shape mismatch: ' + key)
    result = {key: comparison[key] - nominal[key] for key in ('P_world_mid', 'G_world_mid')}
    out = dict(P_delta_world_m=result['P_world_mid'], G_delta_world_m=result['G_world_mid'],
               P_distance_m=np.linalg.norm(result['P_world_mid'], axis=-1),
               G_distance_m=np.linalg.norm(result['G_world_mid'], axis=-1),
               Q_degrees=_rotation_degrees(nominal['Q_world_mid'], comparison['Q_world_mid']))
    delta = comparison['skate_origin_world_mid'] - nominal['skate_origin_world_mid']
    ids = np.asarray(list(LEG_JOINTS)) - 1
    out.update(skate_origin_delta_world_m=delta, skate_origin_distance_m=np.linalg.norm(delta, axis=-1),
               skate_R_degrees=_rotation_degrees(nominal['skate_R_world_mid'], comparison['skate_R_world_mid']),
               leg_local_R_degrees=_rotation_degrees(nominal['local_R_mid'][:, ids], comparison['local_R_mid'][:, ids]))
    return out


def _delta_summary(delta, pts, rows):
    selected = np.asarray(rows)
    summary = {key: distribution(delta[key][selected], pts[selected])
               for key in ('P_distance_m', 'G_distance_m', 'Q_degrees')}
    summary['shoes'] = {
        foot: dict(O_distance_m=distribution(delta['skate_origin_distance_m'][selected, i], pts[selected]),
                   R_degrees=distribution(delta['skate_R_degrees'][selected, i], pts[selected]))
        for i, foot in enumerate(('left', 'right'))}
    summary['leg_local_R_degrees'] = {
        name: distribution(delta['leg_local_R_degrees'][selected, i], pts[selected])
        for i, name in enumerate(LEG_JOINTS.values())}
    return summary


def _step(packet, i):
    return dict(P_delta_world_m=(packet['P_world_mid'][i + 1] - packet['P_world_mid'][i]).tolist(),
                G_delta_world_m=(packet['G_world_mid'][i + 1] - packet['G_world_mid'][i]).tolist(),
                Q_degrees=float(_rotation_degrees(packet['Q_world_mid'][i], packet['Q_world_mid'][i + 1])),
                skate_origin_delta_world_m=(packet['skate_origin_world_mid'][i + 1] - packet['skate_origin_world_mid'][i]).tolist(),
                skate_R_degrees=_rotation_degrees(packet['skate_R_world_mid'][i], packet['skate_R_world_mid'][i + 1]).tolist(),
                leg_local_R_degrees=_rotation_degrees(packet['local_R_mid'][i, np.asarray(list(LEG_JOINTS)) - 1],
                                                      packet['local_R_mid'][i + 1, np.asarray(list(LEG_JOINTS)) - 1]).tolist())


def _guards(metrics, baseline):
    saved = metrics.get('primary_guards', {})
    expected = set(baseline.get('source', {})) | set(saved)
    if baseline.get('torso') is not None:
        expected.add('torso')
    rows = {}
    for name in sorted(expected):
        raw = saved.get(name)
        if not isinstance(raw, dict) or not isinstance(raw.get('rejected'), bool):
            rows[name] = dict(status='missing', saved=raw)
            continue
        row = dict(status='fail' if raw['rejected'] else 'pass', saved=raw)
        if 'before' in raw and 'after' in raw:
            radius = raw.get('original_radius_px')
            allowed = .25 * radius if radius is not None else (max(2., .1 * raw['before']) if name == 'torso' else None)
            if allowed is not None:
                limit = raw['before'] + allowed
                row.update(saved_policy_limit_px=limit, remaining_margin_px=limit - raw['after'],
                           saved_rejection_consistent=bool(raw['rejected'] == (raw['after'] > limit)))
        rows[name] = row
    return dict(available=isinstance(metrics.get('primary_guards'), dict),
                counts={status: sum(row['status'] == status for row in rows.values())
                        for status in ('pass', 'fail', 'missing')},
                all_saved_expected_pass=bool(rows) and all(row['status'] == 'pass' for row in rows.values()),
                rows=rows, meaning='Saved finite engineering source guards; not independent 3D accuracy gates.')


def _initializations(stage):
    """Follow only recorded resume ancestry; do not guess sibling experiment names."""
    found, seen = [], set()
    current = Path(stage).resolve()
    while current not in seen:
        seen.add(current)
        init, resume = current / 'initialization.json', current / 'resume.json'
        if init.is_file():
            found.append(dict(kind='initialization', file=file_identity(init), saved=_read_json(init)))
        if not resume.is_file():
            break
        record = _read_json(resume)
        found.append(dict(kind='resume', file=file_identity(resume), saved=record))
        parent = record.get('parameter')
        if not parent:
            break
        current = Path(parent).resolve().parent.parent
    return found


def _validate_scope(preparation, pts, shots, tb):
    scope_name = preparation.get('scope')
    if not scope_name:
        return dict(available=False, authority='prepared/input.npz only; no registered scope metadata')
    path = Path(scope_name)
    if not path.is_file():
        raise FileNotFoundError('Recorded full-programme scope unavailable: ' + str(path))
    identity = file_identity(path)
    if preparation.get('scope_sha256') and preparation['scope_sha256'] != identity['sha256']:
        raise ValueError('Registered programme scope SHA changed')
    scope = _read_json(path)
    clock = scope['source_clock']
    step = int(clock['native_pts_step'])
    expected, labels = [], []
    if step <= 0:
        raise ValueError('Invalid registered native PTS step')
    for shot in scope['shots']:
        p = shot['processing']
        values = np.arange(int(p['first_pts']), int(p['last_pts']) + 1, step, dtype=np.int64)
        if len(values) != int(p['native_frame_count']) or not len(values) or values[-1] != int(p['last_pts']):
            raise ValueError('Registered shot range/count mismatch')
        expected.append(values)
        labels.extend([shot['id']] * len(values))
    expected = np.concatenate(expected)
    _equal_clock(pts, expected, 'registered full programme')
    _equal_clock(shots, np.asarray(labels), 'registered shots')
    _equal_clock(tb, np.array([clock['time_base_num'], clock['time_base_den']]), 'registered time_base')
    target = scope['processing_scope']
    if (len(pts) != target['native_frame_count'] or pts[0] != target['first_pts']
            or pts[-1] != target['last_pts'] or pts[-1] + step != target['stop_pts_exclusive']):
        raise ValueError('Registered whole-programme range/count mismatch')
    return dict(available=True, file=identity, native_pts_step=step,
                expected_frames=len(expected), complete=True)


def read_stage(stage, prepared):
    stage, prepared = Path(stage).resolve(), Path(prepared).resolve()
    result_path, parameter_path = stage / 'fitted/result.npz', stage / 'solve/candidate.npz'
    summary = _read_json(stage / 'run_summary.json')
    initial, final = _read_json(stage / 'initial_metrics.json'), _read_json(stage / 'final_metrics.json')
    names = ('pts', 'source_pts', 'time_base', 'source_shot_ids', 'COCO_uv') + tuple(STATE_SHAPES) + DIAGNOSTICS + META_FIELDS
    result = _fields(result_path, names)
    p = _fields(prepared / 'input.npz', ('pts', 'time_base', 'shot_ids', 'shared_betas', 'body_scale', 'metres_per_scene_unit'))
    pts, tb = _clock(result['pts'], result['time_base'])
    _equal_clock(pts, p['pts'], 'result versus prepared input')
    _equal_clock(tb, p['time_base'], 'result versus prepared time_base')
    if 'source_pts' in result:
        _equal_clock(pts, result['source_pts'], 'saved source_pts alias')
    shots = result.get('source_shot_ids', p.get('shot_ids'))
    if shots is None or np.asarray(shots).shape != pts.shape:
        raise ValueError('Saved whole-programme shot labels required')
    result['source_shot_ids'] = shots = np.asarray(shots).astype(str)
    if 'shot_ids' in p:
        _equal_clock(shots, p['shot_ids'], 'result versus prepared shots')
    for name, tail in STATE_SHAPES.items():
        a = result.get(name)
        if a is None or a.shape != (len(pts),) + tail or not np.isfinite(a).all():
            raise ValueError('Missing/invalid saved full-programme state: ' + name)
    preparation = _read_json(prepared / 'preparation.json') if (prepared / 'preparation.json').is_file() else {}
    scope = _validate_scope(preparation, pts, shots, tb)
    if 'source_frames' in summary and summary['source_frames'] != len(pts):
        raise ValueError('Run summary full-programme frame count mismatch')
    for name in ('shared_betas', 'body_scale'):
        if name in p and not np.isfinite(p[name]).all():
            raise ValueError('Nonfinite prepared shape/scale: ' + name)
    if 'body_scale' in p and (p['body_scale'].shape != () or float(p['body_scale']) <= 0):
        raise ValueError('Positive shared prepared body_scale required')
    if 'frames' in preparation and preparation['frames'] != len(pts):
        raise ValueError('Preparation full-programme frame count mismatch')
    unique = list(dict.fromkeys(shots.tolist()))
    by_shot = {name: dict(frames=int((shots == name).sum()), first_pts=int(pts[shots == name][0]),
                         last_pts=int(pts[shots == name][-1])) for name in unique}
    if 'by_shot' in preparation and preparation['by_shot'] != {k: v['frames'] for k, v in by_shot.items()}:
        raise ValueError('Preparation full-programme shot count mismatch')
    files = {'result': file_identity(result_path), 'parameter': file_identity(parameter_path)}
    files['prepared'] = {path.name: file_identity(path) for path in sorted(prepared.iterdir())
                         if path.is_file() and path.suffix.lower() in ('.json', '.npz')}
    files['stage'] = {name: file_identity(stage / name) for name in
                      ('run_summary.json', 'initial_metrics.json', 'final_metrics.json',
                       'primary_quality_baseline.json', 'objective_policy.json',
                       'objective_block_weights.npz', 'source_initialization_guard.json') if (stage / name).is_file()}
    identity_keys = {'prepared_input_sha256': 'input.npz', 'prepared_observations_sha256': 'observations.npz',
                     'source_evidence_sha256': 'source_evidence.json'}
    parameter_meta = _fields(parameter_path, tuple(identity_keys) +
                             ('state_kind', 'ice_soft_policy_json', 'initialization_provenance_json'))
    bindings = {}
    for key, name in identity_keys.items():
        if key not in parameter_meta:
            bindings[key] = dict(status='metadata_missing')
            continue
        actual = files['prepared'].get(name)
        declared = str(parameter_meta[key])
        matches = actual is not None and actual['sha256'] == declared
        bindings[key] = dict(declared_sha256=declared, matches=matches)
        if not matches:
            raise ValueError('Saved parameter is not bound to supplied prepared input: ' + key)
    baseline = _read_json(stage / 'primary_quality_baseline.json') if (stage / 'primary_quality_baseline.json').is_file() else {}
    raw = {name: result[name] for name in DIAGNOSTICS if name in result and name != 'skate_edge_world_mid'}
    if 'skate_edge_world_mid' in result:
        edge = result['skate_edge_world_mid']
        if edge.shape[0] != len(pts) or edge.shape[1] != 2 or edge.shape[-1] != 3:
            raise ValueError('Saved raw ice runner geometry has invalid shape')
        raw['minimum_runner_edge_height_m'] = edge[..., 2].min(axis=tuple(range(2, edge.ndim - 1)))
        del result['skate_edge_world_mid']
    raw_report = {}
    for name, a in raw.items():
        if a.shape[0] != len(pts):
            raise ValueError('Saved raw diagnostic does not use full source clock: ' + name)
        raw_report[name] = dict(whole=distribution(a, pts), by_shot={shot: distribution(a[shots == shot], pts[shots == shot]) for shot in unique})
        if a.ndim >= 2 and a.shape[1] == 2:
            raw_report[name]['by_foot'] = {foot: distribution(a[:, i], pts)
                                                 for i, foot in enumerate(('left', 'right'))}
    receipt = dict(stage=str(stage), prepared=str(prepared), files=files,
                   array_shapes=dict(result=_inventory(result_path), parameter=_inventory(parameter_path),
                                     input=_inventory(prepared / 'input.npz')),
                   parameter_input_bindings=bindings,
                   coverage=dict(frames=len(pts), first_pts=int(pts[0]), last_pts=int(pts[-1]),
                                 first_last_seconds=(pts[[0, -1]] * float(tb[0]) / tb[1]).tolist(),
                                 time_base=tb.tolist(), exact_prepared_clock=True, scope=scope, by_shot=by_shot,
                                 cut_count=int(np.count_nonzero(shots[1:] != shots[:-1]))),
                   shape_and_scale_assumptions=dict(prepared=_metadata(p), result=_metadata(result),
                                                    missing_result_shape_fields=[k for k in ('shared_betas', 'shared_body_scale') if k not in result],
                                                    interpretation=LIMITATION),
                   initialization_ancestry=_initializations(stage),
                   saved_parameter_metadata={key: json.loads(str(value)) if key.endswith('_json') else _plain(value)
                                             for key, value in parameter_meta.items()},
                   saved_run_summary=summary, saved_initial_metrics=initial, saved_final_metrics=final,
                   costs=dict(initial=classified_costs(initial), final=classified_costs(final)),
                   source_guards=_guards(final, baseline),
                   raw_ice_and_structure=dict(saved_initial_structure=initial.get('structure'),
                                              saved_final_structure=final.get('structure'), arrays=raw_report,
                                              meaning='Original saved template/reference-Z=0 and cavity/collar diagnostics; no reconstruction, no physical truth/contact acceptance.'))
    return result, receipt


def _comparison(nominal, other, nominal_receipt, other_receipt):
    delta = compare_arrays(nominal, other)
    pts, shots, tb = nominal['pts'], nominal['source_shot_ids'], nominal['time_base']
    cuts = []
    for i in np.flatnonzero(shots[1:] != shots[:-1]):
        rows = np.array([i, i + 1])
        cuts.append(dict(left_shot=str(shots[i]), right_shot=str(shots[i + 1]),
                         left_pts=int(pts[i]), right_pts=int(pts[i + 1]), dt_s=float((pts[i + 1] - pts[i]) * tb[0] / tb[1]),
                         same_pts_delta=_delta_summary(delta, pts, rows),
                         endpoint_deltas={key: value[rows].tolist() for key, value in delta.items()},
                         nominal_adjacent_step=_step(nominal, i), comparison_adjacent_step=_step(other, i)))
    cost_changes = {}
    for phase in ('initial', 'final'):
        a, b = nominal_receipt['costs'][phase]['weighted_parts'], other_receipt['costs'][phase]['weighted_parts']
        cost_changes[phase] = {name: dict(category=part_category(name), nominal=a.get(name), comparison=b.get(name),
                                         change=None if name not in a or name not in b else b[name] - a[name])
                                for name in sorted(set(a) | set(b))}
    ng, og = nominal_receipt['source_guards']['rows'], other_receipt['source_guards']['rows']
    nominal_shape = nominal_receipt['shape_and_scale_assumptions']['prepared']
    other_shape = other_receipt['shape_and_scale_assumptions']['prepared']
    nominal_files, other_files = nominal_receipt['files']['prepared'], other_receipt['files']['prepared']
    assumptions = dict(shared_betas_equal=nominal_shape.get('shared_betas') == other_shape.get('shared_betas'),
                       prepared_body_scale_ratio=(other_shape['body_scale'] / nominal_shape['body_scale']
                                                  if 'body_scale' in nominal_shape and 'body_scale' in other_shape else None),
                       same_prepared_file_sha256={name: nominal_files.get(name, {}).get('sha256') == other_files.get(name, {}).get('sha256')
                                                  if name in nominal_files and name in other_files else None
                                                  for name in ('camera_context.json', 'model_identity.json', 'observations.npz', 'source_evidence.json', 'fixed_binding.npz')},
                       changed_inputs_retained_in_each_receipt=True,
                       same_numerical_objective_not_assumed_from_a_total_cost=True)
    report = dict(assumption_comparison=assumptions, whole=_delta_summary(delta, pts, np.arange(len(pts))),
                  by_shot={str(shot): _delta_summary(delta, pts, np.flatnonzero(shots == shot))
                           for shot in dict.fromkeys(shots.tolist())},
                  cuts=cuts, cut_count=len(cuts), weighted_part_changes=cost_changes,
                  each_source_guard={name: dict(nominal=ng.get(name, {'status': 'missing'}),
                                                comparison=og.get(name, {'status': 'missing'}))
                                     for name in sorted(set(ng) | set(og))},
                  same_source_clock=True, no_alignment_or_scale_normalization_applied=True,
                  interpretation=LIMITATION)
    return delta, report


def parse_comparison(value):
    parts = value.split('=', 2)
    if len(parts) != 3 or re.fullmatch(r'[A-Za-z0-9_-]+', parts[0]) is None or not all(parts):
        raise argparse.ArgumentTypeError('Use --comparison label=stage=prepared with an ASCII label')
    return parts[0], Path(parts[1]), Path(parts[2])


def generate_report(nominal_stage, nominal_prepared, comparisons, output, *, data_root=None):
    output = Path(output).resolve()
    if data_root is None:
        from rinkspace.storage import AssetStore
        data_root = AssetStore(Path(__file__).resolve().parents[3]).data
    data_root = Path(data_root).resolve()
    if output == data_root or not output.is_relative_to(data_root):
        raise ValueError('Output must be a new directory under the configured Data root')
    if output.exists():
        raise FileExistsError('Preserve existing report; choose a new --output: ' + str(output))
    comparisons = list(comparisons)
    labels = [entry[0] for entry in comparisons]
    if not labels or len(set(labels)) != len(labels) or any(re.fullmatch(r'[A-Za-z0-9_-]+', name) is None for name in labels):
        raise ValueError('At least one comparison with distinct ASCII labels is required')
    nominal, nominal_receipt = read_stage(nominal_stage, nominal_prepared)
    observations, body_weights, body_evidence = _common_body_inputs(nominal_stage, nominal_prepared, nominal_receipt)
    nominal_score = body_evidence_rescore(nominal, observations, body_weights)
    nominal_equivalence = body_cost_equivalence(nominal_score, nominal_receipt['saved_final_metrics']['weighted_parts'].get('body_pixels'))
    nominal_receipt['nominal_body_evidence_review'] = nominal_score
    body_evidence['nominal_body_cost_equivalence'] = nominal_equivalence
    report = dict(schema='rinkspace.saved-programme-sensitivity.v2', interpretation=LIMITATION,
                  category_definitions=CATEGORY_MEANING, leg_joint_ids_and_names=LEG_JOINTS,
                  rotation_change='Geodesic degrees from saved matrices; no FK/Jacobian/body restoration.',
                  nominal=nominal_receipt, common_body_evidence_review=body_evidence,
                  comparisons={}, candidate_selection_performed=False)
    packets = {}
    for label, stage, prepared in comparisons:
        other, receipt = read_stage(stage, prepared)
        delta, summary = _comparison(nominal, other, nominal_receipt, receipt)
        filename = 'comparison_' + label + '.npz'
        packets[filename] = dict(pts=nominal['pts'], time_base=nominal['time_base'],
                                 source_shot_ids=nominal['source_shot_ids'],
                                 leg_joint_ids=np.array(list(LEG_JOINTS)), **delta)
        common_score = body_evidence_rescore(other, observations, body_weights)
        report['comparisons'][label] = dict(input=receipt, differences=summary,
                                           nominal_body_evidence_review=common_score, per_pts_arrays=filename)
    # Reject non-JSON saved metadata before creating any new output as well.
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    # All clocks/bindings/costs are validated before creating any new output.
    output.mkdir(parents=True, exist_ok=False)
    for filename, packet in packets.items():
        np.savez_compressed(output / filename, **packet)
        report['comparisons'][filename[len('comparison_'):-4]]['per_pts_file'] = file_identity(output / filename)
    destination = output / 'report.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return dict(report=str(destination), comparisons=labels, candidate_selection_performed=False,
                nominal_body_cost_equivalence=nominal_equivalence)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nominal-stage', type=Path, required=True)
    parser.add_argument('--nominal-prepared', type=Path, required=True)
    parser.add_argument('--comparison', type=parse_comparison, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate_report(args.nominal_stage, args.nominal_prepared, args.comparison, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
