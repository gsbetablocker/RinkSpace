"""Same-source pixel and physical diagnostics; no fit or accuracy acceptance."""
from pathlib import Path
import hashlib
import json

import numpy as np

from .boot_integration import camera_centres
from .footwear_binding import transform
from .footwear_fit import sample_polyline
from .footwear_layout import OBSERVATION_LAYOUT
from .footwear_observations import visible_shell_edges
from .footwear_pixel_factors import nearest, nearest_segment
from .fullbody_fit import footwear_rows
from .rebuild_prepare import load, digest
from .skate_diagnostics import ice_objective_mode, _legacy_ice_diagnostics
from .skate_physics import (fk_state, constraints, ALLOWANCE, SUPPORT_BAND,
                            FEASIBILITY_TOLERANCE)


CORE = (35.20, 36.80)
PARENTS = {
    'fullbody_1102': 'fullbody_1102/full/result.npz',
    'boot_integration_1102': 'boot_integration_1102/fitted/result.npz',
    'opening_hand_1103': 'opening_timing_1103/hand_polish/fitted/result.npz',
}


def _jsonable(value):
    if isinstance(value, dict):return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):return _jsonable(value.tolist())
    if isinstance(value, np.generic):return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):return None
    return value


def _identical(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _array_identity(value):
    a = np.asarray(value)
    return dict(shape=list(a.shape), dtype=str(a.dtype),
                bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest())


def _stats(values, pts):
    """Keep failed predictions in the denominator, never silently drop them."""
    values, pts = np.asarray(values, float).ravel(), np.asarray(pts, np.int64).ravel()
    if len(values) != len(pts):raise ValueError('Diagnostic values/PTS length mismatch')
    ok = np.isfinite(values)
    result = dict(count=len(values), finite_count=int(ok.sum()), missing_count=int((~ok).sum()),
                  median=None, p95=None, max=None, worst_pts=None, worst_source_seconds=None)
    if ok.any():
        selected = np.flatnonzero(ok);worst = selected[np.argmax(values[ok])]
        result.update(median=float(np.median(values[ok])), p95=float(np.percentile(values[ok], 95)),
                      max=float(values[worst]), worst_pts=int(pts[worst]),
                      worst_source_seconds=float(pts[worst]/12800.))
    return result


def _scopes(values, pts):
    pts = np.asarray(pts, np.int64);mask = (pts >= round(CORE[0]*12800)) & (pts <= round(CORE[1]*12800))
    return dict(all_source_batch=_stats(values, pts), core_35_20_to_36_80=_stats(np.asarray(values)[mask], pts[mask]))


def _pixels(packet, rows, layout, faces, centres):
    pts = np.asarray(packet['pts']);uv = np.asarray(packet['equipment_uv']).reshape(len(pts), layout.equipment_count, 2)
    world = np.asarray(packet['gear_world_mid']).reshape(len(pts), layout.equipment_count, 3)
    samples = {name: [[], []] for name in ('heel_toe_finite_outer_edges', 'old_blade_curves', 'visible_boot_source_contour')}
    choices, failures = [], []
    def add(name, row, values):
        values = np.asarray(values, float).ravel()
        samples[name][0].extend(values);samples[name][1].extend([int(row['pts'])]*len(values))
    for row in rows.landmark_rows:
        for kind in ('heel', 'toe'):
            ids = layout.footwear_landmarks(row['foot'], kind)
            try:values = nearest_segment([row[kind]], uv[row['frame']], np.asarray([ids]))[0]
            except ValueError as error:
                values = [np.nan];failures.append(dict(kind=kind, pts=row['pts'], foot=row['foot'], reason=str(error)))
            add('heel_toe_finite_outer_edges', row, values)
    for row in rows.blade_rows:
        source = sample_polyline(row['uv'], 24)
        candidates = [layout.footwear_landmarks(row['foot'], kind) for kind in ('edge_negative_y', 'edge_positive_y')]
        if all(np.isfinite(uv[row['frame'], ids]).all() for ids in candidates):
            distances = [nearest(source, uv[row['frame'], ids])[0] for ids in candidates]
            side = int(np.argmin([d@d for d in distances]));values = distances[side]
            choices.append(dict(pts=row['pts'], foot=row['foot'], candidate=('negative_y', 'positive_y')[side],
                                source_side_confirmed=False, source_samples=24))
        else:
            values = np.full(24, np.nan)
            failures.append(dict(kind='blade', pts=row['pts'], foot=row['foot'], reason='nonfinite curve projection'))
        add('old_blade_curves', row, values)
    for row in rows.visible_rows:
        ids = layout.footwear_landmarks(row['foot'], 'shell');fi = row['frame']
        try:
            edges = visible_shell_edges(world[fi, ids], faces, centres[fi])['edges']
            values = nearest_segment(row['observed'], uv[fi, ids], edges)[0]
        except ValueError as error:
            values = np.full(len(row['observed']), np.nan)
            failures.append(dict(kind='boot_contour', pts=row['pts'], foot=row['foot'], reason=str(error)))
        add('visible_boot_source_contour', row, values)
    return dict(errors_px={name: _scopes(*value) for name, value in samples.items()},
                whole_curve_side_choices=choices, missing_predictions=failures)


def _max_error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():return None
    return float(np.max(np.abs(a-b), initial=0.))


def _source_contract(root, out, ref, packet, parent):
    keys = ('pts', 'source_pts', 'time_base', 'source_lower_body_inferred', 'opening_inferred_mask',
            'association_corrected_pts', 'contact_state', 'user_accepted', 'coordinate_convention', 'observation_ledger_id',
            'uv', 'use', 'sigma_px', 'lower_body_inferred', 'source_id', 'source_qualified')
    preserved = {key: dict(identical=key in packet and _identical(parent[key], packet[key]),
                           source=_array_identity(parent[key])) for key in keys if key in parent}
    obs_path = root/'boot_integration_1102/observations.npz';obs = load(obs_path)
    external = {}
    saved = load(out/'observations.npz') if (out/'observations.npz').is_file() else None
    for key, value in obs.items():
        item = dict(source=_array_identity(value))
        if saved is not None:item['output_observation_bytes_identical'] = key in saved and _identical(value, saved[key])
        if key in ('pts', 'time_base', 'uv', 'use', 'sigma_px', 'lower_body_inferred', 'source_id'):
            # A result's COCO_uv is a prediction, never a substitute for source uv.
            if key in packet:item['packet_bytes_identical'] = _identical(value, packet[key])
            else:item['packet_storage'] = 'external observation ledger; not a result-packet field'
        external[key] = item
    P = np.asarray(packet['P_world_mid']);G = P.copy();G[:, 2] = 0.
    return dict(parent_arrays=preserved, all_parent_arrays_identical=all(v['identical'] for v in preserved.values()),
                reference_pts_bytes_identical=_identical(ref.p['pts'], packet['pts']),
                observation_ledger=dict(path=str(obs_path), sha256=digest(obs_path), arrays=external),
                G_exactly_vertical_projection_of_P=_identical(G, packet['G_world_mid']))


def _replay(ref, packet, ratio, pts):
    targets = [pts[0], pts[-1], *[round(t*12800) for t in (26.22, 35.20, 35.60, 35.88, 36.02, 36.80, 39.78, 43.86)]]
    frames = np.array(sorted({int(np.searchsorted(pts, p)) for p in targets if p in pts}))
    t = ref.times[frames];state = ref.world(ref.at(t));physics = fk_state(ref, ref.at(t), np.zeros((len(t), 69)))
    checks = {target: _max_error(packet[target][frames], state[key]*scale) for target, key, scale in (
        ('P_world_mid', 'P', ratio), ('Q_world_mid', 'Q', 1.), ('local_R_mid', 'local_R', 1.),
        ('native22_world_mid', 'native', ratio), ('COCO_world_mid', 'coco', ratio), ('gear_world_mid', 'gear', ratio))}
    for key, value in [('skate_R_world_mid', physics['R']), ('skate_origin_world_mid', physics['O']*ratio),
                       ('skate_lower_world_mid', transform(physics['R'], physics['O'], ref.binding.packet()['template']['vertices'])*ratio)]:
        checks[key] = None if key not in packet else _max_error(packet[key][frames], value)
    return dict(source_pts=pts[frames], max_absolute_errors=checks, engineering_replay_tolerance=1e-9,
                all_within_replay_tolerance=all(v is not None and v <= 1e-9 for v in checks.values()),
                meaning='same body/FK T_WB, binding and template; numerical identity is not source-image acceptance')


def _physics(ref, packet, pts, evidence):
    t = np.asarray(ref.times);state = fk_state(ref, ref.at(t), np.zeros((len(t), 69)))
    result = constraints(ref.binding, state, t, pts[0]/12800., evidence)
    clearance = result['clearance'];violation = np.maximum(-clearance, 0.)
    flat_pts = np.repeat(pts, 2);features = result['features'];height = features['height']
    runner = np.array([s == 'runner' for s in features['semantics']])
    source_rows = []
    for row in evidence:
        found = np.flatnonzero(pts == row['pts'])
        if len(found) != 1:
            source_rows.append(dict(id=row['id'], pts=row['pts'], status='missing_source_exposure'));continue
        i = int(found[0]);members = row.get('feet', [0, 1]);kind = row.get('mode', 'glide')
        if kind in ('unknown','airborne'):
            source_rows.append(dict(id=row['id'],pts=int(pts[i]),status=kind,source=row.get('source')));continue
        eligible = ref.binding.packet()['template']['support_feature_ids'][kind]
        candidates = [(float(height[i, f, j]), f, name) for f in members for j, name in enumerate(features['ids']) if name in eligible]
        h, foot, feature = min(candidates);gap_violation = max(SUPPORT_BAND[0]-h, h-SUPPORT_BAND[1], 0.)
        source_rows.append(dict(id=row['id'], pts=int(pts[i]), source_seconds=float(pts[i]/12800.),
            eligible_minimum_height_m=h, support_band_violation_m=gap_violation,
            legacy_hard_ice_diagnostics=_legacy_ice_diagnostics(
                exceeds_numerical_tolerance=gap_violation > FEASIBILITY_TOLERANCE),
            model_selected_candidate=dict(foot=foot, feature=feature),
            source_physical_side=row.get('physical_side', 'unknown'), source_edge=row.get('edge', 'unknown'),
            human_confirmed=row.get('human_confirmed', False), source=row.get('source'),
            temporal_scope='reviewed exposure midpoint only; no contact-label propagation'))
    worst = np.unravel_index(np.argmax(violation), violation.shape)
    counts = dict(zip(*np.unique(packet['contact_state'], return_counts=True))) if 'contact_state' in packet else {}
    return result, dict(checked_source_exposures=len(pts), tested_foot_exposures=2*len(pts),
        source_time_base=[1, 12800], source_pts_step=256,
        signed_minimum_clearance_m=float(clearance.min()),
        max_nonpenetration_violation_m=float(violation[worst]),
        worst_nonpenetration=dict(pts=int(pts[worst[0]]), source_seconds=float(pts[worst[0]]/12800.), foot=int(worst[1])),
        foot_exposures_over_numerical_tolerance=int(np.sum(violation > FEASIBILITY_TOLERANCE)),
        nonpenetration_violation_m=_scopes(violation.ravel(), flat_pts),
        runner_penetration_depth_m=_scopes((-height[:, :, runner].min(2)).ravel(), flat_pts),
        runner_statistic='reported statistics are depth=-minimum raw runner height; positive means below ice',
        lower_skate_raw_minimum_height_m=float(height.min()),
        source_reviewed_support=source_rows, inherited_contact_state_counts=counts,
        ice_mode=ice_objective_mode(packet),current_ice_quality_verdict=None,
        legacy_hard_ice_diagnostics=_legacy_ice_diagnostics(
            physical_failure=bool(np.any(violation > FEASIBILITY_TOLERANCE) or any(
                r.get('legacy_hard_ice_diagnostics',{}).get('exceeds_numerical_tolerance',True)
                for r in source_rows if r.get('status') not in ('unknown','airborne'))),
            support_band_m=SUPPORT_BAND,numerical_feasibility_tolerance_m=FEASIBILITY_TOLERANCE),
        semantic_allowance_m=ALLOWANCE, support_band_m=SUPPORT_BAND,
        numerical_feasibility_tolerance_m=FEASIBILITY_TOLERANCE,
        numerical_threshold_scope='legacy_hard_ice_objective only; no current ice quality criterion',
        scope='source 50Hz only; caller solver owns 2.5ms dense checks; finite sampling is not continuous proof',
        unknown='unreviewed physical side/edge and inherited unknown labels stay unknown; a model minimum is only a candidate')


def review_candidate(root, out, ref, camera, info, packet, evidence):
    """Write validation.json, preserving common denominators and failed states."""
    root, out = Path(root), Path(out);out.mkdir(parents=True, exist_ok=True)
    layout = getattr(ref, 'layout', OBSERVATION_LAYOUT);pts = np.asarray(packet['pts'])
    if not np.array_equal(pts, np.arange(283136, 575489, 256)):
        raise ValueError('Review requires the complete 1143 S01 source exposures')
    if not np.array_equal(packet['time_base'], [1, 12800]):raise ValueError('Unexpected source time base')
    rows, sources = footwear_rows(root, np.arange(len(pts)), pts)
    if len(rows.blade_rows) != 7:raise ValueError('The fixed old blade batch changed; review its provenance before comparison')
    ratio = float(info['length_export_ratio']);centres = camera_centres(camera, np.arange(len(pts)))*ratio
    parents=dict(PARENTS)
    previous='skate_contact_1103/fitted/result.npz'
    if (root/previous).is_file() and (root/previous).resolve()!=(out/'fitted/result.npz').resolve():
        parents['previous_skate_1103']=previous
    variants = {name: load(root/path) for name, path in parents.items()};variants['candidate'] = packet
    # ref.parent is the accepted hand/body reference with the legacy binding.
    legacy_faces = ref.parent.binding.faces
    comparisons = {};obs = load(root/'boot_integration_1102/observations.npz')
    for name, data in variants.items():
        if not _identical(data['pts'], packet['pts']) or not _identical(data['time_base'], packet['time_base']):
            raise ValueError('Variant has a different source batch: '+name)
        faces = (ref.binding.adapter_faces if name == 'candidate' else
            np.asarray(data['boot_surface_faces']) if name=='previous_skate_1103' else legacy_faces)
        comparisons[name] = _pixels(data, rows, layout, faces, centres)
        recorded = json.loads(str(data['camera_binding_json']))
        frozen_keys = ('state_sha256', 'shot', 'metres_per_scene_unit', 'solver_metres_per_scene_unit',
                       'length_export_ratio', 'source_time_base')
        comparisons[name]['frozen_camera_fields_equal'] = {
            key: _jsonable(recorded.get(key)) == _jsonable(info.get(key)) for key in frozen_keys}
        fi, joint = np.nonzero(obs['use'])
        error = np.linalg.norm(data['COCO_uv'][fi, joint]-obs['uv'][fi, joint], axis=1)
        comparisons[name]['body_used_source_points_px'] = _scopes(error, pts[fi])
        if name != 'candidate':comparisons[name]['result'] = dict(path=str(root/parents[name]), sha256=digest(root/parents[name]))
    physical, physical_report = _physics(ref, packet, pts, evidence)
    known = []
    for seconds in (26.22, 39.78, 43.86):
        i = int(np.flatnonzero(pts == round(seconds*12800))[0]);record = dict(pts=int(pts[i]), source_seconds=seconds)
        record['legacy_sampled_blade_minimum_height_m'] = {name: data['gear_world_mid'][i, :, layout.shell_count:, 2].min(1)
            for name, data in variants.items() if name != 'candidate'}
        runner = np.array([s == 'runner' for s in physical['features']['semantics']])
        record['new_exact_runner_minimum_height_m'] = physical['features']['height'][i][:, runner].min(1)
        record['new_max_nonpenetration_violation_m'] = float(np.maximum(-physical['clearance'][i], 0.).max())
        record['source_support_candidate'] = [r for r in physical_report['source_reviewed_support'] if r['pts'] == int(pts[i])]
        record['contact_state'] = packet['contact_state'][i] if 'contact_state' in packet else ['unknown', 'unknown']
        known.append(record)
    mask = np.zeros(len(pts), bool)
    for state, _ in ref.new_refinements:mask |= np.any(state.basis(ref.times) != 0., axis=1)
    parent = variants['opening_hand_1103']
    outside = {key: _max_error(packet[key][~mask], parent[key][~mask]) for key in ('P_world_mid', 'Q_world_mid', 'local_R_mid')}
    report = dict(schema='rinkspace.same-source-skate-review.v1', user_accepted=False, independent_3d_truth=False,
        diagnosis='Model/source-development diagnostics only. Legacy hard-ice threshold results and unknown contact states are retained; no current ice quality verdict.',
        source_time_base=[1, 12800], source_seconds=[float(pts[0]/12800.), float(pts[-1]/12800.)],
        observation_sources=sources, common_denominator=dict(blade_curves=len(rows.blade_rows), blade_samples_per_curve=24,
            heel_toe_rows=len(rows.landmark_rows), contour_rows=len(rows.visible_rows),
            contour_source_samples=sum(len(r['observed']) for r in rows.visible_rows),
            rules='same observed source samples for every variant; finite heel/toe edges; one side per entire blade curve; midpoint-visible shell segments'),
        image_comparisons=comparisons, source_preservation=_source_contract(root, out, ref, packet, parent),
        physical_source_50Hz=physical_report, known_failure_exposures=known,
        body_optimization_scope=dict(intervals_seconds=[list(ws.seconds) for ws, _ in ref.new_refinements],
            active_source_exposures=int(mask.sum()), other_source_exposures='shared template replay only; no body optimization',
            outside_body_max_absolute_difference_from_hand_parent=outside),
        same_state_template_replay=_replay(ref, packet, ratio, pts),
        provenance=dict(implementation_sha256=digest(__file__), camera=info,
            source_evidence_sha256=hashlib.sha256(json.dumps(_jsonable(evidence), sort_keys=True).encode()).hexdigest(),
            candidate_path=str(out/'fitted/result.npz'),
            candidate_file_sha256=digest(out/'fitted/result.npz') if (out/'fitted/result.npz').is_file() else None))
    report = _jsonable(report)
    (out/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return report
