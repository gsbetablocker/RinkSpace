"""Independent saved-state audit. No prepare, optimizer, inference or GUI calls."""
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

P = Path('D:/OneDrive/codex_projs/RinkSpace')
D = Path('D:/CodexData/RinkSpace_Data')
B = D/'runs/body_s01_20261029'
OUT = B/'blade_modes_1105'
sys.path[:0] = [str(P), str(P/'src')]
os.environ['RINKSPACE_BODY_NATIVE_LIBRARY'] = str(D/'runs/body_perf_1103/native_cpu/rinkspace_core.dll')
from calibration.program_tracking.body_tracking.blade_projection import SavedBladeProjection
from calibration.program_tracking.body_tracking.skate_physics import batch_features, fk_state
from calibration.program_tracking.body_tracking.support_image import SupportImageFactors


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ident(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def py(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def stats(a):
    a = np.asarray(a, float)
    return dict(n=a.size, min=float(a.min()), q25=float(np.quantile(a, .25)),
                median=float(np.median(a)), q75=float(np.quantile(a, .75)),
                p95=float(np.quantile(a, .95)), max=float(a.max())) if a.size else dict(n=0)


def counts(a):
    return dict(Counter(np.asarray(a).tolist()))


def reprojections(model, document, archived_metrics, multiplier):
    """Call the real factor with zero increments around saved final controls."""
    runtime = copy.deepcopy(document)
    for row in runtime['observations']:
        row['objective_weight'] *= multiplier
    state = SimpleNamespace(m=1, basis=lambda t: np.ones((len(t), 1)),
                            increments=lambda x, t: np.zeros((len(t), 69)))
    problem = SimpleNamespace(optimize_shared=False, binding=model.binding,
                              reference=model.reference, camera=model.camera,
                              body=SimpleNamespace(state=state, tau=float(model.camera.p['rig'][37])))
    factors = SupportImageFactors(problem, runtime, include_review=True,
                                  arc_ids=archived_metrics.get('fixed_arc_ids', {}))
    current = factors.evaluate(np.empty(0), jacobian=False)
    archived = {r['id']: r for r in archived_metrics['observations']}
    delta = []
    for r in current['metrics']['observations']:
        s = archived[r['id']]
        delta.extend(np.asarray(r['uv'])-s['uv'])
        delta.extend(np.asarray(r['actual_geometry_uv'])-s['actual_geometry_uv'])
        for key in ('centre_error_px', 'outside_region_px',
                    'actual_geometry_centre_distance_px', 'actual_geometry_outside_region_px'):
            delta.append(r[key]-s[key])
    fit_factors = SupportImageFactors(problem, runtime, arc_ids=archived_metrics.get('fixed_arc_ids', {}))
    return dict(method='Real SupportImageFactors.evaluate; zero deltas about saved final absolute curve; Jacobian disabled; no optimization',
                real_fit_row_ids=[r[0]['id'] for r in fit_factors.rows],
                real_fit_groups=len(fit_factors.rows),
                real_fit_cost=float(fit_factors.evaluate(np.empty(0), jacobian=False)['cost']),
                include_review_cost=float(current['cost']),
                include_review_residual=current['residual'],
                reevaluated_metrics=current['metrics'],
                max_abs_archived_after_metric_difference=float(np.max(np.abs(delta))))


def source_export(stage, summary, model):
    document_path = stage/'support_and_free_shoe_regions.json'
    document = load(document_path)
    relation = load(stage/'grasp_relations.json')
    rows = copy.deepcopy(document['observations'])
    real = reprojections(model, document, summary['source_regions_after'],
                         relation['factor_parameters']['shoe_region_weight_multiplier'])
    fit = set(real['real_fit_row_ids'])
    review = {r['id'] for r in real['reevaluated_metrics']['observations']}
    lookup = set(model.pts.tolist())
    for row in rows:
        row.update(exact_native_pts_present=row['pts'] in lookup,
                   actual_fit_factor_admitted=row['id'] in fit,
                   actual_include_review_factor_admitted=row['id'] in review,
                   source_semantic=row.get('source_semantic', row['kind']),
                   source_visibility=row.get('source_visibility', row['visibility']))
    saved = {}
    for key in ('source_regions_before', 'source_regions_after'):
        block = copy.deepcopy(summary[key])
        for row in block['observations']:
            row['actual_minus_nominal_uv_px'] = (np.asarray(row['actual_geometry_uv'])-row['uv']).tolist()
            row['actual_minus_nominal_centre_distance_px'] = row['actual_geometry_centre_distance_px']-row['centre_error_px']
            row['actual_minus_nominal_outside_region_px'] = row['actual_geometry_outside_region_px']-row['outside_region_px']
        saved[key] = block
    groups = []
    for foot in (0, 1):
        for kind in sorted(set(r['kind'] for r in rows)):
            ids = {r['id'] for r in rows if r['foot'] == foot and r['kind'] == kind}
            before = [r for r in saved['source_regions_before']['observations'] if r['id'] in ids]
            after = [r for r in saved['source_regions_after']['observations'] if r['id'] in ids]
            groups.append(dict(foot=foot, kind=kind, rows=len(after),
                               before_actual_centre_px=stats([r['actual_geometry_centre_distance_px'] for r in before]),
                               after_actual_centre_px=stats([r['actual_geometry_centre_distance_px'] for r in after]),
                               before_actual_outside_px=stats([r['actual_geometry_outside_region_px'] for r in before]),
                               after_actual_outside_px=stats([r['actual_geometry_outside_region_px'] for r in after]),
                               before_objective_geometry_centre_px=stats([r['centre_error_px'] for r in before]),
                               after_objective_geometry_centre_px=stats([r['centre_error_px'] for r in after]),
                               before_objective_geometry_outside_px=stats([r['outside_region_px'] for r in before]),
                               after_objective_geometry_outside_px=stats([r['outside_region_px'] for r in after])))
    axis = copy.deepcopy(document.get('boot_axis_observations', []))
    for row in axis:
        row['exact_native_pts_present'] = row['pts'] in lookup
    return dict(document_identity=ident(document_path), observations=rows,
                boot_axis_observations=axis, source_id=document['source_id'],
                pixel_domain=document['pixel_domain'], time_base=document['time_base'],
                rows=len(rows), unique_pts=len({r['pts'] for r in rows}),
                roles=counts([r['role'] for r in rows]), visibility=counts([r['visibility'] for r in rows]),
                semantics=counts([r['source_semantic'] for r in rows]),
                all_exact_pts=all(r['exact_native_pts_present'] for r in rows+axis),
                per_foot_kind_before_after=groups, saved_existing_run_metrics=saved,
                fresh_readonly_saved_state_factor_evaluation=real,
                metric_scope='Before/after copied from the historical run summary, after independently reevaluated on final saved state. This audit performed no new optimization. Nominal ice projection is an objective reference; actual_geometry is actual current model geometry, neither is a certified source contact measurement.')


def profile(label, stage):
    summary_path = stage/'run_summary.json'
    summary = load(summary_path)
    model = SavedBladeProjection(P, stage, summary['parameter'])
    ref = model.reference.at(model.times)
    state = fk_state(model.reference, ref, np.zeros((len(model.pts), 69)))
    R, O = state['R'], state['O']
    f = batch_features(model.binding, R, O, state)
    h, q = f['height'], f['local_point']
    template = model.binding.packet()['template']
    eligible = template['support_feature_ids']['glide']
    rid = np.array([i for i, name in enumerate(f['ids']) if name in eligible])
    best = rid[np.argmin(h[..., rid], axis=-1)]
    ix = np.arange(len(R))[:, None]
    ft = np.arange(2)[None]
    qr = q[ix, ft, best]
    hr = h[ix, ft, best]
    p = model.binding.packet()['parameters']
    xrear = -p.runner_rear_extension_m
    xfront = p.sole_length_m-p.runner_front_inset_m
    u = (qr[..., 0]-xrear)/(xfront-xrear)
    endpoint = np.zeros_like(best)
    # Endpoint identity is the whole rocker rear/front boundary, never the
    # internal main/secondary curvature join. Use analytic theta equality.
    for a in template['contact_arcs']:
        j = f['ids'].index(a['id'])
        normal = R[..., 2, :]
        ac, asi = np.asarray(a['axis_cos']), np.asarray(a['axis_sin'])
        A, BB = normal@ac, normal@asi
        lo, hi = a['theta_start_rad'], a['theta_end_rad']
        inside = np.arctan2(-BB, -A)
        inside += 2*np.pi*np.round(((lo+hi)/2-inside)/(2*np.pi))
        ts = np.stack([np.full_like(A, lo), np.full_like(A, hi), np.clip(inside, lo, hi)], -1)
        theta = np.take_along_axis(ts, np.argmin(A[..., None]*np.cos(ts)+BB[..., None]*np.sin(ts), axis=-1)[..., None], axis=-1)[..., 0]
        if a['id'].endswith('_main'):
            endpoint[(best == j) & (theta == lo)] = -1
        elif a['id'].endswith('_secondary'):
            endpoint[(best == j) & (theta == hi)] = 1
    raw = np.argmin(h, axis=-1)
    ids = np.asarray(f['ids'])
    semantics = np.asarray(f['semantics'])
    F, L, U = R[..., :, 0], R[..., :, 1], R[..., :, 2]
    dot = lambda v: np.clip(v[..., 2], -1., 1.)
    asin = lambda v: np.degrees(np.arcsin(dot(v)))
    acos = lambda v: np.degrees(np.arccos(dot(v)))
    fore = asin(F)
    bank = np.degrees(np.arctan2(L[..., 2], U[..., 2]))
    transverse = asin(L)
    lean = acos(U)
    plane = np.degrees(np.arccos(np.abs(dot(L))))
    plane_oriented = acos(L)
    old_pitch = np.degrees(np.arctan2(R[..., 2, 0], np.hypot(R[..., 0, 0], R[..., 1, 0])))
    world_min = (O+np.einsum('nfij,nfj->nfi', R, qr))*model.ratio
    with np.load(summary['result'], allow_pickle=False) as z:
        saved_R_err = float(np.max(np.abs(z['skate_R_world_mid']-R)))
        saved_O_err = float(np.max(np.abs(z['skate_origin_world_mid']-O*model.ratio)))
        saved_upper_height_err = float(np.max(np.abs(np.min(z['skate_upper_world_mid'][..., 2], axis=-1)-h[..., -1])))
        saved_binding = json.loads(str(z['footwear_binding_json']))
    neutral = model.reference.body.J
    final_R_FB = model.binding.parameters()[0]
    edge_map = []
    for foot, ankle in enumerate((7, 8)):
        medial = (neutral[(8, 7)[foot]]-neutral[ankle]).copy()
        norm = float(np.linalg.norm(medial))
        medial_unit = medial/norm if norm > 1e-12 else np.zeros(3)
        lateral = final_R_FB[foot, :, 1]
        alignment = float(lateral@medial_unit)
        side = 'positive_y' if alignment > 1e-8 else 'negative_y' if alignment < -1e-8 else 'unknown'
        edge_map.append(dict(foot=foot, neutral_native_ankle=ankle,
                             neutral_ankle_position=neutral[ankle],
                             medial_toward_other_neutral_ankle=medial,
                             medial_unit=medial_unit, final_R_FB=final_R_FB[foot],
                             local_positive_y_in_native= lateral,
                             positive_y_dot_medial=alignment,
                             conditional_inside_local_y=side,
                             conditional_outside_local_y=('negative_y' if side == 'positive_y' else 'positive_y' if side == 'negative_y' else 'unknown'),
                             actual_current_edge_state='unknown',
                             basis='Fixed neutral native ankle 7/8 anatomy and final saved R_FB; never current lowest feature. Model anatomy conditional mapping only; not source-confirmed physical foot or contact.'))
    arr = dict(pts=model.pts, seconds=model.pts/12800., R_world=R,
               origin_world_conditional_m=O*model.ratio, forward_world=F,
               lateral_world=L, up_world=U,
               ice_normal_world=np.broadcast_to([0., 0., 1.], F.shape),
               fore_aft_signed_deg=fore, bank_deg=bank, transverse_signed_deg=transverse,
               total_lean_deg=lean, blade_plane_normal_world=L,
               blade_plane_normal_unoriented_to_ice_normal_deg=plane,
               blade_plane_normal_oriented_to_ice_normal_deg=plane_oriented,
               legacy_pitch_deg=old_pitch, legacy_roll_deg=bank,
               runner_u=u, whole_runner_endpoint=endpoint,
               runner_feature=ids[best], runner_min_height_conditional_m=hr,
               runner_min_local_point_solver=qr,
               runner_min_world_conditional_m=world_min,
               feature_ids=ids, feature_semantics=semantics,
               all_feature_min_height_conditional_m=h,
               all_feature_min_local_point_solver=q,
               raw_lowest_feature=ids[raw], raw_lowest_semantic=semantics[raw],
               raw_lowest_height_conditional_m=np.min(h, axis=-1))
    npz_path = OUT/(label+'_diagnostics_951.npz')
    np.savez_compressed(npz_path, **arr)
    mask = (model.pts >= 1111040) & (model.pts <= 1183488)
    ss = endpoint[mask, 1]
    seq = ids[best][mask, 1]
    switch = np.flatnonzero(seq[1:] != seq[:-1])+1
    pwin = model.pts[mask]
    changes = [dict(pts=int(pwin[i]), seconds=float(pwin[i]/12800.),
                    from_feature=seq[i-1], to_feature=seq[i],
                    same_local_y_side=seq[i-1].rsplit('_', 1)[0] == seq[i].rsplit('_', 1)[0],
                    meaning='rocker main/secondary curvature-segment migration; not inside/outside edge change') for i in switch]
    window = dict(foot=1, first_last_pts=pwin[[0, -1]], exposure_count=int(mask.sum()),
                  toe_pick_lowest=int(np.sum(semantics[raw][mask, 1] == 'toe_pick')),
                  runner_front_endpoint=int(np.sum(ss == 1)), runner_interior=int(np.sum(ss == 0)),
                  runner_rear_endpoint=int(np.sum(ss == -1)),
                  runner_pitch_median_deg=float(np.median(old_pitch[mask, 1])),
                  runner_roll_median_deg=float(np.median(bank[mask, 1])),
                  runner_height_median=float(np.median(hr[mask, 1])),
                  runner_u_stats=stats(u[mask, 1]), runner_feature_counts=counts(seq),
                  runner_feature_switches=[r['seconds'] for r in changes], switch_details=changes,
                  angles={key:stats(arr[key][mask, 1]) for key in ('fore_aft_signed_deg', 'bank_deg', 'transverse_signed_deg', 'total_lean_deg', 'blade_plane_normal_unoriented_to_ice_normal_deg', 'blade_plane_normal_oriented_to_ice_normal_deg')})
    record = dict(stage=str(stage), parameter=ident(summary['parameter']),
                  result=ident(summary['result']), run_summary=ident(summary_path),
                  source_frames=len(model.pts), first_last_pts=model.pts[[0, -1]],
                  length_export_ratio=model.ratio,
                  saved_final_FK_max_abs_errors=dict(R=saved_R_err, origin_conditional_m=saved_O_err,
                                                    deformed_upper_min_height_conditional_m=saved_upper_height_err),
                  fixed_registration=saved_binding,
                  conditional_edge_anatomy_mapping=edge_map,
                  independent_arrays=ident(npz_path), support_window=window,
                  source=source_export(stage, summary, model),
                  existing_run_status={k:summary[k] for k in ('accepted_steps', 'retained_step', 'stop', 'wall_seconds', 'initial_parts', 'final_parts', 'accuracy_accepted', 'user_accepted', 'numerical_convergence', 'not_default') if k in summary})
    print(json.dumps(dict(event='profile_complete', candidate=label, window=window,
                          saved_final_FK_max_abs_errors=record['saved_final_FK_max_abs_errors'],
                          source_after_max_abs_error=record['source']['fresh_readonly_saved_state_factor_evaluation']['max_abs_archived_after_metric_difference']), default=py), flush=True)
    return record


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = D/'handoff/1105/RinkSpace_4a06f11e_Blade_Audit_Codex_Integration.zip'
    with zipfile.ZipFile(bundle) as z:
        expected = json.loads(z.read('AUDIT_METRICS.json'))
    profiles = {name:profile(name, path) for name, path in (
        ('boot_axis_entry', B/'blade_orientation_1105/entry_81_2_82_0'),
        ('runner_support', B/'blade_support_profile_1105/runner_support_run'))}
    checks = []
    for name in profiles:
        actual = profiles[name]['support_window']
        for key, value in expected['runner_mode'][name].items():
            found = actual[key]
            checks.append(dict(candidate=name, key=key, expected=value, actual=found,
                               difference=float(found-value), match=bool(np.isclose(found, value, rtol=0, atol=1e-10))))
    for key in ('runner_feature_counts', 'runner_feature_switches'):
        value = expected['runner_mode'][key]
        found = profiles['runner_support']['support_window'][key]
        checks.append(dict(candidate='runner_support', key=key, expected=value, actual=found, match=found == value))
    obs = load(B/'blade_observability_1105/blade_observability_audit.json')
    code_dir = P/'calibration/program_tracking/body_tracking'
    record = dict(schema='rinkspace.blade-modes-independent-reproduction.v1',
                  head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
                  target_commit=expected['baseline_commit'], integration_bundle=ident(bundle),
                  script=ident(Path(__file__)),
                  code_identities=[ident(code_dir/n) for n in ('blade_projection.py', 'skate_physics.py', 'skate_binding.py', 'skate_template.py', 'support_image.py', 'support_anchor_factors.py', 'runner_support_factors.py')],
                  boundary='Read saved final absolute controls + frozen binding; current FK, batch_features and real SupportImageFactors reevaluation. No prepare/parent-chain replay, optimizer, network inference, GUI, image rereview, or modification of existing source/results.',
                  definitions=dict(axes='F=R e_x heel-to-toe; L=R e_y lateral; U=R e_z boot-up; n=[0,0,1]',
                                   fore_aft_signed_deg='degrees(asin(F dot n))',
                                   bank_deg='degrees(atan2(L dot n,U dot n))',
                                   transverse_signed_deg='degrees(asin(L dot n))',
                                   total_lean_deg='degrees(acos(U dot n))',
                                   blade_plane_normal='L',
                                   blade_plane_normal_unoriented_to_ice_normal_deg='degrees(acos(abs(L dot n)))',
                                   blade_plane_normal_oriented_to_ice_normal_deg='degrees(acos(L dot n))',
                                   runner_u='(local_x-runner_rear_x)/(runner_front_x-runner_rear_x)',
                                   whole_runner_endpoint='-1 rear, +1 front, 0 interior; exact analytic arc parameter at whole rocker boundary; main/secondary join is interior',
                                   units='All reported heights/world arrays are conditional exported metres, scale ratio applied exactly once. Local arrays labeled solver retain solver units. All angles are model diagnostics, never source measurements.',
                                   feature_switch='main/secondary are rocker curvature segments on the same y side. Four observed runner feature switches are not evidence of inside/outside edge change.'),
                  profiles=profiles, expected_value_checks=checks,
                  all_audit_targets_match=all(r['match'] for r in checks),
                  existing_visual_review_ledger_identity=ident(B/'blade_observability_1105/blade_observability_audit.json'),
                  existing_visual_review_ledger=obs['reviewed_source_exposures'],
                  source_observability=dict(reviewed_pts=len(obs['reviewed_source_exposures']),
                                            unreviewed_exposures=951-len(obs['reviewed_source_exposures']),
                                            source_role='Saved historical source-only review, copied without image rereview or promotion into factors.',
                                            unknown_coverage=obs['unknown_coverage'], window_coverage=obs['window_coverage']),
                  physical_contact_truth='unknown', actual_edge_mode='unknown',
                  accuracy_accepted=False, new_optimization_performed=False,
                  elapsed_seconds=time.perf_counter()-started)
    (OUT/'reproduction.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, default=py, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(OUT/'reproduction.json'), all_audit_targets_match=record['all_audit_targets_match'],
                          elapsed_seconds=record['elapsed_seconds'], edge_mappings={n:r['conditional_edge_anatomy_mapping'] for n,r in profiles.items()}), default=py), flush=True)


if __name__ == '__main__':
    main()
