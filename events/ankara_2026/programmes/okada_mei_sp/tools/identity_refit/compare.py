"""Measure A, fixed-pose B and refitted C on the registered native clock."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.programme_projection import generate, merge_review_assets
from calibration.program_tracking.body_tracking.programme_whole_skate import (
    DirectedBootSourceFactor, WholeSkateSourceFactor,
)
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from calibration.program_tracking.body_tracking.sequence_skate_state import SequenceState
from experiments.skate_video_v1.programme_motion_followup import bind as bind_motion, changed_support
from prepare import write
from source_review import BASELINE, NAMES, OPPOSITE, PAIRS, PROGRAMME, PROJECT, baseline, digest, read


NEW = 'okada_body_identity_1114'


def bind(prepared, evidence, parameter):
    result = bind_motion(prepared, evidence, parameter)
    ref, _, _, _, packet = result
    if 'fitted_programme_body_scale' in packet:
        from shared_stature import apply_body_scale
        apply_body_scale(ref, packet['fitted_programme_body_scale'])
    return result

GROUPS = dict(face=[0,1,2,3,4], torso=[5,6,11,12], elbow=[7,8], wrist=[9,10], knee=[13,14], ankle=[15,16])


def ranges(mask, pts):
    ix = np.flatnonzero(mask)
    if not len(ix):
        return []
    parts = np.split(ix, np.flatnonzero(np.diff(ix) > 1)+1)
    return [dict(first_pts=int(pts[r[0]]), last_pts=int(pts[r[-1]]), exposures=len(r),
                 seconds=(pts[r[[0,-1]]]/12800).tolist()) for r in parts]


def distribution(values, at_pts, use=None):
    values = np.asarray(values)
    stamps = np.broadcast_to(np.asarray(at_pts).reshape((-1,)+(1,)*(values.ndim-1)), values.shape)
    mask = np.isfinite(values) if use is None else np.isfinite(values) & use
    raw = values[mask]
    if not len(raw):
        return dict(n=int(len(raw)), worst_pts=None)
    quantiles = np.percentile(raw,[50,90,95,99,100]).tolist()
    return dict(n=len(raw), **dict(zip(('p50','p90','p95','p99','max'), quantiles)),
                gt100=int(np.sum(raw>100)), worst_pts=int(stamps[mask][raw.argmax()]))


def result_fields(path, keys):
    with np.load(path, allow_pickle=False) as saved:
        return {k:saved[k] for k in keys}


def rotation_step(a):
    k = a[:-1].swapaxes(-1,-2)@a[1:]
    return np.rad2deg(Rotation.from_matrix(k.reshape(-1,3,3)).magnitude()).reshape(k.shape[:-2])


def stage_distributions(motion, anatomy, pts, take):
    use_step = take[1:] & take[:-1]
    shoe = motion['skate_R_world_mid']
    position = motion['skate_origin_world_mid']
    return dict(
        joint_domain={str(j):dict(positive_exposures=int(np.count_nonzero(np.any(anatomy['normalized_violation'][take,i]>0,axis=-1))),
            **distribution(anatomy['normalized_violation'][take,i].max(-1),pts[take]))
            for i,j in enumerate(anatomy['joint_names'])},
        local_so3_deg=distribution(rotation_step(motion['native_body_local_R_mid'])[use_step],pts[1:][use_step]),
        shoe_so3_deg=distribution(rotation_step(shoe)[use_step],pts[1:][use_step]),
        shoe_axis_deg=distribution(np.degrees(np.arccos(np.clip(np.sum(shoe[:-1,:,:,0]*shoe[1:,:,:,0],axis=-1),-1.,1.)))[use_step],pts[1:][use_step]),
        shoe_step_m=distribution(np.linalg.norm(np.diff(position,axis=0),axis=-1)[use_step],pts[1:][use_step]),
        foot_cavity_m=distribution(motion['foot_cavity_max_violation_m_mid'][take],pts[take]),
        runner_clearance_m=distribution(motion['skate_clearance_m_mid'][take],pts[take]),
        heel_drift_m=distribution(motion['heel_drift_mid'][take],pts[take]),
        forefoot_error_m=distribution(motion['forefoot_error_mid'][take],pts[take]),
        com_acceleration_mps2=distribution(np.linalg.norm(motion['com_acceleration_world_mps2'][take],axis=-1),pts[take]),
        contact_candidate_modes=dict(Counter(map(str,motion['source_support_mode_mid'][take].ravel()))),
        measured_contact_count=0)


def skate_rows(a, c, pts):
    left, right = ({r['id']:r for r in v['source_and_physics']['new_source_skate_regions']} for v in (a,c))
    if set(left) != set(right):
        raise ValueError('Shoe contour populations differ')
    for key, old in left.items():
        new = right[key]
        for k in ('pts','foot','role','source_polyline_uv','pixel_uncertainty_radius_px'):
            if old[k] != new[k]:
                raise ValueError('Shoe contour source changed: '+key)
    def scores(doc):
        keep = [r for r in doc.values() if r['pts'] in pts]
        d = [(p,r['pts']) for r in keep for p in r['distance_px']]
        return dict(source_feature_rows=len(keep), exposures=len({r['pts'] for r in keep}),
                    pixel_distance=distribution(*np.array(d).reshape(-1,2).T))
    return dict(A_B=scores(left), C=scores(right))


def skate_reprojection(reference, camera, document, controls):
    """Replay the saved curves on the existing native shoe factors."""
    pts = reference.p['pts']
    state = SequenceState(reference, controls['seconds'])
    delta = np.zeros(state.size)
    problem = SimpleNamespace(reference=reference, camera=camera, binding=reference.binding,
                              body=SimpleNamespace(state=state))
    spec = document['whole_skate_source']
    whole = WholeSkateSourceFactor(problem, spec)
    whole.restore_checkpoint(controls)
    whole.evaluate(delta, False)
    directed = np.full((len(pts), 2), np.nan)
    rows = DirectedBootSourceFactor(problem, spec['directed_boot_rows']).evaluate(delta, False)
    for row in rows['metrics']['directed_boot_source']:
        directed[np.searchsorted(pts, row['pts']), row['foot']] = row['distance_px']
    return dict(outline_median_distance_px=whole.last_distances, boot_chord_distance_px=directed)


def load_surface(stage):
    summary = json.loads(read(stage/'retained_surface.json'))
    path = Path(summary['surface_path'])
    if digest(path) != summary['surface_sha256']:
        raise ValueError('Surface scan does not match retained candidate')
    return summary, result_fields(path,('source_seconds','crossing_counts','grip_crossing_counts','topology_uncertain_candidates'))


def audit(data, prepared, run, inputs, stage, output, *, surface_base=None):
    scope, paths, old = baseline(data)
    parent = paths['solve_state'].parents[1]
    registered = json.loads(read(BASELINE))
    new_path = stage/'fitted/result.npz'
    parameter = stage/'solve/candidate.npz'
    new = result_fields(new_path, old.keys())
    obs = load(inputs/'observations.npz')
    revision = json.loads(read(inputs/'revision.json'))
    verified_review = json.loads(read(PROGRAMME/'fit_v11/body_identity_1114_review.json'))
    fit_review = revision['source_review_events']
    # A source-evidence clarification can be appended after a ledger is frozen
    # only when every affected field of every decision is exactly the same.
    evidence_fields = ('reason','source_sheet','full_source_sheet')
    def decision_identity(events):
        return [{k:v for k,v in e.items() if k not in evidence_fields} for e in events]
    if decision_identity(fit_review) != decision_identity(verified_review['events']):
        raise ValueError('Published source decisions differ from the fitted ledger')
    revision['source_review_events'] = verified_review['events']
    pts, shot = old['source_pts'], old['source_shot_ids']
    for k in ('source_pts','time_base','source_shot_ids','source_uv_native','source_sigma_px'):
        np.testing.assert_array_equal(old[k],new[k],err_msg=k)
    for k,j in (('source_pts','pts'),('source_observation_objective_use','use'),('source_observation_use','use'),
                ('source_body_model_joint_ids','model_joint_ids'),('source_uv_native','uv'),('source_sigma_px','sigma_px')):
        np.testing.assert_array_equal(new[k],obs[j],err_msg=k)
    assert digest(inputs/'observations.npz') == str(new['source_observation_ledger_sha256'])
    fixed = ('camera_binding_json','shared_betas','shared_body_scale','skate_dimensions_json','template_version',
             'binding_version','body_length_ratio','metres_per_scene_unit','solver_metres_per_scene_unit')
    before, after = (result_fields(p,fixed) for p in (paths['fitted_state'],new_path))
    for k in fixed:
        np.testing.assert_array_equal(before[k],after[k],err_msg=k)
    r0,camera,info,doc,p0 = bind(prepared,paths['source_evidence'],paths['solve_state'])
    r1,camera1,_,doc1,p1 = bind(prepared,inputs/'source.json',parameter)
    for k in ('base_skate_parameters_json','R_FB_initial','t_FB_initial_solver','fixed_boot_geometry_json',
              'relative_skate_rotation_controls','absolute_knots','sequence_contact_modes','sequence_contact_status',
              'action_objective_root_anchor_solver'):
        np.testing.assert_array_equal(p0[k],p1[k],err_msg=k)
    np.testing.assert_array_equal(p0['x'][:17],p1['x'][:17])
    np.testing.assert_array_equal(p0['whole_skate_match_sha256'],p1['whole_skate_match_sha256'])
    if info['state_sha256'] != registered['native_camera_sha256'] or info['scene_values_sha256'] != registered['shared_scene_values_sha256']:
        raise ValueError('Fixed camera/scene changed')
    changed = changed_support(p0,p1,r0.times)
    dp = p1['absolute_position_controls']-p0['absolute_position_controls']
    dr = p0['absolute_rotation_controls'].swapaxes(-1,-2) @ p1['absolute_rotation_controls']
    magnitude = np.rad2deg(Rotation.from_matrix(dr.reshape(-1,3,3)).magnitude()).reshape(dr.shape[:2])
    control_report = dict(root_translation_controls=int(np.any(dp!=0,axis=1).sum()),root_translation_peak_m=float(np.linalg.norm(dp,axis=-1).max()*r0.binding.ratio),
                          root_and_local_rotation_controls=(magnitude>1e-9).sum(0).tolist(),
                          root_and_local_rotation_peak_deg=magnitude.max(0).tolist(),exact_spline_support=ranges(changed,pts))

    original_use, new_use = old['source_observation_objective_use'], new['source_observation_objective_use']
    prior_ids, ids = old['source_body_model_joint_ids'], new['source_body_model_joint_ids']
    uv = old['source_uv_native']; f = np.arange(len(pts))[:,None]
    error = {name:np.linalg.norm(pred[f,assignment]-uv,axis=-1) for name,pred,assignment in (
        ('A',old['COCO_uv'],prior_ids),('B',old['COCO_uv'],ids),('C',new['COCO_uv'],ids))}
    remapped = prior_ids!=ids
    np.testing.assert_array_equal(remapped.sum(),revision['changed_correspondence_entries'])
    np.testing.assert_array_equal(np.sum(original_use & ~new_use),revision['changed_use_entries'])
    reviewed_unknown = np.zeros_like(original_use)
    for e in revision['source_review_events']:
        if e['status'] in ('unknown_keep', 'unknown_retired'):
            row_indices = np.searchsorted(pts, e['observation_pts'])
            reviewed_unknown[np.ix_(row_indices, e['observed_slots'])] = True
    retired_unknown = reviewed_unknown & original_use & ~new_use
    # Where only a subset of an unknown event is retired, its other source
    # points still belong to their visible chains and remain eligible.
    uncertain = np.zeros_like(original_use)
    for e in revision['source_review_events']:
        if e['status']=='unknown_keep':
            uncertain[np.ix_(np.searchsorted(pts,e['observation_pts']),e['observed_slots'])] = True
    scan = load(run/'scan/scan.npz')
    flagged = scan['legacy']|scan['locator']|scan['source_swap_boundary']|scan['source_unilateral_boundary']
    for pair, slot in enumerate(PAIRS):
        flagged[:,slot] |= scan['possible_duplicate'][:,pair,None]
    supported_model = np.zeros_like(original_use)
    for e in revision['source_review_events']:
        if e['status']=='model_pose':
            supported_model[np.ix_(np.searchsorted(pts,e['observation_pts']),e['observed_slots'])] = True
    # Apply the conservative comparison mask identically to A, B and C.
    common=original_use & new_use
    unresolved=common & flagged & ~remapped & ~supported_model
    flagged_inactive=flagged & ~original_use
    comparable=common & ~(unresolved|uncertain)
    groups = dict(programme=np.ones(len(pts),bool), **{s['id']:shot==s['id'] for s in scope['shots']})
    for e in doc['source_episode_review']['episodes']:
        lo,hi = e['start_pts'],e['end_pts_inclusive']
        selected = (pts>=lo)&(pts<=hi)
        if np.any(selected & changed) and e['kind'] in ('rotation_source_path','flight_visible_candidate',
                'support_candidate','step_relation','hand_to_raised_shoe_grasp_candidate'):
            groups[e['episode_id']] = selected
    groups['edited_neighbourhoods'] = changed

    def abc(take, role):
        return {name:distribution(err[take],pts[take],role[take]) for name,err in error.items()}

    group_stats = {}
    for name,take in groups.items():
        counts = dict(exposures=int(take.sum()), changed_exposures=int((take&changed).sum()),
                      slot_capacity=int(take.sum())*17, finite_source_points=int(np.isfinite(uv[take]).all(axis=-1).sum()),
                      prepared_initial_use=int(old['source_observation_initial_use'][take].sum()),
                      original_admitted=int(old['source_observation_use'][take].sum()), original_objective=int(original_use[take].sum()),
                      revised_admitted=int(new_use[take].sum()), revised_objective=int(new['source_observation_objective_use'][take].sum()),
                      historical_mapped_entries=int((prior_ids[take]!=np.arange(17)).sum()), new_mapped_entries=int(remapped[take].sum()),
                      retired=int((original_use & ~new_use)[take].sum()), added=int((new_use & ~original_use)[take].sum()),
                      uncertain=int((uncertain&common)[take].sum()), reviewed_unknown_retired=int(retired_unknown[take].sum()),
                      unresolved_candidates=int(unresolved[take].sum()),
                      previously_inactive_candidates=int(flagged_inactive[take].sum()))
        group_stats[name]=dict(counts=counts, all_common=abc(take,common), reviewed_comparison=abc(take,comparable),
            by_joint_family={k:dict(exposures=int(take.sum()),screen_population=int((flagged&original_use)[take][:,j].sum()),
                original_objective=int(original_use[take][:,j].sum()), revised_objective=int(new_use[take][:,j].sum()),
                conservative={s:distribution(e[take][:,j],pts[take],comparable[take][:,j]) for s,e in error.items()})
                for k,j in GROUPS.items()})

    motion_keys=('P_world_mid','Q_world_mid','native_body_local_R_mid','skate_R_world_mid','skate_origin_world_mid',
                 'foot_cavity_max_violation_m_mid','skate_clearance_m_mid','com_acceleration_world_mps2',
                 'source_support_mode_mid','com_world_mid','palm_to_segment_m_mid','native22_world_mid',
                 'heel_drift_mid','forefoot_error_mid')
    motion=[result_fields(p,motion_keys) for p in (paths['fitted_state'],new_path)]
    anatomy=[result_fields(p/'anatomical_frames.npz',('joint_names','source_pts','normalized_violation')) for p in (parent,stage)]
    for z in anatomy:
        np.testing.assert_array_equal(z['source_pts'], pts)
    delta={key:float(np.max(np.abs(motion[0][key][~changed]-motion[1][key][~changed]),initial=0.)) for key in (
        'P_world_mid','Q_world_mid','native_body_local_R_mid','native22_world_mid','skate_R_world_mid','skate_origin_world_mid','com_world_mid')}
    if max(delta.values())>1e-8:
        raise ValueError('Remote spline complement changed: '+repr(delta))
    actions=[json.loads(read(p/'final_action_factors.json')) for p in (parent,stage)]
    physical=[json.loads(read(p/'final_metrics.json')) for p in (parent,stage)]
    surface=[load_surface(p) for p in (parent,stage if surface_base is None else surface_base)]
    shoe_pixels = [skate_reprojection(*args) for args in ((r0,camera,doc,p0), (r1,camera1,doc1,p1))]
    for metric in shoe_pixels[0]:
        np.testing.assert_array_equal(np.isfinite(shoe_pixels[0][metric]), np.isfinite(shoe_pixels[1][metric]))
    replay = load(stage/'scoped_guard_baseline.npz')
    for key, metric in (('silhouette','outline_median_distance_px'), ('directed','boot_chord_distance_px')):
        np.testing.assert_allclose(shoe_pixels[0][metric], replay[key], rtol=0, atol=1e-6)
    np.savez_compressed(stage/'skate_source_replay.npz', source_pts=pts,
                        **{f'{k}_{role}': v for role, replay in zip(('AB','C'),shoe_pixels,strict=True)
                           for k,v in replay.items()})
    for name,take in groups.items():
        if take.any():
            g=group_stats[name]
            g['motion']={s:stage_distributions(m,a,pts,take) for s,m,a in zip(('A_B','C'),motion,anatomy,strict=True)}
            g['change_m']={key:distribution(np.linalg.norm((motion[1][key]-motion[0][key])[take],axis=-1),pts[take])
                           for key in ('P_world_mid','skate_origin_world_mid','native22_world_mid')}
            g['skate_source_contours_px']=skate_rows(*actions,set(map(int,pts[take])))
            g['skate_projection_px'] = {
                role: {metric: distribution(v[take], pts[take]) for metric,v in shoe.items()}
                for role,shoe in zip(('A_B','C'),shoe_pixels,strict=True)}
            rows=[]
            for receipt,z in surface:
                sample_pts=np.rint(z['source_seconds']*12800).astype(int)
                left=np.searchsorted(pts,sample_pts,side='right')-1
                right=np.searchsorted(pts,sample_pts,side='left')
                selected=take[left] & take[right]
                rows.append({k:int(np.sum(z[k][selected])) for k in ('crossing_counts','grip_crossing_counts','topology_uncertain_candidates')})
            g['surface_counts']={'A_B':rows[0],'C':rows[1]}

    bounds=np.unique(np.r_[np.flatnonzero(shot[1:]!=shot[:-1])+1,np.flatnonzero(changed[1:]!=changed[:-1])+1])
    boundary=[]
    for i in bounds:
        entry=dict(source_pts_before=int(pts[i-1]),source_pts_after=int(pts[i]),
                   kind='cut' if shot[i-1]!=shot[i] else 'changed_support')
        for key,m in zip(('A_B','C'),motion,strict=True):
            r=slice(i-1,i+1)
            entry[key]=dict(root_step_m=float(np.linalg.norm(np.diff(m['P_world_mid'][r],axis=0))),
                skate_step_m=np.linalg.norm(np.diff(m['skate_origin_world_mid'][r],axis=0),axis=-1)[0].tolist(),
                skate_step_deg=rotation_step(m['skate_R_world_mid'][r])[0].tolist(),
                joint_step_peak_deg=float(rotation_step(m['native_body_local_R_mid'][r]).max()))
        before, after = entry['A_B'], entry['C']
        entry['delta_C_minus_B'] = {
            k:(np.asarray(after[k])-np.asarray(v)).tolist() for k,v in before.items()}
        boundary.append(entry)
    turn={}
    for e in doc['source_episode_review']['episodes']:
        name=e['episode_id']
        key=f'episode/{name}/metrics'
        metrics=[v['source_and_physics'].get(key) for v in actions]
        if all(m and 'path_source_pts' in m for m in metrics):
            rows=np.searchsorted(pts,metrics[0]['path_source_pts'])
            signs=[m['selected_direction_sign'] for m in metrics]
            assert signs[0]==signs[1]
            turn[name]=dict(source_first_last=pts[rows[[0,-1]]].tolist(), selected_sign=signs[0], states={})
            for label,m,phase in zip(('A_B','C'),motion,metrics,strict=True):
                R=m['Q_world_mid'][rows]; step=Rotation.from_matrix(R[1:]@R[:-1].swapaxes(-1,-2)).as_rotvec()[:,2]
                turn[name]['states'][label]=dict(signed_turns_z=float(step.sum()/(2*np.pi)),
                    reverse_steps=int(np.sum(signs[0]*step<0)),peak_abs_step_deg=float(np.rad2deg(np.max(abs(step)))),
                    source_facing_rows=phase['facing_rows'])
    grasp={s:[{k:e[k] for k in ('hand','foot','source_seconds','samples','median_distance_m','p95_distance_m','max_distance_m')}
             for e in p['physical_diagnostics']['source_supported_grasp']['episodes']] for s,p in zip(('A_B','C'),physical,strict=True)}
    anchors=[]
    for stamp,j in ((2009600,10),(2100992,15),(2350336,15)):
        i=np.searchsorted(pts,stamp)
        anchors.append(dict(pts=stamp,slot=j,A_px=float(error['A'][i,j]),B_px=float(error['B'][i,j]),C_px=float(error['C'][i,j]),
            opposite_A_px=float(np.linalg.norm(old['COCO_uv'][i,OPPOSITE[prior_ids[i,j]]]-uv[i,j])),
            model_id_A_B_C=[int(prior_ids[i,j]),int(ids[i,j]),int(ids[i,j])],objective_A_B_C=[bool(original_use[i,j]),bool(new_use[i,j]),bool(new_use[i,j])],
            events=[e['id'] for e in revision['source_review_events'] if stamp in e['observation_pts']]))
    model_pose_cases = []
    for event in revision['source_review_events']:
        if event['status']!='model_pose':
            continue
        rows = np.searchsorted(pts,event['observation_pts'])
        slots = event['observed_slots']
        model_pose_cases.append(dict(event=event['id'], source_pts=pts[rows].tolist(), slots=slots,
            px_B=error['B'][np.ix_(rows,slots)].tolist(), px_C=error['C'][np.ix_(rows,slots)].tolist()))

    output.mkdir(exist_ok=False)
    with (output/'candidate_events.csv').open('x',newline='',encoding='utf-8-sig') as stream:
        w=csv.writer(stream)
        w.writerow(['shot','first_pts','last_pts','observation_pts','slots','status','origin','evidence_pts','details','full_source_sheet'])
        for e in revision['source_review_events']:
            t=e['observation_pts']
            slots=sorted({j for pair in e.get('swap_pairs',[]) for j in pair}|set(e.get('retire_slots',[]))
                         |set(e.get('observed_slots',[]))|{r['slot'] for r in e.get('retire_rows',[])})
            w.writerow([e['shot'],t[0],t[-1],';'.join(map(str,t)),';'.join(map(str,slots)),e['status'],e['source_sheet'],
                        ';'.join(map(str,e['evidence_pts'])),e['reason'],e.get('full_source_sheet','')])
        for status, mask in (('unresolved_candidate_keep',unresolved&~uncertain),
                             ('screen_only_originally_inactive',flagged_inactive)):
            for j,name in enumerate(NAMES):
                for shot_id in np.unique(shot):
                    for span in ranges(mask[:,j]&(shot==shot_id),pts):
                        lo,hi=span['first_pts'],span['last_pts']; i,jj=np.searchsorted(pts,[lo,hi]); rr=slice(i,jj+1)
                        triggers=[key for key in ('legacy','locator','source_swap_boundary','source_unilateral_boundary')
                                  if scan[key][rr,j].any()]
                        w.writerow([shot_id,lo,hi,';'.join(map(str,pts[rr])),j,status,
                                    ';'.join(triggers) or 'possible_duplicate','',name,''])
    for tag, use in (('surviving_remapped',common&remapped),('retired',original_use&~new_use),('model_pose_reviewed',common&supported_model)):
        group_stats[tag]=abc(np.ones(len(pts),bool),use)
    if not changed.any():
        raise ValueError('Refitted candidate must contain an actual saved motion update')
    guard=json.loads(read(stage/'final_metrics.json'))['primary_guards']
    report=dict(baseline=registered['candidate_id'],candidate=NEW,pts=pts[[0,-1]].tolist(),native_exposures=len(pts),
        time_base=old['time_base'].tolist(),reference_hashes=dict(camera=info['state_sha256'],scene=info['scene_values_sha256'],video=scope['source']['sha256']),
        by_scope=group_stats,source_scan={k:dict(rows=int(v.sum()),exposures=int(v.any(1).sum())) for k,v in scan.items() if v.dtype==bool},
        anchor_rows=anchors, model_pose_cases=model_pose_cases, controls=control_report,
        boundary_steps=boundary, remote_state_max_abs_delta=delta,
        whole_skate_outline_population=dict(exposures=len(np.unique(p0['whole_skate_match_frames'])),
            source_exposure_foot_rows=len(p0['whole_skate_match_feet']),fixed_correspondence=True),
        turn_direction_and_phase=turn, source_supported_grasp=grasp, surface_receipts=[s[0] for s in surface],
        physical_diagnostics=[{k:p['physical_diagnostics'][k] for k in ('surface_intersection','actual_model_com_acceleration')} for p in physical],
        weighted_parts=dict(A=physical[0]['weighted_parts'],
            B=json.loads(read(stage/'initial_metrics.json'))['weighted_parts'],C=physical[1]['weighted_parts']),
        fixed_pose_replay=json.loads(read(stage/'scoped_import.json'))['same_pose_objective'],
        guard_failures={k:v for k,v in guard.items() if v['rejected']},
        run=json.loads(read(stage/'run_summary.json')),accuracy_accepted=False,user_accepted=False)
    write(output/'abc.json',report)
    print(json.dumps(dict(metrics=str(output/'abc.json'),anchors=anchors,programme=group_stats['programme']['all_common']),ensure_ascii=False),flush=True)


def projections(data, prepared, run, inputs, stage):
    _, paths, _ = baseline(data)
    a = run/'projection_A'
    c = stage/'projection'
    args = dict(review_evidence=paths['source_evidence'], allow_missing_review_source=True)
    old_id=json.loads(read(BASELINE))['candidate_id']
    for projected, parameter, result, source, variant in (
        (a,paths['solve_state'],paths['fitted_state'],paths['source_evidence'],old_id),
        (c,stage/'solve/candidate.npz',stage/'fitted/result.npz',inputs/'source.json',NEW)):
        if not (projected/'projection_registration.json').is_file():
            generate(PROJECT,prepared,parameter,result,projected,variant=variant,source_evidence=source,**args)
    target = stage/'projection_compare/programme_blade_review.json'
    if target.is_file():
        return target
    return merge_review_assets([p/'programme_blade_review.json' for p in (a,c)],target.parent,primary_variant=NEW)


def source_comparison(data, stage, projection, seconds):
    """Draw selected adjacent native exposures using the player's source layers."""
    from PIL import Image, ImageDraw
    from rinkspace.storage import AssetStore
    from rinkspace.viewer3d.blade_review import BladeReview, overlay_primitives
    from rinkspace.viewer3d.source import SourceVideo, body_overlay_primitives, read_body_overlay

    scope, paths, old = baseline(data)
    pts = old['source_pts']
    reader = SourceVideo(scope['source']['registered_path'], old['time_base'])
    review = BladeReview(projection, assets=AssetStore(PROJECT), time_base=tuple(old['time_base']))
    inputs = {json.loads(read(BASELINE))['candidate_id']:paths['fitted_state'], NEW:stage/'fitted/result.npz'}
    layers = {key:read_body_overlay(path,pts) for key,path in inputs.items()}
    output = stage/'native_source_review'
    output.mkdir(exist_ok=False)
    receipts = []
    try:
        for start, end in seconds:
            first, last = [round(s*12800) for s in (start,end)]
            indices = np.flatnonzero((pts>=first) & (pts<=last))
            tiles = []
            for i in indices:
                stamp = int(pts[i])
                rgb = reader.read(stamp)
                body = old['source_uv_native'][i,5:]
                body = body[np.isfinite(body).all(axis=-1)]
                lo = np.maximum(0, np.floor(body.min(axis=0))-65).astype(int)
                hi = np.minimum(rgb.shape[1::-1], np.ceil(body.max(axis=0))+65).astype(int)
                left, top = lo
                tile = Image.new('RGB',(300,2*230),(24,24,26))
                for c,(variant, layer) in enumerate(layers.items()):
                    plot = Image.fromarray(rgb[top:hi[1], left:hi[0]].copy())
                    draw = ImageDraw.Draw(plot)
                    items = overlay_primitives(review.at(variant,stamp),
                        include_model_reference=False,include_model_contours=True,
                        include_model_body=True,include_source_structures=True,include_source_legacy=False)
                    items += body_overlay_primitives(layer,i,show_model=False)
                    for item in items:
                        points = [tuple(x) for x in item['points']-lo]
                        color = tuple(int(round(255*v)) for v in item['color'][:3])
                        if item['kind']=='line':
                            draw.line(points,fill=color,width=max(1,round(item['width'])))
                        elif item['kind']=='point':
                            x,y = points[0]
                            draw.ellipse((x-3,y-3,x+3,y+3),fill=color,outline=(12,12,12))
                    plot.thumbnail((300,207))
                    tile.paste(plot,((300-plot.width)//2,230*c+23))
                    ImageDraw.Draw(tile).text((7,230*c+6),
                        f"{'A' if not c else 'C'} | {stamp/12800:.2f}s | PTS {stamp}",(241,241,242))
                tiles.append(tile)
            span = output/f'{first}-{last}'
            span.mkdir()
            for k in range(0,len(tiles),5):
                group = tiles[k:k+5]
                row = Image.new('RGB',(300*len(group),460),(24,24,26))
                for m,tile in enumerate(group):
                    row.paste(tile,(300*m,0))
                row.save(span/f'row_{k//5+1:03}.jpg',quality=91)
            receipts.append(dict(first_pts=first,last_pts=last,native_exposures=len(indices),images=str(span),
                rendered_pts=pts[indices].tolist()))
    finally:
        reader.close()
    write(output/'source_checks.json',dict(source_sha256=scope['source']['sha256'],time_base=[1,12800],
        ranges=receipts,variants=list(layers),cyan_left_orange_right='Saved active body points after per-variant mapping',
        steel_and_body='Saved candidate native projections',source_contours='Previously annotated visible structure'))


def viewer(data, stage, audit_path, projection):
    _, paths, _ = baseline(data)
    source = json.loads(read(paths['viewer_configuration']))
    result, parameter = stage/'fitted/result.npz',stage/'solve/candidate.npz'
    stats = json.loads(read(audit_path/'abc.json'))
    old=source['variants'][0]
    new=dict(old,id=NEW,candidate_id=NEW,label='C · 身体身份修订重拟合',result=str(result),result_sha256=digest(result),
        parameter=str(parameter),parameter_sha256=digest(parameter),body_reference_label=old['id'],
        body_refinement_intervals_seconds=[r['seconds'] for r in stats['controls']['exact_spline_support']],
        body_refinement_intervals_pts=[[r['first_pts'],r['last_pts']] for r in stats['controls']['exact_spline_support']],
        candidate_scope_label='全节目，同源时钟，未验收',blade_review_available=True)
    old=dict(old,label='A · 原始人体与鞋刀',blade_review_available=True)
    motion_spans = [
        dict(kind='identity_refit',start_pts=left,end_pts=right,worst_pts=anchor,
             label=f'{left/12800:.2f}–{right/12800:.2f} 秒 {label}')
        for (left,right,anchor,label) in (
            (2003200,2115840,2009600,'燕式至提鞋'),
            (2054912,2115840,2100992,'手鞋关系与腿链'),
            (2310656,2361856,2350336,'末旋提鞋与遮挡'))]
    nav=json.loads(read(paths['navigation']));nav['events'].extend(motion_spans)
    nav['events'].sort(key=lambda e:e['start_pts'])
    write(stage/'navigation.json',nav)
    source.update(variants=[new,old],comparison_variants=[NEW,old['id']],session_id=NEW,
        default_variant=NEW,latest_review_variant=NEW,initial_source_pts=2009600,source_navigation=str(stage/'navigation.json'),
        body_scope_label='身体身份对照候选，原片 PTS 同步',source_body_observations_default=True,
        blade_review_asset=str(projection),source_review_model_runner_default=True,source_review_model_body_default=True,
        source_review_model_trajectory_default=False,
        initial_visible=dict(source['initial_visible'],trajectory=False,com=False,com_projection=False,boards=False,rink_lines=False))
    write(stage/'viewer3d.json',source)
    (stage/'START_IDENTITY_3D.cmd').write_bytes(paths['viewer'].read_bytes())


def register(data, run, inputs, stage):
    base = json.loads(read(BASELINE))
    compare = stage/'comparison'
    metrics = json.loads(read(compare/'abc.json'))
    player = stage/'native_review_clean/checks.json'
    if not json.loads(read(player))['passed']:
        raise ValueError('The actual OpenGL player must load and switch both candidates')
    root = PROGRAMME/'fit_v11'
    checkpoint = stage/'solve/candidate.npz'
    controls = root/'body_identity_1114_controls.npz'
    binding = root/'body_identity_1114.json'
    pending = controls.with_suffix('.pending.npz')
    for p in (controls, pending, binding):
        if len(str(p).encode('utf-16-le'))//2 >= 260 or any(len(part.encode('utf-16-le'))//2 > 255 for part in p.parts):
            raise ValueError('OneDrive path too long: '+str(p))
        if p.exists():
            raise FileExistsError(p)
    old_controls = load(root/'body_first_spin_1113_controls.npz')
    z = load(checkpoint)
    registered_fields = {'source_pts','source_time_base','shared_body_scale','source_video_sha256','camera_sha256'}
    compact = {k:old_controls[k] if k in registered_fields else z[k]
               for k in old_controls if k != 'candidate_sha256'}
    compact['candidate_sha256'] = np.array(digest(checkpoint))
    compact['body_observation_ledger_sha256'] = np.array(digest(inputs/'observations.npz'))
    def identity(path, parent):
        return dict(path=path.relative_to(parent).as_posix(),sha256=digest(path))
    runtime = dict(solve_state=checkpoint,result=stage/'fitted/result.npz',main_log=stage/'progress.jsonl',
        run_summary=stage/'run_summary.json',anatomical_frames=stage/'anatomical_frames.npz',
        console_log=run/(stage.name+'.console.log'),first_refit_log=run/'refit/progress.jsonl',
        expanded_refit_log=run/'reviewed_refit/progress.jsonl',trust_refinement_log=run/'final_refit/progress.jsonl',
        preceding_candidate_log=run/'candidate/progress.jsonl',tail_review_run_log=run/'delivery/progress.jsonl',
        observations=inputs/'observations.npz',source_evidence=inputs/'source.json',revision=inputs/'revision.json',
        guard_population=inputs/'quality.json',all_events=compare/'candidate_events.csv',abc=compare/'abc.json',
        a_b_pose_replay=stage/'scoped_import.json',guard_results=stage/'final_metrics.json',
        inherited_motion_proposal=stage/'proposal_inputs.json',
        viewer=stage/'START_IDENTITY_3D.cmd',viewer_config=stage/'viewer3d.json',viewer_check=player,
        comparison_projection=stage/'projection_compare/programme_blade_review.json',
        previous_viewer_check=run/'candidate/native_review/checks.json',
        source_image_check=stage/'native_source_review/source_checks.json',
        projection=stage/'projection/projection_registration.json',
        tests=run/'tests_verified.log',source_scan=run/'scan/scan.npz',skate_reprojection=stage/'skate_source_replay.npz')
    if (stage/'retained_surface.json').is_file():
        runtime['surface']=stage/'retained_surface.json'
    runtime = {k:identity(p,data) for k,p in runtime.items()}
    with pending.open('xb') as f:
        np.savez_compressed(f,**compact)
    pending.rename(controls)
    write(binding,dict(schema='rinkspace.okada_body_identity_candidate.v1',programme='okada_mei_sp',candidate_id=NEW,
        baseline=identity(BASELINE,PROJECT),reviewed_source_decisions=identity(root/'body_identity_1114_review.json',PROJECT),
        source_exposures=metrics['native_exposures'],source_first_last_pts=metrics['pts'],source_time_base=metrics['time_base'],
        source_video_sha256=metrics['reference_hashes']['video'],native_camera_sha256=base['native_camera_sha256'],
        shared_scene_values_sha256=base['shared_scene_values_sha256'],body_shape_and_skate_geometry_fixed=True,
        source_clock_controls=identity(controls,PROJECT),data_root_artifacts=runtime,
        review_report=identity(root/'body_identity_1114_report.md',PROJECT),
        engineering_comparison_candidate=True,substantive_motion_improvement_verified=False,
        user_accepted=False,accuracy_accepted=False))


def main():
    parser=argparse.ArgumentParser()
    for key in ('data','prepared','run','inputs','stage'):
        parser.add_argument(key,type=lambda s:Path(s).resolve())
    parser.add_argument('command',choices=('audit','project-A','project','viewer','images','register'))
    parser.add_argument('--surface-from',type=Path)
    parser.add_argument('--span', nargs=2, type=float, action='append', metavar=('FIRST_SECOND','LAST_SECOND'))
    args=parser.parse_args()
    data,prepared,run,inputs,stage=args.data,args.prepared,args.run,args.inputs,args.stage
    if args.command=='project-A':
        _, paths, _ = baseline(data)
        generate(PROJECT,prepared,paths['solve_state'],paths['fitted_state'],run/'projection_A',
            variant=json.loads(read(BASELINE))['candidate_id'],source_evidence=paths['source_evidence'],
            review_evidence=paths['source_evidence'],allow_missing_review_source=True)
    elif args.command=='audit':
        audit(data,prepared,run,inputs,stage,stage/'comparison',surface_base=args.surface_from)
    elif args.command=='project':
        projections(data,prepared,run,inputs,stage)
    elif args.command=='viewer':
        projection=projections(data,prepared,run,inputs,stage)
        viewer(data,stage,stage/'comparison',projection)
    elif args.command=='images':
        if not args.span:
            parser.error('images requires at least one --span')
        source_comparison(data,stage,stage/'projection_compare/programme_blade_review.json',args.span)
    else:
        register(data,run,inputs,stage)


if __name__=='__main__':
    main()
