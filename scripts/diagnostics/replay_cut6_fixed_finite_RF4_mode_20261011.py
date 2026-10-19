"""One fixed finite cut6 mode, original values only; stdout, no artifacts."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import replay_cut6_F3_existing_RF4_source_response_20261011 as single
import joint_absolute_source_context_20261011 as ctx
import joint_absolute_global_tail_context_20261011 as global_tail
from verify_original_measured_locations_actual_20261011 import install_on_source_context, current_revised
from calibration.solver_runtime.source_loss import weighted_source_cost, SourceLossWeights, loss_weight_arrays
from calibration.solver_runtime.legacy.correlated_roll_prior_candidate.explicit_bias_source import theta_and_gradient

np, cv2, a = single.np, single.cv2, ctx.a


def original_value(c, index, revised, value, raw=False):
    item = c.originals[index]
    row, keep = current_revised(c, index, revised)
    at = int(item['group_actual_pts'])
    c.provider.register_row(item['source_row'], actual_pts=at,
        allow_previously_qualified_held_ad2_representation=item['collection_role'] == 'HELD')
    bound = c.provider.rebind(row, value[c.world_indices])
    if raw:
        bound = dict(bound)
        bound.pop('original_boundary_location_observation', None)
        bound.pop('authenticated_measured_location', None)
        bound = a.raw_row(bound)
    return np.asarray(c.calls[item['block']](ctx.camera(c, value, at), [bound], False)), row, keep


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('fixed','zoom-rate'),default='fixed')
    mode=parser.parse_args().mode
    a.RUN, a.CHILD, a.CONFIG = single.DIRECTORY.parent.parent, single.Q, single.CONFIG
    c = ctx.load_context()
    frozen = c.q.copy()
    shots = a.read(a.ROOT / 'calibration/program_tracking/source_shots.json')['shots'][1:10]
    shot = shots[5]
    lo, hi = int(shot['start_pts']), int(shot['end_pts'])
    points = sorted(p for p in c.state.bypts if lo <= p < hi)
    if not points or single.PTS not in points:
        raise ValueError('Existing complete cut6 exposure range required')
    blocks = {c.state.bypts[p].name for p in points}
    if blocks != {'secondary'}:
        raise ValueError('One original secondary installation must own cut6')
    columns = np.concatenate([c.state.bypts[p].start + 4*c.state.bypts[p].bundle.index[p] + np.arange(4) for p in points])
    if np.intersect1d(columns, c.fixed_evaluation).size:
        raise ValueError('Proposed fixed mode changes an original fixed RF4 coordinate')
    ref = c.state.bypts[single.PTS]
    start = ref.start + 4*ref.bundle.index[single.PTS]
    individual = c.q.copy()
    individual[start:start+4] += single.DELTA
    camera0, camera1 = [ctx.camera(c, value, single.PTS) for value in (c.q, individual)]
    R0, R1 = [cv2.Rodrigues(value[3:6])[0] for value in (camera0, camera1)]
    Q = R0.T @ R1
    angle_scan_scale=np.diag([np.exp(-single.DELTA[3]),np.exp(-single.DELTA[3]),1.])
    trial = c.q.copy()
    rotation_errors = []
    relative_angles=[]
    for p in points:
        b = c.state.bypts[p]
        j = b.start + 4*b.bundle.index[p]
        previous = ctx.camera(c, c.q, p)
        R = cv2.Rodrigues(previous[3:6])[0]
        omega=cv2.Rodrigues(R@R0.T)[0].ravel()
        relative_angles.append(float(np.linalg.norm(omega)))
        if relative_angles[-1]>=np.pi-1e-6:
            raise ValueError('Relative rotation approaches the Log branch boundary')
        expected=R@Q if mode=='fixed' else cv2.Rodrigues(angle_scan_scale@omega)[0]@R1
        trial[j:j+3] = cv2.Rodrigues(expected)[0].ravel()
        trial[j+3] += single.DELTA[3]
        # Preserve the exact supplied reference Rodrigues representation.
        # This does not change its matrix; the roundtrip alone changes ULPs.
        if p == single.PTS:
            trial[j:j+4] = individual[j:j+4]
        actual = cv2.Rodrigues(ctx.camera(c, trial, p)[3:6])[0]
        rotation_errors.append(float(np.max(abs(actual-expected))))
    if not np.array_equal(ctx.camera(c, trial, single.PTS), camera1):
        raise ValueError('Finite common mode does not reproduce exact reference camera25')
    untouched = np.ones(len(trial), bool)
    untouched[columns] = False
    if not np.array_equal(trial[untouched], c.q[untouched]):
        raise ValueError('A world/shared/fixed/outside-cut coordinate changed')
    canonical = c.state.canonical_parameter_layout.canonical_to_evaluation
    changed_canonical = c.state.canonical_parameter_layout.evaluation_to_canonical[columns]
    keep_canonical = np.ones(len(c.q), bool)
    keep_canonical[changed_canonical] = False
    if not np.array_equal(trial[canonical][keep_canonical], c.arrays['q'][keep_canonical]):
        raise ValueError('Original complete canonical untouched values changed')
    single.emit('fixed_complete_cut6_identity', mode=mode, source_interval=[lo, hi], actual_exposures=len(points),
        actual_first_last=[points[0], points[-1]], original_q_sha256=single.Q,
        reference_camera25_bitexact=True, right_rotation_matrix=Q,
        common_native_logf_change=float(single.DELTA[3]),
        camera_axis_angle_sweep_multiplier=angle_scan_scale.diagonal(),
        relative_rotation_degrees_min_max=np.rad2deg([min(relative_angles),max(relative_angles)]),
        public_rotation_matrix_max_error=max(rotation_errors),
        all_other_evaluation_and_canonical_values_bitexact=True, no_new_views_or_time_points=True,
        frozen_profile_token=c.identity['frozen_profile_token'], script_sha256=single.sha(__file__),
        zoom_changes_original_rays_not_claimed_as_gauge=True)
    with np.load(c.path, allow_pickle=False) as z:
        ordinal=z['aux__block_names'].tolist().index('secondary')
        keys=['aux__block_names','aux__block_lengths','aux__canonical_pts',
              'aux__current_depth_and_material_domain_contract_json','profile__weights_'+str(ordinal)]
        saved={key:z[key].copy() for key in keys}
    data=single.secondary_inputs(c.manifest, saved)
    selected=np.isin(data['pa'], points)|np.isin(data['pb'], points)
    data['incident']=selected
    course,cq,_=c.provider.print_course_binding(c.q[c.world_indices])
    border=np.asarray(course.sample_perimeter(cq,with_jacobian=False)[0])
    if not np.array_equal(border,np.asarray(a.read(c.directory/'scene_UNACCEPTED.json')['ice_boundary']['points_xyz'])):
        raise ValueError('Current public full course differs from saved scene')
    r0, check0, trace0=single.motion_value(c.state,c.q,data,border)
    r1, check1, trace1=single.motion_value(c.state,trial,data,border)
    for label,mask in [('all_cut_touching',selected),('inside_cut_both_endpoints',selected&np.isin(data['pa'],points)&np.isin(data['pb'],points)),
        ('cross_cut_one_endpoint',selected&~(np.isin(data['pa'],points)&np.isin(data['pb'],points))),
        ('finite_board',selected&data['board']),('finite_ice',selected&data['finite_mask']&~data['board']),
        ('profiled_unknown',selected&~data['finite_mask'])]:
        b,d=single.motion_cost(r0,data,mask),single.motion_cost(r1,data,mask)
        single.emit('complete_cut6_original_motion',kind=label,points=int(mask.sum()),before=b,after=d,change=d-b,
            original_frozen_profile_and_sigma_retained=True,
            raw_p95_before_after=[float(np.quantile(np.linalg.norm(r[mask],axis=1),.95)) if mask.any() else None for r in (r0,r1)])
    pair_changes=[]
    for pa,pb in sorted(set(zip(data['pa'][selected].tolist(),data['pb'][selected].tolist()))):
        mask=(data['pa']==pa)&(data['pb']==pb)
        b,d=single.motion_cost(r0,data,mask),single.motion_cost(r1,data,mask)
        pair_changes.append(dict(source_actual_pts=pa,target_actual_pts=pb,points=int(mask.sum()),before=b,after=d,change=d-b))
    single.emit('complete_cut6_directed_pair_summary',pairs=len(pair_changes),counts=dict(Counter('better' if r['change']<0 else 'worse' if r['change']>0 else 'equal' for r in pair_changes)),
        largest_cost_changes=sorted(pair_changes,key=lambda r:abs(r['change']),reverse=True)[:4],
        native_before=check0,native_after=check1,untouched_raw_max_change=float(abs(r1[~selected]-r0[~selected]).max()))
    packet,revised=install_on_source_context(c,c.options['--original-measured-transition-location-source'])
    originals=defaultdict(lambda:dict(before=0.,after=0.,rows=0,points=0,raw=[]))
    for index,item in enumerate(c.originals):
        if int(item['group_actual_pts']) not in points:
            continue
        b,row,keep=original_value(c,index,revised,c.q)
        if not len(keep):
            continue
        d,_,_=original_value(c,index,revised,trial)
        rb,_,_=original_value(c,index,revised,c.q,True)
        rd,_,_=original_value(c,index,revised,trial,True)
        sigma=np.repeat(np.broadcast_to(np.asarray(row['sigma'],float),(len(keep),)),2)
        raw=[(r*sigma).reshape(-1,2) for r in (rb,rd)]
        key=item['block']+':'+item['collection_role'];summary=originals[key]
        summary['rows']+=1;summary['points']+=len(keep)
        record=dict(index=index,id=row['id'],actual_pts=int(item['group_actual_pts']),role=item['collection_role'],
            mean_xy_before_after=[r.mean(0).tolist() for r in raw],
            raw_p90_before_after=[float(np.quantile(np.linalg.norm(r,axis=1),.9)) for r in raw])
        retired=(item['block']=='secondary'and int(item['group_actual_pts'])==230445824 and item['group_order']==131 and item['row_order']==2)
        record['structurally_retired']=retired
        if item['collection_role']=='FIT':
            costs=[0.,0.] if retired else [weighted_source_cost(r,ctx.original_quadratic(c,index),ctx.original_loss(c,index)) for r in (b,d)]
            summary['before']+=costs[0];summary['after']+=costs[1];record['actual_cost_before_after']=costs
        summary['raw'].append(record)
    for family,summary in originals.items():
        single.emit('complete_cut6_original_absolute',family=family,rows=summary['rows'],points=summary['points'],
            original_saved_location_rows_restored=True,actual_cost_before=summary['before'] if family.endswith(':FIT') else None,
            actual_cost_after=summary['after'] if family.endswith(':FIT') else None,
            actual_cost_change=summary['after']-summary['before'] if family.endswith(':FIT') else None,
            raw_metric_counts=dict(Counter('better' if r['raw_p90_before_after'][1]<r['raw_p90_before_after'][0] else 'worse' if r['raw_p90_before_after'][1]>r['raw_p90_before_after'][0] else 'equal' for r in summary['raw'])),
            largest_raw_changes=sorted(summary['raw'],key=lambda r:abs(r['raw_p90_before_after'][1]-r['raw_p90_before_after'][0]),reverse=True)[:4])
    for family,factor in [('ICE',c.ice),('BOARD',c.board),('MATERIAL',c.material),('CIRCLE',c.circle)]:
        active=factor._qualified_active_fit_view() if hasattr(factor,'_qualified_active_fit_view') else None
        active=factor if active is None else active
        name={'ICE':'additional_real_native_ice_curves','BOARD':'additional_real_native_board_edges',
              'MATERIAL':'additional_real_native_board_material_points','CIRCLE':'additional_real_native_board_circles'}[family]
        payload=c.integration._payloads.get(name);w,norm=loss_weight_arrays(payload,active.row_count)
        offset=0;costs=[0.,0.];stats=[]
        for row in active.rows:
            stop=offset+2*len(row['uv'])
            if row['actual_pts'] in points:
                loss=None if w is None else SourceLossWeights(w[offset:stop],None if norm is None else norm[offset:stop])
                values=[ctx.native_value(c,factor,value,[row]) for value in (c.q,trial)]
                cs=[weighted_source_cost(r,active.quadratic[offset:stop],loss) for r in values]
                costs=[x+y for x,y in zip(costs,cs)]
                raw=[(r*np.repeat(row['sigma'],2)).reshape(-1,2) for r in values]
                stats.append(dict(id=row['id'],actual_pts=row['actual_pts'],role='FIT',points=len(row['uv']),
                    cost_before_after=cs,cost_change=cs[1]-cs[0],curve=row.get('curve'),
                    source_optical_support_is_None=row.get('_source_optical_support') is None,
                    typed_raw_mean_before_after=[float(np.linalg.norm(r,axis=1).mean()) for r in raw],
                    mean_xy_before_after=[r.mean(0).tolist() for r in raw]))
            offset=stop
        for row in factor.held:
            if row['actual_pts'] not in points:
                continue
            raw=[(ctx.native_value(c,factor,value,[row],raw=True)*np.repeat(row['sigma'],2)).reshape(-1,2) for value in (c.q,trial)]
            stats.append(dict(id=row['id'],actual_pts=row['actual_pts'],role='HELD',points=len(row['uv']),curve=row.get('curve'),
                typed_raw_mean_before_after=[float(np.linalg.norm(r,axis=1).mean()) for r in raw],
                mean_xy_before_after=[r.mean(0).tolist() for r in raw]))
        single.emit('complete_cut6_native_absolute',family=family,actual_cost_before=costs[0],actual_cost_after=costs[1],
            actual_cost_change=costs[1]-costs[0],FIT_rows=sum(r['role']=='FIT' for r in stats),
            HELD_rows=sum(r['role']=='HELD' for r in stats),
            raw_counts={role:dict(Counter('better' if r['typed_raw_mean_before_after'][1]<r['typed_raw_mean_before_after'][0] else 'worse' if r['typed_raw_mean_before_after'][1]>r['typed_raw_mean_before_after'][0] else 'equal' for r in stats if r['role']==role)) for role in ('FIT','HELD')},
            reference_rows=[r for r in stats if r['actual_pts']==single.PTS],
            largest_FIT_cost_changes=sorted([r for r in stats if r['role']=='FIT'],key=lambda r:abs(r['cost_change']),reverse=True)[:3],
            largest_HELD_raw_changes=sorted([r for r in stats if r['role']=='HELD'],key=lambda r:abs(r['typed_raw_mean_before_after'][1]-r['typed_raw_mean_before_after'][0]),reverse=True)[:3])
    from audit_low6_private_RF4_causal_response_20261011 import guard_factors,guard_values,guarded_report
    factors=guard_factors(c);caps=a.read(c.directory/'truth_restoration_state.json')
    base,_=guard_values(factors,c.q);guard=guarded_report(factors,trial,caps,base)
    single.emit('complete_cut6_all86_actual_truth',**{k:v for k,v in guard.items() if k!='rows'},
        changed_groups=sum(r['value_change_px']!=0. for r in guard['rows']),
        largest_changed=[r for r in sorted(guard['rows'],key=lambda r:abs(r['value_change_px']),reverse=True)[:6] if r['value_change_px']!=0.])
    for tail in global_tail.prepare_tails(c):
        if not np.intersect1d(tail['columns'],columns).size:
            single.emit('complete_cut6_global_tail_invariance',family=tail['family'],proved_by_exact_unchanged_dependency_columns=True)
            continue
        r=tail['evaluate'](trial)
        b,d=[weighted_source_cost(v,tail['quadratic'],tail['loss']) for v in (tail['actual'],r)]
        single.emit('complete_cut6_global_tail',family=tail['family'],scalar_rows=len(r),before=b,after=d,change=d-b,
            touched_actual_source_pts=sorted(set(int(row['actual_pts']) for row in tail['row_labels'] if row.get('actual_pts') in points)),
            touched_source_label_count=sum(row.get('actual_pts') in points for row in tail['row_labels']))
    roll=json.loads(str(c.arrays['aux__correlated_roll_prior_contract_json'].item()))
    inst=roll['installation_order'].index(c.state.bypts[single.PTS].bundle.calibration.installation_id)
    bias=c.q[c.state.structured_programme_coordinate_chart.bias_indices[inst]]
    sigma=float(roll['component_sigma_radians'])
    positions=np.array([c.state.bypts[p].start+4*c.state.bypts[p].bundle.index[p] for p in points])
    priors=[(theta_and_gradient(np.array([v[j:j+3] for j in positions]))[0]-bias)/sigma for v in (c.q,trial)]
    costs=[weighted_source_cost(r,np.ones(len(r),bool),None) for r in priors]
    single.emit('complete_cut6_correlated_roll_engineering_prior',before=costs[0],after=costs[1],change=costs[1]-costs[0],
        component_sigma_radians=sigma,bias_unchanged=True,physical_pixel_observation=False)
    if not np.array_equal(c.q,frozen):
        raise ValueError('Frozen parent state mutated')
    single.emit('complete_cut6_scope_limits',candidate_state_saved=False,source_images_decoded=0,source_images_written=0,
        source_J_created=False,new_fit_or_report=False,
        remaining_track_cross_manual_and_finite_extent_checks_not_yet_run=True)


if __name__=='__main__':
    main()
