"""Compare two saved complete states on unchanged original sources; values only.

Production review values are reused. Original FIT and all motion are replayed
only where their per-source values were not archived. No J, fitting or images.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict, Counter
import argparse
import importlib.util
import json
import time

import joint_absolute_source_context_20261011 as ctx
import audit_low6_private_RF4_causal_response_20261011 as local
from audit_production_motion_coverage import certify, compact_indices, token
from calibration.solver_runtime.source_loss import weighted_source_cost
from verify_original_measured_locations_actual_20261011 import install_on_source_context,current_revised

a, np = ctx.a, ctx.np
SIX = local.SIX


def emit(**value):
    print(json.dumps(value, ensure_ascii=False, default=a.plain), flush=True)


def shot_for(pts, shots):
    matches=[i+1 for i,s in enumerate(shots) if int(s['start_pts'])<=int(pts)<int(s['end_pts'])]
    return matches[0] if len(matches)==1 else None


def row_key(row):
    return (row.get('id'),row.get('split'),row.get('actual_pts',row.get('pts')),
            row.get('source_pts',row.get('source_native_pts')),row.get('target_pts',row.get('target_native_pts')))


def review_metric(row):
    raw=row.get('raw_source_pixel_error',{})
    if raw.get('p95_px') is not None:return 'raw_p95_native_px',float(raw['p95_px'])
    for name in ('p90_native_px','mean_native_px','mean_raw_native_distance_px','median_native_px','native_error_norm_px'):
        if row.get(name) is not None:return name,float(row[name])
    if 'mean_outside_source_support_px' in row:return 'mean_outside_source_support_px',float(row['mean_outside_source_support_px'])
    return None,None


def original_raw_review_changes(before, after, shots):
    """Compare archived raw display errors separately from support exceedance.

    A zero outside-support value is not zero pixel error. Keep the exact
    display operator and original split, and omit any unavailable raw metric.
    """
    key=lambda r:(r['block'],r['group_order'],r['row_order'],r['actual_pts'],r['id'])
    first={key(r):r for r in before['rows']};second={key(r):r for r in after['rows']}
    if len(first)!=225 or first.keys()!=second.keys():
        raise ValueError('Original raw review source instances changed')
    result=[]
    for k,old in first.items():
        new=second[k]
        for field in ('observation_signature','original_split_present','original_source_split','UV_count'):
            if old[field]!=new[field]:raise ValueError('Original raw review source changed: '+str(k))
        left=old.get('original_pixel_error_report',{});right=new.get('original_pixel_error_report',{})
        if left.get('observation_kind')!=right.get('observation_kind'):
            raise ValueError('Original raw display operator changed: '+str(k))
        if 'median_px' not in left or 'median_px' not in right:continue
        metrics={name:dict(before=float(left[name]),after=float(right[name]),delta=float(right[name]-left[name]))
                 for name in ('median_px','p95_px','maximum_px') if name in left and name in right}
        result.append(dict(block=old['block'],id=old['id'],actual_pts=old['actual_pts'],
            shot=shot_for(old['actual_pts'],shots),original_source_split=old['original_source_split'],
            original_split_present=old['original_split_present'],
            display_operator=left.get('observation_kind','original_source_pixel_display'),
            metrics=metrics))
    return result


def axis_location_review_change(before, after):
    """Keep conditional source-axis errors distinct from nearest-curve display."""
    old=before.get('axis_scanline_location_objective')
    new=after.get('axis_scanline_location_objective')
    if old is None and new is None:return None
    if old is None or new is None:
        return dict(id=before['id'],before_available=old is not None,after_available=new is not None,
            comparable=False,legacy_nearest_distance_is_not_an_axis_error=True)
    for key in ('fixed_axis','branch'):
        if old[key]!=new[key]:raise ValueError('Conditional source-axis operator changed')
    return dict(id=before['id'],actual_pts=before.get('actual_pts',before.get('pts')),
        split=before.get('split'),fixed_axis=old['fixed_axis'],branch=old['branch'],comparable=True,
        metrics={name:dict(before=old[name],after=new[name],delta=new[name]-old[name])
                 for name in ('mean_horizontal_error_px','p90_horizontal_error_px',
                              'squared_normalized_horizontal_residual')},
        signed_horizontal_error_before=old['signed_horizontal_error_px'],
        signed_horizontal_error_after=new['signed_horizontal_error_px'])


def saved_reviews(before,after,shots):
    for directory in (before,after):
        identity=a.read(directory/'state_identity.json')
        if a.sha(directory/'source_review_UNACCEPTED.json')!=identity['source_review_sha256']:
            raise ValueError('Saved production review identity changed')
    data=[a.read(d/'source_review_UNACCEPTED.json') for d in (before,after)]
    rows=[];families=[];axis_rows=[]
    for family,old in data[0].items():
        if not family.startswith('additional_') or not isinstance(old,dict) or 'rows' not in old:continue
        new=data[1][family];first={row_key(r):r for r in old['rows']};second={row_key(r):r for r in new['rows']}
        if first.keys()!=second.keys() or len(first)!=len(old['rows']):raise ValueError('Production review source identity changed: '+family)
        summary=Counter()
        for key,b in first.items():
            d=second[key];metric,x=review_metric(b);metric2,y=review_metric(d)
            axis=axis_location_review_change(b,d)
            if axis is not None:axis_rows.append(dict(axis,family=family))
            if metric!=metric2:raise ValueError('Production review metric changed')
            pts=b.get('actual_pts',b.get('target_actual_pts',b.get('target_pts',b.get('pts'))))
            if metric is None:
                summary[str(b.get('split'))+':no_active_numeric_metric']+=1
                continue
            split=b.get('split');status='improved' if y<x else 'worsened' if y>x else 'unchanged'
            summary[str(split)+':'+status]+=1
            rows.append(dict(family=family,id=b['id'],split=split,actual_pts=pts,shot=shot_for(pts,shots) if pts is not None else None,
                physical_entity=b.get('feature_id', 'Ad'+str(b['advertisement_number']) if 'advertisement_number' in b else b['id']),
                metric=metric,before=x,after=y,delta=y-x,status=status,
                active_points=b.get('active_FIT_accuracy_uv_count',b.get('source_uv_count',b.get('points')))))
        metric_names=sorted({r['metric']for r in rows if r['family']==family})
        families.append(dict(family=family,source_rows=len(first),same_source_instances=True,
            metrics=metric_names,counts=dict(summary)))
    low=[r for r in rows if r['family']=='additional_native_board_edges' and r['actual_pts'] in SIX and r['split']=='fit']
    grouped=[]
    for family in sorted({r['family'] for r in rows}):
        for shot in range(1,10):
            for split in sorted({str(r['split'])for r in rows if r['family']==family}):
                rr=[r for r in rows if r['family']==family and r['shot']==shot and str(r['split'])==split]
                if rr:grouped.append(dict(family=family,shot=shot,split=split,rows=len(rr),
                    metrics=sorted({r['metric']for r in rr}),counts=dict(Counter(r['status'] for r in rr)),
                    maximum_absolute_same_metric_change=max(abs(r['delta'])for r in rr)))
    held=[];raw_held=[]
    paths=[d/'original225_review_UNACCEPTED.json' for d in (before,after)]
    if all(p.exists() for p in paths):
        from structured_programme_source_candidate.original_review_source import compare_original_review_reports
        original_reports=[a.read(p) for p in paths]
        held=compare_original_review_reports(*original_reports)
        raw_held=original_raw_review_changes(*original_reports,shots)
    receipts=[a.read(d.parent.parent/'actual_trial_family_cost_pass_0001.json') for d in (before,after)]
    for directory,receipt in zip((before,after),receipts):
        identity=a.read(directory/'state_identity.json')
        if receipt.get('complete')is not True or receipt.get('error')is not None or receipt['complete_q_sha256']!=identity['q_sha256']:
            raise ValueError('Saved complete family values belong to another state or failed pass')
        if receipt['frozen_profile_token']!=identity['frozen_profile_token']:raise ValueError('Saved family values use another frozen profile')
    logs=[receipt['families']for receipt in receipts]
    if len(logs[0])!=len(logs[1]):raise ValueError('Original family count changed')
    costs=[]
    for ordinal,(old,new) in enumerate(zip(*logs)):
        identity=old['original_row_ledger'];identity2=new['original_row_ledger']
        for field in ('family','original_start','original_stop'):
            if identity[field]!=identity2[field]:raise ValueError('Original target family layout changed')
        x,y=float(old['normalized_cost']),float(new['normalized_cost'])
        costs.append(dict(ordinal=ordinal,family=identity['family'],scalar_rows=identity['original_stop']-identity['original_start'],before=x,after=y,delta=y-x))
    result=dict(reused_production_reviews=True,review_references=[a.ref(d/'source_review_UNACCEPTED.json') for d in (before,after)],
        native_rows=rows,native_family_counts=families,native_per_shot=grouped,LOW6=low,
        original225_HELD=held,original225_raw_pixel_rows=raw_held,
        native_axis_location_rows=axis_rows,family_costs=costs,
        raw_metric_row_counts_are_not_full_shot_accuracy_certificates=True)
    print_review_summary(result)
    return result


def print_review_summary(result):
    """Small stdout only; no common pixel average or automatic report file."""
    costs=result['family_costs'];rows=result['native_rows']
    emit(stage='complete_saved_production_review',total_cost_before=sum(r['before']for r in costs),
        total_cost_after=sum(r['after']for r in costs),family_cost_delta={str(r['ordinal'])+':'+r['family']:r['delta']for r in costs})
    emit(stage='LOW6_same_raw_p95_native_px',rows=[{k:r[k]for k in ('actual_pts','before','after','delta','status')}for r in result['LOW6']])
    emit(stage='native_family_same_metric_counts',families=result['native_family_counts'])
    abbreviations={
        'additional_native_ice_curves':'ICE','additional_native_board_edges':'BOARD',
        'additional_native_board_material_points':'MATERIAL','additional_native_static_board_tracks':'TRACK',
        'additional_native_board_circles':'CIRCLE','additional_native_persistent_ice_long_baseline':'PERSISTENT'}
    for shot in range(1,10):
        rr=[r for r in result['native_per_shot']if r['shot']==shot]
        emit(stage='shot_source_review',shot=shot,families={abbreviations.get(r['family'],r['family'])+':'+r['split']:
            dict(counts=r['counts'],maximum_absolute_same_metric_change=r['maximum_absolute_same_metric_change'])for r in rr})
    held=result['original225_HELD'];counts=defaultdict(Counter)
    if held:
        for r in held['rows']:
            split=str(r['original_source_split'])if r['original_split_present']else 'absent_original_split'
            if 'delta_median_px'not in r:status='not_evaluated'
            else:status='improved'if r['delta_median_px']<0 else 'worsened'if r['delta_median_px']>0 else 'unchanged'
            counts[split][status]+=1
        numeric=[r for r in held['rows']if 'delta_median_px'in r]
        largest=sorted(numeric,key=lambda r:abs(r['delta_median_px']),reverse=True)[:3]
        emit(stage='original225_HELD_same_supported_median_px',counts={k:dict(v)for k,v in counts.items()},
            largest_changes=[{k:r[k]for k in ('id','actual_pts','original_source_split','before_median_px','after_median_px','delta_median_px')}for r in largest],
            pending_and_absent_splits_not_promoted_to_independent_validation=True)
    else:emit(stage='original225_HELD_unavailable',missing_original_review=True)
    held_native=[r for r in rows if r['split']!='fit']
    emit(stage='native_HELD_largest_same_metric_changes',rows=[{k:r[k]for k in ('family','id','split','metric','before','after','delta')}
        for r in sorted(held_native,key=lambda r:abs(r['delta']),reverse=True)[:3]],
        no_cross_metric_pixel_average=True,source_images_decoded=0,source_images_written=0)


def state_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        keys=['q','aux__block_names','aux__block_lengths','aux__block_start','aux__canonical_pts',
              'aux__current_evaluation_to_canonical_parameter_indices','aux__current_depth_and_material_domain_contract_json','quadratic']
        keys += ['profile__weights_'+str(i) for i in range(len(z['aux__block_names']))]
        return {k:z[k].copy() for k in keys}


def prepare_states(c,after):
    identity=a.read(after/'state_identity.json');run=after.parent.parent;manifest=a.read(run/'input_manifest.json')
    for path,key in ((after/'complete_state_UNACCEPTED.npz','state_npz_sha256'),(after/'scene_UNACCEPTED.json','scene_sha256'),
            (Path(identity['camera_path']),'camera_sha256'),(after/'source_review_UNACCEPTED.json','source_review_sha256'),(run/'input_manifest.json','input_manifest_sha256')):
        if a.sha(path)!=identity[key]:raise ValueError('Saved candidate reference changed: '+str(path))
    first=state_arrays(c.path);second=state_arrays(after/'complete_state_UNACCEPTED.npz')
    for key in first:
        if key=='q':continue
        if not np.array_equal(first[key],second[key]):raise ValueError('Original source/chart/frozen profile changed: '+key)
    if c.identity['frozen_profile_token']!=identity['frozen_profile_token']:raise ValueError('Frozen profile identity changed')
    for key in ('original_motion_far_board_role_correction','original_source_role_retirement','static_motion_source_role_retirement',
                'original_motion_edge_counts','original_motion_rows','actual_exposures','native_presentations'):
        if c.manifest[key]!=manifest[key]:raise ValueError('Original source contract changed: '+key)
    for key in ('schema','low_candidate','main_candidate','removed_motion_points','removed_motion_scalar_rows','original_retirement_scalar_remap'):
        if c.manifest['original_motion_input_compaction'][key]!=manifest['original_motion_input_compaction'][key]:raise ValueError('Compaction source contract changed: '+key)
    value=c.state.evaluation_q(second['q']);c.provider.validate_parameters(value[c.world_indices])
    borders=[];camera_checks=[]
    for directory,ident,q in ((c.directory,c.identity,c.q),(after,identity,value)):
        course,world,_=c.provider.print_course_binding(q[c.world_indices]);border=np.asarray(course.sample_perimeter(world,with_jacobian=False)[0])
        np.testing.assert_array_equal(border,np.asarray(a.read(directory/'scene_UNACCEPTED.json')['ice_boundary']['points_xyz']))
        borders.append(border)
        with np.load(ident['camera_path'],allow_pickle=False) as z:pts=z['pts'];export=z['camera25']
        real=np.asarray(sorted(c.state.bypts),int);indices=np.searchsorted(pts,real)
        if not np.array_equal(pts[indices],real):raise ValueError('Candidate camera source clock changed')
        actual=np.asarray([ctx.camera(c,q,int(p)) for p in real]);error=float(abs(actual-export[indices]).max())
        if error>1e-9:raise ValueError('Public camera25 and saved complete export differ: '+str(error))
        camera_checks.append(dict(actual_exposures=len(real),public_camera25_export_max_error=error,
            current_perimeter_saved_scene_bitexact=True))
    return value,first,second,identity,manifest,borders,camera_checks


def verify_common_initial_state(before,after,common_cap_state=None):
    initial=[]
    for directory in (before,after):
        path=directory.parent.parent/'actual_initial_normal_state.npz'
        with np.load(path,allow_pickle=False)as z:
            initial.append(dict(q=z['q'].copy(),profile=str(z['frozen_profile_token'].item())))
    np.testing.assert_array_equal(initial[0]['q'],initial[1]['q'])
    if initial[0]['profile']!=initial[1]['profile']:raise ValueError('The comparison cycles did not use the same initial frozen profile')
    q_sha=token(initial[0]['q'])
    if common_cap_state is not None:
        caps=a.read(Path(common_cap_state)/'truth_restoration_state.json')
        if caps['complete_q_sha256']!=q_sha:raise ValueError('Common cap parent differs from the actual initial state')
    emit(stage='same_initial_complete_state_and_profile',complete_q_sha256=q_sha,frozen_profile_token=initial[0]['profile'],
         original_two_complete_cycles_not_sequentially_accumulated=True)
    return dict(complete_q_sha256=q_sha,frozen_profile_token=initial[0]['profile'])


def original_absolute(c,value,shots):
    rows=[];receipts=defaultdict(lambda:dict(before=0.,after=0.,operator_rows=0,scalar_rows=0))
    retired=c.manifest['original_source_role_retirement'];locator=retired.get('locator',retired.get('original_locator',{}))
    # The sole retired Ad7/right_top is the historical wrong material row.
    # The distinct corrected Ad7/right_top row remains active.
    for index,item in enumerate(c.originals):
        if item['collection_role']!='FIT':continue
        revised=getattr(c,'_comparison_revised_original_locations',{})
        row,keep=current_revised(c,index,revised)
        if not len(keep):continue
        def actual_at(q):
            at=int(item['group_actual_pts']);c.provider.register_row(item['source_row'],actual_pts=at)
            return np.asarray(c.calls[item['block']](ctx.camera(c,q,at),[c.provider.rebind(row,q[c.world_indices])],False))
        before=actual_at(c.q);after=actual_at(value)
        if before.shape!=after.shape or not np.isfinite(after).all():raise ValueError('Original absolute value shape/domain changed')
        inactive=(item['block']=='secondary' and int(item['group_actual_pts'])==230445824 and item['group_order']==131 and item['row_order']==2)
        loss=ctx.original_loss(c,index);quad=ctx.original_quadratic(c,index);scales=np.repeat(np.asarray(row['sigma']),2)
        if quad.shape!=before.shape:raise ValueError('Original absolute scalar layout changed')
        x,y=(0.,0.) if inactive else (weighted_source_cost(before,quad,loss),weighted_source_cost(after,quad,loss))
        b=c.provider._binding(item['source_row'])[1];physical=b.get('name')or '+'.join('Ad'+str(n) for n in sorted({n for n,_ in b.get('ids',[])}))or b['kind']
        record=dict(packet_row_index=index,block=item['block'],id=row['id'],actual_pts=int(item['group_actual_pts']),
            physical_entity=physical,shot=shot_for(item['group_actual_pts'],shots),points=len(keep),scalar_rows=len(before),
            original_cost_before=x,original_cost_after=y,cost_delta=y-x,structurally_retired=inactive,
            supported_native_component_p95_before=float(np.quantile(abs(before*scales),.95)),
            supported_native_component_p95_after=float(np.quantile(abs(after*scales),.95)))
        rows.append(record);r=receipts[item['block']];r['before']+=x;r['after']+=y;r['operator_rows']+=1;r['scalar_rows']+=len(before)
    if sum(r['structurally_retired'] for r in rows)!=1:raise ValueError('Original material retirement row not uniquely bound')
    grouped=[]
    for shot in range(1,10):
        rr=[r for r in rows if r['shot']==shot and not r['structurally_retired']]
        for entity in sorted({r['physical_entity'] for r in rr}):
            gg=[r for r in rr if r['physical_entity']==entity]
            grouped.append(dict(shot=shot,physical_entity=entity,operators=len(gg),
                before=sum(r['original_cost_before']for r in gg),after=sum(r['original_cost_after']for r in gg),
                improved=sum(r['cost_delta']<0 for r in gg),worsened=sum(r['cost_delta']>0 for r in gg)))
    for block in ('main1','main7'):
        saved=a.read(a.RUN/f'actual_trial_{block}_source_row_cost_pass_0001.json')
        gold=next(r['actual_after_cost']for r in saved['actual_source_partition']if r['family']=='absolute_original_FIT')
        receipts[block]['saved_q8_absolute_cost']=gold;receipts[block]['baseline_cost_error']=receipts[block]['before']-gold
        if abs(receipts[block]['before']-gold)>2e-8:raise ValueError('Original absolute baseline did not reproduce saved partition')
    emit(stage='all_original_absolute_FIT_values',operators=len(rows),blocks=dict(receipts))
    for shot in range(1,10):
        rr=[r for r in grouped if r['shot']==shot]
        emit(stage='whole_shot_original_absolute_FIT',shot=shot,operators=sum(r['operators']for r in rr),
            before=sum(r['before']for r in rr),after=sum(r['after']for r in rr),
            improved=sum(r['improved']for r in rr),worsened=sum(r['worsened']for r in rr),
            largest_entity_changes=sorted(rr,key=lambda r:abs(r['after']-r['before']),reverse=True)[:2])
    return dict(rows=rows,by_block=dict(receipts),by_shot_entity=grouped,
        active_compaction_current_UV_sigma_support_and_original_loss_preserved=True)


def reconstruct_all_motion(c,arrays):
    coverage_path=a.H/'current_motion_role_whole_programme_exact_947_20261011.json';coverage=a.read(coverage_path)
    authority=coverage['actual_runtime_manifest']
    if a.sha(authority['path'])!=authority['sha256']:raise ValueError('Prior exact motion source authentication changed')
    manifest=dict(c.manifest);previous=a.read(authority['path'])
    manifest['persistent_source_replay_identity']=previous['persistent_source_replay_identity']
    reference=coverage['reconstruction_module']
    if a.sha(reference['path'])!=reference['sha256']:raise ValueError('Motion replay source changed')
    spec=importlib.util.spec_from_file_location('rs_complete_source_numeric_replay',reference['path']);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.STATE=c.directory;old.RUN=a.RUN
    binding=Path(reference['path']).parent/'actual_pan_pairs_canonical_RF4_export_binding_27fa_699_readonly_20261011.json'
    bindings=next(r for r in a.read(binding)['models']if r['model']=='699')['pairs']
    compaction=a.read(a.RUN/'actual_original_motion_input_compaction.json');blocks,checks,refs=old.reconstruct(manifest,compaction,bindings)
    main=a.read(compaction['main_candidate']['path']);base_planes={}
    for record in main['classification']:
        with np.load(record['map']['path'],allow_pickle=False) as mp,np.load(record['original_admission_cache']['path'],allow_pickle=False) as z:
            base_planes[record['block']]=z['active_finite_planes'][mp['keep_old_point_indices']].copy()
    base_planes['low']=np.zeros((len(blocks['low']['a']),4));base_planes['low'][blocks['low']['finite_mask']]=[0.,0.,1.,0.]
    base_planes['secondary']=np.zeros((len(blocks['secondary']['a']),4));base_planes['secondary'][blocks['secondary']['finite_mask']]=[0.,0.,1.,0.]
    typed=a.R/'prepared_full8503_1791035775270492900/complete_original_sources/low_source/low_source_factory/finite_material_source_binding.json'
    lowboard=[r['global_bundle_index']for r in a.read(typed)if r['surface_kind']=='finite_board_face']
    expected=manifest['persistent_source_replay_identity']['actual_original_source_input_arrays'];out={};cursor=0
    contract=json.loads(str(arrays['aux__current_depth_and_material_domain_contract_json'].item()))
    for ordinal,(name,length) in enumerate(zip(arrays['aux__block_names'].tolist(),arrays['aux__block_lengths']//4)):
        src=blocks[name];stamps=arrays['aux__canonical_pts'][cursor:cursor+length];cursor+=int(length)
        certify(stamps,expected['programme:'+name+':stamps'],name+':stamps')
        board=np.linalg.norm(base_planes[name][:,:2],axis=1)>0
        if name=='low':
            pos,removed=compact_indices(lowboard,src['active_to_original'])
            if len(removed):raise ValueError('Original LOW finite board unexpectedly retired')
            board[pos]=True
        pos,removed=compact_indices(c.manifest['original_motion_far_board_role_correction']['original_global_bundle_indices_by_block'][name],src['active_to_original'])
        board[pos]=True;certify(np.flatnonzero(board),expected['programme:'+name+':course_board_indices'],name+':board')
        static=np.flatnonzero(src['finite_mask']&~board)
        certify(static,expected['programme:'+name+':static_noncourse_finite_point_indices'],name+':static')
        floor=next(r['candidate_minimum_source_Z_m']for r in contract['blocks']if r['block']==name)/contract['scene_unit_metres']
        profile=arrays['profile__weights_'+str(ordinal)]
        if profile.shape!=(len(board),2):raise ValueError('Saved profile shape changed')
        out[name]=dict(source=src,stamps=stamps,board=board,base_planes=base_planes[name],profile=profile,floor=floor,
            pa=stamps[src['ia']],pb=stamps[src['ib']],old=old)
    return out,dict(coverage=a.ref(coverage_path),prior_array_authentication=authority,source_hash_checks=checks,
        source_authorities=refs,new_actor_input_array_digest_capture_claimed=False)


def motion_values(c,motion,value,border):
    from structured_programme_source_candidate.original_motion_far_board_roles import OriginalMotionFarBoardRoleBinding
    from shared_board_course_candidate.board_course_material import vertical_plane_value_jacobian
    import cv2
    src=motion['source'];camera=np.asarray([ctx.camera(c,value,int(p))for p in motion['stamps']])
    rotations=np.asarray([cv2.Rodrigues(v[3:6])[0]for v in camera]);planes=motion['base_planes'].copy()
    if not np.array_equal(camera[:,8:10],np.broadcast_to(camera[0,8:10],camera[:,8:10].shape)) or not np.all(camera[:,11]==camera[0,11]):raise ValueError('Block intrinsics are not shared')
    edge=border[:-1]if np.array_equal(border[0],border[-1])else border;zero=np.zeros((3,0))
    for view in np.unique(src['ia'][motion['board']]):
        ids=np.flatnonzero(motion['board']&(src['ia']==view));course=SimpleNamespace(sample_perimeter=lambda q,with_jacobian=False:(border,))
        selector=SimpleNamespace(_facet_course_cache=None,records=[None]*len(ids),source_row_groups={0:np.arange(len(ids))},
            source_cameras={0:camera[view]},source_uv=src['a'][ids],exact_cache_statistics=dict(facet_group_hits=0,facet_group_misses=0))
        facets=OriginalMotionFarBoardRoleBinding._current_source_facets(selector,course,None,('public_current_border',token(border)))
        planes[ids]=np.asarray([vertical_plane_value_jacobian(edge[i],edge[(i+1)%len(edge)],zero,zero)[0]for i in facets])
    sigma=src['motion_sigma'].reshape(-1,2);result=np.empty_like(src['a']);checks=[]
    # Retain original point order inside each source batch; no profiling weight
    # is applied again outside the original nuisance kernel.
    for begin in range(0,len(result),65536):
        ids=np.arange(begin,min(begin+65536,len(result)));ca,cb=camera[src['ia'][ids]],camera[src['ib'][ids]]
        r,check=motion['old'].native_motion(a=src['a'][ids],b=src['b'][ids],fa=np.exp(ca[:,6]),fb=np.exp(cb[:,6]),
            ra=rotations[src['ia'][ids]],rb=rotations[src['ib'][ids]],ca=ca[:,:3],cb=cb[:,:3],da=None,db=None,
            dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,pp=camera[0,8:10],k1=camera[0,11],
            planes=planes[ids],sigma=sigma[ids],profile_weight=motion['profile'][ids],jacobian=False,
            minimum_source_depth_scene_units=motion['floor'])
        result[ids]=np.asarray(r).reshape(-1,2);checks.append(check)
    if not np.isfinite(result).all():raise ValueError('Original native motion produced nonfinite values')
    return result,dict(native_batches=len(checks),native_check_records=checks,original_production_chunk_points=65536,
        actual_principal_point_px=camera[0,8:10].tolist(),actual_k1=float(camera[0,11]),
        public_current_source_ray_facets_and_border=True)


def compare_motion(c,motions,value,borders,shots):
    rows=[];pairs=[];blocks=[];total=0
    for name,m in motions.items():
        emit(stage='complete_motion_value_replay_started',block=name,points=len(m['source']['a']))
        raw0,check0=motion_values(c,m,c.q,borders[0]);raw1,check1=motion_values(c,m,value,borders[1])
        src=m['source'];sigma=src['motion_sigma'].reshape(-1,2);normalized=[raw0/sigma,raw1/sigma];active=np.ones_like(raw0,bool)
        if name=='low':
            retirement=c.manifest['static_motion_source_role_retirement'];slots=np.asarray(retirement['global_scalar_slots'],int)-int(retirement['programme_block']['original_start'])
            active.ravel()[slots]=False
        for r in normalized:r[~active]=0.
        cost=[np.where(abs(r)<=2.,.5*r*r,2.*abs(r)-2.).sum(axis=1)for r in normalized]
        blocks.append(dict(block=name,points=len(raw0),before=float(cost[0].sum()),after=float(cost[1].sum()),delta=float((cost[1]-cost[0]).sum()),
            structural_inactive_scalars=int((~active).sum()),native_checks_before=check0,native_checks_after=check1))
        assigned=np.zeros(len(raw0),bool)
        for i,s in enumerate(shots,1):
            lo,hi=int(s['start_pts']),int(s['end_pts']);mask=(m['pa']>=lo)&(m['pa']<hi)&(m['pb']>=lo)&(m['pb']<hi)
            if not mask.any():continue
            assigned|=mask
            for branch,bmask in (('all_motion',mask),('finite_board',mask&m['board']),('finite_other',mask&src['finite_mask']&~m['board']),('unknown_depth',mask&~src['finite_mask'])):
                ids=np.flatnonzero(bmask)
                if not len(ids):continue
                xy=[r[ids] for r in (raw0,raw1)];before,after=[float(v[ids].sum())for v in cost]
                rows.append(dict(shot=i,block=name,branch=branch,points=len(ids),before=before,after=after,delta=after-before,
                    native_norm_p95_before=float(np.quantile(np.linalg.norm(xy[0],axis=1),.95)),native_norm_p95_after=float(np.quantile(np.linalg.norm(xy[1],axis=1),.95))))
            for pa,pb in sorted(set(zip(m['pa'][mask].tolist(),m['pb'][mask].tolist()))):
                ids=np.flatnonzero(mask&(m['pa']==pa)&(m['pb']==pb));x,y=[float(v[ids].sum())for v in cost]
                pairs.append(dict(shot=i,block=name,source_actual_pts=pa,target_actual_pts=pb,points=len(ids),before=x,after=y,delta=y-x))
        if not assigned.all():raise ValueError('Original cross-shot/unassigned motion rows exist')
        total+=len(raw0);emit(stage='complete_motion_block_done',block=name,points=len(raw0),before=blocks[-1]['before'],after=blocks[-1]['after'],delta=blocks[-1]['delta'])
    coverage=a.read(a.H/'current_motion_role_whole_programme_exact_947_20261011.json')
    if total!=coverage['totals']['motion_points']:raise ValueError('Complete original motion count changed')
    for shot in coverage['shots']:
        observed=sum(r['points']for r in rows if r['shot']==shot['shot_index']and r['branch']=='all_motion')
        if observed!=shot['motion_points']:raise ValueError('Original full-shot motion coverage changed')
    emit(stage='all9_shot_original_motion_complete',points=total,shots=len(shots))
    for shot in range(1,10):
        emit(stage='whole_shot_motion_material_response',shot=shot,
            materials=[{k:r[k]for k in ('branch','points','before','after','delta','native_norm_p95_before','native_norm_p95_after')}for r in rows if r['shot']==shot])
    return dict(total_points=total,by_block=blocks,by_shot_material=rows,actual_directed_pairs=pairs,
        source_profile_sigma_role_retirement_and_Huber2_preserved=True,full_production_finite_extent_guard_reconstructed=False)


def camera_parameter_ranges(directories,shots,scene_unit_m):
    """Authenticated full actual camera exports; no probe/factor/J needed."""
    states=[]
    for directory in directories:
        identity=a.read(directory/'state_identity.json');camera_path=Path(identity['camera_path'])
        if a.sha(camera_path)!=identity['camera_sha256']:raise ValueError('Camera export SHA changed')
        if a.sha(directory/'complete_state_UNACCEPTED.npz')!=identity['state_npz_sha256']:raise ValueError('Complete state SHA changed')
        with np.load(directory/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:actual=z['aux__canonical_pts'].copy()
        with np.load(camera_path,allow_pickle=False)as z:
            archive={k:z[k].copy()for k in ('pts','exposure_representative_pts','camera25','installation_id','metadata_json')}
        aliases=archive['exposure_representative_pts'];indices={}
        for i,p in enumerate(aliases):
            p=int(p)
            if p in indices:
                if not np.array_equal(archive['camera25'][indices[p]],archive['camera25'][i]):raise ValueError('Alias has an independent camera')
            else:indices[p]=i
        if any(int(p)not in indices for p in actual):raise ValueError('An actual source view is absent')
        select=np.asarray([indices[int(p)]for p in actual],int);camera=archive['camera25'][select]
        install=archive['installation_id'][select];meta=json.loads(str(archive['metadata_json'].item()))
        calibrations=[]
        for name,cal in meta['calibrations'].items():
            pivot=np.asarray(cal['pivot_xyz'])*scene_unit_m;lever=np.asarray(cal['lever_xyz'])*scene_unit_m
            ids=np.flatnonzero(install==name)
            calibrations.append(dict(installation_id=name,pivot_xyz_working_m=pivot.tolist(),pivot_Z_working_m=float(pivot[2]),
                lever_xyz_working_m=lever.tolist(),lever_length_working_m=float(np.linalg.norm(lever)),
                optical_C_xyz_working_m_min_p50_max=np.quantile(camera[ids,:3]*scene_unit_m,[0,.5,1],axis=0).tolist(),
                focal_native_px_min_max=np.exp([camera[ids,6].min(),camera[ids,6].max()]).tolist(),
                principal_x_fraction=cal['principal_x_fraction_coefficients'],principal_y_fraction=cal['principal_y_fraction_coefficients'],k1=cal['k1_coefficients']))
        shot_rows=[]
        for j,shot in enumerate(shots,1):
            ids=np.flatnonzero((actual>=int(shot['start_pts']))&(actual<int(shot['end_pts'])))
            if not len(ids):raise ValueError('An original whole shot has no actual source views')
            shot_rows.append(dict(shot=j,actual_views=len(ids),source_first_last=[int(actual[ids].min()),int(actual[ids].max())],
                installations=sorted(set(install[ids].tolist())),
                optical_C_xyz_working_m_min_p50_max=np.quantile(camera[ids,:3]*scene_unit_m,[0,.5,1],axis=0).tolist(),
                focal_native_px_min_max=np.exp([camera[ids,6].min(),camera[ids,6].max()]).tolist()))
        states.append(dict(state_q_sha256=identity['q_sha256'],actual_source_views=len(actual),installations=calibrations,shots=shot_rows))
    if states[0]['actual_source_views']!=states[1]['actual_source_views']:raise ValueError('Whole source view count changed')
    for label,state in zip(('before','after'),states):emit(stage='complete_installation_camera_ranges',state=label,**state)
    return states


def common_parent_guard_comparison(c,value,cap_state):
    factors=local.guard_factors(c)
    if cap_state is None:cap_state=c.directory
    cap_state=Path(cap_state);caps=a.read(cap_state/'truth_restoration_state.json')
    reference=caps['complete_state_reference']
    if a.sha(reference['path'])!=reference['sha256']:raise ValueError('Original common-parent cap state SHA changed')
    with np.load(reference['path'],allow_pickle=False)as z:
        parent_q=z['q'].copy()
        if token(parent_q)!=caps['complete_q_sha256']:raise ValueError('Common-parent cap q changed')
    parent=c.state.evaluation_q(parent_q);base,ids=local.guard_values(factors,parent)
    before=local.guarded_report(factors,c.q,caps,base);after=local.guarded_report(factors,value,caps,base)
    differences=[]
    for old,new in zip(before['rows'],after['rows']):
        if (old['family'],old['id'],old['original_q8_limit_px'])!=(new['family'],new['id'],new['original_q8_limit_px']):raise ValueError('Original common-parent guard changed')
        differences.append(dict(family=old['family'],id=old['id'],limit_px=old['original_q8_limit_px'],before_px=old['trial_violation_px'],after_px=new['trial_violation_px'],
            delta_px=new['trial_violation_px']-old['trial_violation_px']))
    emit(stage='all86_same_common_parent_caps',parent_q=caps['complete_q_sha256'],
         before={k:v for k,v in before.items()if k!='rows'},after={k:v for k,v in after.items()if k!='rows'},
         largest_changes=sorted(differences,key=lambda r:abs(r['delta_px']),reverse=True)[:6])
    return dict(before=before,after=after,differences=differences,common_parent_reference=a.ref(cap_state/'truth_restoration_state.json'))


def native_ice_cost_breakdown(before,after,options):
    """One thin native-factor value replay, preserving full budget slices."""
    from rinkspace.vision_camera_probe import FrozenCameraSourceProbe
    from calibration.solver_runtime.curve_source_grouping import SourceWithNativeBaseCurveIntegration
    from calibration.solver_runtime.source_loss import SourceLossWeights,loss_weight_arrays
    states=[]
    for directory in (before,after):
        identity=a.read(directory/'state_identity.json')
        probe=FrozenCameraSourceProbe(directory/'complete_state_UNACCEPTED.npz',identity['camera_path'],options['--world-init'],options)
        base=SimpleNamespace(contract={},native_board_factor=probe.native['native_board'],native_ice_factor=probe.native['native_ice'])
        integration=SourceWithNativeBaseCurveIntegration(base,options['--native-base-curve-integration-policy'])
        factor=probe.native['native_ice'];view=factor._qualified_active_fit_view();view=factor if view is None else view
        payload=integration._payloads['additional_real_native_ice_curves'];weights,normalization=loss_weight_arrays(payload,view.row_count)
        cursor=0;rows=[]
        for row in view.rows:
            n=2*len(row['uv']);stop=cursor+n;at=int(row['actual_pts'])
            r=view._residual(probe.q,[row],cameras={at:probe.camera['camera25'][probe.actual_index[at]]})
            loss=None if weights is None else SourceLossWeights(weights[cursor:stop],None if normalization is None else normalization[cursor:stop])
            rows.append(dict(id=row['id'],actual_pts=at,feature_id=row.get('feature_id',row.get('curve',{}).get('feature_id',str(row.get('curve')))),
                cost=weighted_source_cost(r,view.quadratic[cursor:stop],loss)))
            cursor=stop
        if cursor!=view.row_count:raise ValueError('ICE original scalar budget order changed')
        receipt=a.read(directory.parent.parent/'actual_trial_family_cost_pass_0001.json')
        if receipt['complete_q_sha256']!=identity['q_sha256']:raise ValueError('ICE saved cost belongs to another q')
        gold=[r['normalized_cost']for r in receipt['families']if r['original_row_ledger']['family']=='additional_real_native_ice_curves']
        if len(gold)!=1 or abs(sum(r['cost']for r in rows)-gold[0])>2e-8:raise ValueError('ICE per-row values do not reproduce actual family cost')
        states.append(rows)
    if [(r['id'],r['actual_pts'])for r in states[0]]!=[(r['id'],r['actual_pts'])for r in states[1]]:raise ValueError('ICE admitted source rows changed')
    differences=[dict(id=x['id'],actual_pts=x['actual_pts'],feature_id=x['feature_id'],before=x['cost'],after=y['cost'],delta=y['cost']-x['cost'])for x,y in zip(*states)]
    groups=defaultdict(lambda:dict(before=0.,after=0.,rows=0))
    for r in differences:
        g=groups[r['feature_id']];g['before']+=r['before'];g['after']+=r['after'];g['rows']+=1
    emit(stage='native_ICE_actual_original_budget_contribution',total_delta=sum(r['delta']for r in differences),
        by_feature=dict(groups),largest_increases=sorted(differences,key=lambda r:r['delta'],reverse=True)[:6],
        largest_decreases=sorted(differences,key=lambda r:r['delta'])[:3],source_images_decoded=0,source_images_written=0,new_J=False)
    return differences


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--after-state',type=Path,required=True);ap.add_argument('--output',type=Path)
    ap.add_argument('--before-state',type=Path);ap.add_argument('--config',type=Path);ap.add_argument('--common-cap-state',type=Path)
    ap.add_argument('--stdout-values',action='store_true',help='Complete value replay without saving a report')
    ap.add_argument('--reviews-only',action='store_true');args=ap.parse_args();started=time.perf_counter()
    if args.output is not None and args.output.exists():raise FileExistsError(args.output)
    if not args.reviews_only and args.output is None and not args.stdout_values:ap.error('Use --stdout-values for full values without an output report')
    if args.before_state is not None:
        before=args.before_state.resolve();before=before.parent if before.is_file()else before
        a.RUN=before.parent.parent;a.CHILD=before.name
    if args.config is not None:a.CONFIG=args.config.resolve()
    before=a.RUN/'accepted_states'/a.CHILD;after=args.after_state.resolve()
    if after.is_file():after=after.parent
    shots=a.read(a.ROOT/'calibration/program_tracking/source_shots.json')['shots'][1:10]
    reviews=saved_reviews(before,after,shots)
    result=dict(schema='rinkspace.complete-saved-original-source-value-comparison.v1',before=a.ref(before/'state_identity.json'),after=a.ref(after/'state_identity.json'),
        reviews=reviews,script=a.ref(Path(__file__)),source_images_decoded=0,source_images_written=0,new_J_or_fit=False,accuracy_accepted=False)
    if not args.reviews_only:
        result['same_initial_complete_state_and_profile']=verify_common_initial_state(before,after,args.common_cap_state)
        c=ctx.load_context();value,first,second,identity,manifest,borders,camera_checks=prepare_states(c,after)
        if c.options.get('--original-measured-transition-location-source'):
            _,c._comparison_revised_original_locations=install_on_source_context(c,c.options['--original-measured-transition-location-source'])
        result['public_current_camera_and_perimeter_checks']=camera_checks
        result['all86_original_dynamic_caps']=common_parent_guard_comparison(c,value,args.common_cap_state)
        result['camera_parameter_ranges']=camera_parameter_ranges((before,after),shots,c.provider.scene_unit_m)
        result['all_original_absolute_FIT']=original_absolute(c,value,shots)
        motions,receipt=reconstruct_all_motion(c,first);result['motion_source_identity']=receipt
        result['all_original_motion']=compare_motion(c,motions,value,borders,shots)
        result['frozen_profile_token']=identity['frozen_profile_token']
        result['limits']=['Per-source values and all-shot motion are numerical evidence, not whole-shot visual accuracy acceptance.',
            'Production native reviews are reused; their HELD/development source splits are retained.',
            'All86 original protection caps are replayed. The full original programme finite-extent guard is not reconstructed here; saved solver full-domain acceptance remains a separate receipt.',
            'No weights, source UV/sigma/support, role, physical model, camera, or configuration is modified.']
    result['seconds']=time.perf_counter()-started
    if args.output is not None:
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=a.plain,allow_nan=False)+'\n',encoding='utf8')
        emit(stage='comparison_complete',output=str(args.output),sha256=a.sha(args.output),seconds=result['seconds'])
    else:emit(stage='comparison_complete',saved_new_report=False,seconds=result['seconds'])


if __name__=='__main__':main()
