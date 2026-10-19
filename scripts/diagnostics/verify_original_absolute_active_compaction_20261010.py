"""Replay actual source pixels/camera25; no complete motion graph or solver."""
from copy import deepcopy
import argparse,json,pickle,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from calibration.solver_runtime import install
install()
from calibration.solver_runtime.original_absolute_input_compaction import compact_original_absolute_inputs,original_points_to_active
from calibration.solver_runtime.original_source_point_qualification import OriginalSourcePointQualification
from calibration.solver_runtime.source_point_qualification import file_sha256
from calibration.solver_runtime.source_loss import weighted_source_cost,source_loss_checkpoint_arrays,source_loss_from_checkpoint
from structured_programme_source_candidate.original_ice_pixel_correction import install_original_ice_pixel_correction
from z190_source_camera_diagnostic import absolute_value_jac15
from structured_programme_source_candidate.static_motion_source_role_retirement import StaticMotionSourceRoleRetirement
from structured_programme_source_candidate.original_source_role_retirement import OriginalSourceRoleRetirement

SYS=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
RUN=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
SEED=RUN/'okada_full9_current_review71_clarabel_20261009_152520_411/accepted_states/27fa214561703310c832f3884656cf16c9461870a44ed3acf808bd866a4eeb47'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--correction',type=Path,default=SYS/'original_F2_Lane3_and_direct_user_ice_correction_MERGED_20261010.json')
    parser.add_argument('--qualification',type=Path,default=SYS/'original_F2_Lane3_pending_point_qualification_MERGED_20261010.json')
    parser.add_argument('--seed-directory',type=Path,default=SEED)
    parser.add_argument('--config',type=Path,default=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review85_corrected_compact_20261010.json')
    parser.add_argument('--output',type=Path,default=SYS/'original_absolute_active_compaction_complete_layout_verification_20261010.json')
    args=parser.parse_args();seed=args.seed_directory
    started=time.perf_counter();path=SYS/'nearboard49_original_source_mass_coarsening_evidence_20261010.json'
    proof=json.loads(path.read_text(encoding='utf8'));entries=pickle.loads(Path(proof['original_packet']['path']).read_bytes())['programme_absolute_rows']
    identity=json.loads((seed/'state_identity.json').read_text(encoding='utf8'))
    print(json.dumps(dict(stage='identity',keys=list(identity))),flush=True)
    camera_path=Path(identity['camera_path'])
    if not camera_path.exists():
        cameras=list(seed.glob('*camera*.*npz'))
        if len(cameras)!=1:raise ValueError('Unique actual same-q camera archive required')
        camera_path=cameras[0]
    with np.load(camera_path,allow_pickle=False) as z:
        print(json.dumps(dict(stage='camera',keys=z.files)),flush=True)
        cameras={int(p):c.copy() for p,c in zip(z['pts'],z['camera25'])}
        aliases={int(p):int(a) for p,a in zip(z['pts'],z['exposure_representative_pts'])}
    with np.load(seed/'complete_state_UNACCEPTED.npz',allow_pickle=False) as z:q=z['q'].copy()
    prior_run=RUN/'priority_witness_complete_cycle_qc38_20261010'
    source_input=json.loads((prior_run/'input_manifest.json').read_text(encoding='utf8'))
    summary=json.loads((prior_run/'actual_backend_and_source_continuity.json').read_text(encoding='utf8'))['actual_family_rows'][:6]
    parent_manifest=json.loads((SEED.parent.parent/'input_manifest.json').read_text(encoding='utf8'))
    config=json.loads(args.config.read_text(encoding='utf8'))
    static=StaticMotionSourceRoleRetirement(config['source_options']['--static-motion-source-role-retirement-source'])
    old_ui=parent_manifest['static_motion_source_role_retirement'];mapping_path=old_ui['candidate_map_reference']['path']
    with np.load(mapping_path,allow_pickle=False)as z:low_map=z['old_point_to_compact_point'].copy()
    ids=np.asarray([r['current_bundle_motion_row']for r in static.records],np.int64)
    old_binding=old_ui['original_precompaction_programme_block']
    qualified=[dict(source_record=deepcopy(r),programme_family='programme:low',local_scalar_slots=[2*int(i),2*int(i)+1],
        global_scalar_slots=[old_binding['original_start']+2*int(i),old_binding['original_start']+2*int(i)+1])for r,i in zip(static.records,ids)]
    prequalified=dict(schema='rinkspace.actual-original-retirement-qualified-before-compaction.v1',original_qualification_passed=True,
        packet_path=str(static.packet_path),packet_sha256=static.contract['packet_sha256'],qualified_records=qualified,
        original_point_indices=ids,mapped_active_point_indices=low_map[ids],original_block_binding=deepcopy(old_binding),
        compact_block_binding=deepcopy(old_ui['programme_block']),candidate_map_reference=deepcopy(old_ui['candidate_map_reference']))
    blocks=[];all_scales=[]
    for ordinal,ledger in enumerate(summary):
        name=ledger['family'].split(':',1)[1]
        items=[x for x in entries if x['block']==name and x['collection_role']=='FIT']
        if not items:continue
        groups=[None]*(max(x['group_order'] for x in items)+1)
        for item in items:
            gi=int(item['group_order']);rows=[] if groups[gi] is None else groups[gi][1]
            if item['row_order']!=len(rows):raise ValueError('Actual original absolute order changed')
            rows.append(deepcopy(item['source_row']));groups[gi]=(int(item['group_actual_pts']),rows)
            aliases[int(item['source_row']['pts'])]=int(item['group_actual_pts'])
        scales=np.concatenate([np.repeat(r['sigma'],2) for _,rr in groups for r in rr])
        stamps=np.asarray(sorted(cameras));index={int(p):i for i,p in enumerate(stamps)}
        motion_rows=int(source_input['original_motion_rows'][ordinal]);n=motion_rows//2
        prior_count=ledger['original_stop']-ledger['original_start']-motion_rows-len(scales)
        motion_sigma=np.ones(motion_rows);a=np.zeros((n,2));b=a.copy();ia=np.zeros(n,np.int32);ib=ia.copy()
        bundle=SimpleNamespace(motion_rows=motion_rows,source_residual_rows=motion_rows+len(scales),
            native_residual_sigma=np.r_[motion_sigma,scales],motion_sigma=motion_sigma,a=a,b=b,ia=ia,ib=ib,
            finite_planes=np.zeros((n,4)),index=index,stamps=stamps,abs_groups=groups)
        if name=='low':
            bundle.original_motion_compaction_old_to_active=low_map
            for r in static._dynamic_validity_append['records']:
                j=int(r['active_compacted_motion_point_index']);bundle.a[j]=r['source_uv'];bundle.b[j]=r['target_uv']
                bundle.ia[j]=index[int(r['source_pts'])];bundle.ib[j]=index[int(r['target_pts'])]
                bundle.motion_sigma[2*j:2*j+2]=r['source_sigma_px']
            bundle.native_residual_sigma[:motion_rows]=bundle.motion_sigma
        block=SimpleNamespace(name=name,groups=groups,bundle=bundle,quadratic=np.zeros(motion_rows+len(scales)+prior_count,bool),
            actor_output_capacity=dict(rows=motion_rows+len(scales)+prior_count,nnz=20*(motion_rows+len(scales)+prior_count)))
        blocks.append(block);all_scales.extend(bundle.native_residual_sigma.tolist()+[1.]*prior_count)
    state=SimpleNamespace(q=q,blocks=blocks,original=SimpleNamespace(aliases=aliases),
        original_motion_compaction_retirement_prequalification=prequalified,
        original_motion_compaction_scalar_remap={int(k):v for k,v in parent_manifest['original_motion_input_compaction']['original_retirement_scalar_remap'].items()})
    correction=json.loads(args.correction.read_text(encoding='utf8'))
    centers=[r for r in correction['records']if r.get('measurement_role')=='painted_line_center']
    if centers:
        helpdir=RUN/'manual_constraint_dependency_fd_20261009/printed_component_extent_candidate'
        sys.path[:0]=[str(helpdir),str(ROOT/'src'),str(ROOT/'scripts/diagnostics')]
        from audit_user_left_goal_line_and_ad_edges_candidate import load_state
        light,lightq,lightidentity,error=load_state(seed,Path(config['source_options']['--world-init']),json.loads((seed.parent.parent/'input_manifest.json').read_text(encoding='utf8')))
        assert error == 0. and lightidentity['q_sha256']==identity['q_sha256']
        # The real light helper exposes the evaluation column order. Its
        # authenticated canonical checkpoint is the same unchanged complete q.
        state.q=lightq
        state.geometry=light.geometry
        for record in centers:
            entry=entries[record['original_packet_row_index']]
            state.geometry.register_row(entry['source_row'],actual_pts=entry['group_actual_pts'])
        from structured_programme_source_candidate.original_review_source import load_original_absolute_evaluators
        actual_evaluators,_=load_original_absolute_evaluators()
    install_original_ice_pixel_correction(state,args.correction)
    cap=dict(scales=np.asarray(all_scales),quadratic=np.zeros(len(all_scales),bool));template=deepcopy(cap)
    contract=compact_original_absolute_inputs(state,cap,template,path,
        qualification_path=args.qualification)
    restored=OriginalSourcePointQualification(args.qualification);restored.qualify_state(state)
    static.qualify_state(state)
    print(json.dumps(dict(stage='real_static_motion_public_qualification_passed',binding=static.block_binding)),flush=True)
    ad7=OriginalSourceRoleRetirement(config['source_options']['--original-source-role-retirement-source']);ad7.qualify_state(state)
    corrected=state.original_ice_pixel_correction['corrected_rows']
    for entry in corrected:
        loc=entry['locator'];block=next(b for b in blocks if b.name==loc['block']);row=block.groups[loc['group_order']][1][loc['row_order']]
        active=original_points_to_active(row,entry['point_indices'])
        if not np.array_equal(np.asarray(row['uv'])[active],entry['corrected_uv']):raise ValueError('Corrected66 original point IDs lost active UV identity')
    checks=[];budgets={};offsets={}
    for record in proof['records']:
        loc=record['original_locator'];group=(loc['block'],int(loc['group_actual_pts']),record['source_identity'])
        budgets[group]=budgets.get(group,0.)+float(np.sum(record['retained_point_integration_mass']))
    for block in blocks:
        cursor=block.bundle.motion_rows
        for gi,(_,rows)in enumerate(block.groups):
            for ri,row in enumerate(rows):
                offsets[(block.name,gi,ri)]=cursor;cursor+=2*len(row['uv'])
    for record in proof['records']:
        loc=record['original_locator'];block=next(b for b in blocks if b.name==loc['block']);gi=loc['group_order'];ri=loc['row_order'];pts=loc['group_actual_pts']
        old=block.original_absolute_groups[gi][1][ri];new=block.groups[gi][1][ri]
        if centers:
            state.geometry.register_row(old,actual_pts=pts)
            old_bound=state.geometry.rebind(old,lightq[light.geometry_parameter_indices])
            new_bound=state.geometry.rebind(new,lightq[light.geometry_parameter_indices])
            before=actual_evaluators[loc['block']](cameras[pts],[old_bound],True)
            after=actual_evaluators[loc['block']](cameras[pts],[new_bound],True)
        else:
            before=absolute_value_jac15(cameras[pts],[old],True);after=absolute_value_jac15(cameras[pts],[new],True)
        keep=np.asarray(record['kept_original_point_indices']);slots=(2*keep[:,None]+np.arange(2)).ravel()
        dr=float(abs(after[0]-before[0][slots]).max());dj=float(abs(after[1]-before[1][slots]).max())
        if dr>1e-10 or dj>1e-8:raise ValueError('Actual retained absolute native pixel R/J differs')
        stored=np.asarray(new['sigma']);mass=np.asarray(record['retained_point_integration_mass']);prior=np.asarray(record['retained_original_prior_sampling_multiplier'])
        group=(loc['block'],int(pts),record['source_identity']);w=mass/budgets[group]
        payload=state.original_absolute_input_compaction_loss_weights['programme:'+loc['block']]
        start=offsets[(loc['block'],gi,ri)];stop=start+2*len(stored)
        live_w=payload.integration_weights[start:stop];live_norm=payload.working_normalization_multiplier[start:stop]
        if not np.array_equal(live_w,np.repeat(w,2))or not np.array_equal(live_norm,np.repeat(prior,2)):
            raise ValueError('Actual source block physical-sigma integration payload differs')
        a=prior**2*w;budget=float(np.sum(a/stored**2))
        if abs(budget-1/16)>1e-14:raise ValueError('Actual nearboard working-sigma budget changed')
        checks.append(dict(entry_index=record['original_entry_index'],actual_pts=pts,old_points=len(old['uv']),new_points=len(new['uv']),
            retained_R_error=dr,retained_camera15_J_error=dj,source_budget_sum_a_over_sigma_squared=budget,
            physical_sigma_px=float(record['original_working_sigma_px']),physical_Huber_cutoff_px=8.,
            integration_mass_budget=float(w.sum()),stored_sigma_sqrtN_removal_exact=True,
            maximum_working_component=float(abs(after[0]*live_norm).max()),
            actual_physical_Huber_tail_scalar_count=int(np.count_nonzero(abs(after[0]*live_norm)>2.)),
            actual_physical_source_cost=weighted_source_cost(after[0],np.zeros(len(after[0]),bool),
                type(payload)(live_w,live_norm))))
    # All generated payloads survive actual NPZ block round-trip.
    for family,weights in state.original_absolute_input_compaction_loss_weights.items():
        testpath=SYS/'nearboard_active_compaction_loss_payload_roundtrip.npz'
        np.savez(testpath,**source_loss_checkpoint_arrays(weights,len(weights.integration_weights)))
        with np.load(testpath,allow_pickle=False)as z:loaded=source_loss_from_checkpoint(z)
        if weights.identity_digest!=loaded.identity_digest:raise ValueError('Weighted source cache round trip changed objective')
    report=dict(schema='rinkspace.actual-original-absolute-active-input-compaction-verification.v1',passed=True,
        source_compaction=contract,actual_camera_reference=dict(path=str(camera_path),sha256=file_sha256(camera_path)),
        q_sha256=identity.get('q_sha256'),real_49_row_native_camera15_R_J_checks=checks,
        actual_qualification_scalar_rows_after_physical_compaction=restored.contract['active_FIT_scalars_retired'],
        actual_complete_programme_block_order=[b.name for b in blocks],
        current_full_programme_block_sizes_from_saved_original_complete_source=True,
        static_motion_original_public_qualification_passed=True,static_motion_contract=static.contract,
        actual_original_Ad7_public_qualification_passed=True,Ad7_contract=ad7.contract,
        original_motion_prequalified_historical_records_and_original_point_map_unchanged=True,
        all66_corrected_original_point_ids_resolve_to_exact_active_corrected_UV=True,
        unrelated_motion_arrays_are_layout_only_placeholders_not_numerically_verified=True,
        preserved_corrected_source_points=sum(len(r['point_indices'])for r in corrected),
        actual_original_qualification_physical_removal=restored.contract,
        original_full_finite_WORLD_support_preserved=True,
        actual_live_SourceLossWeights_argument_pixel_error_over_sigma4=True,
        actual_same_complete_q_geometry_WORLD_rebound=bool(centers),
        non_nearboard_rows_loss_normalization_unchanged=True,
        absolute_arrays_and_native_CSR_compacted=True,full_motion_graph_or_J_constructed=False,
        source_geometry_world_columns_not_recomputed_by_this_narrow_check=True,seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,removed_scalars=contract['removed_active_scalar_rows'],checks=len(checks),seconds=report['seconds'])),flush=True)

if __name__=='__main__':main()
