"""Bind reviewed native appends and curve budgets to one complete saved q.

Uses the existing public truth factors and startup scope. Does not prepare a
programme graph, reuse a normal/J, change an old cap, or start the solver.
"""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import argparse,hashlib,json,os,sys,time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2];DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
SYS=DATA/'diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/diagnostics')]
from calibration.solver_runtime import install
install()
import audit_current_user_ice_truth as a
from calibration.solver_runtime.original_board_curve_integration import file_reference
from calibration.solver_runtime.curve_source_grouping import wrap_native_base_curve_integration
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.human_stroke_truth import UserConfirmedSourceTruth,human_support_operator_contract
from structured_programme_source_candidate.reviewed_manual_append_restore import CURRENT_AUDIT_SCHEMA,CURRENT_VALUES_SCHEMA,SCHEMA
from z190_streaming_source_normal import SourceRowBlock
from calibration.solver_runtime.source_loss import weighted_source_cost,source_irls_weights
import numpy as np
from scipy.sparse import csr_matrix

def same_except_references(before,after,key):
    expected=deepcopy(after);expected[key]=before[key]
    assert expected==before

def rebind(options,board_path,ice_path,material_path,out):
    refs={'native_board_edge':file_reference(board_path),'native_ice_curve':file_reference(ice_path),
          'native_board_material_point':file_reference(material_path)}
    packets={family:a.read(ref['path'])for family,ref in refs.items()};oldpackets={};changes=[]
    authority=a.read(options['--direct-human-truth-authority']);human=deepcopy(authority)
    confirmed_old=a.read(options['--user-confirmed-source-truth-authority']);confirmed=deepcopy(confirmed_old)
    for kind,document in (('human',human),('confirmed',confirmed)):
        for record in document['records']:
            family=record['family']
            if family not in refs:continue
            old=deepcopy(record['packet_reference']);key=old['path']
            if key not in oldpackets:
                assert file_reference(key)==dict(path=str(Path(key).resolve()),sha256=old['sha256']);oldpackets[key]=a.read(key)
            index=record['original_row_index'];original=oldpackets[key]['rows'][index];current=packets[family]['rows'][index]
            assert current==original and current['id']==record.get('source_row_id',record['id'])
            canonical=json.dumps(current,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
            assert hashlib.sha256(canonical).hexdigest()==record['source_row_canonical_json_sha256']
            record['packet_reference']=refs[family]
            changes.append(dict(kind=kind,id=record['id'],original_row_index=index,old_packet=old,new_packet=refs[family],
                original_source_row_canonical_UV_sigma_support_curve_and_role_exact=True))
    for before,after in zip(authority['records'],human['records']):same_except_references(before,after,'packet_reference')
    for before,after in zip(confirmed_old['records'],confirmed['records']):same_except_references(before,after,'packet_reference')
    hpath=out/'human_same54_new_native_container_bindings.json';a.save(hpath,human)
    cpath=out/'confirmed_same30_new_native_container_bindings.json';a.save(cpath,confirmed)
    oldfull=a.read(options['--direct-human-full-stroke-guard']);full=deepcopy(oldfull);full['primary_authority_reference']=file_reference(hpath)
    fpath=out/'human_same54_full_guard.json';a.save(fpath,full)
    caps=a.read(options['--direct-human-stroke-cap-support']);caps['primary_authority_reference']=file_reference(hpath)
    caps['full_trace_reference']=dict(caps['full_trace_reference'],**file_reference(fpath))
    capath=out/'human_same54_stroke_caps.json';a.save(capath,caps)
    location=a.read(options['--practical-human-location-envelope']);location['authority_references']['native']=file_reference(hpath)
    lpath=out/'human_same54_original_location_envelope.json';a.save(lpath,location)
    next_options=deepcopy(options)
    mapping={'--additional-board-source':board_path,'--additional-ice-source':ice_path,
        '--additional-board-material-point-source':material_path,'--direct-human-truth-authority':hpath,
        '--direct-human-full-stroke-guard':fpath,'--direct-human-stroke-cap-support':capath,
        '--practical-human-location-envelope':lpath,'--user-confirmed-source-truth-authority':cpath}
    for key,path in mapping.items():next_options[key]=str(Path(path).resolve())
    a.save(out/'protected_container_rebinding_receipt.json',dict(schema='rinkspace.unchanged-native-source-append-protected-container-rebinding.v1',passed=True,
        parent_human=file_reference(options['--direct-human-truth-authority']),parent_confirmed=file_reference(options['--user-confirmed-source-truth-authority']),
        new_source_options={k:file_reference(v)for k,v in mapping.items()},records=changes,
        original_human54_and_confirmed30_all_fields_except_packet_reference_exact=True,
        original_full_guard_array_bytes_and_stroke_cap_records_and_location_records_preserved=True))
    return next_options

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-config',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True);p.add_argument('--ice',type=Path,required=True)
    p.add_argument('--material',type=Path,required=True)
    p.add_argument('--native-policy',type=Path,required=True);p.add_argument('--PP6-proof',type=Path,required=True)
    p.add_argument('--output-config',type=Path,required=True);opt=p.parse_args();began=time.perf_counter()
    config=a.read(opt.input_config);previous=config['source_options'];directory=opt.state.resolve();identity=a.read(directory/'state_identity.json')
    parent=a.read(directory/'truth_restoration_state.json');old_manifest=a.read(directory.parent.parent/'input_manifest.json')
    assert parent['complete_q_sha256']==identity['q_sha256'] and sum(len(r['group_ids'])for r in parent['families'].values())==86
    assert parent['source_input_manifest_reference']==file_reference(directory.parent.parent/'input_manifest.json')
    out=SYS/('current_source_budget_append_'+identity['q_sha256'][:8]);out.mkdir(exist_ok=True)
    options=rebind(previous,opt.board,opt.ice,opt.material,out)
    policy=a.read(opt.native_policy);assert policy['packets']['board']==file_reference(opt.board) and policy['packets']['ice']==file_reference(opt.ice)
    if policy.get('separate_declared_packet_sampling_normalization')is not True:
        policy['separate_declared_packet_sampling_normalization']=True
        policy_path=out/'native_base_policy_explicit_packet_sampling_separation.json';a.save(policy_path,policy)
    else:policy_path=opt.native_policy.resolve()
    options['--native-base-curve-integration-policy']=str(policy_path)
    original_policy=SYS/'original_ICE_all79_physical_curve_per_exposure_integration_POLICY_20261011.json'
    original_proof=SYS/'actual_original_ICE79_fixed_qfe074_curve_budget_R_oneFD_and_loss_20261011.json'
    assert a.read(original_proof)['passed'] and a.read(original_proof)['complete_q_sha256']==identity['q_sha256']
    options['--original-ice-curve-integration-policy']=str(original_policy)
    pp=a.read(opt.PP6_proof);assert pp['passed'] and pp['complete_q_sha256']==identity['q_sha256'] and pp['rows']==6
    state,q,loaded,error=a.load_state(directory,Path(previous['--world-init']),old_manifest);assert error==0.
    loaded['_accepted_directory']=str(directory);qbefore=q.tobytes()
    old_board=NativeBoardEdgeFactor(state,previous['--additional-board-source'],len(q));board=NativeBoardEdgeFactor(state,options['--additional-board-source'],len(q))
    old_ice=NativeIceCurveFactor(state,previous['--additional-ice-source'],len(q));ice=NativeIceCurveFactor(state,options['--additional-ice-source'],len(q))
    assert a.read(board.packet_path)['rows'][:len(a.read(old_board.packet_path)['rows'])]==a.read(old_board.packet_path)['rows']
    assert a.read(ice.packet_path)['rows'][:len(a.read(old_ice.packet_path)['rows'])]==a.read(old_ice.packet_path)['rows']
    for old,new in ((old_board,board),(old_ice,ice)):
        assert np.array_equal(old.scales,new.scales[:len(old.scales)]) and np.array_equal(old.quadratic,new.quadratic[:len(old.quadratic)])
    old_human=a.bind_human(state,old_board,old_ice,previous,len(q));human=a.bind_human(state,board,ice,options,len(q))
    wide=a.wide_factor(state,loaded,previous,len(q),previous['--practical-human-location-envelope'])
    scene=a.read(directory/'scene_UNACCEPTED.json')
    old_circle,old_material,old_placement=a.existing_LOW_source_owners(state,q,directory,previous,scene)
    circle,material,placement=a.existing_LOW_source_owners(state,q,directory,options,scene)
    assert a.read(material.packet_path)['rows'][:len(a.read(old_material.packet_path)['rows'])]==a.read(old_material.packet_path)['rows']
    assert np.array_equal(old_material.scales,material.scales[:len(old_material.scales)])
    assert np.array_equal(old_material.quadratic,material.quadratic[:len(old_material.quadratic)])
    old_confirmed=UserConfirmedSourceTruth(state,previous['--user-confirmed-source-truth-authority'],old_board,old_ice,
        parameter_count=len(q),circle_factor=old_circle,material_factor=old_material,placement_source=old_placement)
    confirmed=UserConfirmedSourceTruth(state,options['--user-confirmed-source-truth-authority'],board,ice,
        parameter_count=len(q),circle_factor=circle,material_factor=material,placement_source=placement)
    before=a.values(old_human,wide,old_confirmed,q);after=a.values(human,wide,confirmed,q)
    for key in ('primary','full','wide','confirmed_carriers','confirmed_details','buffers'):assert a.exact(before[key],after[key]),key
    assert q.tobytes()==qbefore
    families={}
    for name,row in parent['families'].items():
        actual=after['buffers'][name];assert actual.shape==np.asarray(row['limits']).shape and np.all(actual<=np.asarray(row['limits'])+1e-8)
        families[name]=dict(group_ids=row['group_ids'],old_count=len(row['group_ids']),violations_px=actual.tolist(),
            limits=deepcopy(row['limits']),uncertainties=deepcopy(row['uncertainties']))
    base=SimpleNamespace(native_board_factor=board,native_ice_factor=ice,contract={},count=len(q),scales=np.r_[board.scales,ice.scales],quadratic=np.r_[board.quadratic,ice.quadratic])
    wrapper=wrap_native_base_curve_integration(base,policy_path)
    native_checks=[];column=int(state.geometry_parameter_indices[14]);h=1e-6
    for name,old,new,family in (('board',old_board,board,'additional_real_native_board_edges'),('ice',old_ice,ice,'additional_real_native_ice_curves')):
        raw=new.block(q,False);oldr=old.block(q,False).r;assert np.array_equal(raw.r[:len(oldr)],oldr)
        active=(new._qualified_active_fit_view()or new)if name=='ice' else new
        plus=q.copy();minus=q.copy();plus[column]+=h;minus[column]-=h
        j=(active._residual(plus,active.rows)-active._residual(minus,active.rows))/(2*h)
        block=SourceRowBlock(raw.r,csr_matrix(j[:,None]),raw.quadratic,raw.scales,dict(family=family))
        weighted=wrapper.apply_block(block);w=weighted.loss_weights
        coeff=source_irls_weights(block.r,block.quadratic,w)
        sampling=wrapper.native_base_curve_integration.get('declared_packet_sampling_normalizations',[])
        packet_samples=[row for row in sampling if row['family']==name]
        normalization=w.working_normalization_multiplier
        normalized_scalar_indices=np.empty(0,int)if normalization is None else np.flatnonzero(normalization!=1.)
        if len(normalized_scalar_indices):
            assert np.all(block.quadratic[normalized_scalar_indices])
            assert np.allclose(coeff[normalized_scalar_indices],1.,rtol=2e-13,atol=2e-13)
        native_checks.append(dict(family=family,old_rows=len(oldr),new_rows=len(raw.r),added_scalars=len(raw.r)-len(oldr),
            old_R_prefix_bitexact=True,new_actual_R_sha256=hashlib.sha256(raw.r.tobytes()).hexdigest(),
            public_world14_FD_column_sha256=hashlib.sha256(j.tobytes()).hexdigest(),
            new_integrated_cost=weighted_source_cost(block.r,block.quadratic,w),new_integrated_gradient_world14=float(j@(coeff*block.r)),
            new_integrated_normal_world14=float(j@(coeff*j)),R_J_scales_quadratic_objects_preserved=True,
            encoded_sampling_scalar_count=len(normalized_scalar_indices),encoded_sampling_rows=packet_samples,
            encoded_quadratic_cost_gradient_normal_coefficients_preserved=True))
    mq=q[:material.parameter_count];old_mq=q[:old_material.parameter_count]
    raw=material.block(mq,False);oldr=old_material.block(old_mq,False).r
    assert np.array_equal(raw.r[:len(oldr)],oldr)
    plus=mq.copy();minus=mq.copy();plus[column]+=h;minus[column]-=h
    j=(material._residual(plus,material.rows)-material._residual(minus,material.rows))/(2*h)
    coeff=source_irls_weights(raw.r,raw.quadratic)
    native_checks.append(dict(family='additional_real_native_board_material_points',old_rows=len(oldr),new_rows=len(raw.r),
        added_scalars=len(raw.r)-len(oldr),old_R_prefix_bitexact=True,
        new_actual_R_sha256=hashlib.sha256(raw.r.tobytes()).hexdigest(),
        public_world14_FD_column_sha256=hashlib.sha256(j.tobytes()).hexdigest(),
        new_integrated_cost=weighted_source_cost(raw.r,raw.quadratic,None),
        new_integrated_gradient_world14=float(j@(coeff*raw.r)),new_integrated_normal_world14=float(j@(coeff*j)),
        declared_quadratic_curve_budget_encoded_in_sigma_once=True,scales_quadratic_prefix_bitexact=True))
    source_delta=sum(r['added_scalars']for r in native_checks);expected_count=old_manifest['source_rows']+source_delta
    transaction=out/'source_target_transaction.json'
    a.save(transaction,dict(schema='rinkspace.same-q-reviewed-native-source-and-original-curve-budget-target-revision.v1',passed=True,
        complete_q_sha256=identity['q_sha256'],complete_state_reference=file_reference(directory/'complete_state_UNACCEPTED.npz'),
        parent_truth_state=file_reference(directory/'truth_restoration_state.json'),parent_runtime_manifest=file_reference(directory.parent.parent/'input_manifest.json'),
        source_rows_before=old_manifest['source_rows'],expected_source_rows_after=expected_count,source_scalar_delta=source_delta,
        native_source_checks=native_checks,native_base_curve_integration=wrapper.native_base_curve_integration,
        original_ICE_budget_actual_proof=file_reference(original_proof),existing86_raw_queries_bounds_caps_uncertainties_and_roles_bitexact=True,
        complete_actor_full_source_count_and_all86_startup_checks_still_required=True,old_complete_normal_J_profiles_cache_reuse_authorized=False,
        fresh_complete_source_J_and_profiles_required=True,solver_started=False,accuracy_accepted=False))
    manual_keys=('--additional-board-source','--additional-ice-source','--additional-board-material-point-source','--direct-human-truth-authority','--direct-human-full-stroke-guard',
        '--direct-human-stroke-cap-support','--practical-human-location-envelope','--user-confirmed-source-truth-authority')
    manual={key:file_reference(options[key])for key in manual_keys}
    qualification=ice.source_point_qualification
    assert qualification is not None;manual['--additional-ice-source']=qualification.contract['original_packet']
    audit=out/'same_q_public_truth_and_native_append_audit.json'
    a.save(audit,dict(schema=CURRENT_AUDIT_SCHEMA,passed=True,complete_q_sha256=identity['q_sha256'],
        complete_state_reference=file_reference(directory/'complete_state_UNACCEPTED.npz'),old_caps_reference=file_reference(directory/'truth_restoration_state.json'),
        parent_manifest_reference=file_reference(directory.parent.parent/'input_manifest.json'),new_source_options=manual,
        total_added_source_scalars=source_delta,added_direct_manual_centre_scalars=0,new_complete_source_scalar_count=expected_count,
        parent_group_count=86,new_group_count=86,families=families,current_source_target_transaction=file_reference(transaction),
        existing86_caps_uncertainties_group_ids_exact=True,existing86_source_support_uncertainties_group_ids_exact=True,
        actual_full86_query_and_carrier_invariance_bitexact=True,actual_native_R_prefix_and_world14_FD=native_checks,
        full_programme_J_graph_or_solver_not_created=True,total_seconds=time.perf_counter()-began))
    values=out/'exact_parent_caps_values.json';a.save(values,dict(schema=CURRENT_VALUES_SCHEMA,complete_q_sha256=identity['q_sha256'],audit=file_reference(audit),families=families))
    operators={'static_motion_source_role_retirement':'--static-motion-source-role-retirement-source',
        'original_motion_far_board_role_correction':'--original-motion-far-board-role-source','original_ice_pixel_correction':'--original-ice-pixel-correction-source',
        'original_source_point_qualification':'--original-source-point-qualification-source','original_absolute_input_compaction':'--original-absolute-input-compaction-source',
        'original_board_curve_integration':'--original-board-curve-integration-policy','original_ice_curve_integration':'--original-ice-curve-integration-policy',
        'native_base_curve_integration':'--native-base-curve-integration-policy'}
    scope=dict(schema=SCHEMA,same_q_public_audit=file_reference(audit),exact_caps_values=file_reference(values),
        current_source_target_transaction=file_reference(transaction),source_qualified_native_packet_sources={'additional_real_native_ice_curves':file_reference(opt.ice)},
        admitted_operator_sources={key:dict(reference=file_reference(options[option]))for key,option in operators.items()},
        admitted_engineering_prior_append=dict(actual_same_q_public_audit=file_reference(opt.PP6_proof),rows=6,new_rows_in_this_revision=0),
        all86_existing_human_groups_and_caps_preserved=True)
    binding=out/'startup_binding.json';a.save(binding,scope)
    with np.load(directory/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:
        damping=float(z['damping']);fields=len(z.files);fullq=z['q'].copy();fraction=float(z['next_line_search_fraction'])
    assert hashlib.sha256(fullq.tobytes()).hexdigest()==identity['q_sha256'] and fields==77
    options.update({'--camera-seed':str(directory/'complete_state_UNACCEPTED.npz'),'--truth-restoration-state':str(directory/'truth_restoration_state.json'),
        '--initial-damping':damping,'--truth-restoration-reviewed-manual-append-audit':str(binding),
        '--save-initial-linearization-cache':True,'--save-complete-source-directions':True})
    for key in list(options):
        if key.startswith('--reuse-')and key!='--reuse-source-jacobian':options.pop(key)
    options.pop('--accepted-steps',None)
    config['source_options']=options;config['source_manifest_sha256']=a.sha(ROOT/'calibration/solver_runtime/source_manifest.json')
    config['continuation']=dict(complete_accepted_state=str(directory/'complete_state_UNACCEPTED.npz'),checkpoint_file_sha256=identity['state_npz_sha256'],
        truth_restoration_state=str(directory/'truth_restoration_state.json'),truth_restoration_state_sha256=a.sha(directory/'truth_restoration_state.json'),
        state=dict(q_sha256=identity['q_sha256'],parameters=len(fullq),fields=77,profiles=18,damping=damping,next_line_search_fraction=fraction),
        expected_complete_active_source_scalars=expected_count,current_truth_group_count=86,current_source_target_transaction=file_reference(transaction),
        reviewed_source_target_startup_binding=file_reference(binding),same_q_actual86_audit=file_reference(audit),
        next_start_requires_fresh_source_J=True,no_cached_J_H_g_or_prefix_imported=True,solver_run_started=False,accuracy_accepted=False)
    config['preparation_status']=dict(final_programme_configuration=True,ready_to_start=True,
        actual_full86_query_and_native_R_oneFD_budget_proof_passed=True,complete_startup_inventory_and_protected_truth_still_required=True)
    a.save(opt.output_config,config)
    print(json.dumps(dict(config=file_reference(opt.output_config),binding=file_reference(binding),actual_audit=file_reference(audit),
        source_scalar_delta=source_delta,expected_source_scalars=expected_count,total_seconds=time.perf_counter()-began,solver_started=False)))

if __name__=='__main__':main()
