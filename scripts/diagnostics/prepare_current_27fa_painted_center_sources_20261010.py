"""Source-only correction transaction and exact q27fa/85 startup metadata."""
from pathlib import Path
from copy import deepcopy
import argparse, hashlib, json, os, pickle, sys, time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','RINKSPACE_MOTION_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
DATA=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
RUNS=DATA/'calibration/runs/solver_acceleration_20261003'
SYS=DATA/'diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010'
HELP=RUNS/'manual_constraint_dependency_fd_20261009/printed_component_extent_candidate'
STATE=RUNS/'okada_full9_current_review71_clarabel_20261009_152520_411/accepted_states/27fa214561703310c832f3884656cf16c9461870a44ed3acf808bd866a4eeb47'
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/diagnostics'),str(HELP)]
from audit_user_left_goal_line_and_ad_edges_candidate import load_state,bind_human,wide_factor,values,exact
from existing_LOW_circle_material_owner_light import existing_LOW_source_owners
from structured_programme_source_candidate.native_board_edge_source import NativeBoardEdgeFactor
from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
from structured_programme_source_candidate.human_stroke_truth import UserConfirmedSourceTruth
from structured_programme_source_candidate.original_ice_pixel_correction import install_original_ice_pixel_correction
from structured_programme_source_candidate.original_review_source import load_original_absolute_evaluators
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf8')
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else v.item()if isinstance(v,np.generic)else str(v))+'\n',encoding='utf8')

def refresh_current_location_only(config_path,receipt_path,output_config):
    """Requery only an appended group's changed support at the same complete q."""
    began=time.perf_counter();config=read(config_path);options=config['source_options'];receipt=read(receipt_path)
    scope=read(options['--truth-restoration-reviewed-manual-append-audit'])
    audit=read(scope['same_q_public_audit']['path']);frozen=read(scope['exact_caps_values']['path'])
    proof=read(audit['current_source_target_transaction']['path'])
    old_location=read(options['--practical-human-location-envelope']);new_ref=receipt['paths']['location']
    if ref(new_ref['path'])!=new_ref:raise ValueError('Corrected location receipt bytes changed')
    location=read(new_ref['path']);allowed={'normal_expansion_px','tangent_expansion_px',
        'engineering_uncorrelated_location_scale_px','desired_minimum_signed_location_radius_px'}
    if old_location['records'][:-1]!=location['records'][:-1]:raise ValueError('Only new TOP support may change')
    before=old_location['records'][-1];after=location['records'][-1]
    if {k:v for k,v in before.items()if k not in allowed}!={k:v for k,v in after.items()if k not in allowed}or any(after[k]!=0. for k in allowed):
        raise ValueError('New TOP change must remove only the extra engineering expansion')
    manual=dict(board='--additional-board-source',ice='--additional-ice-source',human_authority='--direct-human-truth-authority',
        full_guard='--direct-human-full-stroke-guard',stroke_caps='--direct-human-stroke-cap-support',confirmed_authority='--user-confirmed-source-truth-authority')
    for name,option in manual.items():
        if ref(options[option])!=receipt['paths'][name]:raise ValueError('Existing same-q packet changed: '+name)
    seed=Path(options['--camera-seed']).parent
    state,q,identity,error=load_state(seed,Path(options['--world-init']),read(seed.parent.parent/'input_manifest.json'))
    if error!=0. or identity['q_sha256']!=audit['complete_q_sha256']:raise ValueError('Source support refresh complete q differs')
    board=NativeBoardEdgeFactor(state,options['--additional-board-source'],len(q))
    ice=NativeIceCurveFactor(state,options['--additional-ice-source'],len(q))
    options['--practical-human-location-envelope']=new_ref['path'];human=bind_human(state,board,ice,options,len(q))
    from structured_programme_source_candidate.human_stroke_truth import direct_manual_centre_factors
    actual_centres=direct_manual_centre_factors(state,human,options['--additional-board-source'],options['--additional-ice-source'],len(q))
    parent_manifest=read(audit['parent_manifest_reference']['path'])
    old_centre_count=parent_manifest['source_contract']['direct_manual_centre_secondary']['added_scalar_rows']
    new_centre_count=sum(f.row_count for f in actual_centres);direct_delta=new_centre_count-old_centre_count
    # The new human TOP also enters the already-enabled sparse centre target.
    # Count its real factor rows, in addition to the NativeBoard append.
    audit['total_added_source_scalars']+=direct_delta-audit['added_direct_manual_centre_scalars']
    audit['added_direct_manual_centre_scalars']=direct_delta
    audit['new_complete_source_scalar_count']=parent_manifest['source_rows']+audit['total_added_source_scalars']
    if human.records[-1]['row']['id']!=after['id']or frozen['families']['native']['old_count']!=len(human.records)-1:
        raise ValueError('Only the one genuinely appended TOP group may be requeried')
    primary=human._queries_for_records(q,human.records[-1:])[0]['query']
    full=human._queries_for_records(q,human.full_stroke_records[-1:])[0]['query']
    violation=max(float(abs(primary['residual']).max()),float(abs(full['residual']).max()))
    native=frozen['families']['native'];native['limits'][-1]=violation;native['uncertainties'][-1]=0.;native['violations_px'][-1]=violation
    proof['actual_all_groups_violations_at_exact_original_q']['native'][-1]=violation
    proof.update(current_user_append_receipt=ref(receipt_path),source_location_revision=dict(
        parent_actual_all86_transaction=audit['current_source_target_transaction'],
        same_q_old85_source_arrays_support_and_caps_exact=True,old85_queries_not_recomputed=True,
        new_TOP_only_actual_primary_and_full_guard_recomputed=True,new_location=new_ref,
        engineering_expansion_and_uncertainty_px=0.,actual_primary_violation_px=float(abs(primary['residual']).max()),
        actual_full_guard_violation_px=float(abs(full['residual']).max()),new_initial_recovery_cap_px=violation,
        old_direct_manual_centre_scalar_count=old_centre_count,actual_new_direct_manual_centre_scalar_count=new_centre_count,
        actual_added_direct_manual_centre_scalars=direct_delta,centre_FIT_rows_counted_from_real_public_factors=True,
        original_user_ink_cell_and_measured_registration_support_only=True,seconds=time.perf_counter()-began))
    tag=f'q{identity["q_sha256"][:8]}_truth86_TOP_zero_extra_current_target'
    proofpath=SYS/(tag+'_transaction.json');write(proofpath,proof)
    audit['families']=deepcopy(frozen['families']);audit['new_source_options']['--practical-human-location-envelope']=new_ref
    audit['current_source_target_transaction']=ref(proofpath)
    auditpath=SYS/(tag+'_startup_audit.json');write(auditpath,audit)
    frozen['audit']=ref(auditpath);valuespath=SYS/(tag+'_caps.json');write(valuespath,frozen)
    scope.update(same_q_public_audit=ref(auditpath),exact_caps_values=ref(valuespath),current_source_target_transaction=ref(proofpath))
    scopepath=SYS/(tag+'_startup_binding.json');write(scopepath,scope)
    options['--truth-restoration-reviewed-manual-append-audit']=str(scopepath)
    config['continuation'].update(current85_caps_source=ref(valuespath),startup_binding=ref(scopepath),
        current_source_target_transaction=ref(proofpath),new_TOP_zero_engineering_expansion=True,
        expected_complete_active_source_scalars=audit['new_complete_source_scalar_count'])
    config['continuation']['current_caps_source']=ref(valuespath)
    config['continuation']['same_q_actual_source_audits']=[ref(proofpath),ref(auditpath)]
    config['source_manifest_sha256']=sha(ROOT/'calibration/solver_runtime/source_manifest.json')
    write(output_config,config)
    print(json.dumps(dict(config=str(output_config),proof=str(proofpath),actual_new_TOP_violation_px=violation,
        engineering_uncertainty_px=0.,old85_caps_exact=True,seconds=time.perf_counter()-began)),flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--extra-correction',type=Path,action='append',default=[])
    ap.add_argument('--board-packet',type=Path)
    ap.add_argument('--ice-packet',type=Path)
    ap.add_argument('--source-append-receipt',type=Path)
    ap.add_argument('--refresh-current-location-config',type=Path)
    ap.add_argument('--qualification-parent',type=Path,action='append',default=[])
    ap.add_argument('--seed-directory',type=Path,default=STATE)
    ap.add_argument('--output-config',type=Path,default=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review85_painted_center_ad4_corrected_compact_20261010.json')
    a=ap.parse_args();started=time.perf_counter()
    if a.refresh_current_location_config:
        if a.source_append_receipt is None:raise ValueError('Exact changed-support append receipt required')
        return refresh_current_location_only(a.refresh_current_location_config,a.source_append_receipt,a.output_config)
    if a.board_packet is None:raise ValueError('Actual board source required')
    basepath=ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_review85_corrected_compact_20261010.json'
    config=read(basepath);options=config['source_options'];seed=a.seed_directory
    original_confirmed_reference=options['--user-confirmed-source-truth-authority']
    append_receipt=read(a.source_append_receipt)if a.source_append_receipt else None
    if append_receipt is not None:
        for key,option in dict(human_authority='--direct-human-truth-authority',full_guard='--direct-human-full-stroke-guard',
                stroke_caps='--direct-human-stroke-cap-support',location='--practical-human-location-envelope').items():
            reference=append_receipt['paths'][key]
            if ref(reference['path'])!=reference:raise ValueError('Source append receipt bytes changed: '+key)
            options[option]=reference['path']
        if ref(a.board_packet)!=append_receipt['paths']['board']or a.ice_packet is None or ref(a.ice_packet)!=append_receipt['paths']['ice']:
            raise ValueError('Board/ice source differs from exact current-user append receipt')
    parents=[Path(options['--original-ice-pixel-correction-source']),
             SYS/'original_C_seven_center_RGB_correction_DERIVED_20261010.json',
             SYS/'original_Lane1_six_center_RGB_correction_DERIVED_20261010.json']+a.extra_correction
    packets=[read(p)for p in parents];merged=deepcopy(packets[0]);records=[];evidence=[];seen=set();evidence_seen=set()
    for p in packets:
        for field in ('schema','source_video_sha256','time_base','original_absolute_packet'):
            if p[field]!=merged[field]:raise ValueError('Correction parents differ: '+field)
        for r in p['records']:
            i=r['original_packet_row_index']
            if i in seen:raise ValueError('Correction parents overlap one original row')
            seen.add(i);records.append(deepcopy(r))
        for r in p['source_evidence']:
            key=canonical(r)
            if key not in evidence_seen:evidence_seen.add(key);evidence.append(deepcopy(r))
    qualification_parents=[Path(options['--original-source-point-qualification-source'])]+a.qualification_parent
    qualifications=[read(p)for p in qualification_parents];qualified=deepcopy(qualifications[0]);qualified_records=[];qualified_seen=set()
    for p in qualifications:
        for field in ('schema','source_family','source_video_sha256','time_base','original_packet'):
            if p[field]!=qualified[field]:raise ValueError('Qualification parents differ: '+field)
        for r in p['records']:
            if r['original_packet_row_index']in qualified_seen:raise ValueError('Qualification parent rows overlap')
            qualified_seen.add(r['original_packet_row_index']);qualified_records.append(deepcopy(r))
    retired_count=sum(len(r['excluded_point_indices'])for r in qualified_records)
    qualified.update(records=qualified_records,derived_qualification_parents=[ref(p)for p in qualification_parents])
    qualificationpath=SYS/f'original_absolute_{retired_count}_pending_point_qualification_MERGED_20261010.json';write(qualificationpath,qualified)
    state,q,identity,camera_error=load_state(seed,Path(options['--world-init']),read(seed.parent.parent/'input_manifest.json'))
    assert camera_error==0.;identity['_accepted_directory']=str(seed);before_q=q.tobytes();qshort=identity['q_sha256'][:8]
    source=merged['original_absolute_packet'];assert ref(source['path'])==source
    original=pickle.loads(Path(source['path']).read_bytes())['programme_absolute_rows']
    # This source was already measured as a fine ordinary paint center. Keep
    # its three actual pixels, retire the independently ambiguous original
    # point, and remove the historical paint-width location band.
    for r in records:
        if r['original_packet_row_index']==5171:
            e=original[5171];state.geometry.register_row(e['source_row'],actual_pts=e['group_actual_pts'])
            _,binding=state.geometry._binding(e['source_row'])
            r.update(measurement_role='painted_line_center',original_geometry_identity=binding['contract']['physical_identity'],
                paint_footprint_is_not_location_uncertainty=True,remove_original_normal_interval=True,
                corrected_normal_interval_px=None,retired_original_point_indices=[0])
        if r.get('retired_original_point_indices'):
            r['source_point_qualification_reference']=ref(qualificationpath)
    points=sum(len(r['point_indices'])for r in records)
    merged.update(records=records,source_evidence=evidence,corrected_point_count=points,
                  derived_correction_parents=[ref(p)for p in parents],old66_measured_UV_sigma_exact=True,
                  ordinary_paint_center_semantics_explicit=True,multi_layer_target_ring_boundaries_unchanged=True,
                  no_world_radius_sigma_weight_or_user_truth_change=True)
    correctionpath=SYS/f'original_ice_{points}_boundary_and_painted_center_RGB_correction_MERGED_20261010.json';write(correctionpath,merged)
    # The actual source rows and original group positions are retained. No
    # motion arrays/actors or complete source graph are needed by this installer.
    for b in state.blocks:
        items=[e for e in original if e['block']==b.name and e['collection_role']=='FIT']
        groups=[None]*(max(e['group_order']for e in items)+1)
        for e in items:
            gi=e['group_order'];rows=[]if groups[gi]is None else groups[gi][1]
            assert e['row_order']==len(rows)
            rows.append(deepcopy(e['source_row']));groups[gi]=(e['group_actual_pts'],rows)
        b.groups=groups
    lookup={b.name:b for b in state.blocks}
    centers=[r for r in records if r.get('measurement_role')=='painted_line_center']
    for r in centers:
        e=original[r['original_packet_row_index']];state.geometry.register_row(e['source_row'],actual_pts=e['group_actual_pts'])
    contract=install_original_ice_pixel_correction(state,correctionpath)
    evaluators,authorities=load_original_absolute_evaluators()
    with np.load(identity['camera_path'],allow_pickle=False)as z:
        camera_by_pts={int(p):v.copy()for p,v in zip(z['pts'],z['camera25'])}
    checks=[]
    for r in centers:
        i=r['original_packet_row_index'];e=original[i];old=e['source_row'];new=lookup[e['block']].groups[e['group_order']][1][e['row_order']]
        for field in ('world','sigma','normal'):
            assert np.array_equal(old.get(field),new.get(field)),field
        assert all(new.get(k)==old.get(k)for k in ('point_correspondence','curve_observation_kind','scanline_branch','scanline_fixed_axis','extent_side'))
        assert 'normal_interval'not in new and'normal_interval_px'not in new
        ow=state.geometry.rebind(old,q[state.geometry_parameter_indices]);nw=state.geometry.rebind(new,q[state.geometry_parameter_indices])
        br,bj=evaluators[e['block']](camera_by_pts[e['group_actual_pts']],[ow],True)
        from calibration.solver_runtime.original_absolute_input_compaction import slice_observed_points
        retired=r.get('retired_original_point_indices',[]);keep=np.setdiff1d(np.arange(len(new['uv'])),retired)
        active=slice_observed_points(nw,keep)if retired else nw
        ar,aj=evaluators[e['block']](camera_by_pts[e['group_actual_pts']],[active],True)
        assert np.isfinite(ar).all()and np.isfinite(aj).all()and len(ar)==2*len(keep)
        checks.append(dict(original_packet_row_index=i,physical_identity=r['original_geometry_identity'],
            original_scalar_rows=len(br),before_normalized_residual=br.tolist(),after_normalized_residual=ar.tolist(),
            before_camera15_J_norm=float(np.linalg.norm(bj)),after_camera15_J_norm=float(np.linalg.norm(aj)),
            after_camera15_J_nonzero_values=int(np.count_nonzero(aj)),all_original_UV_sigma_world_and_roles_exact_except_authorized_centre_UV_and_band=True))
    oldboard=read(options['--additional-board-source']);newboard=read(a.board_packet)
    assert newboard['rows'][:len(oldboard['rows'])]==oldboard['rows']
    board_delta=sum(2*len(r['uv'])for r in newboard['rows'][len(oldboard['rows']):]if r['split']=='fit')
    oldconfirmed=read(original_confirmed_reference);confirmed=deepcopy(oldconfirmed);changes=[]
    for r in confirmed['records']:
        if r['family']not in ('native_board_edge','native_ice_curve'):continue
        oldref=r['packet_reference'];assert ref(oldref['path'])==oldref
        candidate=a.board_packet if r['family']=='native_board_edge' else a.ice_packet
        if candidate is None:continue
        oldpacket=read(oldref['path']);i=r['original_row_index'];actual=read(candidate)['rows'][i]
        assert canonical(oldpacket['rows'][i])==canonical(actual)
        assert hashlib.sha256(canonical(actual)).hexdigest()==r['source_row_canonical_json_sha256']
        assert all(np.array_equal(r[k],actual[k])for k in ('uv','sigma_px'))
        changes.append(dict(id=r['id'],family=r['family'],original_row_index=i,original_packet_reference=oldref,new_packet_reference=ref(candidate),original_row_and_uv_sigma_caps_and_identity_exact=True))
        r['packet_reference']=ref(candidate)
    confirmed.update(parent_authority_reference=ref(original_confirmed_reference),
                     current_board_packet_exact_append_binding=dict(packet=ref(a.board_packet),records=changes))
    # Complete parents retain their referenced immutable authority. A new
    # container binding must never overwrite that parent's source identity.
    confirmedpath=SYS/f'confirmed_same30_q{qshort}_ice{sha(a.ice_packet)[:8]if a.ice_packet else "unchanged"}_board{len(newboard["rows"])}_painted{points}_binding_20261010.json'
    if append_receipt is None:write(confirmedpath,confirmed)
    if append_receipt is not None:
        supplied=append_receipt['paths']['confirmed_authority']
        if ref(supplied['path'])!=supplied or canonical(read(supplied['path'])['records'])!=canonical(confirmed['records']):
            raise ValueError('Current-user append confirmed30 differs from exact original row/UV/sigma binding')
        confirmedpath=Path(supplied['path'])
    options['--additional-board-source']=str(a.board_packet);options['--original-ice-pixel-correction-source']=str(correctionpath)
    options['--user-confirmed-source-truth-authority']=str(confirmedpath)
    options['--original-source-point-qualification-source']=str(qualificationpath)
    if a.ice_packet:options['--additional-ice-source']=str(a.ice_packet)
    board=NativeBoardEdgeFactor(state,options['--additional-board-source'],len(q));ice=NativeIceCurveFactor(state,options['--additional-ice-source'],len(q))
    human=bind_human(state,board,ice,options,len(q));wide=wide_factor(state,identity,options,len(q),options['--practical-human-location-envelope'])
    old71=read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_user_review71_20261009.json')['source_options']
    circle,material,placement=existing_LOW_source_owners(state,q,seed,old71,read(seed/'scene_UNACCEPTED.json'))
    confirmed_factor=UserConfirmedSourceTruth(state,confirmedpath,board,ice,parameter_count=len(q),circle_factor=circle,material_factor=material,placement_source=placement)
    current=values(human,wide,confirmed_factor,q);parentcaps=read(seed/'truth_restoration_state.json')
    parent_groups=sum(len(v['group_ids'])for v in parentcaps['families'].values())
    if parent_groups not in (71,85):raise ValueError('This source transaction supports the actual old71 cold or current85 complete parent')
    caps_to_keep=parentcaps if parent_groups==85 else read(SYS/'current85_caps_values_bound_to_exact27fa_v1.json')
    if caps_to_keep.get('complete_q_sha256')!=identity['q_sha256']:raise ValueError('Current85 limits need their exact complete q')
    from run_current_named66_full_programme import truth_restoration_group_ids
    # Actual public factors are recomputed at this exact accepted complete q.
    families={}
    for family,factor in dict(native=human,wide=wide,confirmed=confirmed_factor).items():
        saved=caps_to_keep['families'][family]
        actual_ids=truth_restoration_group_ids(factor,family);n=len(saved['group_ids'])
        assert actual_ids[:n]==saved['group_ids'],family
        if family=='confirmed':uncertainty=np.asarray([item['radius'].min()for item in factor.records],float)
        else:
            policy=read(options['--practical-human-location-envelope'])
            uncertainty=np.asarray([r.get('engineering_uncorrelated_location_scale_px',r['normal_expansion_px'])for r in policy['records']if r['family']==family],float)
        assert np.array_equal(uncertainty[:n],saved['uncertainties']),family
        assert np.all(current['buffers'][family][:n]<=np.asarray(saved['limits'])+1e-8),family
        extra=current['buffers'][family][n:];extra_caps=np.where(extra<=1e-8,0.,extra+uncertainty[n:])
        families[family]=dict(old_count=len(parentcaps['families'][family]['group_ids']),group_ids=actual_ids,
            limits=list(saved['limits'])+extra_caps.tolist(),uncertainties=uncertainty.tolist(),violations_px=current['buffers'][family].tolist())
    group_count=sum(len(v)for v in current['buffers'].values());board_count=len(newboard['rows'])
    expected_groups=append_receipt['new_total_truth_groups']if append_receipt is not None else 85
    assert group_count==expected_groups and q.tobytes()==before_q
    proofpath=SYS/f'actual_q{qshort}_ice{points}_painted_center_board{board_count}_truth{group_count}_transaction_20261010.json'
    proof=dict(schema='rinkspace.actual-q27fa-painted-centre-and-board-source-transaction.v1',passed=True,
        complete_q_sha256=identity['q_sha256'],complete_state_reference=ref(seed/'complete_state_UNACCEPTED.npz'),
        correction_packet=ref(correctionpath),parents=[ref(p)for p in parents],corrected_source_points=points,
        actual_public_installer_passed=True,actual_ordinary_paint_source_R_camera15_J=checks,
        old66_measured_UV_sigma_preserved_exact=all(all(records[i][k]==r[k]for k in ('point_indices','corrected_uv','original_sigma'))for i,r in enumerate(packets[0]['records'])),
        board_packet=ref(a.board_packet),board_added_scalars=board_delta,old_board104_rows_canonical_exact=True,
        confirmed_authority=ref(confirmedpath),confirmed_board_packet_changes=changes,
        actual_all_groups_violations_at_exact_original_q=current['buffers'],all85_original_caps_uncertainties_and_groups_preserved=True,
        new_group_count=group_count,current_user_append_receipt=None if a.source_append_receipt is None else ref(a.source_append_receipt),
        actual_native_ice_active_scalar_rows=ice.row_count,actual_native_board_active_scalar_rows=board.row_count,
        original_qualification=ref(qualificationpath),actual_original_points_to_physically_retire=retired_count,
        original_density_compaction_source_unchanged=True,
        full_source_graph_or_solver_loaded=False,current_source_target_differs_from_previous_corrected66=True,
        previous_source_J_profiles_H_g_replay_authorized=False,source_evidence_accuracy_is_not_complete_fit_accuracy=True,
        seconds=time.perf_counter()-started);write(proofpath,proof)
    oldscope=read(options['--truth-restoration-reviewed-manual-append-audit'])
    oldmanifest=read(parentcaps['source_input_manifest_reference']['path'])
    if parent_groups==85:
        oldretired=oldmanifest['original_absolute_input_compaction']['removed_unqualified_absolute_points']
        delta=board.row_count-oldmanifest['additional_real_native_board_edges']['scalar_rows']-2*(retired_count-oldretired);direct_delta=0
        if append_receipt is not None:
            from structured_programme_source_candidate.human_stroke_truth import direct_manual_centre_factors
            actual_centres=direct_manual_centre_factors(state,human,options['--additional-board-source'],options['--additional-ice-source'],len(q))
            direct_delta=sum(f.row_count for f in actual_centres)-oldmanifest['source_contract']['direct_manual_centre_secondary']['added_scalar_rows']
            delta+=direct_delta
    else:
        origin_audit=read(oldscope['same_q_public_audit']['path'])
        delta=origin_audit['total_added_source_scalars']+board_delta-2*(retired_count-12)
        direct_delta=origin_audit['added_direct_manual_centre_scalars']
    manualkeys=('--additional-board-source','--additional-ice-source','--direct-human-truth-authority','--direct-human-full-stroke-guard','--direct-human-stroke-cap-support','--practical-human-location-envelope','--user-confirmed-source-truth-authority')
    newmanual={k:ref(options[k])for k in manualkeys}
    if a.ice_packet:newmanual['--additional-ice-source']=read(read(a.ice_packet)['source_only_review_qualification']['path'])['original_packet']
    audit=dict(schema='rinkspace.same-complete-q-current-user-source-and-exact-parent-caps-append-audit.v1',passed=True,
        complete_q_sha256=identity['q_sha256'],complete_state_reference=ref(seed/'complete_state_UNACCEPTED.npz'),
        old_caps_reference=ref(seed/'truth_restoration_state.json'),parent_manifest_reference=parentcaps['source_input_manifest_reference'],
        new_source_options=newmanual,added_direct_manual_centre_scalars=direct_delta,total_added_source_scalars=delta,
        new_complete_source_scalar_count=oldmanifest['source_rows']+delta,parent_group_count=parent_groups,new_group_count=group_count,families=families,
        current_source_target_transaction=ref(proofpath),existing85_caps_uncertainties_group_ids_exact=True,
        actual_public_same_q_factors_recomputed=True,old_source_J_profiles_or_H_g_not_reused=True)
    auditpath=SYS/f'same_q{qshort}_truth{group_count}_ice{points}_board{board_count}_current_target_startup_audit.json';write(auditpath,audit)
    capvalues=dict(schema='rinkspace.exact-parent-plus-current-source-cap-values.v1',complete_q_sha256=identity['q_sha256'],audit=ref(auditpath),families=families)
    valuespath=SYS/f'current{group_count}_ice{points}_board{board_count}_caps_bound_to_exact{qshort}.json';write(valuespath,capvalues)
    scope=deepcopy(oldscope)
    if parent_groups==85:scope.pop('protected_same_q_caps',None)
    scope['same_q_public_audit']=ref(auditpath);scope['exact_caps_values']=ref(valuespath)
    scope['admitted_operator_sources']['original_ice_pixel_correction']['reference']=ref(correctionpath)
    scope['admitted_operator_sources']['original_source_point_qualification']['reference']=ref(qualificationpath)
    if a.ice_packet:scope['source_qualified_native_packet_sources']['additional_real_native_ice_curves']=ref(a.ice_packet)
    scope['current_source_target_transaction']=ref(proofpath)
    scopepath=SYS/f'same_q{qshort}_truth{group_count}_ice{points}_board{board_count}_current_target_startup_binding.json';write(scopepath,scope)
    options['--truth-restoration-reviewed-manual-append-audit']=str(scopepath)
    options['--camera-seed']=str(seed/'complete_state_UNACCEPTED.npz');options['--truth-restoration-state']=str(seed/'truth_restoration_state.json')
    with np.load(seed/'complete_state_UNACCEPTED.npz',allow_pickle=False)as z:
        damping=float(z['damping'])if 'damping'in z else float(read(seed/'state_identity.json')['damping'])
        actual_checkpoint_state=dict(q_sha256=identity['q_sha256'],parameters=len(z['q']),damping=damping,
            fields=len(z.files),next_line_search_fraction=float(z['next_line_search_fraction'])if 'next_line_search_fraction'in z else None)
    options['--initial-damping']=damping
    config['continuation'].update(current85_caps_source=ref(valuespath),startup_binding=ref(scopepath),
        current_source_target_transaction=ref(proofpath),original_corrected_source_pixels=points,
        old66_corrected_source_pixels_preserved=True,Ad4_four_actual_RGB_edges_appended=True,
        complete_accepted_state=str(seed/'complete_state_UNACCEPTED.npz'),checkpoint_file_sha256=sha(seed/'complete_state_UNACCEPTED.npz'),
        truth_restoration_state=str(seed/'truth_restoration_state.json'),truth_restoration_state_sha256=sha(seed/'truth_restoration_state.json'),
        original_unqualified_absolute_points_physically_removed=retired_count,
        expected_complete_active_source_scalars=audit['new_complete_source_scalar_count'],
        next_start_requires_fresh_source_J=True,no_cached_J_H_g_or_prefix_imported=True,solver_run_started=False)
    config['source_manifest_sha256']=sha(ROOT/'calibration/solver_runtime/source_manifest.json')
    config['continuation'].update(current_truth_group_count=group_count,nearboard_physical_working_sigma_px=4.,nearboard_Huber_component_cutoff_px=8.,
        state=actual_checkpoint_state,current_caps_source=ref(valuespath),same_q_actual_source_audits=[ref(proofpath),ref(auditpath)],
        previous_input_manifest=parentcaps['source_input_manifest_reference'])
    configpath=a.output_config;write(configpath,config)
    print(json.dumps(dict(config=str(configpath),proof=str(proofpath),correction=str(correctionpath),points=points,
                         same_q85_prefix_exact=True,truth_group_count=group_count,source_rows=audit['new_complete_source_scalar_count'],seconds=proof['seconds'])),flush=True)

if __name__=='__main__':main()
