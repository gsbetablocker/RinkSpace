"""Continue the entire native-source programme on a revised common rink."""
from calibration.solver_runtime.config import data_root as _runtime_data_root, data_module_dir as _runtime_data_module_dir
from pathlib import Path
import os,sys,json,time,argparse,hashlib,cProfile,pstats,shutil,math
from dataclasses import asdict,replace
from contextlib import nullcontext
BASE=_runtime_data_root()
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ.update(RINKSPACE_MOTION_BACKEND='native',RINKSPACE_MOTION_THREADS='3',RINKSPACE_SCATTER_BACKEND='native',
    RINKSPACE_SCATTER_THREADS='3',RINKSPACE_PROJECTION_BACKEND='native',RINKSPACE_PROJECTION_VALUE_BACKEND='opencv',
    RINKSPACE_GEOMETRY_SPARSE_FD='1',RINKSPACE_GEOMETRY_FD_STEP_M='1e-7',RINKSPACE_PROFILE_MOTION_ONLY='1',
    RINKSPACE_PATTERN_WORKERS='1',RINKSPACE_SOLVER_ACCELERATION='auto')
from isolated_solver_bootstrap import install
install()
import numpy as np
from scipy.sparse import vstack
from run_shared_print_pairs_and_mount_full_source_candidate import prepare,DEFAULT_SEED,write,sha,qtoken
from advertisement_source_binding_candidate.corrected_advertisement_source_geometry import install_corrected_advertisement_source_geometry
from shared_ice_hockey_geometry_candidate.hockey_graph import install_shared_ice_hockey_geometry
from shared_ice_hockey_geometry_candidate.hockey_driver_capture import capture_hockey_driver
from shared_ice_hockey_geometry_candidate.hockey_source_chart import HockeyCoordinateChart
from shared_ice_hockey_geometry_candidate.hockey_wide_source import HockeyWideHighSourceFactor
from correlated_roll_prior_candidate.explicit_bias_source import ExplicitBiasSource
from wide_anchor_candidate.complete_wide_free_optics_source import CompleteWideSource
from structured_programme_source_candidate.paired_programme_blocks import install_structured_programme_blocks
from structured_programme_source_candidate.paired_programme_chart import CORE_COUNT,WORLD_INDICES,OLD,CompleteProgrammeParameterLayout,CanonicalCompleteProgrammeSource
from structured_programme_source_candidate.current_full_source import CurrentProgrammeSource
from structured_programme_source_candidate.current_wide_source import CurrentWideSource
from shared_unmeasured_advertisement_geometry_candidate.paired_locations_shared_sizes_geometry import PairedLocationsSharedSizesGeometry
from simple_wide_perspective_reset_candidate.audit_structured_current_axis_reference import make_structured_geometry
from z190_joint_source_solver import solve_joint_source,_frozen_profile_token,make_profile_update
from z190_streaming_source_normal import accumulate_source_normal,_block_cost
from calibration.solver_runtime.source_loss import source_loss_checkpoint_arrays
from calibration.solver_runtime.checkpoint_history import read_checkpoint_history, read_solver_history
from structured_programme_source_candidate.named66_soft_geometry_references import wrap_named66_soft_geometry_references
from structured_programme_source_candidate.runtime_backend_contract import restore_native_backend_contract,install_source_worker_backend_contract

# Windows actors import __main__ before their bootstrap. Restore the reviewed
# runtime after diagnostic imports, so child processes inherit it as well.
restore_native_backend_contract(stage='after_module_imports')
SOURCE_WORKER_BACKEND_INSTALLATION=install_source_worker_backend_contract()

TRACE=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_contours_20261004/edge1_2e2bd471877d/edge1_source_uv.json')
BASE_FULL_COUNT=CORE_COUNT+28

def circle_height_step_constraints(circle,q,selected):
    """Four existing circle height faces in the selected evaluation chart.

    These are feasible-step halfspaces, never observations. The original
    nonlinear guard and its strictly positive radius check remain unchanged.
    """
    q=np.asarray(q,float);selected=np.asarray(selected,int)
    cq=circle._course(q)
    _,bound,chain=circle.geometry.print_course_binding(q[circle.world_indices])
    chain=np.asarray(chain,float)
    if not np.array_equal(cq,bound)or chain.shape!=(6,len(circle.world_indices)):
        raise ValueError('Existing complete circle height chain required')
    if selected.ndim!=1 or len(np.unique(selected))!=len(selected):
        raise ValueError('Distinct selected evaluation coordinates required')
    mapping={int(column):position for position,column in enumerate(selected)}
    dh=np.zeros(len(selected))
    for column,coefficient in zip(circle.world_indices,chain[5]):
        if coefficient!=0.:
            if int(column)not in mapping:raise ValueError('Circle board height depends on an omitted shared column')
            dh[mapping[int(column)]]=coefficient
    rows=[];margins=[];bounds=[]
    for number,center_slot in ((11,1),(24,4)):
        material=circle._material(q,number);z=float(material[1]);r=float(material[2])
        zcol=int(circle.circle_indices[center_slot])
        rcol=int(circle.circle_indices[2 if circle.radius_shared or number==11 else 5])
        if zcol not in mapping or rcol not in mapping:
            raise ValueError('Circle center and radius must be selected shared columns')
        dz=np.zeros(len(selected));dr=dz.copy()
        dz[mapping[zcol]]=1.;dr[mapping[rcol]]=1.
        for face,row,margin in (('bottom',dz-dr,z-r),('top',dh-dz-dr,float(cq[5])-z-r)):
            rows.append(row);margins.append(margin)
            bounds.append(dict(ad_number=number,bound=face,margin_m=float(margin)))
    return dict(matrix=np.asarray(rows,float),lower=-np.asarray(margins,float),
        report=dict(kind='existing_complete_printed_circle_height_domains',bounds=bounds,
            radius_shared=bool(circle.radius_shared),strict_positive_radius_guard_unchanged=True,
            nonlinear_trial_guard_retained=True,new_source_or_prior_rows=False,
            new_minimum_radius_or_height_tolerance=False))

def role_target_from_restored_manifest(manifest):
    from structured_programme_source_candidate.truth_source_role_migration import role_target
    return role_target(manifest)

def truth_restoration_source_target(manifest):
    raw=json.dumps(manifest['source_contract'],sort_keys=True,separators=(',',':'),allow_nan=False,
        default=lambda v:v.tolist()if isinstance(v,np.ndarray)else v.item()if isinstance(v,np.generic)else v)
    return dict(parameter_count=int(manifest['full_parameters']),source_rows=int(manifest['source_rows']),
        source_contract_sha256=hashlib.sha256(raw.encode('utf8')).hexdigest())

def truth_restoration_group_ids(factor,family):
    return [r['record']['id']if family=='wide'else r['row']['id']for r in factor.records]

def authenticate_truth_source_append(proof_path,state_path,packet,original,manifest):
    """Authenticate only the reviewed Ad26 material append; preserve every cap."""
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    proof_path=Path(proof_path).resolve();proof=json.loads(proof_path.read_text(encoding='utf8'))
    if proof.get('schema')!='rinkspace.authenticated-Ad26-material-append-truth-proof.v1':
        raise ValueError('Explicit reviewed material append truth proof required')
    def reference(ref):
        path=Path(ref['path']).resolve()
        if sha(path)!=ref['sha256']:raise ValueError('Source append proof reference bytes changed')
        return path
    if reference(proof['original_truth_state'])!=Path(state_path).resolve():
        raise ValueError('Source append proof belongs to another complete truth state')
    if (proof['complete_q_sha256']!=packet['complete_q_sha256']
        or proof['existing_manual_group_count']!=sum(len(r['group_ids'])for r in packet['families'].values())):
        raise ValueError('Source append proof complete q/group inventory differs')
    if proof['original_source_target']!=truth_restoration_source_target(original):
        raise ValueError('Source append proof original source target differs')
    old_ref=proof['original_material_packet'];new_ref=proof['appended_material_packet']
    old_path=reference(old_ref);new_path=reference(new_ref)
    old=json.loads(old_path.read_text(encoding='utf8'));new=json.loads(new_path.read_text(encoding='utf8'))
    admission_path=reference(proof['strict_append_admission_proof'])
    admission=json.loads(admission_path.read_text(encoding='utf8'))
    if (admission.get('schema')!='rinkspace.strict-Ad26-pure-additive-source-admission.v1'
        or admission['parent_packet']!=old_ref or admission['admitted_packet']!=new_ref):
        raise ValueError('Reviewed material append admission identity differs')
    n=len(old['rows']);extra=new['rows'][n:]
    if (n!=11 or len(extra)!=2 or canonical(new['rows'][:n])!=canonical(old['rows'])
        or any(new.get(key)!=value for key,value in old.items()if key not in('rows','derived_from_native_source_parent'))
        or new.get('derived_from_native_source_parent')!=old_ref):
        raise ValueError('Material append changed an original source field or row')
    expected_rows=proof['appended_row_inventory']
    observed=[dict(id=r['id'],pts=r['pts'],actual_pts=r['actual_pts'],split=r['split'],points=len(r['uv']))for r in extra]
    if observed!=expected_rows:raise ValueError('Reviewed append row inventory differs')
    if sorted((r['split'],len(r['uv']))for r in extra)!=[('fit',10),('holdout',8)]:
        raise ValueError('Only the reviewed 10-point FIT and 8-point check append is authorized')
    for row in extra:
        if (row['curve']!=dict(kind='shared_board_pattern_circle_curve',advertisement_id=26,edge='outermost_visible_circular_print_stroke')
            or row.get('source_RGB_actual_measurement')is not True
            or row.get('model_projection_used_to_generate_or_select_uv')is not False
            or row.get('source_optical_normal_support_authorized')is not False
            or row.get('solver_admitted')is not True or row.get('world')is not None
            or not np.array_equal(np.asarray(row['sigma_px'],float),np.full(len(row['uv']),6.))
            or sha(row['source_raw_frame'])!=row['source_file_sha256']):
            raise ValueError('Reviewed Ad26 source observation semantics differ')
    delta=sum(2*len(r['uv'])for r in extra if r['split']=='fit')
    if delta!=20 or proof['added_FIT_scalar_rows']!=delta:
        raise ValueError('Only the reviewed 20 source scalar rows may be added')
    old_factor=original['additional_real_native_board_material_points']
    new_factor=manifest['additional_real_native_board_material_points']
    if (Path(old_factor['packet_path']).resolve()!=old_path or old_factor['packet_sha256']!=old_ref['sha256']
        or Path(new_factor['packet_path']).resolve()!=new_path or new_factor['packet_sha256']!=new_ref['sha256']):
        raise ValueError('Actual loaded material source differs from the append proof')
    fit=[r for r in new['rows']if r['split']=='fit'];held=[r for r in new['rows']if r['split']=='holdout']
    expected_factor=json.loads(canonical(old_factor))
    expected_factor.update(packet_path=str(new_path),packet_sha256=new_ref['sha256'],
        fit_ids=[r['id']for r in fit],held_ids=[r['id']for r in held],
        scalar_rows=old_factor['scalar_rows']+delta,
        native_circle_curve_ids=[r['id']for r in fit+held if r['curve']['kind']=='shared_board_pattern_circle_curve'])
    if canonical(new_factor)!=canonical(expected_factor):
        raise ValueError('Loaded material factor changed more than the reviewed append')
    counters={
        'rinkspace.original-source-canonical-complete-parameter-layout.v1':('source_rows',),
        'rinkspace.complete-source-plus-native-board-tracks.v1':('original_source_rows','total_rows'),
        'rinkspace.static-motion-source-role-retirement.v1.complete-source':('total_rows',),
        'rinkspace.complete-source-plus-real-persistent-ice-long-baseline.v1':('original_source_rows','total_rows'),
        'rinkspace.complete-source-with-independent-pattern-placements.v1':('source_rows',),
        'rinkspace.source-proven-original-print-role-retirement.v1.complete-source':('total_rows',),
        'rinkspace.complete-source-plus-six-slot-five-active-shared-radius-board-circle-materials.v1':('original_source_rows','total_rows'),
        'rinkspace.complete-source-plus-real-native-board-material-points.v1':('additional_scalar_rows','total_rows'),
        'rinkspace.shared-print-template-independent-physical-placements.v1':('original_source_scalar_rows','source_scalar_rows')}
    replacements=[]
    def expected_contract(value,path='source_contract'):
        if isinstance(value,list):return [expected_contract(v,path+'/'+str(i))for i,v in enumerate(value)]
        if not isinstance(value,dict):return value
        if value.get('schema')=='rinkspace.additional-real-native-board-material-point-factor.v1':
            if canonical(value)!=canonical(old_factor):raise ValueError('Original material contract differs')
            replacements.append(path);return expected_factor
        result={key:expected_contract(v,path+'/'+key)for key,v in value.items()}
        for key in counters.get(value.get('schema'),()):
            if key not in value or type(value[key])is not int:raise ValueError('Source append counter inventory differs')
            result[key]=value[key]+delta
        return result
    expected=expected_contract(original['source_contract'])
    if len(replacements)!=1 or canonical(manifest['source_contract'])!=canonical(expected):
        raise ValueError('Original source operators, masks, scales or contracts changed outside the material append')
    if (manifest['full_parameters']!=original['full_parameters']
        or manifest['source_rows']!=original['source_rows']+delta):
        raise ValueError('Material append changed parameter or source row inventory')
    for key in ('highest_direct_human_hard_truth','highest_user_confirmed_native_sources'):
        if canonical(manifest[key])!=canonical(original[key]):
            raise ValueError('Manual authority, complete UV/support or confirmed source changed: '+key)
    return dict(proof=dict(path=str(proof_path),sha256=sha(proof_path)),
        original_truth_state=proof['original_truth_state'],original_source_target=packet['source_target'],
        actual_current_source_target=truth_restoration_source_target(manifest),added_FIT_scalar_rows=delta,
        original_source_contract_retained_except_exact_material_append=True,
        original_complete_state_bytes_and_q_identity_retained=True,
        all_manual_UV_support_authorities_and_group_caps_retained=True,
        group_count=sum(len(r['group_ids'])for r in packet['families'].values()),
        original_truth_packet_not_rewritten=True,old_source_J_or_profile_cache_not_authorized=True)


def load_truth_restoration_state(path,q,manifest,factors,*,source_append_proof=None,source_role_proof=None,source_revision_proof=None,source_location_revision_proof=None,reviewed_soft_source_proof=None,wide_sampling_loss_runtime=None):
    packet=json.loads(Path(path).read_text(encoding='utf8'))
    if packet.get('schema')!='rinkspace.joint-source-truth-restoration-state.v1':
        raise ValueError('Explicit complete truth restoration state required')
    reference=packet['complete_state_reference'];source=packet['source_input_manifest_reference']
    if sha(reference['path'])!=reference['sha256']or sha(source['path'])!=source['sha256']:
        raise ValueError('Truth restoration state/source reference bytes changed')
    with np.load(reference['path'],allow_pickle=False)as arrays:
        if not np.array_equal(arrays['q'],q):raise ValueError('Truth restoration complete seed differs')
    if packet['complete_q_sha256']!=qtoken(q):raise ValueError('Truth restoration seed q identity differs')
    target=truth_restoration_source_target(manifest)
    original=json.loads(Path(source['path']).read_text(encoding='utf8'))
    original_target=truth_restoration_source_target(original)
    if packet['source_target']!=original_target:
        raise ValueError('Truth restoration original source target differs')
    from structured_programme_source_candidate.truth_source_role_migration import (enforce_role_change_opt_in,authenticate_role_correction,authenticate_reviewed_source_revision,role_target)
    if packet.get('source_role_operator_target',role_target(original))!=role_target(original):
        raise ValueError('Original saved source role operator target differs')
    if source_location_revision_proof is not None and any(proof is not None for proof in (source_append_proof,source_role_proof,source_revision_proof)):
        raise ValueError('Measured boundary location revision cannot be combined with another source migration')
    role_changed=enforce_role_change_opt_in(original,manifest,source_role_proof if source_role_proof is not None else source_revision_proof)
    if source_revision_proof is not None and source_role_proof is not None:raise ValueError('Only one exact source migration proof may be supplied')
    role_migration=None;revision_migration=None
    if role_changed:
        if source_append_proof is not None:raise ValueError('Role correction cannot be combined with another source append migration')
        from structured_programme_source_candidate import original_motion_far_board_roles as role_api
        if source_revision_proof is not None:
            from structured_programme_source_candidate import original_motion_input_compaction as compaction_api
            revision_migration=authenticate_reviewed_source_revision(source_revision_proof,path,q,manifest,role_api.OriginalMotionFarBoardRoleBinding,Path(role_api.__file__),Path(compaction_api.__file__))
        else:role_migration=authenticate_role_correction(source_role_proof,path,q,manifest,role_api.OriginalMotionFarBoardRoleBinding,Path(role_api.__file__))
    migration=None;location_migration=None;loss_migration=None;soft_migration=None
    if reviewed_soft_source_proof is not None:
        if any(proof is not None for proof in (source_append_proof,source_role_proof,source_revision_proof,source_location_revision_proof)):
            raise ValueError('Only one reviewed soft-source continuation proof may be supplied')
        from calibration.solver_runtime.reviewed_source_continuation import authenticate_reviewed_source_revision
        authenticated=authenticate_reviewed_source_revision(reviewed_soft_source_proof,path,q,manifest,factors)
        packet=authenticated['packet'];soft_migration=authenticated['migration']
    elif original_target!=target:
        if source_location_revision_proof is not None:
            from calibration.solver_runtime.truth_source_location_revision import authenticate_location_revision
            authenticated=authenticate_location_revision(source_location_revision_proof,path,q,manifest,factors)
            packet=authenticated['packet'];location_migration=authenticated['migration']
        elif wide_sampling_loss_runtime is not None and all(proof is None for proof in (source_append_proof,source_role_proof,source_revision_proof)):
            from calibration.solver_runtime.original_wide_curve_integration import authenticate_wide_sampling_truth_continuation
            loss_migration=authenticate_wide_sampling_truth_continuation(original,manifest,wide_sampling_loss_runtime)
            loss_migration.update(original_source_target=original_target,actual_current_source_target=target)
        elif revision_migration is None:
            if source_append_proof is None:raise ValueError('Truth restoration mathematical source target differs')
            migration=authenticate_truth_source_append(source_append_proof,path,packet,original,manifest)
    elif source_append_proof is not None or source_location_revision_proof is not None:
        raise ValueError('Source migration proof requires an actual mathematical target change')
    expected={family:factor for family,factor in factors.items()if factor is not None}
    if set(packet['families'])!=set(expected):raise ValueError('Truth restoration family inventory differs')
    for family,factor in expected.items():
        record=packet['families'][family];authority=factor.contract['authority'];saved=record['authority']
        if Path(saved['path']).resolve()!=Path(authority['path']).resolve()or saved['sha256']!=authority['sha256']or sha(saved['path'])!=saved['sha256']:
            raise ValueError('Truth restoration source authority differs: '+family)
        ids=truth_restoration_group_ids(factor,family)
        if record['group_ids']!=ids:raise ValueError('Truth restoration group identity/order differs: '+family)
        for key in ('limits','uncertainties'):
            values=np.asarray(record[key],float)
            if values.shape!=(len(ids),)or not np.isfinite(values).all()or np.any(values<0.):
                raise ValueError('Finite complete truth restoration limits required: '+family)
    if migration is not None:
        packet=dict(packet,authenticated_additive_source_migration=migration)
    if role_migration is not None:
        packet=dict(packet,authenticated_original_motion_role_migration=role_migration)
    if revision_migration is not None:
        packet=dict(packet,authenticated_reviewed_source_revision=revision_migration)
    if location_migration is not None:
        packet=dict(packet,authenticated_source_location_revision=location_migration)
    if loss_migration is not None:
        packet=dict(packet,authenticated_sampling_loss_continuation=loss_migration)
    if soft_migration is not None:
        packet=dict(packet,authenticated_reviewed_soft_source_revision=soft_migration)
    return packet

def truth_restoration_state_packet(value,stage,manifest_path,manifest,factors,limits,uncertainties,*,progress_policy='historical_ratchet',seed_envelopes=None,seed_engineering_scale_source=None):
    from structured_programme_source_candidate.truth_source_role_migration import role_target
    return dict(schema='rinkspace.joint-source-truth-restoration-state.v1',complete_q_sha256=qtoken(value),
        complete_state_reference=dict(path=str(stage/'complete_state_UNACCEPTED.npz'),sha256=sha(stage/'complete_state_UNACCEPTED.npz')),
        source_input_manifest_reference=dict(path=str(manifest_path),sha256=sha(manifest_path)),
        source_target=truth_restoration_source_target(manifest),source_role_operator_target=role_target(manifest),
        families={family:dict(authority=factor.contract['authority'],group_ids=truth_restoration_group_ids(factor,family),
            limits=limits[family].tolist(),uncertainties=uncertainties[family].tolist())for family,factor in factors.items()if factor is not None},
        progress_policy=progress_policy,fixed_seed_envelopes=seed_envelopes if progress_policy=='fixed_seed_envelope'else None,
        seed_engineering_scale_source=seed_engineering_scale_source,
        historical_group_minimum_limits_preserved=progress_policy=='historical_ratchet',
        source_rows_uncertainties_and_nonlinear_guards_unchanged=progress_policy=='historical_ratchet',
        source_rows_uncertainties_and_nonlinear_support_operators_unchanged=True)

def component_state(factor,value):
    """Old factors retain their exact prefix when a material suffix is added."""
    value=np.asarray(value,float);count=int(factor.parameter_count)
    layout=getattr(factor,'canonical_parameter_layout',None)
    if layout is not None and len(value)==layout.count:value=layout.evaluation_q(value)
    if value.ndim!=1 or len(value)<count:raise ValueError('Factor state prefix is absent')
    return value[:count]

def current_seed_parts(seedq,*,circle_enabled=False):
    """Map current full checkpoints to old camera chart without losing suffix."""
    seedq=np.asarray(seedq,float)
    if seedq.shape not in ((BASE_FULL_COUNT,),(BASE_FULL_COUNT+6,)):
        return seedq,None,None
    suffix=seedq[BASE_FULL_COUNT:].copy()
    if len(suffix) and not circle_enabled:
        raise ValueError('Six-circle-column seed requires the same explicit board-circle source packet')
    world=seedq[WORLD_INDICES].copy()
    legacy=np.r_[seedq[:OLD.core_count],seedq[CORE_COUNT:BASE_FULL_COUNT]]
    return legacy,world,suffix if len(suffix) else None


def actual_source_replay_inputs(state,source,wide_factor):
    """Stable live source inputs; current planes/world/J are derived from q."""
    arrays={};records={}
    for block in state.blocks:
        bundle=block.bundle;prefix='programme:'+block.name+':'
        for field in ('a','b','ia','ib','stamps','motion_sigma','finite_mask','indices','pointer','swap_rows','optimized_local_columns'):
            arrays[prefix+field]=np.asarray(getattr(bundle,field))
        board=np.asarray(block.course_board_indices,int)
        arrays[prefix+'course_board_indices']=board
        arrays[prefix+'course_board_record_positions']=np.asarray(block.course_board_record_positions)
        noncourse=np.flatnonzero(bundle.finite_mask & ~np.isin(np.arange(len(bundle.a)),board))
        arrays[prefix+'static_noncourse_finite_point_indices']=noncourse
        arrays[prefix+'static_noncourse_finite_planes']=bundle.finite_planes[noncourse]
        original_records=block.course_original_board_records
        old=np.asarray([i for i,r in enumerate(original_records)if 'actual_source_record'not in r],int)
        arrays[prefix+'original_static_board_plane_positions']=old
        arrays[prefix+'original_static_board_planes']=block.course_original_board_planes[old]
        stable_board=[]
        for record in original_records:
            if 'actual_source_record'not in record:stable_board.append(record)
            else:
                stable_board.append({key:record[key]for key in ('global_bundle_index','surface_kind',
                    'physical_surface_identity','source_role_binding_id','source_role_block','source_native_pair_pts',
                    'original_source_role','material_identity','actual_source_record')})
        nonboard_records=[r for r in bundle.finite_records if r.get('surface_kind')!='finite_board_face']
        records[prefix+'metadata']=dict(source_clock_index=bundle.index,absolute_original_groups=block.groups,
            original_static_nonboard_finite_records=nonboard_records,original_stable_board_records=stable_board,
            current_board_planes_extents_facets_are_q_derived_not_source_input=True)
    records['actual_cross_source_fit']=state.cross_fit
    records['original_wide_input']=dict(observations=wide_factor.observations,advertisement_packet=wide_factor.ad_packet,
        entity_support=wide_factor.entity_support,trace_packet=wide_factor.trace_packet)
    for name in ('native_ice_factor','native_board_factor','native_board_material_point_factor','circle_factor',
                 'native_board_track_factor','persistent_ice_factor','reference_factor'):
        factor=getattr(source,name,None)
        if factor is not None:
            item={key:getattr(factor,key)for key in ('rows','held','records','packet','contract')if hasattr(factor,key)}
            if item:records['actual_factor:'+name]=item
    records['actual_exposure_aliases']=state.original.aliases
    return arrays,records

def check_current_board_edge_exports(edge,value,scene):
    """Verify every admitted edge against the actual exported finite material."""
    q=component_state(edge,value);checked=[];seen=set()
    keys={'left_vertical':'left_vertical_xyz','right_vertical':'right_vertical_xyz',
          'top_edge':'top_edge_xyz','bottom_edge':'bottom_edge_xyz'}
    for row in edge.rows+edge.held:
        curve=row['curve']
        if curve.get('kind')in('ice_board_contact','board_top'):
            from rinkspace.vision_camera_initialization import frozen_scene_curve
            identity=(curve['kind'],curve.get('source_structure'),curve.get('course_primitive'),curve.get('physical_boundary'))
            if identity in seen:continue
            seen.add(identity)
            exported=frozen_scene_curve(scene,edge.effective_curve(curve));actual=edge._world(q,curve)
            if actual.shape!=exported.shape:raise ValueError('Current physical course facet shape differs from scene export')
            difference=float(np.max(abs(actual-exported)))
            if difference>5e-14:raise ValueError('Actual top/contact course facet and exported finite physical boundary differ')
            checked.append(dict(source_structure=curve['source_structure'],edge=curve['kind'],
                course_primitive=curve.get('course_primitive'),physical_boundary=curve.get('physical_boundary'),
                world_shape=list(actual.shape),export_world_max_difference=difference,
                current_actual_course_material_producer_verified=True))
            continue
        if curve.get('kind')=='printed_component_contour':
            number=int(curve['advertisement_number'])
            identity=(number,curve['template_id'],curve['component_id'])
            if identity in seen:continue
            seen.add(identity)
            ads=[a for a in scene['advertisements']if int(a['user_number'])==number]
            if len(ads)!=1 or ads[0].get('source_template_id')!=curve['template_id']:
                raise ValueError('Actual exported printed template identity missing or different')
            contours=[c for c in ads[0].get('external_contours',[])if c['id']==curve['component_id']]
            if len(contours)!=1:raise ValueError('Actual exported printed component missing or duplicated')
            exported=np.asarray(contours[0]['world_xyz_scene_units'],float)
            actual=edge._world(q,curve)
            if actual.shape!=exported.shape:raise ValueError('Actual printed contour shape differs from scene export')
            difference=float(np.max(abs(actual-exported)))
            if difference>5e-14:raise ValueError('Actual printed contour and exported material differ')
            material_uv=edge._printed_contour_material(curve)
            material=edge.geometry.source_template_material(number,*material_uv.T,q[edge.world_indices])[0]
            if not np.array_equal(actual,material):raise ValueError('Printed contour bypassed its certified material identity')
            checked.append(dict(advertisement_number=number,curve_kind=curve['kind'],
                source_template_id=curve['template_id'],component_id=curve['component_id'],
                world_shape=list(actual.shape),export_world_max_difference=difference,
                actual_material_parameterization_byte_exact=True))
            continue
        number=int(curve['advertisement_number']);key=keys[curve['edge']]
        identity=(number,key)
        if identity in seen:continue
        seen.add(identity)
        ads=[a for a in scene['advertisements']if int(a['user_number'])==number]
        if len(ads)!=1:raise ValueError('Actual exported advertisement identity missing or duplicated')
        exported=np.asarray(ads[0]['fitted_boundary_curves_xyz'][key],float)
        actual=edge._world(q,curve)
        if actual.shape!=exported.shape:raise ValueError('Actual finite print edge shape differs from export')
        difference=float(np.max(abs(actual-exported)))
        if difference>5e-14:raise ValueError('Actual board factor and exported print material differ')
        u,v=edge.geometry._materials[identity]
        material=edge.geometry.material(number,u,v,q[edge.world_indices])[0]
        if not np.array_equal(actual,material):raise ValueError('Board edge bypassed its actual material identity')
        checked.append(dict(advertisement_number=number,edge=curve['edge'],material_edge_key=key,
            world_shape=list(actual.shape),export_world_max_difference=difference,
            actual_material_parameterization_byte_exact=True))
    return checked

def export_camera(state,core,path):
    from z190_physical_camera import PhysicalCameraModel,CanonicalView
    from z190_continuous_source_bundle import SOURCE_SHA
    calibrations=dict(state.original.model.calibrations);views=dict(state.original.views)
    for b in state.blocks:
        local=b.local(core);cal=b.bundle.current_calibration(local)
        mount_contract=getattr(state,'current_user_mount_constraint_contract',None)
        if mount_contract is not None:
            cal=replace(cal,field_settings=dict(cal.field_settings,current_user_mount_working_constraints=mount_contract))
        pivot_contract=getattr(state,'current_user_pivot_outside_constraint_contract',None)
        if pivot_contract is not None:
            cal=replace(cal,field_settings=dict(cal.field_settings,
                current_user_pivot_outside_constraints=pivot_contract,
                physical_source_core_q_sha256=qtoken(core)))
        calibrations[cal.installation_id]=cal
        for p in b.bundle.stamps:views[p]=CanonicalView(cal.installation_id,b.bundle.view(local,p))
    # Canonical exporter intentionally creates immutable files exclusively.
    # Repeated callbacks for the same state reuse its authenticated archive;
    # a new coupled state receives a new artifact instead of overwriting it.
    path=path.with_name(path.stem+'_'+qtoken(core)[:16]+path.suffix)
    if path.exists():
        with np.load(path,allow_pickle=False)as z:
            actual=z['camera25'];pts=z['pts']
        model=PhysicalCameraModel(calibrations)
        expected=np.asarray([model.camera25(views[state.original.aliases[int(p)]])for p in pts])
        if not np.array_equal(actual,expected):raise ValueError('Existing immutable camera artifact differs from exact same core')
    else:PhysicalCameraModel(calibrations).save_npz(path,state.original.native,state.original.aliases,views,source_sha256=SOURCE_SHA,time_base='1/15360')
    return path

def main():
    core_count=CORE_COUNT;world_indices=WORLD_INDICES.copy()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--world-init',type=Path);parser.add_argument('--camera-seed',type=Path)
    parser.add_argument('--strict-source-fit-dataset',type=Path,
        help='Authenticated finite-source dataset; replace every legacy pixel objective with admitted FIT rows')
    parser.add_argument('--strict-motion-inputs',type=Path,
        help='Authenticated strict ice/board paired-motion admission manifest')
    parser.add_argument('--camera-mount-constraint-source',type=Path,
        help='User-confirmed fixed-tripod setup and adopted camera-local lever working ranges')
    parser.add_argument('--reinitialize-invalid-mounts',action='store_true',
        help='Explicitly restore an out-of-range lever to the recorded user initializer and preserve its reference optical centre')
    parser.add_argument('--camera-pivot-exterior-constraint-source',type=Path,
        help='User-confirmed installation pivots outside the actual common rink boundary')
    parser.add_argument('--reinitialize-invalid-pivot-positions',action='store_true',
        help='Explicitly initialize an invalid declared pivot XY outside the current rink; preserve its Z, lever and camera RF states')
    parser.add_argument('--initial-pose-feasibility-source',type=Path,
        help='Exact-state source-domain feasible initializer for a declared nonreference actual camera pose; validated by the original source operator')
    parser.add_argument('--absolute-camera-initializer-source',type=Path,
        help='Explicit same-source nonreference Rodrigues/log-focal initializer; preserve shared hardware/world and recompute the complete source normal')
    parser.add_argument('--reinitialize-invalid-qualified-source-poses',action='store_true',
        help='Restore all invalid original finite-source ray domains after a shared-world edit with initialization-only RF4; preserve current direct manual exposures and every source fact')
    parser.add_argument('--roll-prior-sigma-degrees',type=float,default=.5,
        help='Explicit unmeasured engineering roll/installation-bias prior sigma; default preserves historical 0.5 degrees')
    parser.add_argument('--principal-point-working-prior',action='store_true',help='Opt-in existing shared PP6 engineering identifiability candidate: native [960,540], sigma32 pixels once per installation; not user hard truth or a measured calibration')
    parser.add_argument('--additional-native-board-track-source',type=Path,
        help='Reviewed same-static-material paired native board pixels on the shared current finite course')
    parser.add_argument('--estimate-unverified-physical-geometry',action='store_true',
        help='Estimate previously provisional print locations, frozen source print dimensions/locations and independent Lane right extents; retain canonical old complete state prefix')
    parser.add_argument('--derive-hockey-center-from-course',action='store_true',help='Confirmed C0/current course center relation; retain legacy center initializer slots in saved states, derive actual center for every source and export')
    parser.add_argument('--shared-hockey-curling-axis',action='store_true',help='User-confirmed Lane long edges perpendicular to C/B1/B2; derive the hockey axis from the current common curling axis while retaining the historical yaw slot in complete states')
    parser.add_argument('--strict-half-circle-crease',action='store_true',help='Explicit user-confirmed half-circle blue crease with current free goal frame and shared radius; historical snapshots without this flag retain their previous inferred unit shape')
    parser.add_argument('--estimate-unverified-goal-crease-placement',action='store_true',help='Estimate each blue crease center independently along its current goal baseline; preserve shared radius and all goal/trapezoid parameters')
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--save-initial-linearization-cache',action='store_true',
        help='Explicit diagnostic export of one initial r/J/normal cache; default saves checkpoints only, never per-iteration matrices')
    parser.add_argument('--accepted-steps',type=int,default=0,help='Optional intermediate review batch: freeze after this many new accepted coupled steps; default 0 continues to the solver stop condition')
    parser.add_argument('--serial-source',action='store_true',help='Diagnostic comparison only; default uses a memory-bounded resident source actor')
    parser.add_argument('--serial-current-patterns',action='store_true',help='Diagnostic comparison only; default uses the verified private current-geometry frame pool')
    parser.add_argument('--original-motion-low-compaction-source',type=Path,help='Authenticated source-clock/self/explicit-retirement input map')
    parser.add_argument('--original-motion-main-compaction-source',type=Path,help='Authenticated exact same-pair UV/operator/sigma duplicate input maps')
    parser.add_argument('--geometry-fd-workers',type=int,default=1,help='Light private absolute-material FD workers; one parent source graph and original source operators')
    parser.add_argument('--normal-workers',type=int,default=1,help='Threads for unchanged weighted CSR row Gram accumulation')
    parser.add_argument('--residual-source-workers',type=int,default=0,help='Six original persistent source actors for real residual/profile evaluations; fresh Jacobians use the parent FD pool')
    parser.add_argument('--residual-native-threads',type=int,default=3,help='Native motion threads per resident residual actor')
    parser.add_argument('--conditional-view-backtracking',action='store_true',help='Opt-in exact conditional original-system view backsolve while shortening declared shared steps; unchanged full source cost and all truth/physical guards; fallback to original whole-step search')
    parser.add_argument('--schur-view-lm-ratio',type=float,choices=(0.,1.),default=1.,
        help='Numerical Schur view-only LM ratio: 1 preserves the existing path; 0 uses the existing undamped view elimination. Original source H/g, objective and guards are unchanged')
    parser.add_argument('--bounded-motion-nuisance',action='store_true',
        help='Retain the original unknown-depth lambda lower bound in each numerical camera QP; original source points, profiles, costs and physical guards are unchanged')
    parser.add_argument('--save-complete-source-directions',action='store_true',help='Persist the actual complete canonical source step and q/profile/layout identity before endpoint scheduling; no additional J/source evaluation')
    parser.add_argument('--reuse-source-jacobian',action='store_true',help='Retain one source J in RAM for modified-Newton steps with fresh residuals/profiles/Huber H/g and real full-source trial cost; refresh on poor model agreement')
    parser.add_argument('--source-jacobian-refresh-ratio',type=float,default=0.,help='Opt-in existing modified-Newton refresh threshold: regenerate the admitted-source J after accepted actual/predicted reduction falls below this value; objective and guards unchanged')
    parser.add_argument('--adaptive-source-backtracking',action='store_true',help='Reuse the learned successful whole-source step fraction; save it in the complete checkpoint')
    parser.add_argument('--complete-scene-material-diagnostic',action='store_true',help='Diagnostic comparison only; default reuses exact bounded ice-material producer views')
    parser.add_argument('--additional-ice-curves','--additional-ice-source',type=Path,help='Explicit additional real native-image FIT/HELD packet')
    parser.add_argument('--additional-board-source',type=Path,help='Explicit admitted native board-edge source packet, appended after the unchanged ice target')
    parser.add_argument('--additional-board-material-point-source',type=Path,help='Explicit admitted printed shield-material point source on existing q9 geometry')
    parser.add_argument('--additional-board-circle-source',type=Path,help='Explicit admitted independent Ad11/Ad24 circle material source, with six appended material parameters')
    parser.add_argument('--truth-restoration-state',type=Path,help='Authenticated complete same-target truth-group state; historical ratchet is the default')
    parser.add_argument('--truth-restoration-progress-policy',choices=('historical_ratchet','fixed_seed_envelope'),default='historical_ratchet',
        help='Opt-in fixed seed violation plus existing declared engineering uncertainty for unresolved groups; source-zero groups stay protected, full source cost must decrease, no historical minimum ratchet')
    parser.add_argument('--truth-restoration-seed-engineering-scale-source',type=Path,
        help='Explicit authenticated existing positional engineering scales for a fixed-seed policy; never measured precision or a FIT sigma/support edit')
    parser.add_argument('--truth-restoration-source-revision-proof',type=Path,help='Exact authenticated reviewed material anchor append plus main1/main4 unknown-to-finite source roles; preserve all complete seed/cap bytes and require fresh source derivatives')
    parser.add_argument('--truth-restoration-source-location-revision-proof',type=Path,help='Authenticate measured RGB boundary location metadata at the same complete seed; preserve every source row, support and truth cap, and require fresh source derivatives')
    parser.add_argument('--truth-restoration-reviewed-soft-source-proof',type=Path,help='Authenticate reviewed soft source UV/observability changes at the saved complete seed; retain hard observations, original limits/envelopes and fresh initial derivatives')
    parser.add_argument('--truth-restoration-source-role-proof',type=Path,help='Explicit authenticated source-only UNKNOWN_DEPTH to existing finite-board role correction; retain every original complete seed/cap and require fresh source J')
    parser.add_argument('--truth-restoration-source-append-proof',type=Path,help='Explicit authenticated reviewed material append: retain original complete seed and every truth-group cap, validate actual new source contract, and require fresh source J')
    parser.add_argument('--truth-restoration-reviewed-manual-append-audit',type=Path,help='Explicit same-complete-q source-only manual append audit: preserve every parent cap and bind new actual source groups to the actual startup manifest')
    parser.add_argument('--physical-course-scope-source',type=Path,help='Exact source/RGB-authorized interpretation of existing connected course scope; retain original human packet')
    parser.add_argument('--direct-human-truth-authority',type=Path,help='Registered direct human support envelopes: highest hard truth on the existing shared q')
    parser.add_argument('--direct-human-full-stroke-guard',type=Path,help='Complete authenticated human strokes as hard guards, without extra independent FIT weights')
    parser.add_argument('--original-wide-human-truth-authority',type=Path,help='Original direct wide TOP strokes on their existing private25 camera operator')
    parser.add_argument('--direct-human-stroke-cap-support',type=Path,help='Actual drawing cap pixel cells and registration support; never a physical endpoint measurement')
    parser.add_argument('--practical-human-location-envelope',type=Path,help='Fixed engineering manual-location drift guard from the visually reviewed initializer; no pixel-exact physical precision claim')
    parser.add_argument('--user-confirmed-source-truth-authority',type=Path,help='Explicitly confirmed existing native RGB measurements on their original finite operators; no extra FIT weight')
    parser.add_argument('--joint-source-truth-restoration',action='store_true',help='Restore violated manual/confirmed groups inside the full original objective, with practical group uncertainty and accepted violations ratcheted downward; never a human-only initializer')
    parser.add_argument('--truth-restoration-original-step-bounds',action='store_true',help='Use original native/wide/confirmed support bounds for the joint-restoration QP direction; retain existing nonlinear ratchet guards and true complete-source cost acceptance')
    parser.add_argument('--truth-restoration-curvature-compensation',action='store_true',help='Opt-in one same-H/g QP re-solve with only measured outward source-face curvature compensated inside original step bounds; original guards and full-source cost acceptance unchanged')
    parser.add_argument('--finite-rectangle-station-model',action='store_true',help='Use eliminated local foot-station faces at authenticated current finite straight-line ends; original nonlinear human supports and guards unchanged')
    parser.add_argument('--truth-restoration-priority',action='store_true',help='Contract actual violated original truth supports before source-cost-only optimization; all source costs and nonlinear guards remain evaluated')
    parser.add_argument('--direct-human-centre-target',action='store_true',help='Existing sparse finite source operators target direct manual centres as correlated trace means; practical support remains a hard guard')
    parser.add_argument('--original-ice-pixel-correction-source',type=Path,help='Source-only reviewed pixel/support correction on exact original FIT ice rows; preserve historical bytes and row layout')
    parser.add_argument('--original-measured-transition-location-source',type=Path,help='Authenticate saved original FIT boundary estimators; preserve UV/sigma/support/typed geometry and evaluate the measured location')
    parser.add_argument('--original-source-role-retirement-source',type=Path,help='Reviewed source-only false print-role retirement; keeps original scalar slots and history, neutralizes only certified r/J rows')
    parser.add_argument('--original-source-point-qualification-source',type=Path,help='Source-only RGB qualification of exact original absolute FIT points; preserve UV/sigma/world/history and scalar slots, exclude reviewed r/J rows and accuracy metrics')
    parser.add_argument('--original-absolute-input-compaction-source',type=Path,help='Source-certified original RGB-polyline representatives and reviewed invalid points; physically compact active native inputs, preserve original source history and full guards')
    parser.add_argument('--original-board-curve-integration-policy',type=Path,help='Opt-in authenticated original board curve/exposure source-support budget outside Huber; preserve active UV, pointwise sigma, R/J, roles and existing qualified near-board loss policy; fresh source objective required')
    parser.add_argument('--original-ice-curve-integration-policy',type=Path,help='Opt-in original ICE physical curve/exposure budget outside Huber; separate explicitly declared historical sampling factors while retaining original UV, sigma, R/J and all hard supports; fresh objective required')
    parser.add_argument('--original-wide-curve-integration-policy',type=Path,help='Opt-in original wide45 physical curve/exposure integration outside Huber; preserve UV, working sigma, literal weight4, outer4, integer row layout and all hard supports; fresh objective required')
    parser.add_argument('--native-base-curve-integration-policy',type=Path,help='Opt-in authenticated native curve support weights on base blocks only; preserve raw sigma, mean-normalized human centre factors and all hard truth supports')
    parser.add_argument('--static-motion-source-role-retirement-source',type=Path,help='Source-only broadcast overlay role correction; retains original LK history and scalar layout, neutralizes certified static-world motion r/J rows')
    parser.add_argument('--original-motion-far-board-role-source',type=Path,help='Actual original LK points identified as current far-board material; retain point/scalar inventory and source sigma while estimating current finite surface depth')
    parser.add_argument('--independent-board-pattern-placement-source',type=Path,help='Direct-user same-print authority: append only observed Ad25 station/Z placement, retaining shared template and sizes')
    parser.add_argument('--persistent-ice-long-baseline-source',type=Path,help='Frozen same-material ice anchor/target UV correspondences; append real full-Brown camera reprojection rows without new parameters')
    parser.add_argument('--reuse-wide-linearization',type=Path,help='Authenticated same-state small wide r/J cache; never reuses programme blocks')
    parser.add_argument('--reuse-complete-linearization',type=Path,help='Authenticated complete same-q/profile source r/J and H/g; no source reevaluation')
    parser.add_argument('--reuse-source-blocks-from',type=Path,help='Strict same-q/profile original r/J for append-only board/ice target; calculate only added FIT rows and reaccumulate H/g')
    parser.add_argument('--backend-equivalence-evidence',type=Path,help='Verified native/NumPy mathematical equivalence for saved block reuse')
    parser.add_argument('--verify-source-blocks-from',type=Path,help='Actually recompute accelerated r/J and compare every unchanged block at the saved q/profile')
    parser.add_argument('--verify-activated-programme-families',action='store_true',help='At the existing complete same-q/profile normal, compare the independent complete main1/secondary family r/J/roles/ledger to its already-retained replay cache before solving')
    parser.add_argument('--verified-source-prefix-report',type=Path,help='Authenticate already verified native-equivalent programme prefix at the exact initial q/profile')
    parser.add_argument('--initial-damping',type=float,help='Explicit numerical damping for a declared new target; otherwise inherit a same-target complete checkpoint')
    parser.add_argument('--watch-native-evidence',action='store_true',help='Engineering review: freeze every accepted state and stop on raw source regression; never an accuracy claim')
    opt=parser.parse_args()
    # Keep legacy requests readable, but joint restoration must use the
    # current nonlinear group ratchet as its numerical step domain. The old
    # one-shot original-support and curvature routes cannot override it.
    fixed_seed_policy=opt.truth_restoration_progress_policy=='fixed_seed_envelope'
    if fixed_seed_policy and not(opt.joint_source_truth_restoration and opt.truth_restoration_priority):
        parser.error('Fixed seed envelope requires the complete existing joint source/SOC restoration interface')
    seed_engineering_scale_overrides={};seed_engineering_scale_source=None
    if opt.truth_restoration_seed_engineering_scale_source is not None:
        if not fixed_seed_policy:parser.error('Existing engineering scale override requires the explicit fixed seed policy')
        from calibration.solver_runtime.fixed_seed_truth_envelope import load_existing_source_engineering_scales
        seed_engineering_scale_overrides,seed_engineering_scale_source=load_existing_source_engineering_scales(
            opt.truth_restoration_seed_engineering_scale_source,native_authority_path=opt.direct_human_truth_authority,
            location_envelope_path=opt.practical_human_location_envelope)
    effective_original_step_bounds=bool(opt.truth_restoration_original_step_bounds and not opt.joint_source_truth_restoration)
    if opt.truth_restoration_priority and not opt.joint_source_truth_restoration:
        parser.error('Truth restoration priority requires the full joint source restoration domain')
    effective_curvature_compensation=bool(opt.truth_restoration_priority or (opt.truth_restoration_curvature_compensation and not opt.joint_source_truth_restoration))
    if opt.finite_rectangle_station_model and not opt.truth_restoration_priority:
        raise ValueError('Finite rectangle station model requires the complete truth-restoration priority route')
    if opt.truth_restoration_curvature_compensation and not opt.truth_restoration_original_step_bounds and not opt.joint_source_truth_restoration:
        parser.error('Measured source curvature compensation requires an explicit truth step domain')
    if not np.isfinite(opt.source_jacobian_refresh_ratio)or opt.source_jacobian_refresh_ratio<0.:
        parser.error('Finite nonnegative source Jacobian prediction refresh ratio required')
    if opt.truth_restoration_original_step_bounds and not opt.joint_source_truth_restoration:
        parser.error('Original truth step bounds require --joint-source-truth-restoration')
    from calibration.solver_runtime.original_wide_curve_integration import checked_wide_curve_integration_reuse
    checked_wide_curve_integration_reuse(opt)
    if opt.truth_restoration_reviewed_soft_source_proof is not None:
        if opt.truth_restoration_state is None or not opt.joint_source_truth_restoration:
            raise ValueError('Reviewed soft-source continuation requires saved complete truth state and joint restoration')
        if any(value is not None for value in (opt.truth_restoration_source_revision_proof,opt.truth_restoration_source_role_proof,opt.truth_restoration_source_append_proof,opt.truth_restoration_source_location_revision_proof,opt.truth_restoration_reviewed_manual_append_audit)):
            raise ValueError('Reviewed soft-source continuation requires its own single exact proof')
        if any(value is not None for value in (opt.reuse_complete_linearization,opt.reuse_source_blocks_from,opt.reuse_wide_linearization)):
            raise ValueError('Changed soft-source inputs require fresh initial profiles/J/normal')
    if opt.truth_restoration_source_location_revision_proof is not None:
        if opt.truth_restoration_state is None or not opt.joint_source_truth_restoration:
            raise ValueError('Measured location revision requires the complete parent truth state and joint restoration')
        if any(value is not None for value in (opt.truth_restoration_source_revision_proof,opt.truth_restoration_source_role_proof,opt.truth_restoration_source_append_proof,opt.truth_restoration_reviewed_manual_append_audit)):
            raise ValueError('Measured location revision requires its own exact source migration proof')
        if any(value is not None for value in (opt.reuse_complete_linearization,opt.reuse_source_blocks_from,opt.reuse_wide_linearization)):
            raise ValueError('Measured location revision requires fresh source/J/profile normal generation')
    if opt.truth_restoration_source_revision_proof is not None:
        if opt.truth_restoration_state is None or not opt.joint_source_truth_restoration or opt.original_motion_far_board_role_source is None or opt.additional_native_board_track_source is None:
            raise ValueError('Reviewed source revision requires complete truth state, joint restoration and both exact source packets')
        if opt.truth_restoration_source_role_proof is not None or opt.truth_restoration_source_append_proof is not None:
            raise ValueError('Only one exact source migration proof may be authorized')
        if any(value is not None for value in (opt.reuse_complete_linearization,opt.reuse_source_blocks_from,opt.reuse_wide_linearization)):
            raise ValueError('Reviewed source target change requires fresh source/J/profile normal generation')
    if opt.truth_restoration_source_role_proof is not None:
        if opt.truth_restoration_state is None or not opt.joint_source_truth_restoration or opt.original_motion_far_board_role_source is None:
            raise ValueError('Typed-role proof requires original complete truth state, joint restoration and explicit physical role packet')
        if opt.truth_restoration_source_append_proof is not None:raise ValueError('Only one exact source migration may be authorized')
        if any(value is not None for value in (opt.reuse_complete_linearization,opt.reuse_source_blocks_from,opt.reuse_wide_linearization)):
            raise ValueError('Typed-role target change requires fresh source/J/profile normal generation')
    if opt.truth_restoration_source_append_proof is not None:
        if opt.truth_restoration_state is None or not opt.joint_source_truth_restoration:
            parser.error('A source append proof requires the original truth restoration state and joint truth restoration')
        if any(value is not None for value in(opt.reuse_complete_linearization,opt.reuse_source_blocks_from,opt.reuse_wide_linearization)):
            parser.error('A source append truth target requires fresh complete source profiles and J')
    human_options=(opt.direct_human_truth_authority,opt.direct_human_full_stroke_guard,opt.original_wide_human_truth_authority,opt.direct_human_stroke_cap_support,opt.practical_human_location_envelope)
    if any(x is not None for x in human_options)and not all(x is not None for x in human_options):
        parser.error('All direct native and existing wide human hard-truth authorities must be supplied together')
    if not np.isfinite(opt.roll_prior_sigma_degrees)or opt.roll_prior_sigma_degrees<=0:
        parser.error('Finite positive explicit engineering roll-prior sigma required')
    if opt.original_motion_far_board_role_source is not None and any(value is not None for value in
            (opt.verify_source_blocks_from,opt.reuse_wide_linearization)):
        parser.error('Original motion material-role correction requires fresh complete source derivatives')
    if opt.additional_native_board_track_source is not None and any(value is not None for value in
            (opt.verify_source_blocks_from,opt.reuse_wide_linearization)):
        parser.error('New actual board tracks require fresh complete source derivatives')
    if opt.reinitialize_invalid_mounts and opt.camera_mount_constraint_source is None:
        parser.error('Explicit user mount constraint source required for mount reinitialization')
    if opt.reinitialize_invalid_pivot_positions and opt.camera_pivot_exterior_constraint_source is None:
        parser.error('Explicit user pivot exterior constraint source required for pivot reinitialization')
    if (opt.camera_mount_constraint_source is not None or opt.camera_pivot_exterior_constraint_source is not None) and any(value is not None for value in
            (opt.verify_source_blocks_from,opt.reuse_wide_linearization)):
        parser.error('Restored camera physical domain uses the actual current source normal, not a saved matrix')
    # RAM Jacobian reuse starts with this run's fresh complete normal. It never
    # imports an old derivative, including after an initializer or pixel edit.
    # Saved matrices from the preceding objective remain forbidden here.
    if opt.absolute_camera_initializer_source is not None and any(value is not None for value in
            (opt.verify_source_blocks_from,opt.reuse_wide_linearization)):
        parser.error('A source camera initializer requires fresh complete profiles/J, never a cached derivative')
    if opt.original_ice_pixel_correction_source is not None and any(value is not None for value in
            (opt.verify_source_blocks_from,opt.reuse_wide_linearization)):
        parser.error('Corrected original ice pixels require fresh complete source profiles/J')
    if opt.original_source_point_qualification_source is not None and any(value is not None for value in
            (opt.verify_source_blocks_from,opt.reuse_wide_linearization,opt.reuse_complete_linearization,opt.reuse_source_blocks_from)):
        parser.error('Original source point qualification requires fresh complete source profiles/J')
    if min(opt.geometry_fd_workers,opt.normal_workers)<1:parser.error('Positive geometry FD and normal worker counts required')
    if opt.residual_native_threads<1:parser.error('Positive residual native thread count required')
    if opt.residual_source_workers not in (0,6):parser.error('Residual source worker count must be zero or the six distinct original bundles')
    if opt.geometry_fd_workers>1 and not opt.serial_source:parser.error('Light geometry FD workers require --serial-source to keep one source graph and avoid nested full-graph actors')
    if (opt.original_motion_low_compaction_source is None)!=(opt.original_motion_main_compaction_source is None):
        parser.error('Both explicit original-motion compaction authorities are required')
    if opt.original_motion_low_compaction_source is not None:
        if opt.static_motion_source_role_retirement_source is None:parser.error('Compaction preserves the explicit original overlay retirement authority')
        if opt.verify_source_blocks_from is not None:
            parser.error('Changed active input inventory requires fresh complete source J')
    os.environ['RINKSPACE_NORMAL_WORKERS']=str(opt.normal_workers)
    if opt.reuse_source_blocks_from is not None and opt.reuse_complete_linearization is not None:
        parser.error('Choose unchanged-target complete reuse or rebuilt append-only target reuse')
    if opt.verify_source_blocks_from is not None and (opt.reuse_source_blocks_from is not None or opt.reuse_complete_linearization is not None):
        parser.error('Actual accelerated r/J verification must not reuse a cached normal')
    if opt.verified_source_prefix_report is not None and opt.verify_source_blocks_from is None:
        parser.error('Verified prefix is usable only inside the same-state source audit')
    if opt.independent_board_pattern_placement_source is not None:
        if opt.additional_board_material_point_source is None:
            parser.error('Independent Ad25 placement requires its actual admitted material-point source')
    # Presence of a correction/domain option does not mean it changed on this
    # resume. The actual sealed identity gates below compare every target,
    # physical domain, layout, mask, frozen profile and implementation file.
    restore_native_backend_contract(stage='before_prepare')
    if opt.initial_damping is not None and (not np.isfinite(opt.initial_damping)or opt.initial_damping<=0):parser.error('Finite positive explicit numerical damping required')
    out=opt.output.resolve()
    if not out.is_relative_to(BASE.resolve())or out.exists():parser.error('New Data candidate output required')
    t=time.perf_counter();print(json.dumps(dict(stage='build_preserved_complete_original_programme_source_graph')),flush=True)
    # This preparation uses only the unchanged graph/source initializer.
    # The requested complete checkpoint is restored below, after preparation;
    # its q9 is never evaluated against the old initializer's local course/H.
    state,args,template,parent=prepare(DEFAULT_SEED,out)
    if opt.physical_course_scope_source is not None:state.current_user_physical_course_scope_source=opt.physical_course_scope_source
    install_corrected_advertisement_source_geometry(state);install_shared_ice_hockey_geometry(state)
    args.source_processes=False;args.source_workers=1;args.compact_source_review=True
    args.residual_source_workers=0 if opt.strict_source_fit_dataset is not None else opt.residual_source_workers
    args.residual_native_threads=opt.residual_native_threads
    capold=capture_hockey_driver(args,state)
    factorold=HockeyWideHighSourceFactor(state)
    oldfull=np.r_[state.q,template['q'][-3:],factorold.initial_camera_state25]
    inherited=_frozen_profile_token(make_profile_update(state.blocks))
    g62,q62,_,_,_,_=make_structured_geometry()
    physical_geometry_initializer={}
    if opt.estimate_unverified_physical_geometry:
        packet={}if opt.additional_native_board_track_source is None else json.loads(opt.additional_native_board_track_source.read_text(encoding='utf8'))
        templates=packet.get('frozen_source_print_templates',[])
        if len(templates)!=3:raise ValueError('Three explicit q0 frozen source print templates are required for the unified physical geometry')
        physical_geometry_initializer=dict(additional_independent_locations=[15,16,17,18,19],
            frozen_source_print_templates=templates,independent_lane_right_extensions=True)
    if opt.strict_half_circle_crease:physical_geometry_initializer['strict_half_circle_crease']=True
    if opt.derive_hockey_center_from_course:physical_geometry_initializer['derive_hockey_center_from_course']=True
    if opt.shared_hockey_curling_axis:physical_geometry_initializer['shared_hockey_curling_axis']=True
    if opt.estimate_unverified_goal_crease_placement:physical_geometry_initializer['independent_goal_crease_transverse_placements']=True
    geometry=PairedLocationsSharedSizesGeometry(g62.parent,q62[:g62.parent_parameter_count],
        additional_independent_locations=physical_geometry_initializer.get('additional_independent_locations',()),
        source_print_templates=physical_geometry_initializer.get('frozen_source_print_templates',()),
        independent_lane_right_extensions=physical_geometry_initializer.get('independent_lane_right_extensions',False),
        strict_half_circle_crease=physical_geometry_initializer.get('strict_half_circle_crease',False),
        derive_hockey_center_from_course=physical_geometry_initializer.get('derive_hockey_center_from_course',False),
        independent_goal_crease_transverse_placements=physical_geometry_initializer.get('independent_goal_crease_transverse_placements',False),
        shared_hockey_curling_axis=physical_geometry_initializer.get('shared_hockey_curling_axis',False))
    initial=geometry.initial.copy()
    if opt.world_init:
        payload=json.loads(opt.world_init.read_text(encoding='utf8'))
        initial=np.asarray(payload.get('world66',payload.get('q',payload.get('parameters'))),float)
        if initial.shape==(66,)and geometry.parameter_count>66:initial=np.r_[initial,geometry.initial[66:]]
        if initial.shape!=(geometry.parameter_count,):raise ValueError('Explicit complete physical world init required')
        if payload.get('geometry_initializer_auxiliary')is not None:
            from user_corrected_programme_candidate.initialize_latest9a8c_world66 import apply_latest_geometry_auxiliary
            apply_latest_geometry_auxiliary(geometry,payload)
    # Fresh producer materials remain from original constructor; a changed
    # initial vector is the candidate q, not a rewritten identity certificate.
    geometry.validate_parameters(initial)
    from structured_programme_source_candidate.course_station_cache import install_exact_course_station_cache
    install_exact_course_station_cache(geometry.print_course_binding(initial)[0])
    inherited_current_world=None;inherited_circle_suffix=None;inherited_pattern_suffix=None;inherited_geometry_tail=None;continued_damping=None;continued_damping_identity=None
    pending_compacted_profiles=None;inherited_checkpoint_q9=None
    continued_line_search_fraction=1.;inherited_adaptive_line_search=False
    if opt.camera_seed:
        print(json.dumps(dict(stage='restore_complete_latest_camera_nonworld_state',camera_seed=str(opt.camera_seed))),flush=True)
        with np.load(opt.camera_seed,allow_pickle=False)as z:
            seedq=z['q'].copy();profiles={k[9:]:z[k].copy()for k in z.files if k.startswith('profile__')}
            if 'aux__shared_pattern_start'in z:
                pattern_start=int(z['aux__shared_pattern_start'])
                if not 0<=pattern_start<=OLD.core_count-9:
                    raise ValueError('Saved q9 must belong to the preserved original nonworld prefix')
                inherited_checkpoint_q9=seedq[pattern_start:pattern_start+9].copy()
            if 'aux__current_complete_parameter_layout_json'in z:
                saved_layout=json.loads(str(z['aux__current_complete_parameter_layout_json'].item()))
                old_complete=int(saved_layout['original_complete_parameter_count']);extra=int(saved_layout['additional_world_parameter_count'])
                intended_extra=geometry.parameter_count-66
                compatible_append=(opt.estimate_unverified_goal_crease_placement and extra==32 and intended_extra==34)
                if (seedq.shape!=(old_complete+extra,)or (extra!=intended_extra and not compatible_append)
                        or old_complete not in(34302,34306)):raise ValueError('Exact compatible canonical complete geometry extension required')
                inherited_geometry_tail=seedq[old_complete:].copy();seedq=seedq[:old_complete].copy()
            if 'next_line_search_fraction' in z:
                continued_line_search_fraction=float(z['next_line_search_fraction'])
                inherited_adaptive_line_search=True
            elif opt.adaptive_source_backtracking and 'history_json' in z:
                accepted=[r for r in read_checkpoint_history(z,opt.camera_seed)if r.get('accepted')]
                if accepted and accepted[-1].get('global_schur_trial_records'):
                    continued_line_search_fraction=float(accepted[-1]['global_schur_trial_records'][-1]['whole_source_step_fraction'])
            placement_prefix_count=BASE_FULL_COUNT+(6 if opt.additional_board_circle_source is not None else 0)
            if (opt.independent_board_pattern_placement_source is not None and
                    seedq.shape in ((placement_prefix_count+2,),(placement_prefix_count+4,))):
                inherited_pattern_suffix=seedq[placement_prefix_count:].copy();seedq=seedq[:placement_prefix_count].copy()
            if seedq.shape in ((BASE_FULL_COUNT,),(BASE_FULL_COUNT+6,)):
                continued_damping=float(z['damping'])
                if not np.isfinite(continued_damping)or continued_damping<=0:raise ValueError('Finite continued optimizer damping required')
                continued_damping_identity=dict(quadratic=z['quadratic'].copy(),scales=z['scales'].copy(),identity=json.loads(str(z['identity_json'].item())))
        if seedq.shape in ((BASE_FULL_COUNT,),(BASE_FULL_COUNT+6,)):
            # Continue the new complete state without changing its fixed
            # material/source initializer. Only numerical q comes from the
            # accepted checkpoint; the supplied initializer's auxiliary
            # carrier references remain unchanged.
            seedq,inherited_current_world,inherited_circle_suffix=current_seed_parts(seedq,
                circle_enabled=opt.additional_board_circle_source is not None)
        if seedq.shape==(34253,):seedq=OLD.extend_full_q(seedq)
        if seedq.shape!=oldfull.shape:raise ValueError('Explicit old complete camera/world chart required for camera inheritance')
        keep=np.ones(OLD.core_count,bool);keep[np.asarray(OLD.world_indices)]=False
        oldfull[:OLD.core_count][keep]=seedq[:OLD.core_count][keep];oldfull[OLD.core_count:]=seedq[OLD.core_count:]
        state.q=oldfull[:OLD.core_count].copy()
        if profiles:
            compatible=all(np.asarray(profiles['weights_'+str(i)]).shape in ((0,2),(len(b.bundle.a),2))for i,b in enumerate(state.blocks))
            if compatible:
                make_profile_update(state.blocks).restore_checkpoint(profiles);inherited=_frozen_profile_token(make_profile_update(state.blocks))
            elif opt.original_motion_low_compaction_source is not None:
                pending_compacted_profiles=profiles
            else:raise ValueError('Saved profile point inventory requires the same explicit source compaction authority')
    before=[]
    for b in state.blocks:
        local=b.local(state.q);before.append(np.array([b.bundle.camera(local,p)for p in b.bundle.stamps]))
    # Named66 installation uses constructor initial solely for source binding;
    # place the intended revised candidate values into its returned chart.
    state.source_actors=None
    contract,q=install_structured_programme_blocks(state,geometry,oldfull,inherited_profile_token=inherited)
    core_count=state.structured_programme_coordinate_chart.core_count
    world_indices=np.asarray(state.geometry_parameter_indices,int)
    evaluation_base_full_count=BASE_FULL_COUNT+geometry.parameter_count-66
    install_exact_course_station_cache(state.geometry.source_course.course)
    candidate_world=initial.copy()
    if inherited_current_world is not None:candidate_world[:66]=inherited_current_world
    if inherited_geometry_tail is not None:candidate_world[66:66+len(inherited_geometry_tail)]=inherited_geometry_tail
    q[world_indices]=candidate_world
    state.q=q[:core_count].copy()
    camera_error=0.
    for b,previous in zip(state.blocks,before):
        local=b.local(state.q);actual=np.array([b.bundle.camera(local,p)for p in b.bundle.stamps]);camera_error=max(camera_error,float(abs(actual-previous).max()))
    if camera_error!=0.:raise ValueError('Camera values changed while replacing only the shared world')
    if not opt.complete_scene_material_diagnostic:
        from structured_programme_source_candidate.current_ice_material_scene import install_source_ice_material_view
        install_source_ice_material_view(state.geometry,max_cache_bytes=32*1024**2)
        print(json.dumps(dict(stage='exact_current_ice_material_view_installed',maximum_cache_bytes=32*1024**2,
            original_feature_transform_and_footprint_checks_preserved=True)),flush=True)
    mount_working_constraints=None;mount_reinitialization=None;mount_parameter_bounds={};pose_feasibility_initializer=None
    absolute_camera_initializer=None
    pivot_outside_constraints=None;pivot_reinitialization=None;shared_linear_constraints=None
    if opt.camera_mount_constraint_source is not None:
        from installation_mount_candidate.mount_graph import (
            install_mount_working_constraints,reinitialize_invalid_mounts,validate_mounts)
        mount_working_constraints=install_mount_working_constraints(state,opt.camera_mount_constraint_source)
        state.current_user_mount_constraint_contract=mount_working_constraints
        mount_parameter_bounds=state.mount_working_parameter_bounds
    if opt.camera_pivot_exterior_constraint_source is not None:
        from installation_mount_candidate.mount_graph import (
            install_pivot_outside_constraints,reinitialize_invalid_pivots_outside,
            pivot_outside_linear_step_constraints,pivot_outside_domain,validate_mounts)
        # Install the facts before workers copy the state. The inherited q can
        # remain invalid until the explicit mount and pivot stages below.
        pivot_outside_constraints=install_pivot_outside_constraints(state,opt.camera_pivot_exterior_constraint_source)
        state.current_user_pivot_outside_constraint_contract=pivot_outside_constraints
        shared_linear_constraints=lambda value,selected: pivot_outside_linear_step_constraints(state,value,selected)
    args.source_processes=not opt.serial_source;args.source_actor_threads=3
    ice_pixel_correction=None
    if opt.original_ice_pixel_correction_source is not None:
        from structured_programme_source_candidate.original_ice_pixel_correction import install_original_ice_pixel_correction,review_corrected_ice_pixels,require_painted_retirement_compaction
        ice_pixel_correction=install_original_ice_pixel_correction(state,opt.original_ice_pixel_correction_source)
        require_painted_retirement_compaction(ice_pixel_correction,opt.original_source_point_qualification_source,opt.original_absolute_input_compaction_source)
    original_measured_locations=None
    if opt.original_measured_transition_location_source is not None:
        from calibration.solver_runtime.original_measured_transition_locations import install_original_measured_transition_locations
        original_measured_locations=install_original_measured_transition_locations(state,opt.original_measured_transition_location_source)
    far_board_role_correction=None
    if opt.original_motion_far_board_role_source is not None:
        from structured_programme_source_candidate.original_motion_far_board_roles import install_original_motion_far_board_roles
        far_board_role_correction=install_original_motion_far_board_roles(state,opt.original_motion_far_board_role_source,q)
    retired_motion_compaction=None
    if opt.strict_source_fit_dataset is not None:
        if opt.strict_motion_inputs is None:
            raise ValueError('Strict FIT graph requires authenticated zero-motion inputs')
        from calibration.solver_runtime.strict_optimizer_resume import retired_motion_compaction_policy
        retired_motion_compaction=retired_motion_compaction_policy(
            dict(path=str(opt.strict_motion_inputs.resolve()),sha256=sha(opt.strict_motion_inputs)),state.blocks)
    motion_input_compaction=None
    if opt.original_motion_low_compaction_source is not None:
        from structured_programme_source_candidate.original_motion_input_compaction import compact_original_motion_inputs
        motion_input_compaction=compact_original_motion_inputs(state,capold,template,
            opt.original_motion_low_compaction_source,opt.original_motion_main_compaction_source,
            retirement_packet=opt.static_motion_source_role_retirement_source,
            verify_value_J=retired_motion_compaction is None)
        if retired_motion_compaction is not None:
            motion_input_compaction['strict_retired_numerical_check']=retired_motion_compaction
        if pending_compacted_profiles is not None:make_profile_update(state.blocks).restore_checkpoint(pending_compacted_profiles)
        print(json.dumps(dict(stage='actual_original_motion_input_compaction_passed'if retired_motion_compaction is None else'archived_original_motion_input_mapping_authenticated',
            removed_points=motion_input_compaction['removed_motion_points'],
            removed_scalar_rows=motion_input_compaction['removed_motion_scalar_rows'],
            all_camera_parameters_preserved=True,
            archived_motion_numerical_check='PASSED'if retired_motion_compaction is None else'NOT_APPLICABLE_RETIRED_PIXEL_OBJECTIVE')),flush=True)
    active_backend_report=restore_native_backend_contract(stage='before_source_actor_construction')
    absolute_input_compaction=None
    if opt.original_absolute_input_compaction_source is not None:
        if opt.original_source_point_qualification_source is None:
            raise ValueError('Absolute compaction needs the exact original point qualification certificate')
        from calibration.solver_runtime.original_absolute_input_compaction import compact_original_absolute_inputs
        absolute_input_compaction=compact_original_absolute_inputs(state,capold,template,
            opt.original_absolute_input_compaction_source,qualification_path=opt.original_source_point_qualification_source)
        print(json.dumps(dict(stage='actual_original_absolute_input_compaction_passed',
            removed_scalar_rows=absolute_input_compaction['removed_active_scalar_rows'],
            all_camera_parameters_and_original_full_truth_preserved=True)),flush=True)
    existing17_check=None
    if far_board_role_correction is not None and opt.strict_source_fit_dataset is None:
        from structured_programme_source_candidate.paired_programme_blocks import check_existing_finite_perimeter_sources
        print(json.dumps(dict(stage='actual_existing17_continuous_perimeter_preflight_started')),flush=True)
        # Keep the driver's new-output creation boundary intact; preparation
        # evidence already belongs to its existing sibling bootstrap folder.
        existing17_path=out.with_name(out.name+'.bootstrap')/'actual_existing17_continuous_perimeter_value_J_crossregion_check.json'
        existing17_check=check_existing_finite_perimeter_sources(state,q,output_path=existing17_path)
        write(existing17_path,existing17_check)
        print(json.dumps(dict(stage='actual_existing17_continuous_perimeter_preflight_passed',
            original_points=17,actual_world_count=state.geometry.parameter_count,
            FD_max=max(c['FD_error']for b in existing17_check['block_checks']for c in b['FD_checks']))),flush=True)
    raw=CurrentProgrammeSource(state,args,capold)
    if inherited_checkpoint_q9 is not None:
        if state.patterns is None or not np.array_equal(q[state.patterns.start:state.patterns.start+9],inherited_checkpoint_q9):
            raise ValueError('The complete checkpoint q9 changed during graph/layout restoration')
        # The physical guard now belongs to the actual complete current course,
        # including its actual board height and periodic material station.
        with state.patterns.bind_current_geometry(q[:core_count]):
            state.patterns.geometry.guard(inherited_checkpoint_q9)
        print(json.dumps(dict(stage='complete_checkpoint_q9_restored_on_current_continuous_course',
            original_checkpoint_q9_bitexact=True,legacy_bootstrap_domain_not_applied_to_checkpoint=True)),flush=True)
    cap=raw.capture()
    current_pattern_pool=None
    if state.patterns is not None and not opt.serial_current_patterns and opt.strict_source_fit_dataset is None:
        from structured_programme_source_candidate.current_geometry_pattern_pool import install_current_geometry_pattern_pool
        current_pattern_pool=install_current_geometry_pattern_pool(state.patterns,workers=4,enabled=True)
    # The legacy capture owns preparation-only closures and blocks. Current
    # source owns copied original masks/scales and its own complete factories.
    import gc
    from rinkspace.runtime_memory import process_identity,sample_owned_tree
    memory_before_capture_release=sample_owned_tree([process_identity(os.getpid())])
    raw.old_cap=None;del capold;gc.collect()
    memory_after_capture_release=sample_owned_tree([process_identity(os.getpid())])
    preparation_capture_release=dict(legacy_capture_retained=False,source_masks_scales_and_current_factories_retained=True,
        original_clock_camera_metadata_retained=True,before=asdict(memory_before_capture_release),after=asdict(memory_after_capture_release),
        rss_difference_bytes=memory_before_capture_release.rss_bytes-memory_after_capture_release.rss_bytes,
        private_difference_bytes=memory_before_capture_release.private_bytes-memory_after_capture_release.private_bytes,
        source_geometry_q_profiles_not_changed=True)
    print(json.dumps(dict(stage='preserved_graph_ready_for_complete_material_suffixes')),flush=True)
    base=ExplicitBiasSource(cap,state,component_sigma_degrees=opt.roll_prior_sigma_degrees)
    base.initial_bias(state.q);baseq=np.r_[state.q,q[core_count:core_count+3]]
    print(json.dumps(dict(stage='preserved_bias_chart_ready')),flush=True)
    factor=CurrentWideSource(factorold,state,json.loads(TRACE.read_text(encoding='utf8')))
    print(json.dumps(dict(stage='preserved_wide_material_geometry_ready')),flush=True)
    source=CompleteWideSource(base,factor,4)
    complete_wide_source=source
    source=wrap_named66_soft_geometry_references(source,state.geometry,world_indices)
    print(json.dumps(dict(stage='preserved_soft_geometry_references_ready')),flush=True)
    if opt.additional_ice_curves is not None:
        from structured_programme_source_candidate.native_ice_curve_source import wrap_additional_native_ice_curves
        source=wrap_additional_native_ice_curves(source,state,opt.additional_ice_curves)
        print(json.dumps(dict(stage='preserved_native_ice_material_geometry_ready')),flush=True)
    if opt.additional_board_source is not None:
        from structured_programme_source_candidate.native_board_edge_source import wrap_additional_native_board_edges
        source=wrap_additional_native_board_edges(source,state,opt.additional_board_source)
        print(json.dumps(dict(stage='preserved_native_board_material_geometry_ready')),flush=True)
    if opt.additional_board_material_point_source is not None:
        from structured_programme_source_candidate.native_board_material_point_source import wrap_additional_native_board_material_points
        source=wrap_additional_native_board_material_points(source,state,opt.additional_board_material_point_source)
    # Construct the historical complete prefix first. Wide indices stay at
    # their original explicit slots; circle materials append after that prefix.
    camera_indices_before_circle=source.camera_indices.copy()
    q=source.initial(baseq,q[-25:])
    if q.shape!=(evaluation_base_full_count,):raise ValueError('Complete evaluation parameter chart changed')
    if opt.additional_board_circle_source is not None:
        from structured_programme_source_candidate.native_board_circle_source import wrap_additional_native_board_circles
        source=wrap_additional_native_board_circles(source,state,opt.additional_board_circle_source)
        prefix=q.copy();q=source.extend_initial(prefix,suffix=inherited_circle_suffix)
        if (source.prefix_count!=evaluation_base_full_count or source.count!=evaluation_base_full_count+6 or
            not np.array_equal(q[:evaluation_base_full_count],prefix)or
            not np.array_equal(source.camera_indices,camera_indices_before_circle)or
            not np.array_equal(source.circle_indices,np.arange(evaluation_base_full_count,evaluation_base_full_count+6))):
            raise ValueError('Independent circle suffix changed the original parameter or camera prefix')
        if inherited_circle_suffix is not None and not np.array_equal(q[source.circle_indices],inherited_circle_suffix):
            raise ValueError('Restart lost the actual independent circle material parameters')
    retirement=None
    if opt.original_source_role_retirement_source is not None:
        from structured_programme_source_candidate.original_source_role_retirement import wrap_original_source_role_retirement
        source=wrap_original_source_role_retirement(source,state,opt.original_source_role_retirement_source)
        retirement=source.original_role_retirement
    placements=None
    if opt.independent_board_pattern_placement_source is not None:
        from structured_programme_source_candidate.course_patterns_independent_placements import wrap_pattern_independent_placements
        prefix=q.copy();source=wrap_pattern_independent_placements(source,state,opt.independent_board_pattern_placement_source)
        q=source.extend_initial(prefix,suffix=inherited_pattern_suffix);placements=source
        if (not np.array_equal(q[:source.prefix_count],prefix)or source.count!=source.prefix_count+source.placement_count or
            not np.array_equal(source.camera_indices,camera_indices_before_circle)):
            raise ValueError('Independent print placement changed the complete original camera/material prefix')
        if inherited_pattern_suffix is not None and not np.array_equal(q[source.placement_indices[:len(inherited_pattern_suffix)]],inherited_pattern_suffix):
            raise ValueError('Restart lost the actual independent print placement parameters')
    persistent_ice=None
    if opt.persistent_ice_long_baseline_source is not None:
        from structured_programme_source_candidate.persistent_ice_long_baseline_source import wrap_persistent_ice_long_baseline
        prefix_masks=source.quadratic.copy();prefix_scales=source.scales.copy();prefix_count=source.count
        source=wrap_persistent_ice_long_baseline(source,state,opt.persistent_ice_long_baseline_source)
        persistent_ice=source.persistent_ice_factor
        if (source.count!=prefix_count or q.shape!=(prefix_count,) or
            not np.array_equal(source.quadratic[:len(prefix_masks)],prefix_masks) or
            not np.array_equal(source.scales[:len(prefix_scales)],prefix_scales) or
            not np.array_equal(source.camera_indices,camera_indices_before_circle)):
            raise ValueError('Persistent ice source changed the original parameter, camera, mask or scale prefix')
    motion_retirement=None
    if opt.static_motion_source_role_retirement_source is not None:
        from structured_programme_source_candidate.static_motion_source_role_retirement import wrap_static_motion_source_role_retirement
        source=wrap_static_motion_source_role_retirement(source,state,opt.static_motion_source_role_retirement_source)
        motion_retirement=source.static_motion_role_retirement
    board_tracks=None
    if opt.additional_native_board_track_source is not None:
        from structured_programme_source_candidate.native_board_track_source import wrap_native_board_tracks
        source=wrap_native_board_tracks(source,state,opt.additional_native_board_track_source)
        board_tracks=source.native_board_track_factor
    original_point_qualification=None
    print(json.dumps(dict(stage='preserved_material_suffixes_ready')),flush=True)
    if opt.original_source_point_qualification_source is not None:
        from calibration.solver_runtime.original_source_point_qualification import wrap_original_source_point_qualification
        source=wrap_original_source_point_qualification(source,state,opt.original_source_point_qualification_source)
        original_point_qualification=source.original_point_qualification
        print(json.dumps(dict(stage='preserved_original_source_qualification_ready')),flush=True)
    if absolute_input_compaction is not None:
        from calibration.solver_runtime.original_absolute_input_compaction import wrap_original_absolute_input_compaction
        source=wrap_original_absolute_input_compaction(source,state)
    original_board_curve_integration=None
    original_board_curve_integration_preparation_seconds=None
    if opt.original_board_curve_integration_policy is not None and opt.strict_source_fit_dataset is None:
        from calibration.solver_runtime.original_board_curve_integration import wrap_original_board_curve_integration
        source=wrap_original_board_curve_integration(source,state,opt.original_board_curve_integration_policy)
        original_board_curve_integration=source.original_board_curve_integration
        original_board_curve_integration_preparation_seconds=source.original_board_curve_integration_preparation_seconds
    original_ice_curve_integration=None
    original_ice_curve_integration_preparation_seconds=None
    if opt.original_ice_curve_integration_policy is not None and opt.strict_source_fit_dataset is None:
        from calibration.solver_runtime.original_ice_curve_integration import wrap_original_ice_curve_integration
        source=wrap_original_ice_curve_integration(source,state,opt.original_ice_curve_integration_policy)
        original_ice_curve_integration=source.original_ice_curve_integration
        original_ice_curve_integration_preparation_seconds=source.original_ice_curve_integration_preparation_seconds
    original_wide_curve_integration=None
    if opt.original_wide_curve_integration_policy is not None and opt.strict_source_fit_dataset is None:
        from calibration.solver_runtime.original_wide_curve_integration import wrap_original_wide_curve_integration
        source=wrap_original_wide_curve_integration(source,factor,opt.original_wide_curve_integration_policy,wide_source=complete_wide_source)
        original_wide_curve_integration=source.original_wide_curve_integration
    strict_dataset_factor=None
    strict_source=None
    if opt.strict_source_fit_dataset is not None:
        print(json.dumps(dict(stage='strict_dataset_full_inventory_construction_begin')),flush=True)
        from calibration.solver_runtime.strict_source_dataset_adapter import StrictDatasetFactor
        from calibration.solver_runtime.strict_reviewed_programme_source import StrictReviewedProgrammeSource
        strict_dataset_reference=dict(path=str(opt.strict_source_fit_dataset.resolve()),sha256=sha(opt.strict_source_fit_dataset))
        strict_motion_reference=dict(path=str(opt.strict_motion_inputs.resolve()),sha256=sha(opt.strict_motion_inputs))
        strict_dataset_factor=StrictDatasetFactor(state,strict_dataset_reference,source.count,
            circle_factor=source.circle_factor,material_factor=source.native_board_material_point_factor,pattern_source=source)
        strict_source=StrictReviewedProgrammeSource(source,raw,base,strict_dataset_factor,
            motion_reference=strict_motion_reference)
        source=strict_source
        print(json.dumps(dict(stage='strict_dataset_full_inventory_construction_ready',contract=strict_source.contract)),flush=True)
    out.mkdir(exist_ok=False)
    if original_board_curve_integration is not None:
        write(out/'actual_original_board_curve_integration_initial.json',dict(
            contract=original_board_curve_integration,complete_q_sha256=qtoken(q),
            preparation_seconds=original_board_curve_integration_preparation_seconds,
            original_active_R_J_scales_roles_and_hard_truth_unchanged=True,
            source_loss_objective_revised=True,old_original_cost_not_a_second_acceptance_veto=True))
    if original_ice_curve_integration is not None:
        write(out/'actual_original_ice_curve_integration_initial.json',dict(
            contract=original_ice_curve_integration,complete_q_sha256=qtoken(q),
            preparation_seconds=original_ice_curve_integration_preparation_seconds,
            original_active_R_J_scales_roles_and_hard_truth_unchanged=True,
            source_loss_objective_revised=True,old_original_cost_not_a_second_acceptance_veto=True))
    if original_wide_curve_integration is not None:
        write(out/'actual_original_wide_curve_integration_initial.json',dict(
            contract=original_wide_curve_integration,complete_q_sha256=qtoken(q),
            original_UV_R_J_scales_masks_profiles_and_all_hard_supports_unchanged=True,
            source_loss_objective_revised=True,old_original_cost_not_a_second_acceptance_veto=True))
    if existing17_check is not None:write(out/'actual_existing17_continuous_perimeter_value_J_crossregion_check.json',existing17_check)
    if motion_input_compaction is not None:write(out/'actual_original_motion_input_compaction.json',motion_input_compaction)
    if opt.camera_mount_constraint_source is not None:
        if opt.reinitialize_invalid_mounts:
            q,mount_reinitialization=reinitialize_invalid_mounts(state,q)
            write(out/'explicit_mount_working_range_reinitialization.json',mount_reinitialization)
        validate_mounts(state,q,check_pivot_exterior=pivot_outside_constraints is None)
        print(json.dumps(dict(stage='user_mount_working_constraints_restored',
            source_sha256=sha(opt.camera_mount_constraint_source),
            reinitialization=mount_reinitialization,nominal_q_sha256=qtoken(q))),flush=True)
    if pivot_outside_constraints is not None:
        if opt.reinitialize_invalid_pivot_positions:
            q,pivot_reinitialization=reinitialize_invalid_pivots_outside(state,q)
            write(out/'explicit_pivot_outside_reinitialization.json',pivot_reinitialization)
        print(json.dumps(dict(stage='user_pivot_exterior_constraints_ready',
            source_sha256=sha(opt.camera_pivot_exterior_constraint_source),
            contract=pivot_outside_constraints,reinitialization=pivot_reinitialization,
            actual_domain=pivot_outside_domain(state,q),q_sha256=qtoken(q))),flush=True)
    if opt.initial_pose_feasibility_source is not None:
        from structured_programme_source_candidate.qualified_ring_pose_initializer import apply_qualified_ring_pose_initializer
        q,pose_feasibility_initializer=apply_qualified_ring_pose_initializer(state,q,opt.initial_pose_feasibility_source)
        write(out/'explicit_source_domain_pose_feasibility_initialization.json',pose_feasibility_initializer)
        print(json.dumps(dict(stage='actual_source_domain_pose_feasibility_initialized',
            report=pose_feasibility_initializer,q_sha256=qtoken(q))),flush=True)
    if opt.absolute_camera_initializer_source is not None:
        from structured_programme_source_candidate.absolute_camera_pose_initializer import apply_absolute_camera_pose_initializer
        q,absolute_camera_initializer=apply_absolute_camera_pose_initializer(state,q,opt.absolute_camera_initializer_source)
        continued_line_search_fraction=1.;inherited_adaptive_line_search=False
        write(out/'explicit_absolute_source_camera_initialization.json',absolute_camera_initializer)
        print(json.dumps(dict(stage='source_camera_pose_initializer_applied',
            report=absolute_camera_initializer,q_sha256=qtoken(q))),flush=True)
    if opt.reinitialize_invalid_qualified_source_poses:
        from structured_programme_source_candidate.absolute_camera_pose_initializer import reinitialize_invalid_qualified_source_views
        q,qualified_pose_initializer=reinitialize_invalid_qualified_source_views(state,q,protected_actual_pts=(230024192,))
        absolute_camera_initializer=qualified_pose_initializer if absolute_camera_initializer is None else dict(
            explicit=absolute_camera_initializer,qualified_finite_domain=qualified_pose_initializer)
        continued_line_search_fraction=1.;inherited_adaptive_line_search=False
        write(out/'explicit_original_finite_source_pose_initialization.json',qualified_pose_initializer)
        print(json.dumps(dict(stage='actual_all_original_finite_source_domains_restored',
            report=qualified_pose_initializer,q_sha256=qtoken(q))),flush=True)
    if pivot_outside_constraints is not None:
        # Stage initialization can temporarily start outside the new domain;
        # the actual full candidate must now satisfy the original hard guard.
        validate_mounts(state,q)
    # Existing component factories evaluate their expanded physical core.
    # The solver/checkpoint chart preserves every old complete-state slot;
    # all component suffixes and derivative columns use this exact permutation.
    layout=CompleteProgrammeParameterLayout(source.count-(geometry.parameter_count-66),geometry.parameter_count-66)
    evaluation_world_indices=world_indices.copy()
    q=layout.canonical_q(q);source=CanonicalCompleteProgrammeSource(source,layout)
    world_indices=layout.canonical_indices(evaluation_world_indices)
    evaluation_core=source.evaluation_core
    evaluation_q=source.evaluation_q
    write(out/'matrix_storage_policy.json',dict(
        persistent_initial_linearization=bool(opt.save_initial_linearization_cache),
        persistent_following_linearizations=False,
        default_full_matrix_files=0,
        explicit_initial_cache_maximum_per_run=1,
        transient_native_replay_lifetime='One source J derivative model in RAM across accepted steps'if opt.reuse_source_jacobian else'One exact current linearization in memory; released before trial costs',
        complete_numeric_checkpoints_preserved=True,
        source_residuals_profiles_masks_scales_and_true_cost_acceptance_unchanged=True,
        exact_current_J_every_iteration=not opt.reuse_source_jacobian))
    runtime=out/'actual_source_blocks';runtime.mkdir();os.environ['RINKSPACE_SOURCE_RUNTIME_DIR']=str(runtime)
    geometry_fd_pool=None
    if opt.geometry_fd_workers>1:
        from calibration.solver_runtime.absolute_geometry_fd_pool_candidate import freeze_private_material_owner,AbsoluteGeometryGroupPool
        material_owner=freeze_private_material_owner(state.geometry,
            [group for block in state.blocks for group in block.groups],out/'private_absolute_material_owner.pkl')
        geometry_fd_pool=AbsoluteGeometryGroupPool(material_owner,workers=opt.geometry_fd_workers,
            max_inflight=opt.geometry_fd_workers,groups_per_job=16,parent_geometry=state.geometry)
        for block in state.blocks:block._absolute_geometry_fd_pool=geometry_fd_pool
        print(json.dumps(dict(stage='private_absolute_geometry_FD_pool_ready',workers=opt.geometry_fd_workers,
            material_owner=material_owner,whole_programme_bundle_transmitted=False)),flush=True)
    if strict_source is not None:
        from calibration.solver_runtime.strict_dataset_execution_runtime import install_strict_dataset_execution
        strict_execution=install_strict_dataset_execution(strict_dataset_factor)
        write(out/'strict_dataset_execution_receipt.json',strict_execution)
    def release_finished_geometry_fd_stage():
        if geometry_fd_pool is not None:
            result=geometry_fd_pool.release_idle()
            print(json.dumps(dict(stage='completed_programme_J_idle_FD_pool_released',result=result,
                next_fresh_J_workers=opt.geometry_fd_workers)),flush=True)
    def release_finished_pattern_stage():
        if current_pattern_pool is not None:
            result=current_pattern_pool.release_idle()
            print(json.dumps(dict(stage='completed_pattern_idle_pool_released',result=result,
                next_pattern_workers=4)),flush=True)
    raw.after_programme_jacobian=release_finished_geometry_fd_stage
    raw.after_shared_pattern_evaluation=release_finished_pattern_stage
    if raw.actors is not None:raw.actors.runtime_report_path=out/'source_runtime_memory_live.json'
    if not np.array_equal(base.quadratic,template['quadratic'])or not np.array_equal(base.scales,template['scales']):raise ValueError('Original complete source mask/scales changed')
    shared=layout.canonical_indices(np.r_[cap['shared_schur_indices'],base.slots])
    if opt.additional_board_circle_source is not None:
        shared=np.r_[shared,source.circle_indices]
        if len(np.unique(shared))!=len(shared):raise ValueError('New shared circle indices overlap original Schur columns')
    if placements is not None:
        shared=np.r_[shared,source.placement_indices]
        if len(np.unique(shared))!=len(shared):raise ValueError('Independent placement indices overlap original Schur columns')
    truth_restoration_limits={}
    truth_restoration_uncertainties={}
    fixed_seed_envelopes={}
    restored_truth_state=None
    def truth_violation_buffer(factor,value,*,family):
        evaluated=evaluation_q(value)
        if family=='confirmed':
            _,_,_,groups=factor._carriers(evaluated)
            return np.asarray([r['maximum_violation_px']for r in groups],float)
        rows=factor.queries(evaluated)
        values=np.asarray([abs(row['query']['residual']).max()for row in rows],float)
        if family=='native':
            full=factor.queries(evaluated,full_stroke=True)
            values=np.maximum(values,[abs(row['query']['residual']).max()for row in full])
        return values
    def truth_guard(factor,value,*,family):
        if not opt.joint_source_truth_restoration:
            return factor.guard(evaluation_q(value))
        violation=truth_violation_buffer(factor,value,family=family)
        if family not in truth_restoration_limits:
            if qtoken(value)!=qtoken(q):raise ValueError('Only the complete restored source state may establish initial truth recovery limits')
            if family=='confirmed':
                uncertainty=np.asarray([item['radius'].min()for item in factor.records],float)
            else:
                policy=json.loads(opt.practical_human_location_envelope.read_text(encoding='utf8'))
                uncertainty=np.asarray([r.get('engineering_uncorrelated_location_scale_px',r['normal_expansion_px'])for r in policy['records']if r['family']==family],float)
            if uncertainty.shape!=violation.shape or not np.isfinite(uncertainty).all() or np.any(uncertainty<0.):raise ValueError('Each restored source group needs its declared finite nonnegative practical engineering uncertainty')
            truth_restoration_uncertainties[family]=uncertainty
            # Bounds already include the drawing/registration engineering
            # support. A feasible group stays inside that original support;
            # only an actually violated group needs temporary restoration room.
            initial_limits=np.where(violation<=1e-8,0.,violation+uncertainty)
            if restored_truth_state is not None:
                saved=restored_truth_state['families'][family]
                limits=np.asarray(saved['limits'],float)
                if not np.array_equal(np.asarray(saved['uncertainties'],float),uncertainty):
                    raise ValueError('Truth restoration uncertainty differs: '+family)
                if not fixed_seed_policy:
                    if np.any(limits>initial_limits):raise ValueError('Truth restoration state would expand default limits: '+family)
                    initial_limits=limits.copy()
                elif np.any(violation>limits+1e-8):
                    raise ValueError('Fixed envelope seed is not admitted by its authenticated parent truth state: '+family)
            if fixed_seed_policy:
                from calibration.solver_runtime.fixed_seed_truth_envelope import make_fixed_seed_envelope
                resumed_envelope=None if restored_truth_state is None else (restored_truth_state.get('fixed_seed_envelopes')or{}).get(family)
                seed_uncertainty=uncertainty.copy()
                group_ids=truth_restoration_group_ids(factor,family)
                if family=='native':
                    for index,identity in enumerate(group_ids):
                        if identity in seed_engineering_scale_overrides:seed_uncertainty[index]=seed_engineering_scale_overrides[identity]
                initial_limits,seed_envelope=make_fixed_seed_envelope(violation,seed_uncertainty,
                    seed_q_sha256=qtoken(q),group_ids=truth_restoration_group_ids(factor,family),resumed=resumed_envelope)
                if resumed_envelope is not None and not np.array_equal(initial_limits,limits):
                    raise ValueError('Continued fixed seed truth limits differ from their original envelope: '+family)
                seed_envelope['original_recorded_engineering_uncertainties_px']=uncertainty.tolist()
                if resumed_envelope is None:
                    seed_envelope['existing_source_engineering_scale_override']=seed_engineering_scale_source
                fixed_seed_envelopes[family]=seed_envelope
                write(out/'actual_fixed_seed_truth_envelopes.json',dict(
                    complete_initial_q_sha256=qtoken(q),progress_policy='fixed_seed_envelope',families=fixed_seed_envelopes,
                    source_scale_override=seed_engineering_scale_source,historical_group_minimum_limits_preserved=False,
                    original_source_pixels_supports_sigmas_and_physical_domains_unchanged=True,accuracy_accepted=False))
            truth_restoration_limits[family]=initial_limits
        allowed=truth_restoration_limits[family]
        if violation.shape!=allowed.shape or np.any(violation>allowed+1e-8):
            increase=float(np.max(violation-allowed))if violation.shape==allowed.shape else float('inf')
            detail=''
            if violation.shape==allowed.shape:
                index=int(np.argmax(violation-allowed))
                identity=truth_restoration_group_ids(factor,family)[index]
                detail=f'; group_index={index}, group_id={identity}, violation={violation[index]:.12g}, allowed={allowed[index]:.12g}'
            raise ValueError(f'Joint full-source truth restoration would worsen {family} group beyond practical uncertainty by {increase:.12g} px{detail}')
        return True
    human_truth=None;wide_human_truth=None
    if opt.direct_human_truth_authority is not None:
        from structured_programme_source_candidate.human_stroke_truth import HumanStrokeTruth,WideStrokeTruth
        human_truth=HumanStrokeTruth(state,opt.direct_human_truth_authority,source.native_board_factor,source.native_ice_factor,parameter_count=len(q))
        human_truth.bind_full_stroke_guard(opt.direct_human_full_stroke_guard)
        human_truth.bind_stroke_cap_support(opt.direct_human_stroke_cap_support)
        wide_human_truth=WideStrokeTruth(factor,opt.original_wide_human_truth_authority,len(q),layout.canonical_to_evaluation[source.camera_indices])
        human_truth.bind_practical_location_envelope(opt.practical_human_location_envelope)
        wide_human_truth.bind_practical_location_envelope(opt.practical_human_location_envelope)
        shared=np.unique(np.r_[shared,layout.canonical_indices(human_truth.active),layout.canonical_indices(wide_human_truth.active)])
        original_complete_source_guard=source.guard
        def complete_source_and_direct_human_guard(value):
            original_complete_source_guard(value)
            truth_guard(human_truth,value,family='native');truth_guard(wide_human_truth,value,family='wide')
            return True
        source.guard=complete_source_and_direct_human_guard
    confirmed_truth=None
    if opt.user_confirmed_source_truth_authority is not None:
        from structured_programme_source_candidate.human_stroke_truth import UserConfirmedSourceTruth
        confirmed_truth=UserConfirmedSourceTruth(state,opt.user_confirmed_source_truth_authority,
            source.native_board_factor,source.native_ice_factor,parameter_count=len(q),
            circle_factor=getattr(source,'circle_factor',None),material_factor=getattr(source,'native_board_material_point_factor',None),
            placement_source=source.base if placements is not None else None)
        shared=np.unique(np.r_[shared,layout.canonical_indices(confirmed_truth.active)])
        original_source_and_human_guard=source.guard
        def complete_source_and_confirmed_native_guard(value):
            original_source_and_human_guard(value)
            truth_guard(confirmed_truth,value,family='confirmed')
            return True
        source.guard=complete_source_and_confirmed_native_guard
    pivot_shared_linear_constraints=shared_linear_constraints
    if opt.direct_human_centre_target and strict_source is None:
        if human_truth is None:raise ValueError('Direct manual centre targets require their actual human support guards')
        from structured_programme_source_candidate.human_stroke_truth import SourceWithDirectManualCentres
        source=SourceWithDirectManualCentres(source,state,human_truth,opt.additional_board_source,opt.additional_ice_curves,layout)
    trial_feasibility_recovery=None
    if pivot_outside_constraints is not None:
        def recover_pivot_trial(parent,proposed):
            evaluated=evaluation_q(proposed);domain=pivot_outside_domain(state,evaluated)
            report=None;corrected=evaluated
            if not domain['valid']:
                corrected,report=reinitialize_invalid_pivots_outside(state,evaluated,require_coordinate_fd_margin=False)
                changed=np.flatnonzero(layout.canonical_q(corrected)!=proposed)
                allowed={k for entry in pivot_outside_constraints['by_installation'].values()
                    for k in entry['pivot_parameter_indices'][:2]}
                if not set(changed.tolist()).issubset(allowed):
                    raise ValueError('Trial pivot feasibility recovery changed a nonpivot parameter')
                report.update(application='explicit numerical nonlinear trial feasibility recovery',
                    accepted_parent_state_unchanged=True,new_source_or_physical_prior=False,
                    actual_trial_source_cost_and_all_nonlinear_guards_required=True)
            from structured_programme_source_candidate.current_print_roundoff_recovery import recover_ad17_print_roundoff
            world,recovery=recover_ad17_print_roundoff(state.geometry.current,corrected[evaluation_world_indices])
            if recovery is not None:
                corrected=corrected.copy();corrected[evaluation_world_indices]=world
                if report is None:report=dict(schema='rinkspace.existing-domain-numerical-trial-recovery.v1',
                    accepted_parent_state_unchanged=True,new_source_or_physical_prior=False,
                    actual_trial_source_cost_and_all_nonlinear_guards_required=True)
                report['Ad17_print_roundoff_recovery']=recovery
            final=proposed if report is None else layout.canonical_q(corrected)
            if effective_curvature_compensation:
                from calibration.solver_runtime.physical_signed_margin_SOC import require_identity_trial_recovery
                require_identity_trial_recovery(proposed,final)
            return dict(q=final,report=report)
        trial_feasibility_recovery=recover_pivot_trial
    priority_restoration_domain={}
    def priority_face_map(family,factor,response,offset):
        if 'row_carrier_projection'in response:
            ids=np.asarray(response['row_group_index'],int);added=np.asarray(response['row_limit_expansion'],float)
            limits=truth_restoration_limits[family]
            if (ids.ndim!=1 or added.shape!=ids.shape or len(ids)!=response['row_carrier_projection'].shape[0]
                or np.any(ids<0)or np.any(ids>=len(limits))or not np.array_equal(added,limits[ids])):
                raise ValueError('Exact numerical finite-station face/group ownership required')
            return ids+offset,added
        if family=='native':
            records=factor.full_stroke_records
            sizes=[2*len(row['uv'])for row in records]
        elif family=='wide':
            records=factor.records
            sizes=[2*len(row['uv'])for row in records]
        else:
            records=factor.records
            sizes=[item.get('carrier_size',len(item['row']['uv'])*(2 if item['normals']is not None else 1))for item in records]
        group=np.repeat(np.arange(len(sizes),dtype=int)+offset,sizes)
        expansion=np.repeat(truth_restoration_limits[family],sizes)
        lower_group=group.copy();lower_expansion=expansion.copy()
        if family=='confirmed':
            cursor=0
            for item,size in zip(records,sizes):
                if not item.get('expand_lower_faces',item['normals']is not None):
                    lower_group[cursor:cursor+size]=-1
                    lower_expansion[cursor:cursor+size]=0.
                cursor+=size
        mask=np.asarray(response['retained_face_mask'],bool)
        if mask.shape!=(2*len(group),):raise ValueError('Exact original truth face/group mapping required')
        return np.r_[lower_group,group][mask],np.r_[lower_expansion,expansion][mask]
    def complete_truth_violations(value):
        values=[];labels=[]
        for family,factor in [('native',human_truth),('wide',wide_human_truth),('confirmed',confirmed_truth)]:
            if factor is not None:
                part=truth_violation_buffer(factor,value,family=family)
                values.append(part);labels.extend((family,index)for index in range(len(part)))
        return np.concatenate(values),labels
    def source_trial_acceptance(parent,trial,*,before_cost,after_cost,default_source_cost_accepts,trial_context):
        from structured_programme_source_candidate.feasibility_priority_restoration import actual_restoration_trial_decision
        previous,labels=complete_truth_violations(parent)
        if fixed_seed_policy:
            from calibration.solver_runtime.fixed_seed_truth_envelope import fixed_seed_trial_decision
            actual,trial_labels=complete_truth_violations(trial)
            if labels!=trial_labels:raise ValueError('All source truth group identities must remain unchanged')
            family_order=list(dict.fromkeys(family for family,_ in labels))
            allowed=np.concatenate([truth_restoration_limits[family]for family in family_order])
            return fixed_seed_trial_decision(previous,actual,allowed,before_cost=before_cost,after_cost=after_cost,
                default_source_cost_accepts=default_source_cost_accepts)
        if not np.any(previous>1e-8):
            return dict(accept=default_source_cost_accepts,mode='source_objective')
        if priority_restoration_domain.get('q_sha256')!=qtoken(parent):
            raise ValueError('Truth restoration trial does not match its complete parent domain')
        actual,trial_labels=complete_truth_violations(trial)
        if labels!=trial_labels:raise ValueError('All source truth group identities must remain unchanged')
        alpha=trial_context.get('line_search_fraction',trial_context.get('alpha'))
        result=actual_restoration_trial_decision(previous,actual,
            common_fraction=priority_restoration_domain['common_fraction'],step_fraction=alpha,
            physical_guard_passed=True,previous_source_cost=before_cost,trial_source_cost=after_cost)
        return dict(accept=result['accepted'],mode='truth_restoration',report=result,rejection_reason=result['reason'])
    def shared_linear_constraints(value,selected):
        from structured_programme_source_candidate.source_step_curvature_tightening import constraints_on_declared_free_chart
        # These are tangents of already enforced physical domains, not source
        # observations. All nonlinear source/print/pivot guards still run.
        domain=state.geometry.current.print_vertical_domain_step_constraints(value[world_indices])
        world_matrix=np.asarray(domain['matrix'],float)
        selected=np.asarray(selected,int)
        if world_matrix.ndim!=2 or world_matrix.shape[1]!=len(world_indices):
            raise ValueError('Complete physical print domain column inventory changed')
        reduced=np.zeros((len(world_matrix),len(selected)))
        mapping={int(column):position for position,column in enumerate(selected)}
        for world_column,column in enumerate(world_indices):
            if int(column)not in mapping:
                if np.any(world_matrix[:,world_column]):raise ValueError('Print vertical domain depends on an omitted shared world column')
                continue
            reduced[:,mapping[int(column)]]=world_matrix[:,world_column]
        lower=np.asarray(domain['lower'],float)
        circle_domain=None
        if opt.additional_board_circle_source is not None:
            circle_domain=circle_height_step_constraints(source.circle_factor,evaluation_q(value),
                layout.canonical_to_evaluation[selected])
            reduced=np.vstack((reduced,circle_domain['matrix']));lower=np.r_[lower,circle_domain['lower']]
        previous=None if pivot_shared_linear_constraints is None else pivot_shared_linear_constraints(evaluation_q(value),layout.canonical_to_evaluation[selected])
        if previous is not None:
            reduced=np.vstack((previous['matrix'],reduced));lower=np.r_[previous['lower'],lower]
        truth_report=None
        curvature_responses={}
        if human_truth is not None:
            key=(qtoken(value),np.asarray(selected,int).tobytes())
            if hard_truth_step_cache.get('key')!=key:
                evaluated=evaluation_q(value);columns=layout.canonical_to_evaluation[selected]
                fixed_evaluation=layout.canonical_to_evaluation[np.asarray(fixed,int)]
                native_constraints=constraints_on_declared_free_chart
                if opt.finite_rectangle_station_model:
                    from calibration.solver_runtime.finite_rectangle_station_step_constraints import finite_rectangle_station_constraints
                    native_constraints=finite_rectangle_station_constraints
                native=native_constraints(human_truth,evaluated,columns,fixed_evaluation,full_stroke=True,
                    group_limits=truth_restoration_limits.get('native')if opt.joint_source_truth_restoration else None,
                    endpoint_inward_restoration=effective_original_step_bounds,
                    return_step_carrier_response=effective_curvature_compensation)
                wide=constraints_on_declared_free_chart(wide_human_truth,evaluated,columns,fixed_evaluation,
                    group_limits=truth_restoration_limits.get('wide')if opt.joint_source_truth_restoration else None,
                    return_step_carrier_response=effective_curvature_compensation)
                hard_truth_step_cache.clear();hard_truth_step_cache.update(key=key,
                    matrix=np.vstack((native['matrix'],wide['matrix'])),lower=np.r_[native['lower'],wide['lower']],
                    report=dict(native=native['report'],wide=wide['report']),
                    curvature_responses=dict(native=native.get('step_carrier_response'),wide=wide.get('step_carrier_response')),
                    native_rows=len(native['lower']))
                if effective_curvature_compensation:
                    from calibration.solver_runtime.native_truth_nominal_chart import capture_native_nominal_query_chart
                    chart=capture_native_nominal_query_chart(human_truth,evaluated)
                    native_response=hard_truth_step_cache['curvature_responses']['native']
                    count=native_response.get('original_native_carrier_count',len(native_response['nominal_carrier']))
                    if not np.array_equal(chart['nominal_carrier'],native_response['nominal_carrier'][:count]):
                        raise ValueError('Native constraint J and nominal query carrier charts differ')
                    native_response['native_nominal_query_chart']=chart
            if effective_curvature_compensation:
                cursor=len(lower)
                for family,count in [('native',hard_truth_step_cache['native_rows']),('wide',len(hard_truth_step_cache['lower'])-hard_truth_step_cache['native_rows'])]:
                    response=hard_truth_step_cache['curvature_responses'][family]
                    curvature_responses[family]=dict(response,row_start=cursor,row_stop=cursor+count);cursor+=count
            reduced=np.vstack((reduced,hard_truth_step_cache['matrix']));lower=np.r_[lower,hard_truth_step_cache['lower']]
            truth_report=hard_truth_step_cache['report']
        confirmed_report=None
        if confirmed_truth is not None:
            key=(qtoken(value),np.asarray(selected,int).tobytes())
            if confirmed_truth_step_cache.get('key')!=key:
                confirmed=constraints_on_declared_free_chart(confirmed_truth,evaluation_q(value),layout.canonical_to_evaluation[selected],
                    layout.canonical_to_evaluation[np.asarray(fixed,int)],
                    group_limits=truth_restoration_limits.get('confirmed')if opt.joint_source_truth_restoration else None,
                    return_step_carrier_response=effective_curvature_compensation)
                confirmed_truth_step_cache.clear();confirmed_truth_step_cache.update(key=key,**confirmed)
            if effective_curvature_compensation:
                curvature_responses['confirmed']=dict(confirmed_truth_step_cache['step_carrier_response'],row_start=len(lower),row_stop=len(lower)+len(confirmed_truth_step_cache['lower']))
            reduced=np.vstack((reduced,confirmed_truth_step_cache['matrix']))
            lower=np.r_[lower,confirmed_truth_step_cache['lower']]
            confirmed_report=confirmed_truth_step_cache['report']
        priority_report=None
        original_priority_lower=None
        if opt.truth_restoration_priority and not fixed_seed_policy:
            from structured_programme_source_candidate.feasibility_priority_restoration import contract_original_support_bounds
            ownership=np.full(len(lower),-1,dtype=int);expansion=np.zeros(len(lower))
            offset=0
            for family,factor in [('native',human_truth),('wide',wide_human_truth),('confirmed',confirmed_truth)]:
                if factor is None:continue
                response=curvature_responses[family]
                ids,added=priority_face_map(family,factor,response,offset)
                begin,end=response['row_start'],response['row_stop']
                if len(ids)!=end-begin:raise ValueError('Original source face block length changed')
                ownership[begin:end]=ids;expansion[begin:end]=added
                offset+=len(truth_restoration_limits[family])
            violations,labels=complete_truth_violations(value)
            step_lower=np.full(len(selected),-np.inf);step_upper=np.full(len(selected),np.inf)
            for column,interval in mount_parameter_bounds.items():
                position=mapping[int(column)]
                step_lower[position]=interval[0]-value[column];step_upper[position]=interval[1]-value[column]
            for column in pivot_height_columns:
                if value[column]<=1e-6:
                    position=mapping[int(column)];step_lower[position]=max(0.,step_lower[position])
            original_priority_lower=lower+expansion
            contracted=contract_original_support_bounds(step_lower,step_upper,reduced,original_priority_lower,ownership,violations)
            lower=contracted['lower'];priority_report=contracted['report']
            priority_restoration_domain.clear();priority_restoration_domain.update(
                q_sha256=qtoken(value),common_fraction=contracted.get('common_fraction'),
                current_original_violations=violations.tolist(),group_labels=labels,report=priority_report)
        if effective_curvature_compensation:
            # Freeze the existing parent's actual per-group target. Local
            # station faces are derivative models, not nonlinear guard values.
            demanded=(priority_report.get('demanded_group_violations_px')if priority_report is not None else None)
            offset=0
            for family,factor in [('native',human_truth),('wide',wide_human_truth),('confirmed',confirmed_truth)]:
                if factor is None:continue
                response=curvature_responses[family]
                records=factor.full_stroke_records if family=='native'else factor.records
                groups_count=len(records)
                target=(np.asarray(demanded[offset:offset+groups_count],float)if demanded is not None else
                    np.asarray(truth_restoration_limits.get(family,np.zeros(groups_count)),float).copy())
                if 'row_carrier_projection'in response:
                    group=np.asarray(response['row_group_index'],int).copy()
                else:
                    sizes=([item.get('carrier_size',len(item['row']['uv'])*(2 if item['normals']is not None else 1))for item in records]
                           if family=='confirmed'else[2*len(item['uv'])for item in records])
                    carrier_groups=np.repeat(np.arange(groups_count),sizes)
                    lower_group=carrier_groups.copy()
                    if family=='confirmed':
                        position=0
                        for item,size in zip(records,sizes):
                            if not item.get('expand_lower_faces',item['normals']is not None):lower_group[position:position+size]=-1
                            position+=size
                    group=np.r_[lower_group,carrier_groups][np.asarray(response['retained_face_mask'],bool)]
                response.update(dynamic_group_limits=target,dynamic_row_group_index=group,
                    dynamic_guard_comparison_tolerance_px=1e-8 if opt.joint_source_truth_restoration or family=='confirmed'else 0.,
                    dynamic_group_target_basis='same_parent_original_priority_demand'if demanded is not None else'same_parent_original_guard_cap')
                offset+=groups_count
            if demanded is not None and offset!=len(demanded):raise ValueError('Complete original priority group target ownership required')
        result=dict(matrix=reduced,lower=lower,curvature_responses=curvature_responses,report=dict(
            schema='rinkspace.existing-pivot-and-print-physical-step-domains.v1',
            pivot_domain=None if previous is None else previous.get('report'),
            print_vertical_domain=domain['report'],original_nonlinear_guards_retained=True,
            additional_circle_height_domain=None if circle_domain is None else circle_domain['report'],
            highest_direct_human_hard_truth=truth_report,old_soft_cost_cannot_trade_outside_human_stroke_support=True,
            highest_user_confirmed_native_sources=confirmed_report,
            new_source_rows_or_priors=0,truth_feasibility_priority=priority_report))
        if (effective_curvature_compensation and priority_report is not None
                and priority_report.get('restoration_required')):
            from calibration.solver_runtime.trial_fraction_source_domain import attach_trial_fraction_domain
            result=attach_trial_fraction_domain(result,
                original_constraint_lower=original_priority_lower,row_group_index=ownership,
                previous_group_violations=violations,common_fraction=contracted['common_fraction'])
        if fixed_seed_policy and effective_curvature_compensation:
            from calibration.solver_runtime.fixed_seed_truth_envelope import attach_fixed_seed_trial_fraction_domain
            result=attach_fixed_seed_trial_fraction_domain(result)
        if effective_curvature_compensation and previous is not None:
            from calibration.solver_runtime.physical_signed_margin_SOC import attach_original_signed_margin_rows
            rows=previous['report']['installations']
            result=attach_original_signed_margin_rows(result,row_indices=np.arange(len(rows),dtype=int),
                nominal_margins=[row['outward_margin_working_m']for row in rows],
                row_ids=[row['installation_id']for row in rows])
        return result
    shared_step_curvature_tightening=None
    if effective_curvature_compensation:
        def tighten_source_trial_faces(value,step,selected,domain):
            from structured_programme_source_candidate.source_step_curvature_tightening import measured_outward_step_tightening
            base_value=evaluation_q(value);trial_value=evaluation_q(value+step)
            try:
                physical_correction=None
                if 'original_signed_margin_rows'in domain:
                    from calibration.solver_runtime.physical_signed_margin_SOC import measured_signed_margin_step_tightening
                    from calibration.solver_runtime.optional_soc_runtime import observe_original_physical_SOC_measurement
                    physical=pivot_outside_domain(state,trial_value)['installations']
                    physical_correction,measurement=measured_signed_margin_step_tightening(
                        domain['matrix'],domain['lower'],step[selected],domain['original_signed_margin_rows'],
                        [row['outward_margin_working_m']for row in physical],
                        row_ids=[row['installation_id']for row in physical])
                    observe_original_physical_SOC_measurement(measurement)
                actual={}
                actual_groups={}
                if human_truth is not None:
                    native_response=domain['curvature_responses']['native']
                    from calibration.solver_runtime.native_truth_nominal_chart import native_nominal_query_carriers
                    original_carriers=native_nominal_query_carriers(human_truth,trial_value,native_response['native_nominal_query_chart'])
                    if 'finite_station_contexts'in native_response:
                        from calibration.solver_runtime.finite_rectangle_station_step_constraints import actual_station_model_carriers
                        actual['native']=actual_station_model_carriers(human_truth,trial_value,native_response,original_carriers=original_carriers)
                    else:actual['native']=original_carriers
                    actual_groups['native']=truth_violation_buffer(human_truth,value+step,family='native')
                if wide_human_truth is not None:
                    actual['wide']=wide_human_truth._carriers(trial_value)
                    actual_groups['wide']=truth_violation_buffer(wide_human_truth,value+step,family='wide')
                if confirmed_truth is not None:
                    actual['confirmed']=confirmed_truth._carriers(trial_value)[0]
                    actual_groups['confirmed']=truth_violation_buffer(confirmed_truth,value+step,family='confirmed')
                source_correction=measured_outward_step_tightening(domain['matrix'],domain['lower'],step[selected],domain['curvature_responses'],actual,
                    actual_group_violations=actual_groups)
                if physical_correction is None:return source_correction
                if source_correction is None:return physical_correction
                return dict(lower=np.maximum(source_correction['lower'],physical_correction['lower']),
                    report=dict(source_correction['report'],physical_signed_margin_SOC=physical_correction['report']))
            except ValueError:
                # Existing physical-domain backtracking remains authoritative.
                return None
            finally:
                if human_truth is not None:human_truth.queries(base_value)
                if wide_human_truth is not None:wide_human_truth.queries(base_value)
                if confirmed_truth is not None:confirmed_truth._carriers(base_value)
        shared_step_curvature_tightening=tighten_source_trial_faces
    source_physical_endpoint_event=None
    if human_truth is not None and opt.truth_restoration_priority:
        def locate_physical_endpoint_event(value,step,maximum_fraction):
            from structured_programme_source_candidate.source_step_curvature_tightening import physical_endpoint_branch_event
            if opt.save_complete_source_directions:
                directory=out/'actual_complete_source_directions';directory.mkdir(exist_ok=True)
                direction_sha=qtoken(step);identity_sha=qtoken(value)
                path=directory/(identity_sha+'_'+direction_sha+'.npz')
                if not path.exists():
                    np.savez(path,q=np.asarray(value),step=np.asarray(step),
                        canonical_to_evaluation=layout.canonical_to_evaluation,
                        frozen_profile_token=np.asarray(_frozen_profile_token(source.updater)),
                        source_input_manifest_path=np.asarray(str(out/'input_manifest.json')),
                        source_input_manifest_sha256=np.asarray(sha(out/'input_manifest.json')),
                        complete_q_sha256=np.asarray(identity_sha),complete_step_sha256=np.asarray(direction_sha),
                        requested_shared_indices=shared,fixed_parameter_indices=fixed)
            # The complete layout is a pure permutation. Permute the exact
            # displacement directly; subtracting two rounded states moves a
            # near-boundary event to the wrong side of its physical endpoint.
            return physical_endpoint_branch_event(human_truth,evaluation_q(value),evaluation_q(step),maximum_fraction)
        source_physical_endpoint_event=locate_physical_endpoint_event
    hard_truth_step_cache={}
    confirmed_truth_step_cache={}
    principal_point_prior=None
    if opt.principal_point_working_prior:
        from calibration.solver_runtime.current_principal_point_working_prior import wrap_current_principal_point_working_prior
        source=wrap_current_principal_point_working_prior(source,state)
        principal_point_prior=source.principal_point_working_prior
        pp_values,pp_J=principal_point_prior.evaluate(q,True)
        write(out/'actual_principal_point_working_prior_initial.json',dict(
            contract=principal_point_prior.contract,complete_q_sha256=qtoken(q),
            actual_native_principal_points_px=principal_point_prior.native_principal_points(q).tolist(),
            actual_normalized_residual=pp_values.tolist(),
            actual_J_shape=list(pp_J.shape),actual_J_nonzeros=int(pp_J.nnz),
            actual_J_columns=pp_J.indices.tolist(),actual_J_values=pp_J.data.tolist(),
            original_scalar_prefix=source.prior_start,new_scalar_rows=6,
            prior_is_engineering_hypothesis_not_source_pixel_truth=True))
    initial_damping=.001;damping_continuation=dict(inherited=False,value=initial_damping)
    if continued_damping is not None and opt.initial_damping is None:
        authority=continued_damping_identity['identity'].get('input_manifest')
        if authority is None:raise ValueError('Numerical initializer has no optimizer source manifest; use explicit --initial-damping for the declared new target')
        seed_manifest=Path(authority['path'])
        if sha(seed_manifest)!=authority['sha256']:raise ValueError('Continued optimizer source manifest changed')
        seed_manifest_data=json.loads(seed_manifest.read_text(encoding='utf8'))
        current_packet=None if opt.additional_ice_curves is None else source.native_ice_factor.contract
        current_board_packet=None if opt.additional_board_source is None else source.native_board_factor.contract
        current_material_point_packet=None if opt.additional_board_material_point_source is None else source.native_board_material_point_factor.contract
        current_circle_packet=None if opt.additional_board_circle_source is None else source.circle_factor.contract
        as_json=lambda v:json.loads(json.dumps(v,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x)))
        same_source=(np.array_equal(continued_damping_identity['quadratic'],source.quadratic)and
            np.array_equal(continued_damping_identity['scales'],source.scales)and
            seed_manifest_data.get('additional_real_native_ice_curves')==as_json(current_packet) and
            seed_manifest_data.get('additional_real_native_board_edges')==as_json(current_board_packet) and
            seed_manifest_data.get('additional_real_native_board_material_points')==as_json(current_material_point_packet) and
            seed_manifest_data.get('additional_real_native_board_circles')==as_json(current_circle_packet) and
            seed_manifest_data.get('independent_board_pattern_placements')==as_json(None if placements is None else placements.placement_contract) and
            seed_manifest_data.get('additional_real_native_persistent_ice_long_baseline')==as_json(None if persistent_ice is None else persistent_ice.contract) and
            seed_manifest_data.get('original_source_role_retirement')==as_json(None if retirement is None else retirement.contract) and
            seed_manifest_data.get('original_source_point_qualification')==as_json(None if original_point_qualification is None else original_point_qualification.contract) and
            seed_manifest_data.get('static_motion_source_role_retirement')==as_json(None if motion_retirement is None else motion_retirement.contract) and
            seed_manifest_data.get('original_motion_far_board_role_correction')==as_json(far_board_role_correction) and
            seed_manifest_data.get('original_motion_input_compaction')==as_json(motion_input_compaction) and
            seed_manifest_data.get('original_board_curve_integration')==as_json(original_board_curve_integration) and
            seed_manifest_data.get('original_ice_curve_integration')==as_json(original_ice_curve_integration) and
            seed_manifest_data.get('original_wide_curve_integration')==as_json(original_wide_curve_integration) and
            seed_manifest_data.get('latest9a8c_reference18_preserved')==as_json(source.reference_factor.contract) and
            seed_manifest_data.get('camera_mount_working_constraints')==as_json(mount_working_constraints) and
            seed_manifest_data.get('camera_pivot_exterior_constraints')==as_json(pivot_outside_constraints) and
            seed_manifest_data.get('original_ring_source_qualification')==as_json(getattr(state,'original_ring_source_qualification',None)) and
            seed_manifest_data.get('original_ice_pixel_correction')==as_json(ice_pixel_correction) and
            seed_manifest_data.get('original_measured_transition_locations')==as_json(original_measured_locations) and
            seed_manifest_data.get('additional_native_static_board_tracks')==as_json(None if board_tracks is None else board_tracks.contract) and
            float(seed_manifest_data.get('roll_prior_component_sigma_degrees',.5))==opt.roll_prior_sigma_degrees and
            seed_manifest_data.get('user_trace_sha256')==sha(TRACE))
        physical_initialization_changed=any(report is not None and report.get('changed_parameter_indices')
            for report in (mount_reinitialization,pivot_reinitialization,pose_feasibility_initializer,absolute_camera_initializer))
        if strict_source is not None:
            from calibration.solver_runtime.strict_optimizer_resume import same_strict_optimizer_target
            same_source=same_strict_optimizer_target(seed_manifest_data,strict_source.contract,
                old_quadratic=continued_damping_identity['quadratic'],quadratic=source.quadratic,
                old_scales=continued_damping_identity['scales'],scales=source.scales,
                protected_fields=dict(
                    full_parameters=len(q),complete_parameter_layout=layout.manifest(),
                    geometry_world_indices=world_indices.tolist(),physical_geometry_initializer=physical_geometry_initializer,
                    latest9a8c_reference18_preserved=source.reference_factor.contract,
                    correlated_roll_prior=base.contract,
                    principal_point_engineering_prior=None if principal_point_prior is None else principal_point_prior.contract,
                    camera_mount_working_constraints=mount_working_constraints,
                    camera_pivot_exterior_constraints=pivot_outside_constraints,
                    roll_prior_component_sigma_degrees=opt.roll_prior_sigma_degrees,user_trace_sha256=sha(TRACE)),
                full_truth_resume_required=bool(opt.joint_source_truth_restoration and opt.truth_restoration_state is not None),
                physical_initialization_changed=physical_initialization_changed)
        same_source=bool(same_source and not physical_initialization_changed)
        if same_source:
            initial_damping=continued_damping;damping_continuation=dict(inherited=True,value=initial_damping,
                same_complete_source_masks_scales_native_packet_reference18_and_user_trace=strict_source is None,
                same_strict_active_dataset_motion_operator_priors_and_trace=strict_source is not None,
                protected_full_truth_resume_required_before_solver=strict_source is not None,
                initial_line_search_fraction=continued_line_search_fraction,seed_manifest=authority)
        else:
            continued_line_search_fraction=1.
            damping_continuation=dict(inherited=False,value=initial_damping,reason='Changed source objective, camera initializer or declared physical working domain requires fresh damping and step initialization')
    if opt.initial_damping is not None:
        initial_damping=float(opt.initial_damping)
        continued_line_search_fraction=1.;inherited_adaptive_line_search=False
        damping_continuation=dict(inherited=False,value=initial_damping,explicit_numerical_initialization=True,
            latest_complete_seed_damping=continued_damping,
            reason='Explicit initialization for declared target revision; numerical state and profiles retained')
    elif opt.adaptive_source_backtracking and not inherited_adaptive_line_search:
        initial_damping=.001
        damping_continuation=dict(inherited=False,value=initial_damping,
            latest_complete_seed_damping=continued_damping,
            reason='One-time migration from reset-to-one backtracking to learned-step numerical policy; complete parent state retained')
    if not np.isfinite(continued_line_search_fraction)or not 0<continued_line_search_fraction<=1:
        raise ValueError('Finite saved whole-source line search fraction in (0, 1] required')
    # This is optimizer provenance, outside the source-math/cache identity.
    # Checkpoints still hash the complete manifest and actual implementation.
    damping_continuation['numerical_runtime_options']=dict(
        schur_view_lm_ratio=opt.schur_view_lm_ratio,
        bounded_motion_nuisance=opt.bounded_motion_nuisance,
        schur_backtracking=True,schur_backtracking_aware_damping=True,
        original_source_objective_and_guards_unchanged=True,
        affects_next_numerical_step_only=True)
    print(json.dumps(dict(stage='continued_optimizer_damping_ready',**damping_continuation)),flush=True)
    native_base_curve_integration=None
    if opt.native_base_curve_integration_policy is not None and strict_source is None:
        from calibration.solver_runtime.curve_source_grouping import wrap_native_base_curve_integration
        source=wrap_native_base_curve_integration(source,opt.native_base_curve_integration_policy)
        native_base_curve_integration=source.native_base_curve_integration
    from structured_programme_source_candidate.human_stroke_truth import human_support_operator_contract
    human_support_operator=human_support_operator_contract()
    source.contract=dict(source.contract,human_truth_support_operator=human_support_operator)
    if original_measured_locations is not None:
        source.contract=dict(source.contract,original_measured_transition_locations=original_measured_locations)
    fixed=[]if state.external is None else[state.external_start+4*state.external.index[p]+c for p in state.external.held_pts for c in range(4)]
    if opt.shared_hockey_curling_axis:
        # Preserve the historical complete-state slot, but remove its inactive
        # duplicate direction from the actual optimization. All physical
        # producers and their J bind the single curling/common yaw owner.
        fixed.append(int(layout.canonical_indices([state.geometry_parameter_indices[2]])[0]))
    manifest=dict(schema='rinkspace.complete-original-programme-current-named66-user-correction.v1',
        additional_native_static_board_tracks=None if board_tracks is None else board_tracks.contract,
        roll_prior_component_sigma_degrees=opt.roll_prior_sigma_degrees,
        correlated_roll_prior=base.contract,
        parent_b691_sha256=sha(DEFAULT_SEED),camera_seed=None if opt.camera_seed is None else dict(path=str(opt.camera_seed),sha256=sha(opt.camera_seed)),
        original_camera_replacement_max_absolute=camera_error,geometry_world_indices=world_indices.tolist(),
        numerical_current_world_inherited=inherited_current_world is not None,
        absolute_source_camera_initializer=absolute_camera_initializer,
        original_complete_parameter_prefix_count=layout.original_complete_count,
        complete_parameter_layout=layout.manifest(),physical_geometry_initializer=physical_geometry_initializer,
        numerical_independent_circle_material_suffix_inherited=inherited_circle_suffix is not None,
        numerical_independent_pattern_placement_suffix_inherited=inherited_pattern_suffix is not None,
        independent_pattern_placement_parameter_indices=[]if placements is None else source.placement_indices.tolist(),
        independent_board_pattern_placements=None if placements is None else placements.placement_contract,
        additional_real_native_persistent_ice_long_baseline=None if persistent_ice is None else persistent_ice.contract,
        independent_pattern_new_slot_initializer=None if placements is None else dict(
            parameter_values=q[source.placement_indices].tolist(),complete_previous_q_prefix_byte_exact=True,
            shared_template_size_and_radius_unchanged=True,initializer_is_not_observation=True),
        independent_circle_material_parameter_indices=[]if opt.additional_board_circle_source is None else source.circle_indices.tolist(),
        original_wide_camera_indices_after_suffix=source.camera_indices.tolist(),
        original_wide_camera_indices_byte_exact=np.array_equal(source.camera_indices,layout.canonical_indices(camera_indices_before_circle)),
        fixed_geometry_initializer_auxiliary_preserved=True,
        optimizer_damping_continuation=damping_continuation,
        actual_exposures=8503,native_presentations=10123,original_motion_edge_counts=[len(b.bundle.a)for b in state.blocks],
        original_motion_rows=[b.bundle.motion_rows for b in state.blocks],native_backend_requested={k:v for k,v in os.environ.items()if k.startswith('RINKSPACE_')},
        original_source_masks_scales_and_roles_exact=retirement is None and motion_retirement is None and far_board_role_correction is None and original_point_qualification is None,source_rows=len(source.scales),full_parameters=len(q),shared_indices=shared.tolist(),
        original_source_masks_scales_and_historical_raw_roles_exact=True,
        original_source_role_retirement=None if retirement is None else retirement.contract,
        static_motion_source_role_retirement=None if motion_retirement is None else motion_retirement.contract,
        original_motion_far_board_role_correction=far_board_role_correction,
        original_motion_input_compaction=motion_input_compaction,
        original_absolute_input_compaction=absolute_input_compaction,
        original_board_curve_integration=original_board_curve_integration,
        original_ice_curve_integration=original_ice_curve_integration,
        active_original_FIT_scalars_retired=(0 if retirement is None else 2)+(0 if motion_retirement is None else motion_retirement.contract['active_original_motion_FIT_scalars_retired'])+(0 if original_point_qualification is None else original_point_qualification.contract['active_FIT_scalars_retired']),
        latest9a8c_reference18_preserved=source.reference_factor.contract,
        hard_structural_relations_and_soft_engineering_references_distinguished=True,
        geometric60x30_objective_added=False,metric_conversion_applied_after_geometry=True,
        original_midpoint_row_now_redundant_structural_zero=True,user_trace_sha256=sha(TRACE),
        additional_real_native_ice_curves=None if opt.additional_ice_curves is None else source.native_ice_factor.contract,
        additional_real_native_board_edges=None if opt.additional_board_source is None else source.native_board_factor.contract,
        additional_real_native_board_material_points=None if opt.additional_board_material_point_source is None else source.native_board_material_point_factor.contract,
        additional_real_native_board_circles=None if opt.additional_board_circle_source is None else source.circle_factor.contract,
        exact_ice_material_view_installed=not opt.complete_scene_material_diagnostic,
        source_ice_material_cache_budget_bytes=0 if opt.complete_scene_material_diagnostic else 32*1024**2,
        current_geometry=contract,source_contract=source.contract,accuracy_accepted=False,full_video_generated=False)
    manifest['camera_mount_working_constraints']=mount_working_constraints
    if strict_source is not None:
        manifest['strict_source_dataset']=strict_source.contract
        manifest['legacy_pixel_objective_replaced']=True
        manifest['original_source_masks_scales_and_historical_raw_roles_exact']=False
    manifest['principal_point_engineering_prior']=None if principal_point_prior is None else principal_point_prior.contract
    manifest['native_base_curve_integration']=native_base_curve_integration
    if original_wide_curve_integration is not None:manifest['original_wide_curve_integration']=original_wide_curve_integration
    manifest['human_truth_support_operator']=human_support_operator
    manifest['explicit_mount_working_range_reinitialization']=mount_reinitialization
    manifest['camera_pivot_exterior_constraints']=pivot_outside_constraints
    manifest['explicit_pivot_outside_reinitialization']=pivot_reinitialization
    manifest['physical_domain_initial_q_sha256']=qtoken(q)
    manifest['initial_pivot_exterior_domain']=None if pivot_outside_constraints is None else pivot_outside_domain(state,evaluation_q(q))
    manifest['explicit_source_domain_pose_feasibility_initialization']=pose_feasibility_initializer
    manifest['original_ring_source_qualification']=getattr(state,'original_ring_source_qualification',None)
    manifest['original_ice_pixel_correction']=ice_pixel_correction
    if original_measured_locations is not None:
        manifest['original_measured_transition_locations']=original_measured_locations
    if original_point_qualification is not None:
        manifest['original_source_point_qualification']=original_point_qualification.contract
    manifest['physical_course_scope_source']=None if opt.physical_course_scope_source is None else dict(path=str(opt.physical_course_scope_source.resolve()),sha256=sha(opt.physical_course_scope_source))
    manifest['highest_direct_human_hard_truth']=None if human_truth is None else dict(native=human_truth.contract,wide=wide_human_truth.contract,
        full_complete_strokes_guard_every_actual_trial=True,all_human_halfspaces_in_shared_step=True,
        extra_independent_FIT_weight_rows=0,old_soft_objective_cannot_offset_human_violation=True)
    manifest['highest_user_confirmed_native_sources']=None if confirmed_truth is None else confirmed_truth.contract
    manifest['joint_source_truth_restoration']=dict(enabled=opt.joint_source_truth_restoration,
        explicit_source_revision_proof=None if opt.truth_restoration_source_revision_proof is None else dict(path=str(opt.truth_restoration_source_revision_proof.resolve()),sha256=sha(opt.truth_restoration_source_revision_proof)),
        explicit_source_role_proof=None if opt.truth_restoration_source_role_proof is None else dict(path=str(opt.truth_restoration_source_role_proof.resolve()),sha256=sha(opt.truth_restoration_source_role_proof)),
        explicit_source_append_proof=None if opt.truth_restoration_source_append_proof is None else dict(path=str(opt.truth_restoration_source_append_proof.resolve()),sha256=sha(opt.truth_restoration_source_append_proof)),
        legacy_original_step_bounds_requested=bool(opt.truth_restoration_original_step_bounds),
        legacy_source_face_curvature_compensation_requested=bool(opt.truth_restoration_curvature_compensation),
        original_support_bounds_in_step=effective_original_step_bounds,
        native_endpoint_inward_branch_in_step=effective_original_step_bounds,
        measured_source_face_curvature_compensation=effective_curvature_compensation,
        finite_rectangle_station_model=None if not opt.finite_rectangle_station_model else dict(
            numerical_step_only=True,original_source_and_nonlinear_support_operator_unchanged=True,
            actual_original_guards_and_full_trial_source_cost_required=True,
            implementation_sources=__import__('calibration.solver_runtime.finite_rectangle_station_step_constraints',fromlist=['station_model_implementation_sources']).station_model_implementation_sources()),
        progress_policy=opt.truth_restoration_progress_policy,
        seed_engineering_scale_source=seed_engineering_scale_source,
        truth_feasibility_priority=bool(opt.truth_restoration_priority and not fixed_seed_policy),
        step_group_bound_mode='fixed_seed_envelope'if fixed_seed_policy else('strict_current_violation_contraction'if opt.truth_restoration_priority else('restoration_limits'if opt.joint_source_truth_restoration else'original_support')),
        nonlinear_group_guard_mode='fixed_seed_envelope'if fixed_seed_policy else('restoration_ratchet'if opt.joint_source_truth_restoration else'original_support'),
        historical_group_minimum_limits_preserved=not fixed_seed_policy,
        fixed_seed_policy_implementation=None if not fixed_seed_policy else dict(
            path=str(Path(__import__('calibration.solver_runtime.fixed_seed_truth_envelope',fromlist=['POLICY']).__file__).resolve()),
            sha256=sha(Path(__import__('calibration.solver_runtime.fixed_seed_truth_envelope',fromlist=['POLICY']).__file__))),
        complete_initial_q_sha256=qtoken(q),full_original_source_objective_retained=True,
        human_only_phase_I=False,initial_violations_not_claimed_feasible=True,
        group_limits_ratcheted_with_existing_engineering_uncertainty=not fixed_seed_policy,
        feasible_groups_keep_original_practical_support=True,
        restoration_room_only_for_actually_violated_groups=True,
        restored_groups_latched_to_original_support=not fixed_seed_policy,
        subpixel_per_query_monotonicity_not_imposed=True,
        actual_practical_band_halfspaces_retained=True)
    manifest['actual_active_backend_contract']=active_backend_report
    manifest['source_worker_backend_installation']=SOURCE_WORKER_BACKEND_INSTALLATION
    manifest['preparation_only_capture_release']=preparation_capture_release
    manifest['absolute_geometry_fd_parallelism']=dict(requested_workers=opt.geometry_fd_workers,
        material_owner=None if geometry_fd_pool is None else geometry_fd_pool.material_owner,
        one_parent_source_graph=True,original_FD_width_material_operator_masks_scales_and_order_retained=True)
    manifest['weighted_gram_parallelism']=dict(requested_workers=opt.normal_workers,
        internal_contiguous_rows=True,floating_reduction_order_changed_up_to_roundoff=True)
    manifest['source_jacobian_reuse']=dict(enabled=opt.reuse_source_jacobian,
        optional_prediction_refresh_ratio=opt.source_jacobian_refresh_ratio,
        derivative_model='modified Newton with current residuals and lagged source J',
        current_profiles_Huber_weights_H_g_and_full_nonlinear_trial_cost_recomputed=True,
        refresh_on_invalid_prediction_or_rejected_lagged_iteration=True,
        numerical_convergence_requires_fresh_source_J=True,
        retained_matrix_copies_in_RAM=1,persistent_full_matrix_files=1 if opt.save_initial_linearization_cache else 0,
        persistent_matrices_are_one_explicit_sealed_same_state_cache=bool(opt.save_initial_linearization_cache))
    manifest['adaptive_source_backtracking']=dict(enabled=opt.adaptive_source_backtracking,
        initial_line_search_fraction=continued_line_search_fraction,
        fraction_inherited_from_adaptive_checkpoint=inherited_adaptive_line_search,
        true_nonlinear_cost_acceptance_unchanged=True)
    manifest['residual_source_parallelism']=dict(requested_workers=opt.residual_source_workers,
        native_threads_per_actor=opt.residual_native_threads,
        output_jacobian=False,source_admission_memory_reserve_bytes=0,
        fresh_source_J_uses_parent_and_geometry_FD_workers=True)
    manifest['current_geometry_pattern_parallelism']=dict(enabled=current_pattern_pool is not None,
        requested_workers=0 if current_pattern_pool is None else 4,
        independent_current_geometry_snapshots=True,original_context_order_and_FD_unchanged=True,
        complete_inventory_equivalence_evidence=str(BASE/'structured_programme_source_candidate/current_geometry_pattern_pool_full_inventory_check_20261005_v1.json'),
        evidence_sha256=sha(BASE/'structured_programme_source_candidate/current_geometry_pattern_pool_full_inventory_check_20261005_v1.json'))
    manifest['accepted_state_review_policy']=dict(enabled=opt.watch_native_evidence,
        new_FIT_mean_increase_stop_px=.01,F1_HELD_mean_increase_from_initial_stop_px=1.,
        last_F2_inner_right_original_outside_support_median_increase_from_initial_stop_px=.5,
        role='Engineering progress monitor; not user accuracy criteria, fitted constraint, covariance, or goal pause',
        every_accepted_q_profiles_scene_camera_reference_and_source_review_frozen=True,
        following_actual_source_normals_r_J_H_g_streamed_to_Data=False,
        following_linearizations_in_memory_only=True)
    conditional_view_dependencies=[layout.evaluation_to_canonical[np.asarray(factor.active,int)]
        for factor in (human_truth,wide_human_truth,confirmed_truth)if factor is not None]if opt.conditional_view_backtracking else[]
    conditional_view_protected=np.unique(np.concatenate(conditional_view_dependencies))if conditional_view_dependencies else np.empty(0,int)
    if opt.conditional_view_backtracking:
        if not np.isin(conditional_view_protected,shared).all():
            raise ValueError('Conditional view backtracking may not move any original hard-truth dependency outside the shared chart')
        manifest['adaptive_source_backtracking']['conditional_view_backtracking']=dict(enabled=True,
            protected_actual_truth_dependency_canonical_parameter_indices=conditional_view_protected.tolist(),
            all_protected_dependencies_in_original_shared_chart=True,
            original_source_cost_guards_profiles_rows_sigma_and_DOF_unchanged=True)
    from structured_programme_source_candidate.complete_same_state_replay import (
        build_persistent_source_replay_identity,seal_complete_same_state_cache,safe_same_state_replay_predicate)
    actual_shared_mask=np.isin(np.arange(len(q)),shared)
    # Keep the actual pending source identity if strict cache preparation fails.
    # This is diagnostic input, never a sealed matrix-reuse authorization.
    write(out/'actual_unsealed_source_semantic_manifest.json',manifest)
    def prepare_actual_cache_identity():
        actual_source_arrays,actual_source_records=actual_source_replay_inputs(state,source,factor)
        return build_persistent_source_replay_identity(manifest,
            fixed_parameter_indices=fixed,shared_parameter_mask=actual_shared_mask,
            source_input_arrays=actual_source_arrays,source_input_records=actual_source_records,
            implementation_files=[Path(__file__),Path(__file__).parents[1]/'runtime.py',
                Path(__file__).parents[1]/'config.py',Path(__file__).parents[1]/'__init__.py'])
    if opt.truth_restoration_reviewed_manual_append_audit is not None:
        if not opt.joint_source_truth_restoration or opt.truth_restoration_state is None:
            raise ValueError('Reviewed manual append requires the complete parent truth state and joint restoration')
        if any(getattr(opt,name,None)is not None for name in ('reuse_complete_linearization','reuse_source_blocks_from','reuse_wide_linearization')):
            raise ValueError('Reviewed source target revision requires fresh source J/profiles')
        manifest['joint_source_truth_restoration']['explicit_reviewed_manual_append_audit']=dict(path=str(opt.truth_restoration_reviewed_manual_append_audit.resolve()),sha256=sha(opt.truth_restoration_reviewed_manual_append_audit))
    if opt.save_initial_linearization_cache or opt.reuse_complete_linearization is not None or opt.reuse_source_blocks_from is not None:
        try:manifest['persistent_source_replay_identity']=prepare_actual_cache_identity()
        except Exception as error:
            # Cache reuse is never allowed to fall back. Optional cache creation
            # must not stop an independently guarded fresh full-source solve.
            if opt.reuse_complete_linearization is not None or opt.reuse_source_blocks_from is not None:raise
            opt.save_initial_linearization_cache=False
            unavailable=dict(status='UNAVAILABLE_NOT_REUSABLE',error_type=type(error).__name__,reason=str(error),
                original_UV_sigma_role_domain_and_native_source_guards_still_required=True,
                full_source_J_will_be_fresh=True,no_matrix_reuse_authorized=True)
            manifest['optional_persistent_cache_preflight']=unavailable
            manifest['source_jacobian_reuse']['persistent_full_matrix_files']=0
            manifest['source_jacobian_reuse']['persistent_matrices_are_one_explicit_sealed_same_state_cache']=False
            write(out/'actual_optional_cache_preflight_failure.json',unavailable)
            print(json.dumps(dict(stage='optional_persistent_cache_unavailable_fresh_solve_continues',**unavailable)),flush=True)
    write(out/'input_manifest.json',manifest)
    truth_factors=dict(native=human_truth,wide=wide_human_truth,confirmed=confirmed_truth)
    resume_truth_path=opt.truth_restoration_state
    if opt.truth_restoration_reviewed_manual_append_audit is not None:
        from structured_programme_source_candidate.reviewed_manual_append_restore import prepare_reviewed_manual_append_state
        def declared_truth_uncertainties(factor,family):
            if family=='confirmed':return np.asarray([item['radius'].min()for item in factor.records],float)
            policy=json.loads(opt.practical_human_location_envelope.read_text(encoding='utf8'))
            return np.asarray([r.get('engineering_uncorrelated_location_scale_px',r['normal_expansion_px'])for r in policy['records']if r['family']==family],float)
        resume_truth_path=prepare_reviewed_manual_append_state(opt.truth_restoration_reviewed_manual_append_audit,opt.truth_restoration_state,
            out/'input_manifest.json',manifest,q,truth_factors,group_ids=truth_restoration_group_ids,
            violation_buffer=truth_violation_buffer,uncertainties=declared_truth_uncertainties,
            source_target=truth_restoration_source_target,role_target=role_target_from_restored_manifest,write=write)
    if resume_truth_path is not None:
        if not opt.joint_source_truth_restoration:raise ValueError('Saved truth limits require the existing joint truth restoration mode')
        if strict_source is not None:
            from calibration.solver_runtime.strict_dataset_truth_resume import load_for_strict_revision
            restored_truth_state=load_for_strict_revision(resume_truth_path,q,manifest,truth_factors,
                archived_loader=load_truth_restoration_state,source_target=truth_restoration_source_target,
                role_target=role_target_from_restored_manifest)
            write(out/'strict_dataset_preserved_human_caps.json',restored_truth_state)
        else:
            restored_truth_state=load_truth_restoration_state(resume_truth_path,q,manifest,truth_factors,
                source_append_proof=opt.truth_restoration_source_append_proof,source_role_proof=opt.truth_restoration_source_role_proof,source_revision_proof=opt.truth_restoration_source_revision_proof,
                source_location_revision_proof=opt.truth_restoration_source_location_revision_proof,
                reviewed_soft_source_proof=opt.truth_restoration_reviewed_soft_source_proof,
                wide_sampling_loss_runtime=original_wide_curve_integration)
        restored_policy=restored_truth_state.get('progress_policy','historical_ratchet')
        if restored_policy not in ('historical_ratchet','fixed_seed_envelope'):
            raise ValueError('Unknown truth restoration progress policy in checkpoint')
        if restored_policy=='fixed_seed_envelope'and not fixed_seed_policy:
            raise ValueError('Fixed seed envelope checkpoint cannot silently become a historical ratchet state')
        if restored_policy=='fixed_seed_envelope':
            if set(restored_truth_state.get('fixed_seed_envelopes')or{})!=set(restored_truth_state['families']):
                raise ValueError('Every fixed seed family needs its original envelope provenance')
            if restored_truth_state.get('seed_engineering_scale_source')!=seed_engineering_scale_source:
                raise ValueError('Fixed seed existing-engineering scale source changed on continuation')
        write(out/'actual_truth_restoration_state_resume.json',dict(path=str(resume_truth_path),
            sha256=sha(resume_truth_path),complete_q_sha256=qtoken(q),source_target=restored_truth_state['source_target'],
            groups=sum(len(r['group_ids'])for r in restored_truth_state['families'].values()),
            progress_policy=opt.truth_restoration_progress_policy,
            default_limits_never_expanded=not fixed_seed_policy,
            historical_group_minimum_limits_preserved=not fixed_seed_policy,
            actual_current_source_target=truth_restoration_source_target(manifest),
            actual_current_source_role_target=role_target_from_restored_manifest(manifest),
            authenticated_reviewed_source_revision=restored_truth_state.get('authenticated_reviewed_source_revision'),
            authenticated_sampling_loss_continuation=restored_truth_state.get('authenticated_sampling_loss_continuation'),
            authenticated_source_location_revision=restored_truth_state.get('authenticated_source_location_revision'),
            authenticated_reviewed_soft_source_revision=restored_truth_state.get('authenticated_reviewed_soft_source_revision'),
            authenticated_original_motion_role_migration=restored_truth_state.get('authenticated_original_motion_role_migration'),
            authenticated_additive_source_migration=restored_truth_state.get('authenticated_additive_source_migration')))
    review_initial=None;review_previous=None;review_initial_tail=None
    first_complete_normal_seconds=None;accepted_cycle_wall_times=[];last_accepted_wall_time=None
    from collections import OrderedDict
    review_cache=OrderedDict()
    from calibration.program_tracking.current_source_point_collection import collect_current_active_source_points
    from calibration.program_tracking.current_source_points import freeze_current_source_point_inputs,save_current_source_point_binding
    frozen_current_point_inputs=None
    current_point_bindings={}
    def current_source_point_binding(value,camera_path):
        nonlocal frozen_current_point_inputs
        if frozen_current_point_inputs is None:
            if strict_source is not None:
                from calibration.program_tracking.strict_source_dataset_points import freeze_dataset_source_point_inputs
                frozen_current_point_inputs=freeze_dataset_source_point_inputs(
                    out/'current_active_source_pixels_FROZEN.npz',strict_dataset_reference,
                    input_manifest=out/'input_manifest.json',camera_path=camera_path)
                receipts=[strict_source.contract]
            else:
                groups,receipts,exposures=collect_current_active_source_points(state,source,factor)
                frozen_current_point_inputs=freeze_current_source_point_inputs(
                    out/'current_active_source_pixels_FROZEN.npz',groups,actual_pts=exposures,
                    input_manifest=out/'input_manifest.json',family_receipts=receipts)
            print(json.dumps(dict(stage='current_active_source_pixels_frozen_once',
                **frozen_current_point_inputs,source_families=len(receipts))),flush=True)
        key=(qtoken(value),sha(camera_path))
        if key not in current_point_bindings:
            current_point_bindings[key]=save_current_source_point_binding(
                out/('current_source_points_UNACCEPTED_'+key[0][:16]+'.npz'),camera_path,
                frozen_current_point_inputs,q_sha256=key[0],input_manifest=out/'input_manifest.json')
        return current_point_bindings[key]
    def review(value,event=None):
        nonlocal review_initial,review_previous,review_initial_tail,last_accepted_wall_time
        if opt.joint_source_truth_restoration and event and event.get('checkpoint_kind')=='accepted':
            progress=[]
            for family,truth_factor in (('native',human_truth),('wide',wide_human_truth),('confirmed',confirmed_truth)):
                if truth_factor is None:continue
                violation=truth_violation_buffer(truth_factor,value,family=family)
                previous=truth_restoration_limits[family]
                if np.any(violation>previous+1e-8):raise ValueError('Accepted full-source state exceeded a protected truth group practical uncertainty')
                if not fixed_seed_policy:
                    restored=np.where(violation<=1e-8,0.,violation+truth_restoration_uncertainties[family])
                    truth_restoration_limits[family]=np.minimum(previous,restored)
                progress.append(dict(family=family,groups=len(violation),
                    maximum_violation_px=float(violation.max()),squared_violation=float(violation@violation),
                    practical_band_feasible=bool(np.max(violation)<=1e-8)))
            write(out/'actual_joint_source_truth_restoration_progress.json',dict(
                complete_q_sha256=qtoken(value),groups=progress,full_original_source_objective_retained=True,
                progress_policy=opt.truth_restoration_progress_policy,historical_group_minimum_limits_preserved=not fixed_seed_policy,
                human_only_phase_I=False,accuracy_accepted=False))
        review_key=(qtoken(value),_frozen_profile_token(source.updater))
        cached=review_cache.get(review_key)
        if cached is not None:
            camera_path=cached['camera_path']
            if sha(camera_path)!=cached['camera_sha256']:
                raise ValueError('Authenticated same-state camera evidence changed on disk')
            scene=cached['scene'];original_review=cached['original_review']
            current_review=dict(cached['current_review'],event=event)
            write(out/'latest_candidate_scene_UNACCEPTED.json',scene)
            if sha(out/'latest_candidate_scene_UNACCEPTED.json')!=current_review['scene_sha256']:
                raise ValueError('Authenticated same-state scene evidence changed')
            write(out/'original225_review_source_latest.json',original_review)
            write(out/('original225_review_source_'+review_key[0][:16]+'.json'),original_review)
            review_cache.move_to_end(review_key)
            print(json.dumps(dict(stage='reuse_complete_same_q_profile_source_review',q_sha256=review_key[0],
                camera_sha256=cached['camera_sha256'],complete_source_review_values_unchanged=True)),flush=True)
        else:
            core=evaluation_core(value);world=value[world_indices];scene=state.geometry.scene(world)
            from rinkspace.user_contours import apply_user_visible_markings
            scene=apply_user_visible_markings(scene,pairs=[('F1','R1'),('F2','R2'),('F3','R3'),('F4','R4')],
                authority='D:/CodexData/RinkSpace_Data/events/ankara_2026/evidence/user_contours_20261004/user_corrections_current.json')
            if state.patterns is not None:
                if placements is None:scene['shared_board_patterns']=state.patterns.descriptor(core)
                else:scene=source.export_pattern_scene(scene,value)
            if opt.additional_board_circle_source is not None:scene=source.export_additional_circle_scene(scene,value)
            if board_tracks is not None:
                scene=board_tracks.export_print_scene(scene,evaluation_q(value))
                descriptor=scene.get('source_conditioned_board_prints')
                if descriptor is not None:
                    descriptor['evaluation_q_sha256']=descriptor['q_sha256'];descriptor['q_sha256']=qtoken(value)
            scene['source_event_state_contract']=dict(q_sha256=qtoken(value),q_shape=list(value.shape),frozen_profile_sha256=_frozen_profile_token(source.updater))
            scene['full_programme_source_binding']=dict(path=str(out/'input_manifest.json'),source_pts=[229114112,231705600],native_presentations=10123,actual_exposures=8503)
            scene['frame_cameras']=[dict(source_pts=56218880,camera25=factor.camera(value[source.camera_indices]).tolist(),role='geometry evidence wide observer')]
            write(out/'latest_candidate_scene_UNACCEPTED.json',scene)
            camera_path=export_camera(state,core,out/'complete_physical_camera_model_UNACCEPTED.npz')
            original_review=None
            try:
                from structured_programme_source_candidate.original_review_source import evaluate_original_review_rows
                original_review=evaluate_original_review_rows(state.geometry,world,camera_path,complete_q_sha256=qtoken(value))
                if retirement is not None:original_review=retirement.annotate_original_review(original_review)
                if motion_retirement is not None:original_review=motion_retirement.annotate_original_review(original_review)
                write(out/'original225_review_source_latest.json',original_review)
                write(out/('original225_review_source_'+qtoken(value)[:16]+'.json'),original_review)
            except Exception as exc:
                # This independent review never selects or changes fitting rows.
                # Preserve a diagnostic failure without blocking coupled fitting.
                write(out/'original225_review_source_error.json',dict(error_type=type(exc).__name__,error=str(exc),q_sha256=qtoken(value)))
            current_review=dict(event=event,wide=None if strict_source is not None else factor.review(core,value[source.camera_indices]),
                original_ice_pixel_correction=(None if strict_source is not None or ice_pixel_correction is None else
                    review_corrected_ice_pixels(state,core)),
                additional_native_ice_curves=None if strict_source is not None or opt.additional_ice_curves is None else source.native_ice_factor.review(component_state(source.native_ice_factor,value)),
                additional_native_board_edges=None if strict_source is not None or opt.additional_board_source is None else source.native_board_factor.review(component_state(source.native_board_factor,value)),
                additional_native_board_material_points=None if strict_source is not None or opt.additional_board_material_point_source is None else (
                    source.native_board_material_point_factor.review(component_state(source.native_board_material_point_factor,value))
                    if placements is None else source.material_point_source_review(value)),
                additional_native_static_board_tracks=None if strict_source is not None or board_tracks is None else board_tracks.review(evaluation_q(value)),
                additional_native_board_circles=None if strict_source is not None or opt.additional_board_circle_source is None else source.circle_factor.review(component_state(source.circle_factor,value)),
                independent_board_pattern_source=None if strict_source is not None or placements is None else source.pattern_source_review(value),
                additional_native_persistent_ice_long_baseline=None if strict_source is not None or persistent_ice is None else persistent_ice.review(evaluation_q(value)),
                original_source_role_retirement=None if retirement is None else retirement.history_review(),
                **({} if original_point_qualification is None else dict(original_source_point_qualification=original_point_qualification.history_review())),
                static_motion_source_role_retirement=None if motion_retirement is None else motion_retirement.history_review(),
                scene_sha256=sha(out/'latest_candidate_scene_UNACCEPTED.json'),camera_path=str(camera_path),camera_sha256=sha(camera_path),accuracy_accepted=False)
            if strict_source is not None:
                current_review['strict_dataset_original_HELD']=strict_dataset_factor.review(evaluation_q(value))
                current_review['legacy_pixel_diagnostics_are_not_active_fit']=True
            if original_review is not None:
                review_cache[review_key]=dict(scene=scene,original_review=original_review,
                    current_review=dict(current_review,event=None),camera_path=camera_path,camera_sha256=sha(camera_path))
                while len(review_cache)>2:review_cache.popitem(last=False)
        current_points=current_source_point_binding(value,camera_path)
        current_review['current_source_points']=current_points
        write(out/'latest_source_review.json',current_review)
        if event and (event.get('checkpoint_kind')in('initial','resumed')or event.get('stage')=='initial_current_complete_unaccepted'):
            write(out/'initial_source_review_before_coupled_step.json',json.loads((out/'latest_source_review.json').read_text(encoding='utf8')))
        if event and event.get('checkpoint_kind')in('initial','resumed'):
            initial_stage=out/'initial_states'/qtoken(value);initial_stage.mkdir(parents=True,exist_ok=False)
            shutil.copy2(out/'latest_candidate_scene_UNACCEPTED.json',initial_stage/'scene_UNACCEPTED.json')
            shutil.copy2(out/'latest_source_review.json',initial_stage/'source_review_UNACCEPTED.json')
            shutil.copy2(out/'solver/latest_accepted_state.npz',initial_stage/'complete_state_UNACCEPTED.npz')
            write(initial_stage/'state_identity.json',dict(q_sha256=qtoken(value),
                frozen_profile_token=_frozen_profile_token(source.updater),parameter_count=len(value),
                state_kind='explicit_initial_candidate_not_an_accepted_update',
                state_npz_sha256=sha(initial_stage/'complete_state_UNACCEPTED.npz'),
                scene_sha256=sha(initial_stage/'scene_UNACCEPTED.json'),
                camera_path=str(camera_path),camera_sha256=sha(camera_path),
                current_source_points=current_points,
                source_review_sha256=sha(initial_stage/'source_review_UNACCEPTED.json'),
                input_manifest_sha256=sha(out/'input_manifest.json'),accuracy_accepted=False))
        print(json.dumps(dict(stage='complete_current_review_exported',event=event,camera_path=str(camera_path))),flush=True)
        extra=current_review['additional_native_ice_curves']
        def tail_error(report):
            if report is None:return None
            rows=[r for r in report['rows']if r.get('actual_pts')==230368768 and r.get('id')=='F2_inner:right'
                and r.get('evaluation_status')=='evaluated']
            return None if len(rows)!=1 else rows[0].get('original_operator_outside_support_median_px')
        if review_initial is None and extra is not None:
            review_initial={r['id']:dict(r)for r in extra['rows']};review_previous=review_initial
            review_initial_tail=tail_error(original_review)
        if event and event.get('checkpoint_kind')=='accepted':
            stage=out/'accepted_states'/qtoken(value);stage.mkdir(parents=True,exist_ok=False)
            shutil.copy2(out/'latest_candidate_scene_UNACCEPTED.json',stage/'scene_UNACCEPTED.json')
            shutil.copy2(out/'latest_source_review.json',stage/'source_review_UNACCEPTED.json')
            shutil.copy2(out/'solver/latest_accepted_state.npz',stage/'complete_state_UNACCEPTED.npz')
            if original_review is not None:write(stage/'original225_review_UNACCEPTED.json',original_review)
            write(stage/'state_identity.json',dict(q_sha256=qtoken(value),frozen_profile_token=_frozen_profile_token(source.updater),
                parameter_count=len(value),original_complete_parameter_prefix_count=layout.original_complete_count,
                independent_circle_material_parameter_indices=[]if opt.additional_board_circle_source is None else source.circle_indices.tolist(),
                independent_pattern_placement_parameter_indices=[]if placements is None else source.placement_indices.tolist(),
                state_npz_sha256=sha(stage/'complete_state_UNACCEPTED.npz'),scene_sha256=sha(stage/'scene_UNACCEPTED.json'),
                camera_path=str(camera_path),camera_sha256=sha(camera_path),source_review_sha256=sha(stage/'source_review_UNACCEPTED.json'),
                current_source_points=current_points,
                input_manifest_sha256=sha(out/'input_manifest.json'),accuracy_accepted=False,
                accepted_state_J_status='Generated and frozen by the following actual optimizer linearization; pending if directed review stops before then'))
            if opt.joint_source_truth_restoration:
                write(stage/'truth_restoration_state.json',truth_restoration_state_packet(value,stage,
                    out/'input_manifest.json',manifest,truth_factors,truth_restoration_limits,truth_restoration_uncertainties,
                    progress_policy=opt.truth_restoration_progress_policy,seed_envelopes=fixed_seed_envelopes,
                    seed_engineering_scale_source=seed_engineering_scale_source))
            accepted_wall_time=time.perf_counter()
            accepted_cycle_wall_times.append(dict(q_sha256=qtoken(value),
                derivative_model=(event.get('source_jacobian_reuse')or{}).get('derivative_model','fresh_source_J'),
                elapsed_from_run_start_seconds=accepted_wall_time-t,
                accepted_to_accepted_wall_seconds=None if last_accepted_wall_time is None else accepted_wall_time-last_accepted_wall_time,
                solver_elapsed_seconds=event.get('elapsed_seconds'),
                source_stage_timing_seconds=event.get('source_stage_timing_seconds')))
            last_accepted_wall_time=accepted_wall_time
            write(out/'actual_accepted_cycle_performance.json',dict(
                first_complete_source_J_and_normal_seconds=first_complete_normal_seconds,
                first_accepted_includes_preparation=True,cycles=accepted_cycle_wall_times,
                accepted_intervals_include_source_review_checkpoints_profiles_J_and_trial_cost=True,
                historical_hot_first_step_is_not_complete_cold_runtime=True))
            if opt.watch_native_evidence and extra is not None:
                regression=[]
                for row in extra['rows']:
                    prior=review_previous[row['id']];initial=review_initial[row['id']]
                    increase=float(row['mean_native_px']-prior['mean_native_px'])
                    if row['split']=='fit'and increase>.01:
                        regression.append(dict(id=row['id'],role='new FIT raw pixel mean',before=prior['mean_native_px'],
                            after=row['mean_native_px'],increase_px=increase,engineering_trigger_px=.01))
                    if row['split']=='holdout'and row['feature_id']=='F1'and row['mean_native_px']-initial['mean_native_px']>1.:
                        regression.append(dict(id=row['id'],role='F1 original HELD raw pixel mean',before=initial['mean_native_px'],
                            after=row['mean_native_px'],engineering_trigger_px=1.))
                tail=tail_error(original_review)
                if tail is not None and review_initial_tail is not None and tail-review_initial_tail>.5:
                    regression.append(dict(id='F2_inner:right@230368768',role='original HELD outside-support median',
                        before=review_initial_tail,after=tail,engineering_trigger_px=.5))
                policy=dict(q_sha256=qtoken(value),actual_source_regressions=regression,
                    numerical_acceptance_is_not_accuracy=True,engineering_monitor_not_user_acceptance=True,
                    every_source_observation_and_original_split_preserved=True)
                write(stage/'native_source_progress_review.json',policy)
                print(json.dumps(dict(stage='actual_native_source_progress_review',**policy)),flush=True)
                review_previous={r['id']:dict(r)for r in extra['rows']}
                if regression:return dict(stop_branch=True,reason='Actual native source progress regressed; accepted state preserved for directed review',
                    actual_regressions=regression)
        if event and event.get('checkpoint_kind')=='accepted'and opt.accepted_steps>0:
            count=sum(bool(r.get('accepted'))for r in read_solver_history(out/'solver/iterations.json'))
            if count>=opt.accepted_steps:return dict(stop_branch=True,reason='Actual coupled accepted-state review phase ready; numerical acceptance is not visual accuracy')
    trial_cost_pass=0
    programme_row_baselines={}
    def remember_programme_row_baseline(value,block):
        family=block.row_ledger.get('family')
        if family not in ('programme:main1','programme:main7'):return
        from structured_programme_source_candidate.existing_main7_trial_row_cost import capture_existing_programme_baseline
        try:
            programme_row_baselines[family]=capture_existing_programme_baseline(value,
                _frozen_profile_token(source.updater),block,state,sha(out/'input_manifest.json'))
        except Exception as diagnostic_error:
            programme_row_baselines.pop(family,None)
            write(out/('actual_'+family.split(':',1)[1]+'_row_baseline_error.json'),
                dict(error_type=type(diagnostic_error).__name__,error=str(diagnostic_error),
                    complete_q_sha256=qtoken(value),extra_source_calls=0,source_or_solver_changed=False))
    def logged_source_blocks(value,jacobian=True):
        # Observe the already-produced original residuals. No second source
        # evaluation, mutation, mask selection, or different objective occurs.
        nonlocal trial_cost_pass
        if jacobian:
            family_times=[];begin=time.perf_counter();complete=False
            try:
                for block in source.blocks(value,True):
                    ready=time.perf_counter()
                    remember_programme_row_baseline(value,block)
                    family_times.append(dict(family=block.row_ledger.get('family'),
                        rows=len(block.r),source_generation_seconds=ready-begin))
                    yield block
                    begin=time.perf_counter()
                complete=True
            finally:
                write(out/'latest_full_J_family_timings.json',dict(q_sha256=qtoken(value),
                    complete=complete,families=family_times,
                    source_generation_excludes_normal_accumulation=True,
                    original_residual_J_and_source_operators_reused=True))
            return
        trial_cost_pass+=1;rows=[];started=time.perf_counter();complete=False;error=None
        identity=dict(complete_q_sha256=qtoken(value),frozen_profile_token=_frozen_profile_token(source.updater),
            source_manifest_sha256=sha(out/'input_manifest.json'))
        path=out/('actual_trial_family_cost_pass_'+str(trial_cost_pass).zfill(4)+'.json')
        def save():
            write(path,dict(schema='rinkspace.observed-original-source-family-cost.v1',**identity,
                families=rows,complete=complete,error=error,scalar_rows=sum(x['scalar_rows']for x in rows),
                normalized_cost=math.fsum(x['normalized_cost']for x in rows),seconds=time.perf_counter()-started,
                extra_source_value_calls=0,source_arrays_and_order_unchanged=True,accuracy_accepted=False))
        try:
            for ordinal,block in enumerate(source.blocks(value,False)):
                rows.append(dict(ordinal=ordinal,original_row_ledger=block.row_ledger,scalar_rows=len(block.r),
                    normalized_cost=_block_cost(block.r,np.asarray(block.quadratic,bool),block.loss_weights)))
                # Use residuals from the live linearization even when full-J
                # disk export is disabled. No second evaluator or retained J.
                family=block.row_ledger['family']
                if family in programme_row_baselines:
                    try:
                        from structured_programme_source_candidate.existing_main7_trial_row_cost import write_existing_main7_trial_row_cost
                        write_existing_main7_trial_row_cost(out,value,identity['frozen_profile_token'],block,state,trial_cost_pass,
                            baseline=programme_row_baselines[family])
                    except Exception as diagnostic_error:
                        write(out/('actual_trial_'+family.split(':',1)[1]+'_source_row_cost_error_pass_'+str(trial_cost_pass).zfill(4)+'.json'),
                            dict(error_type=type(diagnostic_error).__name__,error=str(diagnostic_error),
                                **identity,extra_source_calls=0,source_or_solver_changed=False))
                save()
                yield block
            complete=True
        except Exception as exc:
            error=dict(type=type(exc).__name__,message=str(exc))
            raise
        finally:save()
    def evaluate(value,jacobian=True):
        blocks=list(logged_source_blocks(value,jacobian));r=np.concatenate([b.r for b in blocks])
        return(r,vstack([b.J for b in blocks],format='csr'))if jacobian else r
    from structured_programme_source_candidate.fixed_pivot_positive_domain_fd import (
        positive_pivot_height_columns,positive_pivot_guard_direction,
        check_positive_pivot_boundary_column,print_domain_directional_stencil)
    pivot_height_columns=positive_pivot_height_columns(state.blocks)
    physical_directional_checks=[]
    def real_physical_directional(evaluation,value,direction,h,base,*,canonical=False,support_branches=None,branches_only_at_print_boundary=False):
        def validate(candidate):
            core=evaluation_core(candidate)if canonical else candidate
            state.geometry._q(core[evaluation_world_indices])
        initial_h=float(h);branch_trials=[]
        nominal_branch=None
        while True:
            try:stencil=print_domain_directional_stencil(value,direction,h,validate)
            except ValueError as error:
                physical_directional_checks.append(dict(passed=False,error=str(error),q_sha256=qtoken(value),
                    initial_requested_step=initial_h,actual_support_branch_trials=branch_trials))
                write(out/'actual_initial_physical_directional_steps.json',physical_directional_checks)
                raise
            samples=[value+offset*direction for offset in stencil['offsets']]
            if support_branches is not None and (not branches_only_at_print_boundary or stencil['method']!='original_symmetric'):
                if nominal_branch is None:nominal_branch=support_branches(value)
                branches=[support_branches(sample)for sample in samples]
                changed=[int(np.count_nonzero(branch!=nominal_branch))for branch in branches]
                branch_trials.append(dict(step=stencil['step'],offsets=stencil['offsets'],changed_support_points=changed))
                if any(changed):h=stencil['step']*.1;continue
            values=[np.asarray(base)if offset==0. else np.asarray(evaluation(sample))
                for offset,sample in zip(stencil['offsets'],samples)]
            if stencil['method']=='original_symmetric':observed=(values[1]-values[0])/(2*stencil['step'])
            else:observed=stencil['side']*(4*(values[1]-values[0])-(values[2]-values[0]))/(2*stencil['step'])
            stencil.update(initial_requested_step=initial_h,actual_support_branch_trials=branch_trials,
                original_source_values_sigmas_supports_and_guard_preserved=True)
            physical_directional_checks.append(dict(q_sha256=qtoken(value),stencil=stencil,
                direction_sha256=qtoken(direction),derivative_evaluated_without_changing_nominal_state=True))
            write(out/'actual_initial_physical_directional_steps.json',physical_directional_checks)
            return observed,stencil,values
    def qualify_pivot_boundary_columns(value,evaluation,matrix,base,guard,allowed,filename):
        checks=[]
        for item in guard['boundary_columns']:
            col=item['column']
            _,check=check_positive_pivot_boundary_column(evaluation,value,col,
                matrix[:,col].toarray().ravel(),1e-6,pivot_height_columns,base=base,allowed=allowed)
            checks.append(check)
            write(out/filename,dict(guard=guard,column_checks=checks,
                actual_complete_q_sha256=qtoken(q),source_values_and_J_unchanged=True))
            if not check['passed']:raise ValueError('Actual positive-pivot inward source derivative failed: '+str(col))
        return checks
    try:
        if opt.reuse_complete_linearization is None and strict_source is None:
            if opt.additional_board_source is not None:
                print(json.dumps(dict(stage='actual_additional_native_board_edge_small_check')),flush=True)
                edge=source.native_board_factor;edgeq=component_state(edge,q);edgeb=edge.block(edgeq,True)
                if edgeb.J.shape!=(edge.row_count,edge.parameter_count)or not np.isfinite(edgeb.r).all()or not np.isfinite(edgeb.J.data).all():
                    raise ValueError('Actual board-edge source values/J are invalid')
                direction=np.zeros(edge.parameter_count);direction[edge.active]=np.random.default_rng(230519552).normal(size=len(edge.active))
                direction/=np.linalg.norm(direction);h=1e-6
                direction,edge_pivot_guard=positive_pivot_guard_direction(edgeq,direction,h,pivot_height_columns)
                observed,edge_support_step,_=real_physical_directional(lambda value:edge.block(value,False).r,
                    edgeq,direction,h,edgeb.r,support_branches=lambda value:edge._support_branches(value)[0])
                h=edge_support_step['step']
                edge_support_step['same_support_branch_step_found']=True
                edge_check=dict(fit_UV=sum(len(r['uv'])for r in edge.rows),scalar_rows=edge.row_count,
                    J_shape=list(edgeb.J.shape),J_nnz=edgeb.J.nnz,support_directional_step=edge_support_step,
                    positive_pivot_guard=edge_pivot_guard,
                    original_source_UV_sigma_roles_profiles_and_model_DOF_unchanged=True,accuracy_accepted=False)
                if not edge_support_step['same_support_branch_step_found']:
                    edge_check.update(passed=False,failure='No representable symmetric step on nominal source support branch')
                    write(out/'actual_additional_native_board_source_check.json',edge_check)
                    raise ValueError('Actual board-edge local support branch FD step unavailable')
                predicted=edgeb.J@direction;error=float(np.max(abs(observed-predicted)))
                allowed=2e-3+2e-5*float(np.max(abs(observed)))
                edge_check.update(coupled_directional_max_error=error,allowed=allowed,passed=error<=allowed)
                write(out/'actual_additional_native_board_source_check.json',edge_check)
                if error>allowed:raise ValueError('Actual board-edge coupled world/camera FD failed')
                edge_boundary_checks=qualify_pivot_boundary_columns(edgeq,
                    lambda value:edge.block(value,False).r,edgeb.J,edgeb.r,edge_pivot_guard,allowed,
                    'actual_board_positive_pivot_boundary_qualification.json')
                scene_for_edge=state.geometry.current.source_scene(q[world_indices])
                edge_exports=check_current_board_edge_exports(edge,edgeq,scene_for_edge)
                difference=max((item['export_world_max_difference']for item in edge_exports),default=0.)
                edge.block(edgeq,False)
                edge_check.update(positive_pivot_boundary_checks=edge_boundary_checks,
                    current_factor_export_world_max_difference=difference,current_material_export_checks=edge_exports,
                    review=edge.review(edgeq))
                write(out/'actual_additional_native_board_source_check.json',edge_check)
                print(json.dumps(dict(stage='actual_additional_native_board_edge_check_passed',FD_error=error,source_rows=edge.row_count)),flush=True)
            if opt.additional_ice_curves is not None and opt.reuse_complete_linearization is None:
                print(json.dumps(dict(stage='actual_additional_native_ice_curves_small_check')),flush=True)
                extra=source.native_ice_factor;extraq=component_state(extra,q);eb=extra.block(extraq,True)
                if extra.contract.get('objective_revision')is not None:
                    from structured_programme_source_candidate.check_native_ice_raw_sigma_revision import check_raw_sigma_revision
                    transform=check_raw_sigma_revision(extra,extraq)
                    write(out/'actual_native_ice_raw_sigma_revision_check.json',transform)
                    print(json.dumps(dict(stage='actual_native_ice_raw_sigma_revision_check_passed',**transform)),flush=True)
                if eb.J.shape!=(extra.row_count,extra.parameter_count)or not np.isfinite(eb.r).all()or not np.isfinite(eb.J.data).all():
                    raise ValueError('Additional real native-ice source values/J are invalid')
                active=np.unique(eb.J.indices);direction=np.zeros(extra.parameter_count)
                direction[active]=np.random.default_rng(19066).uniform(-.1,.1,len(active))
                step=1e-6
                direction,ice_pivot_guard=positive_pivot_guard_direction(extraq,direction,step,pivot_height_columns)
                ice_active=extra._qualified_active_fit_view()
                if ice_active is None:ice_active=extra
                def ice_actual_branch(value):
                    from structured_programme_source_candidate.native_ice_directional_diagnostic import actual_native_ice_row_branch_signature
                    cameras=ice_active._cameras(value);parts=[]
                    for row in ice_active.rows:
                        parts.extend(actual_native_ice_row_branch_signature(
                            cameras[row['actual_pts']],ice_active._world(value,row['curve']),row))
                    return np.concatenate(parts)
                observed,ice_physical_step,ice_values=real_physical_directional(lambda value:extra.block(value,False).r,
                    extraq,direction,step,eb.r,support_branches=ice_actual_branch,branches_only_at_print_boundary=True)
                step=ice_physical_step['step'];vm,vp=ice_values[0],ice_values[-1];predicted=eb.J@direction
                err=float(abs(observed-predicted).max());allowed=3e-3+1e-3*float(abs(observed).max())
                ice_boundary_checks=qualify_pivot_boundary_columns(extraq,
                    lambda value:extra.block(value,False).r,eb.J,eb.r,ice_pivot_guard,allowed,
                    'actual_ice_positive_pivot_boundary_qualification.json')
                ice_checked_err=err;ice_branch_summary=None
                if err>allowed:
                    if ice_physical_step['method']!='original_symmetric':
                        write(out/'actual_native_ice_physical_domain_directional_failure.json',dict(physical_domain_step=ice_physical_step,
                            actual_error=err,allowed=allowed,source_and_nominal_J_unchanged=True))
                        raise ValueError('Actual feasible-side native-ice coupled source derivative failed')
                    from structured_programme_source_candidate.native_ice_directional_diagnostic import record_native_ice_directional_diagnostic,check_actual_native_ice_directional_branches
                    ice_diag=record_native_ice_directional_diagnostic(extra,extraq,q,eb,direction,step,vp,vm,predicted,out,
                        dict(frozen_profile_token=_frozen_profile_token(source.updater),input_manifest_sha256=sha(out/'input_manifest.json')))
                    checked_observed,ice_branch_summary=check_actual_native_ice_directional_branches(extraq,eb,direction,step,vp,vm,predicted,ice_diag)
                    ice_checked_err=float(abs(checked_observed-predicted).max())
                    write(out/'actual_native_ice_directional_branch_qualification.json',ice_branch_summary)
                    print(json.dumps(dict(stage='actual_native_ice_directional_failure_preserved',
                        original_error=err,original_allowed=allowed,numeric_path=ice_diag['numeric_path'],
                        branch_replay_completed=ice_diag.get('branch_replay_completed',False),
                        checked_error=ice_checked_err,qualified_UV=len(ice_branch_summary['qualified_UV']))),flush=True)
                    if ice_checked_err>allowed:raise ValueError('Additional current-camera/current-world coupled directional derivative failed')
                extra.block(extraq,False)
                write(out/'actual_additional_native_ice_source_check.json',dict(scalar_fit_rows=extra.row_count,
                    native_fit_UV=sum(len(r['uv'])for r in ice_active.rows),
                    original_native_fit_UV=sum(len(r['uv'])for r in extra.rows),native_held_UV=sum(len(r['uv'])for r in extra.held),
                    actual_J_shape=list(eb.J.shape),actual_J_nnz=eb.J.nnz,actual_active_columns=active.tolist(),
                    coupled_directional_max_absolute=ice_checked_err,coupled_directional_tolerance=allowed,
                    positive_pivot_guard=ice_pivot_guard,positive_pivot_boundary_checks=ice_boundary_checks,
                    physical_domain_directional_step=ice_physical_step,
                    original_two_sided_directional_max_absolute=err,piecewise_branch_qualification=ice_branch_summary,
                    original_source_UV_sigma_and_splits_changed=False,review=extra.review(extraq),accuracy_accepted=False))
            print(json.dumps(dict(stage='actual_all_small_source_values_and_jacobians',seconds=time.perf_counter()-t)),flush=True)
            small=[]
            with (nullcontext() if placements is None else source.pattern_placement_context(q)):
                for family,r,J in raw.tail_evaluations(evaluation_core(q),True):
                    if not np.isfinite(r).all()or not np.isfinite(J.data).all():raise ValueError('Small original source family values/J invalid: '+family)
                    small.append(dict(family=family,actual_rows=len(r),actual_nnz=J.nnz))
            if opt.reuse_wide_linearization is not None:
                from structured_programme_source_candidate.wide_same_state_replay import install_same_state_wide_replay
                wide_replay_report=install_same_state_wide_replay(factor,q,source.camera_indices,opt.reuse_wide_linearization,manifest,core_getter=evaluation_core)
                write(out/'actual_same_state_wide_linearization_replay.json',wide_replay_report)
            wr,wc,wp=factor.evaluate(evaluation_core(q),q[source.camera_indices],True)
            if not np.isfinite(wr).all()or not np.isfinite(wc.data).all()or not np.isfinite(wp).all():raise ValueError('Complete wide source values/J invalid')
            direction=np.zeros(len(q));wide_cols=np.r_[layout.canonical_indices(factor.core_active_indices),source.camera_indices]
            direction[wide_cols]=np.random.default_rng(43425808).uniform(-.01,.01,len(wide_cols))
            step=1e-6
            direction,wide_pivot_guard=positive_pivot_guard_direction(q,direction,step,pivot_height_columns)
            def wide_actual_branch(value):
                from structured_programme_source_candidate.wide_same_state_replay import _branch
                world,camera,*_=factor._inputs(evaluation_core(value),value[source.camera_indices]);curves,top,_=factor._curve_worlds(world)
                parts=[]
                for curve,uv in [*zip(curves,factor.uvs),*((top,uv)for uv in factor.trace_uv)]:
                    branch=_branch(camera,curve,uv);parts.extend((branch['index'],branch['category']))
                return np.concatenate(parts)
            observed,wide_physical_step,_=real_physical_directional(
                lambda value:factor.evaluate(evaluation_core(value),value[source.camera_indices],False)[0],
                q,direction,step,wr,canonical=True,support_branches=wide_actual_branch,branches_only_at_print_boundary=True)
            step=wide_physical_step['step'];plus=q+step*direction;minus=q-step*direction
            predicted=wc@evaluation_core(direction)+wp@direction[source.camera_indices]
            err=float(abs(observed-predicted).max());allowed=3e-3+1e-3*float(abs(observed).max())
            wide_boundary_checks=qualify_pivot_boundary_columns(q,
                lambda value:factor.evaluate(evaluation_core(value),value[source.camera_indices],False)[0],
                wc,wr,wide_pivot_guard,allowed,'actual_wide_positive_pivot_boundary_qualification.json')
            from structured_programme_source_candidate.wide_same_state_replay import check_actual_wide_directional_branches
            if wide_physical_step['method']=='original_symmetric':
                branch_check=check_actual_wide_directional_branches(factor,q,source.camera_indices,direction,step,wr,observed,predicted,core_getter=evaluation_core)
            else:
                branch_check=dict(checked_observed=observed,seams=[],physical_domain_step=wide_physical_step,
                    source_values_and_nominal_J_and_true_objective_unchanged=True)
            checked_observed=np.asarray(branch_check.pop('checked_observed'))
            err=float(abs(checked_observed-predicted).max());allowed=3e-3+1e-3*float(abs(checked_observed).max())
            families=[];cursor=0
            for row,uv in zip(factor.observations,factor.uvs):
                count=2*len(uv);families.append((str(row.get('id',row.get('target'))),cursor,cursor+count));cursor+=count
            families.extend([('original_advertisement_points',4288,4340),('same_installation_entity',4340,4342)])
            cursor=4342
            for row,uv in zip(factor.trace_packet['contours'],factor.trace_uv):
                count=2*len(uv);families.append((row['id'],cursor,cursor+count));cursor+=count
            def derivative_stats(actual,expected):
                delta=abs(actual-expected);at=int(np.argmax(delta))
                return dict(max_absolute=float(delta[at]),argmax_scalar_row=at,observed=float(actual[at]),predicted=float(expected[at]),
                    family=next(name for name,start,stop in families if start<=at<stop),
                    per_family=[dict(family=name,start=start,stop=stop,max_absolute=float(delta[start:stop].max()),
                        argmax_scalar_row=int(start+np.argmax(delta[start:stop])),observed_max_absolute=float(abs(actual[start:stop]).max()))
                        for name,start,stop in families])
            diag=dict(q_sha256=qtoken(q),frozen_profile_token=_frozen_profile_token(source.updater),
                source_manifest_sha256=sha(out/'input_manifest.json'),step=step,allowed=allowed,
                positive_pivot_guard=wide_pivot_guard,positive_pivot_boundary_checks=wide_boundary_checks,
                physical_domain_directional_step=wide_physical_step,
                coupled=derivative_stats(observed,predicted),original_small_families=small,passed=err<=allowed)
            diag['actual_local_branch_check']=branch_check
            diag['checked_max_absolute']=err
            diag['checked_coupled']=derivative_stats(checked_observed,predicted)
            # Cache the actual linearization before any diagnostic can fail. It is
            # small (5808 x 97 active columns), and no programme actor has started.
            numeric=dict(q=q,direction=direction,wide_private_indices=source.camera_indices,wide_core_active_indices=factor.core_active_indices,
                residual=wr,core_J_data=wc.data,core_J_indices=wc.indices,core_J_indptr=wc.indptr,core_J_shape=np.asarray(wc.shape),
                private_J=wp,observed_directional=observed,predicted_directional=predicted,
                source_manifest_sha256=np.asarray(diag['source_manifest_sha256']),frozen_profile_token=np.asarray(diag['frozen_profile_token']))
            np.savez(out/'actual_wide_linearization_diagnostic.npz',**numeric)
            write(out/'actual_wide_directional_diagnostic.json',diag)
            print(json.dumps(dict(stage='actual_wide_directional_diagnostic_cached',error=err,allowed=allowed,
                argmax=diag['coupled']['argmax_scalar_row'],family=diag['coupled']['family'])),flush=True)
            if err>allowed:
                if wide_physical_step['method']!='original_symmetric':
                    raise ValueError('Complete wide feasible-side current world/camera coupled source derivative failed; actual numeric diagnosis preserved')
                for name,indices in [('world',world_indices),('entity_target_and_lever',np.r_[factor.secondary_xyz_indices,factor.secondary_mount_indices]),
                    ('private_camera',source.camera_indices)]:
                    separate=np.zeros(len(q));separate[indices]=direction[indices];p=q+step*separate;m=q-step*separate
                    actual=(factor.evaluate(evaluation_core(p),p[source.camera_indices],False)[0]-factor.evaluate(evaluation_core(m),m[source.camera_indices],False)[0])/(2*step)
                    expected=wc@evaluation_core(separate)+wp@separate[source.camera_indices]
                    diag[name]=derivative_stats(actual,expected);numeric[name+'_observed']=actual;numeric[name+'_predicted']=expected
                    print(json.dumps(dict(stage='wide_derivative_component',component=name,**{k:v for k,v in diag[name].items()if k!='per_family'})),flush=True)
                    write(out/'actual_wide_directional_diagnostic.json',diag)
                for label,value in [('base',q),('plus',plus),('minus',minus)]:
                    world,camera,*_=factor._inputs(evaluation_core(value),value[source.camera_indices]);curves,top,ads=factor._curve_worlds(world)
                    numeric[label+'_camera25']=camera;numeric[label+'_world66']=world;numeric[label+'_top']=top;numeric[label+'_ads']=ads
                    for i,(curve,uv,sigma)in enumerate(zip(curves,factor.uvs,factor.sigmas)):
                        numeric[label+'_curve_'+str(i)]=curve
                        if label=='base':numeric['curve_uv_'+str(i)]=uv;numeric['curve_sigma_'+str(i)]=sigma
                for i,uv in enumerate(factor.trace_uv):numeric['trace_uv_'+str(i)]=uv
                np.savez(out/'actual_wide_linearization_diagnostic.npz',**numeric)
                raise ValueError('Complete wide current world/camera coupled directional derivative failed; actual numeric diagnosis preserved')
            # The two value-only directional states evict the tiny LRU. Restore
            # this authenticated base result, avoiding another 66-world pass.
            base_key=factor._inputs(evaluation_core(q),q[source.camera_indices])[-1]
            factor._cache[base_key]=(wr,wc,wp)
            while len(factor._cache)>2:factor._cache.popitem(last=False)
            source.reference_factor.block(component_state(source.reference_factor,q),True)
            write(out/'actual_all_small_source_value_J_check.json',dict(original_small_families=small,
                complete_wide_curves=len(factor.observations),complete_original_wide_curve_scalar_rows=4288,
                complete_original_ad_points=len(factor.ad_packet['rows']),complete_original_entity_scalar_rows=2,
                complete_user_stroke_count=len(factor.trace_uv),complete_user_UV=sum(len(u)for u in factor.trace_uv),
                actual_wide_rows=len(wr),actual_core_J_shape=list(wc.shape),actual_core_J_nnz=wc.nnz,
                actual_private_J_shape=list(wp.shape),coupled_directional_max_absolute=err,
                coupled_directional_tolerance=allowed,finite_support_and_original_UV_sigma_preserved=True,accuracy_accepted=False))
            print(json.dumps(dict(stage='all_small_source_value_J_check_passed',wide_directional_error=err)),flush=True)
        print(json.dumps(dict(stage='current_source_guard',seconds=time.perf_counter()-t)),flush=True);source.guard(q)
        if strict_source is not None:
            strict_preflight=strict_dataset_factor.preflight(evaluation_q(q))
            write(out/'strict_dataset_source_value_J_preflight.json',strict_preflight)
            print(json.dumps(dict(stage='strict_dataset_source_value_J_preflight_passed',report=strict_preflight)),flush=True)
        # A restored complete normal already owns an authenticated frozen
        # profile. Otherwise prepare it once before computing that normal.
        if opt.reuse_complete_linearization is None and opt.reuse_source_blocks_from is None and opt.verify_source_blocks_from is None:source.updater(q)
        np.savez_compressed(out/'actual_initial_frozen_profile_arrays.npz',**source.updater.checkpoint_arrays())
        np.savez_compressed(out/'actual_initial_normal_state.npz',q=q,
            frozen_profile_token=np.asarray(_frozen_profile_token(source.updater)),source_manifest_sha256=np.asarray(sha(out/'input_manifest.json')))
        arrays={k[5:]:v for k,v in template.items()if k.startswith('aux__')}
        arrays.update(current_shared_world_geometry_parameter_indices=world_indices,current_named66_parameter_names=np.asarray(state.geometry.parameter_names),
            current_coordinate_contract_json=np.array(json.dumps(manifest['current_geometry'])),correlated_roll_prior_contract_json=np.array(json.dumps(base.contract)),
            current_shared_mount_parameter_indices=state.mount_parameter_indices,current_all_installation_optics_parameter_indices=state.optics_parameter_indices,
            current_wide_camera_parameter_indices=source.camera_indices)
        arrays.update(current_complete_parameter_layout_json=np.asarray(json.dumps(layout.manifest())),
            current_evaluation_to_canonical_parameter_indices=layout.evaluation_to_canonical,
            current_physical_geometry_initializer_json=np.asarray(json.dumps(physical_geometry_initializer)))
        if far_board_role_correction is not None:
            arrays.update(current_original_motion_far_board_role_correction_json=np.asarray(json.dumps(far_board_role_correction)))
        if mount_working_constraints is not None:
            arrays.update(current_camera_mount_working_constraints_json=np.asarray(json.dumps(mount_working_constraints)))
        if pivot_outside_constraints is not None:
            arrays.update(current_camera_pivot_exterior_constraints_json=np.asarray(json.dumps(pivot_outside_constraints)),
                explicit_pivot_outside_reinitialization_json=np.asarray(json.dumps(pivot_reinitialization)),
                physical_domain_initial_q_sha256=np.asarray(manifest['physical_domain_initial_q_sha256']))
        if placements is not None:
            arrays.update(current_independent_pattern_placement_parameter_indices=source.placement_indices,
                current_independent_pattern_placement_contract_json=np.asarray(json.dumps(placements.placement_contract)))
        if persistent_ice is not None:
            arrays.update(current_persistent_ice_long_baseline_contract_json=np.asarray(json.dumps(persistent_ice.contract)),
                current_persistent_ice_long_baseline_existing_camera_parameter_indices=persistent_ice.active)
        if motion_retirement is not None:
            arrays.update(current_static_motion_source_role_retirement_contract_json=np.asarray(json.dumps(motion_retirement.contract)))
        if motion_input_compaction is not None:
            arrays.update(current_original_motion_input_compaction_json=np.asarray(json.dumps(motion_input_compaction)))
        # Validate the actual existing solver-domain assembly before full J.
        for column,interval in mount_parameter_bounds.items():
            if (int(column)not in shared or len(interval)!=2 or
                    not np.isfinite(interval).all()or not interval[0]<interval[1]):
                raise ValueError('Finite selected shared working intervals required before full J')
        if not np.all(np.isin(pivot_height_columns,shared))or np.any(q[pivot_height_columns]<=0.):
            raise ValueError('Positive selected tripod-height coordinates required before full J')
        # The joint solver reuses the existing explicit fixed-normal protocol
        # and excludes fixed slots from its LP/QP chart. Prepare exactly that
        # tangent once, while preserving requested/shared checkpoint metadata.
        initial_domain_columns=shared[~np.isin(shared,fixed)]
        initial_step_domain=shared_linear_constraints(q,initial_domain_columns)
        def first_domain_identity(value,selected):
            columns=np.asarray(selected)
            limits=tuple((family,np.asarray(values).dtype.str,np.asarray(values).shape,
                np.asarray(values).tobytes())for family,values in sorted(truth_restoration_limits.items()))
            return (qtoken(value),columns.dtype.str,columns.shape,columns.tobytes(),
                _frozen_profile_token(source.updater),limits,
                json.dumps(priority_restoration_domain,sort_keys=True,allow_nan=False))
        initial_domain_identity=first_domain_identity(q,initial_domain_columns)
        pending_initial_domain=[initial_step_domain]
        def solver_shared_linear_constraints(value,selected):
            # Consume once. Same q alone is insufficient: accepted reviews can
            # tighten truth limits, and priority contraction owns live state.
            if pending_initial_domain:
                saved_domain=pending_initial_domain.pop()
                if first_domain_identity(value,selected)==initial_domain_identity:
                    print(json.dumps(dict(stage='reuse_exact_preflight_shared_domain',
                        q_sha256=qtoken(value),rows=len(saved_domain['lower']),
                        columns=len(selected),q_profiles_limits_priority_and_columns_verified=True)),flush=True)
                    return saved_domain
            return shared_linear_constraints(value,selected)
        if (initial_step_domain['matrix'].shape!=(len(initial_step_domain['lower']),len(initial_domain_columns))or
                not np.isfinite(initial_step_domain['matrix']).all()or
                not np.isfinite(initial_step_domain['lower']).all()):
            raise ValueError('Finite actual shared domain tangent and lower margins required')
        write(out/'actual_initial_shared_step_domain.json',dict(
            finite_working_intervals={str(k):list(v)for k,v in mount_parameter_bounds.items()},
            domain_rows=len(initial_step_domain['lower']),shared_count=len(initial_domain_columns),
            requested_shared_count=len(shared),requested_shared_parameter_indices=shared.tolist(),
            effective_shared_parameter_indices=initial_domain_columns.tolist(),
            algebraically_eliminated_declared_fixed_shared_indices=shared[np.isin(shared,fixed)].tolist(),
            report=initial_step_domain['report'],actual_initial_domain_check_passed=True,
            source_observations_weights_and_nonlinear_guards_unchanged=True))
        # Validate the exact live checkpoint contracts before expensive full J.
        # Every JSON auxiliary was encoded above with the standard encoder.
        json.dumps(manifest,allow_nan=False)
        if any(np.asarray(v).dtype.hasobject for v in arrays.values()):
            raise TypeError('Checkpoint auxiliary must use numeric or explicit JSON arrays')
        for key,value in arrays.items():
            if key.endswith('_json'):json.loads(str(np.asarray(value)))
        write(out/'actual_checkpoint_serialization_preflight.json',dict(
            live_complete_input_and_source_manifest_standard_JSON_passed=True,
            all_auxiliary_JSON_decoded=True,object_arrays_present=False,
            actual_auxiliary_count=len(arrays),parameter_count=len(q),
            q_sha256=qtoken(q),no_default_string_conversion=True))
        print(json.dumps(dict(stage='actual_checkpoint_serialization_preflight_passed',
            auxiliary_count=len(arrays),q_sha256=qtoken(q))),flush=True)
        review(q,dict(stage='initial_current_complete_unaccepted'))
        print(json.dumps(dict(stage='actual_first_complete_native_source_normal',rows=len(source.scales),parameters=len(q))),flush=True)
        family_rows=[]
        numeric_blocks=out/'initial_normal_source_blocks'
        if opt.save_initial_linearization_cache:numeric_blocks.mkdir()
        def recorded_initial_blocks(iterator):
            begin=time.perf_counter()
            for ordinal,b in enumerate(iterator):
                ready=time.perf_counter()
                remember_programme_row_baseline(q,b)
                family_rows.append(dict(b.row_ledger,actual_rows=len(b.r),actual_J_shape=list(b.J.shape),actual_nnz=b.J.nnz,
                    actual_normalized_cost=_block_cost(b.r,np.asarray(b.quadratic,bool),b.loss_weights),
                    source_generation_seconds=ready-begin))
                write(out/'latest_initial_J_family_timings.json',dict(q_sha256=qtoken(q),complete=False,
                    families=family_rows,source_generation_excludes_normal_accumulation=True))
                print(json.dumps(dict(stage='actual_initial_source_family_ready',ordinal=ordinal,
                    family=b.row_ledger.get('family'),rows=len(b.r),source_generation_seconds=ready-begin)),flush=True)
                # Persistent full matrices are an explicit single-cache export.
                # Ordinary solves retain the current linearization in memory
                # only until the solver completes its GN/native replay.
                if opt.save_initial_linearization_cache:
                    path=numeric_blocks/(str(ordinal).zfill(3)+'.npz')
                    np.savez(path,r=b.r,J_data=b.J.data,J_indices=b.J.indices,J_indptr=b.J.indptr,
                        J_shape=np.asarray(b.J.shape),quadratic=b.quadratic,scales=b.scales,
                        ledger_json=np.asarray(json.dumps(b.row_ledger)),q_sha256=np.asarray(qtoken(q)),
                        frozen_profile_token=np.asarray(_frozen_profile_token(source.updater)),
                        **source_loss_checkpoint_arrays(b.loss_weights,len(b.r)))
                    write(numeric_blocks/'completed_prefix.json',dict(q_sha256=qtoken(q),
                        frozen_profile_token=_frozen_profile_token(source.updater),source_manifest_sha256=sha(out/'input_manifest.json'),
                        completed_blocks=family_rows,completed_original_rows=sum(x['actual_rows']for x in family_rows),
                        mathematical_state_changed=False,cache_role='Explicit initial source r/J export; subsequent iterations are not persisted'))
                yield b
                begin=time.perf_counter()
            write(out/'latest_initial_J_family_timings.json',dict(q_sha256=qtoken(q),complete=True,
                families=family_rows,source_generation_excludes_normal_accumulation=True))
        def actual_profiled_blocks():
            yield from recorded_initial_blocks(source.blocks(q,True))
        profiler=cProfile.Profile();profiler.enable()
        # Profiling is diagnostic work, not part of normal repeated solves.
        # Keep a valid minimal Stats object when detailed profiling is off.
        if os.environ.get('RINKSPACE_SOLVER_PROFILE')!='1':profiler.disable()
        initial_normal_begin=time.perf_counter()
        if opt.verify_source_blocks_from is not None:
            from structured_programme_source_candidate.source_block_target_extension_replay import audit_same_state_ice_extension_blocks
            audit_factory,audit_report=audit_same_state_ice_extension_blocks(opt.verify_source_blocks_from,q,source,manifest,
                _frozen_profile_token(source.updater),verified_prefix_report=opt.verified_source_prefix_report,raw=raw)
            try:
                normal=accumulate_source_normal(lambda:recorded_initial_blocks(audit_factory()),len(q),expected_row_count=len(source.scales),
                    frozen_profile_token=_frozen_profile_token(source.updater),fixed_parameter_indices=fixed,
                    retain_native_replay=not opt.evaluate_only,source_state=q if not opt.evaluate_only else None)
            finally:write(out/'actual_accelerated_same_state_source_block_equivalence.json',audit_report)
        elif opt.reuse_source_blocks_from is not None:
            from structured_programme_source_candidate.source_block_target_extension_replay import prepare_same_state_source_extension_blocks
            replay_factory,replay_report=prepare_same_state_source_extension_blocks(opt.reuse_source_blocks_from,q,source,manifest,
                _frozen_profile_token(source.updater),fixed_parameter_indices=fixed,shared_parameter_mask=actual_shared_mask)
            normal=accumulate_source_normal(lambda:recorded_initial_blocks(replay_factory()),len(q),expected_row_count=len(source.scales),
                frozen_profile_token=_frozen_profile_token(source.updater),fixed_parameter_indices=fixed,
                retain_native_replay=not opt.evaluate_only,source_state=q if not opt.evaluate_only else None)
            write(out/'actual_appended_source_blocks_replay.json',replay_report)
        elif opt.reuse_complete_linearization is None:
            normal=accumulate_source_normal(actual_profiled_blocks,len(q),expected_row_count=len(source.scales),
                frozen_profile_token=_frozen_profile_token(source.updater),fixed_parameter_indices=fixed,
                retain_native_replay=not opt.evaluate_only,source_state=q if not opt.evaluate_only else None)
        else:
            from structured_programme_source_candidate.complete_same_state_replay import load_complete_same_state_normal
            replay_gate=safe_same_state_replay_predicate(opt.reuse_complete_linearization,q,source,manifest,
                _frozen_profile_token(source.updater),fixed_parameter_indices=fixed,shared_parameter_mask=actual_shared_mask)
            write(out/'actual_complete_source_replay_preflight.json',replay_gate)
            if not replay_gate['allowed']:raise ValueError('Complete same-state cache refused: '+replay_gate['reason'])
            normal=load_complete_same_state_normal(opt.reuse_complete_linearization,q,source,manifest,
                _frozen_profile_token(source.updater),fixed_parameter_indices=fixed,shared_parameter_mask=actual_shared_mask)
            write(out/'actual_complete_source_linearization_replay.json',dict(source_directory=str(opt.reuse_complete_linearization),
                original_source_rows=normal.contract.row_count,original_source_blocks=len(normal.contract.signatures),
                exact_q_profile_and_all_block_inventory_r_J_signatures_verified=True,original_H_g_cost_reused=True,
                programme_sources_reevaluated_for_normal=False))
        first_complete_normal_seconds=time.perf_counter()-initial_normal_begin
        profiler.disable()
        statistics=pstats.Stats(profiler);native=[]
        for (file,line,function),value in statistics.stats.items():
            if any(token in (file+' '+function).lower()for token in ('native','scatter','profile_motion','central_valid','finite_plane_motion')):
                primitive,calls,total,cumulative,_=value
                native.append(dict(file=file,line=line,function=function,calls=calls,self_seconds=total,cumulative_seconds=cumulative))
        from z190_motion_native import metadata as motion_metadata
        from z190_csr_scatter_native import metadata as scatter_metadata
        write(out/'actual_backend_and_source_continuity.json',dict(actual_family_rows=family_rows,actual_total_rows=sum(x['actual_rows']for x in family_rows),
            actual_native_motion_backend=motion_metadata(),actual_native_scatter_backend=scatter_metadata(),
            actual_family_times=raw.actual_family_times,
            actual_actor_last_stats=[]if raw.actors is None else raw.actors.last_stats,
            actual_memory_samples=[]if raw.actors is None else raw.actors.memory_samples,
            actual_resident_source_actors=raw.actors is not None,
            actual_hot_calls=sorted(native,key=lambda x:x['cumulative_seconds'],reverse=True),
            original_motion_pairs=2195232,original_motion_scalar_rows=4390464,
            actual_active_motion_pairs=sum(len(b.bundle.a)for b in state.blocks),
            actual_active_motion_scalar_rows=sum(b.bundle.motion_rows for b in state.blocks),all8503_exposure_r3_logf_carried=True,
            source_exposure_aliases_preserved=True,accuracy_accepted=False))
        write(out/'first_complete_source_normal.json',dict(normal.report,full_cost=float(normal.cost),
            source_J_and_normal_seconds=first_complete_normal_seconds,
            seconds_since_start=time.perf_counter()-t))
        if opt.save_initial_linearization_cache:
            np.savez_compressed(out/'first_complete_source_normal_numeric.npz',q=q,normal_data=normal.normal.data,
                normal_indices=normal.normal.indices,normal_indptr=normal.normal.indptr,normal_shape=np.asarray(normal.normal.shape),
                gradient=normal.gradient,cost=np.asarray(normal.cost),frozen_profile_token=np.asarray(normal.contract.frozen_profile_token),
                source_contract_json=np.asarray(json.dumps(asdict(normal.contract))),report_json=np.asarray(json.dumps(normal.report)))
            try:
                write(out/'actual_complete_cache_seal.json',seal_complete_same_state_cache(out,q,source,manifest,
                    normal.contract.frozen_profile_token,fixed_parameter_indices=fixed,shared_parameter_mask=actual_shared_mask))
            except Exception as error:
                unavailable=dict(status='UNSEALED_NOT_REUSABLE',error_type=type(error).__name__,reason=str(error),
                    actual_complete_normal_retained_in_RAM=True,no_matrix_reuse_authorized=True)
                write(out/'actual_optional_cache_seal_failure.json',unavailable)
                print(json.dumps(dict(stage='optional_cache_seal_failed_actual_full_normal_continues',**unavailable)),flush=True)
        print(json.dumps(dict(stage='actual_first_complete_native_source_normal_done',cost=float(normal.cost),seconds=time.perf_counter()-t)),flush=True)
        if opt.evaluate_only:return
        # Reuse the normal just computed at the EXACT same q and frozen
        # profiles.  The existing solver retains all guards, GN/Schur/native
        # step checks and complete source line-search costs.
        import z190_streaming_source_normal as normal_module
        original_accumulate=normal_module.accumulate_source_normal;reused=False;normal_serial=0
        class FirstFrozenProfileReuse:
            def __init__(self,delegate):self.delegate=delegate;self.first=True
            def __getattr__(self,name):return getattr(self.delegate,name)
            def __call__(self,value):
                if self.first:
                    if not np.array_equal(value,q)or _frozen_profile_token(self.delegate)!=normal.contract.frozen_profile_token:
                        raise ValueError('Cannot reuse already-updated profiles: q or frozen arrays changed')
                    self.first=False
                    print(json.dumps(dict(stage='reuse_exact_already_updated_initial_profiles',profile_token=normal.contract.frozen_profile_token)),flush=True)
                    return
                return self.delegate(value)
        solver_profiles=FirstFrozenProfileReuse(source.updater)
        def save_current_complete_cache(value,computed):
            """One explicitly requested fresh stop-point export, without J copies."""
            cache=computed.native_replay_cache
            token=_frozen_profile_token(source.updater)
            if (cache is None or not np.array_equal(cache.source_state,value)
                or computed.contract.frozen_profile_token!=token
                or cache.contract!=computed.contract or computed.contract.row_count!=len(source.scales)
                or computed.report['fixed_parameter_indices']!=list(fixed)):
                raise ValueError('Current complete cache export requires exact fresh q/profile/source/fixed policy')
            destination=out/'current_complete_linearization_cache'
            if 'persistent_source_replay_identity'not in manifest:
                manifest['persistent_source_replay_identity']=prepare_actual_cache_identity()
            destination.mkdir()
            write(destination/'input_manifest.json',manifest)
            np.savez(destination/'actual_initial_normal_state.npz',q=value,
                frozen_profile_token=np.asarray(token),source_manifest_sha256=np.asarray(sha(destination/'input_manifest.json')))
            np.savez(destination/'actual_initial_frozen_profile_arrays.npz',**dict(source.updater.iter_checkpoint_arrays()))
            blocks=destination/'initial_normal_source_blocks';blocks.mkdir();inventory=[]
            for ordinal,block in enumerate(cache.blocks):
                np.savez(blocks/(str(ordinal).zfill(3)+'.npz'),r=block.r,J_data=block.J.data,
                    J_indices=block.J.indices,J_indptr=block.J.indptr,J_shape=np.asarray(block.J.shape),
                    quadratic=block.quadratic,scales=block.scales,ledger_json=np.asarray(json.dumps(block.row_ledger)),
                    q_sha256=np.asarray(qtoken(value)),frozen_profile_token=np.asarray(token),
                    **source_loss_checkpoint_arrays(block.loss_weights,len(block.r)))
                inventory.append(dict(block.row_ledger,actual_rows=len(block.r),actual_J_shape=list(block.J.shape),actual_nnz=block.J.nnz))
            write(blocks/'completed_prefix.json',dict(q_sha256=qtoken(value),frozen_profile_token=token,
                source_manifest_sha256=sha(destination/'input_manifest.json'),completed_blocks=inventory,
                completed_original_rows=sum(r['actual_rows']for r in inventory),mathematical_state_changed=False,
                cache_role='Explicit fresh complete current q/profile stop-point export; original arrays streamed without copies'))
            np.savez(destination/'first_complete_source_normal_numeric.npz',q=value,normal_data=computed.normal.data,
                normal_indices=computed.normal.indices,normal_indptr=computed.normal.indptr,
                normal_shape=np.asarray(computed.normal.shape),gradient=computed.gradient,cost=np.asarray(computed.cost),
                frozen_profile_token=np.asarray(token),source_contract_json=np.asarray(json.dumps(asdict(computed.contract))),
                report_json=np.asarray(json.dumps(computed.report)))
            sealed=seal_complete_same_state_cache(destination,value,source,manifest,token,
                fixed_parameter_indices=fixed,shared_parameter_mask=actual_shared_mask)
            receipt=dict(directory=str(destination),q_sha256=qtoken(value),frozen_profile_token=token,
                actual_complete_rows=computed.contract.row_count,actual_blocks=len(cache.blocks),seal=sealed,
                copied_J_or_normal_arrays=False,full_source_derivative_fresh_at_stop=True)
            write(out/'actual_control_requested_current_complete_cache.json',receipt)
            return receipt
        def reuse_first_normal(factory,parameter_count,**options):
            nonlocal reused,normal_serial
            if not reused:
                if (parameter_count!=len(q)or options['expected_row_count']!=len(source.scales)or
                    options['frozen_profile_token']!=normal.contract.frozen_profile_token or
                    not np.array_equal(options.get('source_state'),q)):
                    raise ValueError('First normal q/profile/source inventory changed; refusing stale numerical reuse')
                reused=True
                print(json.dumps(dict(stage='reuse_exact_first_complete_source_normal',q_sha256=qtoken(q),profile_token=normal.contract.frozen_profile_token)),flush=True)
                return normal
            value=np.asarray(options['source_state'],float)
            normal_serial+=1
            # The existing streaming accumulator already retains one exact
            # r/J state for GN/native replay and releases it before trial costs.
            # Do not duplicate that transient state into every accepted folder.
            computed=original_accumulate(factory,parameter_count,**options)
            print(json.dumps(dict(stage='following_complete_native_source_normal_ready',q_sha256=qtoken(value),
                profile_token=computed.contract.frozen_profile_token,rows=computed.contract.row_count,cost=computed.cost,
                normal_serial=normal_serial,persistent_matrix_cache=False)),flush=True)
            return computed
        normal_module.accumulate_source_normal=reuse_first_normal
        from structured_programme_source_candidate.activated_small_source_blocks import make_activated_small_source_block_refresh
        activated_source_block_refresh=None if strict_source is not None else make_activated_small_source_block_refresh(source)
        try:
            if opt.verify_activated_programme_families:
                from structured_programme_source_candidate.activated_small_source_blocks import verify_cached_programme_family_refresh
                verify_begin=time.perf_counter()
                verification=verify_cached_programme_family_refresh(source,q,normal.native_replay_cache,
                    activated_source_block_refresh,families=(
                        'programme:secondary',
                        'programme:main1',
                        'physical_target:low5_to_main_B_fixed_panhead_support_230021888',
                        'physical_target:low5_to_main_B_fixed_panhead_support_230023424',
                        'physical_target:low5_to_main_B_fixed_panhead_support_230024960',
                        'physical_target:secondary6_to_low_fixed_panhead_lower_support_230408448',
                        'physical_target:secondary6_to_low_fixed_panhead_lower_support_230409472',
                        'physical_target:secondary6_to_low_fixed_panhead_lower_support_230410496',
                        'external_main_to_low',
                        'shared_board_patterns',
                        'fixed_pivot_board_side:okada_sp_installation_0_conditional_z190',
                        'fixed_pivot_board_side:okada_sp_installation_1_conditional_z190',
                        'fixed_pivot_board_side:okada_sp_installation_2_conditional_z190',
                        'strong_high_wide_source',
                        'additional_real_native_ice_curves',
                        'additional_real_native_board_edges',
                    ))
                verification.update(q_sha256=qtoken(q),frozen_profile_token=normal.contract.frozen_profile_token,
                    startup_existing_source_owner_and_retained_normal_cache=True,normal_H_g_recomputed_for_check=False,
                    seconds=time.perf_counter()-verify_begin)
                write(out/'actual_activated_programme_family_equivalence.json',verification)
                print(json.dumps(dict(stage='actual_activated_programme_family_equivalence_passed',
                    **verification)),flush=True)
            bounded_motion_model_factory=None
            if opt.bounded_motion_nuisance and strict_source is None:
                from calibration.solver_runtime.bounded_motion_schur import BoundedMotionSchurModel
                from calibration.solver_runtime.motion_bounded_schur_terms import prepare_programme_bounded_terms
                def bounded_motion_model_factory(value):
                    return BoundedMotionSchurModel(prepare_programme_bounded_terms(source,value,
                        profile_token=_frozen_profile_token(source.updater)),len(value))
            result=solve_joint_source(evaluate,q,source.quadratic,source.scales,out/'solver',source.guard,update_profiles=solver_profiles,
                checkpoint_callback=review,checkpoint_identity=dict(input_manifest=dict(path=str(out/'input_manifest.json'),sha256=sha(out/'input_manifest.json'))),
                checkpoint_arrays=arrays,global_schur_count=len(shared),global_schur_indices=shared,source_block_factory=logged_source_blocks,
                fixed_parameter_indices=fixed,schur_backtracking=True,schur_view_lm_ratio=opt.schur_view_lm_ratio,schur_backtracking_aware_damping=True,
                trial_feasibility_recovery=trial_feasibility_recovery,
                save_current_linearization_cache=save_current_complete_cache,
                cache_native_source_replay=True,source_stage_timing=True,initial_damping=initial_damping,
                reuse_source_jacobian=opt.reuse_source_jacobian,source_jacobian_refresh_ratio=opt.source_jacobian_refresh_ratio,
                adaptive_source_backtracking=opt.adaptive_source_backtracking,
                initial_line_search_fraction=continued_line_search_fraction,
                positive_parameter_indices=pivot_height_columns,positive_parameter_resolution=1e-6,
                shared_parameter_bounds=mount_parameter_bounds,
                shared_linear_constraints=solver_shared_linear_constraints,
                shared_step_curvature_tightening=shared_step_curvature_tightening,
                source_physical_endpoint_event=source_physical_endpoint_event,
                source_trial_acceptance=source_trial_acceptance if opt.truth_restoration_priority else None,
                bounded_motion_model_factory=bounded_motion_model_factory,
                refresh_activated_source_block=activated_source_block_refresh,
                conditional_view_backtracking=opt.conditional_view_backtracking,
                conditional_view_protected_parameter_indices=conditional_view_protected)
        finally:normal_module.accumulate_source_normal=original_accumulate
        review(result.q,dict(stage='final_unaccepted',status=result.status,stop_reason=result.stop_reason))
    except Exception as exc:
        write(out/'interrupted_source_evaluation.json',dict(error_type=type(exc).__name__,error=str(exc),
            seconds_since_start=time.perf_counter()-t,actual_family_times=raw.actual_family_times,
            actor_stats=[]if raw.actors is None else raw.actors.last_stats,
            memory_samples=[]if raw.actors is None else raw.actors.memory_samples,
            preserved_q_sha256=qtoken(q),accuracy_accepted=False,full_video_generated=False))
        raise
    finally:
        if geometry_fd_pool is not None:
            write(out/'actual_absolute_geometry_FD_pool_runtime.json',geometry_fd_pool.report())
            geometry_fd_pool.close()
        from structured_programme_source_candidate.weighted_gram_rows_candidate import close_weighted_gram_pool
        close_weighted_gram_pool()
        if current_pattern_pool is not None:
            write(out/'actual_current_geometry_pattern_pool_runtime.json',current_pattern_pool.report())
            current_pattern_pool.close()
        if raw.actors is not None:raw.actors.close()
        from z190_parallel_pattern_evaluation import close_pattern_pools
        close_pattern_pools()
if __name__=='__main__':main()
