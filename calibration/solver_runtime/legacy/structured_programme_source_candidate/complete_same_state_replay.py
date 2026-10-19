"""Resume an authenticated completed normal without reevaluating its sources."""
from pathlib import Path
import json,hashlib
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix,csc_matrix
from z190_streaming_source_normal import (SourceRowBlock,BlockSignature,SourceBlockContract,StreamingSourceNormal,
    FrozenNativeReplayCache,_checked_block,_check_replay,_state_digest)

def load_complete_same_state_normal(directory,q,source,manifest,profile_token,*,fixed_parameter_indices=None,shared_parameter_mask=None):
    directory=Path(directory);mp=directory/'input_manifest.json'
    old=json.loads(mp.read_text(encoding='utf8'))
    current=json.loads(json.dumps(manifest,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x)))
    guard_bridge=_check_strict_identity(old,current,fixed_parameter_indices=fixed_parameter_indices,shared_parameter_mask=shared_parameter_mask)
    _sealed_inventory(directory,q,source,profile_token)
    # This field records which historical profile was inherited at graph
    # construction, before its required first update. The complete normal
    # authenticates the actual updated profile separately below. On resume
    # the inherited checkpoint is that updated profile, not its predecessor.
    inherited_profile_provenance=dict(previous_graph_seed=old['current_geometry']['profile_source_token'],
        resumed_graph_seed=current['current_geometry']['profile_source_token'],actual_frozen_normal_profile=profile_token)
    # Graph-seed token is provenance. The actual complete profile NPZ arrays,
    # live updater and normal token have already been authenticated above.
    old['current_geometry']=dict(old['current_geometry']);current['current_geometry']=dict(current['current_geometry'])
    old['current_geometry'].pop('profile_source_token');current['current_geometry'].pop('profile_source_token')
    for key in ('current_geometry','source_contract','latest9a8c_reference18_preserved','user_trace_sha256','additional_real_native_ice_curves'):
        if old[key]!=current[key]:raise ValueError('Complete normal source identity differs: '+key)
    with np.load(directory/'first_complete_source_normal_numeric.npz',allow_pickle=False)as z:
        if not np.array_equal(q,z['q']):raise ValueError('Complete normal q differs')
        if str(z['frozen_profile_token'])!=profile_token:raise ValueError('Complete normal frozen profiles differ')
        payload=json.loads(str(z['source_contract_json']))
        if payload['parameter_count']!=len(q)or payload['frozen_profile_token']!=profile_token:
            raise ValueError('Complete normal contract parameter/profile signature differs')
        contract=SourceBlockContract(payload['parameter_count'],payload['row_count'],
            tuple(BlockSignature(**x)for x in payload['signatures']),payload['frozen_profile_token'])
        normal=csc_matrix((z['normal_data'].copy(),z['normal_indices'].copy(),z['normal_indptr'].copy()),shape=tuple(z['normal_shape']))
        gradient=z['gradient'].copy();cost=float(z['cost']);report=json.loads(str(z['report_json']))
    prefix=directory/'initial_normal_source_blocks';completion=json.loads((prefix/'completed_prefix.json').read_text(encoding='utf8'))
    if completion['source_manifest_sha256']!=hashlib.sha256(mp.read_bytes()).hexdigest():raise ValueError('Complete prefix manifest hash changed')
    raw_q_sha256=_array_signature(np.asarray(q,dtype=np.float64))['sha256']
    if (completion['q_sha256']!=raw_q_sha256 or completion['frozen_profile_token']!=profile_token or
            len(completion['completed_blocks'])!=len(contract.signatures) or completion['completed_original_rows']!=contract.row_count):
        raise ValueError('Complete cached block/state/profile inventory differs')
    if (report['fixed_parameter_indices']!=list(fixed_parameter_indices) or normal.shape!=(len(q),len(q)) or
            gradient.shape!=(len(q),) or not np.isfinite(normal.data).all() or not np.isfinite(gradient).all() or not np.isfinite(cost)):
        raise ValueError('Complete sealed normal fixed policy/numerics/dimensions differ')
    blocks=[];cursor=0
    for ordinal,expected in enumerate(contract.signatures):
        with np.load(prefix/(str(ordinal).zfill(3)+'.npz'),allow_pickle=False)as z:
            if str(z['frozen_profile_token'])!=profile_token:raise ValueError('Complete prefix frozen profiles differ')
            if str(z['q_sha256'])!=raw_q_sha256:raise ValueError('Complete prefix exact state differs')
            from npz_readonly import mapped_array
            stored=prefix/(str(ordinal).zfill(3)+'.npz')
            block=SourceRowBlock(mapped_array(stored,'r'),csr_matrix((mapped_array(stored,'J_data'),mapped_array(stored,'J_indices'),mapped_array(stored,'J_indptr')),shape=tuple(z['J_shape']),copy=False),
                mapped_array(stored,'quadratic'),mapped_array(stored,'scales'),json.loads(str(z['ledger_json'])),
                __import__('calibration.solver_runtime.source_loss',fromlist=['source_loss_from_checkpoint']).source_loss_from_checkpoint(z))
        r,J,quadratic,scales,signature=_checked_block(block,len(q),True);_check_replay(signature,ordinal,contract,True)
        stop=cursor+len(r)
        if block.row_ledger.get('original_start')!=cursor or block.row_ledger.get('original_stop')!=stop:
            raise ValueError('Complete cached block source row layout differs')
        if not np.array_equal(quadratic,source.quadratic[cursor:stop])or not np.array_equal(scales,source.scales[cursor:stop]):
            raise ValueError('Current complete source mask/scales differ from authenticated prefix')
        blocks.append(block);cursor=stop
        print(json.dumps(dict(stage='authenticated_saved_source_block_loaded',ordinal=ordinal,rows=len(r))),flush=True)
    if cursor!=len(source.scales)or cursor!=contract.row_count:raise ValueError('Complete cached source rows differ')
    state,state_sha=_state_digest(q,len(q));state=state.copy();state.flags.writeable=False
    if state_sha!=report['same_state_native_replay_retention']['source_state_sha256']:raise ValueError('Complete native replay state signature differs')
    cache=FrozenNativeReplayCache(state,state_sha,contract,tuple(blocks),report['same_state_native_replay_retention'])
    report['complete_replay_initial_profile_provenance']=inherited_profile_provenance
    if guard_bridge:
        report['complete_replay_explicit_compatibility']=guard_bridge
        if guard_bridge['mode']=='explicit_manual_trial_envelope_extension_same_source_normal':
            report['explicit_guard_only_compatibility']=guard_bridge
    return StreamingSourceNormal(normal,gradient,cost,contract,report,cache)


IDENTITY_KEY='persistent_source_replay_identity'
IDENTITY_SCHEMA='rinkspace.strict-complete-source-replay-identity.v1'
INVENTORY_SCHEMA='rinkspace.sealed-complete-source-replay-inventory.v1'
# Only seed provenance and execution telemetry are excluded. New source/domain
# fields remain in the comparison, so an unknown objective fails closed.
PROVENANCE_KEYS=('camera_seed','original_camera_replacement_max_absolute','numerical_current_world_inherited',
    'numerical_independent_circle_material_suffix_inherited','numerical_independent_pattern_placement_suffix_inherited',
    'optimizer_damping_continuation','preparation_only_capture_release','source_jacobian_reuse',
    'adaptive_source_backtracking','accepted_state_review_policy','explicit_mount_working_range_reinitialization',
    'explicit_pivot_outside_reinitialization','explicit_source_domain_pose_feasibility_initialization',
    'absolute_source_camera_initializer','physical_domain_initial_q_sha256','initial_pivot_exterior_domain')
EXECUTION_ENV_KEYS={'RINKSPACE_NORMAL_WORKERS','RINKSPACE_GEOMETRY_FD_WORKERS','RINKSPACE_RESIDUAL_SOURCE_WORKERS',
    'RINKSPACE_RESIDUAL_NATIVE_THREADS','RINKSPACE_NATIVE_THREADS','RINKSPACE_PATTERN_WORKERS',
    'RINKSPACE_MOTION_THREADS','RINKSPACE_SCATTER_THREADS','RINKSPACE_SOLVER_PROFILE'}


def _plain(value):
    return json.loads(json.dumps(value,allow_nan=False,default=lambda x:x.tolist()if isinstance(x,np.ndarray)
        else x.item()if isinstance(x,np.generic)else str(x)))


def _sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(_plain(value),sort_keys=True,separators=(',',':'),ensure_ascii=False,
        allow_nan=False).encode('utf8')).hexdigest()


def _array_signature(value):
    a=np.asarray(value)
    if a.dtype.hasobject:raise ValueError('Object arrays are not replay identity')
    # Hash contiguous original buffers directly; large CSR/source arrays must
    # not create a second bytes object solely for an identity check.
    contiguous=a if a.flags.c_contiguous else np.ascontiguousarray(a)
    digest=hashlib.sha256()
    if contiguous.size:digest.update(memoryview(contiguous).cast('B'))
    return dict(dtype=a.dtype.str,shape=list(a.shape),sha256=digest.hexdigest())


def _source_record_identity(value):
    # Hash shared array references without expanding them into repeated world
    # coordinate lists. A function/object repr cannot authenticate source input.
    if isinstance(value,(np.ndarray,np.generic)):return {'numeric_array':_array_signature(value)}
    from calibration.solver_runtime.original_measured_location_mode import AuthenticatedMeasuredLocation
    if type(value)is AuthenticatedMeasuredLocation:
        return {'authenticated_measured_location':value.source_identity()}
    from source_edge_intervals import SourceNormalIntervals
    if isinstance(value,SourceNormalIntervals):
        return {'source_normal_intervals':dict(normals=_array_signature(value.normals),
            bounds=_array_signature(value.bounds),eligible=_array_signature(value.eligible))}
    if isinstance(value,dict):return {str(k):_source_record_identity(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_source_record_identity(v)for v in value]
    if isinstance(value,Path):return str(value)
    if isinstance(value,float)and not np.isfinite(value):return {'nonfinite_python_float':value.hex()}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise ValueError('Source record must have explicit numeric/JSON identity, not '+type(value).__name__)


def _semantic_manifest(manifest):
    value=deepcopy(_plain(manifest));value.pop(IDENTITY_KEY,None)
    for key in PROVENANCE_KEYS:value.pop(key,None)
    if isinstance(value.get('current_geometry'),dict):value['current_geometry'].pop('profile_source_token',None)
    if isinstance(value.get('native_backend_requested'),dict):
        for key in EXECUTION_ENV_KEYS:value['native_backend_requested'].pop(key,None)
    for key in ('absolute_geometry_fd_parallelism','weighted_gram_parallelism','residual_source_parallelism','current_geometry_pattern_parallelism'):
        if isinstance(value.get(key),dict):
            for field in ('requested_workers','native_threads_per_actor','material_owner'):value[key].pop(field,None)
    if isinstance(value.get('source_worker_backend_installation'),dict):value['source_worker_backend_installation'].pop('already_installed',None)
    compaction=value.get('original_motion_input_compaction')
    if isinstance(compaction,dict):
        for block in compaction.get('blocks',[]):
            for field in ('actual_retained_r_J_check_seconds','same_state_retained_local_r_max_difference',
                    'same_state_retained_local_J_max_difference','retained_r_max_scale','retained_J_max_scale',
                    'same_state_retained_r_J_bitexact'):
                block.pop(field,None)
    def clean_backend(node):
        if isinstance(node,dict):return {k:clean_backend(v)for k,v in node.items()if k not in
            ('module_object_id','stats_at_enable','native_threads','pattern_workers','threads','restored_environment_before')}
        if isinstance(node,list):return [clean_backend(x)for x in node]
        return node
    if 'actual_active_backend_contract'in value:value['actual_active_backend_contract']=clean_backend(value['actual_active_backend_contract'])
    return value


def _file_references(value):
    refs={}
    def walk(node):
        if isinstance(node,dict):
            for name,p in node.items():
                hname='sha256'if name=='path'else name[:-5]+'_sha256'if name.endswith('_path')else None
                digest=node.get(hname)if hname else None
                if isinstance(p,str)and isinstance(digest,str)and len(digest)==64:
                    declared=str(Path(p).resolve());resolved=str(_registered_reference_path(declared))
                    if declared in refs and refs[declared]['sha256']!=digest:
                        raise ValueError('Conflicting source hashes: '+declared)
                    refs[declared]=dict(path=resolved,declared_path=declared,sha256=digest)
            for item in node.values():walk(item)
        elif isinstance(node,list):
            for item in node:walk(item)
    walk(value)
    actual={}
    for item in refs.values():
        if item['path']in actual and actual[item['path']]!=item['sha256']:
            raise ValueError('Conflicting resolved source hashes: '+item['path'])
        actual[item['path']]=item['sha256']
    return [refs[k]for k in sorted(refs)]


def _registered_reference_path(value):
    from calibration.solver_runtime.config import PROJECT_ROOT
    from rinkspace.storage import AssetStore
    store=AssetStore(PROJECT_ROOT);path=Path(value).resolve()
    # Retained original manifests use the project's registered migration
    # aliases. External explicitly hashed binaries remain literal references.
    if any(path.is_relative_to(base)for base in (store.project,store.legacy_root,store.data)):
        return store.resolve(path)
    return path


def _verify_files(records):
    for item in records:
        if 'declared_path'in item and str(_registered_reference_path(item['declared_path']))!=item['path']:
            raise ValueError('Replay registered source resolution changed: '+item['declared_path'])
        if not Path(item['path']).is_file()or _sha(item['path'])!=item['sha256']:
            raise ValueError('Replay bound code/native/source file changed or missing: '+item['path'])


def _runtime_implementation_files(extra=()):
    from calibration.solver_runtime.config import PACKAGE_ROOT,PROJECT_ROOT
    mp=PACKAGE_ROOT/'source_manifest.json';payload=json.loads(mp.read_text(encoding='utf8'))
    refs=[dict(path=str(mp.resolve()),sha256=_sha(mp))]
    # The resolver and its registry are part of the actual file identity gate.
    for path in (PROJECT_ROOT/'src/rinkspace/storage.py',PROJECT_ROOT/'configs/storage.json'):
        refs.append(dict(path=str(path.resolve()),sha256=_sha(path)))
    for row in list(payload['modules'].values())+payload['native_sources']:
        refs.append(dict(path=str((PACKAGE_ROOT/row['path']).resolve()),sha256=row['sha256']))
    for row in payload.get('shared_repo_dependencies',[]):
        refs.append(dict(path=str((PROJECT_ROOT/row['path']).resolve()),sha256=row['sha256']))
    for item in extra:
        refs.append(dict(path=str(Path(item['path']if isinstance(item,dict)else item).resolve()),
            sha256=item['sha256']if isinstance(item,dict)else _sha(item)))
    _verify_files(refs);unique={r['path']:r for r in refs}
    return [unique[k]for k in sorted(unique)]


def _scientific_backend_identity():
    import sys,scipy,cv2
    refs=[]
    # OpenCV performs the original UV/J projection; pin its actual extension,
    # rather than assuming the Python wrapper version proves binary identity.
    for p in Path(cv2.__file__).parent.glob('cv2*.pyd'):
        refs.append(dict(path=str(p.resolve()),sha256=_sha(p)))
    return dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__,
        projection_binary_files=refs)


def build_persistent_source_replay_identity(manifest,*,fixed_parameter_indices,shared_parameter_mask,
        implementation_files=(),source_input_arrays=None,source_input_metadata=None,source_input_records=None):
    """Before saving the manifest: pass actual canonical fixed/shared policy."""
    semantic=_semantic_manifest(manifest);n=int(semantic['full_parameters'])
    fixed=np.asarray(fixed_parameter_indices);shared=np.asarray(shared_parameter_mask)
    if fixed.ndim!=1 or (fixed.size and fixed.dtype.kind not in 'iu')or np.any(fixed<0)or np.any(fixed>=n)or len(np.unique(fixed))!=len(fixed):
        raise ValueError('Actual unique canonical fixed parameter indices required')
    if shared.shape!=(n,)or shared.dtype.kind!='b':raise ValueError('Actual full canonical boolean shared mask required')
    if not np.array_equal(np.flatnonzero(shared),np.asarray(semantic['shared_indices'],int)):
        raise ValueError('Manifest shared indices differ from actual shared mask')
    required=('complete_parameter_layout','geometry_world_indices','current_geometry','source_contract','source_rows',
        'full_parameters','native_backend_requested','camera_mount_working_constraints','camera_pivot_exterior_constraints',
        'original_source_role_retirement','static_motion_source_role_retirement','original_motion_far_board_role_correction',
        'original_motion_input_compaction','original_ring_source_qualification','original_ice_pixel_correction')
    for key in required:
        if key not in semantic:raise ValueError('Replay semantic identity is incomplete: '+key)
    if not isinstance(source_input_arrays,dict)or not source_input_arrays:
        raise ValueError('Actual unchanged source UV/endpoint/sigma/finite-role/layout array inventory required')
    if source_input_records is not None:
        if source_input_metadata is not None:raise ValueError('Pass source_input_records or source_input_metadata, not both')
        source_input_metadata=source_input_records
    if not isinstance(source_input_metadata,dict)or not source_input_metadata:
        raise ValueError('Actual original source clock and typed row/group/finite-role record inventory required')
    refs=_file_references(semantic);_verify_files(refs)
    return dict(schema=IDENTITY_SCHEMA,semantic_manifest_sha256=_json_sha(semantic),
        fixed_parameter_indices=fixed.astype(np.int64).tolist(),shared_parameter_mask=_array_signature(shared),
        actual_original_source_input_arrays={str(k):_array_signature(v)for k,v in sorted(source_input_arrays.items())},
        actual_original_source_input_metadata_sha256=_json_sha(_source_record_identity(source_input_metadata)),
        scientific_backend_identity=_scientific_backend_identity(),
        implementation_files=_runtime_implementation_files(implementation_files),bound_source_domain_and_native_files=refs,
        all_unknown_source_geometry_domain_keys_compared=True)


def _manual_guard_only_bridge(old,current):
    """One frozen, authorized trial-envelope revision; never an r/J domain alias."""
    old_sha='0b76f7e7fbbb6129e1cd7f13233c92edf2a575e8b14f2045ebf7a41d653c10c7'
    new_sha='92be871bbaae911ee241f5d9f9810f267e584bee97db923333d2f79e7222b3ec'
    a,b=_semantic_manifest(old),_semantic_manifest(current);references=[]
    for family in ('native','wide'):
        previous=a.get('highest_direct_human_hard_truth',{}).get(family,{}).get('practical_manual_location_envelope',{})
        revised=b.get('highest_direct_human_hard_truth',{}).get(family,{}).get('practical_manual_location_envelope',{})
        if previous.get('sha256')!=old_sha or revised.get('sha256')!=new_sha:return None
        before,after=deepcopy(previous),deepcopy(revised)
        for row in (before,after):row.pop('path');row.pop('sha256')
        if before!=after:raise ValueError('Manual guard bridge changed non-reference contract fields')
        references.append((previous,revised))
        b['highest_direct_human_hard_truth'][family]['practical_manual_location_envelope']=deepcopy(previous)
    if a!=b:raise ValueError('Guard-only bridge cannot change source/layout/physical domain/backend semantics')
    previous,revised=references[0]
    if references[1]!=references[0]:raise ValueError('Native/wide manual guard policy references differ')
    _verify_files(_file_references({'previous':previous,'revised':revised}))
    left=json.loads(_registered_reference_path(previous['path']).read_text(encoding='utf8'))
    right=json.loads(_registered_reference_path(revised['path']).read_text(encoding='utf8'))
    if right.get('previous_location_policy_reference')!={k:previous[k]for k in ('path','sha256')}:
        raise ValueError('Guard revision does not authenticate its exact previous policy')
    for key in ('schema','canonical_q_sha256','complete_initializer_reference','authority_references',
            'source_UV_sigma_and_historical_packets_unchanged'):
        if left.get(key)!=right.get(key):raise ValueError('Guard-only source authority/initializer changed: '+key)
    if [(r['family'],r['id'],r['full_vertices'])for r in left['records']]!=[
            (r['family'],r['id'],r['full_vertices'])for r in right['records']]:
        raise ValueError('Guard-only complete stroke identities/coverage changed')
    for before,after in zip(left['records'],right['records']):
        for key in ('normal_expansion_px','tangent_expansion_px'):
            if not np.isfinite(after[key])or after[key]<before[key]:raise ValueError('Guard revision is not a finite envelope extension')
    from calibration.solver_runtime.config import PACKAGE_ROOT
    allowed={str(Path(__file__).resolve()):'63d0716ce939020828e987bd07849e955e870c60b6a951c247bd596b6ee08cf9',
        str((PACKAGE_ROOT/'source_manifest.json').resolve()):'ed358fa20fdf7dd98bd6020978ae21545e492bd4a642c9ea475495527d335c6c'}
    old_files=old[IDENTITY_KEY]['implementation_files'];new_files=current[IDENTITY_KEY]['implementation_files']
    if {r['path']:r['sha256']for r in old_files if r['path']in allowed}!=allowed:
        raise ValueError('Guard bridge is not the authenticated previous load-only implementation')
    if {r['path']:r['sha256']for r in new_files if r['path']in allowed}!={p:_sha(p)for p in allowed}:
        raise ValueError('Current guard bridge loader/manifest implementation differs')
    if [r for r in old_files if r['path']not in allowed]!=[r for r in new_files if r['path']not in allowed]:
        raise ValueError('Guard bridge changed an r/J producer/core/human/native implementation')
    return dict(mode='explicit_manual_trial_envelope_extension_same_source_normal',previous_policy=previous,
        revised_policy=revised,changed_feasibility_domain_explicit=True,source_normal_domain_changed=False,
        halfspaces_must_be_recomputed_under_current_guard=True,allowed_load_only_implementation_paths=list(allowed))


_QP_OUTPUT_PATH_KEYS=('RINKSPACE_BENCHMARK_DIAGNOSTIC_DIR',
    'RINKSPACE_BENCHMARK_TIMINGS_PATH','RINKSPACE_SOURCE_RUNTIME_DIR')


def _qp_compare_source_semantics(value):
    result=deepcopy(value)
    if isinstance(result.get('native_backend_requested'),dict):
        for key in _QP_OUTPUT_PATH_KEYS:result['native_backend_requested'].pop(key,None)
    return result


def _qp_step_only_source_cache_bridge(old,current):
    """One exact QP helper revision; cached source normals retain every gate."""
    from calibration.solver_runtime.config import PACKAGE_ROOT
    helper=(PACKAGE_ROOT/'legacy/snapshot/three_frame_refit/program_tracking/z190_global_schur_linear_step.py').resolve()
    old_helper='957008c99f868e9c29db763eeea91be199c35b39c19ff8bba0c15d64ab2fd699'
    new_helper='e6167e8eb08af2ca4bdfa451b0e760e2faf5a9954c08c4d7dd9868579b4e55d3'
    left=old.get(IDENTITY_KEY,{}).get('implementation_files',[])
    right=current.get(IDENTITY_KEY,{}).get('implementation_files',[])
    old_files={r['path']:r['sha256']for r in left}
    new_files={r['path']:r['sha256']for r in right}
    if old_files.get(str(helper))!=old_helper or new_files.get(str(helper))!=new_helper:return None
    allowed={str(helper):old_helper,
        str(Path(__file__).resolve()):'57ac9934c70e8e84168bfceedfb11097677ee76b097d67c2f2b4b3cdcb80ac6c',
        str((PACKAGE_ROOT/'source_manifest.json').resolve()):'ddea74bf90e14e20e6da071efcab2eefb22764560fb0f46ac3cbd8627fb1f254'}
    if {p:old_files.get(p)for p in allowed}!=allowed:
        raise ValueError('QP bridge does not bind the exact original sealed helper/loader/manifest')
    if {p:new_files.get(p)for p in allowed}!={p:_sha(p)for p in allowed}:
        raise ValueError('QP bridge current helper/loader/manifest differs from actual files')
    if [r for r in left if r['path']not in allowed]!=[r for r in right if r['path']not in allowed]:
        raise ValueError('QP bridge changed an r/J producer/core/guard/native implementation')
    a,b=_semantic_manifest(old),_semantic_manifest(current)
    if _qp_compare_source_semantics(a)!=_qp_compare_source_semantics(b):
        raise ValueError('QP bridge changed source/layout/physical domain/backend semantics')
    authority=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_joint_morning_c134965_commonaxis_board79_fulltarget_20261008_184808_452')
    snapshot=authority/'QP_original_helper_authority_snapshot.py'
    proof_path=authority/'actual_QP_primal_working_face_equivalence_check.json'
    proof_sha='c2ea5379d5a11bf54098d6b6bccbed19a905088c5a63d861449e009d89306398'
    _verify_files([dict(path=str(snapshot),sha256=old_helper),
        dict(path=str(proof_path),sha256=proof_sha),dict(path=str(helper),sha256=new_helper)])
    proof=json.loads(proof_path.read_text(encoding='utf8'))
    if (proof.get('schema')!='rinkspace.equivalent-primal-route-shared-QP-proof.v1'
            or proof.get('old_authority_snapshot')!={'path':str(snapshot),'sha256':old_helper}
            or proof.get('new_actual_helper')!={'path':str(helper),'sha256':new_helper}
            or proof.get('all_constraints_retained')is not True
            or proof.get('source_r_J_H_g_profile_domain_or_guard_changed')is not False
            or proof.get('original_final_KKT_check_unchanged')is not True):
        raise ValueError('QP bridge lacks its exact original helper and unchanged source/constraint proof')
    # Outside this optimizer-only function the actual helper must be identical.
    # Comments/newlines are not numerical code; the old snapshot byte SHA above
    # separately binds the complete historical file, without normalization.
    import ast
    def remainder(path):
        tree=ast.parse(path.read_text(encoding='utf8'))
        selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))
            and n.name=='_linearly_constrained_shared_step']
        if len(selected)!=1:raise ValueError('QP bridge helper function identity is ambiguous')
        tree.body.remove(selected[0]);return ast.dump(tree,include_attributes=False)
    if remainder(snapshot)!=remainder(helper):raise ValueError('QP helper revision changed non-QP numerical code')
    return dict(mode='explicit_QP_step_helper_revision_same_source_normal',
        previous_helper=dict(path=str(snapshot),sha256=old_helper),
        revised_helper=dict(path=str(helper),sha256=new_helper),
        equivalence_proof=dict(path=str(proof_path),sha256=proof_sha),
        source_normal_domain_changed=False,source_r_J_H_g_profiles_unchanged=True,
        physical_and_manual_feasibility_domain_unchanged=True,
        optimizer_step_must_be_recomputed=True,cached_QP_solution_reused=False,
        ignored_execution_output_path_keys=list(_QP_OUTPUT_PATH_KEYS),
        allowed_load_only_implementation_paths=list(allowed))



def _restoration_native_QP_source_cache_bridge(old,current):
    """Exact same-source replay after an explicit joint-step policy repair."""
    from calibration.solver_runtime.config import PACKAGE_ROOT
    helper=(PACKAGE_ROOT/'legacy/snapshot/three_frame_refit/program_tracking/z190_global_schur_linear_step.py').resolve()
    driver=(PACKAGE_ROOT/'legacy/run_current_named66_full_programme.py').resolve()
    left=old.get(IDENTITY_KEY,{}).get('implementation_files',[])
    right=current.get(IDENTITY_KEY,{}).get('implementation_files',[])
    old_files={r['path']:r['sha256']for r in left}
    new_files={r['path']:r['sha256']for r in right}
    if (old_files.get(str(helper))!='e6167e8eb08af2ca4bdfa451b0e760e2faf5a9954c08c4d7dd9868579b4e55d3' or new_files.get(str(helper))!='96051dea416062e0215a1d96a9bb5038d6fef6719ffacf7f4b1cce55268dc520'
            or old_files.get(str(driver))!='b6b68f51c313f79f0a82d7a49837739cf2d1b80837da8b0d5962c8d28ee96b36' or new_files.get(str(driver))!='72a49aaa11ec61c72a1428621b1ba4c107bf265a9bd451943ccd45fdeab8eb00'):return None
    allowed={str(helper):'e6167e8eb08af2ca4bdfa451b0e760e2faf5a9954c08c4d7dd9868579b4e55d3',str(driver):'b6b68f51c313f79f0a82d7a49837739cf2d1b80837da8b0d5962c8d28ee96b36',
        str(Path(__file__).resolve()):'b2ff83801f04f9395e88746e7ffa9e6d6e51ee48e7adbab1f0e0a310adc39cd5',
        str((PACKAGE_ROOT/'source_manifest.json').resolve()):'cfd6ad1c87e8835287bced6241ad7afa2a366a395813473fa5e20776931c39ef'}
    if {p:old_files.get(p)for p in allowed}!=allowed:
        raise ValueError('Restoration/QP bridge does not bind exact previous driver/helper/loader/manifest')
    if {p:new_files.get(p)for p in allowed}!={p:_sha(p)for p in allowed}:
        raise ValueError('Restoration/QP bridge current code/manifest differs from actual files')
    if [r for r in left if r['path']not in allowed]!=[r for r in right if r['path']not in allowed]:
        raise ValueError('Restoration/QP bridge changed another r/J producer/core/guard/native implementation')
    a,b=_qp_compare_source_semantics(_semantic_manifest(old)),_qp_compare_source_semantics(_semantic_manifest(current))
    previous=a.get('joint_source_truth_restoration',{});revised=b.get('joint_source_truth_restoration',{})
    if previous.get('enabled')is not True or revised.get('enabled')is not True:
        raise ValueError('Restoration/QP bridge requires the same joint nonlinear truth ratchet')
    changes=dict(original_support_bounds_in_step=(True,False),native_endpoint_inward_branch_in_step=(True,False),
        measured_source_face_curvature_compensation=(True,False),step_group_bound_mode=('original_support','restoration_limits'))
    for key,(before,after)in changes.items():
        if previous.get(key)!=before or revised.get(key)!=after:
            raise ValueError('Restoration/QP bridge has an unexpected numerical step policy: '+key)
        revised[key]=before
    for key in ('legacy_original_step_bounds_requested','legacy_source_face_curvature_compensation_requested'):
        if key in previous or type(revised.get(key))is not bool:
            raise ValueError('Restoration/QP bridge lacks an explicit legacy request/effective policy record')
        revised.pop(key)
    if a!=b:raise ValueError('Restoration/QP bridge changed source/layout/final physical/truth guard/backend semantics')
    proof_path=Path('D:\\CodexData\\RinkSpace_Data\\events\\ankara_2026\\diagnostics\\current56_shared_active_set_filter_20261009\\verified_restoration_native_QP_source_normal_reuse_authority.json');proof_sha='df91e11bbb08171b7710a155dac15619d92133876ddc68e503bde5cbe2228ad5'
    _verify_files([dict(path=str(proof_path),sha256=proof_sha)])
    proof=json.loads(proof_path.read_text(encoding='utf8'))
    if (proof.get('schema')!='rinkspace.verified-public-restoration-native-QP-source-normal-reuse.v1'
            or proof.get('source_r_J_H_g_cost_frozen_profiles_and_all_observations_supports_unchanged')is not True
            or proof.get('physical_final_guard_and_nonlinear_56_group_ratchet_unchanged')is not True
            or proof.get('optimizer_step_policy_explicitly_changed')is not True
            or proof.get('source_LM_matrix_gradient_and_physical_bound_arrays_actual_bitexact')is not True
            or proof.get('actual_optimizer_step_must_be_recomputed')is not True
            or proof.get('cached_buggy_QP_step_not_reused')is not True
            or proof.get('previous_buggy_and_revised_optimizer_step_equality_claimed')is not False):
        raise ValueError('Restoration/QP bridge lacks exact source-normal-only authority')
    _verify_files([proof[key]for key in ('previous_driver','revised_driver','previous_helper','revised_helper',
        'driver_policy_proof','native_QP_proof','actual_current_domain_QP_audit','actual_current_domain_input',
        'actual_source_normal_capture','native_backend')])
    driver_authority=json.loads(Path(proof['driver_policy_proof']['path']).read_text(encoding='utf8'))
    original=Path(proof['previous_driver']['path']).read_text(encoding='utf8')
    for revision in driver_authority['exact_text_replacements']:
        if original.count(revision['previous'])!=revision['occurrences']:
            raise ValueError('Restoration driver original source/policy context changed')
        original=original.replace(revision['previous'],revision['revised'])
    if original!=driver.read_text(encoding='utf8'):
        raise ValueError('Restoration driver changed source/guard code outside exact policy revisions')
    import ast
    def parts(path):
        tree=ast.parse(path.read_text(encoding='utf8'))
        selected=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_linearly_constrained_shared_step']
        if len(selected)!=1:raise ValueError('Native QP helper function identity is ambiguous')
        tree.body.remove(selected[0]);return tree,selected[0]
    before,old_fn=parts(Path(proof['previous_helper']['path']));after,new_fn=parts(helper)
    if ast.dump(before,include_attributes=False)!=ast.dump(after,include_attributes=False):
        raise ValueError('Native QP helper changed numerical code outside constrained QP')
    def assignment(fn,name):
        result=[n for n in fn.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets)]
        if len(result)!=1:raise ValueError('Native QP original row/gate assignment is ambiguous')
        return ast.dump(result[0],include_attributes=False)
    for name in ('finite_lower','finite_upper','eye','rows','targets','tolerance','kkt_tolerance'):
        if assignment(old_fn,name)!=assignment(new_fn,name):
            raise ValueError('Native QP bridge changed original row/gate assignment: '+name)
    def gate(fn):
        result=[n for n in fn.body if isinstance(n,ast.If)and any(isinstance(v,ast.Constant)and v.value=='Shared half-space quadratic solution failed its KKT check'for v in ast.walk(n))]
        if len(result)!=1:raise ValueError('Native QP final KKT gate is ambiguous')
        return ast.dump(result[0],include_attributes=False)
    if gate(old_fn)!=gate(new_fn):raise ValueError('Native QP bridge changed original final KKT gate')
    relocations=[dict(key=key,previous=old.get('native_backend_requested',{}).get(key),revised=current.get('native_backend_requested',{}).get(key))
        for key in _QP_OUTPUT_PATH_KEYS if old.get('native_backend_requested',{}).get(key)!=current.get('native_backend_requested',{}).get(key)]
    return dict(mode='explicit_verified_restoration_native_QP_same_source_normal',
        source_normal_reuse_authority=dict(path=str(proof_path),sha256=proof_sha),
        source_normal_domain_changed=False,source_r_J_H_g_cost_and_frozen_profiles_unchanged=True,
        physical_and_final_nonlinear_56_group_guard_unchanged=True,
        optimizer_step_policy_explicitly_changed=True,optimizer_step_must_be_recomputed=True,
        cached_QP_solution_reused=False,previous_buggy_and_revised_optimizer_step_equality_claimed=False,
        effective_joint_step_domain='existing restoration_limits',effective_joint_curvature=False,
        ignored_execution_output_path_keys=list(_QP_OUTPUT_PATH_KEYS),execution_output_path_relocations=relocations,
        allowed_load_only_implementation_paths=list(allowed))



def _restoration_damping_actual_provenance_pair(old,current,proof):
    """Authenticate only this cache's graph-seed and completed-check telemetry."""
    p=proof['actual_initialization_provenance']
    _verify_files([p[k]for k in('original_cached_manifest','actual_new_owner_manifest','sealed_inventory','actual_normal_q')])
    if p.get('source_input_arrays_metadata_bound_files_fixed_shared_and_backend_actual_exact')is not True:
        raise ValueError('Actual numerical cache bridge has no exact source input identity proof')
    a,b=_qp_compare_source_semantics(_semantic_manifest(old)),_qp_compare_source_semantics(_semantic_manifest(current))
    previous=a['joint_source_truth_restoration']['complete_initial_q_sha256']
    current_q=b['joint_source_truth_restoration']['complete_initial_q_sha256']
    if previous!=p['original_graph_initial_q_sha256']or current_q!=p['current_graph_and_sealed_q_sha256']:
        raise ValueError('Actual numerical bridge graph seed is not its exact sealed state')
    inventory=json.loads(Path(p['sealed_inventory']['path']).read_text(encoding='utf8'))
    if inventory['q']['sha256']!=current_q:raise ValueError('Actual numerical bridge current graph and cache q differ')
    with np.load(p['actual_normal_q']['path'],allow_pickle=False)as z:
        q=z['q']
        if _array_signature(q)!=inventory['q']:raise ValueError('Actual numerical bridge sealed q array differs')
        if q[np.asarray(p['sealed_current_placement_indices'],int)].tolist()!=p['sealed_current_placement_values']:
            raise ValueError('Actual numerical bridge placement proof is not from exact sealed q')
    previous_initializer=a['independent_pattern_new_slot_initializer'];current_initializer=b['independent_pattern_new_slot_initializer']
    if (previous_initializer.get('initializer_is_not_observation')is not True
            or current_initializer.get('initializer_is_not_observation')is not True
            or previous_initializer['parameter_values']!=p['original_graph_placement_values']
            or current_initializer['parameter_values']!=p['sealed_current_placement_values']
            or b['independent_pattern_placement_parameter_indices']!=p['sealed_current_placement_indices']):
        raise ValueError('Actual numerical bridge changed original placement initializer or sealed coordinates')
    current_initializer['parameter_values']=deepcopy(previous_initializer['parameter_values'])
    b['joint_source_truth_restoration']['complete_initial_q_sha256']=previous
    for ordinal in p['compaction_diagnostic_block_indices']:
        before=a['original_motion_input_compaction']['blocks'][ordinal]
        after=b['original_motion_input_compaction']['blocks'][ordinal]
        if (before.get('same_state_retained_r_J_passed_tight_compaction_numerical_tolerance')is not True
                or after.get('same_state_retained_r_J_passed_tight_compaction_numerical_tolerance')is not True
                or before.get('same_state_retained_CSR_row_column_identity_bitexact')is not True
                or after.get('same_state_retained_CSR_row_column_identity_bitexact')is not True):
            raise ValueError('Actual numerical bridge cannot normalize a failed compaction value/J check')
        for block in(before,after):
            fraction=block['compaction_numerical_tolerance']['J_max_error_fraction']
            residual=block['same_exposure_actual_r_max']
            if not np.isfinite(fraction)or not 0<=fraction<=1 or not np.isfinite(residual)or residual<0:
                raise ValueError('Actual numerical bridge compaction diagnostic is invalid or above original gate')
        after['compaction_numerical_tolerance']['J_max_error_fraction']=before['compaction_numerical_tolerance']['J_max_error_fraction']
        after['same_exposure_actual_r_max']=before['same_exposure_actual_r_max']
    return a,b

def _restoration_damping_only_source_cache_bridge(old,current):
    """One audited numerical LM-control revision, retaining source cache gates."""
    from calibration.solver_runtime.config import PACKAGE_ROOT
    solver=(PACKAGE_ROOT/'legacy/snapshot/three_frame_refit/program_tracking/z190_joint_source_solver.py').resolve()
    loader=(PACKAGE_ROOT/'legacy/structured_programme_source_candidate/complete_same_state_replay.py').resolve()
    registry=(PACKAGE_ROOT/'source_manifest.json').resolve()
    left=old.get(IDENTITY_KEY,{}).get('implementation_files',[])
    right=current.get(IDENTITY_KEY,{}).get('implementation_files',[])
    old_files={r['path']:r['sha256']for r in left};new_files={r['path']:r['sha256']for r in right}
    if old_files.get(str(solver))!='8785e8c5db79389c7de3b4aac5f368928b13651c1b2db0b79eb4e79dc3f905cd' or new_files.get(str(solver))!='519357b823d886ab40c0a93de5167e83679a0e5a04d7e008c8f431001da7db7c':return None
    allowed={str(solver):'8785e8c5db79389c7de3b4aac5f368928b13651c1b2db0b79eb4e79dc3f905cd',str(loader):'b7662cea84ea84c1cdd8539d3c6c824ea5d9324274a597d9231597ef6ae042b7',str(registry):'1f5a7a65cb57b49237541a0e3cf3d376dbdfdb9c2882f08ac722252dc2c23a9b'}
    if {p:old_files.get(p)for p in allowed}!=allowed:
        raise ValueError('Numerical damping bridge previous solver/loader/registry differs')
    if {p:new_files.get(p)for p in allowed}!={p:_sha(p)for p in allowed}:
        raise ValueError('Numerical damping bridge current solver/loader/registry differs from actual files')
    if [r for r in left if r['path']not in allowed]!=[r for r in right if r['path']not in allowed]:
        raise ValueError('Numerical damping bridge changed another source/guard/native implementation')
    proof_path=Path('D:\\CodexData\\RinkSpace_Data\\events\\ankara_2026\\diagnostics\\current56_shared_active_set_filter_20261009\\verified_restoration_damping_actual_initialization_source_normal_reuse_authority.json');proof_sha='58bba47893c50e6a026816f89f9ab744a6906bc0665b67c24a35b474c24967dc'
    _verify_files([dict(path=str(proof_path),sha256=proof_sha)])
    proof=json.loads(proof_path.read_text(encoding='utf8'))
    if (proof.get('schema')!='rinkspace.exact-truth-restoration-numerical-damping-source-normal-reuse.v1'
            or proof.get('source_r_J_H_g_cost_frozen_profiles_observations_sigmas_weights_layout_fixed_shared_and_all_guards_unchanged')is not True
            or proof.get('changes_next_numerical_LM_control_only')is not True
            or proof.get('actual_optimizer_step_must_be_recomputed')is not True
            or proof.get('previous_and_revised_optimizer_steps_equality_claimed')is not False
            or proof.get('cached_QP_step_reused')is not False):
        raise ValueError('Numerical damping bridge lacks exact source-normal-only authority')
    _verify_files([proof[key]for key in('previous_solver','revised_solver','previous_loader','previous_source_manifest','controls')])
    a,b=_restoration_damping_actual_provenance_pair(old,current,proof)
    if a!=b:raise ValueError('Numerical damping bridge changed original source/profile/layout/physical/truth semantics')
    if (proof['previous_solver']['sha256']!=old_files[str(solver)]
            or proof['revised_solver']['sha256']!=new_files[str(solver)]
            or proof['previous_loader']['sha256']!=old_files[str(loader)]
            or proof['previous_source_manifest']['sha256']!=old_files[str(registry)]):
        raise ValueError('Numerical damping proof does not bind exact cache/current implementation')
    original=Path(proof['previous_solver']['path']).read_text(encoding='utf8')
    reconstructed=original
    for change in proof['exact_solver_text_replacements']:
        if reconstructed.count(change['previous'])!=change['occurrences']:
            raise ValueError('Numerical damping original control context changed')
        reconstructed=reconstructed.replace(change['previous'],change['revised'])
    revised=solver.read_text(encoding='utf8')
    if reconstructed!=revised:
        raise ValueError('Numerical damping solver changed outside exact reviewed control revisions')
    import ast
    def source_parts(text):
        tree=ast.parse(text)
        tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef)or n.name not in('accepted_schur_damping','solve_joint_source')]
        return ast.dump(tree,include_attributes=False)
    if source_parts(original)!=source_parts(revised):
        raise ValueError('Numerical damping bridge changed a source/profile/J producer or numerical helper')
    if json.loads(Path(proof['controls']['path']).read_text(encoding='utf8')).get('passed')is not True:
        raise ValueError('Numerical damping actual-alpha/default/guard controls failed')
    relocations=[dict(key=key,previous=old.get('native_backend_requested',{}).get(key),revised=current.get('native_backend_requested',{}).get(key))
        for key in _QP_OUTPUT_PATH_KEYS if old.get('native_backend_requested',{}).get(key)!=current.get('native_backend_requested',{}).get(key)]
    return dict(mode='explicit_truth_restoration_numerical_damping_same_source_normal',
        source_normal_reuse_authority=dict(path=str(proof_path),sha256=proof_sha),
        source_r_J_H_g_cost_and_profiles_unchanged=True,original_all_physical_truth_guards_unchanged=True,
        changes_next_numerical_LM_control_only=True,optimizer_step_must_be_recomputed=True,
        cached_QP_solution_reused=False,previous_and_revised_optimizer_steps_equality_claimed=False,
        ignored_execution_output_path_keys=list(_QP_OUTPUT_PATH_KEYS),execution_output_path_relocations=relocations,
        allowed_load_only_implementation_paths=list(allowed))

def _check_strict_identity(old,current,*,append_normalized_pair=None,fixed_parameter_indices=None,shared_parameter_mask=None):
    bridge=None if append_normalized_pair is not None else _manual_guard_only_bridge(old,current)
    if bridge is None and append_normalized_pair is None:
        bridge=_qp_step_only_source_cache_bridge(old,current)
    if bridge is None and append_normalized_pair is None:
        bridge=_restoration_native_QP_source_cache_bridge(old,current)
    if bridge is None and append_normalized_pair is None:
        bridge=_restoration_damping_only_source_cache_bridge(old,current)
    if bridge is None and append_normalized_pair is None:
        bridge=_q4511_truth_QP_only_source_cache_bridge(old,current)
    allowed=set(bridge['allowed_load_only_implementation_paths'])if bridge else set()
    for manifest in (old,current):
        identity=manifest.get(IDENTITY_KEY)
        if not isinstance(identity,dict)or identity.get('schema')!=IDENTITY_SCHEMA:
            raise ValueError('Saved/current cache lacks strict source identity; recompute and seal')
        if identity.get('semantic_manifest_sha256')!=_json_sha(_semantic_manifest(manifest)):
            raise ValueError('Strict source semantic manifest signature changed')
        _verify_files([r for r in identity['implementation_files']if r['path']not in allowed]);_verify_files(identity['bound_source_domain_and_native_files'])
        backend=identity.get('scientific_backend_identity')
        if backend!=_scientific_backend_identity():raise ValueError('Actual scientific projection/backend environment differs')
    left,right=deepcopy(old[IDENTITY_KEY]),deepcopy(current[IDENTITY_KEY])
    for record in (left,right):
        record.pop('semantic_manifest_sha256');record.pop('bound_source_domain_and_native_files')
        if bridge:record['implementation_files']=[r for r in record['implementation_files']if r['path']not in allowed]
    if left!=right:raise ValueError('Replay code/native DLL/fixed/shared policy identity differs')
    a,b=append_normalized_pair or (_semantic_manifest(old),_semantic_manifest(current))
    if bridge:
        if bridge['mode']=='explicit_verified_restoration_native_QP_same_source_normal':
            a,b=_qp_compare_source_semantics(a),_qp_compare_source_semantics(b)
            for key in ('original_support_bounds_in_step','native_endpoint_inward_branch_in_step',
                    'measured_source_face_curvature_compensation','step_group_bound_mode'):
                b['joint_source_truth_restoration'][key]=a['joint_source_truth_restoration'][key]
            for key in ('legacy_original_step_bounds_requested','legacy_source_face_curvature_compensation_requested'):
                b['joint_source_truth_restoration'].pop(key)
        elif bridge['mode']=='explicit_truth_restoration_numerical_damping_same_source_normal':
            proof=json.loads(Path(bridge['source_normal_reuse_authority']['path']).read_text(encoding='utf8'))
            a,b=_restoration_damping_actual_provenance_pair(old,current,proof)
        elif bridge['mode']in ('explicit_QP_step_helper_revision_same_source_normal','explicit_q4511_truth_QP_same_source_normal'):
            # Compare-only telemetry normalization: original stored semantic
            # signatures above still authenticate their original full values.
            a,b=_qp_compare_source_semantics(a),_qp_compare_source_semantics(b)
        else:
            for family in ('native','wide'):
                b['highest_direct_human_hard_truth'][family]['practical_manual_location_envelope']=deepcopy(
                    a['highest_direct_human_hard_truth'][family]['practical_manual_location_envelope'])
    if a!=b:
        keys=sorted(k for k in set(a)|set(b)if a.get(k)!=b.get(k))
        raise ValueError('Replay source/layout/physical-domain/backend differs: '+','.join(keys))
    if fixed_parameter_indices is None or shared_parameter_mask is None:
        raise ValueError('Replay requires current actual fixed indices and full shared mask')
    if old[IDENTITY_KEY]['fixed_parameter_indices']!=np.asarray(fixed_parameter_indices,int).tolist():
        raise ValueError('Actual fixed parameter policy differs')
    if old[IDENTITY_KEY]['shared_parameter_mask']!=_array_signature(shared_parameter_mask):
        raise ValueError('Actual shared parameter mask differs')
    return bridge


def _authenticate_profiles(directory,updater,token):
    from z190_joint_source_solver import _frozen_profile_token
    values=[(str(k),np.asarray(v))for k,v in updater.iter_checkpoint_arrays()]
    with np.load(Path(directory)/'actual_initial_frozen_profile_arrays.npz',allow_pickle=False)as z:
        if z.files!=[k for k,v in values]:raise ValueError('Frozen profile array inventory/order differs')
        for k,v in values:
            if _array_signature(z[k])!=_array_signature(v):raise ValueError('Frozen profile dtype/shape/bytes differ: '+k)
    if _frozen_profile_token(updater)!=token:raise ValueError('Actual frozen profile token differs')
    return [dict(name=k,**_array_signature(v))for k,v in values]


def seal_complete_same_state_cache(directory,q,source,manifest,profile_token,*,fixed_parameter_indices,shared_parameter_mask):
    """Seal actual existing-format cache only AFTER all blocks/normal are saved."""
    directory=Path(directory);old=json.loads((directory/'input_manifest.json').read_text(encoding='utf8'))
    _check_strict_identity(old,_plain(manifest),fixed_parameter_indices=fixed_parameter_indices,shared_parameter_mask=shared_parameter_mask)
    profiles=_authenticate_profiles(directory,source.updater,profile_token)
    cp=directory/'initial_normal_source_blocks/completed_prefix.json';completion=json.loads(cp.read_text(encoding='utf8'))
    with np.load(directory/'first_complete_source_normal_numeric.npz',allow_pickle=False)as z:
        contract=json.loads(str(z['source_contract_json']));report=json.loads(str(z['report_json']))
        if (_array_signature(q)!=_array_signature(z['q'])or str(z['frozen_profile_token'])!=profile_token or
                contract['parameter_count']!=len(q)or contract['frozen_profile_token']!=profile_token or
                report['fixed_parameter_indices']!=list(fixed_parameter_indices)or contract['row_count']!=len(source.scales)or
                len(contract['signatures'])!=len(completion['completed_blocks'])):
            raise ValueError('Cannot seal incomplete/wrong state/profile/fixed policy normal')
    names=['input_manifest.json','actual_initial_normal_state.npz','actual_initial_frozen_profile_arrays.npz','first_complete_source_normal_numeric.npz']
    names+=['initial_normal_source_blocks/'+str(i).zfill(3)+'.npz'for i in range(len(contract['signatures']))]
    payload=dict(schema=INVENTORY_SCHEMA,q=_array_signature(q),frozen_profile_token=profile_token,profile_arrays=profiles,
        source_scales=_array_signature(source.scales),source_quadratic=_array_signature(source.quadratic),
        strict_identity_sha256=_json_sha(old[IDENTITY_KEY]),files=[dict(relative_path=p,sha256=_sha(directory/p))for p in names])
    target=directory/'strict_complete_source_replay_inventory.json'
    with target.open('x',encoding='utf8')as stream:json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
    completion['strict_replay_inventory']=dict(relative_path=target.name,sha256=_sha(target))
    cp.write_text(json.dumps(completion,ensure_ascii=False,indent=2),encoding='utf8')
    return dict(path=str(target),sha256=_sha(target),actual_profile_arrays=len(profiles))


def _sealed_inventory(directory,q,source,token,*,append=False):
    directory=Path(directory);prefix=json.loads((directory/'initial_normal_source_blocks/completed_prefix.json').read_text(encoding='utf8'))
    bound=prefix.get('strict_replay_inventory')
    if not isinstance(bound,dict):raise ValueError('Cache lacks actual sealed r/J/H/g/profile inventory; no legacy recovery')
    path=directory/bound['relative_path']
    if path.parent.resolve()!=directory.resolve()or _sha(path)!=bound['sha256']:raise ValueError('Sealed inventory path/hash differs')
    payload=json.loads(path.read_text(encoding='utf8'))
    if payload['schema']!=INVENTORY_SCHEMA or payload['q']!=_array_signature(q)or payload['frozen_profile_token']!=token:
        raise ValueError('Sealed replay exact q/profile differs')
    for item in payload['files']:
        p=(directory/item['relative_path']).resolve()
        if not p.is_relative_to(directory.resolve())or _sha(p)!=item['sha256']:raise ValueError('Sealed block/normal/state/profile bytes changed')
    old=json.loads((directory/'input_manifest.json').read_text(encoding='utf8'))
    if payload['strict_identity_sha256']!=_json_sha(old[IDENTITY_KEY]):raise ValueError('Sealed strict replay policy changed')
    if payload['profile_arrays']!=_authenticate_profiles(directory,source.updater,token):raise ValueError('Sealed profile arrays changed')
    if not append and (payload['source_scales']!=_array_signature(source.scales)or payload['source_quadratic']!=_array_signature(source.quadratic)):
        raise ValueError('Actual complete source scales/quadratic mask differ')
    return payload


def safe_same_state_replay_predicate(directory,q,source,manifest,profile_token,*,fixed_parameter_indices,shared_parameter_mask):
    """Preflight gate. Missing historical r/J files always require fresh work."""
    try:
        directory=Path(directory);old=json.loads((directory/'input_manifest.json').read_text(encoding='utf8'))
        bridge=_check_strict_identity(old,_plain(manifest),fixed_parameter_indices=fixed_parameter_indices,shared_parameter_mask=shared_parameter_mask)
        _sealed_inventory(directory,q,source,profile_token)
        return dict(allowed=True,mode=bridge['mode']if bridge else 'identical_complete_target',
            same_q_profiles_source_policy_verified=True,explicit_cache_compatibility=bridge,
            explicit_guard_only_compatibility=bridge if bridge and bridge['mode']=='explicit_manual_trial_envelope_extension_same_source_normal'else None)
    except (OSError,KeyError,TypeError,ValueError)as exc:
        return dict(allowed=False,mode='fresh_complete_source_required',reason=str(exc))

def _q4511_truth_QP_only_source_cache_bridge(old,current):
    """Frozen q4511 source normal; reviewed hard-query/step code only."""
    from calibration.solver_runtime.config import PACKAGE_ROOT
    import ast
    human=(PACKAGE_ROOT/'legacy/structured_programme_source_candidate/human_stroke_truth.py').resolve()
    qp=(PACKAGE_ROOT/'legacy/snapshot/three_frame_refit/program_tracking/z190_global_schur_linear_step.py').resolve()
    loader=(PACKAGE_ROOT/'legacy/structured_programme_source_candidate/complete_same_state_replay.py').resolve()
    registry=(PACKAGE_ROOT/'source_manifest.json').resolve()
    helper=(human.parent/'normal_scanline_human_query_acceleration.py').resolve()
    kernel=(human.parent/'native/normal_scanline_human_query.c').resolve()
    left=old.get(IDENTITY_KEY,{}).get('implementation_files',[])
    right=current.get(IDENTITY_KEY,{}).get('implementation_files',[])
    old_files={r['path']:r['sha256']for r in left};new_files={r['path']:r['sha256']for r in right}
    if old_files.get(str(human))!='f6f107ab73c4f9a23c12c5109d700366ca6377cc36e9f73dcb994883ee5bea78' or new_files.get(str(human))!='db2be46a9550da4a17739db2426b0a82f3677f50ff2a5befea2a815d289e3a84':return None
    if old_files.get(str(qp))!='e347dffb951c37ba975ede50d826e638cd9f9b35359fbf54a7af52f471384fa1' or new_files.get(str(qp))!='b06b12c27ae6118e83c4c1d75ccf5b7868eaddfc3c7e9fe8996ba694d6959428':return None
    authority_path=Path('D:\\CodexData\\RinkSpace_Data\\events\\ankara_2026\\evidence\\user_video_review_20261009\\a13_complete_programme\\combined_current_review_20261009\\new_C0_C_Lane_user_strokes_20261009\\actual_q4511_truth_constraint_performance\\q4511_truth_QP_source_normal_only_authority.json')
    proof=json.loads(authority_path.read_text(encoding='utf8'))
    if proof.get('schema')!='rinkspace.q4511-reviewed-truth-QP-source-normal-only-authority.v1' or proof.get('ready')is not True:
        raise ValueError('q4511 load-only bridge authority is incomplete')
    _verify_files([proof[k]for k in ('original_cached_manifest','sealed_inventory','old_human','old_QP','old_loader','truth_all82_proof','QP_original_gate_proof')])
    if _plain(old)!=json.loads(Path(proof['original_cached_manifest']['path']).read_text(encoding='utf8')):
        raise ValueError('q4511 bridge is not the exact original sealed manifest')
    inventory=json.loads(Path(proof['sealed_inventory']['path']).read_text(encoding='utf8'))
    if inventory['q']['sha256']!='4511ce2da459573e4ce1dcf0b0a5185fc94284b99ca40dc28d9c7f923f224af5' or inventory['frozen_profile_token']!='af561db70289ad205d5e1124ae23640861dad8d7ceba79b4f32c7f5697e3f351':
        raise ValueError('q4511 bridge exact q/profile authority changed')
    previous={str(human):proof['old_human']['sha256'],str(qp):proof['old_QP']['sha256'],str(loader):proof['old_loader']['sha256'],str(registry):'42b9c0a4af8800767a05990ec4bfe3391e5b17cf46f2d7b250f1dcf7b26a55b4'}
    if {p:old_files.get(p)for p in previous}!=previous:raise ValueError('q4511 bridge previous implementation identity differs')
    expected={str(human):'db2be46a9550da4a17739db2426b0a82f3677f50ff2a5befea2a815d289e3a84',str(qp):'b06b12c27ae6118e83c4c1d75ccf5b7868eaddfc3c7e9fe8996ba694d6959428',
        str(helper):'078ab16ac1c6bce0517ef3e7d7dbadafb99cca9298ec7ad0fd8540229945e308',str(kernel):'817ca0b1dfe4bb2b009a8567a6498419de7421d7943d46e4e856df016f943019'}
    expected.update({str(loader):_sha(loader),str(registry):_sha(registry)})
    if {p:new_files.get(p)for p in expected}!=expected:raise ValueError('q4511 bridge current exact implementation identity differs')
    _verify_files([dict(path=p,sha256=h)for p,h in expected.items()])
    allowed=set(expected)
    if [r for r in left if r['path']not in allowed]!=[r for r in right if r['path']not in allowed]:
        raise ValueError('q4511 bridge changed another source/guard/backend implementation')
    def tree(path):return ast.parse(Path(path).read_text(encoding='utf8'))
    def source_remainder(t):
        t.body=[n for n in t.body if not isinstance(n,(ast.FunctionDef,ast.ClassDef))or n.name not in ('finite_stroke_queries','_confirmed_pure_query_memo','UserConfirmedSourceTruth')]
        return ast.dump(t,include_attributes=False)
    if source_remainder(tree(proof['old_human']['path']))!=source_remainder(tree(human)):
        raise ValueError('q4511 bridge human source r/J or another producer changed')
    def qp_parts(t):
        selected=[n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='_linearly_constrained_shared_step']
        if len(selected)!=1:raise ValueError('q4511 QP function identity is ambiguous')
        t.body.remove(selected[0]);return ast.dump(t,include_attributes=False),selected[0]
    before,old_qp=qp_parts(tree(proof['old_QP']['path']));after,new_qp=qp_parts(tree(qp))
    if before!=after:raise ValueError('q4511 bridge QP changed code outside numerical step')
    def kkt_gate(fn):
        nodes=[n for n in fn.body if isinstance(n,ast.If)and any(isinstance(v,ast.Constant)and v.value=='Shared half-space quadratic solution failed its KKT check'for v in ast.walk(n))]
        if len(nodes)!=1:raise ValueError('q4511 original KKT gate is ambiguous')
        return ast.dump(nodes[0],include_attributes=False)
    if kkt_gate(old_qp)!=kkt_gate(new_qp):raise ValueError('q4511 bridge changed original KKT acceptance')
    def loader_remainder(t,revised):
        if revised:
            extra=[n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='_q4511_truth_QP_only_source_cache_bridge']
            if len(extra)!=1:raise ValueError('q4511 bridge loader scope is ambiguous')
            t.body.remove(extra[0]);check=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='_check_strict_identity')
            calls=[n for n in check.body if isinstance(n,ast.If)and any(isinstance(v,ast.Call)and isinstance(v.func,ast.Name)and v.func.id=='_q4511_truth_QP_only_source_cache_bridge'for v in ast.walk(n))]
            if len(calls)!=1:raise ValueError('q4511 bridge dispatch is ambiguous')
            check.body.remove(calls[0])
        return ast.dump(t,include_attributes=False)
    # The only other loader revision adds this mode to compare-only execution telemetry.
    original_loader=Path(proof['old_loader']['path']).read_text(encoding='utf8')
    original_loader=original_loader.replace("elif bridge['mode']=='explicit_QP_step_helper_revision_same_source_normal':","elif bridge['mode']in ('explicit_QP_step_helper_revision_same_source_normal','explicit_q4511_truth_QP_same_source_normal'):")
    if loader_remainder(ast.parse(original_loader),False)!=loader_remainder(tree(loader),True):
        raise ValueError('q4511 bridge changed other loader/authentication gates')
    truth=json.loads(Path(proof['truth_all82_proof']['path']).read_text(encoding='utf8'))
    if truth.get('passed')is not True or truth['combined_candidate']['sha256']!=expected[str(human)] or any(not f['report_byte_exact']or any(not c['byte_exact']for c in f['comparisons'].values())for f in truth['families'].values()):
        raise ValueError('q4511 bridge lacks actual all82 matrix/lower/response equivalence')
    if _qp_compare_source_semantics(_semantic_manifest(old))!=_qp_compare_source_semantics(_semantic_manifest(current)):
        raise ValueError('q4511 bridge changed source/layout/UV/sigma/physical/truth semantics')
    return dict(mode='explicit_q4511_truth_QP_same_source_normal',source_normal_reuse_authority=dict(path=str(authority_path),sha256=_sha(authority_path)),
        source_r_J_H_g_cost_profiles_UV_sigmas_masks_rows_unchanged=True,original_all_truth_and_physical_guards_unchanged=True,
        current_constraints_and_QP_step_must_be_recomputed=True,cached_QP_solution_reused=False,
        allowed_load_only_implementation_paths=sorted(allowed),ignored_execution_output_path_keys=list(_QP_OUTPUT_PATH_KEYS))

