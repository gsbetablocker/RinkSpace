"""Independent whole-source replay of a complete immutable physical state.

Source identity, true implicit RS, finite material/branch, genuine query line,
raw pixel coverage, and original grouped statistics are separate ledgers.
Optimizer success is never a physical certificate.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,time,sys
import numpy as np
from .data import Observations,sha256_file
from .pipeline import read_run
from .runtime import atomic_json,atomic_npz
from .native_camera import VERSION as CAMERA_VERSION
from .native_scene import NativeScene,camera_from_event,VERSION as SCENE_VERSION
from .native_query import load_query_packet,recover_query_roots,rational_query
from .semantic_order import audit_source_runs
from .native_query_certificate import certify_profile_roots
from .semantic_resolution import load_semantics,resolve_materials
from .semantic_exclusion import verify_exclusion_source
from .native_query_global import certify_global_profiles,pack_proof_tree

VERSION='rinkfit.independent_native_replay.active_source.v1.2'


def read_array(root,spec):
    path=Path(root)/spec['file']
    if sha256_file(path)!=spec['sha256']:raise ValueError('immutable binary changed: '+str(path))
    a=np.fromfile(path,dtype=spec['dtype'])
    if a.size!=int(np.prod(spec['shape'])):raise ValueError('immutable binary shape mismatch')
    return a.reshape(spec['shape'])


def row_state(state,state_path,n):
    if 'row_state' in state:
        arrays={k:np.asarray(v) for k,v in state['row_state'].items()}
    elif 'row_state_files' in state:
        arrays={k:read_array(Path(state_path).parent,v) for k,v in state['row_state_files'].items()}
    else:raise ValueError('Complete snapshot lacks explicit all-source station/branch/root row state')
    for key in ['source_rows','stations','branches','root_witnesses']:
        if key not in arrays or arrays[key].shape!=(n,):raise ValueError('Incomplete row state: '+key)
    if not np.array_equal(arrays['source_rows'],np.arange(n)):raise ValueError('snapshot source row order changed')
    branch=arrays['branches']
    if not np.isfinite(branch).all() or np.any(branch!=np.floor(branch)) or np.any(branch<0):raise ValueError('snapshot has illegal material branch identities')
    if np.any(arrays['root_witnesses']<0) or np.any(arrays['root_witnesses']>1079):raise ValueError('snapshot lifted RS row outside legal image interval')
    if not np.isfinite(arrays['stations']).all() or not np.isfinite(arrays['root_witnesses']).all():raise ValueError('nonfinite full row state')
    return arrays


def semantic_arrays(root,n,manifest_path=None,expected_sha256=None,*,include_resolution=False):
    path=Path(manifest_path) if manifest_path is not None else Path(root)/'semantic_packet/manifest.json'
    m,arrays,path,resolution=load_semantics(path,n,expected_sha256=expected_sha256)
    if 'profile_normal' in arrays:arrays['normal']=arrays['profile_normal']
    return (m,arrays,path,resolution) if include_resolution else (m,arrays,path)


def _group_summary(keys,error,ratio,valid,active=None):
    active=np.ones(len(valid),bool) if active is None else active
    result=[]
    for k in np.unique(keys):
        rows=keys==k;selected=rows&active;v=valid[selected];d=error[selected][v];q=ratio[selected][v]
        result.append({'id':int(k) if isinstance(k,np.integer) else str(k),'rows':int(rows.sum()),'valid_rows':int(v.sum()),
          'active_rows':int(selected.sum()),'excluded_rows':int(np.count_nonzero(rows&~active)),
          'invalid_rows':int((~v).sum()),'over_budget':int(np.count_nonzero(q>1)),
          'median_px':float(np.median(d)) if len(d) else None,'p95_px':float(np.quantile(d,.95)) if len(d) else None,
          'max_px':float(d.max()) if len(d) else None,'max_ratio_valid':float(q.max()) if len(q) else None})
    return result


def replay_native(run_dir,model_path,state_path,output_dir=None,*,semantic_manifest_path=None,packet_manifest_path=None):
    started=time.perf_counter();root,run=read_run(run_dir);obs=Observations.load(run['data_dir']);n=len(obs.source_uv)
    # Freeze the actual evaluator sources as well as its data. Version labels
    # alone cannot distinguish a corrected ledger from an older evaluator.
    package=Path(__file__).parent
    source_bytes={p.name:p.read_bytes() for p in sorted(package.glob('*.py'))}
    source_hashes={name:hashlib.sha256(value).hexdigest() for name,value in source_bytes.items()}
    source_bundle_id=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    direct_sources=['native_replay.py','native_camera.py','native_scene.py','native_query.py',
                    'native_query_certificate.py','native_query_global.py','semantic_order.py','semantic_resolution.py','semantic_order_overlay.py','source_generator_recovery.py','source_component_recovery.py','semantic_exclusion.py','data.py','pipeline.py','runtime.py']
    model_path=Path(model_path);state_path=Path(state_path)
    # Read each complete file exactly once, then deep own its immutable content.
    model_bytes=model_path.read_bytes();state_bytes=state_path.read_bytes();model=json.loads(model_bytes);state=json.loads(state_bytes)
    frozen_state_sha=hashlib.sha256(state_bytes).hexdigest();frozen_model_sha=hashlib.sha256(model_bytes).hexdigest()
    source_expected=run['input_hashes']['observations.npz']
    if model.get('source_identity')!=source_expected and model.get('source_hashes',{}).get('observations.npz')!=source_expected:
        raise ValueError('model source identity not bound to immutable observations')
    identity=state.get('identity',{})
    selected_packet=packet_manifest_path or state.get('packet_manifest_path')
    packet_path=Path(selected_packet) if selected_packet else root/'backend_input/manifest.json'
    if not packet_path.is_absolute():packet_path=state_path.parent/packet_path
    packet=json.loads(packet_path.read_text(encoding='utf-8'))
    selected_semantic=semantic_manifest_path or state.get('semantic_manifest_path')
    if selected_semantic is None and selected_packet:
        selected_semantic=packet.get('semantic_packet',{}).get('manifest')
    if selected_semantic is not None:
        selected_semantic=Path(selected_semantic)
        if not selected_semantic.is_absolute():selected_semantic=state_path.parent/selected_semantic
    rows=row_state(state,state_path,n)
    semantic,sem,semantic_path,resolution=semantic_arrays(root,n,selected_semantic,
        identity.get('statistics_id'),include_resolution=True)
    sem_hash=sha256_file(semantic_path)
    exclusion=resolution['source_exclusion'];active=exclusion['active_mask']
    active_n=int(active.sum())
    verify_exclusion_source(exclusion,obs,source_expected,packet['arrays']['engineering_budget_px']['sha256'])
    if exclusion['enabled']:
        for key in ['source_exclusion_sha256','active_source_mask_sha256','active_statistics_sha256']:
            if identity.get(key)!=exclusion[key]:
                raise ValueError('Snapshot source exclusion identity mismatch: '+key)
    if state.get('semantic_manifest_sha256',sem_hash)!=sem_hash:
        raise ValueError('Snapshot semantic path/hash identity mismatch')
    if resolution['enabled']:
        for key,expected_digest in [('resolved_material_sha256',semantic['resolved_material_index']['sha256']),
                ('resolved_variant_sha256',semantic['resolved_variant_index']['sha256']),
                ('confirmed_bindings_sha256',semantic['confirmed_bindings']['sha256'])]:
            if identity.get(key)!=expected_digest:
                raise ValueError('Snapshot resolved binding identity mismatch: '+key)
    query_path=Path(packet.get('query_packet',{}).get('manifest',root/'query_packet/manifest.json'))
    if not query_path.is_absolute():query_path=packet_path.parent/query_path
    query_hash=sha256_file(query_path)
    if semantic.get('immutable_source_arrays')!=obs.manifest['arrays']:raise ValueError('semantic five-array identity mismatch')
    expected={'source_id':source_expected,'model_id':model['model_id'],
        'model_sha256':frozen_model_sha,'statistics_id':sem_hash,
        'budget_id':packet['arrays']['engineering_budget_px']['sha256'],'query_id':query_hash}
    identity_failures=[key for key,value in expected.items() if key!='model_id' and identity.get(key)!=value]
    if identity.get('model_id') not in (model['model_id'],frozen_model_sha):identity_failures.append('model_id')
    if identity_failures and not state.get('initialization_only',False):raise ValueError('snapshot immutable identity mismatch: '+','.join(identity_failures))
    frozen_identity_complete=not identity_failures
    scene=NativeScene(model,state);camera=camera_from_event(model,state);scene_physics=scene.check_physics();optical_physics=camera.optical_domain_certificate()
    table_spec=packet['string_tables'];tables_path=packet_path.parent/table_spec['file']
    if sha256_file(tables_path)!=table_spec['sha256']:raise ValueError('Raw source string registry changed')
    tables=json.loads(tables_path.read_text(encoding='utf-8'));lookup={k:i for i,k in enumerate(tables['material_labels'])}
    if len(model['materials'])<len(lookup) or any(model['materials'][i]['label']!=label
            for i,label in enumerate(tables['material_labels'])):
        raise ValueError('Selected model changed the immutable raw material registry prefix')
    original_material=np.array([lookup[k] for k in obs.source_target_labels],dtype='<u4');branch=rows['branches'].astype(np.int64);station=rows['stations'].astype(float)
    original_variants=np.asarray(semantic['variant_ids'])[sem['variant_index']]
    material,variant_expected,resolved_bindings=resolve_materials(resolution,original_material,
        original_variants,obs,model,branch,source_expected)
    variant_legal=np.zeros(n,bool)
    for m,b in np.unique(np.c_[material,branch],axis=0):
        ii=np.flatnonzero((material==m)&(branch==b))
        try:variant_legal[ii]=scene.path(m,b).get('variant_id')==variant_expected[ii]
        except (ValueError,IndexError):pass
    for binding in resolved_bindings:
        if not binding['saved_branch_in_confirmed_support']:variant_legal[binding['source_row']]=False
    t=(obs.native_pts-256256)/12800.;world,finite,reason=scene.world_rows(material,branch,station)
    finite &= variant_legal;reason[~variant_legal]='source_variant_branch_mismatch'
    # Invalid source supports remain in the denominator and output ledger.
    safe_world=np.where(finite[:,None],world,np.array([0.,0.,1.]))
    result=camera.project_rs(safe_world,t);predicted=result['uv'].copy();valid=result['valid']&finite
    failure=np.where(~finite,reason,np.where(~result['bracketed'],'RS_no_legal_bracket',np.where(~result['certificate']['certified'],'RS_or_Brown_interval_uncertified',np.where(~result['valid'],'invalid_true_RS_root','')))).astype('U160')
    rs_failure=failure.copy()
    current_model=json.loads(model_bytes)
    for definition,value in zip(current_model['blocks'],scene.values):definition['values']=value.tolist()
    order=audit_source_runs(semantic_path,current_model,station,branch)
    explicit_prediction=camera.at(safe_world,t+camera.tau*(rows['root_witnesses']/1079-.5))
    saved_prediction=predicted.copy();replayed_station=station.copy();saved_query_eq=np.zeros(n);profile_saved_alpha_valid=np.zeros(n,bool)
    query_eq=np.zeros(n);candidate_query_eq=np.zeros(n);saved_profile_alpha=np.full(n,np.nan);profile_alpha=np.full(n,np.nan);query_station=station.copy();profile=sem['kind']==2
    normal=sem['normal'];profile_interval_certified=np.zeros(n,bool);profile_local_root_valid=np.zeros(n,bool);profile_roots_recovered=0;profile_root_box_certified=np.zeros(n,bool);profile_certificate=None
    profile_global_unique_certified=np.zeros(n,bool);profile_global_certificate=None
    if np.any(profile):
        ii=np.flatnonzero(profile);query_manifest,curves=load_query_packet(query_path,read_array,resolution['query_semantic_sha256'],ii)
        seed=rows.get('profile_alphas',rows.get('profile_alpha'))
        seed=seed[ii] if seed is not None and seed.shape==(n,) else None
        solved=recover_query_roots(scene,camera,material[ii],branch[ii],station[ii],t[ii],curves,seed)
        profile_roots_recovered=int(solved['valid'].sum())
        profile_certificate=certify_profile_roots(scene,camera,material[ii],branch[ii],solved['stations'],t[ii],curves,solved['alpha'],solved['row'],numerical_valid=solved['valid'])
        solved['relevant_root_interval_certified']=profile_certificate['certified']
        profile_root_box_certified[ii]=profile_certificate['certified']
        profile_global_certificate=certify_global_profiles(scene,camera,material[ii],branch[ii],
            solved['stations'],t[ii],curves,solved['alpha'],solved['row'],numerical_valid=solved['valid'])
        profile_global_unique_certified[ii]=profile_global_certificate['global_unique_certified']
        saved_alpha=rows.get('profile_alphas',rows.get('profile_alpha'))
        alpha_complete=saved_alpha is not None and saved_alpha.shape==(n,)
        if alpha_complete:
            saved_a=saved_alpha[ii];saved_profile_alpha[ii]=saved_a;alpha_legal=np.isfinite(saved_a)&(saved_a>=curves['bounds'][:,0])&(saved_a<=curves['bounds'][:,1])
            saved_q,_,saved_q_good=rational_query(curves,saved_a);alpha_legal &= saved_q_good
            saved_query_eq[ii]=np.linalg.norm(solved['uv']-saved_q,axis=1)
        else:
            alpha_legal=np.zeros(len(ii),bool);saved_query_eq[ii]=np.inf
        profile_saved_alpha_valid[ii]=alpha_legal
        consistency=np.linalg.norm(solved['uv']-saved_prediction[ii],axis=1)
        saved_row_consistency=abs(solved['row']-rows['root_witnesses'][ii])
        good=solved['valid']&profile_certificate['certified']&profile_global_unique_certified[ii]&alpha_legal&(saved_query_eq[ii]<=.01)&(consistency<=.01)&(saved_row_consistency<=.01)
        profile_local_root_valid[ii]=good;profile_interval_certified[ii]=profile_certificate['certified']&profile_global_unique_certified[ii]&good
        query_station[ii]=solved['stations'];profile_alpha[ii]=solved['alpha']
        candidate_query_eq[ii]=solved['query_equality_px']
        # Certified exact query elimination: all final native ledgers use the
        # camera forward at the same recovered station, never Q as prediction.
        selected=ii[good]
        predicted[selected]=solved['uv'][good];replayed_station[selected]=solved['stations'][good]
        world[selected]=scene.world_rows(material[selected],branch[selected],replayed_station[selected])[0]
        for key,value in result.items():
            if key=='certificate':
                for name,array in value.items():
                    if isinstance(array,np.ndarray) and array.shape[:1]==(n,):array[selected]=solved['certificate'][name][good]
            elif isinstance(value,np.ndarray) and value.shape[:1]==(n,):value[selected]=solved[key][good]
        valid[ii] &= good
        for j in ii[~good]:failure[j]=(failure[j]+';' if failure[j] else '')+'profile_query_root_or_saved_witness_failure'
        final_q,_,_=rational_query(curves,profile_alpha[ii]);query_eq[ii]=np.linalg.norm(predicted[ii]-final_q,axis=1)
    witness_error=abs(result['row']-rows['root_witnesses']);witness_passed=bool(np.all(witness_error[valid&active]<=.01))
    explicit_error=explicit_prediction['uv']-obs.source_uv
    error=predicted-obs.source_uv;norm=np.linalg.norm(error,axis=1);coverage_error=norm.copy()
    coverage_error[profile]=abs(np.einsum('ij,ij->i',error[profile],normal[profile]))
    budget_vector=read_array(packet_path.parent,packet['arrays']['engineering_budget_px'])
    if budget_vector.shape!=(n,) or not np.isfinite(budget_vector).all() or np.any(budget_vector<=0):raise ValueError('invalid frozen all-source pixel budgets')
    budget=float(budget_vector[0]) if np.all(budget_vector==budget_vector[0]) else None
    ratio=coverage_error/budget_vector;q=ratio*ratio
    output=Path(output_dir) if output_dir else root/'independent_replay'/frozen_state_sha[:16];output.mkdir(parents=True,exist_ok=True)
    if profile_certificate is not None:
        atomic_npz(output/'profile_root_certificates.npz',{k:v for k,v in profile_certificate.items() if isinstance(v,np.ndarray)})
        atomic_json(output/'profile_root_certificates.json',{**{k:v for k,v in profile_certificate.items() if not isinstance(v,np.ndarray)},'producer_sha256':sha256_file(Path(__file__).with_name('native_query_certificate.py'))})
    if profile_global_certificate is not None:
        global_proof_arrays={k:v for k,v in profile_global_certificate.items() if isinstance(v,np.ndarray)}
        global_proof_arrays['source_rows']=ii
        atomic_npz(output/'profile_global_certificates.npz',global_proof_arrays)
        tree_arrays,tree_schema=pack_proof_tree(profile_global_certificate['proof_tree'])
        atomic_npz(output/'profile_global_proof_tree.npz',tree_arrays)
        atomic_json(output/'profile_global_proof_tree_schema.json',tree_schema)
        global_inputs={'source_rows':ii,'material':material[ii],'branch':branch[ii],
            'station':solved['stations'],'frame_time':t[ii],'alpha':solved['alpha'],
            'row':solved['row'],'numerical_valid':solved['valid']}
        global_inputs.update({'query_'+k:v for k,v in curves.items()})
        atomic_npz(output/'profile_global_frozen_inputs.npz',global_inputs)
        global_constants={'camera':{'R0':camera.R0.tolist(),'n':camera.n.tolist(),
            'b':camera.b.tolist(),'P':camera.P.tolist(),'d':camera.d.tolist(),'a':camera.a,
            'controls':camera.controls.tolist(),'first_control_time':camera.first,
            'dt':camera.dt,'tau':camera.tau,'head':camera.head,'optics':camera.optics},
            'finite_material_line_endpoint_blocks':{str(scene.path(m,b)['geometry_block']):
                scene.values[scene.path(m,b)['geometry_block']].tolist() for m,b in zip(material[ii],branch[ii])},
            'scope':profile_global_certificate['fixed_camera_constants_scope']}
        atomic_json(output/'profile_global_frozen_constants.json',global_constants)
        global_metadata={k:v for k,v in profile_global_certificate.items()
            if not isinstance(v,np.ndarray) and k not in ['local_certificate','proof_tree']}
        global_metadata.update(source_rows=ii.tolist(),semantic_manifest_sha256=sem_hash,
            query_manifest_sha256=query_hash,state_sha256=frozen_state_sha,
            all_original_finite_profiles_unique=bool(profile_global_unique_certified[ii].all()),
            unresolved_source_rows=ii[~profile_global_unique_certified[ii]].tolist(),
            local_certificate_reference='profile_root_certificates.npz',
            files={name:{'file':name,'sha256':sha256_file(output/name)} for name in
                ['profile_global_certificates.npz','profile_global_proof_tree.npz',
                 'profile_global_proof_tree_schema.json','profile_global_frozen_inputs.npz',
                 'profile_global_frozen_constants.json','profile_root_certificates.npz']},
            other_acceptance_gates_unchanged='saved-state witnesses, original W/budget, whole-readout RS, static scene, source order and fresh true-query-operator KKT are separate required gates')
        atomic_json(output/'profile_global_certificates.json',global_metadata)
    base={'source_rows':np.arange(n,dtype=np.uint64),'native_pts':obs.native_pts,'source_point_ids':obs.source_point_ids,
          'active_source_mask':active,
          'source_target_labels':obs.source_target_labels,'predicted_native_uv':predicted,'world_L_units':world,'stations':replayed_station,'branches':branch,
          'original_material_indices':original_material,'resolved_material_indices':material,
          'original_variant_ids':original_variants,'resolved_variant_ids':variant_expected,
          'resolved_model_target_labels':np.asarray([model['materials'][int(i)]['label'] for i in material])}
    atomic_npz(output/'final_predictions.npz',base)
    atomic_npz(output/'final_source_residuals.npz',{'source_rows':base['source_rows'],'error_native_uv':error,'error_2d_px':norm,
      'coverage_error_px':coverage_error,'budget_native_px':budget_vector,'q':q,'budget_ratio':ratio,'valid':valid,'active_source_mask':active,'failure_reason':failure,
      'saved_station_RS_failure_reason':rs_failure,'RS_failure_reason':np.where(~finite,reason,np.where(~result['bracketed'],'RS_no_legal_bracket',np.where(~result['certificate']['certified'],'RS_or_Brown_interval_uncertified',np.where(~result['valid'],'invalid_true_RS_root','')))),'RS_valid':result['valid']&finite,
      'stations':replayed_station,'saved_station_witnesses':station,'saved_predicted_native_uv':saved_prediction,'saved_explicit_predicted_native_uv':explicit_prediction['uv'],'profile_saved_alpha_valid':profile_saved_alpha_valid,'saved_query_equality_px':saved_query_eq,'query_root_station':query_station,'branches':branch,'root_rows':result['row'],
      'saved_root_witnesses':rows['root_witnesses'],'true_RS_equation_px':result['root_equality_px'],
      'RS_witness_error_px':witness_error,'profile_local_root_valid':profile_local_root_valid,'profile_interval_certified':profile_interval_certified,'profile_root_box_certified':profile_root_box_certified,
      'profile_global_unique_certified':profile_global_unique_certified,
      'RS_contraction_bound':result['certificate']['contraction_bound'],'depth_lower':result['certificate']['depth_lower'],
      'Brown_no_fold_lower':result['certificate']['brown_no_fold_lower'],'query_equality_px':query_eq,'candidate_query_equality_px':candidate_query_eq,'profile_alpha':profile_alpha,'saved_profile_alpha_witnesses':saved_profile_alpha})
    atomic_json(output/'source_order.json',order)
    atomic_json(output/'by_frame.json',_group_summary(obs.frame_indices,coverage_error,ratio,valid,active))
    atomic_json(output/'by_material.json',_group_summary(obs.source_target_labels,coverage_error,ratio,valid,active))
    if resolution['enabled']:
        atomic_json(output/'confirmed_material_bindings.json',{'schema':'rinkfit.replayed_confirmed_material_bindings.v1',
            'semantic_manifest':str(semantic_path),'semantic_manifest_sha256':sem_hash,
            'parent_semantic_sha256':resolution['parent_semantic_sha256'],
            'overlay_sha256':resolution['overlay_sha256'],'bindings':resolved_bindings,
            'raw_source_five_arrays_unchanged':True})
        atomic_json(output/'by_resolved_material.json',_group_summary(base['resolved_model_target_labels'],coverage_error,ratio,valid,active))
    if exclusion['enabled']:
        atomic_json(output/'authorized_source_exclusions.json',{'schema':'rinkfit.replayed_authorized_source_exclusions.v1',
            'semantic_manifest':str(semantic_path),'semantic_manifest_sha256':sem_hash,
            'raw_observations':n,'active_observations':active_n,
            'source_exclusion_sha256':exclusion['source_exclusion_sha256'],
            'active_source_mask_sha256':exclusion['active_source_mask_sha256'],
            'active_statistics_sha256':exclusion['active_statistics_sha256'],
            'raw_source_arrays_record_and_archived_W_unchanged':True,'records':exclusion['records']})
    # Preserve exact grouped W, including its cross-point/corner rank. Never
    # subtract a fitted common mode from the physical pixel coverage ledger.
    statistics={'complete':False,'rank':0,'half_squared_norm':None,'source_groups':0}
    if 'weights' in sem and 'group_rows' in sem:
        groups_path=semantic_path.parent/semantic.get('groups_file',semantic['assets']['groups.jsonl']['file'])
        if semantic.get('groups_sha256') and sha256_file(groups_path)!=semantic['groups_sha256']:raise ValueError('statistical group provenance changed')
        count=np.zeros(n,dtype=np.int32);sse=0.;explicit_sse=0.;rank=0;group_count=0
        with groups_path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():continue
                group=json.loads(line);start=int(group['member_start']);m=int(group['member_count']);r=int(group['rank']);w=int(group['weight_start'])
                ii=sem['group_rows'][start:start+m].astype(np.int64)
                W=sem['weights'][w:w+r*2*m].reshape(r,2*m)
                replacement=exclusion['replacements'].get(group['group_id'])
                if replacement is not None:ii=replacement['source_rows'];W=replacement['W'];r=replacement['rank']
                if not np.all(active[ii]):raise ValueError('Statistical group contains an inactive row without an exact marginal replacement')
                count[ii]+=1
                res=W@error[ii].reshape(-1);sse+=float(res@res);saved_res=W@explicit_error[ii].reshape(-1);explicit_sse+=float(saved_res@saved_res);rank+=r;group_count+=1
        if 'ordinary_rows' in sem:
            ii=sem['ordinary_rows'].astype(np.int64);ii=ii[active[ii]];count[ii]+=1
            independent=error[ii]/sem['source_sigma'][ii,None]
            sse+=float(np.sum(independent*independent));saved_independent=explicit_error[ii]/sem['source_sigma'][ii,None];explicit_sse+=float(np.sum(saved_independent*saved_independent));rank+=2*len(ii)
        membership=bool(np.array_equal(count,active.astype(np.int32)))
        statistics={'complete':bool(membership&np.all(valid[active])),'rank':rank,'half_squared_norm':.5*sse if np.isfinite(sse) else None,'source_groups':group_count,
                    'half_squared_norm_saved_lifted_witnesses':.5*explicit_sse if np.isfinite(explicit_sse) else None,
                    'root_restoration_cost_change_diagnostic':.5*(sse-explicit_sse) if np.isfinite(sse+explicit_sse) else None,
                     'saved_lifted_arithmetic_complete':bool(np.all((finite&explicit_prediction['valid'])[active])),
                     'membership_exactly_once':membership,'active_observations':active_n,'raw_observations':n,
                     'exact_marginal_replacements':len(exclusion['replacements']),
                     'statistics_identity':sem_hash,'residual_is_unmodified_raw_native_error':True}
    profile_recovery_change=float(np.max(np.linalg.norm(predicted[profile]-saved_prediction[profile],axis=1),initial=0))
    numeric=state.get('outer',state.get('outer_resume_state',{}));certificate=state.get('fresh_numerical_certificate',{})
    cert_values=[certificate.get(k) for k in ['primal_violation','equality_violation','complementarity','scaled_stationarity']]
    fresh=bool(state.get('state_token') and certificate.get('state_token')==state['state_token'] and certificate.get('complete_current_derivatives') is True
        and certificate.get('scales_id')==identity.get('scales_id') and certificate.get('scales_id'))
    numerical_passed=bool(fresh and all(isinstance(v,(int,float)) and np.isfinite(v) for v in cert_values)
        and cert_values[0]<=1e-4 and cert_values[1]<=.01 and cert_values[2]<=1e-5 and cert_values[3]<=1e-5
        and certificate.get('phase')==2 and certificate.get('phase_two_completed') is True)
    lifted_numerical_passed=numerical_passed
    numerical_profile_operator_passed=bool(numerical_passed and certificate.get('profile_operator_scope')=='certified_true_query_root_v1')
    numerical_passed=numerical_passed and numerical_profile_operator_passed
    root_certified=bool(np.all(profile_interval_certified[profile]))
    active_valid=valid&active;all_active_valid=bool(np.all(valid[active]))
    good_norm=coverage_error[active_valid];counts={k:int(np.count_nonzero((failure==k)&active)) for k in np.unique(failure[active]) if k}
    result_report={'schema':VERSION,'state_sha256':frozen_state_sha,'model_sha256':frozen_model_sha,'semantic_manifest_sha256':sha256_file(semantic_path),
      'semantic_manifest_path':str(semantic_path),'packet_manifest_path':str(packet_path),
      'material_resolution':{'enabled':resolution['enabled'],'confirmed_source_rows':[b['source_row'] for b in resolved_bindings],
        'overlay_sha256':resolution.get('overlay_sha256'),'parent_semantic_sha256':resolution.get('parent_semantic_sha256'),
        'query_bound_semantic_sha256':resolution['query_semantic_sha256'],'raw_source_labels_unchanged':True},
      'camera_evaluator':CAMERA_VERSION,'scene_evaluator':SCENE_VERSION,'observations':n,'active_observations':active_n,'frames':len(np.unique(obs.frame_indices)),
      'source_exclusions':{'enabled':exclusion['enabled'],'excluded_source_rows':np.flatnonzero(~active).tolist(),
        'raw_rows_retained':n,'active_rows':active_n,'authority_is_explicit_current_user_instruction':exclusion['enabled'],
        'source_exclusion_sha256':exclusion.get('source_exclusion_sha256'),'active_source_mask_sha256':exclusion.get('active_source_mask_sha256'),'active_statistics_sha256':exclusion.get('active_statistics_sha256')},
      'material_labels':len(np.unique(obs.source_target_labels)),'source_identity_exact':frozen_identity_complete,'identity_failures':identity_failures,'all_original_obligations_retained':True,
      'valid_source_rows':int(active_valid.sum()),'invalid_source_rows':int(np.count_nonzero(active&~valid)),
      'raw_valid_source_rows_diagnostic':int(valid.sum()),'raw_invalid_source_rows_diagnostic':int((~valid).sum()),'support_and_physical_failures':counts,
      'overlapping_failure_ledgers':{'RS_invalid_rows':int(np.count_nonzero(active&~(result['valid']&finite))),
        'RS_no_legal_bracket':int(np.count_nonzero(active&finite&~result['bracketed'])),
        'RS_interval_uncertified':int(np.count_nonzero(active&finite&~result['certificate']['certified'])),
        'profile_query_or_saved_witness_invalid':int(np.count_nonzero(profile&~profile_local_root_valid)),
        'profile_global_uniqueness_unproved':int(np.count_nonzero(profile&~profile_global_unique_certified)),
        'source_variant_mismatch':int(np.count_nonzero(active&~variant_legal))},
      'over_budget_valid_rows':int(np.count_nonzero(active_valid&(q>1))),
      't_hat_all_sources':float(np.max(ratio)) if valid.all() and not exclusion['enabled'] else None,
      't_hat_active_sources':float(np.max(ratio[active])) if all_active_valid else None,
      't_hat_valid_subset_diagnostic':float(np.max(ratio[active_valid])) if active_valid.any() else None,
      'raw_native_px_quantiles_valid_subset':np.quantile(good_norm,[.5,.9,.95,.99,1]).tolist() if len(good_norm) else None,
      'statistics':statistics,'budget_native_px':budget,'genuine_profiles':int(profile.sum()),
      'max_profile_recovery_change_px':profile_recovery_change,'max_saved_query_witness_error_px':float(np.max(saved_query_eq[profile],initial=0)),
      'max_query_equality_px':float(np.max(abs(query_eq[profile]),initial=0)),
      'max_true_RS_equality_px_valid_subset':float(np.max(abs(result['root_equality_px'][active_valid]),initial=0)),
      'arclength_numerical_max_error_L_units':scene.arclength_numerical_max_error,
      'pixel_budget_passed':bool(all_active_valid&np.all(q[active]<=1+1e-4)),'physical_source_support_passed':bool(all_active_valid and scene_physics['passed'] and optical_physics['passed'] and order['passed'] and root_certified and witness_passed),
      'static_scene_physics':scene_physics,'full_optical_domain':optical_physics,'source_order':{k:v for k,v in order.items() if k!='records'},
      'profile_relevant_root_interval_certified':root_certified,'profile_numerical_local_roots_recovered':profile_roots_recovered,'profile_local_root_boxes_certified':int(profile_root_box_certified.sum()),'profile_roots_consistent_with_saved_state':int(profile_local_root_valid.sum()),
      'profile_global_finite_domain_unique_certified':bool(np.all(profile_global_unique_certified[profile])),
      'profile_global_unique_roots_certified':int(profile_global_unique_certified.sum()),
      'profile_root_certificate_scope':'certified local root with exclusion of all other roots on the original finite source station x original query-alpha domain; unsupported or unresolved global boxes fail closed',
      'RS_saved_witness_consistency_passed':witness_passed,'max_RS_saved_witness_error_px_valid_subset':float(np.max(witness_error[active_valid],initial=0)),
      'numerical_state':{k:v for k,v in numeric.items() if k not in ['lambda','nu']},
      'dual_multipliers_checkpoint_reference':str(state_path),
      'dual_multiplier_counts':{k:len(numeric.get(k,[])) for k in ['lambda','nu']},'fresh_numerical_certificate':certificate,'numerical_certificate_fresh':fresh,'lifted_state_numerical_passed':lifted_numerical_passed,'replayed_profile_operator_numerical_passed':numerical_profile_operator_passed,'numerical_passed':numerical_passed,'verified_metre_scale':False,
      'elapsed_seconds':time.perf_counter()-started,'output_dir':str(output),
      'whole_shot_fit_accepted':bool(frozen_identity_complete and all_active_valid and np.all(q[active]<=1+1e-4) and scene_physics['passed'] and optical_physics['passed'] and order['passed'] and root_certified and witness_passed and statistics['complete'] and numerical_passed),
      'independent_holdout_validation_completed':False,'original_fit_sources_are_not_holdout':True}
    for name in direct_sources:
        if sha256_file(package/name)!=source_hashes[name]:
            raise ValueError('Independent evaluator changed during replay: '+name)
    source_bundle=root/'native_evaluators'/(source_bundle_id+'.json')
    if not source_bundle.exists():
        atomic_json(source_bundle,{'schema':'rinkfit.native_evaluator_source_bundle.v1','source_sha256':source_hashes,
            'source_utf8':{name:value.decode('utf-8') for name,value in source_bytes.items()}})
    bundle=json.loads(source_bundle.read_text(encoding='utf-8'))
    if bundle.get('source_sha256')!=source_hashes or any(hashlib.sha256(bundle['source_utf8'][name].encode()).hexdigest()!=digest for name,digest in source_hashes.items()):
        raise ValueError('Frozen independent evaluator source bundle changed')
    import scipy
    result_report['evaluator_provenance']={'source_bundle':str(source_bundle),'source_bundle_sha256':sha256_file(source_bundle),
        'direct_module_sha256':{name:source_hashes[name] for name in direct_sources},
        'python':sys.version,'python_executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,
        'direct_sources_unchanged_through_replay':True}
    atomic_json(output/'verification.json',result_report)
    return result_report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--model',required=True);p.add_argument('--state',required=True);p.add_argument('--output-dir')
    p.add_argument('--semantic-manifest');p.add_argument('--packet-manifest')
    a=p.parse_args();print(json.dumps(replay_native(a.run_dir,a.model,a.state,a.output_dir,
        semantic_manifest_path=a.semantic_manifest,packet_manifest_path=a.packet_manifest),indent=2,allow_nan=False))

if __name__=='__main__':main()
