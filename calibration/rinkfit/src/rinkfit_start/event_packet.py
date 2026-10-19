"""Portable full-source ABI for the event-specific Ceres ProblemBuilder.

No pickle, Python-owner imports, video dependencies, source filtering, or
assumed scalar replacement for the original group whitening operator.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import numpy as np

from .data import Observations, sha256_file
from .runtime import atomic_json, Journal, run_lock
from .pipeline import read_run


def _binary(path, array, dtype):
    array=np.ascontiguousarray(array,dtype=np.dtype(dtype))
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as out:
            out.write(array.tobytes());out.flush();os.fsync(out.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    return {'file':path.name,'dtype':array.dtype.str,'shape':list(array.shape),
            'byte_order':'little','layout':'C','sha256':sha256_file(path)}


def export_packet(run_dir):
    root,run=read_run(run_dir); data=Path(run['data_dir']); obs=Observations.load(data)
    packet=root/'backend_input';packet.mkdir(exist_ok=True)
    labels,label_ids=np.unique(obs.source_target_labels,return_inverse=True)
    point_ids,point_indices=np.unique(obs.source_point_ids,return_inverse=True)
    n=len(obs.source_uv)
    with run_lock(packet):
        entries={
          'native_uv':_binary(packet/'uv.f64',obs.source_uv,'<f8'),
          'source_rows':_binary(packet/'row.u64',np.arange(n),'<u8'),
          'frame_index':_binary(packet/'frame.i32',obs.frame_indices,'<i4'),
          'native_pts':_binary(packet/'pts.i64',obs.native_pts,'<i8'),
          'material_index':_binary(packet/'material.u32',label_ids,'<u4'),
          'point_id_index':_binary(packet/'point_id.u32',point_indices,'<u4'),
          'engineering_budget_px':_binary(packet/'budget.f64',np.full(n,run['config']['engineering_budget_native_px']),'<f8')}
        atomic_json(packet/'string_tables.json',{'material_labels':labels.tolist(),
                    'source_point_ids':point_ids.tolist(),'point_id_alone_is_not_primary_key':True})
        assets={}
        for name in ['seeds.json','world_seeds.npz','camera_seeds.npz','bootstrap_state.json','source_sigma.npz','media.json','input_manifest.json']:
            shutil.copyfile(data/name,packet/name)
            assets[name]={'sha256':sha256_file(packet/name),'role':'conditional seed or original source metadata'}
        manifest={'schema':'rinkfit.event_packet.v1','shot_id':obs.manifest['shot_id'],
            'observations':n,'frame_count':1248,'frame_min':1,'frame_max':1248,
            'native_image_wh':obs.manifest['image_wh'],'pixel_centers':'integer',
            'native_time_base':obs.manifest['time_base'],'pts_origin':obs.manifest['pts_origin'],
            'time_origin':'model convention: frame 1 mid-readout at PTS 256256; not a measured sensor timestamp',
            'arrays':entries,'string_tables':{'file':'string_tables.json','sha256':sha256_file(packet/'string_tables.json')},
            'assets':assets,
            'primary_key':['shot_id','native_pts','source_point_id'],
            'immutable_source_hashes':run['input_hashes'],
            'seed_role':'conditional initial state only; shared scene and fixed-head parameters must be fitted',
            'world_gauge':'rink full length L=1; no verified metre scale supplied',
            'statistics':obs.manifest['statistics'],
            'semantics':obs.manifest['semantics'],
            'registration_directory':str(root/'registration'),'graph_directory':str(root/'graph'),
            'required_final_source_rows':n,'H_graph_flow_residuals_in_final_objective':False,
            'event_problem_builder_implemented_in_this_package':False,
            'codex_integration_contract':'docs/event_integration.md'}
        atomic_json(packet/'manifest.json',manifest)
        Journal(root).emit('backend_packet_exported',observations=n,frames=1248,
                          manifest=str(packet/'manifest.json'),physical_fit_completed=False)
    return {'manifest':str(packet/'manifest.json'),'observations':n,'frames':1248}


def check_packet(manifest_path,verify_components=True):
    manifest_path=Path(manifest_path); root=manifest_path.parent
    m=json.loads(manifest_path.read_text(encoding='utf-8'))
    if m['schema']!='rinkfit.event_packet.v1':raise ValueError('Unknown event ABI')
    for name,spec in m['arrays'].items():
        path=root/spec['file']
        if sha256_file(path)!=spec['sha256']:raise ValueError(f'Packet checksum changed: {name}')
        if path.stat().st_size!=int(np.prod(spec['shape']))*np.dtype(spec['dtype']).itemsize:
            raise ValueError(f'Packet byte count changed: {name}')
    tables=m['string_tables']
    if sha256_file(root/tables['file'])!=tables['sha256']:raise ValueError('String table changed')
    for name,spec in m['assets'].items():
        if sha256_file(root/name)!=spec['sha256']:raise ValueError(f'Packet asset changed: {name}')
    for key in ['semantic_packet','query_packet','event_model','station_initialization']:
        if verify_components and key in m:
            desc=m[key];component=Path(desc['manifest'])
            if sha256_file(component)!=desc['sha256']:raise ValueError('Frozen physical component changed: '+key)
            if key=='semantic_packet':
                from .semantic_resolution import load_semantics
                _,_,_,resolution=load_semantics(component,m['observations'],expected_sha256=desc['sha256'])
                if resolution['enabled']:
                    spec=m['arrays']['material_index']
                    original=np.fromfile(root/spec['file'],dtype=np.dtype(spec['dtype'])).reshape(spec['shape'])
                    if not np.array_equal(original,resolution['original_material_index']):
                        raise ValueError('Resolved semantic packet changed the original material registry')
            if key=='station_initialization':
                initial=json.loads(component.read_text(encoding='utf-8'))
                for name in ['station','path']:
                    if name+'_file' in initial and sha256_file(Path(initial[name+'_file']))!=initial[name+'_sha256']:
                        raise ValueError('Frozen initializer changed: '+name)
                for name,spec in initial.get('additional_arrays',{}).items():
                    path=Path(spec['file'])
                    if sha256_file(path)!=spec['sha256'] or path.stat().st_size!=int(np.prod(spec['shape']))*np.dtype(spec['dtype']).itemsize:
                        raise ValueError('Frozen initializer changed: '+name)
                for name,spec in initial.get('additional_assets',{}).items():
                    if sha256_file(Path(spec['file']))!=spec['sha256']:
                        raise ValueError('Frozen initializer asset changed: '+name)
    return {'schema':m['schema'],'observations':m['observations'],'byte_arrays_verified':len(m['arrays']),
            'physical_geometry_verified':False}


def start_physical_backend(run_dir, executable, resume=None, packet_manifest=None,
                           physical_run_dir=None, runtime_path=None,
                           extend_resource_from=None):
    root,run=read_run(run_dir); executable=Path(executable).expanduser().resolve()
    if not executable.is_file():raise ValueError('Build the event-specific rinkfit_event executable first; see CODEX_START.md')
    env=os.environ.copy()
    for name in ['build_recipe.json','direct_build_recipe.json']:
        recipe_path=executable.parent/name
        if recipe_path.is_file():
            recipe=json.loads(recipe_path.read_text(encoding='utf-8'))
            if recipe.get('executable_sha256')!=sha256_file(executable):
                raise ValueError('Backend executable does not match its frozen build recipe')
            if os.name=='nt':env={k.upper():v for k,v in env.items()}
            env['PATH']=os.pathsep.join(recipe.get('runtime_bin_paths',[]))+os.pathsep+env.get('PATH','')
            for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
            break
    probe=subprocess.run([str(executable),'--capabilities-json'],capture_output=True,text=True,check=True,timeout=30,env=env)
    capabilities=json.loads(probe.stdout)
    supported_schemas={'rinkfit.event_backend.v1','rinkfit.event_backend.v1.1.domain_closure.v2','rinkfit.event_backend.v1.1.active_source_exclusion.v1'}
    if capabilities.get('schema') not in supported_schemas:
        raise ValueError('Unsupported physical event backend schema')
    if capabilities['schema'] in {'rinkfit.event_backend.v1.1.domain_closure.v2','rinkfit.event_backend.v1.1.active_source_exclusion.v1'}:
        if capabilities.get('per_source_domain_constraints')!=3 or capabilities.get('AL_extension_is_physical_validity') is not False:
            raise ValueError('Domain-closure backend lacks the complete original per-source domain contract')
    if capabilities['schema']=='rinkfit.event_backend.v1.1.active_source_exclusion.v1':
        for flag in ['active_source_accounting','excluded_raw_rows_retained_in_state','supports_explicit_source_exclusion']:
            if capabilities.get(flag) is not True:
                raise ValueError('Backend lacks the authorized exclusion ledger contract: '+flag)
    required={'backend_kind':'physical_event',
              'shot_id':'ankara_2026/okada_mei_sp/shot1','solver':'Ceres',
              'whole_shot_rows':182017,'independent_native_replay':True,
              'fixed_shared_camera_head':True,'synthetic_demo':False}
    for key,value in required.items():
        if capabilities.get(key)!=value:raise ValueError(f'Backend capability mismatch: {key}; synthetic demos are not event fitters')
    manifest=Path(packet_manifest).resolve() if packet_manifest else root/'backend_input/manifest.json'
    if not manifest.exists():
        if packet_manifest:raise ValueError('Selected immutable event packet does not exist')
        export_packet(root)
    check_packet(manifest)
    selected_packet=json.loads(manifest.read_text(encoding='utf-8'))
    semantic_spec=selected_packet.get('semantic_packet')
    if semantic_spec:
        semantic_file=Path(semantic_spec['manifest'])
        if not semantic_file.is_absolute():semantic_file=manifest.parent/semantic_file
        selected_semantic=json.loads(semantic_file.read_text(encoding='utf-8'))
        for field,flag in [('source_order_overlay','source_order_overlay'),
                           ('source_generator_recovery','source_generator_recovery'),
                           ('source_component_recovery','source_component_recovery')]:
            if field in selected_semantic and capabilities.get(flag) is not True:
                raise ValueError('Backend lacks selected semantic interpretation capability: '+flag)
    output=Path(physical_run_dir).resolve() if physical_run_dir else root
    output.mkdir(parents=True,exist_ok=True)
    cmd=[str(executable),'--packet',str(manifest),'--run-dir',str(output)]
    if resume:cmd+=['--resume',str(Path(resume).resolve())]
    if runtime_path:cmd+=['--runtime',str(Path(runtime_path).resolve())]
    if extend_resource_from:
        if not resume:raise ValueError('Resource extension requires a complete resume checkpoint')
        cmd+=['--extend-resource-from',str(Path(extend_resource_from).resolve())]
    with run_lock(root):
        Journal(root).emit('physical_backend_starting',capabilities=capabilities,
                          packet=str(manifest),packet_sha256=sha256_file(manifest),physical_run_dir=str(output),
                          capability_declaration_is_not_a_verification_certificate=True)
        result=subprocess.run(cmd,check=False,env=env)
        Journal(root).emit('physical_backend_process_finished',returncode=result.returncode,
                          accepted_fit_requires_independent_S6=True)
    return {'returncode':result.returncode,'physical_fit_accepted_by_launcher':False}


def verify_native_export(run_dir, prediction_path):
    """Independent source alignment and native pixel arithmetic, not geometry proof.

Event S6 must separately reproject the saved physical state with a distinct
verification evaluator, certify support/RS/depth, and check the KKT state.
"""
    root,run=read_run(run_dir); obs=Observations.load(run['data_dir']);n=len(obs.source_uv)
    with np.load(prediction_path,allow_pickle=False) as z:
        required=['source_rows','native_pts','source_point_ids','source_target_labels','predicted_native_uv']
        if any(k not in z for k in required):raise ValueError('Native result lacks original row identity fields')
        if not np.array_equal(z['source_rows'],np.arange(n)):raise ValueError('Result dropped, reordered or duplicated source rows')
        for key in ['native_pts','source_point_ids','source_target_labels']:
            if not np.array_equal(z[key],getattr(obs,key)):raise ValueError(f'Native result changed reviewed source identity: {key}')
        predicted=z['predicted_native_uv'].copy()
    if predicted.shape!=(n,2) or not np.isfinite(predicted).all():raise ValueError('Every original row needs one finite prediction')
    errors=predicted-obs.source_uv; norm=np.linalg.norm(errors,axis=1)
    budget=run['config']['engineering_budget_native_px']; ratio=norm/budget; q=ratio**2
    result={'schema':'rinkfit.native_arithmetic_check.v1','observations':n,'frames':1248,
        'source_identity_exact':True,'budget_native_px':budget,'max_error_native_px':float(norm.max()),
        'rms_native_px':float(np.sqrt(np.mean(norm**2))),'quantiles_native_px':np.quantile(norm,[.5,.9,.95,.99,1]).tolist(),
        'violating_rows':int(np.count_nonzero(q>1)),'max_budget_ratio':float(ratio.max()),
        't_hat':float(ratio.max()),'max_normalized_squared_error_q':float(q.max()),
        'raw_pixel_budget_passed':bool(np.all(q<=1)),
        'independent_physical_reprojection_performed_here':False,'S6_physical_and_numerical_certificates_checked_here':False,
        'whole_shot_fit_accepted':False}
    atomic_json(root/'native_arithmetic_check.json',result)
    atomic_npz_result=root/'native_arithmetic_residuals.npz'
    from .runtime import atomic_npz
    atomic_npz(atomic_npz_result,{'source_rows':np.arange(n),'error_native_uv':errors,'budget_ratio':ratio,'q':q})
    return result



def finalize_event_packet(run_dir,model_path=None,initial_state_dir=None,manifest_path=None,semantic_path=None):
    """Attach immutable full event inputs, never rewriting original source arrays."""
    root,run=read_run(run_dir);base_manifest=root/'backend_input/manifest.json'
    check_packet(base_manifest,verify_components=False);packet=base_manifest.parent
    manifest=Path(manifest_path).resolve() if manifest_path else base_manifest
    if manifest.parent!=packet.resolve():raise ValueError('Candidate manifest must share the immutable ABI array directory')
    selected_model=Path(model_path).resolve() if model_path else packet/'model.json'
    selected_initial=Path(initial_state_dir).resolve() if initial_state_dir else packet
    selected_manifest=packet/(manifest.stem+'.init.json') if manifest_path else packet/'selected_initialization.json'
    def save_selected(path,value):
        if manifest_path and path.exists():
            if json.loads(path.read_text(encoding='utf-8'))!=value:
                raise ValueError('Immutable candidate already frozen with different inputs: '+str(path))
            return
        atomic_json(path,value)
    for name in ['station.f64','path.u32']:
        if not (selected_initial/name).is_file():raise ValueError('Incomplete whole-source initializer: '+name)
    additional={name:{'file':str(selected_initial/name),'dtype':'<f8','shape':[182017],
        'sha256':sha256_file(selected_initial/name)} for name in ['initial_row.f64','initial_alpha.f64'] if (selected_initial/name).is_file()}
    additional_assets={name:{'file':str(selected_initial/name),'sha256':sha256_file(selected_initial/name)}
        for name in ['run_constraint_seed.json','initializer_manifest.json'] if (selected_initial/name).is_file()}
    save_selected(selected_manifest,{'schema':'rinkfit.selected_allsource_initializer.v1','observations':182017,
        'station_file':str(selected_initial/'station.f64'),'path_file':str(selected_initial/'path.u32'),
        'station_sha256':sha256_file(selected_initial/'station.f64'),'path_sha256':sha256_file(selected_initial/'path.u32'),
        'additional_arrays':additional,
        'additional_assets':additional_assets,
        'role':'initial physical capture only; no source observation or final factor changed'})
    selected_semantic=Path(semantic_path).resolve() if semantic_path else root/'semantic_packet/manifest.json'
    needed={'semantic_packet':selected_semantic,'query_packet':root/'query_packet/manifest.json',
            'event_model':selected_model,'station_initialization':selected_manifest}
    for key,path in needed.items():
        if not path.is_file():raise ValueError('Full physical packet component is not ready: '+key)
    model=json.loads(needed['event_model'].read_text(encoding='utf-8'))
    source=model.get('source_hashes',{}).get('observations.npz',model.get('source_identity'))
    if source!=run['input_hashes']['observations.npz']:raise ValueError('Physical scene has a different immutable source identity')
    from .semantic_resolution import load_semantics
    semantic,_,_,resolution=load_semantics(needed['semantic_packet'],182017)
    if resolution['enabled'] and resolution['source_identity']!=run['input_hashes']['observations.npz']:
        raise ValueError('Resolved material confirmation has a different immutable source')
    for name,spec in semantic['arrays'].items():
        path=needed['semantic_packet'].parent/spec['file']
        if sha256_file(path)!=spec['sha256']:raise ValueError('Semantic packet array changed: '+name)
    stations=json.loads(needed['station_initialization'].read_text(encoding='utf-8'))
    for name,key in [('station.f64','station_sha256'),('path.u32','path_sha256')]:
        if sha256_file(selected_initial/name)!=stations[key]:raise ValueError('Station/branch initialization changed')
    with run_lock(packet):
        m=json.loads(base_manifest.read_text(encoding='utf-8'))
        for key,path in needed.items():m[key]={'manifest':str(path),'sha256':sha256_file(path)}
        m['initial_state_directory']=str(selected_initial)
        m['initialization_only_paths']={'curve_graph':str(root/'curve_graph'),'atlas_upgrade':str(root/'atlas_upgrade'),
            'all_derived_H_P_flow_factors_absent_from_final_objective':True}
        m['event_model_exported']=True;m['full_semantic_packet_attached']=True
        m['event_problem_builder_implemented_in_this_package']=True
        m['statistics'].update(semantic['statistics'])
        m['statistics'].pop('scalar_sigma_must_not_replace_full_grouped_W',None)
        order_overlay=resolution.get('source_order_overlay')
        hard_runs=(order_overlay['active_hard_run_count'] if order_overlay
                   else semantic['source_runs']['proven_runs'])
        m['semantics'].update(verified_run_order_included=bool(hard_runs),
                              active_hard_run_count=int(hard_runs),
                              retained_source_navigation_run_count=(int(semantic['source_runs']['proven_runs']) if order_overlay else 0),
                              true_profile_metadata_included=bool(semantic['profiles']),
                              source_generator_recovery_included=bool(resolution.get('source_generator_recovery')),
                              source_component_recovery_included=bool(resolution.get('source_component_recovery')))
        m['physical_fit_completed']=False
        save_selected(manifest,m)
    return {'manifest':str(manifest),'sha256':sha256_file(manifest),'observations':182017,'frames':1248,
            'materials':171,'event_model':str(needed['event_model']),'semantic_packet':str(needed['semantic_packet']),
            'physical_fit_accepted':False}
