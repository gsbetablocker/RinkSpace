"""Export the current complete fitting data; historical evidence is provenance only.

This exporter is an offline data-management operation. The reader consumes the
current package, never an earlier packet, qualifier, source implementation, or
cold-validation report.
"""
from pathlib import Path
import argparse, copy, hashlib, json, shutil, time
import numpy as np
from .effective_input_audit_v1 import strict_json, require

SCHEMA='rinkfit.current_joint_input.v1'

def file_ref(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return dict(file=str(p.absolute()),sha256=h.hexdigest(),byte_count=p.stat().st_size)

def current_query_values(query):
    def plain(value):
        if isinstance(value,dict):
            if 'file'in value and 'sha256'in value:
                return 'offline source provenance: '+str(value['file'])+' sha256:'+str(value['sha256'])
            return {k:plain(v) for k,v in value.items()}
        if isinstance(value,list):return [plain(v) for v in value]
        return value
    result={}
    for k,v in query.items():
        if isinstance(v,dict) and 'file'in v and 'sha256'in v:
            result[k+'_provenance']=plain(v)
        else:result[k]=plain(v)
    return result


def export(resolved_path,selection_path,model_path,motion_path,out_dir,payload_map=None,logical_source_id=None,config_path=None):
    started=time.perf_counter();out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    require(not(out/'manifest.json').exists(),'Refuse to overwrite current input version')
    assets=out/'a';assets.mkdir(exist_ok=True);aliases={};copied={};ledger=[]
    if payload_map:
        for x in strict_json(Path(payload_map).read_bytes())['runtime_payload']:
            require(x['original']['sha256']==x['actual']['sha256'],'Source payload alias changed bytes')
            aliases[(str(Path(x['original']['file']).absolute()).casefold(),x['original']['sha256'])]=Path(x['actual']['file'])
    def save(value,name):
        p=out/name;p.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n','utf8');r=file_ref(p);r['file']=name;return r
    def asset(r):
        expected=r['sha256'];p=Path(r['file']);p=aliases.get((str(p.absolute()).casefold(),expected),p)
        if expected not in copied:
            dest=assets/(expected+'.bin');h=hashlib.sha256()
            with p.open('rb') as src,dest.open('wb') as dst:
                for b in iter(lambda:src.read(8<<20),b''):h.update(b);dst.write(b)
            require(h.hexdigest()==expected,'Current source asset bytes differ: '+str(p))
            copied[expected]=dict(file='a/'+expected+'.bin',sha256=expected,byte_count=dest.stat().st_size)
            ledger.append(dict(name=copied[expected]['file'],bytes=dest.stat().st_size))
        result=dict(r);result['file']=copied[expected]['file'];return result
    def load(r):
        p=Path(r['file']);raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==r['sha256'],'Current source JSON differs: '+str(p));return strict_json(raw)
    def rewrite(value):
        if isinstance(value,dict):
            if 'file'in value and 'sha256'in value:return asset(value)
            return {k:rewrite(v) for k,v in value.items()}
        if isinstance(value,list):return [rewrite(v) for v in value]
        return value
    q=strict_json(Path(resolved_path).read_bytes());sel=strict_json(Path(selection_path).read_bytes());model=strict_json(Path(model_path).read_bytes())
    n=q['row_count'];require(n==sel['source_row_count'],'Current raw denominator differs');logical_source_id=logical_source_id or sel['source_packet']['sha256']
    require(logical_source_id==sel['source_packet']['sha256'],'Current selection logical source differs')
    # These are the current values already consumed by the complete native domain.
    # Fold the final source permission / withdrawal values out of historical
    # containers once, then leave only plain provenance strings in the package.
    current_paths={};withheld=set()
    for r in q.get('resolved_native_print_identity_revision_chain',[]):
        rev=load(r);review=load(rev['review'])
        for x in review['rows']:current_paths[int(x['raw_row'])]=[int(x['new_material_index']),int(x.get('new_path',x.get('old_path',0)))]
    for r in q.get('resolved_native_source_measurement_revision_chain',[]):
        for x in load(r)['rows']:
            mi=int(x['new_material_index'])
            if model['materials'][mi]['label'].startswith('Ad'):current_paths[int(x['raw_row'])]=[mi,int(x['new_path'])]
    for r in q.get('resolved_native_identity_revision_chain',[]):withheld.update(map(int,load(r)['withheld_rows']))
    keep={'schema','independent_statistic_kind','row_count','shot_ids','arrays','source_dataset_ids','source_row_ids','source_family_ids',
          'engineering_scope','groups','queries','independent_rows','stroke_pairs','allowed_paths','source_station_domains','active_rows','admitted_rows','source_clock','statistical_rank'}
    packet={k:q[k] for k in keep if k in q}
    packet['arrays']={k:asset(v) for k,v in q['arrays'].items()}
    eg=q['engineering_pixel_dependency_groups'];packet['engineering_pixel_dependency_groups']={k:eg[k] for k in ('schema','policy_id','groups','normalization_order','source_dependency_components_are_not_independent_noise_samples') if k in eg}
    packet['current_input_data_v1']=dict(schema='rinkfit.current_packet_values.v1',version=1,logical_source_id=logical_source_id)
    packet['current_print_path_overrides']={str(k):v for k,v in sorted(current_paths.items())}
    packet['current_withheld_rows']=sorted(withheld)
    binding=dict(schema='rinkfit.current_material_binding.v1',model_material_prefix_count=len(model['materials']),materials=[])
    for i,m in enumerate(model['materials']):binding['materials'].append(dict(index=i,label=m['label'],paths=[dict(index=j,id=p['id'],physical_point=p.get('physical_point',False)) for j,p in enumerate(m['paths'])]))
    packet['material_binding']=binding
    regkeys={'schema','row_start','count','source_id_offsets','source_id_utf8','source_family_index','source_store_ordinal','source_family_ids','source_dataset_id','allowed_paths_by_material','allowed_path_mask'}
    def registry(r):return rewrite({k:v for k,v in r.items() if k in regkeys})
    packet['dense_source_binding']=registry(q['dense_source_binding'])
    packet['native_ad_source_bindings']=[registry(x) for x in q.get('native_ad_source_bindings',[q['native_ad_source_binding']])]
    packet['native_ad_source_binding']=packet['native_ad_source_bindings'][-1]
    for g in packet['groups']:
        if g['statistic_kind']=='constraint_only_unknown_noise':
            g.pop('unknown_noise_source_refs',None);g['current_unknown_noise_source_rows']=True
        if isinstance(g['W'],dict):g['W']=asset(g['W'])
    if 'source_station_domains'in packet:
        packet['source_station_domains'].pop('source_refs',None)
        packet['source_station_domains']['current_source_values']=True
        rechart={}
        for b in model.get('legacy_line_coordinate_recharts_v1',{}).get('blocks',[]):
            a,z=b['old_station_interval']
            for m in b['materials']:rechart[(m['material_index'],m['path_index'])]=(a,z)
        material=column_dummy=None
        raw_material=np.memmap(q['arrays']['material_index']['file'],dtype='<u4',mode='r',shape=(n,))
        for d in packet['source_station_domains']['rows']:
            chart=rechart.get((int(raw_material[d['row_index']]),d['path_index']))
            if chart:
                a,z=chart;d['allowed_station_intervals']=[[(lo-a)/(z-a),(hi-a)/(z-a)] for lo,hi in d['allowed_station_intervals']]

    # Query identities/channels/windows are current source definitions too.
    # Historical proof addresses are plain provenance, never runtime inputs.
    packet['queries']=[current_query_values(x) for x in packet['queries']]
    packet_ref=save(packet,'packet.json')
    prefix=len(packet['source_row_ids']);id_rows=prefix+sum(int(x['count']) for x in [packet['dense_source_binding'],*packet['native_ad_source_bindings']])
    require(id_rows==n,'Current full source ID registry is incomplete')
    def column(r):return np.memmap(r['file'],dtype=np.dtype(r['dtype']),mode='r',offset=r.get('offset_bytes',0),shape=tuple(r['shape']))
    physical=(np.asarray(column(q['arrays']['admission']))==1)&(np.asarray(column(q['arrays']['fit_role']))==1)
    training=column(sel['training_eligibility']['mask']);fit=column(sel['fit_rows']);inspection=column(sel['inspection_rows'])
    require(len(physical)==n and training.shape==(n,) and np.all((training==0)|(training==1)) and np.all(training<=physical),'Current qualification mask domain differs')
    chosen=np.zeros(n,dtype=np.uint8);chosen[fit]=1;require(len(fit)==int(chosen.sum()) and np.all(training[fit]) and np.all(np.diff(fit)>0),'Current fit is not unique sorted training subset')
    expected=np.flatnonzero(physical&~chosen.astype(bool));require(np.array_equal(inspection,expected),'Current full inspection complement differs')
    roles=load(sel['eligibility_provenance']['frame_roles']);roles={k:roles[k] for k in ('schema','source_video_sha256','frame_roles')}
    roles_ref=save(roles,'frame_roles.json');oldv=load(sel['eligibility_provenance']['source_identity_verdicts'])
    vk={'raw_row','native_pts','shot_index','material_index','path','uv','source_row_id','decision','uses_fitted_model_projection','reason'}
    verdicts=dict(schema='rinkfit.current_source_identity_values.v1',logical_source_id=logical_source_id,rows=[{k:v for k,v in x.items() if k in vk} for x in oldv['rows']])
    qualification=dict(schema='rinkfit.current_source_domain.v1',raw_rows=n,physical_count=int(physical.sum()),training_count=int(np.count_nonzero(training)),fit_count=len(fit),inspection_count=len(inspection),
        physical_mask=asset(q['arrays']['admission']),training_mask=asset(sel['training_eligibility']['mask']),fit_rows=asset(sel['fit_rows']),inspection_rows=asset(sel['inspection_rows']),frame_roles=roles_ref,source_identity_verdicts=save(verdicts,'source_identity.json'),
        no_fit_residual_selection=True,old_domain_backfilled=False,unknown_and_revoked_rows_preserved=True)
    # Current motion table holds both original UVs, PTS, pair/source IDs, and the
    # actual latent initialization; no earlier pair file is opened by the reader.
    motion_old=strict_json(Path(motion_path).read_bytes());mt=dict(motion_old['table'],dtype='<f8',shape=motion_old['table_shape'],layout='C',byte_order='little')
    motion=dict(schema='rinkfit.current_surface_motion.v1',logical_motion_id=file_ref(motion_path)['sha256'],surface='board_inner',latent_coordinates='board_arclength_and_height_fraction',selected_matches=motion_old['selected_matches'],table=asset(mt),source_clock=model['source_clock'],frame_roles=roles_ref,noise_covariance_calibrated=False)
    budget=strict_json((Path(selection_path).parent/'budget_recalculation.json').read_bytes())
    # Canonical installation membership is current numerical input, not an audit
    # report. Original entity labels and all distinct ad offsets remain separate.
    cfgpath=Path(config_path) if config_path else Path(resolved_path).parents[1]/'audit_r10g_spec'/'config.json'
    cfg=strict_json(cfgpath.read_bytes());entities=load(cfg['engineering_support_budget_v2']['canonical_material_entities'])
    common={k:v for k,v in cfg['engineering_support_budget_v2'].items() if k!='canonical_material_entities'}
    common['canonical_material_entities']=dict(schema='rinkfit.current_canonical_material_entities.v1',groups=entities['groups'])
    domain_old=load(cfg['capture_certified_numeric_domain']);numeric_domain=dict(schema='rinkfit.current_capture_numeric_domain.v1',scene=model['blocks'][model['scene_block']]['values'],rigs={k:dict(log_f_range=v['log_f_range'],ray_radius=model['rigs'][k]['camera']['radius_max']) for k,v in domain_old['rigs'].items()},dense_six=True)
    numeric_domain_ref=save(numeric_domain,'numeric_domain.json')
    # Keep numerical model definitions and the complete accepted block layout.
    # Historical executable migrations and audit references are not current data.
    model_keys={'schema','blocks','rigs','shots','materials','scene_block','source_clock','coordinate_gauge','physical_parameter_ownership',
      'linear_control_manifolds','continuous_board_station_v1','s01_original_sparse_storage_v1','s01_affine_recovery_corrections',
      'measurement_widths','stroke_pair_model_bindings','paired_advertisement_templates','confirmed_advertisement_shapes',
      'printed_material_path_refinements','shared_board_material_interfaces','path_domain_semantics','confirmed_tripod_outside_rink_v1','physical_equalities','physical_inequalities','current_ice_definition_v1'}
    current_model={k:v for k,v in model.items() if k in model_keys}
    def drop_old_refs(v):
        if isinstance(v,dict):
            if 'file'in v and 'sha256'in v:return 'offline source lineage: '+str(v['file'])+' sha256:'+v['sha256']
            return {k:drop_old_refs(x) for k,x in v.items()}
        if isinstance(v,list):return [drop_old_refs(x) for x in v]
        return v
    current_model=drop_old_refs(current_model);source_model_ref=save(current_model,'source_model.json')
    # This is a one-time full-primal transport into the current data version;
    # it does not claim the old objective identity or token is unchanged.
    initial=Path(model_path).parent;cp=initial/'engineering_checkpoint.json';raw=initial/'explicit_nuisance'/'manifest.json'
    require(cp.is_file() and raw.is_file(),'Current export needs the complete accepted initial state')
    state_binding=strict_json((initial/'observed_row_initialization_state_binding.json').read_bytes());aux=strict_json(raw.read_bytes())
    cp_dest=out/'initial_checkpoint.json';shutil.copyfile(cp,cp_dest);cp_ref=file_ref(cp_dest);cp_ref['file']='initial_checkpoint.json'
    auxcols=aux.get('arrays',aux.get('columns',{}))
    if not auxcols and 'files'in aux:
        auxcols={k:dict(file=str(initial/'explicit_nuisance'/(k+('.u32' if k=='path' else '.f64'))),sha256=aux['files'][k+('.u32' if k=='path' else '.f64')]['sha256'],dtype='<u4' if k=='path' else '<f8',shape=[n],layout='C',byte_order='little') for k in ('station','row','alpha','path')}
    if not auxcols:auxcols={k:aux[k] for k in ('station','row','alpha','path') if k in aux}
    require(set(auxcols)>={'station','row','alpha','path'},'Complete current state is missing raw auxiliary columns')
    transport=dict(schema='rinkfit.current_complete_primal_transport.v1',initialization='explicit_current_input_rebind',model=source_model_ref,checkpoint=cp_ref,
      raw_rows=n,raw_auxiliary={k:asset(auxcols[k]) for k in ('station','row','alpha','path')},from_state_token=state_binding['state_token'],
      preserve_complete_block_values=True,preserve_full_raw_auxiliary=True,preserve_motion_and_global_duals=True,new_objective_identity_required=True)

    manifest=dict(schema=SCHEMA,version=1,logical_source_id=logical_source_id,raw_rows=n,physical_rows=qualification['physical_count'],training_rows=qualification['training_count'],fit_rows=len(fit),inspection_rows=len(inspection),packet=packet_ref,source_model=source_model_ref,qualification=qualification,motion=motion,initial_primal=transport,numeric_domain=numeric_domain_ref,engineering_support_budget_v2=common,
        provenance=[str(Path(resolved_path)),str(Path(selection_path)),str(Path(model_path)),str(Path(motion_path)),'Original evidence and validator history are offline lineage; no production dependency.'],quality_accepted=False,source_image_identity_proven=False)
    # A current dependency must be a package-relative data address. This is a
    # structural packaging check, not a replay of historical source mathematics.
    def current_refs(v):
        if isinstance(v,dict):
            if 'file'in v and 'sha256'in v:
                p=Path(v['file']);require(not p.is_absolute() and '..'not in p.parts,'External current-input dependency');require((out/p).is_file(),'Current data file missing: '+str(p))
            for x in v.values():current_refs(x)
        elif isinstance(v,list):
            for x in v:current_refs(x)
    current_refs(packet);current_refs(manifest);manifest_ref=save(manifest,'manifest.json')
    result=dict(manifest=dict(manifest_ref,file=str((out/'manifest.json').absolute())),raw_rows=n,source_id_rows=id_rows,fit_rows=len(fit),inspection_rows=len(inspection),copied_payloads=len(copied),payload_bytes=sum(x['bytes'] for x in ledger),elapsed_seconds=time.perf_counter()-started,solver_started=False)
    (out/'export.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n','utf8');return result

def prepare_current_run(spec,spec_ref,paths):
    """Prepare only the current package and program; no historical proof route."""
    from .io_utils import read_json,file_sha256
    cfg=read_json(paths['configuration']);binding=cfg['current_input_v1']
    package=Path(binding['file']);package=(package if package.is_absolute() else paths['configuration'].parent/package).absolute();require(file_sha256(package)==binding['sha256'],'Current input manifest changed')
    m=strict_json(package.read_bytes());require(m['schema']==SCHEMA and m['version']==1,'Current data schema/version differs')
    forbidden={'effective_input_v1','prepared_effective_input_v1','dense_source_qualification','engineering_dense_source_weights_v1','representative_source_selection_v1','representative_continuation_v1','complete_stage_primal_v1','archived_dependency_aliases','native_source_archive_members'}
    require(not(forbidden&cfg.keys()),'Current run contains historical proof/continuation entry: '+','.join(sorted(forbidden&cfg.keys())))
    def local(r):
        p=Path(r['file']);require(not p.is_absolute() and '..'not in p.parts,'Current package has external data dependency');return package.parent/p
    require(paths['packet'] in (package,local(m['packet']).absolute()),'Current CLI packet differs from current package')
    require(cfg.get('engineering_active_shot_ids')==['s01','s02','s03','s04','s05','s06'],'Current six-shot source scope differs')
    require(cfg.get('source_uv_capture_objective')=='rinkfit.sourceUV_observed_row_finite_initialization.v1','Current complete-state entry currently supports observed stage')
    require(cfg.get('write_accepted_physical_preview') is False,'Current run preview policy must be explicit')
    require(m['raw_rows']==m['qualification']['raw_rows'] and m['fit_rows']==m['qualification']['fit_count'] and m['inspection_rows']==m['qualification']['inspection_count'],'Current source domain denominators differ')
    gpu=cfg['gpu_raw_uv'];ptx=Path(gpu['ptx']).absolute();require(file_sha256(ptx)==gpu['ptx_sha256'],'Current GPU kernel bytes differ')
    backend=cfg.get('linear_solver_backend');require(backend in ('cuda_cgnr','cpu_eigen_sparse_schur') and gpu.get('batch_reuse')is True,'Current linear backend/cache must be explicit')
    cache=Path(spec.get('cuda_cache_path') or ptx.parent/'cuda_driver_cache').absolute()
    objective=cfg['source_uv_capture_objective'];out=Path(spec['output']).absolute()
    command=[str(paths['native']),'--engineering-recovery',str(paths['model']),str(paths['packet']),str(out),str(paths['configuration'])]
    return dict(schema='rinkfit.current_joint_run_preparation.v1',spec=spec_ref,output=str(out),model=spec['model'],packet=spec['packet'],native=spec['native'],configuration=spec['configuration'],
      runtime_working_directory=str(package.parent),proofs={},current_input_v1=binding,input_data_version=m['version'],command=command,runtime_bin_paths=[str(paths['native'].parent)],cuda_driver_cache_path=str(cache),
      native_raw_UV_batch_reuse=True,linear_backend=backend,linear_solve_on_GPU=backend=='cuda_cgnr',PTX=dict(file=str(ptx),sha256=gpu['ptx_sha256']),
      raw_rows=m['raw_rows'],parent_raw_rows=0,new_raw_rows=0,active_rows=m['fit_rows'],new_active_rows=0,unknown_new_rows=m['raw_rows']-m['physical_rows'],physical_rows=m['physical_rows'],inspection_rows=m['inspection_rows'],current_source_generation=None,source_full_denominator_retained=True,
      objective_id=objective,numerical_initialization_only=True,true_RS_verification_required_after_this_run=True,phase_II_allowed=False,actual_native_source_gate_required_on_every_execution=False,current_typed_data_checks_required=True,source_accuracy_or_model_quality_accepted=False,
      native_launched=False,accepted_source_anchor_review=None,source_anchor_review_baseline=None,effective_input_preflight=None,
      graph_verification_only=bool(cfg.get('source_uv_capture_preflight_only') or cfg.get('source_uv_capture_construct_problem_only')),full_graph_preflight_without_solver=bool(cfg.get('source_uv_capture_preflight_only')),
      historical_packet_or_qualification_replay=False,historical_source_code_compatibility_gate=False)



def revise_current_qualification(manifest_path, proposal_paths, out_dir, model_path=None, circle_policy_scope_path=None):
    """Issue current qualification values without reopening source history.

    Only withdrawals are supported. Pixels, clocks, IDs, Query/W, motion and full
    raw state are immutable. Positive-rank W units remain indivisible.
    """
    import os
    started=time.perf_counter();src=Path(manifest_path).resolve();root=src.parent
    before=file_ref(src);m=strict_json(src.read_bytes());require(m['schema']==SCHEMA and m['version']==1,'Unsupported current input')
    out=Path(out_dir).resolve();require(not out.exists(),'Refuse to overwrite current data revision')
    def local(r):
        p=Path(r['file']);require(not p.is_absolute() and '..' not in p.parts,'External current dependency');return root/p
    def read(r):
        raw=local(r).read_bytes();require(hashlib.sha256(raw).hexdigest()==r['sha256'],'Current JSON changed: '+r['file']);return strict_json(raw)
    def column(r):return np.memmap(local(r),dtype=np.dtype(r['dtype']),mode='r',offset=r.get('offset_bytes',0),shape=tuple(r['shape']))
    packet=read(m['packet']);q=m['qualification'];n=m['raw_rows'];require(packet['row_count']==n,'Current raw denominator')
    arrays={k:column(packet['arrays'][k]) for k in ('uv','native_pts','shot_index','material_index','path','admission','fit_role')}
    old_fit_sha=q['fit_rows']['sha256'];physical=np.array(column(q['physical_mask']),copy=True);training=np.array(column(q['training_mask']),copy=True);fit=np.array(column(q['fit_rows']),copy=True)
    chosen=np.zeros(n,dtype=bool);chosen[fit]=True;old_chosen=chosen.copy()
    require(np.array_equal(physical.astype(bool),(arrays['admission']==1)&(arrays['fit_role']==1)),'Current physical values differ')
    registries=[packet['dense_source_binding'],*packet['native_ad_source_bindings']];id_cache={}
    def source_id(i):
        prefix=packet['source_row_ids']
        if i<len(prefix):return prefix[i]
        for k,r in enumerate(registries):
            a=int(r['row_start']);z=a+int(r['count'])
            if a<=i<z:
                if k not in id_cache:id_cache[k]=(column(r['source_id_offsets']),column(r['source_id_utf8']))
                offsets,raw=id_cache[k];j=i-a;return bytes(raw[int(offsets[j]):int(offsets[j+1])]).decode('utf8')
        raise ValueError('Current raw ID is absent: '+str(i))
    proposals=[];decisions={}
    for path in proposal_paths:
        p=Path(path);v=strict_json(p.read_bytes());require(v['schema']=='rinkfit.current_source_identity_revision_proposal.v1','Unknown source decision format')
        require(v['current_input_v1']['sha256']==before['sha256'] and v['logical_source_id']==m['logical_source_id'],'Source decision bound to another current input')
        proposals.append(dict(file=str(p.resolve()),sha256=file_ref(p)['sha256']))
        for row in v['rows']:
            i=int(row['raw_row']);require(0<=i<n and row['decision'] in ('unknown','rejected','revoked') and row['uses_fitted_model_projection'] is False,'Only source-evidence withdrawals are supported')
            material,path_index=packet.get('current_print_path_overrides',{}).get(str(i),(int(arrays['material_index'][i]),int(arrays['path'][i])))
            if material==4294967295:material=-1
            require(row['source_row_id']==source_id(i) and int(row['native_pts'])==int(arrays['native_pts'][i]) and int(row['shot_index'])==int(arrays['shot_index'][i]) and int(row['material_index'])==material and int(row['path'])==path_index and np.array_equal(row['uv'],arrays['uv'][i]),'Source decision changed immutable row: '+str(i))
            require(bool(row.get('reason')),'Source decision omitted reason')
            if i in decisions:require(decisions[i]==row,'Conflicting source decisions')
            decisions[i]=row
    require(decisions,'No source decisions to apply')
    withdraw=np.array(sorted(decisions),dtype=np.uint64);old_membership={int(i):dict(physical=bool(physical[i]),training=bool(training[i]),fit=bool(chosen[i])) for i in withdraw}
    physical[withdraw]=0;training[withdraw]=0;chosen[withdraw]=False;removed_groups=[]
    for g in packet['groups']:
        if int(g['rank'])<=0:continue
        rows=np.asarray(g['rows'],dtype=np.uint64);old_rows=rows[old_chosen[rows]]
        # Original W metadata also retains previously marginalized failures.
        # Compare the current consumed unit; do not replay or reject its history.
        if len(old_rows) and not np.all(chosen[old_rows]):
            chosen[old_rows]=False
            removed_groups.append(dict(id=g['id'],rank=int(g['rank']),previous_fit_rows=list(map(int,old_rows)),original_rows=list(map(int,rows))))
    new_fit=np.flatnonzero(chosen).astype('<u8');inspection=np.flatnonzero(physical.astype(bool)&~chosen).astype('<u8')
    removed=np.setdiff1d(fit,new_fit,assume_unique=True);require(np.all(training[new_fit]) and np.all(training<=physical),'Revised fit/training domain invalid')
    if model_path:
        candidate=strict_json(Path(model_path).read_bytes());authority=read(m['source_model'])
        require(candidate['blocks']==authority['blocks'],'Current qualification revision cannot change physical blocks or complete checkpoint values')
        allowed={'physical_equalities','physical_inequalities','Lane_relation_restoration'}
        changed={k for k in set(candidate)|set(authority) if candidate.get(k)!=authority.get(k)}
        require(changed<=allowed,'Current qualification revision changed static model fields: '+','.join(sorted(changed-allowed)))
        for kind in ('physical_equalities','physical_inequalities'):
            old=authority.get(kind,[]);require(candidate.get(kind,[])[:len(old)]==old,'Current model revision replaced an existing physical relation')
    source_model_doc=strict_json(Path(model_path).read_bytes()) if model_path else read(m['source_model'])
    circle_transition=None;circle_scope=None;retired=None;current_material=None
    if circle_policy_scope_path:
        from .joint_model import revise_c0_midline_without_a
        circle_scope=strict_json(Path(circle_policy_scope_path).read_bytes())
        require(circle_scope['schema']=='rinkfit.current_circular_paint_source_scope.v1' and circle_scope['current_input_v1']['sha256']==before['sha256'] and circle_scope['logical_source_id']==m['logical_source_id'],'Circle source scope belongs to another current input')
        require(circle_scope['source_projection_used'] is False and circle_scope['raw_UV_ID_PTS_preserved'] is True,'Circle midpoint scope changed original pixels')
        ref=circle_scope['sets']['C0_bound_midline'];raw=Path(ref['file']).read_bytes()
        require(hashlib.sha256(raw).hexdigest()==ref['sha256'] and ref['dtype']=='<u8','Current C0 midpoint row binding changed')
        midline=np.frombuffer(raw,dtype='<u8')
        require(midline.shape==tuple(ref['shape']) and len(midline)>0 and np.all(midline[1:]>midline[:-1]) and int(midline[-1])<n,'Invalid complete C0 midpoint domain')
        old_materials=source_model_doc['materials'];c0={i for i,x in enumerate(old_materials) if x['label'].split(':')[0]=='ice.C0'}
        require(np.all(np.isin(arrays['material_index'][midline],list(c0))) and np.all(arrays['path'][midline]==0),'C0 midpoint rows are not the same original material path')
        require(circle_scope['existing_midline_max_replay_difference_px']==0,'Unresolved current midpoint replay')
        source_model_doc,circle_transition=revise_c0_midline_without_a(source_model_doc)
        material_map=np.asarray(circle_transition['old_material_to_new'],dtype=np.int64)
        current_material=material_map[np.asarray(arrays['material_index'],dtype=np.int64)]
        retired=current_material<0
        retired|=np.isin(arrays['material_index'],list(c0));retired[midline]=False
        require(not np.any(physical[retired]) and not np.any(training[retired]) and not np.any(chosen[retired]),'Retired circle source rows still have current qualification; supply explicit source decisions')
        current_material[retired]=-1
        require(np.all(current_material[midline]>=0),'Current C0 midline material was deleted')
    # Copy only current required payloads; history/evidence stays plain provenance.
    refs={}
    def collect(value):
        if isinstance(value,dict):
            if 'file' in value and 'sha256' in value:
                p=local(value);key=value['file'];require(p.is_file(),'Current required file missing: '+str(p));require(key not in refs or refs[key]['sha256']==value['sha256'],'Conflicting current file identity');refs[key]=value
            else:
                for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    collect(m);collect(packet);out.mkdir(parents=True);(out/'a').mkdir()
    rewritten={m['packet']['file'],m['source_model']['file'],q['source_identity_verdicts']['file']}
    for name,ref in refs.items():
        if name in rewritten:continue
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(local(ref),dest)
        except OSError:shutil.copyfile(local(ref),dest)
    def save(value,name):
        p=out/name;require(not p.exists(),'Refuse to overwrite revision file '+name);p.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+'\n','utf8');r=file_ref(p);r['file']=name;return r
    def save_array(value,dtype):
        value=np.ascontiguousarray(value,dtype=dtype);raw=value.tobytes();sha=hashlib.sha256(raw).hexdigest();name='a/'+sha+'.bin';p=out/name
        if not p.exists():p.write_bytes(raw)
        return dict(file=name,sha256=sha,dtype=np.dtype(dtype).str,shape=list(value.shape),layout='C',byte_order='little')
    admission=np.array(arrays['admission'],copy=True);admission[withdraw]=0;packet['arrays']['admission']=save_array(admission,'|u1')
    for key in ('active_rows','admitted_rows'):
        if key in packet:packet[key]=int(physical.sum())
    packet['current_withheld_rows']=sorted(set(map(int,packet.get('current_withheld_rows',[])))|set(map(int,withdraw)))
    q['physical_mask']=save_array(physical,'|u1');q['training_mask']=save_array(training,'|u1');q['fit_rows']=save_array(new_fit,'<u8');q['inspection_rows']=save_array(inspection,'<u8')
    q.update(physical_count=int(physical.sum()),training_count=int(training.sum()),fit_count=len(new_fit),inspection_count=len(inspection))
    if circle_transition is not None:
        require('source_material_index' not in packet['arrays'],'Current circle definition already migrated')
        packet['source_material_binding']=copy.deepcopy(packet['material_binding'])
        packet['arrays']['source_material_index']=packet['arrays']['material_index']
        packet['arrays']['material_index']=save_array(current_material.astype('<u4'),'<u4')
        packet['arrays']['model_retired']=save_array(retired.astype('u1'),'|u1')
        packet['material_binding']=dict(schema='rinkfit.current_material_binding.v1',model_material_prefix_count=len(source_model_doc['materials']),materials=[
            dict(index=i,label=x['label'],paths=[dict(index=j,id=y['id'],physical_point=y.get('physical_point',False)) for j,y in enumerate(x['paths'])]) for i,x in enumerate(source_model_doc['materials'])])
        packet['current_model_subset_v1']=dict(schema='rinkfit.current_model_source_binding.v1',retired_raw_rows=int(retired.sum()),retired_material_sentinel=4294967295,source_pixels_and_ids_unchanged=True,source_material_catalog='source_material_binding',current_C0_midline_rows=list(map(int,midline)),C0_definition='arithmetic_width_midpoint_of_two_actual_material_edges',excluded_entities=['A1','A2','A3','A4'])
        packet['allowed_paths']={k:v for k,v in packet.get('allowed_paths',{}).items() if not retired[int(k)]}
        if 'source_station_domains' in packet:packet['source_station_domains']['rows']=[x for x in packet['source_station_domains']['rows'] if not retired[int(x['row_index'])]]
        for key,value in packet.get('current_print_path_overrides',{}).items():
            require(not retired[int(key)],'Retired curve unexpectedly has a print identity override')
            value[0]=int(material_map[value[0]])
        for reg in [packet['dense_source_binding'],*packet['native_ad_source_bindings']]:
            if 'allowed_paths_by_material' in reg:reg['allowed_paths_by_material']={str(int(material_map[int(k)])):v for k,v in reg['allowed_paths_by_material'].items() if material_map[int(k)]>=0}
        packet['native_ad_source_binding']=packet['native_ad_source_bindings'][-1]
        dense=packet['dense_source_binding'];mask=np.array(column(dense['allowed_path_mask']),copy=True);start=int(dense['row_start']);mask[retired[start:start+len(mask)]]=0;dense['allowed_path_mask']=save_array(mask,'|u1')
        removed_pairs=set(circle_transition['removed_stroke_pair_ids']);old_pairs=packet.get('stroke_pairs',[])
        pair_map=np.full(len(old_pairs),-1,dtype=np.int32);new_pairs=[]
        for i,x in enumerate(old_pairs):
            if x['id'] in removed_pairs:
                require(np.all(retired[np.asarray(x['rows'],dtype=np.int64)]),'Deleted width pair retains a current midpoint')
                continue
            pair_map[i]=len(new_pairs);y=copy.deepcopy(x);y['material_index']=int(material_map[x['material_index']]);require(y['material_index']>=0,'Retained width pair has no material');new_pairs.append(y)
        actual_pair_ids={x['id'] for x in old_pairs}
        require(len(old_pairs)-len(new_pairs)==len(removed_pairs&actual_pair_ids),'Deleted consumed model width pairs differ from actual source pairs')
        circle_transition['source_pairs_deleted']=len(old_pairs)-len(new_pairs)
        circle_transition['unused_model_width_bindings_deleted']=len(removed_pairs-actual_pair_ids)
        if not old_pairs:
            for key in ('stroke_pair_index','stroke_source_side'):
                if key in packet['arrays']:require(np.all(column(packet['arrays'][key])==-1),'Empty source pairs retain an active pair index')
        if old_pairs:
            pair_index=np.array(column(packet['arrays']['stroke_pair_index']),copy=True);side=np.array(column(packet['arrays']['stroke_source_side']),copy=True)
            selected=pair_index>=0;pair_index[selected]=pair_map[pair_index[selected]];side[pair_index<0]=-1
            packet['arrays']['source_stroke_pair_index']=packet['arrays']['stroke_pair_index'];packet['arrays']['source_stroke_source_side']=packet['arrays']['stroke_source_side']
            packet['arrays']['stroke_pair_index']=save_array(pair_index,'<i4');packet['arrays']['stroke_source_side']=save_array(side,'|i1');packet['stroke_pairs']=new_pairs
        # Canonical installation/entity budget indices refer to current materials.
        def remap_material_entries(value):
            if isinstance(value,list):
                return [remap_material_entries(x) for x in value if not(isinstance(x,dict) and 'material_index' in x and material_map[int(x['material_index'])]<0)]
            if isinstance(value,dict):
                result={k:remap_material_entries(v) for k,v in value.items()}
                if 'material_index' in value:
                    mi=int(material_map[int(value['material_index'])]);require(mi>=0,'Unretired budget entity has no model material');result['material_index']=mi
                    if 'material_label' in result:result['material_label']=source_model_doc['materials'][mi]['label']
                return result
            return value
        m['engineering_support_budget_v2']=remap_material_entries(m['engineering_support_budget_v2'])
        groups=m['engineering_support_budget_v2']['canonical_material_entities']['groups']
        m['engineering_support_budget_v2']['canonical_material_entities']['groups']=[x for x in groups if x.get('materials')]
    old_verdicts=read(q['source_identity_verdicts']);verdicts={int(x['raw_row']):x for x in old_verdicts['rows']}
    for i,row in decisions.items():
        entry=dict(row);entry['previous_membership']=old_membership[i]
        if i in verdicts:entry['previous_decision']=verdicts[i]['decision']
        verdicts[i]=entry
    if circle_transition is not None:
        for i,row in verdicts.items():
            row['source_material_index']=row['material_index'];row['source_material_label']=packet['source_material_binding']['materials'][int(row['material_index'])]['label'];row['material_index']=int(current_material[i])
    old_verdicts['rows']=[verdicts[i] for i in sorted(verdicts)];q['source_identity_verdicts']=save(old_verdicts,'source_identity.json')
    m['packet']=save(packet,'packet.json')
    if model_path is None and circle_transition is None:
        # Qualification-only changes preserve the exact current model file.
        model_ref=copy.deepcopy(m['source_model']);dest=out/model_ref['file']
        dest.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(local(model_ref),dest)
        except OSError:shutil.copyfile(local(model_ref),dest)
    else:
        model_ref=save(source_model_doc,'source_model.json')
    m['source_model']=model_ref;m['initial_primal']['model']=model_ref
    transport=m['initial_primal'];require(transport['initialization']=='explicit_current_input_rebind','Current revision requires complete explicit initial transport')
    previous_transition=transport.get('current_qualification_transition_v1',{})
    complete_removed=sorted(set(map(int,removed))|set(previous_transition.get('removed_fit_rows',[])))
    if complete_removed:
        transport['current_qualification_transition_v1']=dict(schema='rinkfit.current_qualification_transition.v1',new_objective_identity_required=True,from_state_token=transport['from_state_token'],logical_source_id=m['logical_source_id'],previous_fit_rows_sha256=previous_transition.get('previous_fit_rows_sha256',old_fit_sha),target_fit_rows_sha256=q['fit_rows']['sha256'],removed_fit_rows=complete_removed,reason='Explicit current source identity withdrawals; original raw auxiliaries retained')
    if circle_transition is not None:
        transport['current_model_subset_transition_v1']=dict(schema='rinkfit.current_model_subset_transition.v1',new_objective_identity_required=True,from_state_token=transport['from_state_token'],target_model_sha256=model_ref['sha256'],removed_model_parameter_names=sorted(circle_transition['removed_model_parameter_names']),component_subsets=circle_transition['component_subsets'],reason='User-defined C0 width midline only; A1-A4 excluded from current model')
        save(circle_transition,'model_subset.json')
    m.update(physical_rows=q['physical_count'],training_rows=q['training_count'],fit_rows=q['fit_count'],inspection_rows=q['inspection_count'])
    m['provenance']=[*m.get('provenance',[]),'Previous current version '+before['sha256'],*[x['file']+' sha256:'+x['sha256'] for x in proposals]]
    m['source_image_identity_proven']=False;m['quality_accepted']=False
    receipt=dict(schema='rinkfit.actual_current_qualification_revision.v1',previous_manifest_sha256=before['sha256'],decision_rows=len(decisions),withdrawn_physical_rows=sum(x['physical'] for x in old_membership.values()),withdrawn_training_rows=sum(x['training'] for x in old_membership.values()),removed_fit_rows=list(map(int,removed)),whole_W_groups_removed=removed_groups,raw_rows=n,fit_rows=len(new_fit),inspection_rows=len(inspection),raw_UV_PTS_ID_Query_W_motion_and_complete_auxiliary_preserved=True,weights_must_be_rebuilt_by_current_budget_policy=True,cost_change_is_not_geometry_improvement=True,history_opened=False,solver_started=False,elapsed_seconds=time.perf_counter()-started)
    manifest_ref=save(m,'manifest.json');receipt['manifest']=dict(manifest_ref,file=str(out/'manifest.json'));save(receipt,'revision.json');return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resolved',required=True);p.add_argument('--selection',required=True);p.add_argument('--model',required=True);p.add_argument('--motion',required=True);p.add_argument('--out',required=True);p.add_argument('--payload-map');p.add_argument('--config');a=p.parse_args();print(json.dumps(export(a.resolved,a.selection,a.model,a.motion,a.out,a.payload_map,config_path=a.config),ensure_ascii=False))
