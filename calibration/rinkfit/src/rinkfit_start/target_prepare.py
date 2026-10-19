"""Fresh operators for an explicitly selected new native-source target.

Retains the historical parent at SHA
9d9fca030b558b57c80a755b1b3a235d9beea1ee222bb55ef6d504456c892af3.
No optimizer history, multipliers, or old-target continuation is imported.
"""
from pathlib import Path
import copy,json,os,subprocess,sys,time
import numpy as np
from .state_integrity import (read_json,sha256_file,verify_reference,load_array,
    validate_candidate_state,validate_receipt_bindings,StateIntegrityError)
from .semantic_resolution import load_semantics


def ref(path):
    return dict(path=str(Path(path).resolve()),sha256=sha256_file(path))


def selected_scope(packet,semantic):
    """Verify source authority and exact active partition/rank before launch."""
    packet,semantic=Path(packet),Path(semantic);pk,sem=read_json(packet),read_json(semantic)
    if pk.get('semantic_packet',{}).get('sha256')!=sha256_file(semantic):
        raise StateIntegrityError('Packet is not bound to selected source semantics')
    n=int(pk['observations']);_,a,_,extra=load_semantics(semantic,n,expected_sha256=sha256_file(semantic))
    raw={k:load_array(v,packet.parent)for k,v in pk['arrays'].items()}
    if not np.array_equal(raw['source_rows'],np.arange(n)) or raw['native_uv'].shape!=(n,2):
        raise StateIntegrityError('Selected raw source row/UV identity differs')
    active=load_array(sem['active_source_mask'],semantic.parent)
    if active.shape!=(n,) or not np.isin(active,[0,1]).all():raise StateIntegrityError('Invalid selected source mask')
    active=active.astype(bool);part=np.zeros(n,np.int32);rank=0
    replacements=extra.get('source_exclusion',{}).get('replacements',{})
    gp=verify_reference(sem['assets']['groups.jsonl'],semantic.parent)
    for line in gp.read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        g=json.loads(line);rr=a['group_rows'][g['member_start']:g['member_start']+g['member_count']].astype(int);r=g['rank']
        if g['group_id']in replacements:rep=replacements[g['group_id']];rr,r=rep['source_rows'],rep['rank']
        part[rr]+=1;rank+=r
    independent=a['ordinary_rows'].astype(int);independent=independent[active[independent]];part[independent]+=1;rank+=2*len(independent)
    stats=read_json(verify_reference(sem['active_statistics'],semantic.parent))
    if not np.array_equal(part,active.astype(np.int32)) or stats['formal_residual_rows']!=rank or stats['active_observations']!=int(active.sum()):
        raise StateIntegrityError('Selected active statistics do not cover each active source exactly once')
    if len(np.unique(raw['frame_index']))!=pk['frame_count']:raise StateIntegrityError('Selected frame count differs')
    return dict(raw_rows=n,active_rows=int(active.sum()),profiles=int(np.sum(active&(a['kind']==2))),
        entity_rows=int(np.sum(active&(a['kind']==1))),ordinary_rows=int(np.sum(active&(a['kind']==0))),
        W_rank=rank,frames=pk['frame_count'],source_identity=pk['immutable_source_hashes']['observations.npz'],
        packet_sha256=sha256_file(packet),semantic_sha256=sha256_file(semantic),
        query_sha256=pk['query_packet']['sha256'],active_mask_sha256=sem['active_source_mask']['sha256'],
        active_statistics_sha256=sem['active_statistics']['sha256'],excluded_rows=np.flatnonzero(~active).tolist())


def run(checkpoint,context,output,stop_file=None,*,policy):
    start=time.monotonic();CP,CT,D=Path(checkpoint).resolve(),Path(context).resolve(),Path(output).resolve()
    C,S=read_json(CP),read_json(CT);source=verify_reference(S['source_input'],CT.parent);spec=copy.deepcopy(read_json(source))
    packet=verify_reference(spec['packet'],source.parent);semantic=verify_reference(spec['semantic'],source.parent)
    if S.get('baseline_reference'):verify_reference(S['baseline_reference'],CT.parent)
    checked=validate_candidate_state(CP.parent,packet,semantic);scope=selected_scope(packet,semantic)
    if C.get('schema')=='rinkfit.complete_recomputed_native_candidate.v1':validate_receipt_bindings(C,CP.parent,validated=checked)
    elif C.get('schema')=='rinkfit.complete_continuous_reduced_SLP_checkpoint.v1':
        import hashlib
        body=dict(C);token=body.pop('state_token',None)
        if hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=token:
            raise StateIntegrityError('Complete source checkpoint token differs')
        verify_reference(C['model'],CP.parent,expected_path=CP.parent/'model.json')
        for key,name in [('station','station.f64'),('branch','path.u32'),('row','initial_row.f64'),('alpha','initial_alpha.f64')]:
            verify_reference(C['row_state_files'][key],CP.parent,expected_path=CP.parent/name)
        if C['raw_rows']!=scope['raw_rows'] or C['active_rows']!=scope['active_rows'] or C['metrics']!=checked['report']['best']:
            raise StateIntegrityError('Complete source checkpoint values/scope differ')
    else:raise StateIntegrityError('A complete new-target value receipt/checkpoint is required')
    if stop_file and Path(stop_file).exists():raise InterruptedError('STOP before fresh operator generation')
    if D.exists():raise FileExistsError(str(D))
    D.mkdir(parents=True)
    for key in ['verified_complete_initial_state','prepared_first_step','reuse_parent_full_source_initial_state',
                'algorithm_continuation_context','discarded_execution_boundary','parent_complete_PHR_checkpoint']:
        spec.pop(key,None)
    spec['model']=ref(CP.parent/'model.json')
    spec['initial_state']={key:dict(**ref(CP.parent/name),dtype=dtype,shape=[scope['raw_rows']])
        for key,name,dtype in [('station','station.f64','<f8'),('path','path.u32','<u4'),('row','initial_row.f64','<f8'),('alpha','initial_alpha.f64','<f8')]}
    for key,value in policy.get('source_input_overrides',{}).items():spec[key]=copy.deepcopy(value)
    spec.pop('resident',None)
    spec['selected_target_scope']=scope
    spec['new_target_initial_state']=dict(descriptor=ref(CP),report=ref(CP.parent/'report.json'),
        predictions=ref(CP.parent/'candidate_predictions.npz'),baseline_reference=S.get('baseline_reference'),
        old_optimizer_history_or_duals_imported=False)
    spec['baseline_reference']=S.get('baseline_reference')
    prep=verify_reference(policy['preparer_driver'],CT.parent)
    IP=D/'input.json';IP.write_text(json.dumps(spec,indent=2),encoding='utf8')
    command=[sys.executable,'-B','-X','utf8',str(prep),'--input',str(IP),'--output',str(D/'prepared'),'--prepare-linearization-only']
    env={k.upper():v for k,v in os.environ.items()};env.update(PYTHONIOENCODING='utf8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    execution=dict(schema='rinkfit.new_source_target_operator_preparation.v1',source_complete_value_state=ref(CP),context=ref(CT),
        preparer=ref(prep),entry=ref(__file__),selected_scope=scope,command=command,new_target_initialization=True,
        previous_checkpoint_role='baseline_reference and ancestry only',old_history_duals_or_linearization_reused=False)
    (D/'execution.json').write_text(json.dumps(execution,indent=2),encoding='utf8')
    with(D/'stdout.log').open('x',encoding='utf8')as handle:
        code=subprocess.run(command,cwd=D,env=env,stdout=handle,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).returncode
    result_path=D/'prepared/result.json'
    if not result_path.is_file():raise RuntimeError('New-target preparation failed before complete boundary; return code '+str(code))
    result=read_json(result_path)
    if code or result['status']!='COMPLETE_SAME_STATE_OPERATORS_PREPARED_NO_MASTER_NO_PHYSICAL_STEP':
        raise RuntimeError('New-target operator preparation unfinished: '+result.get('status','missing')+' '+str(result.get('diagnostic',{})))
    N=verify_reference(result['current_complete_checkpoint'],D/'prepared');doc=read_json(N)
    if doc['active_rows']!=scope['active_rows'] or doc['raw_rows']!=scope['raw_rows'] or doc['control']['iteration']!=0:
        raise StateIntegrityError('New-target prepare has wrong scope or imported iteration history')
    tag='i00000_p00000';w=D/'prepared/work'
    transport=ref(w/(tag+'_whole_W_transport/manifest.json'));t=read_json(transport['path'])
    if t['active_rows']!=scope['active_rows'] or t['full_W_rank']!=scope['W_rank'] or not t['statistical_partition_exactly_once']:
        raise StateIntegrityError('Prepared transport source rank/scope differs')
    out=dict(checkpoint=ref(N),linearization=doc['linearization'],transport=transport,
        profile_domains=ref(w/(tag+'_profile_domains/manifest.json')),base_directory=str(w/(tag+'_value_authority')),
        source_complete_value_state=ref(CP),baseline_reference=S.get('baseline_reference'),
        selected_scope=scope,target_id=doc['target_id'],new_target_initialization=True,
        old_history_duals_or_linearization_reused=False,elapsed_seconds=time.monotonic()-start,
        original_initial_capture_reused_not_repeated=True,no_master_or_physical_proposal=True)
    (D/'prepared.json').write_text(json.dumps(out,indent=2),encoding='utf8');print('NEW_TARGET_COMPLETE_OPERATORS_READY',json.dumps(out),flush=True);return out
