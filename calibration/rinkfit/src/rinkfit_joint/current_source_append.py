"""Append current source values and transport a complete current primal prefix.

Consumes one current package, one current complete checkpoint and newly reviewed
source rows. No producer ancestry, historical eligibility or source code is read.
The tail has explicit engineering geometry groups with unknown statistical noise.
"""
from pathlib import Path
from copy import deepcopy
import argparse, hashlib, json, os, shutil
import numpy as np
from .io_utils import file_sha256
from .current_input_v1 import file_ref
from .joint_review_packet import CurrentReviewPacket


def require(ok, reason):
    if not ok:
        raise ValueError('current source append: ' + reason)


def issue_current_source_append(manifest_path, checkpoint_path, closed_rows_path,
                                output, withdrawal_proposals=()):
    src = Path(manifest_path).resolve(); base = src.parent
    source_ref = file_ref(src); m = json.loads(src.read_text('utf8'))
    require(m['schema'] == 'rinkfit.current_joint_input.v1' and m['version'] == 1, 'current schema')
    out = Path(output).resolve(); require(not out.exists(), 'immutable output already exists')
    def pinned(ref, parent=base):
        p = Path(ref['file']); p = p if p.is_absolute() else Path(parent)/p
        require(p.is_file() and file_sha256(p) == ref['sha256'], 'changed current file ' + str(p))
        return p
    def col(ref, parent=base):
        return np.memmap(pinned(ref,parent), dtype=np.dtype(ref['dtype']), mode='r',
                         offset=ref.get('offset_bytes',0), shape=tuple(ref['shape']))
    packet = json.loads(pinned(m['packet']).read_text('utf8')); n0 = int(m['raw_rows'])
    previous_fit_sha=m['qualification']['fit_rows']['sha256']
    require(packet['row_count'] == n0, 'parent raw denominator')
    cpath = Path(checkpoint_path).resolve(); cp = json.loads(cpath.read_text('utf8')); cb = cpath.parent
    require(cp['schema'] == 'rinkfit.current_complete_primal_transport.v1'
            and cp['initialization'] == 'strict_current_checkpoint'
            and cp['current_input_v1']['sha256'] == source_ref['sha256']
            and cp['raw_rows'] == n0, 'complete parent checkpoint/current data binding')
    model_file = pinned(cp['model'], cb); checkpoint_file = pinned(cp['checkpoint'], cb)
    model = json.loads(model_file.read_text('utf8')); authority = json.loads(pinned(m['source_model']).read_text('utf8'))
    def static(j):
        v=deepcopy(j)
        for b in v['blocks']: b.pop('values',None)
        return v
    require(static(model) == static(authority), 'candidate changed current static model')
    old_aux = {k:col(cp['raw_auxiliary'][k],cb) for k in ('station','row','alpha','path')}
    for k,a in old_aux.items(): require(a.shape == (n0,), 'complete raw auxiliary ' + k)
    closed_path=Path(closed_rows_path).resolve(); closed=json.loads(closed_path.read_text('utf8'))
    require(closed['schema']=='rinkfit.current_direct_board_closed_material_readouts.v1'
            and closed['parent_current_input']['sha256']==source_ref['sha256']
            and closed['source_or_model_projection_used_for_identity'] is False
            and closed['native_RGB_sha256'], 'new current source declaration')
    rows=closed['rows']; count=len(rows); n=n0+count
    require(count>0 and closed['passed_readouts']==count, 'new closed source denominator')
    roles=json.loads(pinned(m['qualification']['frame_roles']).read_text('utf8'))['frame_roles']
    material_ids={x['label']:i for i,x in enumerate(model['materials'])}
    old_source_ids={x['label']:x['index'] for x in packet['source_material_binding']['materials']}
    # Only the supplied current candidate initializes finite coordinates. This
    # nearest numerical seed never defines identity or changes measured pixels.
    from .dense_visible_metrics import ReviewScene
    from .joint_review_camera import CurrentJointCameras
    from .dense_joint_packet import _fast_seed_frame
    scene=ReviewScene(model); cameras=CurrentJointCameras(model); tail=[]; seen=set()
    for r in rows:
        rid=r['source_row_id']; sid=r['shot_id']; pts=int(r['native_pts']); uv=np.asarray(r['current_uv'],float)
        require(rid not in seen and r['source_uv']==r['current_uv'] and uv.shape==(2,)
                and np.isfinite(uv).all() and np.all(uv>=[0,0]) and np.all(uv<=[1919,1079]), 'new unchanged UV/ID')
        seen.add(rid); role=roles[str(pts)]
        require(role['shot_id']==sid and role['role'] not in ('validation_guard','reserved_pending_lineage','sealed_validation'), 'new protected source exposure')
        proof=r['current_full_material_sequence_evidence']
        require(proof['decision']=='source_material_order_closed' and not proof['reasons']
                and proof['source_RGB_sha256']==closed['native_RGB_sha256']
                and proof['current_uv']==r['current_uv'] and proof['source_row_id']==rid, 'new complete material sequence proof')
        label=r['material_role']; require(label in ('board.top','board.contact'), 'unsupported direct source primitive')
        mi=material_ids[label]; paths=model['materials'][mi]['paths']
        require(paths and all(p['kind']==3 and p.get('boundary_segment')==-2 for p in paths), 'whole continuous board material identity')
        pi=0; normal=np.asarray(r['normal'],float); seed=np.asarray(r['navigation_seed_uv'],float)
        a=r['source_measurement_audit']['continuous_cubic_material_profile']; offsets=np.asarray(a['actual_profile_normal_offsets'],float)
        intensity=np.asarray(a['actual_profile_intensity'],float); alpha=float(a['normal_offset_px'])
        require(normal.shape==(2,) and abs(np.linalg.norm(normal)-1)<1e-9 and offsets.shape==(49,)
                and np.all(np.diff(offsets)>0) and intensity.shape==offsets.shape and np.isfinite(intensity).all()
                and offsets[0]<=alpha<=offsets[-1] and a['numeric_valid'] and not a['synthetic_window_boundary_count']
                and np.max(np.abs(seed+alpha*normal-uv))<1e-9, 'actual frozen source profile/mean')
        station=.5 # Legal placeholder; existing batch numerical initializer below.
        tail.append(dict(source_row_id=rid,shot_id=sid,native_pts=pts,uv=uv.tolist(),material_index=mi,
            source_material_index=int(old_source_ids[label]),path=pi,station=station,row=float(uv[1]),alpha=alpha,
            normal=normal.tolist(),seed=seed.tolist(),offsets=offsets.tolist(),intensity=intensity.tolist(),
            role=label,source_family_id='direct_current_board/'+sid+'/'+str(pts)+'/'+label,
            numeric_initialization_only=True,entity_identity_from_model=False))
    for sid,pts in sorted({(r['shot_id'],r['native_pts'])for r in tail}):
        ii=[i for i,r in enumerate(tail)if r['shot_id']==sid and r['native_pts']==pts]
        source_rows=[dict(source_uv=tail[i]['uv'],material_index=tail[i]['material_index'],path_index=tail[i]['path'],
            allowed_paths=list(range(len(model['materials'][tail[i]['material_index']]['paths']))))for i in ii]
        local=float(cameras.local_seconds(sid,np.array([pts],np.int64))[0])
        branches,stations,status,errors=_fast_seed_frame(scene,cameras.native_camera(sid),local,source_rows)
        require(np.all(status==1) and np.isfinite(stations).all() and np.all((stations>=0)&(stations<=1)), 'existing finite current numerical source initializer')
        for j,i in enumerate(ii):tail[i]['path']=int(branches[j]);tail[i]['station']=float(stations[j]);tail[i]['numerical_seed_error_px']=float(errors[j])
    print('current source append: existing current batch tail numerical seeds complete',flush=True)
    proposals=[json.loads(Path(file).read_text('utf8')) for file in withdrawal_proposals]
    required_withdrawals={int(r['raw_row']) for proposal in proposals for r in proposal['rows']}
    reader=CurrentReviewPacket(pinned(m['packet']),current_input=source_ref,required_rows=required_withdrawals)
    physical=np.array(reader.current_source_membership_v1['physical'],copy=True)
    training=np.array(reader.current_source_membership_v1['training'],copy=True)
    old_fit=np.asarray(reader.current_source_membership_v1['fit_rows']); chosen=np.zeros(n0,bool);chosen[old_fit]=True
    previous_chosen=chosen.copy(); decisions={}
    raw_values={k:reader.load(k)for k in ('uv','native_pts','shot_index','material_index','path')}
    for proposal in proposals:
        require(proposal['schema']=='rinkfit.current_source_identity_revision_proposal.v1'
                and proposal['current_input_v1']['sha256']==source_ref['sha256']
                and proposal['logical_source_id']==m['logical_source_id'], 'withdrawal parent identity')
        for r in proposal['rows']:
            i=int(r['raw_row']); require(0<=i<n0 and r['decision'] in ('unknown','revoked','rejected')
                and r['uses_fitted_model_projection'] is False and r.get('reason'), 'withdrawal source decision')
            require(reader.source_id(i)==r['source_row_id'] and np.array_equal(raw_values['uv'][i],r['uv'])
                    and int(raw_values['native_pts'][i])==r['native_pts'] and int(raw_values['shot_index'][i])==r['shot_index']
                    and int(raw_values['material_index'][i])==r['material_index'] and int(raw_values['path'][i])==r['path'], 'withdrawal changed raw source')
            require(i not in decisions, 'duplicate withdrawal decision'); decisions[i]=r
    withdraw=np.array(sorted(decisions),dtype=np.uint64); old_membership={int(i):dict(physical=bool(physical[i]),training=bool(training[i]),fit=bool(chosen[i]))for i in withdraw}
    physical[withdraw]=0;training[withdraw]=0;chosen[withdraw]=False
    for group in packet['groups']:
        if int(group['rank'])>0:
            g=np.asarray(group['rows'],np.uint64); previously=g[previous_chosen[g]]
            if len(previously) and not np.all(chosen[previously]): chosen[previously]=False
    removed=np.flatnonzero(previous_chosen & ~chosen).astype('<u8'); new_ids=np.arange(n0,n,dtype='<u8')
    physical=np.r_[physical,np.ones(count,'u1')];training=np.r_[training,np.ones(count,'u1')]
    fit=np.r_[np.flatnonzero(chosen).astype('<u8'),new_ids]; fit_mask=np.zeros(n,bool);fit_mask[fit]=True
    inspection=np.flatnonzero(physical.astype(bool)&~fit_mask).astype('<u8')
    require(np.all(training[fit]) and np.all(training<=physical), 'current source domains')
    logical=hashlib.sha256(json.dumps(dict(schema='rinkfit.current_source_append_identity.v1',
        parent_logical_source_id=m['logical_source_id'],closed_source_sha256=file_sha256(closed_path),
        parent_raw_rows=n0,new_raw_rows=count,tail_source_ids=[r['source_row_id']for r in tail]),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    print('current source append: current withdrawals checked; writing new package',flush=True)
    out.mkdir();(out/'a').mkdir()
    def save_json(value,name):
        p=out/name;require(not p.exists(), 'output collision '+name)
        p.write_text(json.dumps(value,ensure_ascii=False,separators=(',',':'))+chr(10),'utf8')
        return dict(file=name,sha256=file_sha256(p),byte_count=p.stat().st_size)
    def save_array(a,dtype=None):
        a=np.ascontiguousarray(a,dtype=dtype);raw=a.tobytes();digest=hashlib.sha256(raw).hexdigest();name='a/'+digest+'.bin';p=out/name
        if not p.exists():p.write_bytes(raw)
        return dict(file=name,sha256=digest,dtype=a.dtype.str,shape=list(a.shape),layout='C',byte_order='little')
    def extend(ref,values,prefix=None):
        source=pinned(ref) if prefix is None else Path(prefix); dt=np.dtype(ref['dtype']); require(ref['shape'][0]==n0,'raw column prefix shape')
        new=np.ascontiguousarray(values,dtype=dt);require(list(new.shape)==[count,*ref['shape'][1:]],'raw tail column shape')
        tmp=out/'a'/('raw_'+str(len(list((out/'a').iterdir())))+'.tmp');h=hashlib.sha256()
        with source.open('rb')as old,tmp.open('wb')as dest:
            while True:
                b=old.read(4*1024*1024)
                if not b:break
                dest.write(b);h.update(b)
            raw=new.tobytes();dest.write(raw);h.update(raw)
        digest=h.hexdigest();name='a/'+digest+'.bin';tmp.replace(out/name)
        return dict(file=name,sha256=digest,dtype=dt.str,shape=[n,*ref['shape'][1:]],layout='C',byte_order='little')
    extra=dict(uv=[r['uv']for r in tail],query_uv=[r['seed']for r in tail],profile_normal=[r['normal']for r in tail],
        native_pts=[r['native_pts']for r in tail],shot_index=[packet['shot_ids'].index(r['shot_id'])for r in tail],kind=np.zeros(count),
        material_index=[r['material_index']for r in tail],source_material_index=[r['source_material_index']for r in tail],model_retired=np.zeros(count),
        path=[r['path']for r in tail],station=[r['station']for r in tail],row=[r['row']for r in tail],alpha=[r['alpha']for r in tail],
        alpha_bounds=[[r['offsets'][0],r['offsets'][-1]]for r in tail],fit_role=np.ones(count),admission=np.ones(count),
        # Finite ABI placeholders: rank-zero unknown-noise rows never consume
        # source_sigma as a statistical precision. E is the common existing
        # engineering pixel scale, not a measured noise standard deviation.
        source_sigma=np.ones(count),E=np.full(count,2.),stroke_pair_index=np.full(count,-1),stroke_source_side=np.full(count,-1))
    require(set(packet['arrays'])==set(extra),'unsupported current raw column extension')
    parent_arrays=deepcopy(packet['arrays'])
    for key,ref in list(packet['arrays'].items()):
        if key=='admission':
            a=np.array(col(ref),copy=True);a[withdraw]=0;packet['arrays'][key]=save_array(np.r_[a,np.ones(count,'u1')],'u1')
        else:
            prefix=pinned(cp['raw_auxiliary'][key],cb)if key in old_aux else None
            packet['arrays'][key]=extend(ref,extra[key],prefix)
    aux={k:packet['arrays'][k]for k in old_aux}
    ids=[r['source_row_id'].encode('utf8')for r in tail];offset=np.r_[0,np.cumsum([len(x)for x in ids])].astype('<u8')
    families=list(dict.fromkeys(r['source_family_id']for r in tail));fm={s:i for i,s in enumerate(families)}
    registry=dict(schema='rinkfit.dense_source_suffix_registry.v1',row_start=n0,count=count,
        source_dataset_id='direct_current_board_1024',source_id_offsets=save_array(offset),
        source_id_utf8=save_array(np.frombuffer(b''.join(ids),dtype='u1')),
        source_family_ids=families,source_family_index=save_array(np.array([fm[r['source_family_id']]for r in tail],'<u4')),
        source_store_ordinal=save_array(np.arange(count,dtype='<u8')),
        allowed_paths_by_material={str(mi):list(range(len(model['materials'][mi]['paths'])))for mi in sorted({r['material_index']for r in tail})})
    packet['native_ad_source_bindings'].append(registry);packet['native_ad_source_binding']=registry
    for family in families:
        indices=[n0+i for i,r in enumerate(tail)if r['source_family_id']==family]
        packet['groups'].append(dict(id=family,statistic_kind='constraint_only_unknown_noise',rows=indices,rank=0,W=[],current_unknown_noise_source_rows=True))
        packet['engineering_pixel_dependency_groups']['groups'].append(dict(id=family,rows=indices))
    for j,r in enumerate(tail):
        normal=np.asarray(r['normal']);seed=np.asarray(r['seed']);off=np.asarray(r['offsets'])
        packet['queries'].append(dict(row_index=n0+j,alpha=r['offsets'],native_uv=(seed+off[:,None]*normal).tolist(),
            normal=r['normal'],numerator_x=[r['seed'][0],r['normal'][0]],numerator_y=[r['seed'][1],r['normal'][1]],denominator=[1.],alpha_origin=0.,alpha_scale=1.))
    packet.update(row_count=n,admitted_rows=int(physical.sum()),active_rows=int(physical.sum()))
    packet['current_input_data_v1']['logical_source_id']=logical
    packet['current_withheld_rows']=sorted(set(packet['current_withheld_rows'])|set(map(int,withdraw)))
    q=m['qualification'];q.update(raw_rows=n,physical_count=int(physical.sum()),training_count=int(training.sum()),fit_count=len(fit),inspection_count=len(inspection),
        physical_mask=save_array(physical,'u1'),training_mask=save_array(training,'u1'),fit_rows=save_array(fit,'<u8'),inspection_rows=save_array(inspection,'<u8'))
    verdict=json.loads(pinned(q['source_identity_verdicts']).read_text('utf8'));byid={int(r['raw_row']):r for r in verdict['rows']}
    for i,r in decisions.items():byid[i]=dict(r,previous_membership=old_membership[i])
    for j,r in enumerate(tail):byid[n0+j]=dict(raw_row=n0+j,source_row_id=r['source_row_id'],native_pts=r['native_pts'],shot_index=packet['shot_ids'].index(r['shot_id']),
        material_index=r['material_index'],path=r['path'],uv=r['uv'],decision='confirmed',reason='current exact native full RGB material sequence and independently reviewed finite board body; source-noise covariance not issued',
        physical=True,training=True,fit=True,uses_fitted_model_projection=False,qualification_uses_model_residual=False,raw_UV_PTS_ID_preserved=True)
    verdict.update(logical_source_id=logical,rows=[byid[i]for i in sorted(byid)]);q['source_identity_verdicts']=save_json(verdict,'source_identity.json')
    # Actual current accepted files are copied, not reconstructed from history.
    shutil.copyfile(model_file,out/'source_model.json');model_ref=dict(file='source_model.json',sha256=file_sha256(model_file),byte_count=model_file.stat().st_size)
    try:os.link(checkpoint_file,out/'initial_checkpoint.json')
    except OSError:shutil.copyfile(checkpoint_file,out/'initial_checkpoint.json')
    checkpoint_ref=dict(file='initial_checkpoint.json',sha256=file_sha256(checkpoint_file),byte_count=checkpoint_file.stat().st_size)
    transition=dict(schema='rinkfit.current_source_append_transition.v1',new_objective_identity_required=True,
        from_state_token=cp['from_state_token'],parent_current_input_sha256=source_ref['sha256'],parent_logical_source_id=m['logical_source_id'],
        target_logical_source_id=logical,parent_raw_rows=n0,target_raw_rows=n,new_raw_rows=count,
        parent_raw_auxiliary_prefix_sha256={k:cp['raw_auxiliary'][k]['sha256']for k in old_aux},
        retained_parameter_values_exact=True,new_parameters_only_tail_raw_auxiliary=True,
        tail_geometry_primitive_kinds=[3],tail_source_noise_W_issued=False,reason='Explicit current source geometry append and source-identity withdrawals; complete old primal prefix retained')
    transport=dict(schema='rinkfit.current_complete_primal_transport.v1',initialization='explicit_current_input_rebind',raw_rows=n,
        from_state_token=cp['from_state_token'],model=model_ref,checkpoint=checkpoint_ref,raw_auxiliary=aux,
        preserve_complete_block_values=True,preserve_full_raw_auxiliary=True,preserve_motion_and_global_duals=True,new_objective_identity_required=True,
        current_source_append_transition_v1=transition)
    if len(removed):transport['current_qualification_transition_v1']=dict(schema='rinkfit.current_qualification_transition.v1',new_objective_identity_required=True,
        from_state_token=cp['from_state_token'],logical_source_id=logical,previous_fit_rows_sha256=previous_fit_sha,
        target_fit_rows_sha256=q['fit_rows']['sha256'],removed_fit_rows=list(map(int,removed)),reason='Current independently unconfirmed source entity withdrawals; original raw state retained')
    m.update(logical_source_id=logical,raw_rows=n,physical_rows=int(physical.sum()),training_rows=int(training.sum()),fit_rows=len(fit),inspection_rows=len(inspection),
        source_model=model_ref,initial_primal=transport,quality_accepted=False,source_image_identity_proven=False)
    m['engineering_support_budget_v2']['source_packet_sha256']=logical
    packet_ref=save_json(packet,'packet.json');m['packet']=packet_ref
    proof=save_json(dict(schema='rinkfit.current_source_append_values.v1',parent_raw_rows=n0,new_raw_rows=count,
        source_RGB_sha256=closed['native_RGB_sha256'],rows=tail,source_complete_profiles_retained=True,
        source_material_full_sequence_proofs=[r['current_full_material_sequence_evidence']for r in rows],
        geometric_initializers_are_not_identity_evidence=True,source_noise_covariance_issued=False,tail_source_sigma_placeholder=1.,tail_source_sigma_statistically_consumed=False,
        tail_engineering_E=2.,tail_engineering_E_is_calibrated_noise=False),'source_append.json')
    transition['current_tail_source_values']=proof
    # Retained current assets are linked only when referenced by the new package.
    checked_assets={}
    def current_assets(v):
        if isinstance(v,dict):
            if 'file'in v and 'sha256'in v:
                p=Path(v['file']);require(not p.is_absolute()and'..'not in p.parts,'external dependency in new current package')
                dest=out/p;expected=v['sha256'];key=str(dest)
                require(key not in checked_assets or checked_assets[key]==expected,'conflicting current asset identities')
                if key not in checked_assets:
                    if dest.exists(): require(file_sha256(dest)==expected,'existing current output asset hash differs: '+str(p))
                    else:
                        old=pinned(v);dest.parent.mkdir(parents=True,exist_ok=True)
                        try:os.link(old,dest)
                        except OSError:shutil.copyfile(old,dest)
                    checked_assets[key]=expected
            else:
                for x in v.values():current_assets(x)
        elif isinstance(v,list):
            for x in v:current_assets(x)
    current_assets(packet);current_assets(m)
    result=dict(schema='rinkfit.current_source_append_issuance.v1',parent_current_input=source_ref,previous_state_token=cp['from_state_token'],
        parent_raw_rows=n0,new_raw_rows=count,raw_rows=n,withdrawn_physical_rows=sum(v['physical']for v in old_membership.values()),removed_fit_rows=len(removed),
        fit_rows=len(fit),inspection_rows=len(inspection),source_UV_PTS_ID_Query_W_prefix_preserved=True,latest_complete_raw_auxiliary_prefix_preserved=True,
        complete_physical_blocks_motion_and_dual_checkpoint_copied_exactly=True,source_noise_covariance_issued=False,history_replayed=False,solver_started=False)
    m['provenance']=[*m.get('provenance',[]),'Explicit current source append from token '+cp['from_state_token']]
    manifest_ref=save_json(m,'manifest.json');result['manifest']=dict(manifest_ref,file=str(out/'manifest.json'))
    save_json(result,'append_issuance.json');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--closed-rows',required=True);p.add_argument('--withdrawal',action='append',default=[]);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(issue_current_source_append(a.current,a.checkpoint,a.closed_rows,a.out,a.withdrawal),ensure_ascii=False))

if __name__=='__main__':main()
