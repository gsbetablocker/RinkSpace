"""Append source-geometry-qualified rows for an explicit geometric initializer.

No source-noise W is fabricated. Formal statistical eligibility stays false;
the resulting packet must be family/shot normalized and remains initialization.
"""
from pathlib import Path
from .packet_storage import write_packet
from collections import defaultdict
import json
import numpy as np
from .io_utils import read_jsonl,write_json,file_sha256

def append_geometry_sources(parent_packet,geometry_rows,raw_ledger,material_map,model_path,output,gate_manifest,*,write_provisional_W=True):
    src=Path(parent_packet).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('new immutable generation required')
    p=json.loads(src.read_text(encoding='utf8'));mapping=json.loads(Path(material_map).read_text());model=json.loads(Path(model_path).read_text())
    old_binding=p.get('material_binding',{});prefix=old_binding.get('materials',[])
    if not prefix:raise ValueError('parent immutable material identity binding required')
    actual=[dict(index=i,label=m['label'],paths=[dict(index=j,id=q['id'],physical_point=bool(q.get('physical_point',False)))for j,q in enumerate(m['paths'])])for i,m in enumerate(model['materials'][:len(prefix)])]
    if actual!=prefix:raise ValueError('new actual model changes inherited material topology')
    keep=('source_row_id','source_family_id','shot_id','native_pts','source_uv','current_uv','normal','joint_material_id','joint_material_index','point_correspondence','source_geometry_eligible','solver_eligible_for_statistical_objective','statistical_objective_eligible','numeric_statistics_ready','solver_eligible')
    def compact(r):return {k:r[k]for k in keep if k in r}
    maps=mapping['semantic_to_material'];eligible={r['source_row_id']:compact(r)for r in read_jsonl(geometry_rows)};raw=[compact(r)for r in read_jsonl(raw_ledger)];N0=p['row_count'];n=N0+len(raw)
    if len({r['source_row_id']for r in raw})!=len(raw):raise ValueError('duplicate immutable new source ids')
    if set(eligible)-{r['source_row_id']for r in raw}:raise ValueError('eligible geometry rows outside complete source ledger')
    checked={}
    def locate(ref):
        f=Path(ref['file']);f=f if f.is_absolute()else src.parent/f;k=(str(f),ref['sha256'])
        if k not in checked:
            if file_sha256(f)!=ref['sha256']:raise ValueError('changed parent source array or W')
            checked[k]=True
        return f.resolve()
    A={k:np.fromfile(locate(v),v['dtype']).reshape(v['shape'])for k,v in p['arrays'].items()}
    uv=[];active=[];material=[];normals=[];families=defaultdict(list);allowed=dict(p.get('allowed_paths',{}));newflags=[]
    for j,r in enumerate(raw):
        rid=r['source_row_id'];g=eligible.get(rid);ok=g is not None
        if ok:
            if g.get('source_geometry_eligible')is not True or g.get('statistical_objective_eligible')is not False or g.get('numeric_statistics_ready')is not False or g.get('solver_eligible')is not False:raise ValueError('geometry qualification must be separate from formal statistic eligibility')
            for key in ('source_uv','current_uv','native_pts','joint_material_id'):
                if r.get(key)!=g.get(key):raise ValueError('geometry source gate redirects actual source '+key)
            r={**r,**g}
        desc=maps.get(r.get('joint_material_id'),{});index=int(desc.get('material_index',0));q=r.get('current_uv')if r.get('current_uv')is not None else r['source_uv']
        if ok:
            if not desc.get('geometry_ready')or index>=len(model['materials']):raise ValueError('physical source has no actual model topology')
            if r.get('joint_material_index',index)!=index:raise ValueError('source gate material map conflict')
            if r.get('point_correspondence'):raise ValueError('this outer geometry source adapter does not promote entity corners')
            families[(r['shot_id'],int(r['native_pts']),r['source_family_id'])].append(N0+j);allowed[str(N0+j)]=list(range(len(model['materials'][index]['paths'])))
        uv.append(q);active.append(ok);material.append(index);normal=np.asarray(r.get('normal',[0.,0.]),float);normals.append(normal if normal.shape==(2,)else[0.,0.])
        newflags.append(dict(source_row_id=rid,source_geometry_eligible=ok,solver_eligible_for_statistical_objective=False,source_noise_W_supplied=False,source_uv=r['source_uv'],current_uv=r.get('current_uv'),joint_material_index=index))
    out.mkdir(parents=True);uv=np.asarray(uv,float);active=np.asarray(active,np.uint8);N=len(raw)
    extra=dict(uv=uv,native_pts=np.asarray([r['native_pts']for r in raw]),shot_index=np.asarray([int(r['shot_id'][1:])-1 for r in raw]),kind=np.zeros(N),material_index=np.asarray(material),path=np.zeros(N),admission=active,fit_role=np.ones(N),E=np.full(N,2.),station=np.full(N,.5),row=np.where((uv[:,1]>=0)&(uv[:,1]<=1079),uv[:,1],539.5),alpha=np.zeros(N),source_sigma=np.ones(N),profile_normal=np.asarray(normals),query_uv=uv,alpha_bounds=np.zeros((N,2)),stroke_pair_index=np.full(N,-1),stroke_source_side=np.full(N,-1))
    arrays={}
    for key,a in A.items():
        if key not in extra:raise ValueError('new source adapter lacks required parent array '+key)
        b=np.concatenate((a,np.asarray(extra[key],dtype=a.dtype)));f=out/(key+'.bin');b.tofile(f);arrays[key]=dict(file=str(f),sha256=file_sha256(f),dtype=b.dtype.str,shape=list(b.shape),layout='C',byte_order='little')
    groups=[{**g,'W':{**g['W'],'file':str(locate(g['W']))}}for g in p['groups']];start_groups=len(groups);bundle=out/'unnormalized_geometry_W.f64';rank=int(p['joint_statistics_rank'])
    with bundle.open('wb')as stream:
        for(shot,native,family),rows in (sorted(families.items())if write_provisional_W else []):
            m=len(rows);W=np.eye(2*m,dtype='<f8');offset=stream.tell();W.tofile(stream)
            groups.append(dict(id='source_geometry:'+family,source_group_id=family,rows=rows,rank=2*m,W=dict(file=str(bundle),sha256='pending_closed_bundle',dtype='<f8',shape=list(W.shape),offset_bytes=offset),statistic_kind='initialization_surrogate',original_numerical_scheme='actual_source_geometry_no_noise_W_supplied',source_noise_covariance_claimed=False,requires_family_shot_normalization=True));rank+=2*m
    digest=file_sha256(bundle)
    for g in groups[start_groups:]:g['W']['sha256']=digest
    original_start=p['new_source_start'];oldactive=(A['admission']==1)&(A['fit_role']==1);packet={**p,'row_count':n,'arrays':arrays,'groups':groups,'source_dataset_ids':p['source_dataset_ids']+['actual_geometry:'+Path(gate_manifest).resolve().parent.name+':'+r['shot_id']for r in raw],'source_row_ids':p['source_row_ids']+[r['source_row_id']for r in raw],'active_rows':int(oldactive.sum()+active.sum()),'joint_statistics_rank':rank,'new_source_statistics_rank':rank-p['current_original_W_rank'],'newly_appended_source_start':N0,'new_source_start':original_start,'allowed_paths':allowed,'recapture_rows':np.flatnonzero(np.r_[oldactive,active.astype(bool)]&(np.arange(n)>=original_start)).tolist(),'source_manifests':p['source_manifests']+[dict(file=str(Path(x).resolve()),sha256=file_sha256(x))for x in (geometry_rows,raw_ledger,material_map,gate_manifest)],'initialization_objective_only':True,'physical_noise_likelihood_claimed':False,'formal_new_source_statistical_optimum_claimed':False,'new_source_statistics_status':'explicit actual-source family geometry initializer; source-noise numeric eligibility is false','outer_source_geometry_gate':dict(file=str(Path(gate_manifest).resolve()),sha256=file_sha256(gate_manifest)),'source_noise_statistics_supplied_for_appended_rows':False,'source_midpoint_or_material_physical_definition_substituted':False,'parent_actual_source_packet':dict(file=str(src),sha256=file_sha256(src))}
    if 'source_family_ids'in p:packet['source_family_ids']=p['source_family_ids']+[r['source_family_id']for r in raw]
    write_packet(out/'packet.json',packet);write_json(out/'new_geometry_membership.json',newflags)
    report=dict(schema='rinkfit.actual_source_geometry_only_append.v1',parent_rows=N0,appended_raw_rows=N,appended_active_geometry_rows=int(active.sum()),raw_rows=n,active_rows=packet['active_rows'],actual_source_families=len(families),original_s01_W_or_prefix_state_changed=False,source_noise_W_or_numeric_statistics_ready_fabricated=False,source_uv_and_current_mean_distinct=True,requires_family_shot_normalization=True,provisional_geometry_W_written=write_provisional_W,packet_sha256=file_sha256(out/'packet.json'))
    write_json(out/'report.json',report);return report


def bind_existing_geometry_sources(parent_packet,qualified_rows,source_manifest,model_path,material_map,output):
    """Replace all new-source consumption by an explicit source-proved subset.

    Raw order/history remain complete. This can retire a recalled source scope
    and activate source-authoritative manual geometry without inventing noise.
    """
    from collections import Counter
    src=Path(parent_packet).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('new immutable source-qualified geometry generation required')
    p=json.loads(src.read_text(encoding='utf8'));model=json.loads(Path(model_path).read_text(encoding='utf8'));start=p['new_source_start'];N=p['row_count'];checked={}
    def locate(v):
        f=Path(v['file']);f=f if f.is_absolute()else src.parent/f;k=(str(f),v['sha256'])
        if k not in checked:
            if file_sha256(f)!=v['sha256']:raise ValueError('inherited numeric dependency changed')
            checked[k]=True
        return f.resolve()
    A={k:np.fromfile(locate(v),v['dtype']).reshape(v['shape'])for k,v in p['arrays'].items()};before={k:a[:start].tobytes()for k,a in A.items()};role=A['fit_role'];ad=A['admission'];ad[start:]=0;role[start:]=1
    byid={rid:i for i,rid in enumerate(p['source_row_ids'][start:],start)};rows=list(read_jsonl(qualified_rows));families=defaultdict(list);decisions=[];allowed={};family_ids=list(p.get('source_family_ids',[None]*N))
    if len({r['source_row_id']for r in rows})!=len(rows):raise ValueError('qualified subset repeats immutable source row')
    for r in rows:
        rid=r['source_row_id'];i=byid.get(rid)
        if i is None or r.get('source_geometry_eligible')is not True or r.get('statistical_objective_eligible')is not False or r.get('numeric_statistics_ready')is not False:raise ValueError('source-qualified geometry must be separate from statistics eligibility')
        if int(A['native_pts'][i])!=int(r['native_pts'])or p['shot_ids'][int(A['shot_index'][i])]!=r['shot_id']:raise ValueError('qualified source clock differs from immutable row')
        q=np.asarray(r['current_uv'],float)
        if r.get('original_source_is_actual_manual_click'):
            if not np.array_equal(q,np.asarray(r['source_uv']))or not np.array_equal(q,A['uv'][i]):raise ValueError('manual authoritative point would be moved')
        m=int(r['joint_material_index']);corner=r.get('kind')=='entity_corner'or bool(r.get('point_correspondence'))
        if m<0 or m>=len(model['materials']):raise ValueError('qualified source material lacks actual geometry')
        paths=model['materials'][m]['paths']
        if corner and not paths[0].get('physical_point'):raise ValueError('manual entity corner lacks a fixed physical point')
        changed=not np.array_equal(q,A['uv'][i]);A['uv'][i]=q;A['query_uv'][i]=q;A['material_index'][i]=m;A['kind'][i]=1 if corner else 0;A['path'][i]=0;A['station'][i]=0. if corner else .5;A['row'][i]=q[1] if 0<=q[1]<=1079 else 539.5;A['alpha'][i]=0.;A['alpha_bounds'][i]=[0.,0.];ad[i]=1;family_ids[i]=r['source_family_id'];families[(r['shot_id'],int(r['native_pts']),r['source_family_id'])].append(i);allowed[str(i)]=list(range(len(paths)))
        decisions.append(dict(joint_row=i,source_row_id=rid,current_uv=q.tolist(),source_uv=r['source_uv'],current_mean_changed_from_parent=changed,joint_material_index=m,source_receipt_sha256=r['receipt_sha256'],source_geometry_eligible=True,statistical_objective_eligible=False,source_noise_W_claimed=False))
    for k,a in A.items():
        if a[:start].tobytes()!=before[k]:raise ValueError('source-qualified subset modifies original S01 numeric prefix')
    oldbinding=p['material_binding']['materials'];actual=[dict(index=i,label=m['label'],paths=[dict(index=j,id=q['id'],physical_point=bool(q.get('physical_point',False)))for j,q in enumerate(m['paths'])])for i,m in enumerate(model['materials'][:len(oldbinding)])]
    if oldbinding!=actual:raise ValueError('qualified actual model changes inherited material identity topology')
    out.mkdir(parents=True);arrays={}
    for k,a in A.items():
        f=out/(k+'.bin');a.tofile(f);arrays[k]={**p['arrays'][k],'file':str(f),'sha256':file_sha256(f)}
    original=[{**g,'W':{**g['W'],'file':str(locate(g['W']))}}for g in p['groups']if all(i<start for i in g['rows'])];groups=list(original);counts=Counter(k[0]for k in families);bundle=out/'family_geometry_W.f64';rank=p['current_original_W_rank'];pending=[]
    with bundle.open('wb')as f:
        for(shot,native,family),indices in sorted(families.items()):
            m=len(indices);scale=float(np.sqrt(1./(m*counts[shot])));W=np.eye(2*m,dtype='<f8')*scale;off=f.tell();W.tofile(f);spec=dict(file=str(bundle),sha256='pending_closed_bundle',dtype='<f8',shape=list(W.shape),offset_bytes=off);pending.append(spec);groups.append(dict(id='qualified_geometry:'+family,source_group_id=family,rows=indices,rank=2*m,W=spec,statistic_kind='initialization_surrogate',original_numerical_scheme='fixed_actual_source_family_and_shot_native_geometry',geometry_weight=dict(scale=scale,rows_in_family=m,families_in_shot=counts[shot],shot_total_weight=1.),source_noise_covariance_claimed=False));rank+=2*m
    digest=file_sha256(bundle)
    for v in pending:v['sha256']=digest
    active=(ad==1)&(role==1);partition=np.zeros(N,np.int16);partition[np.asarray(p['independent_rows'],int)]+=1
    for g in groups:partition[g['rows']]+=1
    if not np.array_equal(partition,active.astype(np.int16)):raise ValueError('new source-proved subset W partition does not match active membership')
    refs=[dict(file=str(Path(x).resolve()),sha256=file_sha256(x))for x in (qualified_rows,source_manifest,model_path,material_map)];result={**p,'arrays':arrays,'groups':groups,'source_family_ids':family_ids,'active_rows':int(active.sum()),'joint_statistics_rank':rank,'new_source_statistics_rank':rank-p['current_original_W_rank'],'allowed_paths':allowed,'recapture_rows':sorted(d['joint_row']for d in decisions),'queries':[q for q in p.get('queries',[])if q['row_index']<start],'source_manifests':p['source_manifests']+refs,'current_source_qualification_refs':refs,'new_source_statistics_status':'source-proved geometry-only subset; noise is not assigned','initialization_objective_only':True,'physical_noise_likelihood_claimed':False,'source_noise_statistics_supplied_for_appended_rows':False,'formal_new_source_statistical_optimum_claimed':False,'physical_source_audit_complete':False,'source_qualified_subset_parent':dict(file=str(src),sha256=file_sha256(src)),'all_other_new_raw_sources_retained_but_inactive':True,'current_recalled_outer_active_rows':0}
    if 'outer_source_geometry_gate'in result:result['historical_recalled_outer_geometry_gate']=result.pop('outer_source_geometry_gate')
    write_packet(out/'packet.json',result);write_json(out/'qualified_source_membership.json',decisions);report=dict(schema='rinkfit.current_source_qualified_geometry_subset.v1',raw_rows=N,active_rows=int(active.sum()),new_active_rows=len(rows),new_actual_source_families=len(families),new_shots=sorted(counts),current_S01_rank=p['current_original_W_rank'],joint_rank=rank,raw_history_undeleted=True,all_other_new_sources_inactive=True,source_noise_precision_invented=False,original_S01_prefix_arrays_and_W_unchanged=True,packet_sha256=file_sha256(out/'packet.json'))
    write_json(out/'report.json',report);return report


def activate_additional_geometry_sources(parent_packet,qualified_rows,source_manifest,output):
    """Activate an already retained, newly source-proved raw geometry subset.

    This preserves every current source and all UV/clock/auxiliary bytes. A
    following family normalization supplies only explicit geometric weights.
    """
    src=Path(parent_packet).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('new immutable additional-source generation required')
    p=json.loads(src.read_text(encoding='utf8'));checked={}
    def resolve(ref):
        f=Path(ref['file']);f=f if f.is_absolute()else src.parent/f;k=(str(f),ref['sha256'])
        if k not in checked:
            if file_sha256(f)!=ref['sha256']:raise ValueError('changed inherited source numeric dependency')
            checked[k]=True
        return f.resolve()
    A={k:np.fromfile(resolve(v),v['dtype']).reshape(v['shape'])for k,v in p['arrays'].items()};before={k:a.tobytes()for k,a in A.items()};table=p['material_binding']['materials'];allowed=dict(p.get('allowed_paths',{}));changed=[]
    for r in read_jsonl(qualified_rows):
        if r.get('source_geometry_eligible')is not True or r.get('numeric_statistics_ready')is not False or r.get('statistical_objective_eligible')is not False:raise ValueError('unknown-noise geometry source qualification required')
        matches=[i for i,rid in enumerate(p['source_row_ids'])if rid==r['source_row_id']and int(A['native_pts'][i])==int(r['native_pts'])]
        if len(matches)!=1:raise ValueError('additional existing source identity must be unique')
        i=matches[0]
        if i<p['new_source_start']:raise ValueError('no inherited S01 source change authorized')
        if not np.array_equal(A['uv'][i],np.asarray(r['current_uv'],float))or r['source_uv']!=r['current_uv']:raise ValueError('actual authoritative manual UV changed')
        if A['admission'][i]==1:continue
        mi=int(r['joint_material_index'])
        if table[mi]['label']!=r['target_id']:raise ValueError('actual source identity differs from frozen model material')
        if r['kind']!='finite_material_curve' or r.get('entity_station_fixed'):raise ValueError('this additional-source adapter only supports a finite curve')
        A['admission'][i]=1;A['fit_role'][i]=1;A['kind'][i]=0;A['material_index'][i]=mi;A['path'][i]=0;allowed[str(i)]=list(range(len(table[mi]['paths'])));changed.append(dict(joint_row=i,source_row_id=r['source_row_id'],native_pts=r['native_pts'],original_UV=r['source_uv'],source_geometry_eligible=True,source_noise_unknown=True))
    if not changed:raise ValueError('no newly source-proved geometry rows to activate')
    out.mkdir(parents=True);arrays={}
    for k,a in A.items():
        if a.tobytes()==before[k]:arrays[k]={**p['arrays'][k],'file':str(resolve(p['arrays'][k]))}
        else:
            if k not in ('admission','fit_role','kind','material_index','path'):raise ValueError('unexpected source/auxiliary mutation')
            f=out/(k+'.bin');a.tofile(f);arrays[k]={**p['arrays'][k],'file':str(f),'sha256':file_sha256(f)}
    refs=[dict(file=str(Path(x).resolve()),sha256=file_sha256(x))for x in (qualified_rows,source_manifest)];active=(A['admission']==1)&(A['fit_role']==1);result={**p,'arrays':arrays,'groups':[{**g,'W':{**g['W'],'file':str(resolve(g['W']))}}for g in p['groups']],'allowed_paths':allowed,'active_rows':int(active.sum()),'recapture_rows':np.flatnonzero(active&(np.arange(len(active))>=p['new_source_start'])).tolist(),'source_manifests':p['source_manifests']+refs,'current_source_qualification_refs':p.get('current_source_qualification_refs',[])+refs,'additional_geometry_source_parent':dict(file=str(src),sha256=file_sha256(src)),'requires_family_shot_renormalization_before_solver':True}
    write_packet(out/'packet.json',result);write_json(out/'additional_source_rows.json',changed);report=dict(raw_rows=p['row_count'],newly_activated_geometry_rows=len(changed),previously_active_source_rows_unchanged=True,source_UV_clock_and_all_four_aux_bytes_changed=False,original_S01_W_changed=False,automatic_noise_W_assigned=False,packet_sha256=file_sha256(out/'packet.json'));write_json(out/'report.json',report);return report
