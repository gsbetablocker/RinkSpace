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
    groups=[{**g,'W':({**g['W'],'file':str(locate(g['W']))}if isinstance(g['W'],dict)else list(g['W']))}for g in p['groups']];start_groups=len(groups);bundle=out/'unnormalized_geometry_W.f64';rank=int(p['joint_statistics_rank'])
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

