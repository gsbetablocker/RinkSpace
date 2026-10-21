"""Same-packet actual complete-CP source audit for the first four native shots.

Uses exported stored auxiliaries, validity flags and independent fixed-station
RS replay. It never deletes failures or treats profiled exports as the CP.
"""
from pathlib import Path
from collections import Counter
import json,math
import numpy as np
from .io_utils import file_sha256
from .r_inner_prefix_v1 import array
from .accepted_aux_prefix import verify_engineering_checkpoint,_quoted_fields,AUX
from .source_quality_compare_v1 import _windows

FLAGS=('valid','arithmetic_valid','ray_domain_valid','station_domain_valid','visibility_valid','width_domain_valid')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def ref(p):return dict(file=str(Path(p).resolve()),sha256=file_sha256(p))
def metric(e,good):
    e=np.asarray(e,float);good=np.asarray(good,bool)&np.isfinite(e);z=e[good];n=len(e)
    return dict(denominator=n,valid=int(good.sum()),failed=int(n-good.sum()),
        full_RMS_px=float(np.sqrt(np.mean(z*z)))if n and good.all()else None,
        available_RMS_px=float(np.sqrt(np.mean(z*z)))if len(z)else None,
        available_max_px=float(z.max())if len(z)else None,
        available_P95_px=float(np.quantile(z,.95))if len(z)else None)

def compare(packet_path,before_cp,before_rs,after_cp,after_rs,output):
    path=Path(packet_path).resolve();packet=read(path);digest=file_sha256(path);n=packet['row_count'];seen=set()
    if packet['shot_ids']!=['s01','s02','s03','s04']:raise ValueError('Explicit first4 packet required')
    A={k:array(packet,k,path.parent,seen)for k in packet['arrays']}
    active=(A['admission']==1)&(A['fit_role']==1); fixed=A['kind']==1
    qmask=np.zeros(n,bool);qmask[[q['row_index']for q in packet['queries']]]=True
    states={};refs=[ref(path)]
    def load(root,manifest,name,dtype,tail=()):
        f=root/name;d=manifest['files'][name]
        if file_sha256(f)!=d['sha256']or f.stat().st_size!=n*np.dtype(dtype).itemsize*math.prod(tail or(1,)):
            raise ValueError('Changed or truncated actual state export '+str(f))
        refs.append(ref(f));return np.memmap(f,mode='r',dtype=dtype,shape=(n,)+tail)
    for key,cp,rs in [('before',before_cp,before_rs),('after',after_cp,after_rs)]:
        cp=Path(cp);rs=Path(rs);cm=read(cp/'CP_explicit_metrics/manifest.json');am=read(cp/'explicit_nuisance/manifest.json');rm=read(rs/'value_report.json')
        if cm['packet_sha256']!=digest or am['packet_sha256']!=digest or cm['raw_rows']!=n or am['raw_rows']!=n:
            raise ValueError('State source generation does not match exact comparison packet')
        model=read(cp/'model.json');cpfile=cp/'engineering_checkpoint.json'
        with cpfile.open(encoding='utf8')as cf: head=[cf.readline().rstrip('\r\n')for _ in range(8)]
        sourcefile=cp/'source_math.json'
        source=read(sourcefile)if sourcefile.exists()else None
        token=_quoted_fields(head[6])[0]
        if cm['state_token']!=token or (source is not None and token!=source['state_token'])or file_sha256(cp/'model.json')!=rm['model_sha256']or file_sha256(rs/'model.json')!=rm['model_sha256']or not rm['exact_checkpoint_model_blocks']:
            raise ValueError('Independent replay is not the actual complete CP model')
        refs.extend(ref(x)for x in [cp/'CP_explicit_metrics/manifest.json',cp/'explicit_nuisance/manifest.json',rs/'value_report.json',cpfile,cp/'model.json'])
        if source is not None: refs.append(ref(sourcefile))
        else:
            aux={k:load(cp/'explicit_nuisance',am,name,dtype)for k,(name,dtype)in AUX.items()}
            verify_engineering_checkpoint(cpfile,dict(configuration_sha256=_quoted_fields(head[1])[0],candidate_state_token=token,source_packet_sha256=digest),model,aux,active,[q['row_index']for q in packet['queries']],[int(i)for i in np.flatnonzero(active)if A['kind'][i]!=1])
        flags={k:load(cp/'CP_explicit_metrics',cm,'explicit_'+k+'.u8','u1')==1 for k in FLAGS}
        error=load(rs,rm,'raw_station_RS_error_native_uv.f64','<f8',(2,));rawvalid=load(rs,rm,'raw_station_RS_valid.u8','u1')==1
        raw=np.linalg.norm(error,axis=1);operator=np.where(qmask,np.abs(np.sum(error*A['profile_normal'],axis=1)),raw)
        rs_error=np.abs(load(cp/'CP_explicit_metrics',cm,'explicit_RS_equality.f64','<f8'))
        query_error=np.linalg.norm(load(cp/'CP_explicit_metrics',cm,'explicit_query_equality.f64','<f8',(2,)),axis=1)
        states[key]=dict(raw=raw,operator=operator,rawvalid=rawvalid,fullvalid=rawvalid&flags['valid'],flags=flags,
            RS_equality=rs_error,Query_equality=query_error,model=model,model_ref=ref(cp/'model.json'),token=cm['state_token'])
    def cohort(sel):
        out={}
        for key,st in states.items():
            out[key]=dict(RS_fixed_current_station_UV=metric(st['raw'][sel],st['fullvalid'][sel]),
                measurement_operator=metric(st['operator'][sel],st['fullvalid'][sel]),
                fixed_entities_2D_UV=metric(st['raw'][sel&fixed],st['fullvalid'][sel&fixed]),
                explicit_RS_equality=metric(st['RS_equality'][sel],np.isfinite(st['RS_equality'][sel])),
                explicit_Query_equality=metric(st['Query_equality'][sel&qmask],np.isfinite(st['Query_equality'][sel&qmask])),
                failure_counts={k:int((sel&~v).sum())for k,v in st['flags'].items()},
                raw_RS_failed=int((sel&~st['rawvalid']).sum()))
        return out
    shots={}
    for si,sid in enumerate(packet['shot_ids']):
        sel=active&(A['shot_index']==si);row=cohort(sel)
        start,end=[(256256,575488),(575744,720640),(720896,745728),(745984,948736)][si]
        pts=np.arange(start,end+256,256,dtype=np.int64)
        for key,st in states.items():
            sse=np.zeros(len(pts));den=np.zeros(len(pts),int);bad=np.zeros(len(pts),int)
            at=np.searchsorted(pts,A['native_pts'][sel]);e=st['operator'][sel];good=st['fullvalid'][sel]&np.isfinite(e)
            if np.any(at>=len(pts))or not np.array_equal(pts[at],A['native_pts'][sel]):raise ValueError('Non-native or wrong-shot source clock')
            np.add.at(den,at,1);np.add.at(bad,at,~good);np.add.at(sse,at,np.where(good,e*e,0.))
            row[key]['source_exposures']=int((den>0).sum());row[key]['temporal']=_windows(pts,sse,den,bad,den>0)
        shots[sid]=row
    entities={};families=np.asarray(packet['source_family_ids'],object)
    mi=A['material_index']
    for index in np.unique(mi[active]):
        label=states['after']['model']['materials'][int(index)]['label'];sel=active&(mi==index)
        entities[label]=dict(**cohort(sel),source_families=len(set(families[sel])))
    report=dict(schema='rinkfit.first4_actual_CP_same_source_comparison.v1',packet=ref(path),
        states={k:dict(model=v['model_ref'],complete_state_token=v['token'])for k,v in states.items()},
        active_rows=int(active.sum()),retained_raw_rows=n,query_rows=int((active&qmask).sum()),fixed_entity_rows=int((active&fixed).sum()),
        by_shot=shots,by_entity=entities,all=cohort(active),
        original_UV_and_failure_denominators_preserved=True,source_families_are_not_independent_samples=True,
        engineering_E_is_not_formal_W=True,source_noise_statistics_invented=False,quality_accepted=False,
        comparison_semantics='Independent local RS at each complete-CP current station; fixed entities use 2D UV. Curve Query normal residual and explicit closure are separate. No nuisance reprofile is mislabeled as stored CP.',
        exact_model_references=refs,producer=ref(__file__))
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('Immutable audit output exists')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return report
