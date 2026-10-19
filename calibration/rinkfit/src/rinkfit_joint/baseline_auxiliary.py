"""Exact d78 S01 native auxiliary import; no recapture or source mutation."""
from __future__ import annotations
import argparse,codecs,hashlib,json,mmap
from pathlib import Path
import numpy as np
from .io_utils import file_sha256,write_json
from .baseline_recovery import verify_baseline_complete_state,verify_zero_optimization_import
from .joint_model import as_event_view
from rinkfit_start.native_scene import NativeScene,camera_from_event


def packet_header(path):
    with Path(path).open('rb') as f:head=f.read(65536).decode('utf8')
    dec=json.JSONDecoder();result={}
    for key in ('row_count','shot_ids','arrays'):
        marker='"'+key+'":';i=head.index(marker)+len(marker);result[key]=dec.raw_decode(head[i:].lstrip())[0]
    return result


def array_items(path,key):
    """Bounded-memory iteration of one immutable top-level JSON array."""
    marker=('"'+key+'":').encode()
    with Path(path).open('rb') as f:
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
            pos=mm.find(marker)
            if pos<0:raise ValueError('packet field missing: '+key)
            pos+=len(marker)
        f.seek(pos);decoder=json.JSONDecoder();text='';position=0;utf=codecs.getincrementaldecoder('utf8')();started=False;ended=False
        while not ended:
            if position>65536:text=text[position:];position=0
            while position<len(text) and text[position].isspace():position+=1
            if not started:
                if position<len(text):
                    if text[position]!='[':raise ValueError('JSON field is not an array')
                    position+=1;started=True
            if started:
                while position<len(text) and (text[position].isspace() or text[position]==','):position+=1
                if position<len(text) and text[position]==']':return
                try:
                    value,end=decoder.raw_decode(text,position);position=end;yield value;continue
                except json.JSONDecodeError:pass
            chunk=f.read(65536)
            if not chunk:raise ValueError('truncated immutable JSON array')
            text+=utf.decode(chunk)


def checked_array(desc,base):
    p=Path(desc.get('file',desc.get('path')));p=p if p.is_absolute() else Path(base)/p
    if file_sha256(p)!=desc['sha256']:raise ValueError('changed native array: '+str(p))
    return np.memmap(p,mode='r',dtype=np.dtype(desc['dtype']),shape=tuple(desc['shape']),offset=desc.get('offset_bytes',0))



def inherited_semantic_array(manifest_path,key):
    """Resolve an inherited relative descriptor through SHA-bound parents."""
    visited=set()
    def resolve(path,expected=None):
        path=Path(path)
        if str(path) in visited:return None
        visited.add(str(path));doc=json.loads(path.read_text(encoding='utf8'));desc=doc.get(key)
        if desc and (expected is None or desc['sha256']==expected):
            p=Path(desc.get('file',desc.get('path')));p=p if p.is_absolute() else path.parent/p
            if p.is_file():return checked_array(desc,path.parent)
        for name,ref in reversed(list(doc.items())):
            if 'parent_manifest' not in name or not isinstance(ref,dict) or not ref.get('sha256'):continue
            pp=Path(ref.get('path',ref.get('file','')));pp=pp if pp.is_absolute() else path.parent/pp
            if not pp.is_file() or file_sha256(pp)!=ref['sha256']:continue
            result=resolve(pp,expected if expected is not None else desc['sha256'] if desc else None)
            if result is not None:return result
        return None
    result=resolve(manifest_path)
    if result is None:raise ValueError('inherited semantic array unresolved: '+key)
    return result

def build_d78_auxiliary_import(packet_file,model_file,complete_state_file,output):
    out=Path(output)
    if out.exists():raise ValueError('immutable auxiliary import generation exists')
    packet_file=Path(packet_file);complete_state_file=Path(complete_state_file);bound_packet_sha=file_sha256(packet_file);bound_model_sha=file_sha256(model_file);head=packet_header(packet_file)
    manifest=json.loads(complete_state_file.read_text(encoding='utf8'));deps=verify_baseline_complete_state(complete_state_file)
    baseline=json.loads(Path(manifest['model']['path']).read_text(encoding='utf8'));model=json.loads(Path(model_file).read_text(encoding='utf8'))
    equivalence=verify_zero_optimization_import(baseline,model)
    oldpacket=json.loads(Path(manifest['packet']['path']).read_text(encoding='utf8'));semantic_path=Path(manifest['semantic']['path']);semantic=json.loads(semantic_path.read_text(encoding='utf8'));N=manifest['source_rows']
    tables=json.loads(Path(manifest['raw_source_arrays_verified_exact']['string_tables']['path']).read_text(encoding='utf8'))
    def raw(name,dtype,shape):
        ref=manifest['raw_source_arrays_verified_exact'][name];return np.memmap(ref['path'],mode='r',dtype=dtype,shape=shape)
    olduv=raw('native_uv','<f8',(N,2));oldpts=raw('native_pts','<i8',(N,));pid=raw('point_id_index','<u4',(N,));oldids=np.asarray(tables['source_point_ids'],object)[pid]
    oldkind=checked_array(semantic['arrays']['kind'],semantic_path.parent);oldmat=inherited_semantic_array(semantic_path,'resolved_material_index')
    oldnormal=checked_array(semantic['arrays']['profile_normal'],semantic_path.parent);oldqueryuv=checked_array(semantic['arrays']['query_uv'],semantic_path.parent)
    oldactive=checked_array(semantic['active_source_mask'],semantic_path.parent).astype(bool)
    arrays={name:checked_array(head['arrays'][name],packet_file.parent) for name in ('uv','native_pts','shot_index','material_index','kind','admission','fit_role','profile_normal','query_uv','stroke_pair_index')}
    if np.any(arrays['shot_index'][:N]!=head['shot_ids'].index('s01')):raise ValueError('S01 original prefix allocation changed; explicit arbitrary-ID map adapter required')
    current_ids=[]
    for i,value in enumerate(array_items(packet_file,'source_row_ids')):
        if i>=N:break
        current_ids.append(value)
    datasets=[]
    for i,value in enumerate(array_items(packet_file,'source_dataset_ids')):
        if i>=N:break
        datasets.append(value)
    if len(current_ids)!=N or len(datasets)!=N or set(datasets)!={'s01_original'}:raise ValueError('original S01 dataset/ID prefix changed')
    lookup={}
    for i,key in enumerate(zip(oldpts.tolist(),oldids.tolist())):
        if key in lookup:raise ValueError('d78 original native source key duplicate')
        lookup[key]=i
    mapping=np.asarray([lookup.get((int(pt),sid),-1) for pt,sid in zip(arrays['native_pts'][:N],current_ids)],dtype=np.int64)
    if np.any(mapping<0) or len(np.unique(mapping))!=N:raise ValueError('current original prefix does not bijectively match d78 immutable source IDs/native PTS')
    active=arrays['admission'][:N].astype(bool)&(arrays['fit_role'][:N]==1);rows=np.flatnonzero(active)
    if np.any(~oldactive[mapping[rows]]):raise ValueError('auxiliary import would resurrect an originally excluded d78 source')
    if not np.array_equal(arrays['uv'][:N],olduv[mapping]):raise ValueError('same-ID original UV changed')
    if not np.array_equal(arrays['kind'][:N],oldkind[mapping]):raise ValueError('same-ID original measurement kind changed')
    labels0=np.asarray([m['label'] for m in baseline['materials']]);labels1=np.asarray([m['label'] for m in model['materials']])
    if not np.array_equal(labels1[arrays['material_index'][:N]],labels0[oldmat[mapping]]):raise ValueError('same-ID original physical material identity changed')
    if not np.array_equal(arrays['profile_normal'][:N],oldnormal[mapping]) or not np.array_equal(arrays['query_uv'][:N],oldqueryuv[mapping]):raise ValueError('original frozen Query normals/origins changed')
    if np.any(arrays['stroke_pair_index'][:N]>=0):raise ValueError('original S01 operator was replaced with actual-flank width rows; d78 auxiliary transfer is not equivalent')
    query_ref=oldpacket['query_packet'];query_path=Path(query_ref['manifest'])
    if file_sha256(query_path)!=query_ref['sha256']:raise ValueError('d78 frozen Query manifest changed')
    qmeta=json.loads(query_path.read_text(encoding='utf8'));qa={k:checked_array(d,query_path.parent) for k,d in qmeta['arrays'].items()};qlookup={int(r):i for i,r in enumerate(qa['source_rows'])};seen=set()
    for q in array_items(packet_file,'queries'):
        ci=int(q['row_index'])
        if ci>=N:break # production registry appends the unchanged original S01 Query section first
        oi=int(mapping[ci]);j=qlookup.get(oi)
        if j is None or j in seen:raise ValueError('new or duplicated Query inside original S01 prefix')
        seen.add(j);lo,hi=map(int,qa['offsets'][j:j+2])
        for key,target in (('alpha',qa['alpha'][lo:hi]),('native_uv',qa['uv'][lo:hi]),('normal',oldnormal[oi]),('numerator_x',qa['numerator_u'][j]),('numerator_y',qa['numerator_v'][j]),('denominator',qa['denominator'][j])):
            if not np.array_equal(np.asarray(q[key]),target):raise ValueError('frozen original finite Query changed: '+key)
    if len(seen)!=len(qlookup):raise ValueError('original finite Query coverage incomplete')
    def aux(name,dtype):
        ref=manifest['row_state_files'][name];return np.fromfile(ref['path'],dtype=dtype)
    oldstation=aux('station.f64','<f8');oldpath=aux('path.u32','<u4');oldrow=aux('initial_row.f64','<f8');oldalpha=aux('initial_alpha.f64','<f8')
    # Source-specific path identity, not an arbitrary nearest-branch search.
    for ci in rows:
        oi=int(mapping[ci]);mi0=int(oldmat[oi]);mi1=int(arrays['material_index'][ci]);pi=int(oldpath[oi]);before=baseline['materials'][mi0]['paths'][pi]
        if pi>=len(model['materials'][mi1]['paths']) or model['materials'][mi1]['paths'][pi]['id']!=before['id']:raise ValueError('d78 finite source path topology/identity changed')
        after=model['materials'][mi1]['paths'][pi];value=float(oldstation[oi])
        if not any(float(lo)-1e-12<=value<=float(hi)+1e-12 for lo,hi in after.get('allowed_station_intervals',[[0.,1.]])):raise ValueError('d78 station outside current source-qualified finite material path')
    scene=NativeScene(as_event_view(model,'s01'));camera=camera_from_event(as_event_view(model,'s01'));mm=arrays['material_index'][rows];pp=oldpath[mapping[rows]];ss=oldstation[mapping[rows]];rr=oldrow[mapping[rows]]
    world,legal,reasons=scene.world_rows(mm,pp,ss);tt=(arrays['native_pts'][rows]-model['shots']['s01']['time_origin_pts'])/12800.+camera.tau*(rr/1079.-.5);pred=camera.at(world,tt);error=pred['uv']-arrays['uv'][rows];distance=np.linalg.norm(error,axis=1)
    query_rows=np.isin(mapping[rows],qa['source_rows']);distance[query_rows]=np.abs(np.sum(error[query_rows]*arrays['profile_normal'][rows[query_rows]],axis=1))
    if not legal.all() or not pred['valid'].all():raise ValueError('restored original source auxiliary projection domain failed')
    with np.load(manifest['predictions']['path'],allow_pickle=False) as z:
        max_original_pixel_delta=float(np.max(np.abs(pred['uv']-z['uv'][mapping[rows]])))
    if max_original_pixel_delta>2e-8:raise ValueError('d78 actual CP auxiliary import changed original prediction')
    ad2=np.array([labels1[int(m)].startswith('Ad2.') for m in mm])&(arrays['kind'][rows]==1);ar=np.linalg.norm(error[ad2],axis=1)
    if file_sha256(packet_file)!=bound_packet_sha or file_sha256(model_file)!=bound_model_sha:raise ValueError('immutable input changed during auxiliary matching')
    out.mkdir(parents=True);files={}
    for name,value in (('rows.u64',rows.astype('<u8')),('original_rows.u64',mapping[rows].astype('<u8')),('station.f64',ss.astype('<f8')),('path.u32',pp.astype('<u4')),('row.f64',rr.astype('<f8')),('alpha.f64',oldalpha[mapping[rows]].astype('<f8'))):
        p=out/name;value.tofile(p);files[name]=dict(file=str(p.resolve()),sha256=file_sha256(p),dtype=value.dtype.str,shape=list(value.shape))
    report=dict(schema='rinkfit.exact_native_baseline_auxiliary_import.v1',current_packet=dict(file=str(packet_file.resolve()),sha256=bound_packet_sha),model=dict(file=str(Path(model_file).resolve()),sha256=bound_model_sha),baseline_model=manifest['model'],complete_baseline_state=dict(file=str(complete_state_file.resolve()),sha256=file_sha256(complete_state_file)),source_key_fields=['source_dataset_id','shot_id','native_pts','source_row_id'],original_rows=N,bijective_original_prefix_rows=N,original_manifest_active_rows=int(oldactive.sum()),current_original_active_subset_rows=len(rows),current_original_withheld_rows=int(oldactive.sum())-len(rows),preserved_active_rows=len(rows),original_inactive_rows_not_resurrected=True,all_original_UV_ID_PTS_material_kind_normal_Query_exact=True,files=files,full_model_equivalence=equivalence,maximum_saved_d78_prediction_change_px=max_original_pixel_delta,RS_equality_max_px=float(np.max(np.abs(rr-pred['uv'][:,1]))),all_valid_original_rows=len(rows),overall_original_operator_RMS_px=float(np.sqrt(np.mean(distance*distance))),overall_original_euclidean_RMS_px=float(np.sqrt(np.mean(np.sum(error*error,axis=1)))),overall_original_maximum_px=float(np.max(distance)),fit_accuracy_accepted=False,Ad2_same623=dict(n=int(ad2.sum()),RMS_px=float(np.sqrt(np.mean(ar*ar))),median_px=float(np.median(ar)),P95_px=float(np.quantile(ar,.95)),max_px=float(ar.max())),source_membership_UV_W_E_or_qualification_changed=False,v98_auxiliary_state_loaded=False,preservation_policy='install these exact global rows and skip fresh reprofile for them; rebuild remaining qualified rows independently',does_not_supply_a_new_source_qualification_certificate=True)
    write_json(out/'manifest.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('files','full_model_equivalence')},ensure_ascii=False));return report


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--model',required=True);p.add_argument('--complete-state',required=True);p.add_argument('--out',required=True);a=p.parse_args(argv);build_d78_auxiliary_import(a.packet,a.model,a.complete_state,a.out)
if __name__=='__main__':main()
