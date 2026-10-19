"""Exact S01 numerical component rollback, retaining all other accepted values."""
from __future__ import annotations
import argparse,json,itertools
from pathlib import Path
import numpy as np
from .baseline_auxiliary import packet_header,array_items,checked_array
from .io_utils import file_sha256,write_json


def build_s01_component_rollback(input_model,target_packet,accepted_model,baseline_model,baseline_auxiliary,source_packet,output):
    out=Path(output)
    if out.exists():raise ValueError('immutable component rollback generation exists')
    paths={k:Path(v) for k,v in dict(input_model=input_model,packet=target_packet,accepted_model=accepted_model,baseline_model=baseline_model,baseline_auxiliary=baseline_auxiliary,source_packet=source_packet).items()}
    bound={k:dict(file=str(p.resolve()),sha256=file_sha256(p)) for k,p in paths.items()}
    current,accepted,baseline=[json.loads(paths[k].read_text(encoding='utf8')) for k in ('input_model','accepted_model','baseline_model')]
    if len(current['blocks'])!=len(accepted['blocks']) or len(baseline['blocks'])!=len(accepted['blocks']):raise ValueError('physical allocation differs')
    own=[]
    for i,(a,b,c) in enumerate(zip(accepted['blocks'],baseline['blocks'],current['blocks'])):
        if a['name']!=b['name'] or a['name']!=c['name'] or a['values']!=c['values']:raise ValueError('input physical model is not exact accepted g3 values')
        if a['name'].startswith('shot/s01/'):
            if len(a['values'])!=len(b['values']) or not np.isfinite(b['values']).all():raise ValueError('S01 physical allocation differs')
            own.append(i)
        elif not a['name'].startswith('shot/') and a['values']!=b['values']:raise ValueError('shared scene/rig/geometry changed; S01-only restoration is not equivalent')
    if not own or accepted['shots']['s01']!=baseline['shots']['s01'] or current['shots']['s01']!=accepted['shots']['s01']:raise ValueError('S01 native camera function/time scope differs')
    oldhead=packet_header(paths['source_packet']);head=packet_header(paths['packet']);n=oldhead['row_count']
    if head['row_count']<n or oldhead['shot_ids']!=head['shot_ids']:raise ValueError('source prefix clock/allocation differs')
    original={};actual={}
    immutable=('uv','native_pts','shot_index','material_index','kind','E','admission','fit_role','profile_normal','query_uv','stroke_pair_index','stroke_source_side')
    for key in immutable:
        original[key]=checked_array(oldhead['arrays'][key],paths['source_packet'].parent);actual[key]=checked_array(head['arrays'][key],paths['packet'].parent)
        if not np.array_equal(original[key],actual[key][:n]):raise ValueError('immutable accepted source prefix changed: '+key)
    for key in ('source_row_ids','source_dataset_ids'):
        old=array_items(paths['source_packet'],key);new=itertools.islice(array_items(paths['packet'],key),n)
        for a,b in itertools.zip_longest(old,new,fillvalue=None):
            if a!=b:raise ValueError('immutable compound source identity prefix differs')
    oldq=array_items(paths['source_packet'],'queries');newq=array_items(paths['packet'],'queries')
    count=0
    for a in oldq:
        try:b=next(newq)
        except StopIteration:raise ValueError('frozen original Query prefix missing')
        if a!=b:raise ValueError('frozen original Query/operator changed')
        count+=1
    rows=np.flatnonzero(original['shot_index']==oldhead['shot_ids'].index('s01')).astype('<u8')
    current_rows=np.flatnonzero(actual['shot_index']==head['shot_ids'].index('s01'))
    if not np.array_equal(rows,current_rows):raise ValueError('new S01 sources need an explicit correspondence contract, not blanket rollback')
    aux=json.loads(paths['baseline_auxiliary'].read_text(encoding='utf8'))
    if aux['packet_sha256']!=bound['source_packet']['sha256'] or aux['raw_rows']!=n:raise ValueError('baseline auxiliary source generation differs')
    values={}
    for name,dtype in (('station.f64','<f8'),('row.f64','<f8'),('alpha.f64','<f8'),('path.u32','<u4')):
        f=paths['baseline_auxiliary'].parent/name
        if file_sha256(f)!=aux['files'][name]['sha256'] or f.stat().st_size!=aux['files'][name]['size_bytes']:raise ValueError('changed baseline auxiliary: '+name)
        v=np.memmap(f,mode='r',dtype=dtype,shape=(n,));values[name]=np.asarray(v[rows],dtype=dtype)
    active=(original['admission'][rows]==1)&(original['fit_role'][rows]==1)
    if not all(np.isfinite(v).all() for v in values.values()):raise ValueError('nonfinite component auxiliary')
    for k,p in paths.items():
        if file_sha256(p)!=bound[k]['sha256']:raise ValueError('component dependency changed during verification: '+k)
    out.mkdir(parents=True);files={}
    for name,v in dict({'rows.u64':rows},**values).items():
        f=out/name;np.asarray(v).tofile(f);files[name]=dict(file=name,sha256=file_sha256(f),dtype=np.asarray(v).dtype.str,shape=list(v.shape))
    manifest=dict(schema='rinkfit.exact_S01_component_rollback.v1',shot_id='s01',**bound,files=files,component_raw_rows=len(rows),component_active_rows=int(active.sum()),component_physical_block_indices=own,
        immutable_source_prefix_verified=True,immutable_prefix_rows=n,frozen_Query_prefix_verified=count,shared_scene_mainrig_geometry_baseline_accepted_exact=True,
        old_duals_or_objective_resumed=False,source_qualification_or_statistics_granted=False,source_UV_W_membership_or_ID_changed=False,
        current_source_support_and_motion_subspace_must_remain_unchanged=True,caller_must_start_fresh_engineering_objective=True,
        reason='Actual g3 S01 independent native-RS regression; restore complete original function and auxiliaries while retaining accepted other-shot motion values')
    write_json(out/'manifest.json',manifest);print(json.dumps(dict(manifest=str(out/'manifest.json'),sha256=file_sha256(out/'manifest.json'),raw_rows=len(rows),active_rows=int(active.sum()))));return manifest


def main(argv=None):
    p=argparse.ArgumentParser()
    for name in ('input-model','target-packet','accepted-model','baseline-model','baseline-auxiliary','source-packet','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args(argv);build_s01_component_rollback(a.input_model,a.target_packet,a.accepted_model,a.baseline_model,a.baseline_auxiliary,a.source_packet,a.out)

if __name__=='__main__':main()
