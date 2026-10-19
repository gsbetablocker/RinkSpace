"""Replace a re-audited immutable source prefix while retaining a dense suffix."""
from pathlib import Path
import json
import numpy as np
from .io_utils import file_sha256,write_json

def replace_source_prefix(parent_packet,prefix_packet,output):
    parent_path=Path(parent_packet);prefix_path=Path(prefix_packet);parent=json.loads(parent_path.read_text(encoding='utf8'));prefix=json.loads(prefix_path.read_text(encoding='utf8'));out=Path(output)
    if out.exists():raise ValueError('new prefix replacement generation required')
    out.mkdir(parents=True);n=parent['row_count'];cut=prefix['row_count'];start=parent['new_source_start']
    if not start<=cut<n or start!=prefix['new_source_start']:raise ValueError('source prefix boundary does not agree')
    for key in ['source_row_ids','source_dataset_ids']:
        if prefix[key]!=parent[key][:cut]:raise ValueError('replacement changes immutable source ordering: '+key)
    checked={}
    def ref(spec,base):
        p=Path(spec['file']);p=p if p.is_absolute() else base/p;k=(str(p),spec['sha256'])
        if k not in checked:
            if file_sha256(p)!=spec['sha256']:raise ValueError('changed numerical source during prefix replacement')
            checked[k]=True
        return {**spec,'file':str(p)}
    old={};new={};arrays={}
    for key,spec in parent['arrays'].items():
        o=ref(spec,parent_path.parent);r=ref(prefix['arrays'][key],prefix_path.parent)
        if spec['dtype']!=r['dtype'] or spec['shape'][1:]!=r['shape'][1:]:raise ValueError('array contract changed')
        old[key]=np.fromfile(o['file'],o['dtype']).reshape(o['shape']);new[key]=np.fromfile(r['file'],r['dtype']).reshape(r['shape'])
    if old['native_pts'][:cut].tobytes()!=new['native_pts'].tobytes():raise ValueError('source clock changed')
    old_active=(old['admission'][:cut]==1)&(old['fit_role'][:cut]==1);new_active=(new['admission']==1)&(new['fit_role']==1)
    if np.any(old_active&~new_active):raise ValueError('source prefix replacement must not silently remove a previously admitted source')
    if old['uv'][:cut][old_active].tobytes()!=new['uv'][old_active].tobytes():raise ValueError('previously active source UV changed without a separate revision migration')
    for key in ['station','path','row','alpha']:
        if old[key][:start].tobytes()!=new[key][:start].tobytes():raise ValueError('source prefix replacement changed original s01 four auxiliaries')
    for key,spec in parent['arrays'].items():
        value=np.concatenate((new[key],old[key][cut:]));p=out/(key+'.bin');value.tofile(p)
        arrays[key]={'file':p.name,'sha256':file_sha256(p),'dtype':spec['dtype'],'shape':list(value.shape),'layout':'C','byte_order':'little'}
    groups=[];partition=np.zeros(n,np.int16)
    for packet,path,select in [(prefix,prefix_path,lambda rr:True),(parent,parent_path,lambda rr:np.all(rr>=cut))]:
        for g in packet['groups']:
            rr=np.asarray(g['rows'],int)
            if packet is parent and np.any(rr<cut) and np.any(rr>=cut):raise ValueError('source statistical unit crosses replacement boundary')
            if not select(rr):continue
            groups.append({**g,'W':ref(g['W'],path.parent)});partition[rr]+=1
    independent=prefix['independent_rows']+[r for r in parent['independent_rows'] if r>=cut];partition[independent]+=1
    active=np.r_[new_active,((old['admission'][cut:]==1)&(old['fit_role'][cut:]==1))]
    if not np.array_equal(partition,active.astype(np.int16)):raise ValueError('full replaced source W partition does not match source role/admission')
    rank=sum(g['rank'] for g in groups)+2*len(independent)
    result={**parent,'arrays':arrays,'groups':groups,'independent_rows':independent,'active_rows':int(active.sum()),'joint_statistics_rank':rank,
        'new_source_statistics_rank':rank-parent['s01_original_W_rank'],'recapture_rows':np.flatnonzero(active&(np.arange(n)>=start)).tolist(),
        'allowed_paths':{**{k:v for k,v in parent.get('allowed_paths',{}).items() if int(k)>=cut},**prefix.get('allowed_paths',{})},
        'source_manifests':parent['source_manifests']+prefix['source_manifests']+[{'file':str(q),'sha256':file_sha256(q)} for q in [parent_path,prefix_path]],
        'source_prefix_replacement':{'original_raw_prefix_rows':cut,'parent_packet':{'file':str(parent_path),'sha256':file_sha256(parent_path)},'new_prefix_packet':{'file':str(prefix_path),'sha256':file_sha256(prefix_path)},'formerly_active_mean_changed':False,'dense_suffix_changed':False},
        'native_geometry_requires_fixed_group_shot_normalization':True,'formal_new_source_statistical_optimum_claimed':False}
    write_json(out/'packet.json',result)
    report={'schema':'rinkfit.current_source_prefix_replacement.v1','raw_rows':n,'active_rows':int(active.sum()),'updated_source_prefix_rows':cut,
        'admitted_prefix_growth':int(new_active.sum()-old_active.sum()),'dense_suffix_raw_rows':n-cut,'joint_statistics_rank':rank,
        'original_s01_four_aux_prefix_bit_exact':True,'previously_active_current_uv_bit_exact':True,'entire_dense_array_suffix_bit_exact':True,
        'packet_sha256':file_sha256(out/'packet.json'),'source_identity_or_clock_changed':False}
    write_json(out/'report.json',report);return report
