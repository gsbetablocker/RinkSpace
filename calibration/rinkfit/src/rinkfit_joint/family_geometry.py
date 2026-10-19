"""Explicit source-family/shot balanced native geometry initialization.

This is a geometric objective, never a physical noise likelihood. It combines
corner and curve residuals belonging to the same actual source feature family
and recomputes the per-shot family count after physical source withdrawals.
"""
from pathlib import Path
from .packet_storage import write_packet
from collections import defaultdict,Counter
import json
import numpy as np
from .io_utils import read_jsonl,write_json,file_sha256

def compile_family_geometry(parent_packet,source_journals,output,shot_total_weight=1.0,*,preserve_conditional_statistics=False):
    src=Path(parent_packet).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('a new immutable generation is required')
    packet=json.loads(src.read_text(encoding='utf8'));start=int(packet['new_source_start'])
    checked={}
    def resolve(ref):
        p=Path(ref['file']);p=p if p.is_absolute() else src.parent/p
        key=(str(p),ref['sha256'])
        if key not in checked:
            if file_sha256(p)!=ref['sha256']:raise ValueError('changed input '+str(p))
            checked[key]=True
        return p.resolve()
    arrays={k:{**v,'file':str(resolve(v))} for k,v in packet['arrays'].items()}
    def array(name):
        r=arrays[name];return np.fromfile(r['file'],r['dtype']).reshape(r['shape'])
    admission=array('admission');roles=array('fit_role');shots=array('shot_index');pts=array('native_pts')
    source={};refs=[]
    for path in map(Path,source_journals):
        refs.append(dict(file=str(path.resolve()),sha256=file_sha256(path)))
        for row in read_jsonl(path):
            rid=row['source_row_id'];entry=(row['shot_id'],int(row['native_pts']),row['source_family_id'])
            if rid in source and source[rid]!=entry:raise ValueError('contradictory source family '+rid)
            source[rid]=entry
    conditional=[];conditional_rows=set()
    if preserve_conditional_statistics:
        for g in packet['groups']:
            if g.get('statistic_kind')=='native_joint_readout'and all(i>=start for i in g['rows']):
                active=[i for i in g['rows']if admission[i]==1 and roles[i]==1]
                if active and len(active)!=len(g['rows']):raise ValueError('conditional group requires exact Cov marginal before geometry normalization')
                if active:
                    if conditional_rows.intersection(active):raise ValueError('duplicate conditional measurement consumption')
                    conditional_rows.update(active);conditional.append({**g,'W':{**g['W'],'file':str(resolve(g['W']))}})
    families=defaultdict(list);family_ids=[None]*len(admission)
    for i in range(start,len(admission)):
        rid=packet['source_row_ids'][i]
        if rid not in source:raise ValueError('source family journal missing row '+rid)
        sid,native,family=source[rid]
        if packet['shot_ids'][int(shots[i])]!=sid or int(pts[i])!=native:raise ValueError('source clock identity mismatch')
        family_ids[i]=family
        if admission[i]==1 and roles[i]==1 and i not in conditional_rows:families[(int(shots[i]),native,family)].append(i)
    original=[]
    for g in packet['groups']:
        if all(i<start for i in g['rows']):
            original.append({**g,'W':{**g['W'],'file':str(resolve(g['W']))}})
        elif any(i<start for i in g['rows']):raise ValueError('mixed historical and new statistic group')
    if any(i>=start for i in packet['independent_rows']):raise ValueError('unclassified new independent noise statistic')
    counts=Counter(k[0] for k in families);groups=list(original)+conditional;geometry_group_start=len(groups);out.mkdir(parents=True)
    bundle=out/'family_geometry_W.f64';audit=[];stored={};logical_W_bytes=0
    with bundle.open('wb') as f:
        for (shot,native,family),rows in sorted(families.items()):
            m=len(rows);scale=float(np.sqrt(shot_total_weight/(m*counts[shot])))
            key=(2*m,np.float64(scale).tobytes());logical_W_bytes+=(2*m)**2*8
            if key in stored:offset=stored[key]
            else:
                W=np.eye(2*m,dtype='<f8')*scale;offset=f.tell();W.tofile(f);stored[key]=offset
            record=dict(id='init_family:'+family,source_group_id=family,rows=rows,rank=2*m,
                W=dict(file=str(bundle),sha256='pending_closed_bundle',dtype='<f8',shape=[2*m,2*m],offset_bytes=offset),
                statistic_kind='initialization_surrogate',original_numerical_scheme='fixed_actual_source_family_and_shot_native_geometry',
                geometry_weight=dict(formula='I_2m * sqrt(shot_total_weight / (active_rows_in_actual_source_family * actual_source_families_in_shot))',
                    rows_in_family=m,families_in_shot=counts[shot],shot_total_weight=shot_total_weight,scale=scale),
                physical_source_family_covariance_claimed=False,source_local_W_used=False,
                noise_calibration_status='geometric_initialization_only; no source noise likelihood')
            groups.append(record);audit.append(dict(source_family_id=family,shot_id=packet['shot_ids'][shot],native_pts=native,row_count=m,geometry_scale=scale))
    digest=file_sha256(bundle)
    for g in groups[geometry_group_start:]:g['W']['sha256']=digest
    conditional_rank=sum(int(g['rank'])for g in conditional);rank=int(packet['current_original_W_rank'])+2*sum(map(len,families.values()))+conditional_rank
    result={**packet,'arrays':arrays,'groups':groups,'source_family_ids':family_ids,'joint_statistics_rank':rank,
        'new_source_statistics_rank':rank-int(packet['current_original_W_rank']),
        'new_source_statistics_status':('mixed actual source-family/shot geometry initialization and preserved source-qualified conditional native RGB groups'if conditional else 'actual source-family/shot balanced native geometry initialization; source-local noise W not used'),'preserved_conditional_statistics_rows':sorted(conditional_rows),'preserved_conditional_statistics_rank':conditional_rank,
        'initialization_objective_only':True,'physical_noise_likelihood_claimed':False,
        'formal_new_source_statistical_optimum_claimed':False,
        'family_geometry_parent_packet':dict(file=str(src),sha256=file_sha256(src)),
        'source_family_journal_refs':refs,'source_mean_admission_roles_four_aux_or_original_s01_W_changed':False}
    write_packet(out/'packet.json',result);write_json(out/'families.json',audit)
    report=dict(schema='rinkfit.actual_source_family_native_geometry_initialization.v1',
        raw_rows=len(admission),active_rows=int(np.sum((admission==1)&(roles==1))),
        new_active_rows=sum(map(len,families.values()))+len(conditional_rows),new_actual_source_families=len(families),preserved_conditional_rows=len(conditional_rows),preserved_conditional_rank=conditional_rank,
        family_count_by_shot={packet['shot_ids'][k]:v for k,v in counts.items()},
        squared_geometry_weight_per_shot={packet['shot_ids'][k]:sum(2*len(v)*(shot_total_weight/(len(v)*counts[k])) for key,v in families.items() if key[0]==k) for k in counts},
        note='squared_geometry_weight_per_shot sums both UV components; average squared Euclidean residual has total shot weight 1',
        original_s01_W_unchanged=True,UV_clock_auxiliaries_admission_roles_unchanged=True,
        exact_equal_geometry_matrix_slices_reused=True,unique_geometry_W_slices=len(stored),logical_geometry_W_bytes=logical_W_bytes,stored_geometry_W_bytes=bundle.stat().st_size,
        physical_source_approval_added=False,formal_likelihood_claimed=False,joint_statistics_rank=rank,
        packet_sha256=file_sha256(out/'packet.json'),source_journals=refs)
    write_json(out/'report.json',report);return report
