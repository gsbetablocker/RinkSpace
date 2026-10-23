"""Compile verified imported s01 values/statistics into the joint packet schema."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .io_utils import file_sha256, write_json

class SnapshotResolver:
    def __init__(self,snapshot):
        self.root=Path(snapshot).resolve()
        self.receipt=json.loads((self.root/'import_receipt.json').read_text(encoding='utf8'))
        self.paths={d:self.root/p for d,p in self.receipt['dependency_paths_by_sha256'].items()}
        self.checked=set()
    def ref(self,spec):
        digest=spec['sha256']
        if digest not in self.paths:raise ValueError('dependency outside verified snapshot closure: '+digest)
        path=self.paths[digest]
        if digest not in self.checked:
            if file_sha256(path)!=digest:raise ValueError('changed snapshot dependency '+str(path))
            self.checked.add(digest)
        return path
    def role(self,role):
        e=[e for e in self.receipt['dependencies'] if e['role']==role]
        if len(e)!=1:raise ValueError('missing/ambiguous snapshot role '+role)
        return self.ref(e[0])
    def json(self,spec):return json.loads(self.ref(spec).read_text(encoding='utf8'))
    def array(self,spec):
        x=np.fromfile(self.ref(spec),spec['dtype'])
        if x.size!=int(np.prod(spec['shape'])):raise ValueError('array byte size changed')
        return x.reshape(spec['shape'])

def compile_s01(snapshot,output):
    resolver=SnapshotResolver(snapshot);out=Path(output)
    if out.exists():raise ValueError('new packet generation required')
    out.mkdir(parents=True)
    sem=json.loads(resolver.role('semantic_manifest').read_text(encoding='utf8'))
    obj=json.loads(resolver.role('objective_manifest').read_text(encoding='utf8'))
    if 'frozen_query_manifest' not in obj:raise ValueError('incomplete frozen profile query closure')
    q=resolver.json(obj['frozen_query_manifest']);qa={k:resolver.array(v) for k,v in q['arrays'].items()}
    with np.load(resolver.role('observations'),allow_pickle=False) as z:
        uv=z['source_uv'];pts=z['native_pts'];source_ids=z['source_point_ids'].tolist()
    n=len(uv);cp=resolver.root/'checkpoint'
    model=json.loads((cp/'model.json').read_text(encoding='utf8'))
    src={k:resolver.array(v) for k,v in sem['arrays'].items()}
    active=resolver.array(sem['active_source_mask']).astype(bool)
    material=resolver.array(sem['resolved_material_index'])
    saved=np.load(cp/'candidate_predictions.npz',allow_pickle=False)
    arrays={}
    def save(name,value,dtype):
        a=np.asarray(value,dtype=dtype);path=out/(name+'.bin');a.tofile(path)
        arrays[name]={'file':path.name,'sha256':file_sha256(path),'dtype':a.dtype.str,'shape':list(a.shape),'layout':'C','byte_order':'little'}
    for name,value,dtype in [('uv',uv,'<f8'),('native_pts',pts,'<i8'),('shot_index',np.zeros(n),'<u2'),
        ('kind',src['kind'],'|u1'),('material_index',material,'<u4'),('path',saved['branches'],'<u4'),
        ('admission',active,'|u1'),('fit_role',np.ones(n),'|u1'),('E',resolver.array(obj['engineering_budget']),'<f8'),
        ('station',saved['stations'],'<f8'),('row',saved['root_rows'],'<f8'),('alpha',saved['profile_alpha'],'<f8'),
        ('source_sigma',src['source_sigma'],'<f8'),('profile_normal',src['profile_normal'],'<f8'),
        ('query_uv',src['query_uv'],'<f8'),('alpha_bounds',src['alpha_bounds'],'<f8')]:save(name,value,dtype)
    statistics=resolver.json(sem['active_statistics']);replacements={x['group_id']:x for x in statistics['replacements']}
    weights=resolver.ref(sem['arrays']['weights']);wdesc={'file':str(weights),'sha256':file_sha256(weights),'dtype':'<f8','layout':'C','byte_order':'little'}
    groups=[];partition=np.zeros(n,np.int16);rank=0
    for line in resolver.ref(sem['assets']['groups.jsonl']).read_text(encoding='utf8').splitlines():
        if not line.strip():continue
        group=json.loads(line);rr=src['group_rows'][group['member_start']:group['member_start']+group['member_count']].astype(int)
        if group['group_id'] in replacements:
            r=replacements[group['group_id']];rr=np.asarray(r['source_rows'],int);W=dict(r['W']);W['file']=str(resolver.ref(W));grank=r['rank']
        else:
            grank=group['rank'];W={**wdesc,'offset_bytes':8*group['weight_start'],'shape':[grank,2*len(rr)]}
        if not active[rr].all():raise ValueError('inactive row leaked into full W')
        partition[rr]+=1;rank+=grank
        groups.append({'id':'s01:'+group['group_id'],'rows':rr.tolist(),'rank':grank,'W':W,
            'source_group_id':group['group_id'],'statistic_kind':'original',
            'original_matrix_sha256':group['matrix_sha256']})
    independent=src['ordinary_rows'].astype(int);independent=independent[active[independent]]
    partition[independent]+=1;rank+=2*len(independent)
    if not np.array_equal(partition,active.astype(int)) or rank!=statistics['formal_residual_rows']:
        raise ValueError('current original W partition/rank changed during packet compile')
    queries=[]
    for i,row in enumerate(qa['source_rows']):
        start,end=qa['offsets'][i:i+2].astype(int)
        queries.append({'row_index':int(row),'alpha':qa['alpha'][start:end].tolist(),
            'native_uv':qa['uv'][start:end].tolist(),'normal':src['profile_normal'][row].tolist(),
            'numerator_x':qa['numerator_u'][i].tolist(),'numerator_y':qa['numerator_v'][i].tolist(),
            'denominator':qa['denominator'][i].tolist(),'alpha_origin':0.,'alpha_scale':1.})
    packet={'schema':'rinkfit.joint_packet.v1','row_count':n,'shot_ids':['s01','s02','s03','s04','s05','s06','s07','s08','s09'],
        'arrays':arrays,'source_dataset_ids':['s01_original']*n,'source_row_ids':source_ids,
        'groups':groups,'independent_rows':independent.tolist(),'independent_statistic_kind':'original',
        'queries':queries,'source_manifests':[{'file':str(resolver.root/'import_receipt.json'),'sha256':file_sha256(resolver.root/'import_receipt.json')}],
        'current_original_W_rank':rank,'active_rows':int(active.sum()),
        'source_identity':resolver.receipt['source_identity'],
        'source_order_overlay':{'file':str(resolver.ref(sem['source_order_overlay'])),'sha256':sem['source_order_overlay']['sha256']},
        'resolved_variant_index':{'file':str(resolver.ref(sem['resolved_variant_index'])),**{k:v for k,v in sem['resolved_variant_index'].items() if k!='file'}},
        'import_complete_checkpoint_sha256':resolver.receipt['checkpoint_manifest_sha256'],
        'import_model_sha256':resolver.receipt['model_sha256'],
        'nuisance_rows_same_order_as_original':True,'new_joint_geometry_or_predictions_claimed':False}
    write_json(out/'packet.json',packet)
    report={'schema':'rinkfit.joint_s01_packet_compile.v1','raw_rows':n,'active_rows':int(active.sum()),
        'original_group_count':len(groups),'exact_marginal_groups':len(replacements),
        'independent_rows':len(independent),'statistics_rank':rank,'frozen_queries':len(queries),
        'every_active_source_counted_once':True,'packet_sha256':file_sha256(out/'packet.json'),
        'current_model_sha256':resolver.receipt['model_sha256'],'native_replay_performed':False}
    write_json(out/'compile_report.json',report);return report


def append_admitted_sources(s01_packet,solver_rows,statistics_registry,material_map,model_path,output,*,raw_ledger=None,admission_ledger=None,initialization_geometry=False):
    from .io_utils import read_jsonl,object_sha256
    from .source_statistics import supported_factor_whitening
    out=Path(output)
    if out.exists():raise ValueError('new joint packet generation required')
    parent_path=Path(s01_packet);parent=json.loads(parent_path.read_text(encoding='utf8'))
    if parent.get('board_material_source_qualification'):
        raise ValueError('Legacy append cannot rebind qualified source certificates; use a qualified reference append finalizer')
    if parent.get('dense_parent_packet') or parent.get('dense_source_binding'):
        raise ValueError('Legacy append does not support dense packets; use the reference source append entry')
    out.mkdir(parents=True)
    selected=list(read_jsonl(solver_rows));eligible={r['source_row_id']:r for r in selected}
    new=list(read_jsonl(raw_ledger)) if raw_ledger else selected
    if raw_ledger:
        for row in new:
            keep=eligible.get(row['source_row_id'])
            row['solver_eligible']=keep is not None
            if keep is not None:
                for key in ['source_uv','current_uv','native_pts','joint_material_id']:
                    if row.get(key)!=keep.get(key):raise ValueError('promotion redirects immutable/current source key: '+key)
                row.update({k:v for k,v in keep.items() if k in {'joint_material_index','statistical_group_id','kind','entity_station_fixed'}})
    if admission_ledger:
        gate={r['source_row_id']:r for r in read_jsonl(admission_ledger)}
        if set(gate)!={r['source_row_id'] for r in new}:raise ValueError('admission decisions do not cover complete raw ledger')
        for row in new:
            decision=gate[row['source_row_id']]
            if bool(decision.get('solver_eligible'))!=bool(row.get('solver_eligible')):raise ValueError('admission ledger and solver rows disagree')
            row['admission_decision']=decision.get('decision')
            row['admission_reason']=decision.get('promotion_errors',decision.get('source_row_reasons'))
    N0=parent['row_count'];n=N0+len(new)
    registry_path=Path(statistics_registry);registry=json.loads(registry_path.read_text(encoding='utf8'))
    mapping=json.loads(Path(material_map).read_text(encoding='utf8'))['semantic_to_material']
    model=json.loads(Path(model_path).read_text(encoding='utf8'))
    if not json.loads(Path(material_map).read_text(encoding='utf8')).get('s01_material_indices_preserved'):
        raise ValueError('old material indices must remain exactly mapped')
    byid={row['source_row_id']:N0+i for i,row in enumerate(new)}
    if len(byid)!=len(new):raise ValueError('new source ledger row ids are not unique')
    arrays={}
    newuv=np.asarray([r.get('current_uv') if r.get('current_uv') is not None else r['source_uv'] for r in new],float)
    material=np.zeros(len(new),np.uint32);active=np.zeros(len(new),bool);kind=np.zeros(len(new),np.uint8)
    inactive=[]
    for i,row in enumerate(new):
        desc=mapping.get(row.get('joint_material_id'),{})
        permitted=bool(row.get('solver_eligible'))
        if permitted and row.get('joint_material_index',desc.get('material_index'))!=desc.get('material_index'):raise ValueError('promoted shared material index differs from frozen map')
        if permitted and (not desc.get('geometry_ready') or row.get('current_uv') is None):
            raise ValueError('promoted source lacks current pixels or shared physical geometry')
        if desc.get('geometry_ready'):
            material[i]=desc['material_index']
        active[i]=permitted;kind[i]=1 if row.get('point_correspondence') else 0
        if permitted and kind[i] and not model['materials'][material[i]]['paths'][0].get('physical_point'):
            raise ValueError('entity corner bound to non-point shared geometry')
        if not permitted:inactive.append({'source_row_id':row['source_row_id'],'shot_id':row['shot_id'],'native_pts':row['native_pts'],
            'reason':row.get('admission_reason',row.get('source_review_reasons',row.get('source_support_status'))),
            'source_uv':row['source_uv'],'current_uv':row.get('current_uv'),'joint_material_id':row.get('joint_material_id')})
    joint_groups=[];parent_numeric_checked={}
    for old_group in parent['groups']:
        ref=dict(old_group['W']);wp=Path(ref['file']);wp=wp if wp.is_absolute() else parent_path.parent/wp
        if str(wp) not in parent_numeric_checked:
            if file_sha256(wp)!=ref['sha256']:raise ValueError('changed parent source W')
            parent_numeric_checked[str(wp)]=ref['sha256']
        joint_groups.append({**old_group,'W':{**ref,'file':str(wp)}})
    partition=np.zeros(len(new),np.int16);rank=parent.get('joint_statistics_rank',parent['current_original_W_rank']);marginals=[]
    geometric_bundle=out/'native_geometry_basis.f64';geometric_stream=geometric_bundle.open('wb') if initialization_geometry else None
    for gid,group in registry.items():
        ids=group['row_ids'];presence=[rid in byid for rid in ids]
        if not any(presence):continue
        if not all(presence):raise ValueError('statistical family crosses immutable append boundary; joint covariance union required')
        rr=np.asarray([byid[i]-N0 for i in ids],int)
        keep=active[rr]
        if not keep.any():continue
        for rid in np.asarray(ids)[keep]:
            row=new[byid[rid]-N0]
            digest=object_sha256({'native_pts':row['native_pts'],'current_uv':row['current_uv'],'source_row_id':rid,
                'measurement_operator':row['measurement_operator_resolved']})
            if digest!=group['current_measurements_sha256'][rid]:raise ValueError('W source means differ from promoted measurements')
        ref=dict(group['W']);wp=Path(ref['file']);wp=wp if wp.is_absolute() else registry_path.parent/wp
        if file_sha256(wp)!=ref['sha256']:raise ValueError('new source W changed')
        W=np.fromfile(wp,ref['dtype']).reshape(ref['shape'])
        if W.shape!=(group['rank'],2*len(ids)):raise ValueError('new source W shape')
        if initialization_geometry:
            count=int(keep.sum());matrix=np.eye(2*count,dtype='<f8');offset=geometric_stream.tell();matrix.tofile(geometric_stream)
            ref={'file':geometric_bundle.name,'sha256':'pending_closed_bundle','dtype':'<f8','shape':list(matrix.shape),'offset_bytes':offset};grank=2*count
        elif keep.all():ref['file']=str(wp);grank=len(W)
        else:
            U,ss,V=np.linalg.svd(W,full_matrices=False)
            if len(ss)!=len(W) or np.any(ss<=0):raise ValueError('original source W lost its declared row rank')
            F=(V.T/ss)@U.T;coordinates=np.ravel(np.c_[2*np.flatnonzero(keep),2*np.flatnonzero(keep)+1])
            marginal,proof=supported_factor_whitening(F[coordinates]);name=f'marginal_{len(marginals):05d}.f64'
            marginal.astype('<f8').tofile(out/name);ref={'file':name,'sha256':file_sha256(out/name),'dtype':'<f8','shape':list(marginal.shape)};grank=len(marginal)
            marginals.append({'source_group_id':gid,'original_W_sha256':group['W']['sha256'],
                'original_row_ids':ids,'retained_row_ids':np.asarray(ids)[keep].tolist(),
                'original_rank':len(W),'new_rank':grank,'covariance_marginal_proof':proof,
                'W_columns_sliced':False})
        selected=rr[keep];partition[selected]+=1;rank+=grank
        joint_groups.append({'id':'new:'+gid,'source_group_id':gid,'rows':(selected+N0).tolist(),'rank':grank,'W':ref,
            'statistic_kind':'initialization_surrogate','original_numerical_scheme':group.get('original_numerical_scheme',group['statistic_kind']),'original_W_claimed':False,
            'noise_calibration_status':'explicit native geometry only; source covariance not used' if initialization_geometry else 'linearized_initialization_model_pending_nonlinear_source_family_resampling',
            'source_local_W_full_group':{**group['W'],'file':str(wp)},'source_W_row_ids':ids,
            'source_covariance_marginal_claimed':False if initialization_geometry else True,
            'native_geometry_requires_fixed_group_shot_normalization':bool(initialization_geometry)})
    if initialization_geometry:
        geometric_stream.close();digest=file_sha256(geometric_bundle)
        for g in joint_groups[len(parent['groups']):]:g['W']['sha256']=digest
    if not np.array_equal(partition,active.astype(np.int16)):raise ValueError('promoted new sources do not have exactly one complete W statistical unit')
    values={'uv':newuv,'native_pts':np.asarray([r['native_pts'] for r in new]),
        'shot_index':np.asarray([int(r['shot_id'][1:])-1 for r in new]),'kind':kind,'material_index':material,
        'path':np.asarray([(r.get('allowed_path_indices') or [0])[0] for r in new]),'admission':active,'fit_role':np.ones(len(new)),
        'E':np.asarray([r.get('engineering_budget_px',2.) for r in new]),'station':np.full(len(new),.5),
        'row':np.where((newuv[:,1]>=0)&(newuv[:,1]<=1079),newuv[:,1],539.5),
        'alpha':np.zeros(len(new)),'source_sigma':np.ones(len(new)),'profile_normal':np.zeros((len(new),2)),
        'query_uv':newuv,'alpha_bounds':np.zeros((len(new),2))}
    parent_values={}
    for key,ref in parent['arrays'].items():
        base=Path(ref['file']);base=base if base.is_absolute() else parent_path.parent/base
        if file_sha256(base)!=ref['sha256']:raise ValueError('changed s01 parent array')
        a=np.memmap(base,dtype=ref['dtype'],mode='r',offset=int(ref.get('offset_bytes',0)),shape=tuple(ref['shape']));parent_values[key]=a;extra=np.asarray(values[key],dtype=ref['dtype'])
        joined=np.concatenate((a,extra));path=out/(key+'.bin');joined.tofile(path)
        arrays[key]={'file':path.name,'sha256':file_sha256(path),'dtype':joined.dtype.str,'shape':list(joined.shape),'layout':'C','byte_order':'little'}
    original_source_start=parent.get('new_source_start',N0)
    packet={**parent,'row_count':n,'arrays':arrays,'source_dataset_ids':parent['source_dataset_ids']+['current_native_'+r['shot_id'] for r in new],
        'source_row_ids':parent['source_row_ids']+[r['source_row_id'] for r in new],
        'groups':joint_groups,'active_rows':parent['active_rows']+int(active.sum()),'joint_statistics_rank':rank,
        's01_original_W_rank':parent['current_original_W_rank'],'new_source_statistics_rank':rank-parent['current_original_W_rank'],
        'current_original_W_rank':parent['current_original_W_rank'],
        'source_manifests':parent['source_manifests']+[{'file':str(parent_path.resolve()),'sha256':file_sha256(parent_path),'role':'immutable parent packet source order and numeric prefix'}]+[{'file':str(Path(q).resolve()),'sha256':file_sha256(q)} for q in [solver_rows,statistics_registry,material_map,model_path]+([raw_ledger] if raw_ledger else [])+([admission_ledger] if admission_ledger else [])],
        'same_s01_raw_rows_prefix_preserved':original_source_start,'new_source_start':original_source_start,'newly_appended_source_start':N0,
        'recapture_rows':(N0+np.flatnonzero(active)).tolist(),'new_rows_initialized_for_fresh_capture':len(new),
        'allowed_paths':{**parent.get('allowed_paths',{}),**{str(N0+i):row['allowed_path_indices'] for i,row in enumerate(new) if active[i] and row.get('allowed_path_indices')}},
        'new_nuisance_state_recomputed':False,'complete_joint_solution_claimed':False,
        'new_source_statistics_status':'initialization_surrogate; nonlinear source-family resampling pending',
        'formal_new_source_statistical_optimum_claimed':False,
        'new_source_numeric_covariance_used':not initialization_geometry,
        'native_geometry_requires_fixed_group_shot_normalization':bool(initialization_geometry)}
    from .measurement_operator import assert_append_operator_prefix
    assert_append_operator_prefix(parent, packet, N0)
    qa_path=Path(statistics_registry).parent.parent/'pixel_statistics_QA/report.json'
    if qa_path.is_file():
        packet['new_source_noise_calibration_QA']={'file':str(qa_path),'sha256':file_sha256(qa_path)}
    write_json(out/'packet.json',packet);write_json(out/'marginal_W_proofs.json',marginals);write_json(out/'inactive_sources.json',inactive)
    report={'schema':'rinkfit.joint_admitted_packet_compile.v1','raw_rows':n,'active_rows':packet['active_rows'],
        's01_active_rows':int(np.sum(parent_values['admission'][:original_source_start]==1)),'prior_active_rows':parent['active_rows'],'new_active_rows':int(active.sum()),'new_ledger_rows':len(new),
        'joint_statistics_rank':rank,'s01_original_W_rank':parent['current_original_W_rank'],
        'source_covariance_marginal_groups':len(marginals),'all_active_sources_partitioned_once':True,
        'packet_sha256':file_sha256(out/'packet.json'),'model_sha256':file_sha256(model_path),
        'new_active_shots':sorted({new[i]['shot_id'] for i in np.flatnonzero(active)}),
        'current_uv_preserved_exactly':True,'native_capture_required_before_state_claim':True,
        'new_source_numeric_covariance_used':not initialization_geometry,'initialization_geometry_only':bool(initialization_geometry)}
    write_json(out/'compile_report.json',report);return report
