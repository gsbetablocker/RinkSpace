"""Append source-qualified actual paint interfaces and exact covariance marginals.

The centreline midpoint stays inactive. A shared RGB source group is marginalized
in covariance space; tangent conditions are finite source Queries, not precision.
"""
from pathlib import Path
from .packet_storage import write_packet
from collections import defaultdict
import json
import numpy as np
from .io_utils import read_jsonl,write_json,file_sha256,object_sha256
from .source_statistics import supported_factor_whitening


def append_qualified_paint_pairs(parent_packet,source_gate,statistics_registry,output):
    src=Path(parent_packet).resolve();gate=Path(source_gate).resolve();registry_path=Path(statistics_registry).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('new immutable paired-source generation required')
    p=json.loads(src.read_text(encoding='utf8'));approved=json.loads((gate/'approved_pairs.json').read_text(encoding='utf8'));registry=json.loads(registry_path.read_text(encoding='utf8'));factors={r['source_row_id']:r for r in read_jsonl(gate/'factor_rows.jsonl.gz')};checked={}
    def locate(ref,base=src.parent):
        f=Path(ref['file']);f=f if f.is_absolute()else base/f;k=(str(f),ref['sha256'])
        if k not in checked:
            if file_sha256(f)!=ref['sha256']:raise ValueError('changed immutable numeric dependency '+str(f))
            checked[k]=True
        return f.resolve()
    A={k:np.fromfile(locate(v),v['dtype']).reshape(v['shape'])for k,v in p['arrays'].items()};N0=p['row_count'];rawids=set(p['source_row_ids']);selected=[];pairs=[];by_group=defaultdict(list);queries=list(p['queries']);allowed=dict(p.get('allowed_paths',{}));table=p['material_binding']['materials'];old_midpoints=[]
    for entry in approved:
        pair=entry['pair'];pid=pair['id'];mi=int(entry['joint_material_index']);receipt=pair['source_material_receipt'];normal=np.asarray(pair['source_normal'],float);seed=np.asarray(receipt['original_navigation_seed_uv'],float)
        if not entry.get('source_identity_eligible')or entry['actual_target_id']!=table[mi]['label']:raise ValueError('actual finite material identity differs from frozen model topology')
        ids=pair['factor_row_ids'];uv=np.asarray(pair['actual_flank_uv'],float)
        if len(ids)!=2 or uv.shape!=(2,2)or abs(np.linalg.norm(normal)-1)>1e-10:raise ValueError('invalid two actual source flanks')
        measurement_hash=object_sha256(dict(native_pts=pair['native_pts'],source_row_id=pid,actual_flank_uv=uv.tolist(),measurement_operator='finite_width_paint_material_flank_pair'))
        if measurement_hash!=entry['actual_4D_source_mean_sha256']or measurement_hash!=pair['source_mean_sha256']:raise ValueError('whole 4D measured mean differs from source qualification')
        old=[i for i,rid in enumerate(p['source_row_ids'])if rid==pid and int(A['native_pts'][i])==int(pair['native_pts'])]
        if any(A['admission'][i]==1 and A['fit_role'][i]==1 for i in old):raise ValueError('old midpoint is active; reject duplicate source consumption')
        old_midpoints.extend(old);indices=[];pair_index=len(p.get('stroke_pairs',[]))+len(pairs)
        for side,rid in enumerate(ids):
            if rid in rawids:raise ValueError('source flank already exists in parent raw ledger')
            r=factors[rid];g=registry[entry['statistical_group_id']]
            if not all(r.get(k)is True for k in ('solver_eligible','source_identity_eligible','statistical_objective_eligible','numeric_statistics_ready')):raise ValueError('factor lacks actual source/statistical qualification')
            if r['joint_material_index']!=mi or r['current_uv']!=uv[side].tolist()or r['source_uv']!=uv[side].tolist():raise ValueError('same-mean physical factor mismatch')
            source_hash=object_sha256(dict(native_pts=r['native_pts'],source_uv=r['source_uv'],source_row_id=rid));current_hash=object_sha256(dict(native_pts=r['native_pts'],current_uv=r['current_uv'],source_row_id=rid,measurement_operator=r['measurement_operator_resolved']))
            if source_hash!=g['source_measurements_sha256'][rid]or current_hash!=g['current_measurements_sha256'][rid]:raise ValueError('numeric W and actual source factor are not the same measurement')
            windows=receipt['frozen_flank_windows'];central=windows[len(windows)//2][side];bounds=np.asarray(central['normal_window_offsets'],float)
            if any(w[side]['normal_window_offsets']!=bounds.tolist()for w in windows):raise ValueError('query needs an explicit common frozen transverse acquisition window')
            alpha=float(np.dot(uv[side]-seed,normal));index=N0+len(selected)
            if np.linalg.norm(seed+alpha*normal-uv[side])>1e-9 or not bounds[0]<=alpha<=bounds[1]:raise ValueError('actual flank falls outside its frozen source Query')
            selected.append(dict(row=r,uv=uv[side],normal=normal,bounds=bounds,alpha=alpha,pair_index=pair_index,side=side));indices.append(index);by_group[entry['statistical_group_id']].append((index,rid));allowed[str(index)]=list(range(len(table[mi]['paths'])))
            queries.append(dict(row_index=index,alpha=bounds.tolist(),native_uv=[(seed+a*normal).tolist()for a in bounds],normal=normal.tolist(),source_acquisition_window_only=True,source_factor_row_id=rid))
        pairs.append(dict(id=pid,rows=indices,material_index=mi,target_id=entry['actual_target_id'],shot_id=pair['shot_id'],native_pts=pair['native_pts'],width_scope_id=pair['width_scope_id'],width_scope=pair['width_scope'],allowed_world_side_assignments=pair['allowed_world_side_assignments'],actual_flank_uv=uv.tolist(),source_normal=normal.tolist(),source_mean_sha256=measurement_hash,source_family_id=pair['source_family_id'],source_refs=[dict(file=str(gate/'approved_pairs.json'),sha256=file_sha256(gate/'approved_pairs.json')),dict(file=str(gate/'manifest.json'),sha256=file_sha256(gate/'manifest.json')),dict(source_admission_receipt_sha256=entry['source_admission_receipt_sha256'],actual_finite_source_proof=entry['actual_finite_source_proof'])]))
    if not selected:raise ValueError('no actual source-qualified two-flank measurements')
    out.mkdir(parents=True);newgroups=[];proofs=[];newrank=0
    for gid,items in by_group.items():
        g=registry[gid];ref=g['W'];wp=locate(ref,registry_path.parent);shape=ref['shape'];W=np.fromfile(wp,ref['dtype'],count=int(np.prod(shape)),offset=ref.get('offset_bytes',0)).reshape(shape);U,s,V=np.linalg.svd(W,full_matrices=False)
        if np.any(s<=np.finfo(float).eps*max(W.shape)*s[0]*8):raise ValueError('parent source W contains an unsupported advertised rank')
        F=(V.T/s)@U.T;positions=[g['row_ids'].index(rid)for _,rid in items];cols=[2*j+k for j in positions for k in (0,1)];factor=F[cols];C=factor@factor.T;white,proof=supported_factor_whitening(factor)
        replay=locate(g['replay_arrays'],registry_path.parent)
        with np.load(replay,allow_pickle=False)as z:
            if 'full_uv_covariance_4d_px2'in z:full=z['full_uv_covariance_4d_px2']
            else:
                normals=z['profile_normals'];B=np.zeros((len(normals),2*len(normals)))
                for j,v in enumerate(normals):B[j,2*j:2*j+2]=v
                full=B.T@z['normal_covariance_px2']@B
            direct=full[np.ix_(cols,cols)];draws=z['normal_shift_delta_px'].reshape(g['trials'],-1)[:,positions]
        difference=float(np.max(abs(C-direct),initial=0))
        if difference>1e-8*max(1.,float(np.max(abs(direct)))):raise ValueError('parent W covariance and same-source nonlinear MC covariance disagree')
        ident=float(np.max(abs(white@C@white.T-np.eye(len(white))),initial=0));wf=out/('marginal_'+str(len(newgroups))+'.f64');cf=out/('covariance_'+str(len(newgroups))+'.f64');white.astype('<f8').tofile(wf);C.astype('<f8').tofile(cf)
        record=dict(id='qualified_pair_marginal:'+gid,source_group_id=gid,rows=[i for i,_ in items],rank=len(white),W=dict(file=str(wf),sha256=file_sha256(wf),dtype='<f8',shape=list(white.shape)),statistic_kind='native_joint_readout',original_numerical_scheme='exact_supported_covariance_marginal_of_whole_native_RGB_two_flank_source_group',conditional_model=g['conditional_model'],physical_total_error_calibrated=False,parent_group_W={**ref,'file':str(wp)},parent_registry=dict(file=str(registry_path),sha256=file_sha256(registry_path)),source_factor_row_ids=[rid for _,rid in items],PSD_floor_or_independent_representative_noise_added=False)
        newgroups.append(record);newrank+=len(white);proofs.append(dict(source_group_id=gid,parent_rank=g['rank'],parent_row_count=len(g['row_ids']),selected_parent_row_indices=positions,selected_global_rows=record['rows'],rank=len(white),covariance=dict(file=str(cf),sha256=file_sha256(cf),dtype='<f8',shape=list(C.shape)),W=record['W'],parent_W_to_MC_covariance_max_abs_difference=difference,whitening_identity_max_abs_error=ident,MC_normal_shift_bias_px=draws.mean(0).tolist(),support=proof,method='full covariance recovered from parent supported W, select covariance coordinates, re-whiten; never slice W columns'))
    n=len(selected);uv=np.asarray([x['uv']for x in selected]);extra=dict(uv=uv,native_pts=np.asarray([x['row']['native_pts']for x in selected]),shot_index=np.asarray([int(x['row']['shot_id'][1:])-1 for x in selected]),kind=np.zeros(n),material_index=np.asarray([x['row']['joint_material_index']for x in selected]),path=np.zeros(n),admission=np.ones(n),fit_role=np.ones(n),E=np.full(n,2.),station=np.full(n,.5),row=uv[:,1],alpha=np.asarray([x['alpha']for x in selected]),source_sigma=np.ones(n),profile_normal=np.asarray([x['normal']for x in selected]),query_uv=uv,alpha_bounds=np.asarray([x['bounds']for x in selected]),stroke_pair_index=np.asarray([x['pair_index']for x in selected]),stroke_source_side=np.asarray([x['side']for x in selected]));arrays={}
    for key in set(A)|{'stroke_pair_index','stroke_source_side'}:
        a=A.get(key,np.full(N0,-1,dtype='<i4'if key=='stroke_pair_index'else'i1'))
        if key not in extra:raise ValueError('missing explicit flank initializer for inherited array '+key)
        b=np.concatenate((a,np.asarray(extra[key],dtype=a.dtype)));f=out/(key+'.bin');b.tofile(f)
        if b[:N0].tobytes()!=a.tobytes():raise ValueError('immutable source/auxiliary prefix changed')
        arrays[key]=dict(file=str(f),sha256=file_sha256(f),dtype=b.dtype.str,shape=list(b.shape),layout='C',byte_order='little')
    oldgroups=[{**g,'W':{**g['W'],'file':str(locate(g['W']))}}for g in p['groups']];rank=p['joint_statistics_rank']+newrank;newrefs=[dict(file=str(f),sha256=file_sha256(f))for f in [gate/'manifest.json',gate/'approved_pairs.json',gate/'factor_rows.jsonl.gz',registry_path]]
    result={**p,'row_count':N0+n,'arrays':arrays,'groups':oldgroups+newgroups,'active_rows':p['active_rows']+n,'joint_statistics_rank':rank,'new_source_statistics_rank':rank-p['current_original_W_rank'],'source_dataset_ids':p['source_dataset_ids']+['actual_paint_flank_'+x['row']['shot_id']for x in selected],'source_row_ids':p['source_row_ids']+[x['row']['source_row_id']for x in selected],'source_family_ids':p['source_family_ids']+[x['row']['source_family_id']for x in selected],'stroke_pairs':p.get('stroke_pairs',[])+pairs,'queries':queries,'allowed_paths':allowed,'newly_appended_source_start':N0,'recapture_rows':p['recapture_rows']+list(range(N0,N0+n)),'source_manifests':p['source_manifests']+newrefs,'current_source_qualification_refs':p.get('current_source_qualification_refs',[])+newrefs,'parent_qualified_source_packet':dict(file=str(src),sha256=file_sha256(src)),'new_source_statistics_status':'mixed explicit geometric initialization for unknown-noise manual/board sources plus source-qualified conditional native RGB two-flank covariance; source modes remain separately declared','qualified_pair_covariance_marginal_proof':dict(file=str(out/'covariance_marginal_proofs.json'),sha256='pending'),'new_two_flank_conditional_statistics_rank':newrank,'old_midpoint_corresponding_rows_inactive':old_midpoints,'conditional_pair_source_noise_model_is_total_physical_calibration':False,'old_source_queries_or_W_changed':False}
    write_json(out/'covariance_marginal_proofs.json',proofs);result['qualified_pair_covariance_marginal_proof']['sha256']=file_sha256(out/'covariance_marginal_proofs.json');write_json(out/'typed_source_pairs.json',pairs);write_packet(out/'packet.json',result)
    report=dict(schema='rinkfit.qualified_two_flank_covariance_packet.v1',parent_raw_rows=N0,raw_rows=N0+n,appended_flank_rows=n,actual_source_pairs=len(pairs),active_rows=result['active_rows'],original_S01_rank=p['current_original_W_rank'],new_conditional_rank=newrank,joint_statistics_rank=rank,original_raw_source_and_four_aux_prefix_bit_preserved=True,old_midpoint_rows_inactive=old_midpoints,tangent_conditions_from_actual_finite_queries_not_added_precision=True,source_zero_or_positive_width_not_an_E_tolerance=True,formal_full_noise_joint_optimum_claimed=False,packet_sha256=file_sha256(out/'packet.json'),typed_source_pairs=dict(file=str(out/'typed_source_pairs.json'),sha256=file_sha256(out/'typed_source_pairs.json')))
    write_json(out/'report.json',report);return report
