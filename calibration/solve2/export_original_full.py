"""Self-contained current S01 input for the restored original d78 route.

This is a data adapter. It retains every raw row, the complete raw Query and
every delivered group matrix. Numerical scope changes are explicit, and a
whole statistical factor is selected only when all its support is admitted.
No source production, source qualification or optimizer is run here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'calibration/solve2'))
sys.path.insert(0,str(ROOT/'calibration/rinkfit/src'))
from export_original_ad import binary,checked,save_json,sha
from original_full_model import register_full_model
from rinksolve.preparation import load_prepared
from rinksolve.observations import observation_id,split_observation_id
from rinkfit_start.semantic_resolution import load_semantics

SCHEMA='rinkfit.current_full_S01_source_adapter.v1'
PREFIX_ROWS=182017


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def array(desc,base=None):
    path=Path(desc.get('file',desc.get('path','')))
    if not path.is_absolute():path=Path(base)/path
    a=np.fromfile(path,desc['dtype']).reshape(desc['shape'])
    return a


def advertisement(label):
    match=re.match(r'^Ad(\d+)(?:[.:/]|$)',str(label))
    return 0 if match is None else int(match.group(1))


def owners(prepared):
    registry=prepared.manifest['curve_registry']
    ids=np.asarray(prepared.arrays['curve_index'],dtype=np.int64)
    result=np.zeros(prepared.row_count,dtype=np.int8)
    legal=ids>=0
    result[legal]=np.asarray([advertisement(v['label'])for v in registry],np.int8)[ids[legal]]
    return result


def restore_scope(prepared,policy):
    """Remove only the temporary ads-stage exclusion from original non-Ad rows."""
    owner=owners(prepared)
    quality=np.array(prepared.arrays['exclusion_bits'],copy=True)
    original=np.arange(prepared.row_count)<PREFIX_ROWS
    restore=original&(owner==0)&(np.asarray(prepared.arrays['curve_index'])>=0)&(quality==512)
    quality[restore]=0
    active=(quality&np.uint16(65535^16))==0
    records=[]
    for key,spec in policy.get('advertisement_scopes',{}).items():
        ad=int(key.removeprefix('Ad'))
        rows=np.asarray(np.load(spec['keep_prepared_rows'],allow_pickle=False),dtype=np.int64)
        base=rows[rows<prepared.row_count]
        if rows.ndim!=1 or len(np.unique(rows))!=len(rows) or np.any(base<0) or np.any(owner[base]!=ad):
            raise ValueError('Current reviewed source scope crosses advertisement ownership: '+key)
        if np.any(~active[base]):
            raise ValueError('Current scope attempts to revive a prior withdrawn source: '+key)
        pending=np.flatnonzero(active&(owner==ad)&~np.isin(np.arange(prepared.row_count),base))
        active[pending]=False
        records.append(dict(advertisement=ad,keep_prepared_rows=str(Path(spec['keep_prepared_rows']).resolve()),
                            keep_rows=len(base),pending_rows=len(pending),reason=spec['reason'],
                            raw_evidence_retained=True,permanent_rejection=False))
    return quality,owner,active,dict(nonadvertisement_ads_stage_rows_restored=int(restore.sum()),
        only_exact_exclusion_512_cleared=True,other_source_exclusions_unchanged=True,scoped_advertisements=records)


def native_paths(model):
    """Stable current curve identities; original non-Ad branches retain d78 IDs."""
    paths={};variants=[];by_id={}
    legacy=[i for i,m in enumerate(model['materials'])if not m.get('current_source_curve',False)]
    # The complete-current adapter appends exactly the original 171 materials.
    if len(legacy)!=171:
        raise ValueError('Complete native input must retain the original 171 material definitions')
    for mi,mat in enumerate(model['materials']):
        for pi,path in enumerate(mat['paths']):
            variant=str(path['variant_id'])
            if variant not in by_id:by_id[variant]=len(variants);variants.append(variant)
            if mat.get('current_source_curve',False):
                cid=path.get('current_curve_id',path.get('id'))
                if not cid or cid in paths:raise ValueError('Duplicate or missing delivered current path identity')
                paths[cid]=(mi,pi,by_id[variant])
    for original_mi,mi in enumerate(legacy):
        mat=model['materials'][mi]
        if mat.get('unused_legacy_witness',False):continue
        for pi,path in enumerate(mat['paths']):
            cid=f'd78_material_{original_mi:03}/path{pi}'
            if cid in paths:raise ValueError('Current/non-Ad path identity collision')
            paths[cid]=(mi,pi,by_id[str(path['variant_id'])])
    return paths,variants,legacy


def shared_features(model,feature_prepared):
    """Install the existing five shared C stations without per-frame sliding."""
    report=feature_prepared.manifest['new_fixed_intersections']
    landmarks=feature_prepared.manifest['new_intersection_landmarks']
    observations=feature_prepared.manifest['new_intersection_observations']
    if not report.get('active',False) or len(landmarks)!=5 or len(observations)!=8:
        raise ValueError('Current full source must retain the delivered five C features/eight observations')
    paths,_,_=native_paths(model)
    names={v['name']for v in model['blocks']}
    bindings={}
    for item in landmarks:
        if item['position_kind']!='shared_curve_point' or item['station_is_free_per_observation'] or item['station_bounds']!=[0,1]:
            raise ValueError('The delivered C feature is not a shared finite material position')
        mi,pi,_=paths[item['curve_id']]
        parent=model['materials'][mi]['paths'][pi]
        if parent['kind']!=0 or item['station_variable_block_id']in names:
            raise ValueError('C shared-station parent or parameter identity changed')
        bi=len(model['blocks']);names.add(item['station_variable_block_id'])
        model['blocks'].append(dict(name=item['station_variable_block_id'],values=[item['station_seed']],lower=[0.],upper=[1.],scales=[1.],constant=False))
        path=deepcopy(parent)
        path.update(id=item['id'],variant_id=item['id'],shared_station_block=bi,physical_point=True,
                    current_curve_id=item['id'],shared_feature_parent_curve_id=item['curve_id'])
        model['materials'].append(dict(label=item['id'],paths=[path],current_source_curve=True,shared_material_feature=True))
        bindings[item['id']]=(len(model['materials'])-1,item,bi)
    return observations,bindings,report


def append_ad7_corners(model,prepared,active_policy):
    paths,_,_=native_paths(model)
    features={v['id']:v for v in prepared.manifest.get('ad7_fixed_source_features',[])}
    records=[]
    scope=active_policy.get('advertisement_scopes',{}).get('Ad7')
    selected=None if scope is None else set(np.load(scope['keep_prepared_rows'],allow_pickle=False).tolist())
    for i,obs in enumerate(prepared.manifest.get('ad7_fixed_observations',[])):
        feature=features[obs['fixed_feature_id']]
        if feature['station_is_optimizer_variable'] or obs.get('per_observation_world_station',False):
            raise ValueError('Fixed artwork corners cannot acquire local station parameters')
        mi,pi,_=paths[feature['curve_id']];parent=model['materials'][mi]['paths'][pi]
        p=np.asarray(parent['points'],float);z=feature['fixed_station']*max(0,len(p)-1)
        j=min(int(np.floor(z)),max(0,len(p)-2));point=p[0]if len(p)==1 else p[j]+(z-j)*(p[j+1]-p[j])
        cid='fixed_corner/'+feature['id']
        found=next((k for k,m in enumerate(model['materials'])if m.get('fixed_corner_identity')==cid),None)
        if found is None:
            path=deepcopy(parent);path.update(id=cid,variant_id=cid,current_curve_id=cid,points=[point.tolist()],physical_point=True,closed=False)
            found=len(model['materials']);model['materials'].append(dict(label=feature['id'],paths=[path],current_source_curve=True,fixed_corner_identity=cid))
        records.append(dict(**obs,source_full_row=prepared.row_count+i,material=found,path=0,curve_id=cid,
            active=selected is None or prepared.row_count+i in selected,station=0.,kind=1,
            stable_id=observation_id('ad7_fixed_material_corners',int(obs['native_pts']),obs['id'])))
    return records


def statistical_partition(prepared,active,quality):
    """Select complete current original matrices; no covariance reconstruction."""
    stat=np.zeros(len(active),np.uint8);groups=[];members=[];weights=[];withdrawn=[];rank=0
    for i,meta in enumerate(prepared.manifest['groups']):
        factor=prepared.source_group(i);rr=np.asarray(factor.rows,dtype=np.int64)
        W=np.asarray(factor.matrix,dtype='<f8')
        if len(W)==0:continue
        if not np.all(active[rr])or np.any(quality[rr]!=0):
            if np.any(active[rr]):withdrawn.append(dict(id=meta['id'],rank=len(W),member_count=len(rr),constant_support_not_truncated=True))
            continue
        if np.any(stat[rr]):raise ValueError('A source is counted by two current positive W factors')
        stat[rr]=1
        groups.append(dict(group_id=meta['id'],rank=len(W),member_start=len(members),member_count=len(rr),
            weight_start=len(weights),matrix_shape=list(W.shape),matrix_sha256=hashlib.sha256(W.tobytes()).hexdigest(),
            native_pts=int(prepared.arrays['native_pts'][rr[0]]),frame=int(1+(prepared.arrays['native_pts'][rr[0]]-256256)//256),
            covariance_origin=meta.get('covariance_origin','unchanged current complete factor'),
            one_statistical_measurement_per_original_group=True,source_prepared_kind=meta['statistic_kind']))
        members.extend(rr.tolist());weights.extend(W.ravel().tolist());rank+=len(W)
    independent=np.asarray(prepared.arrays['independent_source_rows'],dtype=np.int64)
    independent=independent[active[independent]&(quality[independent]==0)]
    if np.any(stat[independent])or len(np.unique(independent))!=len(independent):raise ValueError('Independent source/group support overlaps')
    sigma=np.asarray(prepared.arrays['source_sigma'])[independent]
    if not np.isfinite(sigma).all()or np.any(sigma<=0):raise ValueError('A current independent source lost its original sigma')
    stat[independent]=1;rank+=2*len(independent)
    return stat,groups,members,weights,independent,rank,withdrawn


def apply_auxiliary_overlays(state,model,curve_ids,ids,overlays):
    used=[];id_to_row={v:i for i,v in enumerate(ids)}
    for overlay in overlays:
        source=Path(overlay['input']).resolve();full=Path(overlay['complete']).resolve()
        packet=read(source/'packet/manifest.json');sem=read(source/'semantic/manifest.json');oldmodel=read(full/'model.json')
        tables=read(packet['string_tables']['file']);oldids=tables['stable_observation_ids']
        oldmaterial=array(sem['resolved_material_index']);oldbranch=np.fromfile(full/'path.u32','<u4')
        values={k:np.fromfile(full/name,dt)for k,name,dt in [('station','station.f64','<f8'),('row','initial_row.f64','<f8'),('alpha','initial_alpha.f64','<f8')]}
        count=0;changed=[]
        for j,sid in enumerate(oldids):
            if sid not in id_to_row:continue
            i=id_to_row[sid];path=oldmodel['materials'][int(oldmaterial[j])]['paths'][int(oldbranch[j])]
            cid=path.get('current_curve_id',path['id'])
            if cid!=curve_ids[i]:changed.append(i);continue
            for key in values:state[key][i]=values[key][j]
            count+=1
        used.append(dict(input=str(source),complete=str(full),retained_current_valid_auxiliary_rows=count,
            changed_binding_rows_not_overwritten=changed,physical_model_values_not_imported=True))
    return used


def export(prepared_directory,native_model_path,original_input_path,feature_prepared_directory,output,*,data_root,policy_path=None):
    out=checked(output);out.relative_to(Path(data_root).resolve())
    if out.exists():raise FileExistsError('Refuse to overwrite a current full input: '+str(out))
    prepared=load_prepared(prepared_directory,data_root=data_root)
    features=load_prepared(feature_prepared_directory,data_root=data_root)
    policy={}if policy_path is None else read(policy_path)
    model=read(native_model_path);original=read(original_input_path)
    quality,owner,active,scope_report=restore_scope(prepared,policy)
    registry=prepared.manifest['curve_registry'];index=np.asarray(prepared.arrays['curve_index'],np.int64)
    curve_ids=np.full(prepared.row_count,'',dtype=object)
    legal=index>=0;curve_ids[legal]=np.asarray([v['id']for v in registry],dtype=object)[index[legal]]
    binding_changes=[]
    for delta_path in policy.get('specific_binding_deltas',[]):
        delta=read(delta_path)
        for item in delta['corrections']:
            i=int(item['prepared_row'])
            sid=prepared.observation_ids([i])[0]
            sid=sid.decode('utf8')if isinstance(sid,(bytes,np.bytes_))else str(sid)
            if sid!=item['stable_id']or int(prepared.arrays['native_pts'][i])!=item['native_pts']or prepared.arrays['immutable_uv'][i].tolist()!=item['uv']or curve_ids[i]!=item['original_curve_id']:
                raise ValueError('A reviewed current per-source binding delta differs from the delivered raw identity')
            curve_ids[i]=item['target_curve_id']
            binding_changes.append(dict(prepared_row=i,stable_id=sid,previous_curve_id=item['original_curve_id'],
                current_curve_id=item['target_curve_id'],reviewed_delta=str(Path(delta_path).resolve()),
                numerical_station_seed_reset_only=True,raw_source_and_W_unchanged=True))
    paths,_,legacy=native_paths(model)
    missing=[]
    for cid in dict.fromkeys(curve_ids[active]):
        if cid in paths:continue
        rr=np.flatnonzero(active&(curve_ids==cid))
        missing.append(dict(curve_id=cid,rows=len(rr),advertisement=int(owner[rr[0]]),
            first_full_rows=rr[:10].tolist(),reason='current finite source identity absent after a delivered geometry replacement',
            raw_UV_ID_PTS_E_sigma_and_complete_W_retained=True))
        active[rr]=False
    observation_features,feature_bindings,feature_contract=shared_features(model,features)
    corner_records=append_ad7_corners(model,prepared,policy)
    paths,variant_ids,legacy=native_paths(model)
    oldmat=array(original['source_arrays']['material_index']).astype(int)
    original_branch=np.fromfile(Path(data_root)/'runs/solve2_review/shot1_runs/d78_path.u32','<u4')
    N=prepared.row_count
    material=np.zeros(N,dtype='<u4');branch=np.zeros(N,dtype='<u4');variant=np.zeros(N,dtype='<u4')
    unresolved_inactive=[]
    for cid in dict.fromkeys(curve_ids):
        rr=np.flatnonzero(curve_ids==cid)
        if cid in paths:
            mi,pi,vi=paths[cid];material[rr]=mi;branch[rr]=pi;variant[rr]=vi
        else:
            if np.any(active[rr]):raise AssertionError('Active source without delivered current path')
            # Inactive rows retain their prior curve identity as evidence. Their
            # legal ABI addresses are never evaluated as a current association.
            prefix=rr[rr<PREFIX_ROWS]
            for i in prefix:
                mi=legacy[oldmat[i]];pi=int(original_branch[i]);material[i]=mi;branch[i]=pi
                variant[i]=variant_ids.index(model['materials'][mi]['paths'][pi]['variant_id'])
            remainder=rr[rr>=PREFIX_ROWS]
            if len(remainder):unresolved_inactive.extend(remainder.tolist())
    # Zero is a legal inactive-only ABI address, not a new material claim.
    if unresolved_inactive:
        path=model['materials'][0]['paths'][0];variant[unresolved_inactive]=variant_ids.index(path['variant_id'])
    pts=np.array(prepared.arrays['native_pts'],dtype='<i8');uv=np.array(prepared.arrays['immutable_uv'],dtype='<f8')
    budget=np.array(prepared.arrays['engineering_budget_px'],dtype='<f8');kind=np.array(prepared.arrays['semantic_kind'],dtype='u1')
    sigma=np.array(prepared.arrays['source_sigma'],dtype='<f8');normal=np.array(prepared.arrays['profile_normal'],dtype='<f8')
    stable=[v.decode('utf8')if isinstance(v,(bytes,np.bytes_))else str(v)for v in prepared.observation_ids()]
    state=dict(station=np.array(prepared.arrays['station_seed'],dtype='<f8'),branch=branch,
               row=np.array(prepared.arrays['row_auxiliary_seed'],dtype='<f8'),alpha=np.array(prepared.arrays['profile_alpha_seed'],dtype='<f8'))
    for item in binding_changes:state['station'][item['prepared_row']]=.5
    unchanged=np.flatnonzero((np.arange(N)<PREFIX_ROWS)&(owner==0)&legal)
    resolved_original=array(original['resolved_material_index']).astype(int)
    for i in unchanged:
        mi=legacy[resolved_original[i]];pi=int(original_branch[i]);path=model['materials'][mi]['paths'][pi]
        material[i]=mi;state['branch'][i]=pi;variant[i]=variant_ids.index(path['variant_id'])
        curve_ids[i]=f'd78_material_{resolved_original[i]:03}/path{pi}'
    olddir=Path(data_root)/'runs/solve2_review/shot1_runs'
    for key,name,dt in [('station','d78_station.f64','<f8'),('row','d78_initial_row.f64','<f8'),('alpha','d78_initial_alpha.f64','<f8')]:
        values=np.fromfile(olddir/name,dt)
        if len(values)!=PREFIX_ROWS:raise ValueError('Current original complete-state row count differs')
        state[key][unchanged]=values[unchanged]
    source_full=np.arange(N,dtype='<u8');supplements=[]
    records=corner_records+[
        dict(**obs,source_full_row=N+len(corner_records)+i,material=feature_bindings[obs['fixed_feature_id']][0],path=0,
             curve_id=obs['fixed_feature_id'],active=True,station=0.,kind=1,
             engineering_budget_px=feature_contract['engineering_target_px'],
             stable_id=observation_id('derived_fixed_ice_intersections',int(obs['native_pts']),obs['id']))
        for i,obs in enumerate(observation_features)]
    for obs in records:
        mi=int(obs['material']);path=model['materials'][mi]['paths'][0]
        material=np.r_[material,np.uint32(mi)];variant=np.r_[variant,np.uint32(variant_ids.index(path['variant_id']))]
        pts=np.r_[pts,np.int64(obs['native_pts'])];uv=np.vstack([uv,np.asarray(obs['uv_native_pixels'],float)])
        budget=np.r_[budget,float(obs['engineering_budget_px'])];kind=np.r_[kind,np.uint8(obs['kind'])]
        sigma=np.r_[sigma,np.nan];normal=np.vstack([normal,[0.,0.]])
        active=np.r_[active,bool(obs['active'])];quality=np.r_[quality,np.uint16(0 if obs['active']else 2048)]
        curve_ids=np.append(curve_ids,obs['curve_id']);source_full=np.r_[source_full,np.uint64(obs['source_full_row'])]
        stable.append(obs['stable_id']);state['station']=np.r_[state['station'],float(obs['station'])]
        state['branch']=np.r_[state['branch'],np.uint32(0)];state['row']=np.r_[state['row'],float(obs['uv_native_pixels'][1])]
        state['alpha']=np.r_[state['alpha'],0.];supplements.append(obs)
    N=len(pts)
    if len(set(stable))!=N:raise ValueError('Duplicate full current stable observation identity')
    if not np.isfinite(np.r_[uv.ravel(),budget,state['station'],state['row'],state['alpha']]).all():raise ValueError('Nonfinite current source/complete numerical seed')
    if np.any((state['station']<0)|(state['station']>1))or np.any((state['row']<0)|(state['row']>1079)):raise ValueError('Current numerical seed is outside the original finite domain')
    overlay_report=apply_auxiliary_overlays(state,model,curve_ids,stable,policy.get('auxiliary_overlays',[]))
    stat,groups,members,weights,independent,rank,withdrawn=statistical_partition(prepared,active,quality)
    # Supplemental engineering observations have no invented source sigma/W.
    if np.any(stat[prepared.row_count:]):raise AssertionError('Supplemental point acquired a statistical weight')
    source_identity=hashlib.sha256(pts.tobytes()+uv.tobytes()+'\n'.join(stable).encode('utf8')).hexdigest()
    model['source_identity']=source_identity
    supported_ads={int(model['materials'][int(material[i])]['paths'][int(state['branch'][i])].get('advertisement_block',-1))for i in np.flatnonzero(active)}
    supported_geometry={int(model['materials'][int(material[i])]['paths'][int(state['branch'][i])].get('geometry_block',-1))for i in np.flatnonzero(active)}
    unobserved=[ad for ad in range(8,13)if next(j for j,b in enumerate(model['blocks'])if b['name']==f'ad/{ad}')not in supported_ads]
    unobserved_geometry=[b['name']for j,b in enumerate(model['blocks'])if b['name'].startswith('ice/')and j not in supported_geometry]
    model,registration=register_full_model(model,unobserved_artwork_ids=unobserved,unobserved_geometry_blocks=unobserved_geometry)
    out.mkdir(parents=True);pd=out/'packet';sd=out/'semantic';qd=out/'query'
    for directory in (pd,sd,qd):directory.mkdir()
    arrays={key:binary(pd,name,value,dt)for key,name,value,dt in [
        ('native_uv','uv.f64',uv,'<f8'),('source_rows','rows.u64',np.arange(N),'<u8'),('native_pts','pts.i64',pts,'<i8'),
        ('frame_index','frames.i32',1+(pts-256256)//256,'<i4'),('material_index','material.u32',material,'<u4'),
        ('point_id_index','point.u32',np.arange(N),'<u4'),('engineering_budget_px','budget.f64',budget,'<f8'),
        ('source_rows_full','source_rows_full.u64',source_full,'<u8')]}
    split=[split_observation_id(v)for v in stable]
    save_json(pd/'string_tables.json',dict(material_labels=[v['label']for v in model['materials']],source_point_ids=stable,
        stable_observation_ids=stable,original_source_ids=[v[2]for v in split],original_source_dataset_ids=[v[0]for v in split],
        raw_prepared_curve_ids=curve_ids.tolist(),unresolved_inactive_abi_address_rows=unresolved_inactive,
        unresolved_inactive_addresses_are_not_current_material_associations=True))
    sa={key:binary(sd,name,value,dt)for key,name,value,dt in [
        ('kind','kind.u8',kind,'|u1'),('variant_index','variant.u32',variant,'<u4'),('profile_normal','normal.f64',normal,'<f8'),
        ('query_uv','query_uv.f64',np.vstack([array(original['semantic_arrays']['query_uv']),np.zeros((N-PREFIX_ROWS,2))]),'<f8'),
        ('alpha_bounds','alpha_bounds.f64',np.vstack([array(original['semantic_arrays']['alpha_bounds']),np.zeros((N-PREFIX_ROWS,2))]),'<f8'),
        ('source_sigma','sigma.f64',sigma,'<f8'),('ordinary_rows','ordinary_rows.u64',independent,'<u8'),
        ('group_rows','group_rows.u64',members,'<u8'),('weights','weights.f64',weights,'<f8')]}
    (sd/'groups.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n'for v in groups),encoding='utf8')
    (sd/'runs.jsonl').write_text('',encoding='utf8')
    am=binary(sd,'active.u8',active,'|u1');sm=binary(sd,'statistical_active.u8',stat,'|u1')
    stats=dict(schema='rinkfit.active_statistics.v1',active_observations=int(active.sum()),statistical_observations=int(stat.sum()),
        formal_residual_rows=rank,original_group_count=len(groups),independent_rows=len(independent),
        active_source_mask_sha256=am['sha256'],statistical_active_mask_sha256=sm['sha256'],statistical_partition_exactly_once=True,
        budget_only_observations=int(active.sum()-stat.sum()),new_W_sigma_or_weight_invented=False)
    save_json(sd/'active_statistics.json',stats)
    scope=dict(schema='rinkfit.current_advertisement_input.v1',scope_id='S01_full',full_physical_solver=True,
        source_frame_count=1248,group_count=len(groups),independent_count=len(independent),formal_W_rank=rank,run_count=0,
        statistical_active_mask=sm,print_only_ad_ids=model['print_only_ad_ids'],unobserved_geometry_blocks=registration['unobserved_geometry_blocks_fixed'])
    model['current_input_scope']=scope;save_json(out/'model.json',model)
    semantic=dict(schema='rinkfit.semantic_packet.v1',observations=N,shot_id='s01',frames=1248,material_labels=len(model['materials']),
        semantic_counts=dict(finite_curve=int(np.sum(kind==0)),entity_corner=int(np.sum(kind==1)),true_profile=int(np.sum(kind==2))),
        arrays=sa,assets={name:dict(file=str(sd/name),sha256=sha(sd/name))for name in ('groups.jsonl','runs.jsonl')},
        variant_ids=variant_ids,resolved_variant_ids=variant_ids,resolved_material_index=arrays['material_index'],resolved_variant_index=sa['variant_index'],
        active_source_mask=am,statistical_active_mask=sm,active_observations=int(active.sum()),
        active_statistics=dict(file=str(sd/'active_statistics.json'),sha256=sha(sd/'active_statistics.json')),
        current_input_scope=scope,source_runs=dict(proven_runs=0),profiles=int(np.sum(kind==2)),statistics=stats,
        current_prepared_export=dict(schema=SCHEMA,prepared_manifest=str(Path(prepared_directory).resolve()/'manifest.json'),
            original_raw_rows=prepared.row_count,source_identity=source_identity,source_rows_full=arrays['source_rows_full'],
            source_preparation_or_qualification_replayed=False,current_complete_group_matrices_preserved=True))
    save_json(sd/'manifest.json',semantic)
    query=deepcopy(original['query']);query.update(schema='rinkfit.immutable_native_query_curve_packet.v1',profiles=961,
        semantic_manifest=dict(path=str(sd/'manifest.json'),sha256=sha(sd/'manifest.json')),
        rational_coefficients='ascending powers; pixel=(numerator_u, numerator_v)/denominator')
    query['certificates']['path']=query['certificates']['file']
    save_json(qd/'manifest.json',query)
    qrows=array(query['arrays']['source_rows']).astype(int)
    qa=binary(out,'query_initial_alpha.f64',state['alpha'][qrows],'<f8')
    save_json(out/'query_initialization.json',dict(query_packet_sha256=sha(qd/'manifest.json'),alpha=qa))
    for key,name,dt in [('station','station.f64','<f8'),('branch','path.u32','<u4'),('row','initial_row.f64','<f8'),('alpha','initial_alpha.f64','<f8')]:binary(out,name,state[key],dt)
    packet=dict(schema='rinkfit.event_packet.v1',observations=N,frame_count=1248,source_frame_count=1248,shot_id='s01',
        native_image_wh=[1920,1080],native_time_base=[1,12800],pts_origin=256256,arrays=arrays,
        string_tables=dict(file=str(pd/'string_tables.json'),sha256=sha(pd/'string_tables.json')),assets={},
        immutable_source_hashes={'observations.npz':source_identity},current_input_scope=scope,
        current_prepared_export=semantic['current_prepared_export'],
        semantic_packet=dict(manifest=str(sd/'manifest.json'),sha256=sha(sd/'manifest.json')),
        query_packet=dict(manifest=str(qd/'manifest.json'),sha256=sha(qd/'manifest.json')),
        event_model=dict(manifest=str(out/'model.json'),sha256=sha(out/'model.json')),initial_state_directory=str(out))
    save_json(pd/'manifest.json',packet)
    save_json(out/'initializer_manifest.json',dict(schema='rinkfit.current_source_numerical_seed.v1',
        semantic_manifest=dict(path=str(sd/'manifest.json'),sha256=sha(sd/'manifest.json')),
        files={name:dict(path=str(out/name),sha256=sha(out/name))for name in ('station.f64','path.u32','initial_row.f64','initial_alpha.f64')},
        seed_only_not_complete_finite_foot_optimization=True,use_original_full2_for_changed_geometry=True,
        unchanged_current_accepted_auxiliaries_preserved=overlay_report))
    save_json(out/'registration.json',registration)
    report=dict(schema=SCHEMA,packet=str(pd/'manifest.json'),semantic=str(sd/'manifest.json'),query=str(qd/'manifest.json'),
        model=str(out/'model.json'),raw_prepared_rows=prepared.row_count,raw_total_rows=N,active_rows=int(active.sum()),
        statistical_sources=int(stat.sum()),W_rank=rank,positive_W_groups=len(groups),independent_rows=len(independent),
        raw_Query_curves=961,active_Query_rows=int(np.sum(active&(kind==2))),shared_C_features=5,shared_C_observations=8,
        supplemental_Ad7_corners=len(corner_records),scope_restore=scope_report,unresolved_current_geometry_sources=missing,
        unchanged_nonadvertisement_original_complete_material_branch_and_four_aux_rows=len(unchanged),
        whole_positive_factors_withdrawn_because_support_is_pending=withdrawn,auxiliary_overlays=overlay_report,
        specific_current_binding_changes=binding_changes,
        input_model=str(Path(native_model_path).resolve()),source_policy=None if policy_path is None else str(Path(policy_path).resolve()),
        original_raw_prefix_UV_PTS_and_kind_bitexact=bool(np.array_equal(uv[:PREFIX_ROWS],array(original['source_arrays']['native_uv']))and np.array_equal(pts[:PREFIX_ROWS],array(original['source_arrays']['native_pts']))and np.array_equal(kind[:PREFIX_ROWS],array(original['semantic_arrays']['kind']))),
        source_qualification_replayed=False,optimizer_started=False,full_camera_ice_board_and_all_static_ad_objects_free=True,
        pending_visual_source_revisions_do_not_authorize_joint_Solve=True)
    save_json(out/'mapping.json',report)
    load_semantics(sd/'manifest.json',N,expected_sha256=sha(sd/'manifest.json'))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--native-model',required=True)
    p.add_argument('--original-input',required=True);p.add_argument('--feature-prepared',required=True)
    p.add_argument('--output',required=True);p.add_argument('--scope-policy');p.add_argument('--data-root',default='D:/CodexData/RinkSpace_Data')
    a=p.parse_args();report=export(a.prepared,a.native_model,a.original_input,a.feature_prepared,a.output,data_root=a.data_root,policy_path=a.scope_policy)
    print(json.dumps({k:v for k,v in report.items()if k not in ('whole_positive_factors_withdrawn_because_support_is_pending',)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
