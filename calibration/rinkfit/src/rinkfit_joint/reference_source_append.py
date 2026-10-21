"""Append immutable source chunks once, keeping unknown-noise engineering factors.

Bulk journals/profiles stay in their original closed chunks. The contiguous
solver arrays are produced once for the complete new batch, not per chunk.
"""
from pathlib import Path
from collections import defaultdict,Counter
import json,shutil
import numpy as np
from .io_utils import file_sha256,read_jsonl,write_json,object_sha256
from .packet_storage import write_packet


def extend_shared_point_material_binding(parent,packet,model_path,material_map,qualification):
    """Freeze newly qualified active point identities in the native binding ABI.

    Allowing unused model suffixes is not enough: JointPacket requires every
    active material to be inside its exact bound table. Old identities remain
    the same prefix; only source-qualified shared point identities extend it.
    """
    old=parent['material_binding'];n=len(old['materials'])
    receipts=[r for r in qualification.values()if r.get('schema') in ('rinkfit.static_ice_tracklet_point_receipt.v1','rinkfit.static_board_tracklet_point_receipt.v1')
        and r.get('new_material_suffix') and r['existing_material_index']>=n]
    if not receipts:return packet
    model=json.loads(Path(model_path).read_text(encoding='utf8'))
    suffix={r['existing_material_index']for r in receipts}
    if not old.get('allow_append_only_model_material_suffix') or suffix!=set(range(n,len(model['materials']))):
        raise ValueError('Active material binding extension must be the exact qualified shared-point suffix')
    table=[dict(index=i,label=m['label'],paths=[dict(index=j,id=p['id'],physical_point=bool(p.get('physical_point',False)))
        for j,p in enumerate(m['paths'])])for i,m in enumerate(model['materials'])]
    if table[:n]!=old['materials']:raise ValueError('Shared point binding changed inherited material identities')
    def ref(f):return dict(file=str(Path(f).resolve()),sha256=file_sha256(f))
    packet['material_binding']={**old,'materials':table,'model_material_prefix_count':len(table),
        'model_topology_ref':ref(model_path),'topological_identity_table_sha256':object_sha256(table),
        'additional_source_material_maps':old.get('additional_source_material_maps',[])+[ref(material_map)],
        'inherited_material_prefix_count':n,'new_active_point_materials_source_qualified':True}
    return packet


def append_constraint_source_chunks(parent_packet,closed_chunks,model_path,material_map,roles_file,output,*,finalize_packet=None,typed_entity_qualification=None,finalize_packet_with_parent=None):
    parent=Path(parent_packet).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('new immutable source chunk append required')
    p=json.loads(parent.read_text(encoding='utf8'));model=json.loads(Path(model_path).read_text(encoding='utf8'));mapping=json.loads(Path(material_map).read_text(encoding='utf8'))['semantic_to_material']
    table=p['material_binding']['materials'];actual=[{'index':i,'label':m['label'],'paths':[{'index':j,'id':q['id'],'physical_point':bool(q.get('physical_point',False))}for j,q in enumerate(m['paths'])]}for i,m in enumerate(model['materials'][:len(table)])]
    if actual!=table:raise ValueError('actual model changes inherited physical material topology')
    if any(g.get('statistic_kind')=='initialization_surrogate'for g in p['groups']):raise ValueError('engineering source extension expects no unknown-noise surrogate precision')
    cache={}
    def ref(f):
        f=Path(f).resolve();k=str(f)
        if k not in cache:cache[k]=file_sha256(f)
        return {'file':k,'sha256':cache[k]}
    def verify(d):
        f=Path(d.get('file',d.get('path')));f=f if f.is_absolute()else parent.parent/f
        if ref(f)['sha256']!=d['sha256']:raise ValueError('changed inherited source dependency: '+str(f))
        return f.resolve()
    from .typed_entity_append import load_qualification,validate_new_entity
    entity_qualification,entity_source_refs=load_qualification(typed_entity_qualification,verify)
    roles=json.loads(Path(roles_file).read_text(encoding='utf8'))['frame_roles']
    n0=int(p['row_count']);parent_pts_desc=p['arrays']['native_pts'];parent_pts=np.memmap(verify(parent_pts_desc),dtype=parent_pts_desc['dtype'],mode='r',shape=tuple(parent_pts_desc['shape']))
    seen={(rid,int(parent_pts[i]))for i,rid in enumerate(p['source_row_ids'])};keep=('source_row_id','source_family_id','shot_id','native_pts','source_uv','current_uv','normal','joint_material_id','joint_material_index','point_correspondence','source_geometry_eligible','statistical_objective_eligible','numeric_statistics_ready','solver_eligible','fit_role','frame_role')
    old_entity_rows=defaultdict(list)
    for i,rid in enumerate(p['source_row_ids']):
        if rid in entity_qualification:old_entity_rows[(rid,int(parent_pts[i]))].append(i)
    old_ad_desc=p['arrays']['admission'];old_ad=np.memmap(verify(old_ad_desc),dtype=old_ad_desc['dtype'],mode='r',shape=tuple(old_ad_desc['shape']))
    raw=[];raw_datasets=[];newactive=[];materials=[];entity_paths=[];chunk_refs=[];journals=[];families=defaultdict(list);family_refs={};frames={};shotcounts=Counter();entity_revisions=[]
    for chunk in closed_chunks:
        mf=verify(chunk);m=json.loads(mf.read_text(encoding='utf8'))
        if m.get('all_requested_frames_accounted')is not True:raise ValueError('only closed source chunks may be appended')
        adapter=m.get('schema')=='rinkfit.actual_source_gate_reference_adapter.v1'
        ledger=verify(m['source_rows_ref'])if adapter else mf.parent/'source_rows.jsonl.gz';geom=verify(m['geometry_rows_ref'])if adapter else mf.parent/'geometry_rows.jsonl.gz';refs=[ref(mf),ref(ledger),ref(geom)]
        if adapter:refs.append(ref(verify(m['source_gate_manifest'])))
        tag=m['source_dataset_tag']if adapter else mf.parent.parent.name+'.'+mf.parent.name
        chunk_refs.append({'manifest':refs[0],'source_rows':refs[1],'geometry_rows':refs[2]});journals.append(ref(ledger))
        eligible={x['source_row_id']:{k:x[k]for k in keep if k in x}for x in read_jsonl(geom)};used=set();count=0
        for x in read_jsonl(ledger):
            r={k:x[k]for k in keep if k in x};rid=r['source_row_id'];pts=int(r['native_pts']);key=(rid,pts)
            if key in seen:
                old_rows=old_entity_rows.get(key,[])
                if rid not in entity_qualification or not old_rows or np.any(old_ad[old_rows]!=0):raise ValueError('new source chunks overlap an existing active or unqualified row: '+rid)
                if any(x['source_row_id']==rid for x in raw):raise ValueError('new entity measurement version repeated in one append')
                entity_revisions.append(dict(source_row_id=rid,native_pts=pts,prior_inactive_rows=old_rows,new_row_index=n0+len(raw),old_raw_unchanged=True))
            seen.add(key);g=eligible.get(rid);ok=g is not None;count+=1
            measurement=r.get('current_uv')if r.get('current_uv')is not None else r.get('source_uv')
            if measurement is None and m.get('retain_unmeasured_candidates_in_source_journal_only')is True:
                if ok:raise ValueError('a source-qualified physical measurement cannot have no actual UV')
                continue
            if np.asarray(measurement).shape!=(2,)or not np.isfinite(measurement).all():raise ValueError('actual numerical row needs an explicit finite source measurement')
            desc=mapping.get(r.get('joint_material_id'),{});mi=int(desc.get('material_index',0))
            entity_path=None
            if ok:
                if not g.get('source_geometry_eligible')or g.get('numeric_statistics_ready')is not False or g.get('statistical_objective_eligible')is not False or g.get('solver_eligible')is not False:raise ValueError('unknown noise source qualification is not explicit')
                if any(r.get(k)!=g.get(k)for k in ('source_uv','current_uv','native_pts','joint_material_id')):raise ValueError('source chunk gate redirects actual mean/operator')
                if not desc.get('geometry_ready')or r.get('joint_material_index')!=mi or mi>=len(model['materials']):raise ValueError('new actual source has no frozen physical material binding')
                if mi>=len(table):
                    receipt=entity_qualification.get(rid,{})
                    if (receipt.get('schema') not in ('rinkfit.static_ice_tracklet_point_receipt.v1','rinkfit.static_board_tracklet_point_receipt.v1')
                        or not receipt.get('new_material_suffix')
                        or not p['material_binding'].get('allow_append_only_model_material_suffix')):
                        raise ValueError('New material suffix requires exact qualified shared physical point')
                if r.get('point_correspondence'):
                    if not any(q.get('role')=='qualified_current_entity_source_journal'and ref(ledger)['sha256']==q['sha256']for q in entity_source_refs):raise ValueError('Entity append ledger is not a source-qualified current journal')
                    # The original flags remain in the parent journal; only this
                    # source-qualified new current measurement version is appended.
                    full={**r,'measurement_operator_resolved':x.get('measurement_operator_resolved')}
                    if entity_qualification.get(rid,{}).get('schema')=='rinkfit.static_board_tracklet_point_receipt.v1':
                        # Its unknown shared-surface qualifier needs the actual
                        # current source ownership/track fields, not only the
                        # compact numerical packet row.
                        full=x
                    entity_path=validate_new_entity(full,entity_qualification.get(rid),model['materials'][mi],mi)
                if roles.get(str(pts),{}).get('role')!='development_candidate'or not(r.get('fit_role')=='development'or r.get('frame_role')=='development_candidate'):raise ValueError('protected/external exposure must not enter source engineering factors')
                family=(r['shot_id'],pts,r['source_family_id']);families[family].append(n0+len(raw));family_refs[family]=refs
                used.add(rid);frames[pts]=r['shot_id'];shotcounts[r['shot_id']]+=1
            raw.append(r);newactive.append(ok);materials.append(mi);entity_paths.append(entity_path);raw_datasets.append('actual_geometry:'+tag+':'+r['shot_id'])
        if used!=set(eligible):raise ValueError('geometry source rows outside complete immutable chunk ledger')
        if count!=m['actual_source_rows']:raise ValueError('closed chunk actual row accounting mismatch')
    out.mkdir(parents=True);n=len(raw);q=np.asarray([r.get('current_uv')if r.get('current_uv')is not None else r['source_uv']for r in raw],float);adnew=np.asarray(newactive,np.uint8)
    extra=dict(uv=q,native_pts=np.asarray([r['native_pts']for r in raw]),shot_index=np.asarray([int(r['shot_id'][1:])-1 for r in raw]),kind=np.asarray([1 if j is not None else 0 for j in entity_paths]),material_index=np.asarray(materials),path=np.asarray([j if j is not None else 0 for j in entity_paths]),admission=adnew,fit_role=np.ones(n),E=np.full(n,2.),station=np.asarray([0. if j is not None else .5 for j in entity_paths]),row=np.where((q[:,1]>=0)&(q[:,1]<=1079),q[:,1],539.5),alpha=np.zeros(n),source_sigma=np.ones(n),profile_normal=np.asarray([r.get('normal',[0.,0.])for r in raw]),query_uv=q,alpha_bounds=np.zeros((n,2)),stroke_pair_index=np.full(n,-1),stroke_source_side=np.full(n,-1))
    arrays={}
    for k,d in p['arrays'].items():
        if k not in extra:raise ValueError('source adapter lacks new array '+k)
        source=verify(d);f=out/(k+'.bin');shutil.copyfile(source,f)
        with f.open('ab')as stream:np.asarray(extra[k],dtype=d['dtype']).tofile(stream)
        arrays[k]={**d,**ref(f),'shape':[n0+n,*d['shape'][1:]]}
    groups=[]
    for g in p['groups']:
        if g.get('statistic_kind')=='constraint_only_unknown_noise':
            if g['rank']!=0 or g['W']!=[]:raise ValueError('unknown-noise group carries invented precision')
            groups.append(g.copy())
        else:groups.append({**g,'W':{**g['W'],'file':str(verify(g['W']))}})
    allowed=dict(p.get('allowed_paths',{}))
    for (shot,pts,family),rows in families.items():
        groups.append({'id':'constraint_source:'+family,'source_group_id':family,'rows':rows,'rank':0,'W':[],
            'statistic_kind':'constraint_only_unknown_noise','unknown_noise_source_refs':family_refs[(shot,pts,family)],
            'unknown_noise_source_has_statistical_precision':False,'E_RS_Query_and_finite_material_constraints_retained':True})
        for i in rows:allowed[str(i)]=[entity_paths[i-n0]]if entity_paths[i-n0]is not None else list(range(len(model['materials'][materials[i-n0]]['paths'])))
    oldad=p['arrays']['admission'];oldrole=p['arrays']['fit_role'];a=np.memmap(verify(oldad),dtype=oldad['dtype'],mode='r',shape=tuple(oldad['shape']));b=np.memmap(verify(oldrole),dtype=oldrole['dtype'],mode='r',shape=tuple(oldrole['shape']));active=np.r_[(a==1)&(b==1),adnew.astype(bool)]
    partition=np.zeros(n0+n,np.int16);partition[np.asarray(p['independent_rows'],int)]+=1
    for g in groups:partition[g['rows']]+=1
    if not np.array_equal(partition,active.astype(np.int16)):raise ValueError('all admitted old/new sources must keep exact statistical-or-constraint partition')
    index={'schema':'rinkfit.closed_source_chunk_reference_index.v1','closed_chunks':chunk_refs,'frames_with_actual_source':len(frames),'actual_source_rows':n,'active_source_rows':int(adnew.sum()),'by_shot':dict(shotcounts),'source_payload_copied':False,'original_chunk_source_profile_and_RGB_receipts_preserved':True}
    index['new_actual_entity_measurement_revisions']=entity_revisions;index['new_entity_qualification_refs']=entity_source_refs
    write_json(out/'source_chunk_index.json',index)
    result={**p,'row_count':n0+n,'arrays':arrays,'groups':groups,'source_row_ids':p['source_row_ids']+[r['source_row_id']for r in raw],
        'source_dataset_ids':p['source_dataset_ids']+raw_datasets,'source_family_ids':p.get('source_family_ids',[None]*n0)+[r['source_family_id']for r in raw],
        'allowed_paths':allowed,'active_rows':int(active.sum()),'recapture_rows':np.flatnonzero(active&(np.arange(n0+n)>=p['new_source_start'])).tolist(),
        'source_family_journal_refs':p.get('source_family_journal_refs',[])+journals,
        'source_manifests':p['source_manifests']+[ref(out/'source_chunk_index.json'),ref(material_map),ref(model_path),ref(roles_file)]+entity_source_refs,
        'constraint_only_unknown_noise_rows':sum(len(g['rows'])for g in groups if g.get('statistic_kind')=='constraint_only_unknown_noise'),
        'constraint_only_unknown_noise_groups':sum(g.get('statistic_kind')=='constraint_only_unknown_noise'for g in groups),
        'additional_source_chunk_index':ref(out/'source_chunk_index.json'),'parent_all_source_constraint_packet':ref(parent),
        'source_noise_statistics_supplied_for_appended_rows':False,'unknown_noise_source_rows_keep_engineering_constraints':True,
        'unknown_noise_source_statistical_precision_supplied':False,'all_new_source_nuisance_requires_fresh_physical_capture':True,
        'formal_new_source_statistical_optimum_claimed':False,'physical_total_error_accuracy_claimed':False}
    result=extend_shared_point_material_binding(p,result,model_path,material_map,entity_qualification)
    if finalize_packet is not None and finalize_packet_with_parent is not None:raise ValueError('Choose one source finalizer')
    if finalize_packet_with_parent is not None:
        # Reuse the already parsed immutable parent; do not materialize the
        # 904k source-ID registry for a second time in one append process.
        result=finalize_packet_with_parent(result,out,p)
    elif finalize_packet is not None:
        result=finalize_packet(result,out)
    final_ad=np.memmap(result['arrays']['admission']['file'],dtype=result['arrays']['admission']['dtype'],mode='r',shape=(n0+n,));final_role=np.memmap(result['arrays']['fit_role']['file'],dtype=result['arrays']['fit_role']['dtype'],mode='r',shape=(n0+n,));final_partition=np.zeros(n0+n,np.int16)
    final_partition[np.asarray(result['independent_rows'],int)]+=1
    for g in result['groups']:final_partition[g['rows']]+=1
    if not np.array_equal(final_partition,((final_ad==1)&(final_role==1)).astype(np.int16)):raise ValueError('final source statistical/engineering partition must be exact')
    write_packet(out/'packet.json',result)
    report={'schema':'rinkfit.reference_chunk_unknown_noise_constraint_append.v1','packet':ref(out/'packet.json'),'parent_packet':ref(parent),'raw_rows':n0+n,'appended_raw_rows':n,'appended_active_unknown_noise_source_rows':int(adnew.sum()),'active_rows':int(result['active_rows']),'new_actual_source_families':len(families),'frames_with_actual_source':len(frames),'chunks':len(chunk_refs),'by_shot':dict(shotcounts),'statistical_rank':p['joint_statistics_rank'],'new_statistical_precision_invented':False,'new_full_arrays_produced_once_for_complete_batch':True,'source_payload_duplicated':False,'source_alias_canonicalization_required':True}
    write_json(out/'report.json',report);return report
