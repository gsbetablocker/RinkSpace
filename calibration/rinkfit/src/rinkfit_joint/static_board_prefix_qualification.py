"""Strict qualified static-board texture appends; locations remain shared latents.

V3/V4/V5 issuers and the current native dispatch remain frozen. This independent
V6 entry checks an actual existing parent plus the exact reviewed native source
subset. It grants neither known world-corner identity nor noise precision.
"""
from pathlib import Path
import json
import numpy as np
from .io_utils import file_sha256,read_jsonl,write_json,object_sha256
from .board_source_qualification_guard import _load_ref,FAMILY_PREFIXES,BOUND_ARRAYS
from .static_ice_prefix_qualification import _array,_prefix_arrays

SCHEMA='rinkfit.board_material_fit_qualification.v6_static_board_prefix'
BUNDLE_SCHEMA='rinkfit.native_static_board_tracklet_qualified_bundle.v1'
GATE_SCHEMA='rinkfit.static_board_tracklet_point_qualification.v1'
RECEIPT_SCHEMA='rinkfit.static_board_tracklet_point_receipt.v1'
OPERATOR='native_static_board_RGB_texture_2D_track_v1'
SOURCE_SHA='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'
AUX_ARRAYS=('station','row','alpha','path')


def _ref(path):
    path=Path(path).resolve();return dict(file=str(path),sha256=file_sha256(path))


EXECUTED_MODULE_REF=_ref(__file__)


def _json(ref,base,seen):
    path=_load_ref(ref,base,seen);return path,json.loads(path.read_text(encoding='utf8'))


def _source_refs(document,base,seen):
    if isinstance(document,dict):
        if ('file'in document or 'path'in document)and 'sha256'in document:_load_ref(document,base,seen)
        else:
            for value in document.values():_source_refs(value,base,seen)
    elif isinstance(document,list):
        for value in document:_source_refs(value,base,seen)


def _model_table(model):
    return [dict(index=i,label=m['label'],paths=[dict(index=j,id=p['id'],physical_point=bool(p.get('physical_point',False)))for j,p in enumerate(m['paths'])])for i,m in enumerate(model['materials'])]


def _shared_board_point(model,mi,tid,instance,old_count):
    if mi<old_count:raise ValueError('Static board texture cannot reuse a known existing world point')
    material=model['materials'][mi]
    if len(material['paths'])!=1:raise ValueError('Static board texture needs one shared point path')
    path=material['paths'][0];ad=7 if instance=='Ad2'else 8 if instance=='Ad3'else -1
    if (path.get('kind')!=2 or path.get('points')!=[[0,0]]or path.get('physical_point')is not True
        or path.get('advertisement_block')!=ad or path.get('geometry_indices')or path.get('geometry_offset',0)!=0):raise ValueError('Static board point is not on the existing unfolded advertisement surface')
    block_index=int(path['geometry_block']);block=model['blocks'][block_index];values=np.asarray(block['values'],float)
    if (values.shape!=(6,)or not np.isfinite(values).all()or values[[0,1,3,4]].tolist()!=[1,0,0,1]
        or block.get('fixed_indices')!=[0,1,3,4]or block.get('constant')is not False
        or any(block['lower'][k]!=0 or block['upper'][k]!=1 or not 0<=values[k]<=1 for k in (2,5))):raise ValueError('Static board latent is not exactly two bounded shared material coordinates')
    if material.get('physical_track_id',tid)!=tid or len(block['lower'])!=6 or len(block['upper'])!=6:raise ValueError('Static board latent material identity/ABI differs')
    return block_index,path


def _prefix(packet,parent,path,parentpath,cert,seen):
    n=int(parent['row_count']);m=int(packet['row_count'])
    if n!=cert['original_raw_rows']or m!=cert['raw_rows']or m<=n or m-n!=cert['appended_raw_rows']or parent['shot_ids']!=packet['shot_ids']:raise ValueError('Static board append raw prefix/scope differs')
    for root in (parentpath.parent,path.parent):
        for name in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json','SOURCE_IDENTITY_EVIDENCE_RECALL.json'):
            if(root/name).exists():raise ValueError('Static board inherited source evidence recalled')
    for key in ('source_row_ids','source_dataset_ids','source_family_ids'):
        if len(packet[key])!=m or packet[key][:n]!=parent[key]:raise ValueError('Static board redirects inherited source registry')
    for key in ('queries','independent_rows','independent_statistic_kind','stroke_pairs','source_station_domains'):
        if packet.get(key)!=parent.get(key):raise ValueError('Static board changes inherited query/statistic/source scope')
    if packet['groups'][:len(parent['groups'])]!=parent['groups']or packet['joint_statistics_rank']!=parent['joint_statistics_rank']:raise ValueError('Static board changes inherited W/groups/statistical rank')
    rows=[]
    for group in packet['groups'][len(parent['groups']):]:
        if group.get('statistic_kind')!='constraint_only_unknown_noise'or group['rank']!=0 or group['W']!=[]or group.get('unknown_noise_source_has_statistical_precision')is not False:raise ValueError('Static board invents source precision')
        rows.extend(group['rows'])
    if sorted(rows)!=list(range(n,m)):raise ValueError('Static board suffix geometry-only partition differs')
    if cert['reader_module_refs']!=[EXECUTED_MODULE_REF]:raise ValueError('Static board prefix issuer version changed')
    for ref in cert['reader_module_refs']:_load_ref(ref,path.parent,seen)
    _prefix_arrays(parent,dict(packet,arrays=cert['issued_arrays']),parentpath.parent,path.parent,n,m,seen)
    for key,d in packet['arrays'].items():
        old=cert['issued_arrays'][key]
        if d['shape']!=old['shape']or d['dtype']!=old['dtype']:raise ValueError('Static board runtime array ABI differs')
        if key not in AUX_ARRAYS and d['sha256']!=old['sha256']:raise ValueError('Static board runtime changed a frozen source/E array')
    changed=any(packet['arrays'][k]['sha256']!=cert['issued_arrays'][k]['sha256']for k in AUX_ARRAYS)
    if changed:
        from .joint_solver import _source_membership_fingerprint
        fp=_source_membership_fingerprint(packet);explicit=False
        for ref in packet.get('source_identity_watchlist',[]):
            if ref.get('source_membership_fingerprint')!=fp:continue
            _,wrapped=_json(ref,path.parent,seen)
            if wrapped.get('board_material_source_qualification')==packet.get('board_material_source_qualification')and _source_membership_fingerprint(wrapped)==fp:explicit=True;break
        if not explicit:raise ValueError('Static board changed numerical aux needs exact same-source wrapper')
        count=int(parent['new_source_start'])
        for key in AUX_ARRAYS:
            a=_array(dict(packet,arrays=cert['issued_arrays']),key,path.parent,seen);b=_array(packet,key,path.parent,seen)
            if not np.array_equal(a[:count],b[:count]):raise ValueError('Static board numerical wrapper changed complete S01 auxiliary prefix')
    return {key:_array(packet,key,path.parent,seen)for key in packet['arrays']}


def _source_proof(packet,parent,path,parentpath,bundlepath,arrays,seen):
    bundle=json.loads(bundlepath.read_text(encoding='utf8'));base=bundlepath.parent
    if bundle.get('schema')!=BUNDLE_SCHEMA or bundle.get('source_ledger_preserved')is not True or not bundle.get('reader_module_refs'):raise ValueError('Static board lacks actual source-qualified bundle')
    for ref in bundle['reader_module_refs']:_load_ref(ref,base,seen)
    refs={key:_load_ref(bundle[key],base,seen)for key in ('model','source_rows_ref','qualification_receipts','qualification_gate','static_surface_qualification','source_generation')}
    model=json.loads(refs['model'].read_text(encoding='utf8'));qual=json.loads(refs['static_surface_qualification'].read_text(encoding='utf8'));generation=json.loads(refs['source_generation'].read_text(encoding='utf8'));gate=json.loads(refs['qualification_gate'].read_text(encoding='utf8'))
    if gate.get('schema')!=GATE_SCHEMA:raise ValueError('Static board gate schema differs')
    _source_refs(gate,refs['qualification_gate'].parent,seen)
    for key in ('source_generation','static_surface_qualification','input_cumulative_journal','qualification_receipts','current_model','track_material_bindings','numeric_source_replay','current_RGB_receipt','roles'):
        expected=bundle[{'input_cumulative_journal':'source_rows_ref','current_model':'model'}.get(key,key)]
        if gate.get(key)!=expected:raise ValueError('Static board gate actual input differs: '+key)
    if gate.get('old_raw_source_flags_false_are_not_overwritten')is not True or gate.get('new_measurement_version_geometry_qualification_only')is not True:raise ValueError('Static board gate overwrites old source authority')
    parentmodelpath,parentmodel=_json(bundle['parent_model'],base,seen)
    if (model['blocks'][:len(parentmodel['blocks'])]!=parentmodel['blocks']or model['materials'][:len(parentmodel['materials'])]!=parentmodel['materials']or any(model.get(k)!=v for k,v in parentmodel.items()if k not in ('blocks','materials'))):raise ValueError('Static board issued model changed inherited physical values')
    bindings_path,bindings=_json(bundle['track_material_bindings'],base,seen);binding_by_track={x['track_id']:x for x in bindings}
    if (qual.get('schema')!='rinkfit.native_board_print_static_texture_source_qualification.v1'or qual['source_video']['sha256']!=SOURCE_SHA
        or qual.get('all_224_source_UV_and_RGB_identities_replayed_exactly')is not True or qual.get('observations_from_model_or_ice_H')is not False
        or qual.get('UV_moved_or_interpolated')is not False or qual.get('statistical_objective_eligible')is not False):raise ValueError('Static board source material qualification is incomplete')
    for key in ('candidate_generation','candidate_rows','candidate_tracks','numeric_replay','current_RGB_receipt'):_load_ref(qual[key],refs['static_surface_qualification'].parent,seen)
    for ref in qual['reviewed_images']:_load_ref(ref,refs['static_surface_qualification'].parent,seen)
    if qual['candidate_generation']['sha256']!=bundle['source_generation']['sha256']or qual['candidate_rows']['sha256']!=generation['candidate_rows']['sha256']or generation['source_video']['sha256']!=SOURCE_SHA or generation['source_video']['time_base']!='1/12800'or generation.get('world_or_ice_H_used')is not False:raise ValueError('Static board acquisition clock/pixel source differs')
    for key in ('scope','roles','source_seed_review'):_load_ref(generation[key],refs['source_generation'].parent,seen)
    for ref in generation['reader_modules']+generation['independent_actor_ledgers']:_load_ref(ref,refs['source_generation'].parent,seen)
    roles=json.loads(_load_ref(generation['roles'],refs['source_generation'].parent,seen).read_text(encoding='utf8'))['frame_roles']
    candidates=list(read_jsonl(_load_ref(qual['candidate_rows'],refs['static_surface_qualification'].parent,seen)))
    _,numeric=_json(qual['numeric_replay'],refs['static_surface_qualification'].parent,seen);_,rgb=_json(qual['current_RGB_receipt'],refs['static_surface_qualification'].parent,seen)
    if (numeric.get('source_UV_modified')is not False or numeric.get('all_current_source_hashes_and_exclusion_masks_valid')is not True
        or numeric.get('all_emitted_rows_current_local_two_direction_supported')is not True or rgb.get('all_candidate_224_UV_and_RGB_hashes_matched')is not True):raise ValueError('Static board actual pixel/visibility/numeric replay failed')
    evidence={(x['track_id'],int(x['native_pts'])):x for x in numeric['rows_evidence']};rgbrows={(x['track_id'],int(x['native_pts'])):x for x in rgb['rows']}
    accepted={x['track_id']:x for x in qual['per_track']if x.get('qualified_for_unknown_shared_board_surface_point_initialization')is True}
    lines={};allkeys=set()
    for index in qual['qualified_source_line_index']:
        line=int(index['candidate_line_1based']);candidate=candidates[line-1];key=(candidate['track_id'],int(candidate['native_pts']))
        if key in lines or index['qualification']!='confirmed'or candidate['track_id']!=index['track_id']or candidate['native_pts']!=index['native_pts']or object_sha256(candidate['source_uv'])!=index['candidate_source_uv_sha256']:raise ValueError('Static board exact reviewed candidate index differs')
        lines[key]=(candidate,index);allkeys.add(key)
    rows=list(read_jsonl(refs['source_rows_ref']));receipts=list(read_jsonl(refs['qualification_receipts']));receiptmap={x['source_row_id']:x for x in receipts};n=int(parent['row_count']);m=int(packet['row_count']);oldcount=len(parent['material_binding']['materials']);proofs=[];used=set();trackmi={};trackblock={};mitype={};blocktype={}
    if len(bindings)!=len(accepted)or len(binding_by_track)!=len(accepted)or gate.get('qualified_new_measurement_rows')!=len(rows):raise ValueError('Static board binding/gate denominator differs')
    if len(rows)!=m-n or len(receiptmap)!=len(rows)or len(rows)!=qual['qualified_source_rows']or len(lines)!=len(rows)or len(accepted)!=qual['qualified_tracks']:raise ValueError('Static board exact qualified source denominator differs')
    table=_model_table(model)
    if table[:oldcount]!=parent['material_binding']['materials']or packet['material_binding']['materials']!=table or len(table)!=oldcount+len(accepted):raise ValueError('Static board material binding does not preserve/extend exact physical topology')
    for key,value in parent['material_binding'].items():
        if key not in ('materials','model_material_prefix_count','model_topology_ref','topological_identity_table_sha256','additional_source_material_maps','inherited_material_prefix_count','new_active_point_materials_source_qualified')and packet['material_binding'].get(key)!=value:raise ValueError('Static board changed inherited material binding authority')
    for j,row in enumerate(rows):
        i=n+j;tid=row['track_id'];pts=int(row['native_pts']);key=(tid,pts);mi=int(row['joint_material_index']);rid='static_board_texture/'+tid+'/'+str(pts);receipt=receiptmap.get(rid)
        if key not in lines or key in used or tid not in accepted or receipt is None:raise ValueError('Static board suffix has unknown/singleton/repeated source row')
        used.add(key);candidate,index=lines[key];track=accepted[tid];instance=candidate['surface_instance_id'];ev=evidence.get(key);rx=rgbrows.get(key);uv=candidate['source_uv']
        if any(row.get(k)!=v for k,v in candidate.items()if k not in ('source_geometry_eligible','physical_position')):raise ValueError('Static board raw candidate evidence changed')
        if (row['source_row_id']!=rid or str(rid).startswith(FAMILY_PREFIXES)or row['current_uv']!=uv or row['source_uv']!=uv or candidate['current_uv']!=uv
            or row.get('current_rgb_sha256')!=candidate['current_rgb_sha256']or row.get('surface_class')!='board'or row.get('surface_instance_id')!=instance
            or row.get('shot_id')!=candidate['shot_id']or candidate['shot_id']!='s02'or row.get('time_base')!=[1,12800]or row.get('source_video_sha256')!=SOURCE_SHA):raise ValueError('Static board suffix redirects current source pixels/clock/surface')
        if (any(row.get(k)is not False or candidate.get(k)is not False for k in ('pixels_from_model','known_world_corner_correspondence','existing_world_corner_or_XY_used','ice_H_used'))
            or row.get('world_position_is_shared_track_latent')is not True or row.get('measurement_operator_resolved')!=OPERATOR
            or row.get('point_correspondence')is not True or row.get('source_geometry_eligible')is not True
            or row.get('numeric_statistics_ready')is not False or row.get('statistical_objective_eligible')is not False or row.get('solver_eligible')is not False
            or row.get('fit_role')!='development'or roles[str(pts)]['role']!='development_candidate'):raise ValueError('Static board unknown shared location/source authority differs')
        diag=candidate['two_direction_tensor_diagnostic']
        if (diag.get('status')!='entity_corner_two_direction_support'or diag['current_uv']!=uv or diag['corner_second_direction_ratio']<.025
            or ev is None or ev.get('same_RGB')is not True or ev.get('current_exclusion_mask_clear_for_LK_support')is not True
            or ev['source_uv']!=uv or ev['two_direction_recomputed']!=diag or rx is None or rx['source_uv']!=uv or rx.get('decoded_hash_matches')is not True):raise ValueError('Static board current RGB/actor/local 2D evidence differs')
        if (track.get('source_material_ownership')!='confirmed_by_current_exact_native_source_review'or track['board_instance_id']!=instance
            or track.get('known_world_corner_correspondence')is not False or track.get('world_coordinates_supplied')is not False
            or track['static_material_track_qualification']!='confirmed_finite_observed_interval'or track['source_rows']<2):raise ValueError('Static board has no finite source-confirmed material continuity')
        if pts!=track['first_pts']and (candidate['forward_backward_px']is None or candidate['forward_backward_px']>numeric['existing_navigation_criteria']['reciprocal_max_px']or candidate['patch_ncc']<numeric['existing_navigation_criteria']['patch_ncc_min']):raise ValueError('Static board reciprocal/appearance motion evidence failed')
        block,pathdesc=_shared_board_point(model,mi,tid,instance,oldcount)
        binding=binding_by_track.get(tid,{})
        if (binding.get('material_index')!=mi or binding.get('path_index')!=0 or binding.get('geometry_block')!=block or binding.get('advertisement_block')!=pathdesc['advertisement_block']or binding.get('board_instance_id')!=instance or binding.get('free_coordinate_indices')!=[2,5]or binding.get('fixed_indices')!=[0,1,3,4]or binding.get('one_shared_latent_per_track')is not True or binding.get('world_coordinate_truth_or_covariance_claimed')is not False or binding.get('shared_native_rows')!=track['source_rows']):raise ValueError('Static board shared material latent binding differs')
        if trackmi.setdefault(tid,mi)!=mi or mitype.setdefault(mi,tid)!=tid or trackblock.setdefault(tid,block)!=block or blocktype.setdefault(block,tid)!=tid:raise ValueError('Static board does not use one distinct shared 2D latent per track')
        if (receipt.get('schema')!=RECEIPT_SCHEMA or receipt.get('native_pts')!=pts or receipt.get('track_id')!=tid
            or receipt.get('existing_material_index',receipt.get('joint_material_index'))!=mi or receipt.get('original_UV_preserved',receipt.get('source_uv'))!=uv
            or receipt.get('static_surface_qualification')!=bundle['static_surface_qualification']
            or receipt.get('source_row_id')!=rid or receipt.get('source_uv')!=uv or receipt.get('shot_id')!=row['shot_id']or receipt.get('current_rgb_sha256')!=row['current_rgb_sha256']
            or receipt.get('status')!='accepted_static_board_print'or receipt.get('geometry_block')!=block or receipt.get('advertisement_block')!=pathdesc['advertisement_block']or receipt.get('source_board_instance_id')!=instance
            or receipt.get('point_path_kind')!=2 or receipt.get('entity_kind')!=1 or receipt.get('entity_station')!=0 or receipt.get('entity_station_fixed')is not True
            or receipt.get('known_world_corner_correspondence')is not False or receipt.get('unknown_shared_surface_coordinates')is not True
            or receipt.get('physical_point')is not True or receipt.get('new_material_suffix')is not True or receipt.get('new_actual_source_geometry_qualified')is not True
            or any(receipt.get(k)is not False for k in ('solver_eligible','statistical_objective_eligible','numeric_statistics_ready'))
            or receipt.get('target_id')!=model['materials'][mi]['label']or receipt.get('existing_point_path')!=pathdesc['id']
            or receipt.get('current_2D_source_mean_sha256')!=object_sha256(dict(source_row_id=rid,native_pts=pts,current_2D_UV=uv,estimator=OPERATOR))):raise ValueError('Static board source receipt differs from exact reviewed material exposure')
        if (packet['source_row_ids'][i]!=rid or int(arrays['native_pts'][i])!=pts or packet['shot_ids'][int(arrays['shot_index'][i])]!=row['shot_id']or(pts-256256)%256
            or not np.array_equal(arrays['uv'][i],uv)or arrays['kind'][i]!=1 or arrays['material_index'][i]!=mi or arrays['path'][i]!=0 or arrays['station'][i]!=0
            or arrays['admission'][i]!=1 or arrays['fit_role'][i]!=1):raise ValueError('Static board packet does not consume exact actual source point/clock')
        proofs.append(dict(source_row=row,qualification_receipt=receipt,candidate=candidate,qualification_index=index,track_qualification=track,numeric_evidence=ev,RGB_evidence=rx,material_index=mi,geometry_block=block,path_index=0,advertisement_block=pathdesc['advertisement_block'],track_material_binding=binding))
    if used!=allkeys or set(trackmi)!=set(accepted)or set(trackmi.values())!=set(range(oldcount,len(model['materials']))):raise ValueError('Static board source/model coverage incomplete')
    for tid in accepted:
        points=sorted(pts for track,pts in used if track==tid)
        if len(points)!=accepted[tid]['source_rows']or points[0]!=accepted[tid]['first_pts']or points[-1]!=accepted[tid]['last_pts']or any(b-a!=256 for a,b in zip(points,points[1:])):raise ValueError('Static board source track crosses a gap/cut or omits exposure')
    return proofs


def assert_static_board_prefix_qualification(packet,packet_path,*,parent_document=None,certificate=None,verified_dependency_cache=None):
    seen=set()if verified_dependency_cache is None else verified_dependency_cache;path=Path(packet_path).resolve();certpath,cert=_json(packet['board_material_source_qualification'],path.parent,seen)if certificate is None else certificate
    if cert.get('schema')!=SCHEMA:raise ValueError('Unsupported static board source certificate')
    parentpath=_load_ref(cert['parent_packet'],Path(certpath).parent,seen);parent=parent_document if parent_document is not None else json.loads(parentpath.read_text(encoding='utf8'))
    from .known_entity_prefix_qualification import assert_board_source_qualification_current
    assert_board_source_qualification_current(parent,parentpath,verified_dependency_cache=seen)
    arrays=_prefix(packet,parent,path,parentpath,cert,seen);bundlepath=_load_ref(cert['actual_append_source_bundle'],Path(certpath).parent,seen);proofs=_source_proof(packet,parent,path,parentpath,bundlepath,arrays,seen)
    if certificate is None and list(read_jsonl(_load_ref(cert['qualified_rows_native'],Path(certpath).parent,seen)))!=proofs:raise ValueError('Static board native proof differs from complete reviewed source data')
    return proofs


def finalize_static_board_tracklet_prefix_certificate(parent_packet_path,appended_packet_document,bundle_path,output_certificate_path,*,parent_document=None):
    parentpath=Path(parent_packet_path).resolve();out=Path(output_certificate_path).resolve();bundle=Path(bundle_path).resolve();packet=appended_packet_document
    if out.exists():raise FileExistsError('Static board source certificate is immutable')
    if _ref(__file__)!=EXECUTED_MODULE_REF:raise ValueError('Static board issuer changed during execution')
    parent=parent_document if parent_document is not None else json.loads(parentpath.read_text(encoding='utf8'))
    cert=dict(schema=SCHEMA,parent_packet=_ref(parentpath),original_raw_rows=int(parent['row_count']),raw_rows=int(packet['row_count']),appended_raw_rows=int(packet['row_count'])-int(parent['row_count']),actual_append_source_bundle=_ref(bundle),issued_arrays=packet['arrays'],reader_module_refs=[EXECUTED_MODULE_REF],original_S01_prefix_rows=parent['new_source_start'],unknown_world_position_preserved=True,new_statistical_precision_invented=False,all_inherited_source_arrays_byte_identical_at_issuance=True)
    proof=assert_static_board_prefix_qualification(packet,out.parent/'packet.json',parent_document=parent,certificate=(out,cert));ledger=out.parent/'qualified_source_rows_native.jsonl'
    if ledger.exists():raise FileExistsError('Static board native proof is immutable')
    with ledger.open('w',encoding='utf8')as stream:
        for row in proof:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    cert['qualified_rows_native']=_ref(ledger);write_json(out,cert);packet['board_material_source_qualification']=_ref(out);packet['all_new_source_nuisance_requires_fresh_physical_capture']=True
    return packet
