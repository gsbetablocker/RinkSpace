"""Explicit qualified board-material successor after the frozen known/V6 chain.

The frozen V6 numeric-prefix reader is delegated verbatim. This new issuer
owns the general source proof and explicit recursive parent contract; no old
issuer is represented as issuing this generation.
"""
from pathlib import Path
import json
import numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256,object_sha256
from . import static_board_prefix_qualification as frozen
from . import known_after_board_prefix_qualification as parent_guard
from .board_source_qualification_guard import _load_ref,FAMILY_PREFIXES,BOUND_ARRAYS
from .static_ice_prefix_qualification import _array
from .static_board_tracklet_append import _ad_block

SCHEMA='rinkfit.board_material_fit_qualification.v7_static_board_successor_prefix'
CONTRACT='static_board_successor_after_known_v1'
BUNDLE_SCHEMA=frozen.BUNDLE_SCHEMA
GATE_SCHEMA=frozen.GATE_SCHEMA
RECEIPT_SCHEMA=frozen.RECEIPT_SCHEMA
OPERATOR=frozen.OPERATOR
SOURCE_SHA=frozen.SOURCE_SHA
_ref=frozen._ref
_json=frozen._json
_source_refs=frozen._source_refs
_model_table=frozen._model_table
EXECUTED_MODULE_REF=_ref(__file__)
PARENT_GUARD_REF=_ref(parent_guard.__file__)
DELEGATED_HELPERS=dict(module=frozen.EXECUTED_MODULE_REF,functions=['_prefix'])


def assert_board_source_qualification_current(packet,packet_path,*,verified_dependency_cache=None):
    seen=set()if verified_dependency_cache is None else verified_dependency_cache
    path=Path(packet_path).resolve();ref=packet.get('board_material_source_qualification')
    if ref:
        cp,cert=_json(ref,path.parent,seen)
        if cert.get('schema')==SCHEMA:
            return assert_static_board_successor_prefix_qualification(packet,path,verified_dependency_cache=seen,certificate=(cp,cert))
    return parent_guard.assert_board_source_qualification_current(packet,path,verified_dependency_cache=seen)


def _shared_board_point(model,mi,tid,instance,old_count):
    if mi<old_count:raise ValueError('Static board texture cannot reuse a known existing world point')
    material=model['materials'][mi]
    if len(material['paths'])!=1:raise ValueError('Static board texture needs one shared point path')
    path=material['paths'][0];ad=_ad_block(model,instance)
    if (path.get('kind')!=2 or path.get('points')!=[[0,0]]or path.get('physical_point')is not True
        or path.get('advertisement_block')!=ad or path.get('geometry_indices')or path.get('geometry_offset',0)!=0):raise ValueError('Static board point is not on the existing unfolded advertisement surface')
    block_index=int(path['geometry_block']);block=model['blocks'][block_index];values=np.asarray(block['values'],float)
    if (values.shape!=(6,)or not np.isfinite(values).all()or values[[0,1,3,4]].tolist()!=[1,0,0,1]
        or block.get('fixed_indices')!=[0,1,3,4]or block.get('constant')is not False
        or any(block['lower'][k]!=0 or block['upper'][k]!=1 or not 0<=values[k]<=1 for k in (2,5))):raise ValueError('Static board latent is not exactly two bounded shared material coordinates')
    if material.get('physical_track_id',tid)!=tid or len(block['lower'])!=6 or len(block['upper'])!=6:raise ValueError('Static board latent material identity/ABI differs')
    return block_index,path


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
    if (qual.get('schema')!='rinkfit.native_board_print_static_texture_source_qualification.v1'or qual.get('authority')!='assistant_current_native_pixel_review'or qual['source_video']['sha256']!=SOURCE_SHA
        or qual.get('all_source_UV_and_RGB_identities_replayed_exactly')is not True or qual.get('observations_from_model_or_ice_H')is not False
        or qual.get('UV_moved_or_interpolated')is not False):raise ValueError('Static board source material qualification is incomplete')
    for key in ('candidate_generation','candidate_rows','candidate_tracks','numeric_replay','current_RGB_receipt'):_load_ref(qual[key],refs['static_surface_qualification'].parent,seen)
    for ref in qual['reviewed_images']:_load_ref(ref,refs['static_surface_qualification'].parent,seen)
    if qual['candidate_generation']['sha256']!=bundle['source_generation']['sha256']or qual['candidate_rows']['sha256']!=generation['candidate_rows']['sha256']or generation['source_video']['sha256']!=SOURCE_SHA or generation['source_video']['time_base']!='1/12800'or generation.get('world_or_ice_H_used')is not False:raise ValueError('Static board acquisition clock/pixel source differs')
    for key in ('scope','roles','source_seed_review'):_load_ref(generation[key],refs['source_generation'].parent,seen)
    for ref in generation['reader_modules']+generation['independent_actor_ledgers']:_load_ref(ref,refs['source_generation'].parent,seen)
    roles=json.loads(_load_ref(generation['roles'],refs['source_generation'].parent,seen).read_text(encoding='utf8'))['frame_roles']
    candidates=list(read_jsonl(_load_ref(qual['candidate_rows'],refs['static_surface_qualification'].parent,seen)))
    scope=read_json(_load_ref(generation['scope'],refs['source_generation'].parent,seen))
    shot_scope={s['id']:s for s in scope['shots']}
    if qual.get('source_rows_replayed')!=len(candidates):raise ValueError('Successor board actual RGB replay denominator differs')

    _,numeric=_json(qual['numeric_replay'],refs['static_surface_qualification'].parent,seen);_,rgb=_json(qual['current_RGB_receipt'],refs['static_surface_qualification'].parent,seen)
    if (numeric.get('source_UV_modified')is not False or numeric.get('all_current_source_hashes_and_exclusion_masks_valid')is not True
        or numeric.get('all_emitted_rows_current_local_two_direction_supported')is not True or rgb.get('all_candidate_UV_and_RGB_hashes_matched')is not True):raise ValueError('Static board actual pixel/visibility/numeric replay failed')
    if rgb.get('source_rows_replayed')!=len(candidates):raise ValueError('Successor board current RGB receipt denominator differs')
    evidence={(x['track_id'],int(x['native_pts'])):x for x in numeric['rows_evidence']};rgbrows={(x['track_id'],int(x['native_pts'])):x for x in rgb['rows']}
    accepted={x['track_id']:x for x in qual['per_track']if x.get('qualified_for_unknown_shared_board_surface_point_initialization')is True}
    lines={};allkeys=set()
    for index in qual['qualified_source_line_index']:
        line=int(index['candidate_line_1based']);candidate=candidates[line-1];key=(candidate['track_id'],int(candidate['native_pts']))
        if key in lines or ('qualification'in index and index['qualification']!='confirmed')or candidate['track_id']!=index['track_id']or candidate['native_pts']!=index['native_pts']or object_sha256(candidate['source_uv'])!=index['candidate_source_uv_sha256']:raise ValueError('Static board exact reviewed candidate index differs')
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
            or row.get('shot_id')!=candidate['shot_id']or candidate['shot_id']=='s01'or row.get('time_base')!=[1,12800]or row.get('source_video_sha256')!=SOURCE_SHA):raise ValueError('Static board suffix redirects current source pixels/clock/surface')
        if (any(row.get(k)is not False or candidate.get(k)is not False for k in ('pixels_from_model','known_world_corner_correspondence','existing_world_corner_or_XY_used','ice_H_used'))
            or row.get('world_position_is_shared_track_latent')is not True or row.get('measurement_operator_resolved')!=OPERATOR
            or row.get('point_correspondence')is not True or row.get('source_geometry_eligible')is not True
            or row.get('numeric_statistics_ready')is not False or row.get('statistical_objective_eligible')is not False or row.get('solver_eligible')is not False
            or row.get('fit_role')!='development'or roles[str(pts)]['role']!='development_candidate'):raise ValueError('Static board unknown shared location/source authority differs')
        source_shot=shot_scope.get(candidate['shot_id'])
        if source_shot is None or not source_shot['first_pts']<=pts<=source_shot['last_pts'] or (pts-source_shot['first_pts'])%256:
            raise ValueError('Successor board candidate is outside its exact native cut/clock')
        diag=candidate['two_direction_tensor_diagnostic']
        if (diag.get('status')!='entity_corner_two_direction_support'or diag['current_uv']!=uv or diag['corner_second_direction_ratio']<.025
            or ev is None or ev.get('same_RGB')is not True or ev.get('current_exclusion_mask_clear_for_LK_support')is not True
            or ev['source_uv']!=uv or ev['two_direction_recomputed']!=diag or rx is None or rx['source_uv']!=uv or rx.get('decoded_hash_matches')is not True):raise ValueError('Static board current RGB/actor/local 2D evidence differs')
        if (track.get('source_material_ownership')!='confirmed_by_current_exact_native_source_review'or track['board_instance_id']!=instance
            or track.get('known_world_corner_correspondence')is not False or track.get('world_coordinates_supplied')is not False
            or track.get('qualified_for_unknown_shared_board_surface_point_initialization')is not True or track['source_rows']<2):raise ValueError('Static board has no finite source-confirmed material continuity')
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



def _contract(cert,base,seen):
    if (cert.get('schema')!=SCHEMA or cert.get('parent_recursion_contract')!=CONTRACT
        or cert.get('issuer_module_ref')!=EXECUTED_MODULE_REF or cert.get('recursive_parent_guard_module_ref')!=PARENT_GUARD_REF
        or cert.get('delegated_numeric_helpers')!=DELEGATED_HELPERS or cert.get('reader_module_refs')!=[frozen.EXECUTED_MODULE_REF]):
        raise ValueError('Successor board actual issuer/helper/recursive contract differs')
    for ref in (EXECUTED_MODULE_REF,PARENT_GUARD_REF,DELEGATED_HELPERS['module']):_load_ref(ref,base,seen)


def assert_static_board_successor_prefix_qualification(packet,packet_path,*,parent_document=None,certificate=None,verified_dependency_cache=None):
    seen=set()if verified_dependency_cache is None else verified_dependency_cache;path=Path(packet_path).resolve()
    cp,cert=_json(packet['board_material_source_qualification'],path.parent,seen)if certificate is None else certificate
    _contract(cert,cp.parent,seen)
    pp=_load_ref(cert['parent_packet'],cp.parent,seen);parent=parent_document if parent_document is not None else read_json(pp)
    if cert.get('inherited_material_count')!=len(parent['material_binding']['materials']):raise ValueError('Successor board parent material view count differs')
    assert_board_source_qualification_current(parent,pp,verified_dependency_cache=seen)
    arrays=frozen._prefix(packet,parent,path,pp,cert,seen)
    bundlepath=_load_ref(cert['actual_append_source_bundle'],cp.parent,seen)
    bundle=read_json(bundlepath);parent_model_path=_load_ref(bundle['parent_model'],bundlepath.parent,seen);parent_model=read_json(parent_model_path)
    if (cert.get('issued_parent_model')!=bundle['parent_model'] or cert.get('inherited_physical_block_count')!=len(parent_model['blocks'])
        or cert.get('inherited_material_count')!=len(parent_model['materials'])
        or cert.get('old_append_provenance_exact')!=parent_model.get('static_board_shared_uv_append')):
        raise ValueError('Successor board actual inherited model/provenance view differs')
    proof=_source_proof(packet,parent,path,pp,bundlepath,arrays,seen)
    if 'qualified_rows_native'in cert and list(read_jsonl(_load_ref(cert['qualified_rows_native'],cp.parent,seen)))!=proof:
        raise ValueError('Successor board native proof differs from complete actual source')
    return proof


def finalize_static_board_successor_prefix_certificate(parent_packet_path,appended_packet_document,bundle_path,output_certificate_path,*,parent_document=None):
    pp=Path(parent_packet_path).resolve();out=Path(output_certificate_path).resolve();bundle=Path(bundle_path).resolve();packet=appended_packet_document
    if out.exists()or(out.parent/'qualified_source_rows_native.jsonl').exists():raise FileExistsError('Successor board certificate is immutable')
    if _ref(__file__)!=EXECUTED_MODULE_REF:raise ValueError('Successor board issuer changed while executing')
    parent=parent_document if parent_document is not None else read_json(pp)
    b=read_json(bundle);pm=_load_ref(b['parent_model'],bundle.parent,set());model=read_json(pm)
    cert=dict(schema=SCHEMA,parent_recursion_contract=CONTRACT,issuer_module_ref=EXECUTED_MODULE_REF,
        recursive_parent_guard_module_ref=PARENT_GUARD_REF,delegated_numeric_helpers=DELEGATED_HELPERS,
        parent_packet=_ref(pp),original_raw_rows=int(parent['row_count']),raw_rows=int(packet['row_count']),
        appended_raw_rows=int(packet['row_count'])-int(parent['row_count']),actual_append_source_bundle=_ref(bundle),
        issued_arrays=packet['arrays'],reader_module_refs=[frozen.EXECUTED_MODULE_REF],original_S01_prefix_rows=parent['new_source_start'],
        inherited_material_count=len(parent['material_binding']['materials']),inherited_physical_block_count=len(model['blocks']),
        issued_parent_model=_ref(pm),old_append_provenance_exact=model.get('static_board_shared_uv_append'),
        unknown_world_position_preserved=True,new_statistical_precision_invented=False,all_inherited_source_arrays_byte_identical_at_issuance=True)
    proof=assert_static_board_successor_prefix_qualification(packet,out.parent/'packet.json',parent_document=parent,certificate=(out,cert))
    native=out.parent/'qualified_source_rows_native.jsonl'
    with native.open('w',encoding='utf8')as stream:
        for row in proof:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    cert['qualified_rows_native']=_ref(native);write_json(out,cert);packet['board_material_source_qualification']=_ref(out)
    packet['all_new_source_nuisance_requires_fresh_physical_capture']=True
    return packet
