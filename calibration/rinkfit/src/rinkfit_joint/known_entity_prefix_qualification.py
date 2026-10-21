"""Strict same-entity native corner source appends with immutable old prefixes.

This separate v5 issuer does not change the archived v3/v4 consumers. New
geometry-only rows require exact existing-corner ownership, source replay and
explicit aliases for old inactive observations. Source pixels never come from
camera predictions or from a fitted station.
"""
from pathlib import Path
import json
import numpy as np
from .io_utils import file_sha256,read_jsonl,write_json,object_sha256
from .board_source_qualification_guard import _load_ref,BOUND_ARRAYS,FAMILY_PREFIXES
from .static_ice_prefix_qualification import _array,_prefix_arrays

SCHEMA='rinkfit.board_material_fit_qualification.v5_known_entity_prefix'
BUNDLE_SCHEMA='rinkfit.native_known_entity_tracklet_qualified_bundle.v1'
SOURCE_SHA='e8b53966d56e72f3842d92c3ecbe6e9cc7e4670b7d0c9bd12f58fd2cff5ad57d'


def _ref(path):
    p=Path(path).resolve();return dict(file=str(p),sha256=file_sha256(p))


EXECUTED_MODULE_REF=_ref(__file__)


def _refs(document,base,seen):
    if isinstance(document,dict):
        if ('file'in document or 'path'in document)and 'sha256'in document:
            _load_ref(document,base,seen)
        else:
            for value in document.values():_refs(value,base,seen)
    elif isinstance(document,list):
        for value in document:_refs(value,base,seen)


def _json_ref(ref,base,seen):
    p=_load_ref(ref,base,seen);return p,json.loads(p.read_text(encoding='utf8'))


def assert_board_source_qualification_current(packet,packet_path,*,verified_dependency_cache=None):
    seen=set()if verified_dependency_cache is None else verified_dependency_cache
    path=Path(packet_path).resolve();ref=packet.get('board_material_source_qualification')
    if ref:
        cp,cert=_json_ref(ref,path.parent,seen)
        if cert.get('schema')==SCHEMA:return assert_known_entity_prefix_qualification(packet,path,verified_dependency_cache=seen,certificate=(cp,cert))
        from .static_ice_prefix_qualification import SCHEMA as ICE_SCHEMA,assert_static_ice_prefix_qualification
        if cert.get('schema')==ICE_SCHEMA:return assert_static_ice_prefix_qualification(packet,path,verified_dependency_cache=seen)
    from .board_source_qualification_guard import assert_board_material_source_qualification
    return assert_board_material_source_qualification(packet,path,verified_dependency_cache=seen)


def _prefix(packet,parent,packet_path,parentpath,cert,seen):
    n=int(parent['row_count']);m=int(packet['row_count'])
    if n!=cert['original_raw_rows']or m!=cert['raw_rows']or m<=n or m-n!=cert['appended_raw_rows']or packet['shot_ids']!=parent['shot_ids']:raise ValueError('Known entity append raw prefix/scope differs')
    for root in (parentpath.parent,packet_path.parent):
        for name in ('SOURCE_IDENTITY_CERTIFICATE_RECALLED.json','SOURCE_IDENTITY_EVIDENCE_RECALL.json'):
            if(root/name).exists():raise ValueError('Known entity append inherited recalled source evidence')
    for key in ('source_row_ids','source_dataset_ids','source_family_ids'):
        if len(packet[key])!=m or packet[key][:n]!=parent[key]:raise ValueError('Known entity append redirects inherited source registry')
    for key in ('queries','independent_rows','independent_statistic_kind','stroke_pairs','source_station_domains','material_binding'):
        if packet.get(key)!=parent.get(key):raise ValueError('Known entity append changes inherited identity/query/statistic scope: '+key)
    if packet['groups'][:len(parent['groups'])]!=parent['groups']:raise ValueError('Known entity append changes inherited group/W')
    suffix=[]
    for group in packet['groups'][len(parent['groups']):]:
        if group.get('statistic_kind')!='constraint_only_unknown_noise'or group['rank']!=0 or group['W']!=[]or group.get('unknown_noise_source_has_statistical_precision')is not False:raise ValueError('Known entity append invents precision')
        suffix.extend(group['rows'])
    if sorted(suffix)!=list(range(n,m))or packet['joint_statistics_rank']!=parent['joint_statistics_rank']:raise ValueError('Known entity append geometry/statistical partition differs')
    if cert['reader_module_refs']!=[EXECUTED_MODULE_REF]:raise ValueError('Known entity source issuer version changed')
    for ref in cert['reader_module_refs']:_load_ref(ref,packet_path.parent,seen)
    _prefix_arrays(parent,dict(packet,arrays=cert['issued_arrays']),parentpath.parent,packet_path.parent,n,m,seen)
    for name in BOUND_ARRAYS:
        current=packet['arrays'][name]
        if current['sha256']!=cert['membership_array_sha256'][name]or current['sha256']!=cert['issued_arrays'][name]['sha256']:raise ValueError('Known entity source arrays changed')
    if any(packet['arrays'][k]['sha256']!=v['sha256']for k,v in cert['issued_arrays'].items()if k not in BOUND_ARRAYS):
        from .joint_solver import _source_membership_fingerprint
        fingerprint=_source_membership_fingerprint(packet);matched=False
        for ref in packet.get('source_identity_watchlist',[]):
            if ref.get('source_membership_fingerprint')!=fingerprint:continue
            _,wrapped=_json_ref(ref,packet_path.parent,seen)
            if wrapped.get('board_material_source_qualification')==packet.get('board_material_source_qualification')and _source_membership_fingerprint(wrapped)==fingerprint:matched=True;break
        if not matched:raise ValueError('Known entity numerical auxiliaries need explicit verified same-source wrapper')
    return {k:_array(packet,k,packet_path.parent,seen)for k in packet['arrays']}


def _source_bands_contain(uv,replay):
    edges=replay.get('material_edge_support',[])
    if len(edges)!=2 or replay.get('chosen_within_both_material_edge_support')is not True:return False
    for edge in edges:
        origin=np.asarray(edge['origin_uv'],float);normal=np.asarray(edge['normal'],float);interval=np.asarray(edge['normal_residual_support_px'],float)
        if origin.shape!=(2,)or normal.shape!=(2,)or interval.shape!=(2,)or not np.isfinite(np.r_[origin,normal,interval]).all()or not .99<np.linalg.norm(normal)<1.01:return False
        residual=float((np.asarray(uv)-origin)@normal)
        if not interval[0]<=residual<=interval[1]:return False
        q=np.asarray(edge['curve_origin_uv'],float);t=np.asarray(edge['curve_tangent'],float);n=np.asarray(edge['curve_normal'],float);a=np.asarray(edge['curve_coefficients'],float);band=np.asarray(edge['curve_residual_support_px'],float)
        if q.shape!=(2,)or t.shape!=(2,)or n.shape!=(2,)or a.shape!=(3,)or band.shape!=(2,)or not np.isfinite(np.r_[q,t,n,a,band]).all():return False
        delta=np.asarray(uv)-q;x=float(delta@t);exact=float(delta@n-a[0]-a[1]*x-a[2]*x*x)
        if not band[0]<=exact<=band[1]:return False
    return True


def _old_outside_all_lobes(uv,replay):
    for edge in replay.get('material_edge_support',[]):
        intervals=edge.get('profile_normal_residual_support_px',[])
        if not intervals:continue
        residual=float((np.asarray(uv)-np.asarray(edge['origin_uv']))@np.asarray(edge['normal']))
        if all(not lo<=residual<=hi for lo,hi in intervals):return True
    return False


def _source_proof(packet,parent,packet_path,parentpath,bundlepath,arrays,seen):
    b=json.loads(bundlepath.read_text(encoding='utf8'));base=bundlepath.parent
    if b.get('schema')!=BUNDLE_SCHEMA or b.get('source_ledger_preserved')is not True or not b.get('reader_module_refs'):raise ValueError('Known entity append lacks actual qualified source bundle')
    for ref in b['reader_module_refs']:_load_ref(ref,base,seen)
    resolved={k:_load_ref(b[k],base,seen)for k in ('model','source_rows_ref','qualification_receipts','qualification_gate','source_mean_replay_QA','ownership_review','anchor_inventory','source_generation')}
    model=json.loads(resolved['model'].read_text(encoding='utf8'));gate=json.loads(resolved['qualification_gate'].read_text(encoding='utf8'));qa=json.loads(resolved['source_mean_replay_QA'].read_text(encoding='utf8'));ownership=json.loads(resolved['ownership_review'].read_text(encoding='utf8'));inventory=json.loads(resolved['anchor_inventory'].read_text(encoding='utf8'));gen=json.loads(resolved['source_generation'].read_text(encoding='utf8'))
    if gate.get('schema')!='rinkfit.known_corner_source_support_qualification.v1'or gate.get('old_raw_source_flags_false_are_not_overwritten')is not True or gate.get('new_measurement_version_geometry_qualification_only')is not True:raise ValueError('Known entity source gate alters historical source authority')
    if gate['input_cumulative_journal']['sha256']!=b['source_rows_ref']['sha256']or gate['qualification_receipts']['sha256']!=b['qualification_receipts']['sha256']or gate['source_mean_and_floating_estimator_replay_QA']['sha256']!=b['source_mean_replay_QA']['sha256']:raise ValueError('Known entity gate actual input refs differ')
    _refs(gate,resolved['qualification_gate'].parent,seen);_refs(qa,resolved['source_mean_replay_QA'].parent,seen)
    if qa.get('passed')is not True:raise ValueError('Known entity actual native source replay failed')
    if gen['source_video']['sha256']!=SOURCE_SHA or gen['source_video']['time_base']!='1/12800'or gen.get('model_or_H_used')is not False:raise ValueError('Known entity native source acquisition identity differs')
    _refs({k:gen[k]for k in ('scope','roles','packet_bound_anchor_reader','executed_source_reader_snapshot','independent_source_bound_actor_ledgers','anchor_inventory')},resolved['source_generation'].parent,seen)
    roles=json.loads(_load_ref(gen['roles'],resolved['source_generation'].parent,seen).read_text(encoding='utf8'))['frame_roles']
    originals={(x['track_id'],int(x['native_pts'])):x for x in read_jsonl(_load_ref(gen['source_rows_ref'],resolved['source_generation'].parent,seen))}
    anchors={int(x['raw_index']):x for x in read_jsonl(_load_ref(inventory['source_rows_ref'],resolved['anchor_inventory'].parent,seen))}
    episodes={x['track_id']:x for x in ownership['reviewed_episodes']}
    if ownership.get('source_body_material_ownership_supported')is not True:raise ValueError('Known corner ownership has not been reviewed')
    rows=list(read_jsonl(resolved['source_rows_ref']));receipts=list(read_jsonl(resolved['qualification_receipts']));n=int(parent['row_count']);m=int(packet['row_count'])
    qrows=qa.get('rows')
    if qrows is None and qa.get('rows_ref'):qrows=list(read_jsonl(_load_ref(qa['rows_ref'],resolved['source_mean_replay_QA'].parent,seen)))
    if qrows is None:raise ValueError('Actual source replay QA needs per-row exact evidence')
    replay={x['source_row_id']:x for x in qrows};receipts_by_id={x['source_row_id']:x for x in receipts}
    if len(rows)!=m-n or len(receipts_by_id)!=len(rows)or gate['qualified_new_measurement_rows']!=len(rows):raise ValueError('Known entity suffix exact source accounting differs')
    oldarrays={k:_array(parent,k,parentpath.parent,seen)for k in ('kind','material_index','native_pts','admission','fit_role','path','uv')}
    by_entity={}
    for i in range(n):
        if int(oldarrays['kind'][i])==1:by_entity.setdefault((int(oldarrays['material_index'][i]),int(oldarrays['path'][i]),int(oldarrays['native_pts'][i])),[]).append(i)
    proofs=[];used=set();new_entities=set()
    from .typed_entity_append import validate_new_entity
    for j,row in enumerate(rows):
        i=n+j;rid=row['source_row_id'];tid=row['track_id'];pts=int(row['native_pts']);mi=int(row['joint_material_index']);uv=row['current_uv'];receipt=receipts_by_id.get(rid);replayed=replay.get(rid);candidate=originals.get((tid,pts));episode=episodes.get(tid)
        if rid in used or str(rid).startswith(FAMILY_PREFIXES)or receipt is None or replayed is None or candidate is None or episode is None:raise ValueError('Known entity suffix contains unreviewed/repeated source identity')
        used.add(rid)
        if not(episode['native_pts_interval'][0]<=pts<=episode['native_pts_interval'][1])or episode['existing_material_index']!=mi or episode['existing_point_path']!=row['existing_point_path']:raise ValueError('Known entity suffix exceeds reviewed physical continuity')
        if (candidate['original_packet_measurement_priority']or candidate['source_mean_pixels_from_model']or candidate['world_or_H_used']or candidate['source_geometry_eligible']or candidate['source_video_sha256']!=SOURCE_SHA or candidate['shot_id']!=row['shot_id']or candidate['target_id']!=row['target_id']or candidate['existing_material_index']!=mi or candidate['existing_point_path']!=row['existing_point_path']or candidate['current_rgb_sha256']!=row['current_rgb_sha256']):raise ValueError('Known entity source journal redirects actual candidate pixels/entity')
        op=candidate['current_source_operator'];peak=candidate['current_uv']
        if not op['valid']or op['current_uv']!=peak or op['pixel_support_status']!='current_native_two_material_edges'or candidate['two_direction_tensor_diagnostic']['corner_second_direction_ratio']<.025:raise ValueError('Known entity source lacks actual two-direction material support')
        expected_operator=dict(original_corner_seed_native_uv=op['source_operator']['intersection_seed_uv'],edge_tangents=op['edge_tangents'])
        if row.get('current_corner_operator')!=expected_operator:raise ValueError('Known entity next feedback reader does not consume the actual native source operator')
        if row.get('source_video_sha256')!=SOURCE_SHA or row.get('time_base')!=[1,12800]or row.get('point_correspondence')is not True or row.get('measurement_operator_resolved')!='physical_corner'or row.get('numeric_statistics_ready')is not False or row.get('statistical_objective_eligible')is not False or row.get('solver_eligible')is not False or row.get('source_geometry_eligible')is not True or row.get('fit_role')!='development'or roles.get(str(pts),{}).get('role')!='development_candidate':raise ValueError('Known entity source clock/role/authority differs')
        if replayed.get('valid',replayed.get('passed'))is not True or replayed.get('current_rgb_sha256')!=row['current_rgb_sha256']or replayed.get('new_peak_uv')!=peak or replayed.get('chosen_uv')!=uv or int(replayed['native_pts'])!=pts:raise ValueError('Known entity exact per-row native replay differs')
        chosen_diag=replayed.get('chosen_two_direction_diagnostic',{})
        if chosen_diag.get('status')!='entity_corner_two_direction_support'or chosen_diag.get('corner_second_direction_ratio',0)<.025:raise ValueError('Chosen native source mean lacks current two-direction support')
        path=validate_new_entity(row,receipt,model['materials'][mi],mi)
        if receipt.get('new_actual_source_geometry_qualified')is not True or receipt['entity_kind']!=1 or not receipt['physical_point']or receipt['solver_eligible']or receipt['statistical_objective_eligible']:raise ValueError('Known entity receipt lacks source geometry-only qualification')
        if (mi,path,pts)in new_entities:raise ValueError('Known entity suffix repeats same physical corner exposure')
        new_entities.add((mi,path,pts))
        prior=by_entity.get((mi,path,pts),[])
        if any(oldarrays['admission'][k]==1 and oldarrays['fit_role'][k]==1 for k in prior):raise ValueError('Known entity append duplicates an active physical corner exposure')
        aliases=row.get('prior_inactive_row_indices',[])
        if prior!=aliases:raise ValueError('Known entity append does not enumerate every old same-source inactive row')
        if not _source_bands_contain(uv,replayed):raise ValueError('Chosen UV lacks both actual material edge support bands')
        if prior:
            if row.get('explicit_new_measurement_version')is not True:raise ValueError('Known entity inactive alias lacks explicit measurement version')
            if any(not np.array_equal(oldarrays['uv'][k],row['source_uv'])for k in prior):raise ValueError('Known entity measurement version lost original inactive source UV')
            if uv!=row['source_uv']:
                if (row.get('source_position_correction_confirmed')is not True or replayed.get('old_UV_outside_all_profile_lobes_on_edge')is not True or not _old_outside_all_lobes(row['source_uv'],replayed)or uv!=peak or not row.get('position_correction_evidence')):raise ValueError('Moving old source UV requires explicit actual wrong-position evidence')
                _refs(row['position_correction_evidence'],resolved['source_rows_ref'].parent,seen)
        elif uv!=peak or row['source_uv']!=uv:raise ValueError('New exposure source mean is not actual current native peak')
        if receipt['original_UV_preserved']!=row['source_uv']:raise ValueError('Known entity source original UV changed')
        anchor_candidates=[a for a in anchors.values()if a['source_row_id']==candidate['qualified_anchor_source_row_id']and a['joint_material_index']==mi]
        if len(anchor_candidates)!=1:raise ValueError('Known entity lacks one exact qualified existing source anchor')
        anchor=anchor_candidates[0];k=int(anchor['raw_index'])
        if k>=n or parent['source_row_ids'][k]!=anchor['source_row_id']or not np.array_equal(oldarrays['uv'][k],anchor['packet_uv'])or int(oldarrays['material_index'][k])!=mi or oldarrays['admission'][k]!=1 or oldarrays['fit_role'][k]!=1 or anchor['source_uv']!=candidate['qualified_anchor_original_UV']:raise ValueError('Known entity current packet anchor/source UV differs')
        if packet['source_row_ids'][i]!=rid or int(arrays['native_pts'][i])!=pts or packet['shot_ids'][int(arrays['shot_index'][i])]!=row['shot_id']or(pts-256256)%256 or not np.array_equal(arrays['uv'][i],uv)or int(arrays['material_index'][i])!=mi or arrays['kind'][i]!=1 or arrays['path'][i]!=path or arrays['station'][i]!=0 or arrays['admission'][i]!=1 or arrays['fit_role'][i]!=1:raise ValueError('Known entity packet does not consume exact fixed-point native source mean/identity')
        if mi>=len(parent['material_binding']['materials'])or model['materials'][mi]['paths'][path]['id']!=row['existing_point_path']:raise ValueError('Known entity suffix creates/replaces physical material')
        brief={key:anchor[key]for key in ('raw_index','source_row_id','native_pts','shot_id','packet_uv','source_uv','joint_material_index','target_id','qualification_gate','qualification_receipt','physical_instance_review','journal_ref')}
        _refs({key:brief[key]for key in ('qualification_gate','physical_instance_review','journal_ref')},resolved['anchor_inventory'].parent,seen)
        proofs.append(dict(source_row=row,qualification_receipt=receipt,replay=replayed,candidate=candidate,anchor=brief,ownership=episode,path_index=path,prior_inactive_row_indices=prior))
    return proofs


def assert_known_entity_prefix_qualification(packet,packet_path,*,verified_dependency_cache=None,parent_document=None,certificate=None):
    seen=set()if verified_dependency_cache is None else verified_dependency_cache;path=Path(packet_path).resolve()
    certpath,cert=_json_ref(packet['board_material_source_qualification'],path.parent,seen)if certificate is None else certificate
    if cert.get('schema')!=SCHEMA:raise ValueError('Unsupported known entity prefix certificate')
    parentpath=_load_ref(cert['parent_packet'],Path(certpath).parent,seen);parent=parent_document if parent_document is not None else json.loads(parentpath.read_text(encoding='utf8'))
    assert_board_source_qualification_current(parent,parentpath,verified_dependency_cache=seen)
    arrays=_prefix(packet,parent,path,parentpath,cert,seen);bundle=_load_ref(cert['actual_append_source_bundle'],Path(certpath).parent,seen)
    proofs=_source_proof(packet,parent,path,parentpath,bundle,arrays,seen)
    if certificate is None:
        native=list(read_jsonl(_load_ref(cert['qualified_rows_native'],Path(certpath).parent,seen)))
        if proofs!=native:raise ValueError('Known entity native qualification ledger differs from actual full source proof')
    return proofs


def finalize_known_entity_tracklet_prefix_certificate(parent_packet_path,appended_packet_document,bundle_path,output_certificate_path,*,parent_document=None):
    parentpath=Path(parent_packet_path).resolve();out=Path(output_certificate_path).resolve();bundlepath=Path(bundle_path).resolve();packet=appended_packet_document
    if out.exists():raise FileExistsError('Known entity prefix certificate is immutable')
    if _ref(__file__)!=EXECUTED_MODULE_REF:raise ValueError('Known entity source issuer changed during execution')
    parent=parent_document if parent_document is not None else json.loads(parentpath.read_text(encoding='utf8'))
    cert=dict(schema=SCHEMA,parent_packet=_ref(parentpath),original_raw_rows=int(parent['row_count']),raw_rows=int(packet['row_count']),appended_raw_rows=int(packet['row_count'])-int(parent['row_count']),actual_append_source_bundle=_ref(bundlepath),membership_array_sha256={k:packet['arrays'][k]['sha256']for k in BOUND_ARRAYS},issued_arrays=packet['arrays'],reader_module_refs=[EXECUTED_MODULE_REF],inherited_qualification_rescanned=True,old_raw_and_old_UV_unchanged=True,new_statistical_precision_invented=False)
    proofs=assert_known_entity_prefix_qualification(packet,out.parent/'packet.json',parent_document=parent,certificate=(out,cert))
    native=out.parent/'qualified_source_rows_native.jsonl'
    if native.exists():raise FileExistsError('Known entity native qualification ledger is immutable')
    with native.open('w',encoding='utf8')as f:
        for row in proofs:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    cert['qualified_rows_native']=_ref(native);write_json(out,cert);packet['board_material_source_qualification']=_ref(out);packet['all_new_source_nuisance_requires_fresh_physical_capture']=True
    return packet
