"""Explicit RGB-qualified original motion rows on the current finite board.

No observation is appended. Explicit original unknown-depth FIT rows change
their physical material operator. Their native UV, clock, sigma and profile
buffers stay intact. Current course intersections/planes are geometry, never
fixed observed XYZ. The same plane/extent/J route runs before actor profiling.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from functools import partial
import numpy as np
from shared_board_course_candidate.board_course_material import vertical_plane_value_jacobian
from secondary_optics_candidate.secondary_optics_graph import SecondaryOpticsFixedXYZRinkBlock
from .native_board_track_source import current_far_board_points

SCHEMA = 'rinkspace.original-motion-source-qualified-far-board-role-admission.v1'
LOW_SCHEMA = 'rinkspace.original-low-motion-source-qualified-far-board-role-admission.v1'
LOW_REVIEW_SCHEMA = 'rinkspace.original-low-FIT-source-qualified-far-board-role-admission.v1'
COMBINED_SCHEMA = 'rinkspace.original-motion-multiple-block-far-board-role-admission.v1'
SECONDARY_REVIEW_SCHEMA = 'rinkspace.original-secondary-motion-source-qualified-far-board-role-admission.v1'
PROGRAMME_REVIEW_SCHEMA = 'rinkspace.original-programme-FIT-source-qualified-far-board-role-admission.v1'
PROGRAMME_QUALIFICATION_SCHEMA = 'rinkspace.original-programme-FIT-source-only-static-board-material-qualification.v1'
LOW_CANDIDATE_SHA = 'd28e4e1ee12766db14a390a69e5a91f4c71f12b834a8ec68ae1cf9a2afc1cc05'
EXPECTED_GLOBAL_INDICES = (27476,27480,27483,27484,27485,27487,27490,27491,
                           27493,27494,27496,27498,27499,27502,27503)
SOURCE_GRAPH_SHA = '71af8c55d5598ba51a59e5f8ac8a6d0f86ed893153a18593d109a3e033f89ae4'
PAIR = (230509824,230510080)
SOURCE_SHA = 'f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_role_admission(candidate_path, output_path, *, authority):
    if not isinstance(authority,str) or not authority.strip():
        raise ValueError('Explicit authorized original-source material-role correction required')
    source=Path(candidate_path).resolve();candidate=json.loads(source.read_text(encoding='utf8'))
    if candidate.get('schema')==PROGRAMME_QUALIFICATION_SCHEMA:
        if (candidate.get('source_video_sha256')!=SOURCE_SHA or candidate.get('time_base')!='1/15360'
                or candidate.get('new_query_or_scalar_rows')!=0
                or not candidate.get('original_UV_PTS_sigma_and_correspondence_unchanged')
                or not candidate.get('HELD_or_diagnostic_tracks_not_promoted_to_FIT')
                or candidate.get('fixed_metric_depth_world_XYZ_or_plane_supplied')is not False):
            raise ValueError('Original programme FIT source-only material qualification must preserve exact queries')
        packet=deepcopy(candidate)
        packet.update(schema=PROGRAMME_REVIEW_SCHEMA,status='explicit_original_motion_source_role_revision_authorized',
            new_source_scalar_rows=0,admission=dict(authority=authority,parent_candidate=dict(path=str(source),sha256=_sha(source)),
                unknown_depth_operator_changed_explicitly=True,source_r_invariance_not_claimed=True,source_scalar_rows_added=0))
        # Source qualification is authenticated before an admitted packet is
        # written. Runtime installation repeats the current bundle check.
        _programme_review_records(packet)
        return _write_immutable_packet(packet,output_path,len(packet['admitted_source_only_rows']))
    if candidate.get('schema')=='rinkspace.original-low-FIT-source-only-static-board-material-qualification.v1':
        _validate_low_review_qualification(candidate)
        packet=deepcopy(candidate)
        packet.update(schema=LOW_REVIEW_SCHEMA,status='explicit_original_motion_source_role_revision_authorized',new_source_scalar_rows=0,
            admission=dict(authority=authority,parent_candidate=dict(path=str(source),sha256=_sha(source)),
                unknown_depth_operator_changed_explicitly=True,source_r_invariance_not_claimed=True,source_scalar_rows_added=0))
        return _write_immutable_packet(packet,output_path,len(candidate['admitted_source_only_rows']))
    if candidate.get('schema')=='rinkspace.original-secondary-FIT-source-only-static-board-material-qualification.v1':
        if (candidate.get('source_video_sha256')!=SOURCE_SHA or candidate.get('time_base')!='1/15360'
                or candidate.get('new_query_or_scalar_rows')!=0 or not candidate.get('original_UV_PTS_sigma_and_correspondence_unchanged')
                or not candidate.get('HELD_or_diagnostic_tracks_not_promoted_to_FIT')):
            raise ValueError('Explicit original secondary source-only RGB material qualification required')
        packet=deepcopy(candidate)
        packet.update(schema=SECONDARY_REVIEW_SCHEMA,status='explicit_original_motion_source_role_revision_authorized',new_source_scalar_rows=0,
            admission=dict(authority=authority,parent_candidate=dict(path=str(source),sha256=_sha(source)),
                unknown_depth_operator_changed_explicitly=True,source_r_invariance_not_claimed=True,source_scalar_rows_added=0))
        return _write_immutable_packet(packet,output_path,len(candidate['admitted_source_only_rows']))
    if candidate.get('schema')=='rinkspace.original-low-motion-far-board-material-role-source-candidate.v1':
        if (_sha(source)!=LOW_CANDIDATE_SHA or candidate.get('source_video_sha256')!=SOURCE_SHA
                or candidate.get('source_rows_added')!=0 or len(candidate['records'])!=53):
            raise ValueError('Exact independently RGB-reviewed fifty-three original low FIT queries required')
        packet=deepcopy(candidate)
        packet.update(schema=LOW_SCHEMA,status='explicit_original_motion_source_role_revision_authorized',
            new_source_scalar_rows=0,source_role_revision_adopted=True,
            admission=dict(authority=authority,parent_candidate=dict(path=str(source),sha256=_sha(source)),
                unknown_depth_operator_changed_explicitly=True,source_r_invariance_not_claimed=True,source_scalar_rows_added=0))
        return _write_immutable_packet(packet,output_path,53)
    if (candidate.get('schema')!='rinkspace.original-motion-explicit-source-material-role-candidate.v1'
            or candidate.get('source_video_sha256')!=SOURCE_SHA
            or candidate.get('time_base')!='1/15360'
            or candidate.get('new_source_scalar_rows')!=0
            or tuple(r['original_global_bundle_index']for r in candidate['records'])!=EXPECTED_GLOBAL_INDICES):
        raise ValueError('Frozen fifteen actual original source rows required')
    packet=deepcopy(candidate)
    packet.update(schema=SCHEMA,status='explicit_original_motion_source_role_revision_authorized',
        source_role_revision_adopted=True,
        admission=dict(authority=authority,parent_candidate=dict(path=str(source),sha256=_sha(source)),
            source_candidate_not_overwritten=True,unknown_depth_operator_changed_explicitly=True,
            source_r_invariance_not_claimed=True,source_scalar_rows_added=0))
    for row in packet['records']:row['status']='qualified_current_far_inner_board_material'
    return _write_immutable_packet(packet,output_path,15)


def _write_immutable_packet(packet,output_path,count):
    target=Path(output_path).resolve()
    serialized=json.dumps(packet,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if target.exists():
        if target.read_text(encoding='utf8')!=serialized:raise ValueError('Immutable role-admission packet differs')
    else:target.write_text(serialized,encoding='utf8')
    return dict(path=str(target),sha256=_sha(target),original_typed_points=count,new_scalar_rows=0)


def combine_role_admissions(packet_paths,output_path,*,authority):
    if not isinstance(authority,str)or not authority.strip():raise ValueError('Explicit source-role correction authority required')
    bindings=[OriginalMotionFarBoardRoleBinding(p)for p in packet_paths]
    if [sorted({r['source_role_block']for r in b.records})for b in bindings[:2]]!=[['secondary'],['low']]:
        raise ValueError('Preserved original secondary fifteen followed by admitted low fifty-three required')
    packet=dict(schema=COMBINED_SCHEMA,status='explicit_original_motion_source_role_revision_authorized',
        source_video_sha256=SOURCE_SHA,time_base='1/15360',new_source_scalar_rows=0,authority=authority,
        source_packets=[dict(path=str(b.packet_path),sha256=b.binding_id)for b in bindings],
        original_unknown_motion_points_retyped=len({(r['source_role_block'],r['original_global_bundle_index'])for b in bindings for r in b.records}),
        original_secondary_fifteen_preserved=True,source_UV_PTS_sigma_profiles_and_rows_preserved=True,
        source_r_invariance_not_claimed=True)
    return _write_immutable_packet(packet,output_path,packet['original_unknown_motion_points_retyped'])


_LOW_SOURCE_FIELDS = ('original_raw_cut5_point_index','original_raw_pair_index','original_raw_pair_local_index',
    'original_selected_cut5_row','original_full_low_bundle_row','current_original_prefix_bundle_index',
    'source_pts','target_pts','source_native_pair_pts','source_uv','target_uv','source_sigma_px')
_LOW_GENERIC_SOURCE_FIELDS = ('shot_index','original_raw_point_index','original_raw_pair_index','original_raw_pair_local_index',
    'original_selected_row','original_full_low_bundle_row','current_original_prefix_bundle_index',
    'source_pts','target_pts','source_native_pair_pts','source_uv','target_uv','source_sigma_px')


def _validate_low_review_qualification(packet):
    """Bind full-cut source roles to exact original ledgers and actual RGB reviews."""
    if (packet.get('source_video_sha256')!=SOURCE_SHA or packet.get('time_base')!='1/15360'
            or packet.get('new_query_or_scalar_rows')!=0
            or not packet.get('original_UV_PTS_sigma_and_correspondence_unchanged')
            or not packet.get('HELD_or_diagnostic_tracks_not_promoted_to_FIT')):
        raise ValueError('Original low FIT material qualification must preserve measurements and rows')
    candidate_ref=packet['source_candidate_packet'];receipt_ref=packet['actual_RGB_receipt']
    for ref in (candidate_ref,receipt_ref,*packet['human_material_reviews']):
        if _sha(ref['path'])!=ref['sha256']:raise ValueError('Frozen original low source/RGB qualification changed')
    candidate=json.loads(Path(candidate_ref['path']).read_text(encoding='utf8'))
    receipt=json.loads(Path(receipt_ref['path']).read_text(encoding='utf8'))
    generic=candidate.get('schema')=='rinkspace.original-low-FIT-full-cut-source-only-board-overlap-candidate.v1'
    if generic:
        shot=int(candidate['shot_index'])
        if (packet.get('shot_index')!=shot or receipt.get('shot_index')!=shot or candidate.get('block')!='low'
                or candidate.get('original_UV_sigma_native_actual_PTS_and_prefix_raw_indices_bitexact')is not True
                or int(candidate['full_original_cut_offset'])<0):
            raise ValueError('Exact original low shot and full-source prefix offset required')
        candidate_preserved=candidate.get('source_measurements_roles_weights_modified')is False
        receipt_schema=f'rinkspace.original-low-cut{shot}-region-candidate-actual-RGB-review-sheets.v1'
        source_fields=_LOW_GENERIC_SOURCE_FIELDS
    else:
        candidate_preserved=(candidate.get('schema')=='rinkspace.original-low-FIT-full-cut5-source-only-board-overlap-candidate.v1'
            and candidate.get('source_measurements_or_roles_modified')is False)
        receipt_schema='rinkspace.original-low-cut5-region-candidate-actual-RGB-review-sheets.v1'
        source_fields=_LOW_SOURCE_FIELDS
    if (not candidate_preserved
            or candidate.get('source_video_sha256')!=SOURCE_SHA or candidate.get('time_base')!='1/15360'
            or candidate.get('new_source_rows')!=0 or candidate.get('HELD_promoted') is not False
            or receipt.get('schema')!=receipt_schema
            or receipt['source_candidates']!=candidate_ref
            or not receipt.get('original_UV_prefix_index_sigma_PTS_bitexact')
            or receipt.get('source_rows_UV_sigma_PTS_weights_and_runtime_target_modified') is not False
            or receipt.get('automatically_material_admitted_points')!=0):
        raise ValueError('Exact source-only original low candidate and actual RGB receipt required')
    for field in ('original_source_ledger','original_raw_motion','original_selected_rows','clock_alias_authority'):
        if packet[field]!=candidate[field]:raise ValueError('Low qualification numeric authority changed')
    sheets={int(s['sheet_index']):s for s in receipt['sheets']}
    for sheet in sheets.values():
        if _sha(sheet['path'])!=sheet['sha256']:raise ValueError('Actual low source/target RGB sheet changed')
    reviewed={}
    for ref in packet['human_material_reviews']:
        review=json.loads(Path(ref['path']).read_text(encoding='utf8'))
        schema=review.get('schema')
        if schema=='rinkspace.original-low-FIT-actual-RGB-source-only-material-review.v1':
            if review['source_RGB_receipt']!=receipt_ref or review['uncertain_source_only_rows'] or review['remaining_sheet_indices']:
                raise ValueError('Incomplete actual low endpoint material review')
            rows=review['qualified_source_only_rows']
            qualified=lambda row:row.get('material_identity_review')=='actual_both_endpoints_static_far_board_printed_material'
        elif schema=='independent_cut5_middle_actual_RGB_material_review_SOURCE_ONLY_v1':
            if (review['source_receipt']!=receipt_ref['path'] or review['source_receipt_sha256']!=receipt_ref['sha256']
                    or review['unknown_points'] or review['rejected_points']):
                raise ValueError('Incomplete independent low endpoint material review')
            rows=review['qualified_material_points']
            qualified=lambda row:(row.get('independent_source_material_review',{}).get('actual_source_and_target_endpoint_visually_reviewed') is True
                and row['independent_source_material_review'].get('selected_point_on_people_ice_UI_near_equipment') is False
                and row['independent_source_material_review'].get('status')=='confirmed_static_far_board_print_or_panel_at_both_actual_native_RGB_endpoints')
        elif schema==(f'rinkspace.original-low-cut{shot}-human-source-material-qualification.v1'if generic
                      else 'rinkspace.original-low-cut5-human-source-material-qualification.v1'):
            if (review['source_RGB_receipt']!=receipt_ref or any(review[k] for k in
                    ('unknown_material_points','nearfield_equipment_points','people_or_UI_or_ice_points'))):
                raise ValueError('Incomplete root low endpoint material review')
            rows=review['records']
            if generic:
                if review.get('original_source_row_UV_sigma_PTS_split_weights_changed') is not False:
                    raise ValueError('Original root low source literals must remain unchanged')
                qualified=lambda row:(row.get('both_end_material_reviewed') is True
                    and row.get('material_identity_review')=='qualified_far_board_print_both_actual_RGB_ends'
                    and row.get('material_family')=='static_far_board_printed_ISU_OrientalBio_or_white_board_face'
                    and row.get('subpixel_correspondence_certified') is False)
            else:
                qualified=lambda row:(row.get('both_end_material_reviewed') is True and row.get('material_role')=='board_face_far'
                    and row.get('source_UV_sigma_PTS_weights_split_modified') is False)
        else:raise ValueError('Explicit actual-RGB human low material review required')
        for row in rows:
            i=int(row['current_original_prefix_bundle_index'])
            if i in reviewed or not qualified(row):raise ValueError('Duplicate or unqualified low original point review')
            reviewed[i]=row
    original={int(r['current_original_prefix_bundle_index']):r for r in receipt['point_to_pair_sheet']}
    rows=packet['admitted_source_only_rows']
    if (len(original)!=receipt['selected_original_points'] or set(reviewed)!=set(original)
            or len(rows)!=packet['source_qualified_far_board_count'] or len(rows)!=len(original)
            or {int(r['current_original_prefix_bundle_index'])for r in rows}!=set(original)):
        raise ValueError('All frozen low candidate endpoints require complete material qualification exactly once')
    for row in rows:
        i=int(row['current_original_prefix_bundle_index']);source=original[i]
        if (any(row[k]!=source[k] or reviewed[i][k]!=source[k]for k in source_fields)
                or not row.get('both_end_original_native_RGB_material_reviewed') or not row.get('source_only_people_clear')
                or row.get('material_role_review')!='confirmed_static_current_far_board_printed_material'
                or row.get('split')!='fit'):
            raise ValueError('Actual low FIT source literals or human material qualification changed')
    return rows


def _low_records(packet):
    """Authenticate the exact original UV/index/clock ledgers, never a pickle."""
    parent=packet['admission']['parent_candidate']
    full_review=packet.get('schema')==LOW_REVIEW_SCHEMA
    if full_review:
        if _sha(parent['path'])!=parent['sha256']:raise ValueError('Frozen low full-cut material qualification changed')
        _validate_low_review_qualification(packet)
    elif parent['sha256']!=LOW_CANDIDATE_SHA or _sha(parent['path'])!=LOW_CANDIDATE_SHA:
        raise ValueError('Frozen source-only low RGB qualification changed')
    refs=[packet[k]for k in ('original_source_ledger','original_raw_motion','original_selected_rows','clock_alias_authority')]
    for ref in refs:
        if _sha(ref['path'])!=ref['sha256']:raise ValueError('Original low numeric ledger or clock changed')
    for frame in packet.get('raw_RGB_source_frames',[]) if full_review else packet['raw_RGB_source_frames']:
        if _sha(frame['path'])!=frame['sha256']:raise ValueError('Original low native RGB changed')
    with np.load(refs[0]['path'],allow_pickle=False)as z:
        source={k:z[k].copy()for k in z.files}
    with np.load(refs[1]['path'],allow_pickle=False)as z:
        raw={k:z[k].copy()for k in ('pair_pts','offsets','xy0','xy1')}
    with np.load(refs[2]['path'],allow_pickle=False)as z:ids=z['original_global_row_indices'].copy()
    with np.load(refs[3]['path'],allow_pickle=False)as z:
        aliases=dict(zip(map(int,z['pts']),map(int,z['exposure_representative_pts'])))
    active=source['active_original_global_index'];records=deepcopy(packet['admitted_source_only_rows']if full_review else packet['records'])
    count=packet['source_qualified_far_board_count']if full_review else 53
    candidate={}if not full_review else json.loads(Path(packet['source_candidate_packet']['path']).read_text(encoding='utf8'))
    generic=candidate.get('schema')=='rinkspace.original-low-FIT-full-cut-source-only-board-overlap-candidate.v1'
    offset=int(candidate['full_original_cut_offset'])if generic else 31987
    raw_field='original_raw_point_index'if generic else 'original_raw_cut5_point_index'
    selected_field='original_selected_row'if generic else 'original_selected_cut5_row'
    if (len(records)!=count or len(active)!=434991
            or (not generic and len(ids)!=319568)
            or (generic and len(ids)!=int(candidate['summary']['selected_original_count']))):
        raise ValueError('Original low prefix inventory changed')
    for row in records:
        i=int(row['current_original_prefix_bundle_index']);g=int(row[raw_field])
        j=int(row['original_raw_pair_index']);local=int(row['original_raw_pair_local_index']);selected=int(row[selected_field])
        full=int(row['original_full_low_bundle_row']);pair=tuple(map(int,raw['pair_pts'][j]))
        if ((generic and row['shot_index']!=candidate['shot_index'])
                or full!=offset+selected or full!=int(active[i])or int(ids[selected])!=g
                or g!=int(raw['offsets'][j])+local or pair!=tuple(row['source_native_pair_pts'])
                or (aliases[pair[0]],aliases[pair[1]])!=(row['source_pts'],row['target_pts'])
                or not np.array_equal(raw['xy0'][g],row['source_uv'])or not np.array_equal(raw['xy1'][g],row['target_uv'])
                or not np.array_equal(source['source_uv_a'][full],row['source_uv'])or not np.array_equal(source['source_uv_b'][full],row['target_uv'])
                or not np.array_equal(source['original_sigma_xy'][full],row['source_sigma_px'])or row['split']!='fit'
                or not row['both_end_original_native_RGB_material_reviewed']or not row['source_only_people_clear']):
            raise ValueError('Original low FIT UV/index/clock/sigma qualification changed')
        row.update(original_global_bundle_index=i,source_role_block='low',physical_material=row['source_only_material_description'])
    if len({r['original_global_bundle_index']for r in records})!=count:raise ValueError('Duplicate original low FIT query')
    return records


def _secondary_review_records(packet):
    parent=packet['admission']['parent_candidate']
    if _sha(parent['path'])!=parent['sha256']:raise ValueError('Frozen two-endpoint RGB source material qualification changed')
    candidate=packet['source_candidate_packet']
    if _sha(candidate['path'])!=candidate['sha256']:raise ValueError('Original query/source-only overlap ledger changed')
    graph=json.loads(Path(candidate['path']).read_text(encoding='utf8'))['source_graph']
    if graph['sha256']!=SOURCE_GRAPH_SHA or _sha(graph['path'])!=SOURCE_GRAPH_SHA:raise ValueError('Exact original secondary FIT graph required')
    records=deepcopy(packet['admitted_source_only_rows'])
    if len(records)!=packet['source_qualified_far_board_count']:raise ValueError('Reviewed source-query count changed')
    with np.load(graph['path'],allow_pickle=False)as archive:
        for r in records:
            i=int(r['original_global_bundle_index']);j=int(r['original_pair_index']);local=int(r['original_local_index'])
            crop=r['pointwise_original_native_RGB']
            if (int(archive['offset'][j])+local!=i or tuple(archive['pair_pts'][j])!=(r['source_pts'],r['target_pts'])
                    or not np.array_equal(archive['source_uv'][i],r['original_source_uv'])
                    or not np.array_equal(archive['target_uv'][i],r['original_target_uv'])
                    or not r['source_only_people_clear']or not r['material_identity_reviewed_on_both_source_endpoints']
                    or not r['model_projection_or_residual_not_used_for_role']
                    or r['material_role_review']!='confirmed_static_current_far_board_printed_material'
                    or not np.array_equal(crop['source_uv'],r['original_source_uv'])
                    or not np.array_equal(crop['target_uv'],r['original_target_uv'])or _sha(crop['path'])!=crop['sha256']):
                raise ValueError('Exact original secondary FIT index/UV/RGB/physical-material certificate changed')
            r.update(source_role_block='secondary',source_uv=r['original_source_uv'],target_uv=r['original_target_uv'],
                source_sigma_px=[2.,2.],split='fit',physical_material='both-end native RGB confirmed stationary printed far inner board')
    if len({r['original_global_bundle_index']for r in records})!=len(records):raise ValueError('Duplicate reviewed secondary query')
    return records


def _programme_review_records(packet):
    """Exact source-only programme queries, indexed before input compaction.

    Only source admissions, source clocks and original duplicate maps are
    read. This adds no camera, world or finite-plane producer.
    """
    hashes={}
    def source_sha(path):
        p=Path(path);s=p.stat();key=(str(p.resolve()),s.st_size,s.st_mtime_ns)
        if key not in hashes:hashes[key]=_sha(p)
        return hashes[key]
    name=packet['source_role_block']
    if name not in ('main1','main7','secondary','low','main9','main4'):
        raise ValueError('Original programme source block identity required')
    references=[packet['admission']['parent_candidate'],packet['source_candidate_packet'],
        packet['original_admission_cache'],packet['original_source_clock'],packet['original_input_compaction_map']]
    for reference in references:
        if source_sha(reference['path'])!=reference['sha256']:
            raise ValueError('Original programme source qualification/input authority changed')
    candidate=json.loads(Path(packet['source_candidate_packet']['path']).read_text(encoding='utf8'))
    for key in ('source_role_block','original_admission_cache','original_source_clock','original_input_compaction_map'):
        if candidate[key]!=packet[key]:raise ValueError('Source candidate original input identity changed: '+key)
    records=deepcopy(packet['admitted_source_only_rows'])
    if len(records)!=packet['source_qualified_far_board_count']:
        raise ValueError('Actual RGB-qualified original query count changed')
    with np.load(packet['original_admission_cache']['path'],allow_pickle=False)as source,\
            np.load(packet['original_source_clock']['path'],allow_pickle=False)as clock,\
            np.load(packet['original_input_compaction_map']['path'],allow_pickle=False)as mapping:
        stamps=np.asarray(clock['canonical_pts'])
        source={key:source[key]for key in ('active_original_row_index','endpoint_index_a','endpoint_index_b',
            'source_uv_a','source_uv_b','source_original_sigma_xy','offsets','pairs','static_factor_active','active_finite_planes')}
        mapping={key:mapping[key]for key in ('keep_old_point_indices','old_point_to_compact_point',
            'removed_exact_duplicate_old_point_indices','preserved_equal_source_point_for_each_removed')}
        active=source['active_original_row_index'];keep=mapping['keep_old_point_indices'];inverse=mapping['old_point_to_compact_point']
        if (len(inverse)!=len(active)or not np.array_equal(inverse[keep],np.arange(len(keep)))
                or np.any(np.diff(keep)<=0)or hashlib.sha256(stamps.tobytes()).hexdigest()!=packet['original_source_clock']['canonical_pts_array_sha256']):
            raise ValueError('Exact original source clock/point compaction bijection required')
        duplicate_keeper=dict(zip(map(int,mapping['removed_exact_duplicate_old_point_indices']),
            map(int,mapping['preserved_equal_source_point_for_each_removed'])))
        for row in records:
            i=int(row['original_global_bundle_index']);compact=int(row['current_compact_bundle_index'])
            raw=int(row['original_admission_row_index']);j=int(row['original_pair_index']);local=int(row['original_local_index'])
            crop=row['pointwise_original_native_RGB']
            removed=compact<0
            representative=duplicate_keeper.get(i)if removed else i
            if removed:
                if (compact!=-1 or representative is None or int(inverse[i])!=-1
                        or row.get('current_runtime_duplicate_keeper_precompaction_bundle_index')!=representative
                        or row.get('current_runtime_equivalent_compact_bundle_index')!=int(inverse[representative])
                        or int(inverse[representative])<0):
                    raise ValueError('Removed original programme source needs its exact historical duplicate keeper')
                keeper_raw=int(active[representative])
                for field in ('source_uv_a','source_uv_b','endpoint_index_a','endpoint_index_b','source_original_sigma_xy'):
                    if not np.array_equal(source[field][raw],source[field][keeper_raw]):
                        raise ValueError('Source-qualified historical duplicate literals differ')
                peer=next((r for r in records if r['original_global_bundle_index']==representative),None)
                if peer is None or any(peer[key]!=row[key]for key in ('source_pts','target_pts','source_uv','target_uv','source_sigma_px','split','pointwise_original_native_RGB')):
                    raise ValueError('Both exact original duplicate indices require the same two-endpoint RGB qualification')
            pair=(int(stamps[source['endpoint_index_a'][raw]]),int(stamps[source['endpoint_index_b'][raw]]))
            if (row['source_role_block']!=name or i<0 or i>=len(active)or compact>=len(keep)
                    or (not removed and int(keep[compact])!=i)or int(inverse[i])!=compact or int(active[i])!=raw
                    or int(source['offsets'][j])+local!=raw or tuple(source['pairs'][j])!=pair
                    or pair!=(int(row['source_pts']),int(row['target_pts']))
                    or not bool(source['static_factor_active'][raw])or np.any(source['active_finite_planes'][i])
                    or not np.array_equal(source['source_uv_a'][raw],row['source_uv'])
                    or not np.array_equal(source['source_uv_b'][raw],row['target_uv'])
                    or not np.array_equal(source['source_original_sigma_xy'][raw],row['source_sigma_px'])
                    or row['split']!='fit' or not row['source_only_people_clear']
                    or not row['both_end_original_native_RGB_material_reviewed']
                    or not row['material_identity_reviewed_on_both_source_endpoints']
                    or not row['model_projection_or_residual_not_used_for_role']
                    or row['physical_world_point_or_measured_depth_supplied']is not False
                    or row['material_role_review']!='confirmed_static_current_far_board_printed_material'
                    or not np.array_equal(crop['source_uv'],row['source_uv'])
                    or not np.array_equal(crop['target_uv'],row['target_uv'])or source_sha(crop['path'])!=crop['sha256']):
                raise ValueError('Exact original programme FIT index/UV/sigma/RGB/material certificate changed')
            for stamp,frame in zip(pair,crop['source_frames']):
                if (frame['native_pts']!=stamp or frame['time_base']!='1/15360'
                        or frame['source_video_sha256']!=SOURCE_SHA or source_sha(frame['path'])!=frame['sha256']):
                    raise ValueError('Original programme exact both-end native RGB identity changed')
            row['physical_material']='Both-end original native RGB confirmed stationary print or white far inner board'
    if len({r['original_global_bundle_index']for r in records})!=len(records):
        raise ValueError('Duplicate original programme FIT material query')
    return records


class OriginalMotionFarBoardRoleBinding:
    """Small source/material certificate; never retains a programme graph."""
    def __init__(self, packet_path):
        self.packet_path=Path(packet_path).resolve()
        self.packet=json.loads(self.packet_path.read_text(encoding='utf8'))
        p=self.packet
        if (p.get('schema')not in (SCHEMA,LOW_SCHEMA,LOW_REVIEW_SCHEMA,SECONDARY_REVIEW_SCHEMA,PROGRAMME_REVIEW_SCHEMA,COMBINED_SCHEMA) or p.get('status')!='explicit_original_motion_source_role_revision_authorized'
                or p.get('source_video_sha256')!=SOURCE_SHA or p.get('time_base')!='1/15360'
                or p.get('new_source_scalar_rows')!=0):
            raise ValueError('Explicit admitted original-motion material-role packet required')
        if p['schema']!=SCHEMA:
            if p['schema']in(LOW_SCHEMA,LOW_REVIEW_SCHEMA):self.records=_low_records(p)
            elif p['schema']==SECONDARY_REVIEW_SCHEMA:self.records=_secondary_review_records(p)
            elif p['schema']==PROGRAMME_REVIEW_SCHEMA:self.records=_programme_review_records(p)
            else:
                self.records=[];seen={}
                for ref in p['source_packets']:
                    if _sha(ref['path'])!=ref['sha256']:raise ValueError('Admitted child source-role packet changed')
                    child=OriginalMotionFarBoardRoleBinding(ref['path'])
                    for row in child.records:
                        key=(row['source_role_block'],int(row['original_global_bundle_index']))
                        if key in seen:
                            prior=self.records[seen[key]]
                            if any(prior[k]!=row[k]for k in ('source_pts','target_pts','source_uv','target_uv','source_sigma_px','split')):
                                raise ValueError('Conflicting physical source-role certificate on the same original query')
                            prior.setdefault('additional_source_role_qualification_packets',[]).append(deepcopy(ref))
                        else:seen[key]=len(self.records);self.records.append(deepcopy(row))
                if len(self.records)!=p['original_unknown_motion_points_retyped']:
                    raise ValueError('Exact frozen unique original-source-role count changed')
            self._finish_certificate()
            return
        graph=p['original_motion_graph']
        if graph['sha256']!=SOURCE_GRAPH_SHA or _sha(graph['path'])!=SOURCE_GRAPH_SHA:
            raise ValueError('Actual original source graph bytes changed')
        for frame in p['source_frames']:
            if _sha(frame['path'])!=frame['sha256']:raise ValueError('Original native source RGB bytes changed')
        self.records=deepcopy(p['records'])
        self.indices=np.asarray([r['original_global_bundle_index']for r in self.records],int)
        if tuple(self.indices)!=EXPECTED_GLOBAL_INDICES:raise ValueError('Exact frozen fifteen original indices required')
        self.source_cameras={};self.source_camera_signature=None
        self.binding_id=_sha(self.packet_path)
        with np.load(graph['path'],allow_pickle=False)as archive:
            for r in self.records:
                i=int(r['original_global_bundle_index']);j=int(r['original_pair_index']);local=int(r['original_local_index'])
                if (tuple(archive['pair_pts'][j])!=PAIR or int(archive['offset'][j])+local!=i
                        or tuple((r['source_pts'],r['target_pts']))!=PAIR
                        or not np.array_equal(archive['source_uv'][i],r['source_uv'])
                        or not np.array_equal(archive['target_uv'][i],r['target_uv'])
                        or r['split']!='fit' or not r.get('both_end_original_source_reviewed')
                        or not r.get('source_only_people_clear')
                        or not np.array_equal(r['source_sigma_px'],[2.,2.])):
                    raise ValueError('Source index/direction/UV/role/sigma certificate changed')
        for row in self.records:row['source_role_block']='secondary'
        self._finish_certificate()

    def _finish_certificate(self):
        self.indices=np.asarray([r['original_global_bundle_index']for r in self.records],int)
        self.source_cameras={};self.source_camera_signature=None;self.source_camera_signatures={}
        self.binding_id=_sha(self.packet_path)
        byblock={name:[int(r['original_global_bundle_index'])for r in self.records if r['source_role_block']==name]
            for name in sorted({r['source_role_block']for r in self.records})}
        if len({(r['source_role_block'],int(r['original_global_bundle_index']))for r in self.records})!=len(self.records):
            raise ValueError('Duplicate block/original-query source role')
        self.source_row_groups={}
        for i,row in enumerate(self.records):self.source_row_groups.setdefault(int(row['source_pts']),[]).append(i)
        self.source_row_groups={p:np.asarray(rows,int)for p,rows in self.source_row_groups.items()}
        self.source_stamps_by_block={name:tuple(sorted({int(r['source_pts'])for r in self.records
            if r['source_role_block']==name}))for name in byblock}
        self.source_uv=np.asarray([r['source_uv']for r in self.records],float)
        self._current_binding_cache=None
        self._facet_course_cache=None
        self._source_camera_map_cache={}
        self.exact_cache_statistics=dict(binding_hits=0,binding_misses=0,camera_hits=0,camera_misses=0,
            facet_group_hits=0,facet_group_misses=0,unchanged_facet_binding_hits=0)
        self.contract=dict(schema=self.packet['schema'],packet_path=str(self.packet_path),packet_sha256=self.binding_id,
            original_global_bundle_indices=self.indices.tolist(),source_pair_pts=list(PAIR)if self.packet['schema']==SCHEMA else None,
            original_source_graph_sha256=SOURCE_GRAPH_SHA if self.packet['schema']==SCHEMA else None,
            authenticated_component_packets=self.packet.get('source_packets',[]),original_unknown_motion_points_retyped=len(self.records),
            original_global_bundle_indices_by_block=byblock,
            actual_source_pairs_by_block={name:sorted({(r['source_pts'],r['target_pts'])for r in self.records if r['source_role_block']==name})for name in byblock},
            additional_source_points_or_scalar_rows=0,old_motion_UV_sigma_order_and_clock_preserved=True,
            original_profile_buffers_not_reweighted=True,
            observation_operator_changed='positive unknown source depth profile -> current finite far inner board',
            finite_board_operator='existing current far-course source ray intersection; per-point current vertical facet',
            fixed_world_XYZ_or_measured_plane_supplied=False,
            model_residual_not_used_for_source_qualification=True,
            source_r_invariance_claimed=False,source_role_admission_before_actors_required=True,
            finite_planes_ignore_unknown_depth_floor=True,accuracy_accepted=False)

    def current_binding(self, geometry, world):
        """Current source-supported board branch and exact current plane J."""
        q=np.asarray(world,float);ng=int(geometry.parameter_count)
        if q.shape!=(ng,):raise ValueError('Complete current logical world required')
        course,cq,chain=geometry.current.print_course_binding(q)
        chain=np.asarray(chain,float)
        if chain.shape!=(6,ng):raise ValueError('Current course/logical-world derivative chain required')
        camera_bytes=[]
        for stamp in sorted(self.source_row_groups):
            camera=self.source_cameras.get(stamp)
            if camera is None:raise ValueError('Actual current source camera must bind before finite-board planes')
            camera_bytes.append(np.asarray(camera,float).tobytes())
        key=(float(geometry.scene_unit_m),np.asarray(cq,float).tobytes(),chain.tobytes(),b''.join(camera_bytes))
        cached=self._current_binding_cache
        if cached is not None and cached[0]==key:
            self.exact_cache_statistics['binding_hits']+=1
            return cached[1]
        self.exact_cache_statistics['binding_misses']+=1
        selected=self._current_source_facets(course,cq,key[:2])
        if (cached is not None and cached[0][:3]==key[:3]
                and np.array_equal(selected,cached[1]['selected_course_facets'])):
            # Current source cameras are still evaluated/applied by the block.
            # With the same course and selected finite faces, their planes,
            # extents and world Jacobians are exactly the same physical values.
            self.exact_cache_statistics['unchanged_facet_binding_hits']+=1
            self._current_binding_cache=(key,cached[1])
            return cached[1]
        bound=self._build_current_binding(geometry,q,course,cq,chain)
        if not np.array_equal(selected,bound['selected_course_facets']):
            raise ValueError('Exact grouped source-facet selection differs from original operator')
        self._current_binding_cache=(key,bound)
        return bound

    def _current_source_facets(self,course,cq,course_key):
        cached=self._facet_course_cache
        if cached is None or cached[0]!=course_key:
            bottom=np.asarray(course.sample_perimeter(cq,with_jacobian=False)[0],float)
            if np.array_equal(bottom[0],bottom[-1]):bottom=bottom[:-1]
            xy=bottom[:,:2];edge=np.roll(xy,-1,axis=0)-xy
            length2=np.einsum('ni,ni->n',edge,edge)
            cached=(course_key,bottom,xy,edge,length2,{})
            self._facet_course_cache=cached
        _,bottom,xy,edge,length2,groups=cached
        selected=np.empty(len(self.records),int)
        for stamp in sorted(self.source_row_groups):
            rows=self.source_row_groups[stamp];camera=self.source_cameras[stamp]
            camera_key=np.asarray(camera,float).tobytes();group=groups.setdefault(stamp,{})
            if camera_key in group:
                self.exact_cache_statistics['facet_group_hits']+=1
                selected[rows]=group[camera_key]
                continue
            self.exact_cache_statistics['facet_group_misses']+=1
            points=current_far_board_points(camera,self.source_uv[rows],bottom)
            t=np.clip(np.einsum('nsi,si->ns',points[:,None,:2]-xy[None,:,:],edge)/length2[None,:],0.,1.)
            distance=np.linalg.norm(points[:,None,:2]-(xy[None,:,:]+t[:,:,None]*edge[None,:,:]),axis=2)
            group[camera_key]=np.argmin(distance,axis=1)
            selected[rows]=group[camera_key]
            while len(group)>3:group.pop(next(iter(group)))
        return selected

    def _build_current_binding(self, geometry, world, course=None, cq=None, chain=None):
        """The original uncached source operator, also used for exact replay."""
        q=np.asarray(world,float);ng=int(geometry.parameter_count)
        if q.shape!=(ng,):raise ValueError('Complete current logical world required')
        if course is None:course,cq,chain=geometry.current.print_course_binding(q)
        chain=np.asarray(chain,float)
        if chain.shape!=(6,ng):raise ValueError('Current course/logical-world derivative chain required')
        bottom,top,jbottom,jtop,*_=course.sample_perimeter(cq,with_jacobian=True)
        bottom=np.asarray(bottom,float);jbottom=np.asarray(jbottom,float)
        if np.array_equal(bottom[0],bottom[-1]):bottom=bottom[:-1];jbottom=jbottom[:-1]
        source=self.source_uv
        points=np.empty((len(source),3))
        for stamp in sorted(self.source_row_groups):
            rows=self.source_row_groups[stamp]
            camera=self.source_cameras.get(stamp)
            if camera is None:raise ValueError('Actual current source camera must bind before finite-board planes')
            points[rows]=current_far_board_points(camera,source[rows],bottom)
        xy=bottom[:,:2];edge=np.roll(xy,-1,axis=0)-xy;length2=np.einsum('ni,ni->n',edge,edge)
        t=np.clip(np.einsum('nsi,si->ns',points[:,None,:2]-xy[None,:,:],edge)/length2[None,:],0.,1.)
        distance=np.linalg.norm(points[:,None,:2]-(xy[None,:,:]+t[:,:,None]*edge[None,:,:]),axis=2)
        selected=np.argmin(distance,axis=1)
        world_j=np.einsum('nik,kj->nij',jbottom,chain)
        planes=[];jplanes=[];extents=[];jextents=[];records=[]
        for source_row,i in zip(self.records,selected):
            i=int(i);j=(i+1)%len(bottom);ends=bottom[[i,j]];jac=world_j[[i,j]]
            plane,dplane=vertical_plane_value_jacobian(ends[0],ends[1],jac[0],jac[1])
            record=dict(global_bundle_index=int(source_row.get('active_global_bundle_index',source_row['original_global_bundle_index'])),
                surface_kind='finite_board_face',extent_start_xy=ends[0,:2].tolist(),extent_end_xy=ends[1,:2].tolist(),
                physical_surface_identity='source_qualified_current_far_inner_board_course',
                source_role_binding_id=self.binding_id,source_role_block=source_row['source_role_block'],
                source_native_pair_pts=[source_row['source_pts'],source_row['target_pts']],
                original_source_role='fit',material_identity=source_row['physical_material'],
                actual_source_record=deepcopy(source_row),current_course_facet_index=i,
                facet_index_is_current_geometry_not_measured_location=True)
            planes.append(plane);jplanes.append(dplane);extents.append(ends);jextents.append(jac);records.append(record)
        high=int(np.argmax(np.asarray(top)[:,2]));zj=np.asarray(jtop)[high,2]@chain
        return dict(planes=np.asarray(planes),plane_jacobian=np.asarray(jplanes),records=records,extent_world=np.asarray(extents),
            extent_world_jacobian=np.asarray(jextents),z_range_scene=np.array([0.,np.asarray(top)[high,2]]),
            z_range_world_jacobian=np.vstack((np.zeros(ng),zj)),selected_course_facets=selected)


class ExistingFinitePerimeterBinding:
    """Existing finite FIT pixels on the current whole board, not a named face.

    Reuse the existing per-source ray/facet operator.  The source pixels,
    clock, sigma and finite row inventory remain exactly those of the bundle.
    The historical plane/straight label is provenance, not a geometry domain.
    """
    current_binding = OriginalMotionFarBoardRoleBinding.current_binding
    _current_source_facets = OriginalMotionFarBoardRoleBinding._current_source_facets
    _build_current_binding = OriginalMotionFarBoardRoleBinding._build_current_binding

    def __init__(self, records):
        self.records = deepcopy(records)
        def numeric_json(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            raise TypeError('Existing finite source metadata must contain only JSON/numpy values')
        encoded = json.dumps(self.records, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, default=numeric_json).encode('utf8')
        self.binding_id = hashlib.sha256(encoded).hexdigest()
        self.source_uv = np.asarray([r['source_uv'] for r in self.records], float)
        self.source_row_groups = {}
        for i, row in enumerate(self.records):
            self.source_row_groups.setdefault(int(row['source_pts']), []).append(i)
        self.source_row_groups = {p: np.asarray(v, int) for p, v in self.source_row_groups.items()}
        self.source_cameras = {}
        self._current_binding_cache = self._facet_course_cache = None
        self.exact_cache_statistics = dict(binding_hits=0, binding_misses=0, camera_hits=0,
            camera_misses=0, facet_group_hits=0, facet_group_misses=0,
            unchanged_facet_binding_hits=0)


def _install_existing_finite_perimeter_binding(binder, role, blocks):
    rows = []
    for name, block in blocks.items():
        bundle = block.bundle
        for record in block.course_original_board_records:
            if record.get('surface_kind') != 'finite_board_face':
                raise ValueError('Existing finite-board geometry requires its unchanged physical role')
            index = int(record['global_bundle_index'])
            if not bundle.finite_mask[index]:
                raise ValueError('Existing finite pixel cannot change its source-depth/profile role')
            source_pts = int(bundle.stamps[bundle.ia[index]])
            target_pts = int(bundle.stamps[bundle.ib[index]])
            identity = deepcopy(record.get('physical_surface_identity'))
            row = dict(original_global_bundle_index=index, source_role_block=name,
                source_pts=source_pts, target_pts=target_pts,
                source_uv=np.asarray(bundle.a[index], float).tolist(),
                target_uv=np.asarray(bundle.b[index], float).tolist(),
                source_sigma_px=np.asarray(bundle.motion_sigma[2*index:2*index+2], float).tolist(),
                split='fit', physical_material=identity,
                original_finite_record=deepcopy(record))
            rows.append(row)
            # Stable provenance survives the existing active-array compaction.
            record['existing_finite_perimeter_key'] = [name, index]
        if rows:
            stamps = {int(r['source_pts']) for r in rows if r['source_role_block'] == name}
            role.source_stamps_by_block[name] = tuple(sorted(stamps | set(role.source_stamps_by_block.get(name, ()))))
    if len(rows) != 17:
        raise ValueError('The original two main1 and fifteen low finite-board pixels must remain present')
    binding = ExistingFinitePerimeterBinding(rows)
    binder._existing_finite_perimeter_binding = binding
    role.contract['existing_finite_geometry_revision'] = dict(
        existing_finite_pixels=len(rows), binding_sha256=binding.binding_id,
        source_UV_sigma_clock_and_finite_role_unchanged=True,
        geometry='same whole continuous current perimeter per-source ray/facet intersection',
        historical_region_labels_are_provenance_only=True,
        complete_current_world_parameter_chain=True, new_source_scalar_rows=0)


def _role_finite_board_records(binder, records, q, *, original_planes):
    role=binder._original_motion_far_board_role_binding
    revised=[i for i,r in enumerate(records)if r.get('source_role_binding_id')==role.binding_id]
    old=[i for i in range(len(records))if i not in revised]
    originals=np.asarray(original_planes,float)
    if originals.shape!=(len(records),4):raise ValueError('Original finite row inventory changed')
    existing = getattr(binder, '_existing_finite_perimeter_binding', None)
    if existing is None:
        legacy=binder._original_motion_far_board_records_method(binder,[records[i]for i in old],q,
            original_planes=originals[old])
    else:
        current = existing.current_binding(binder, q)
        lookup_existing = {(r['source_role_block'], int(r['original_global_bundle_index'])): i
                           for i, r in enumerate(existing.records)}
        selected = [lookup_existing[tuple(records[i]['existing_finite_perimeter_key'])] for i in old]
        legacy = dict(current)
        for field in ('planes', 'plane_jacobian', 'extent_world', 'extent_world_jacobian'):
            legacy[field] = current[field][selected]
        legacy['records'] = []
        for source_index, k in zip(old, selected):
            record = deepcopy(records[source_index])
            for field in ('extent_start_xy', 'extent_end_xy', 'current_course_facet_index',
                          'facet_index_is_current_geometry_not_measured_location'):
                record[field] = deepcopy(current['records'][k][field])
            record['geometry_model'] = 'continuous_current_perimeter_source_ray_facet'
            legacy['records'].append(record)
    if not revised:return legacy
    bound=role.current_binding(binder,q);lookup={(r['source_role_block'],int(r['global_bundle_index'])):i for i,r in enumerate(bound['records'])}
    n,ng=len(records),binder.parameter_count
    out=dict(legacy,planes=np.empty((n,4)),plane_jacobian=np.empty((n,4,ng)),records=[None]*n,
        extent_world=np.empty((n,2,3)),extent_world_jacobian=np.empty((n,2,3,ng)))
    for k,i in enumerate(old):
        for field in ('planes','plane_jacobian','extent_world','extent_world_jacobian'):out[field][i]=legacy[field][k]
        out['records'][i]=legacy['records'][k]
    for i in revised:
        k=lookup[(records[i]['source_role_block'],int(records[i]['global_bundle_index']))]
        for field in ('planes','plane_jacobian','extent_world','extent_world_jacobian'):out[field][i]=bound[field][k]
        out['records'][i]=bound['records'][k]
    out.update(z_range_scene=bound['z_range_scene'],z_range_world_jacobian=bound['z_range_world_jacobian'],
        unknown_depth_relabelled=True,current_source_qualified_original_motion_points=len(revised),
        source_role_admission_sha256=role.binding_id)
    return out


def _role_synchronize_course(block, q):
    role=block._original_motion_far_board_role_binding
    # This original base local() applies the exact physical pivot/mount/lens
    # chart without recursively calling StructuredProgrammeBlock.local().
    local=SecondaryOpticsFixedXYZRinkBlock.local(block,q)
    # Base local() has applied the actual current mount/lens calibration.
    # Include its entire numeric/model manifest: local q alone omits these.
    stamps=role.source_stamps_by_block[block.name]
    camera_key=(np.asarray(local,float).tobytes(),json.dumps(
        block.bundle.calibration.manifest(),sort_keys=True,separators=(',',':')).encode('utf8'))
    cached=role._source_camera_map_cache.get(block.name)
    if cached is not None and cached[0]==camera_key:
        cameras,signature=cached[1:]
        role.exact_cache_statistics['camera_hits']+=1
    else:
        cameras={p:np.asarray(block.bundle.camera(local,p),float).copy()for p in stamps}
        signature=b''.join(cameras[p].tobytes()for p in stamps)
        role._source_camera_map_cache[block.name]=(camera_key,cameras,signature)
        role.exact_cache_statistics['camera_misses']+=1
    if signature!=role.source_camera_signatures.get(block.name):
        block._course_current_geometry_key=None;block._course_camera_cache_key=None
    role.source_cameras.update(cameras);role.source_camera_signature=signature
    role.source_camera_signatures[block.name]=signature
    existing = getattr(block.geometry, '_existing_finite_perimeter_binding', None)
    if existing is not None:
        existing.source_cameras.update(cameras)
    return block._original_motion_far_board_synchronize_method(block,q)


def install_original_motion_far_board_roles(state, packet_path, q):
    """After original block cloning, before actors; all old rows remain present."""
    if getattr(state,'source_actors',None)is not None:
        raise ValueError('Source material roles must be installed before actors')
    if hasattr(state.geometry,'_original_motion_far_board_role_binding'):
        raise ValueError('Original motion far-board material roles already installed')
    role=OriginalMotionFarBoardRoleBinding(packet_path)
    blocks={}
    for row in role.records:
        pair=(int(row['source_pts']),int(row['target_pts']))
        owners=[state.bypts[int(state.original.aliases[p])]for p in pair]
        if owners[0]is not owners[1]or owners[0].name!=row['source_role_block']:
            raise ValueError('Exact original same-cut FIT producer required')
        block=owners[0];bundle=block.bundle;blocks[block.name]=block
        i=int(row['original_global_bundle_index'])
        if (int(bundle.stamps[bundle.ia[i]])!=pair[0]or int(bundle.stamps[bundle.ib[i]])!=pair[1]
                or not np.array_equal(bundle.a[i],row['source_uv'])or not np.array_equal(bundle.b[i],row['target_uv'])
                or not np.array_equal(bundle.motion_sigma[2*i:2*i+2],row['source_sigma_px'])
                or bundle.finite_mask[i]):
            raise ValueError('Actual original untyped FIT row identity or sigma changed')
    # The old main1 finite pixels already had a material role, so they are
    # absent from the later unknown-depth role-admission packet.
    for block in state.blocks:
        if getattr(block, 'course_original_board_records', ()):
            blocks.setdefault(block.name, block)
    binder=state.geometry
    _install_existing_finite_perimeter_binding(binder, role, blocks)
    binder._original_motion_far_board_role_binding=role
    binder._original_motion_far_board_records_method=type(binder).finite_board_records
    # A partial of a module function preserves the self cycle during pickle
    # BUILD. An instance-only bound method would resolve getattr too early.
    binder.finite_board_records=partial(_role_finite_board_records,binder)
    core=np.asarray(q,float)[:state.structured_programme_coordinate_chart.core_count]
    for block in blocks.values():
        block._original_motion_far_board_role_binding=role
        block._original_motion_far_board_synchronize_method=type(block).synchronize_course
        block.synchronize_course=partial(_role_synchronize_course,block)
        local=SecondaryOpticsFixedXYZRinkBlock.local(block,core)
        stamps=role.source_stamps_by_block[block.name]
        role.source_cameras.update({p:np.asarray(block.bundle.camera(local,p),float).copy()for p in stamps})
        binder._existing_finite_perimeter_binding.source_cameras.update(role.source_cameras)
    bound=role.current_binding(binder,core[state.geometry_parameter_indices])
    for name,block in blocks.items():
        bundle=block.bundle;selected=np.array([i for i,r in enumerate(bound['records'])if r['source_role_block']==name],int)
        new_records=[deepcopy(bound['records'][i])for i in selected];new_planes=bound['planes'][selected]
        indices=np.array([r['global_bundle_index']for r in new_records],int)
        old_records=deepcopy(block.course_original_board_records)
        old_planes=block.course_original_board_planes.copy();old_positions=block.course_board_record_positions.tolist()
        new_positions=[]
        for record in new_records:
            new_positions.append(len(bundle.finite_records));bundle.finite_records.append(deepcopy(record))
        all_records=old_records+new_records;order=np.argsort([r['global_bundle_index']for r in all_records])
        block.course_original_board_records=[all_records[i]for i in order]
        block.course_board_indices=np.asarray([r['global_bundle_index']for r in block.course_original_board_records],int)
        block.course_board_record_positions=np.asarray(old_positions+new_positions,int)[order]
        block.course_original_board_planes=np.concatenate((old_planes,new_planes))[order]
        bundle.finite_mask[indices]=True;bundle.finite_planes[indices]=new_planes
        block._course_current_geometry_key=block._course_camera_cache_key=None
        block.synchronize_course(core)
        if hasattr(block,'actor_output_capacity'):
            block.actor_output_capacity=dict(block.actor_output_capacity,
                nnz=int(block.actor_output_capacity['nnz'])+12*len(indices))
    state.original_motion_far_board_roles=role
    return deepcopy(role.contract)
