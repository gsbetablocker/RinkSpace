"""Bind source-qualified static ice textures to shared physical XY latents.

The native point operator is an existing kind-0 carrier at fixed station zero.
Only its first two coordinates are free.  The other endpoint is a frozen ABI
compatibility value and never supplies another world point or an observation.
"""
from collections import Counter, defaultdict
from pathlib import Path
import copy
import json
import numpy as np
from .io_utils import read_json, read_jsonl, write_json, write_jsonl, file_sha256, object_sha256

OPERATOR = 'native_static_ice_RGB_texture_2D_track_v1'
BUNDLE_SCHEMA = 'rinkfit.native_static_ice_tracklet_qualified_bundle.v1'
GATE_SCHEMA = 'rinkfit.static_ice_tracklet_point_qualification.v1'
RECEIPT_SCHEMA = 'rinkfit.static_ice_tracklet_point_receipt.v1'


def _ref(path):
    p = Path(path).resolve()
    return dict(file=str(p), sha256=file_sha256(p))


def _verified(ref):
    p = Path(ref.get('file', ref.get('path'))).resolve()
    if file_sha256(p) != ref['sha256']:
        raise ValueError('Changed static ice source dependency: '+str(p))
    return p


def _source_contract(generation_path, qualification_path, diagnostics_path, roles_path):
    """Check material identity independently of any world-coordinate guess."""
    generation = read_json(generation_path)
    qualification = read_json(qualification_path)
    diagnostics = read_json(diagnostics_path)
    source = generation['source_video']
    tracks_path = _verified(generation['tracks_file'])
    if (generation.get('schema') != 'rinkfit.source_static_ice_texture_tracklet_generation.v1'
        or qualification.get('schema') != 'rinkfit.static_ice_texture_qualification.v1'
        or qualification['tracks_file_sha256'] != file_sha256(tracks_path)
        or qualification['source_video_sha256'] != source['sha256']
        or diagnostics['source_file_sha256'] != file_sha256(tracks_path)
        or source['native_fps'] != '50/1' or source['time_base'] != '1/12800'
        or source['native_pts_step'] != 256):
        raise ValueError('Static ice source generation/qualification/clock differs')
    if file_sha256(source['path']) != source['sha256']:
        raise ValueError('Actual official native programme video changed')
    scope = read_json(_verified(generation['scope']))
    if scope['source']['sha256'] != source['sha256']:
        raise ValueError('Static ice source scope differs')
    shots = {s['id']:s for s in scope['shots']}
    roles = read_json(roles_path)['frame_roles']
    accepted = {}
    evidence = {}
    checked_images = set()
    for q in qualification['tracks']:
        if q['track_id'] in accepted:
            raise ValueError('Repeated static ice track qualification')
        if q['status'] != 'accepted_static_ice':
            continue
        if not q.get('reason') or not q.get('evidence'):
            raise ValueError('Static ice material requires exact source review')
        accepted[q['track_id']] = q
        for e in q['evidence']:
            key = (q['track_id'], int(e['native_pts']))
            if key in evidence:
                raise ValueError('Repeated exact static source exposure evidence')
            evidence[key] = e
            r = e['review_file']; token = (r['file'], r['sha256'])
            if token not in checked_images:
                _verified(r); checked_images.add(token)
    diag = {(r['track_id'], int(r['native_pts'])):r for r in diagnostics['rows']}
    groups = defaultdict(list); used = set()
    for row in read_jsonl(tracks_path):
        key = (row['track_id'], int(row['native_pts']))
        if row['track_id'] not in accepted:
            raise ValueError('Unqualified texture cannot enter physical source gate')
        if key in used:
            raise ValueError('Repeated static source readout')
        used.add(key); e = evidence.get(key); d = diag.get(key)
        sid = row['shot_id']; pts = row['native_pts']; s = shots[sid]
        if (sid == 's01' or not s['first_pts'] <= pts <= s['last_pts']
            or (pts-s['first_pts']) % 256
            or roles.get(str(pts), {}).get('role') != 'development_candidate'
            or row.get('fit_role') != 'development'
            or row.get('surface_class') != 'ice' or row.get('surface_instance_id') != 'ice'
            or row.get('pixels_from_model') is not False
            or row.get('source_video_sha256') != source['sha256']
            or row.get('time_base') != [1,12800]
            or row.get('world_position_is_shared_track_latent') is not True
            or not row.get('stationary_material_evidence')
            or e is None or d is None):
            raise ValueError('Static physical source is outside exact source/material/role scope')
        uv = np.asarray(row['source_uv'],float)
        if (uv.shape != (2,) or not np.isfinite(uv).all()
            or np.any(uv < 0) or np.any(uv > [1919,1079])
            or e['source_uv'] != row['source_uv'] or d['source_uv'] != row['source_uv']
            or e['current_rgb_sha256'] != row['current_rgb_sha256']
            or d['current_rgb_sha256'] != row['current_rgb_sha256']
            or d['status'] != 'entity_corner_two_direction_support'):
            raise ValueError('Static physical source lacks current exact RGB/UV/2D support')
        groups[row['track_id']].append(row)
    if used != set(evidence) or used != set(diag):
        raise ValueError('Static source denominator differs from exact reviewed 2D evidence')
    if (len(used) != qualification['accepted_rows'] or len(groups) != qualification['accepted_tracks']
        or len(used) != generation['observations']):
        raise ValueError('Static ice qualification denominator is incomplete')
    for k, rows in groups.items():
        rows.sort(key=lambda r:r['native_pts'])
        if len(rows)<2 or len({r['shot_id'] for r in rows})!=1 or np.any(np.diff([r['native_pts'] for r in rows])!=256):
            raise ValueError('Shared static ice track crosses a cut or source gap')
        q=accepted[k]['diagnostic']
        if q['n'] != len(rows) or q['two_direction_rows'] != len(rows):
            raise ValueError('Static track does not have whole-track 2D source support')
    return groups


def validate_point_receipt(row, receipt, material, material_index):
    if (not row.get('point_correspondence') or row.get('measurement_operator_resolved') != OPERATOR
        or receipt.get('schema') != RECEIPT_SCHEMA
        or receipt['existing_material_index'] != material_index
        or receipt['target_id'] != material['label'] or row.get('source_uv') != row.get('current_uv')
        or receipt['native_pts'] != row['native_pts']
        or receipt['original_UV_preserved'] != row['source_uv']):
        raise ValueError('Shared ice point receipt redirects source mean or physical identity')
    expected = object_sha256(dict(source_row_id=row['source_row_id'], native_pts=row['native_pts'],
        current_2D_UV=row['current_uv'], estimator=OPERATOR))
    if expected != receipt['current_2D_source_mean_sha256']:
        raise ValueError('Shared ice point source mean changed')
    found = [i for i,p in enumerate(material['paths']) if p['id']==receipt['existing_point_path']]
    if len(found)!=1:
        raise ValueError('Static track must own exactly one corresponding physical point path')
    path=material['paths'][found[0]]
    if path['kind']!=0 or path.get('physical_point') is not True or path.get('closed'):
        raise ValueError('Static track is not the native fixed ice point operator')
    return found[0]


def load_point_qualification(manifest_path, manifest, reference, verify):
    """Production appender calls this gate, rather than trusting new JSON fields."""
    for key in ('source_generation','static_surface_qualification','two_direction_diagnostics','roles',
                'input_cumulative_journal','qualification_receipts','current_model','gate_code'):
        verify(manifest[key])
    groups = _source_contract(verify(manifest['source_generation']),
        verify(manifest['static_surface_qualification']),verify(manifest['two_direction_diagnostics']),
        verify(manifest['roles']))
    actual = {(k,r['native_pts']):r for k,rows in groups.items() for r in rows}
    model = read_json(verify(manifest['current_model']))
    receipts = {r['source_row_id']:r for r in read_jsonl(verify(manifest['qualification_receipts']))}
    rows = list(read_jsonl(verify(manifest['input_cumulative_journal'])))
    if len(rows)!=len(actual) or len(receipts)!=len(actual):
        raise ValueError('Static physical point source gate lost complete denominator')
    for row in rows:
        receipt=receipts[row['source_row_id']]; old=actual[(row['physical_track_id'],row['native_pts'])]
        if (row['source_uv']!=old['source_uv'] or receipt['current_rgb_sha256']!=old['current_rgb_sha256']
            or receipt.get('new_actual_source_geometry_qualified') is not True
            or receipt.get('new_material_suffix') is not True
            or receipt.get('entity_station_fixed') is not True
            or any(receipt.get(k) is not False for k in ('solver_eligible','statistical_objective_eligible','numeric_statistics_ready'))):
            raise ValueError('Static point source qualification or unknown-noise contract differs')
        mi=receipt['existing_material_index']; mat=model['materials'][mi]
        pi=validate_point_receipt(row,receipt,mat,mi);block=model['blocks'][mat['paths'][pi]['geometry_block']]
        if len(block['values'])!=4 or block['constant'] or block.get('fixed_indices')!=[2,3]:
            raise ValueError('Static point requires one free shared XY and frozen unused endpoint')
    return receipts,[dict(file=str(manifest_path),sha256=reference['sha256']),
        manifest['qualification_receipts'],dict(manifest['input_cumulative_journal'],
        role='qualified_current_entity_source_journal')]


def prepare_static_ice_tracklets(parent_model, generation, qualification, diagnostics,
                                roles, screen_state, bridge_report, output):
    """Append seven shared physical points; screen inversion is initialization only."""
    from .screen_recovery_state import ScreenRecoveryState
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('New immutable physical-track source generation required')
    groups=_source_contract(generation,qualification,diagnostics,roles)
    model=read_json(parent_model); result=copy.deepcopy(model)
    state=ScreenRecoveryState(screen_state)
    bridge=read_json(bridge_report)['template']['current_to_reference_H']
    inv=np.linalg.inv(np.asarray(bridge,float))
    rows=[];receipts=[];mapping={};latents=[]
    for k, originals in sorted(groups.items()):
        first=originals[0]; xy,good=state.inverse_ice(first['native_pts'],first['shot_id'],[first['source_uv']])
        if not good.all():raise ValueError('Static track navigation initializer has no ice inverse')
        seed=inv@np.r_[xy[0],1.]; seed=seed[:2]/seed[2]
        if not np.isfinite(seed).all() or np.any(np.abs(seed)>.6):
            raise ValueError('Static ice point initializer leaves the existing L=1 ice chart')
        mi=len(result['materials']);bi=len(result['blocks']);label='ice.tracklet.'+k.replace('/','.')
        pathid='static_track/'+k
        result['blocks'].append(dict(name='texture_shared_X/'+k,values=[*seed,*seed],
            lower=[-.6]*4,upper=[.6]*4,scales=[.5]*4,constant=False,fixed_indices=[2,3]))
        result['materials'].append(dict(label=label,paths=[dict(id=pathid,variant_id='source_static_ice_shared_X',
            kind=0,geometry_block=bi,physical_point=True,closed=False,
            provenance=json.dumps(dict(track_id=k,static_surface_qualification=_ref(qualification),
                geometry_role='shared_unknown_ice_XY_z0',station_fixed=0.,
                inverse_screen_and_bridge_are_initialization_only=True),ensure_ascii=False))]))
        mapping[label]=dict(material_index=mi,geometry_ready=True)
        latents.append(dict(track_id=k,material_index=mi,geometry_block=bi,initial_XY_L=seed.tolist(),
            world_z=0.,shared_native_rows=len(originals),free_coordinate_indices=[0,1],fixed_unused_indices=[2,3],
            world_coordinate_truth_or_covariance_claimed=False))
        for original in originals:
            pts=original['native_pts'];rid='static_ice_texture/'+k+'/'+str(pts)
            row=dict(original,source_row_id=rid,source_family_id=k+'/'+str(pts),physical_track_id=k,
                target_id=label,joint_material_id=label,joint_material_index=mi,
                current_uv=original['source_uv'],point_correspondence=True,measurement_operator_resolved=OPERATOR,
                source_geometry_eligible=True,statistical_objective_eligible=False,
                numeric_statistics_ready=False,solver_eligible=False,frame_role='development_candidate',normal=[0.,0.])
            receipt=dict(schema=RECEIPT_SCHEMA,source_row_id=rid,track_id=k,native_pts=pts,shot_id=original['shot_id'],
                source_uv=original['source_uv'],joint_material_index=mi,status='accepted_static_ice',
                qualification_ref=_ref(qualification),
                current_rgb_sha256=original['current_rgb_sha256'],target_id=label,existing_material_index=mi,
                existing_point_path=pathid,point_path_kind=0,physical_point=True,entity_kind=1,
                entity_station_fixed=True,new_material_suffix=True,new_actual_source_geometry_qualified=True,
                original_UV_preserved=original['source_uv'],current_2D_source_mean_sha256=object_sha256(dict(
                    source_row_id=rid,native_pts=pts,current_2D_UV=original['source_uv'],estimator=OPERATOR)),
                solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False,
                original_flags_preserved=dict(manual_reference_preserved=False),
                static_surface_qualification=_ref(qualification),two_direction_diagnostics=_ref(diagnostics))
            rows.append(row);receipts.append(receipt)
    out.mkdir(parents=True)
    result['static_ice_shared_X_append']=dict(parent_model=_ref(parent_model),
        static_surface_qualification=_ref(qualification),tracks=len(groups),source_rows=len(rows),
        original_scene_rigs_shots_and_blocks_unchanged=True,screen_inverse_is_initializer_only=True)
    write_json(out/'model.json',result);write_json(out/'latents.json',latents)
    write_jsonl(out/'source_rows.jsonl',rows);write_jsonl(out/'qualification_receipts.jsonl',receipts)
    write_json(out/'material_map.json',dict(semantic_to_material=mapping))
    gate=dict(schema=GATE_SCHEMA,source_generation=_ref(generation),static_surface_qualification=_ref(qualification),
        two_direction_diagnostics=_ref(diagnostics),roles=_ref(roles),input_cumulative_journal=_ref(out/'source_rows.jsonl'),
        qualification_receipts=_ref(out/'qualification_receipts.jsonl'),current_model=_ref(out/'model.json'),
        gate_code=_ref(__file__),qualified_new_measurement_rows=len(rows),
        old_raw_source_flags_false_are_not_overwritten=True,new_measurement_version_geometry_qualification_only=True)
    write_json(out/'qualification_gate.json',gate)
    counts=dict(Counter(r['shot_id'] for r in rows))
    chunk=dict(schema='rinkfit.actual_source_gate_reference_adapter.v1',all_requested_frames_accounted=True,
        source_dataset_tag='static_ice_tracklets_physical_v1',source_rows_ref=_ref(out/'source_rows.jsonl'),
        geometry_rows_ref=_ref(out/'source_rows.jsonl'),source_gate_manifest=_ref(out/'qualification_gate.json'),
        actual_source_rows=len(rows),by_shot=counts)
    write_json(out/'chunk_manifest.json',chunk)
    bundle=dict(schema=BUNDLE_SCHEMA,source_generation=_ref(generation),static_surface_qualification=_ref(qualification),
        two_direction_diagnostics=_ref(diagnostics),source_rows_ref=_ref(out/'source_rows.jsonl'),
        qualification_receipts=_ref(out/'qualification_receipts.jsonl'),source_ledger_preserved=True,
        reader_module_refs=[_ref(__file__),_ref(Path(__file__).with_name('source_texture_tracklets.py')),
            _ref(Path(generation).parent.parent/'extend_static_motion_v2.py'),
            _ref(Path(generation).parent.parent/'check_static_motion.py')],
        closed_chunks=[_ref(out/'chunk_manifest.json')],typed_entity_qualification=[_ref(out/'qualification_gate.json')],
        material_map=_ref(out/'material_map.json'),model=_ref(out/'model.json'),roles=_ref(roles),source_rows=len(rows),
        tracks=len(groups),by_shot=counts,world_latent_bindings=_ref(out/'latents.json'),
        source_noise_statistics_supplied=False,absolute_source_UV_only=True,independent_displacement_factors=False)
    write_json(out/'manifest.json',bundle)
    # Execute the same source qualification path used by the real appender.
    load_point_qualification(out/'qualification_gate.json',gate,_ref(out/'qualification_gate.json'),_verified)
    return bundle


def append_prepared_static_ice_tracklets(parent_packet, bundle_path, output, finalize):
    from .reference_source_append import append_constraint_source_chunks
    bundle=read_json(bundle_path)
    if bundle['schema']!=BUNDLE_SCHEMA:raise ValueError('Wrong physical static-track source bundle')
    return append_constraint_source_chunks(parent_packet,bundle['closed_chunks'],_verified(bundle['model']),
        _verified(bundle['material_map']),_verified(bundle['roles']),output,
        typed_entity_qualification=bundle['typed_entity_qualification'],
        finalize_packet_with_parent=lambda result,out,parent:finalize(parent_packet,result,bundle_path,
            out/'qualified_certificate.json',parent_document=parent))
