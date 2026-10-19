"""Qualify source-reviewed finite target interfaces without changing old UV.

The existing signed current-RGB reader and its frozen cubic replay remain the
measurement operator. This gate binds the independently reviewed physical target
and layer, native actor/role exclusions, and exact replay before source admission.
"""
from pathlib import Path
from collections import Counter
import hashlib
import numpy as np
from .io_utils import read_json, read_jsonl, write_json, write_jsonl, file_sha256, object_sha256
from .finite_ring_source import CONFIG, read_target_interface
from .outer_cubic_source import prepare_cubic_channels, replay_outer_cubic
from .measurement_contract import EventMeasurementContract

VERSION = 'native_declared_target_material_interface_gate_v1'


def seal_finite_target_interfaces(acquisition, review, output, *, project):
    import cv2
    out = Path(output)
    if out.exists():
        raise ValueError('New immutable target-interface source generation required')
    hashes = {}
    def ref(path):
        p = Path(path).resolve()
        if str(p) not in hashes:
            hashes[str(p)] = file_sha256(p)
        return dict(file=str(p), sha256=hashes[str(p)])
    def verify(value):
        p = Path(value.get('file', value.get('path'))).resolve()
        if ref(p)['sha256'] != value['sha256']:
            raise ValueError('Changed finite target source evidence: ' + str(p))
        return p
    a = read_json(acquisition)
    if (a.get('schema') != 'rinkfit.native_target_material_interface_acquisition.v1'
        or a.get('world_model_used_for_source_UV') is not False
        or a.get('old_raw_UV_or_membership_changed') is not False):
        raise ValueError('Missing immutable native interface acquisition contract')
    for name in ('model', 'reader', 'frozen_reader', 'source_rows', 'source_mean_replay_QA', 'frames', 'source_roles'):
        verify(a[name])
    if a['reader']['sha256'] != file_sha256(Path(__file__).with_name('finite_ring_source.py')):
        raise ValueError('Actual target interface reader changed after acquisition')
    if a['frozen_reader']['sha256'] != file_sha256(Path(__file__).with_name('outer_cubic_source.py')):
        raise ValueError('Frozen interface reader changed after acquisition')
    contract = EventMeasurementContract(project, 'ankara_2026')
    source = a['source_video']
    if (source['id'] != 'O6j8L8TmfKo' or source['time_base'] != '1/12800'
        or source['native_fps'] != '50/1' or source['native_pts_step'] != 256
        or file_sha256(source['path']) != source['sha256']):
        raise ValueError('Wrong official native video or clock')
    revref = ref(review); rev = read_json(review)
    if (rev.get('schema') != 'rinkfit.native_target_material_interface_assistant_review.v1'
        or rev.get('authority') != 'assistant_native_pixel_review'
        or rev.get('target_layers_are_material_interfaces_not_paint_centres') is not True
        or rev.get('model_projection_or_residual_used_for_identity') is not False
        or rev.get('user_approved_every_source_pixel') is not False):
        raise ValueError('Actual finite physical target and layer review required')
    for evidence in rev['evidence_refs']:
        verify(evidence)
    accepted = set(rev['accepted_source_row_ids'])
    if len(accepted) != len(rev['accepted_source_row_ids']):
        raise ValueError('Repeated target source row in review')
    identities = {(r['shot_id'], int(r['native_pts']), r['target_id']): r for r in rev['finite_target_layer_identity_reviews']}
    for item in identities.values():
        if (item.get('physical_instance_identity_closed') is not True
            or item.get('actual_material_layer_identity_closed') is not True
            or not item.get('source_only_identity_basis')
            or item.get('world_radius_ratio_or_concentricity_used') is not False):
            raise ValueError('Unclosed physical target or actual material layer')
    model = read_json(a['model']['file'])
    roles = read_json(a['source_roles']['file'])['frame_roles']
    frames = {int(x['native_pts']): x for x in read_json(a['frames']['file'])}
    replay = {r['source_row_id']: r for r in read_json(a['source_mean_replay_QA']['file'])['rows']}
    ledgers = {}
    for ledgerref in a['independent_actor_ledgers']:
        for ld in read_jsonl(verify(ledgerref)):
            if int(ld['native_pts']) in frames:
                ledgers[int(ld['native_pts'])] = ld
    rows = []; geometry = []; receipts = []; mapping = {}; counts = Counter(); seen = set()
    current = None; rgb = mask = coeff = None
    for row in read_jsonl(a['source_rows']['file']):
        rid = row['source_row_id']
        if rid in seen:
            raise ValueError('Repeated actual target source ID')
        seen.add(rid)
        new = dict(row, source_geometry_eligible=False)
        if rid not in accepted:
            rows.append(new); continue
        pts = int(row['native_pts']); sid = row['shot_id']; target = row['target_id']
        feature, layer = target[4:].split(':') if target.startswith('ice.') else ('', '')
        if feature not in contract.target_rings or layer not in ('outer','middle','inner'):
            raise ValueError('Not a declared target ring material interface')
        if (sid,pts,target) not in identities:
            raise ValueError('Current native target layer has no physical source review')
        expected = contract.expected(target, 'ice_curve')
        if expected['operator'] != 'material_interface':
            raise ValueError('Current interface conflicts with existing user material definition')
        if pts % 256 or roles.get(str(pts)) != dict(shot_id=sid, role='development_candidate'):
            raise ValueError('Wrong native PTS or protected/nondevelopment source exposure')
        binding = row['native_source_binding']; frame = frames[pts]; ld = ledgers.get(pts)
        if (binding.get('source_video_sha256') != source['sha256']
            or binding.get('native_pts') != pts or binding.get('time_base') != '1/12800'
            or binding.get('shot_id') != sid or binding.get('training_role_verified') is not True
            or binding.get('decoded_native_RGB_sha256') != frame['native_RGB_sha256']
            or row.get('fit_role') != 'development' or row.get('frame_role') != 'development_candidate'):
            raise ValueError('Wrong current source binding')
        if ld is None or ld['shot_id'] != sid or object_sha256(ld) != frame['actor_ledger_record_sha256']:
            raise ValueError('Missing exact independent native actor ledger')
        if current != pts:
            rgb = cv2.cvtColor(cv2.imread(str(verify(frame['native_png']))), cv2.COLOR_BGR2RGB)
            if hashlib.sha256(rgb.tobytes()).hexdigest() != frame['native_RGB_sha256']:
                raise ValueError('Current native RGB bytes differ')
            with np.load(verify(frame['source_mask'])) as archive:
                mask = archive['mask'].copy()
            if rgb.shape != (1080,1920,3) or mask.shape != (1080,1920) or mask.dtype != np.bool_:
                raise ValueError('Wrong native source/mask scalar domain')
            required = np.zeros((1080,1920),bool); required[:270,:620] = True; required[900:,1600:] = True
            for actor in ld.get('current_person_boxes', []):
                ab = actor['source_binding']
                if (ab['source_pts'] != pts or ab['shot_id'] != sid
                    or ab['source_sha256'] != source['sha256']
                    or ab['decoded_native_rgb_sha256'] != frame['native_RGB_sha256']):
                    raise ValueError('Actor/reflection exclusion belongs to another exposure')
                x1,y1,x2,y2 = actor['box_xyxy']
                required[max(0,int(y1)-12):,max(0,int(x1)-24):min(1920,int(x2)+24)] = True
            if not np.array_equal(mask, required):
                raise ValueError('Actual source mask differs from independent actor/reflection and graphics exclusions')
            coeff = prepare_cubic_channels(rgb); current = pts
        q = row['source_operator_result']; receipt = q.get('source_material_receipt', {})
        certificate = receipt.get('actual_target_object_layer_certificate', {})
        polarity = -1 if layer == 'middle' else 1
        if (q.get('status') != 'current_native_declared_target_material_interface'
            or q.get('current_uv') != row['current_uv']
            or receipt.get('operator_version') != CONFIG['version']
            or receipt.get('configuration') != CONFIG
            or receipt.get('layer_role') != layer
            or receipt.get('expected_outward_luminance_polarity') != polarity
            or certificate.get('target_id') != target or certificate.get('layer') != layer
            or certificate.get('native_pts') != pts or certificate.get('shot_id') != sid
            or receipt.get('source_mean_UV_from_world_projection') is not False):
            raise ValueError('Actual current target/layer readout contract differs')
        if (row.get('measurement_operator_resolved') != 'material_interface'
            or row.get('point_correspondence') is not False
            or row.get('source_mean_pixels_from_model') is not False
            or row.get('raw_legacy_UV_or_membership_changed') is not False):
            raise ValueError('Wrong interface semantics or historical mutation')
        normal = np.asarray(q['normal'],float); seed = np.asarray(receipt['original_navigation_seed_uv'],float)
        tangent = np.array([-normal[1],normal[0]])
        coords = seed + np.asarray(receipt['actual_tangent_offsets'])[:,None,None]*tangent + np.asarray(receipt['actual_five_profile_offsets'])[None,:,None]*normal
        xy = np.rint(coords).astype(int)
        if (np.any(xy[:,:,0] < 0) or np.any(xy[:,:,0] >= 1920)
            or np.any(xy[:,:,1] < 0) or np.any(xy[:,:,1] >= 1080)
            or mask[xy[:,:,1],xy[:,:,0]].any()):
            raise ValueError('Current actual target support intersects excluded native pixels')
        # Repeat the real acquisition predicate on these immutable native pixels.
        acquired = read_target_interface(rgb,mask,seed,normal,polarity,coefficients=coeff,
            layer_role=layer,actual_object_layer_certificate=certificate)
        if acquired.get('current_uv') != row['current_uv']:
            raise ValueError('Actual current target material acquisition does not reproduce')
        frozen = replay_outer_cubic(rgb,receipt['frozen_point_operator'],coefficients=coeff)
        err = float(np.max(np.abs(np.asarray(frozen['current_uv']) - row['current_uv'])))
        old = replay.get(rid)
        if (old is None or old['max_abs_replay_px'] != 0 or not frozen['numeric_valid']
            or frozen['synthetic_window_boundary_count'] or err != 0):
            raise ValueError('Frozen actual material-interface source mean does not replay exactly')
        feedback = read_json(verify(row['feedback_receipt']['receipt']))
        if (feedback.get('actual_native_RGB_sha256') != frame['native_RGB_sha256']
            or feedback.get('source',{}).get('pts') != pts or feedback.get('blockers')):
            raise ValueError('Actual source-feedback reader receipt is stale or blocked')
        mi = int(row['joint_material_index']); material = model['materials'][mi]
        if (material['label'] != target or len(material['paths']) != 1
            or material['paths'][0]['kind'] != 1 or material['paths'][0].get('physical_point')):
            raise ValueError('Existing finite target material path binding changed')
        proof = dict(schema='rinkfit.native_target_material_interface_qualification.v1',
            source_row_id=rid,native_pts=pts,shot_id=sid,target_id=target,
            existing_material_index=mi,existing_path_id=material['paths'][0]['id'],
            measurement_operator='material_interface',physical_surface='ice',
            current_2D_source_mean_sha256=object_sha256(dict(source_row_id=rid,native_pts=pts,current_uv=row['current_uv'])),
            source_readout_sha256=object_sha256(receipt),native_pixel_review=revref,
            physical_target_and_layer_review_sha256=object_sha256(identities[(sid,pts,target)]),
            user_material_contract_sha256=contract.digest,source_geometry_qualified=True,
            curve_samples_are_fixed_point_tracks=False,frozen_operator_replay_max_abs_px=err,
            old_raw_source_UV_or_membership_changed=False,statistical_status='unknown_for_current_generation',
            numeric_statistics_ready=False,statistical_objective_eligible=False,solver_eligible=False)
        new.update(source_geometry_eligible=True,source_geometry_review_pending=False,
            source_qualification_receipt_sha256=object_sha256(proof))
        rows.append(new); geometry.append(new); receipts.append(proof); counts[sid] += 1
        mapping[new['joint_material_id']] = dict(material_index=mi,geometry_ready=True)
    if accepted != {r['source_row_id']for r in geometry} or len(rows) != a['rows']:
        raise ValueError('Native target source review/ledger accounting differs')
    contract.verify_unchanged(); out.mkdir(parents=True)
    write_jsonl(out/'source_rows.jsonl.gz',rows); write_jsonl(out/'geometry_rows.jsonl.gz',geometry)
    write_jsonl(out/'qualification_receipts.jsonl.gz',receipts)
    write_json(out/'material_map.json',dict(semantic_to_material=mapping))
    gate = dict(schema='rinkfit.native_target_material_interface_source_gate.v1',gate_version=VERSION,
        acquisition=ref(acquisition),native_pixel_review=revref,gate_code=ref(__file__),
        qualification_receipts=ref(out/'qualification_receipts.jsonl.gz'),source_mean_replay_QA=a['source_mean_replay_QA'],
        source_rows=len(rows),qualified_source_rows=len(geometry),by_shot=dict(counts),
        old_raw_source_UV_or_membership_changed=False,statistical_precision_invented=False)
    write_json(out/'qualification_gate.json',gate)
    chunk = dict(schema='rinkfit.actual_source_gate_reference_adapter.v1',all_requested_frames_accounted=True,
        source_dataset_tag=VERSION,source_rows_ref=ref(out/'source_rows.jsonl.gz'),geometry_rows_ref=ref(out/'geometry_rows.jsonl.gz'),
        source_gate_manifest=ref(out/'qualification_gate.json'),actual_source_rows=len(rows),by_shot=dict(counts),
        all_new_sources_statistical_status='unknown_for_current_generation')
    write_json(out/'chunk_manifest.json',chunk)
    result = dict(schema='rinkfit.native_target_material_interface_qualified_bundle.v1',model=a['model'],
        source_review=revref,acquisition=ref(acquisition),closed_chunks=[ref(out/'chunk_manifest.json')],
        material_map=ref(out/'material_map.json'),source_rows=len(rows),qualified_source_rows=len(geometry),
        by_shot=dict(counts),whole_same_exposure_statistical_status='unavailable_for_new_generation_engineering_only',
        old_raw_source_UV_or_membership_changed=False,source_ledger_preserved=True)
    for p,h in hashes.items():
        if file_sha256(p) != h:
            raise ValueError('Target source evidence changed during sealing: ' + p)
    write_json(out/'manifest.json',result)
    return result
