"""Seal finite-width ice reads with original identity anchors and current review contexts for unknown-noise recovery.

This entry point consumes actual native RGB, an independently bound actor mask,
the existing two-flank paint reader, exact frozen-operator replay, and an explicit
source-only finite-line review. Model coordinates and residuals confer no identity.
It emits the existing chunk adapter; it never edits a parent UV or membership.
"""
from pathlib import Path
from collections import Counter
import numpy as np
from .io_utils import read_json, read_jsonl, write_json, write_jsonl, file_sha256, object_sha256
from .physical_source_operator import CONFIG, replay_paint_midline

VERSION = 'native_finite_paint_midline_chain_source_gate_v1'


def seal_native_finite_paint_chain(acquisition, review, output, *, project):
    from .measurement_contract import EventMeasurementContract
    contract = EventMeasurementContract(project, 'ankara_2026')
    contract.verify_unchanged()
    import cv2
    out = Path(output)
    if out.exists():
        raise ValueError('New immutable ice source generation required')
    hashes = {}
    def ref(path):
        p = Path(path).resolve()
        if str(p) not in hashes:
            hashes[str(p)] = file_sha256(p)
        return dict(file=str(p), sha256=hashes[str(p)])
    def verify(value):
        p = Path(value.get('file', value.get('path'))).resolve()
        if ref(p)['sha256'] != value['sha256']:
            raise ValueError('Changed native ice source evidence: ' + str(p))
        return p
    a = read_json(acquisition)
    if a.get('schema') != 'rinkfit.native_ice_midline_acquisition.v1' or a.get('world_model_used_for_source_UV') is not False or a.get('old_raw_UV_or_membership_changed') is not False:
        raise ValueError('Ice acquisition has no immutable source-only contract')
    for name in ('model', 'reader', 'source_rows', 'source_mean_replay_QA', 'frames', 'source_roles'):
        verify(a[name])
    if a['reader']['sha256'] != file_sha256(Path(__file__).with_name('physical_source_operator.py')):
        raise ValueError('Actual source reader changed after acquisition')
    roles = read_json(a['source_roles']['file'])['frame_roles']
    source = a['source_video']
    if source['id'] != 'O6j8L8TmfKo' or source['time_base'] != '1/12800' or source['native_fps'] != '50/1' or source['native_pts_step'] != 256 or file_sha256(source['path']) != source['sha256']:
        raise ValueError('Wrong official source bytes or native clock')
    reviewref = ref(review)
    rev = read_json(review)
    if rev.get('authority') != 'assistant_native_pixel_review' or rev.get('finite_width_paint_strokes_not_target_ring_interfaces') is not True or rev.get('model_projection_or_residual_used_for_identity') is not False or rev.get('user_approved_every_source_pixel') is not False:
        raise ValueError('Explicit native finite-line identity and material review required')
    for evidence in rev['evidence_refs']:
        verify(evidence)
    accepted = set(rev['accepted_source_row_ids'])
    from .native_ice_chain_gate import validate_native_ice_chain
    selected_temporal_rows = [row for row in read_jsonl(a['source_rows']['file']) if row['source_row_id'] in accepted]
    continuity = validate_native_ice_chain(a, selected_temporal_rows)
    temporal_verified = set(continuity['verified_source_row_ids'])
    if temporal_verified != accepted:
        raise ValueError('Every selected temporal source mean requires its complete native witness')
    claim = rev.get('native_continuity_proof')
    if claim is None or object_sha256(read_json(verify(claim))) != object_sha256(continuity):
        raise ValueError('Current review must bind the exact complete native continuity proof')
    if len(accepted) != len(rev['accepted_source_row_ids']):
        raise ValueError('Repeated reviewed ice source ID')
    identities = {(x['shot_id'], int(x['anchor_native_pts']), x['target_id'], int(x['current_native_review_pts'])): x for x in rev['finite_line_identity_reviews']}
    for item in identities.values():
        if not item.get('physical_instance_identity_closed') or not item.get('source_only_identity_basis') or not item.get('current_source_paint_extent_verified'):
            raise ValueError('Unclosed finite physical line identity')
    model = read_json(a['model']['file'])
    qa = read_json(a['source_mean_replay_QA']['file'])
    replay = {r['source_row_id']: r for r in qa['rows']}
    frames = {int(x['native_pts']): x for x in read_json(a['frames']['file'])}
    masks = {}
    for ledgerref in a['independent_actor_ledgers']:
        for ld in read_jsonl(verify(ledgerref)):
            if int(ld['native_pts']) in frames:
                masks[int(ld['native_pts'])] = ld
    rows = []; geometry = []; receipts = []; mapping = {}; count = Counter(); seen = set()
    current_frame = None; rgb = None; mask = None
    for row in read_jsonl(a['source_rows']['file']):
        rid = row['source_row_id']
        if rid in seen:
            raise ValueError('Repeated actual ice readout')
        seen.add(rid)
        new = dict(row, source_geometry_eligible=False)
        if rid not in accepted:
            rows.append(new)
            continue
        pts = int(row['native_pts']); target = row['target_id']; sid = row['shot_id']
        expected = contract.expected(target, 'ice_curve')
        if expected is None or expected.get('operator') != 'paint_center' or (sid, int(row['anchor_native_pts']), target, int(row.get('current_native_review_pts', -1))) not in identities:
            raise ValueError('Source review does not close this ordinary line instance')
        temporal = row.get('native_temporal_identity', {})
        if (pts % 256 or rid not in temporal_verified
            or int(row.get('current_native_review_pts', -1)) != pts
            or int(row['anchor_native_pts']) != int(temporal.get('anchor_native_pts', -1))):
            raise ValueError('Original temporal anchor or independent current review exposure is not exact')
        b = row['native_source_binding']; frame = frames[pts]; ld = masks.get(pts)
        if roles.get(str(pts)) != dict(shot_id=sid, role='development_candidate'):
            raise ValueError('Actual source role registry does not allow this shot/exposure')
        if b['decoded_native_RGB_sha256'] != frame['native_RGB_sha256']:
            raise ValueError('Per-row RGB binding differs from actual source frame')
        if b['source_video_sha256'] != source['sha256'] or b['native_pts'] != pts or b['time_base'] != '1/12800' or b['shot_id'] != sid or not b['training_role_verified'] or row['fit_role'] != 'development' or row['frame_role'] != 'development_candidate':
            raise ValueError('Wrong native source or protected frame')
        if ld is None or object_sha256(ld) != frame['actor_ledger_record_sha256'] or ld['shot_id'] != sid:
            raise ValueError('Missing current independent actor source ledger')
        if current_frame != pts:
            rgb = cv2.cvtColor(cv2.imread(str(verify(frame['native_png']))), cv2.COLOR_BGR2RGB)
            import hashlib
            if hashlib.sha256(rgb.tobytes()).hexdigest() != frame['native_RGB_sha256'] or b['decoded_native_RGB_sha256'] != frame['native_RGB_sha256']:
                raise ValueError('Actual current RGB differs from native source binding')
            with np.load(verify(frame['source_mask'])) as archive:
                mask = archive['mask'].copy()
            if rgb.shape != (1080, 1920, 3) or mask.shape != (1080, 1920) or mask.dtype != np.bool_:
                raise ValueError('Wrong source frame/mask domain')
            required = np.zeros((1080, 1920), bool)
            required[:270, :620] = True; required[900:, 1600:] = True
            for actor in ld.get('current_person_boxes', []):
                ab = actor['source_binding']
                if ab['source_pts'] != pts or ab['shot_id'] != sid or ab['source_sha256'] != source['sha256'] or ab['decoded_native_rgb_sha256'] != frame['native_RGB_sha256']:
                    raise ValueError('Actor evidence belongs to another native exposure')
                x1,y1,x2,y2 = actor['box_xyxy']
                required[max(0,int(y1)-12):, max(0,int(x1)-24):min(1920,int(x2)+24)] = True
            if not np.array_equal(mask, required):
                raise ValueError('Source mask does not implement the bound actor/reflection and graphics exclusions')
            current_frame = pts
        q = row['source_operator_result']; receipt = q.get('source_material_receipt', {})
        if q.get('status') != 'current_native_same_paint_two_ordered_flanks_midline' or q.get('current_uv') != row['current_uv'] or receipt.get('operator_version') != CONFIG['version'] or not receipt.get('same_material_both_flanks') or not receipt.get('opposite_ordered_transitions') or receipt.get('raw_source_uv_changed') is not False:
            raise ValueError('No actual two-flank ordinary paint midline')
        if row.get('point_correspondence') or row.get('measurement_operator_resolved') != 'paint_center' or row.get('source_mean_pixels_from_model') is not False or row.get('raw_legacy_UV_or_membership_changed') is not False:
            raise ValueError('Line sample was promoted to a point track or changed raw history')
        # Recheck all actual acquisition profiles against the saved current mask.
        seed = np.asarray(receipt['original_navigation_seed_uv']); n = np.asarray(receipt['normal']); t = np.asarray(receipt['tangent'])
        coords = seed + np.asarray(receipt['tangent_offsets'])[:, None, None] * t + np.asarray(receipt['normal_offsets'])[None, :, None] * n
        xy = np.rint(coords).astype(int)
        if np.any(xy[:, :, 0] < 0) or np.any(xy[:, :, 0] >= 1920) or np.any(xy[:, :, 1] < 0) or np.any(xy[:, :, 1] >= 1080) or mask[xy[:, :, 1], xy[:, :, 0]].any():
            raise ValueError('Actual paint support intersects excluded source pixels')
        r = replay.get(rid)
        actual_replay = replay_paint_midline(rgb, receipt)
        err = float(np.max(np.abs(np.asarray(actual_replay['current_uv']) - row['current_uv'])))
        if r is None or r['max_abs_replay_px'] != 0 or not actual_replay['numeric_valid'] or err != 0:
            raise ValueError('Frozen actual source mean does not replay exactly')
        feedback = row['feedback_receipt']; fr = read_json(verify(feedback['receipt']))
        if fr['actual_native_RGB_sha256'] != frame['native_RGB_sha256'] or fr['source']['pts'] != pts or fr.get('blockers'):
            raise ValueError('Source feedback is missing, stale or blocked')
        mi = int(row['joint_material_index']); material = model['materials'][mi]
        feature = target.removeprefix('ice.').split(':')[0]
        expected_path_kind = 1 if feature in contract.circular_lines else 0
        if material['label'] != target or len(material['paths']) != 1 or material['paths'][0]['kind'] != expected_path_kind or material['paths'][0].get('physical_point'):
            raise ValueError('Finite line material binding changed')
        proof = dict(schema='rinkfit.native_finite_paint_midline_qualification.v1', source_row_id=rid,
            native_pts=pts, shot_id=sid, target_id=target, existing_material_index=mi,
            existing_path_id=material['paths'][0]['id'], existing_material_path_kind=expected_path_kind, measurement_operator='paint_center',
            source_measurement_contract_sha256=contract.digest, source_measurement_definition=expected,
            current_2D_source_mean_sha256=object_sha256(dict(source_row_id=rid,native_pts=pts,current_uv=row['current_uv'])),
            source_readout_sha256=object_sha256(receipt), native_pixel_review=reviewref,
            source_only_finite_identity_review_sha256=object_sha256(identities[(sid,int(row['anchor_native_pts']),target,pts)]),
            physical_surface='ice', source_geometry_qualified=True,
            line_samples_are_static_point_tracks=False, frozen_operator_replay_max_abs_px=err,
            old_raw_source_UV_or_membership_changed=False, statistical_status='unknown_for_current_generation',
            numeric_statistics_ready=False, statistical_objective_eligible=False, solver_eligible=False)
        new.update(source_geometry_eligible=True, source_geometry_review_pending=False,
            source_qualification_receipt_sha256=object_sha256(proof))
        rows.append(new); geometry.append(new); receipts.append(proof); count[sid] += 1
        mapping[new['joint_material_id']] = dict(material_index=mi, geometry_ready=True)
    if accepted != {r['source_row_id'] for r in geometry} or len(rows) != a['rows']:
        raise ValueError('Native source review/ledger accounting mismatch')
    out.mkdir(parents=True)
    write_json(out/'native_continuity_proof.json', continuity)
    write_jsonl(out/'source_rows.jsonl.gz',rows); write_jsonl(out/'geometry_rows.jsonl.gz',geometry)
    write_jsonl(out/'qualification_receipts.jsonl.gz',receipts)
    write_json(out/'material_map.json',dict(semantic_to_material=mapping))
    gate = dict(schema='rinkfit.native_finite_paint_midline_source_gate.v1', gate_version=VERSION,
        acquisition=ref(acquisition), native_pixel_review=reviewref, gate_code=ref(__file__),
        qualification_receipts=ref(out/'qualification_receipts.jsonl.gz'), source_mean_replay_QA=a['source_mean_replay_QA'],
        native_temporal_continuity_proof=ref(out/'native_continuity_proof.json'),
        continuity_verifier_code=ref(Path(__file__).with_name('native_ice_chain_gate.py')),
        unchanged_anchor_material_source_gate=ref(Path(__file__).with_name('finite_paint_gate.py')),
        source_rows=len(rows), qualified_source_rows=len(geometry), by_shot=dict(count),
        old_raw_source_UV_or_membership_changed=False, statistical_precision_invented=False)
    write_json(out/'qualification_gate.json',gate)
    chunk = dict(schema='rinkfit.actual_source_gate_reference_adapter.v1', all_requested_frames_accounted=True,
        source_dataset_tag=VERSION, source_rows_ref=ref(out/'source_rows.jsonl.gz'),
        geometry_rows_ref=ref(out/'geometry_rows.jsonl.gz'), source_gate_manifest=ref(out/'qualification_gate.json'),
        actual_source_rows=len(rows), by_shot=dict(count), all_new_sources_statistical_status='unknown_for_current_generation')
    write_json(out/'chunk_manifest.json',chunk)
    result = dict(schema='rinkfit.native_finite_paint_midline_qualified_bundle.v1',
        model=a['model'], source_review=reviewref, acquisition=ref(acquisition),
        closed_chunks=[ref(out/'chunk_manifest.json')], material_map=ref(out/'material_map.json'),
        source_rows=len(rows), qualified_source_rows=len(geometry), by_shot=dict(count),
        whole_same_exposure_statistical_status='unavailable_for_new_generation_engineering_only',
        old_raw_source_UV_or_membership_changed=False, source_ledger_preserved=True)
    for p,h in hashes.items():
        if file_sha256(p) != h:
            raise ValueError('Source evidence changed while sealing: '+p)
    contract.verify_unchanged()
    result['measurement_contract_sha256'] = contract.digest
    result['material_path_kind_is_not_observation_kind'] = True
    result['native_continuity_proof'] = ref(out/'native_continuity_proof.json')
    result['complete_adjacent_native_source_identity_witness_checked'] = True
    result['original_identity_anchor_and_current_review_exposure_are_distinct'] = True
    write_json(out/'manifest.json',result)
    return result
