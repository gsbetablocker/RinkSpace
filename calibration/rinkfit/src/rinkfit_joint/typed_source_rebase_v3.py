"""Consume exact typed source suffixes after immutable common-model rebasing.

The source readers/qualification gates are replayed before accepting a remap.
Only material/block indices and explicitly recorded conditional UV seeds change.
"""
from pathlib import Path
import copy
import math
from .io_utils import read_json, read_jsonl

SCHEMA = 'rinkfit.exact_typed_source_suffix_rebase.v3'


def load_rebased_qualification(manifest_path, manifest, reference, verify):
    from .typed_entity_append import load_qualification, validate_new_entity
    for key in ('original_bundle', 'original_gate', 'original_rows', 'original_model',
                'original_bindings', 'parent_model', 'current_model', 'remap',
                'input_cumulative_journal', 'qualification_receipts',
                'track_material_bindings', 'numeric_initialization', 'gate_code', 'dispatch_code'):
        verify(manifest[key])
    bundle = read_json(verify(manifest['original_bundle']))
    if bundle['qualification_gate'] != manifest['original_gate']:
        raise ValueError('Rebase does not consume the original exact typed gate')
    original_receipts, refs = load_qualification(manifest['original_gate'], verify)
    original_rows = list(read_jsonl(verify(manifest['original_rows'])))
    if bundle['source_rows_ref'] != manifest['original_rows']:
        raise ValueError('Original source bundle and row ledger differ')
    corrected = {}
    if 'source_readout_generation' in manifest:
        from .source_readout_contract_v1 import source_readout_rows
        verify(manifest['source_readout_contract_code'])
        corrected = source_readout_rows(manifest['source_readout_generation'], manifest['source_readout_lineage_QA'], verify)
    original_model = read_json(verify(manifest['original_model']))
    parent = read_json(verify(manifest['parent_model']))
    current = read_json(verify(manifest['current_model']))
    body_tracks = {}; body_ids = set()
    if 'body_chart_binding_contract' in manifest:
        from .body_chart_contract_v1 import load_body_chart_contract
        verify(manifest['body_chart_contract_code'])
        previous = read_json(verify(manifest['previous_common_model']))
        body_tracks, body_ids = load_body_chart_contract(manifest['body_chart_binding_contract'], previous, current, verify)
    nb = len(parent['blocks']); nm = len(parent['materials'])
    if (current['blocks'][:nb] != parent['blocks'] or current['materials'][:nm] != parent['materials']
        or any(current.get(k) != parent.get(k) for k in ('rigs', 'shots', 'scene'))):
        raise ValueError('Source rebase changed an inherited physical state')
    remap_list = read_json(verify(manifest['remap']))
    remap = {r['track_id']: r for r in remap_list}
    if len(remap) != len(remap_list):
        raise ValueError('Rebase has duplicate track mapping')
    rows = list(read_jsonl(verify(manifest['input_cumulative_journal'])))
    receipts_list = list(read_jsonl(verify(manifest['qualification_receipts'])))
    receipts = {r['source_row_id']: r for r in receipts_list}
    latents = read_json(verify(manifest['track_material_bindings']))
    lat = {r['track_id']: r for r in latents}
    numeric = read_json(verify(manifest['numeric_initialization']))
    report = {r['track_id']: r for r in numeric['per_track']}
    if (len(rows) != len(original_rows) or len(receipts) != len(receipts_list)
        or set(receipts) != set(original_receipts) or set(remap) != set(lat)
        or len(lat) != len(latents)):
        raise ValueError('Rebase lost the complete physical source denominator')
    patch_tracks = {}; patch_ids = set()
    if 'Ad5_rebind_patch' in manifest:
        patch = read_json(verify(manifest['Ad5_rebind_patch']))
        verify(patch['domain']); verify(patch['numeric_initialization'])
        load_qualification(patch['typed_entity_qualification'], verify)
        patch_tracks = {r['track_id']: r for r in patch['tracks']}
        patch_ids = set(patch['replace_only_these_105_source_row_ids'])
        if len(patch_tracks) != 25 or len(patch_ids) != 105:
            raise ValueError('Ad5 chart patch denominator differs')
    old_bind = {r['track_id']: r for r in read_json(verify(manifest['original_bindings']))}
    if set(old_bind) != set(remap):
        raise ValueError('Rebase changed the source-qualified track identity set')
    for track, item in remap.items():
        old = old_bind[track]; new = lat[track]
        om = item['old_material_index']; ob = item['old_geometry_block']
        mi = item['new_material_index']; bi = item['new_geometry_block']; ad = item['advertisement_block']
        if (old['material_index'] != om or old['geometry_block'] != ob
            or new['material_index'] != mi or new['geometry_block'] != bi
            or new['advertisement_block'] != ad or mi < nm or bi < nb
            or current['blocks'][bi]['name'] != original_model['blocks'][ob]['name']
            or current['blocks'][ad]['name'] != original_model['blocks'][old['advertisement_block']]['name']):
            raise ValueError('Rebase redirects the unknown track or its physical installation')
        body_patch = body_tracks.get(track)
        patch = body_patch['item'] if body_patch else patch_tracks.get(track)
        expected_mat = copy.deepcopy(patch['material'] if patch else original_model['materials'][om])
        expected_mat['paths'][0]['geometry_block'] = bi
        expected_mat['paths'][0]['advertisement_block'] = ad
        if current['materials'][mi] != expected_mat:
            raise ValueError('Rebase changes the actual point operator or material path')
        expected_block = copy.deepcopy(patch['geometry_block'] if patch else original_model['blocks'][ob])
        block = copy.deepcopy(current['blocks'][bi])
        values = list(block['values']); ev = expected_block['values']
        if (len(values) != 6 or block['constant'] or block.get('fixed_indices') != [0, 1, 3, 4]
            or [values[j] for j in [0, 1, 3, 4]] != [1., 0., 0., 1.]):
            raise ValueError('Unknown material point must retain precisely two free coordinates')
        for j in (2, 5):
            if not math.isfinite(values[j]) or not block['lower'][j] <= values[j] <= block['upper'][j]:
                raise ValueError('Conditional track coordinates leave the source-supported finite domain')
            block['values'][j] = ev[j]
        if block != expected_block:
            raise ValueError('Rebase changed a bounded point field beyond its two conditional coordinates')
        r = report[track]
        if (r['source_row_ids'] != [x['source_row_id'] for x in original_rows if x['physical_track_id'] == track]
            or r['free_unknown_material_coordinates'] != [2, 5]
            or r['physical_source_qualified'] is not True or r['world_anchor_or_optical_accuracy_claimed'] is not False
            or r['after']['unknown_material_uv'] != [values[2], values[5]]
            or new['numeric_initialization_status'] != r['numeric_initialization_status']
            or new['numeric_initialization_failure_reasons'] != r['numeric_initialization_failure_reasons']):
            raise ValueError('Numeric activation is not the actual same-source conditional initialization')
    for old, row in zip(original_rows, rows):
        rid = old['source_row_id']; track = old['physical_track_id']; item = remap[track]; r = report[track]
        expected = copy.deepcopy(corrected.get(rid, old))
        if rid in corrected: expected['original_LK_evidence_is_history_only'] = True
        expected.update(joint_material_index=item['new_material_index'],
                        numeric_initialization_status=r['numeric_initialization_status'],
                        numeric_initialization_failure_reasons=r['numeric_initialization_failure_reasons'],
                        numeric_initialization=manifest['numeric_initialization'],
                        physical_source_qualification_unchanged=True)
        if track in body_tracks:
            source = {x['source_row_id']: x for x in body_tracks[track]['source_rows']}[rid]
            if (source['source_uv'] != expected['source_uv'] or source['current_uv'] != expected['current_uv']
                or source['native_pts'] != expected['native_pts'] or rid not in body_ids):
                raise ValueError('Body chart patch changes its original real source UV or clock')
            expected['source_chart_domain_ref'] = body_tracks[track]['domain_ref']
        if row != expected:
            raise ValueError('Typed rebase changes an original source field or numeric selection')
        receipt = copy.deepcopy(original_receipts[rid])
        receipt.update(existing_material_index=item['new_material_index'], geometry_block=item['new_geometry_block'],
                       advertisement_block=item['advertisement_block'])
        if rid in corrected:
            from .io_utils import object_sha256
            from .static_board_tracklet_append import OPERATOR
            receipt.update(source_uv=expected['source_uv'], original_UV_preserved=expected['source_uv'],
                current_2D_source_mean_sha256=object_sha256(dict(source_row_id=rid, native_pts=expected['native_pts'], current_2D_UV=expected['current_uv'], estimator=OPERATOR)),
                historical_original_UV_preserved=old['source_uv'],
                historical_current_2D_source_mean_sha256=original_receipts[rid]['current_2D_source_mean_sha256'],
                source_readout_generation=manifest['source_readout_generation'],
                current_readout_status=expected['readout_update_status'],
                source_precision_or_covariance_issued=False)
        if track in body_tracks: receipt['source_chart_domain_ref'] = body_tracks[track]['domain_ref']
        if receipts[rid] != receipt:
            raise ValueError('Typed rebase changes an original source receipt beyond the exact current readout')
        validate_new_entity(row, receipts[rid], current['materials'][item['new_material_index']], item['new_material_index'])
        if item['Ad5_body_patch_applied'] != (rid in patch_ids):
            raise ValueError('Ad5 body chart patch was not restricted to its original source row IDs')
    # A role is numerical state, not a source-quality label. Re-evaluate every
    # actual row from this bound current model before accepting its activation.
    import numpy as np
    from collections import defaultdict
    from .joint_model import graph_from_joint, as_event_view
    from .camera_graph import CameraGraph
    from .joint_replay import render_visibility
    from rinkfit_start.native_scene import NativeScene
    cg = CameraGraph(graph_from_joint(current)); scene = NativeScene(as_event_view(current, 's01'))
    by_shot = defaultdict(list)
    for i, row in enumerate(rows): by_shot[row['shot_id']].append(i)
    for sid, ii in by_shot.items():
        selected = [rows[i] for i in ii]
        pts = np.asarray([r['native_pts'] for r in selected], np.int64)
        mi = np.asarray([r['joint_material_index'] for r in selected], np.int64)
        X, ok, _ = scene.world_rows(mi, np.zeros(len(ii), np.uint32), np.zeros(len(ii)))
        times = cg.local_seconds(sid, pts); camera = cg.native_camera(sid)
        projection = camera.project_rs(X, times); prediction = np.asarray(projection['uv'])
        visible = render_visibility(camera, scene, X,
            [dict(begin=0, end=len(ii), label='board.texture', board_material_surface=True)],
            times, prediction)
        by_track = defaultdict(list)
        for j, row in enumerate(selected): by_track[row['physical_track_id']].append(j)
        for track, jj in by_track.items():
            r = report[track]; q = r['after']
            valid = bool(np.all(np.asarray(projection['valid'])[jj]) and np.all(ok[jj]))
            finite = bool(np.isfinite(prediction[jj]).all())
            front = bool(np.all(visible[jj]))
            height = bool(np.all((X[jj, 2] >= 0) & (X[jj, 2] <= scene.height)))
            if (q['all_native_RS_valid'] != valid or q['finite_projection'] != finite
                or q['all_inward_finite_board_visible'] != front or q['all_board_heights_in_domain'] != height):
                raise ValueError('Current native source projection differs from numeric activation evidence')
            if finite:
                observed = np.asarray([selected[j]['source_uv'] for j in jj])
                rms = float(np.sqrt(np.mean(np.sum((prediction[jj] - observed)**2, axis=1))))
                if q['RMS_px'] is None or abs(rms - q['RMS_px']) > 1e-7:
                    raise ValueError('Current all-source UV RMS differs from conditional initialization')
            pending = r['numeric_initialization_status'] == 'pending_numeric_initialization'
            if not pending and not (valid and finite and front and height):
                raise ValueError('Uncaptured numeric point cannot be promoted by source qualification')
    refs += [dict(file=str(manifest_path), sha256=reference['sha256']), manifest['qualification_receipts'],
             dict(manifest['input_cumulative_journal'], role='qualified_current_entity_source_journal')]
    return receipts, refs
