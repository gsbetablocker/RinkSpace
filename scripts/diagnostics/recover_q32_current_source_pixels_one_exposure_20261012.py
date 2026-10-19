"""Recover one q32 exposure's active pixel input from authenticated caches.

No model, geometry, residual, J, or fitting is evaluated. This is explicitly a
single-exposure diagnostic and cannot be supplied to the complete exporter.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from calibration.solver_runtime import install
from calibration.program_tracking.current_source_point_collection import collect_current_active_source_points
from calibration.program_tracking.current_source_points import file_sha

RUN = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003/okada_ce235f_wide_curve_budget_resume_20261012')
Q = '917c0fa45f80af8e16c74d0f2d56486fdb500b591357add3dc2b240b3de64cdf'
PTS = 229702144
CAMERA = RUN / 'complete_physical_camera_model_UNACCEPTED_5126c97aa85b9576.npz'
OUT = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/current_source_points_one_exposure_cache_recovery_20261012')


def auth(reference):
    path = Path(reference['path']).resolve()
    if file_sha(path) != reference['sha256']:
        raise ValueError('Saved source/cache identity changed: ' + str(path))
    return path


def packet(contract):
    reference = dict(path=contract['packet_path'], sha256=contract['packet_sha256'])
    path = auth(reference)
    return path, json.loads(path.read_text(encoding='utf8')), reference


def main():
    manifest_path = RUN / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    identity = json.loads((RUN / 'accepted_states' / Q / 'state_identity.json').read_text(encoding='utf8'))
    if identity['input_manifest_sha256'] != file_sha(manifest_path):
        raise ValueError('q32 saved input manifest changed')
    if identity['q_sha256'] != Q:
        raise ValueError('Wrong saved model identity')
    if auth(dict(path=identity['camera_path'], sha256=identity['camera_sha256'])) != CAMERA.resolve():
        raise ValueError('Camera differs from authoritative q32 saved state')
    with np.load(CAMERA, allow_pickle=False) as camera:
        native = camera['pts'].copy()
        actual = camera['exposure_representative_pts'].copy()
        installation = camera['installation_id'].copy()
        camera_metadata = json.loads(str(camera['metadata_json'].item()))
    aliases = dict(zip(map(int, native), map(int, actual)))
    camera_installations = {int(p): str(i) for p, i in zip(actual, installation)}
    if camera_installations[PTS] != 'okada_sp_installation_0_conditional_z190':
        raise ValueError('Selected q32 exposure is not original installation 0')
    receipts = []

    def same_exposure(row, native_key='pts', actual_key='actual_pts'):
        p = row.get(native_key)
        declared = row.get(actual_key)
        resolved = aliases.get(int(p)) if p is not None else None
        if resolved is not None and declared is not None and resolved != int(declared):
            raise ValueError('Frozen typed source actual exposure alias changed')
        return resolved if resolved is not None else (int(declared) if declared is not None else None)

    compaction = manifest['original_motion_input_compaction']
    main_packet = json.loads(auth(compaction['main_candidate']).read_text(encoding='utf8'))
    main4 = next(r for r in main_packet['classification'] if r['block'] == 'main4')
    current_main4 = next(r for r in compaction['blocks'] if r['block'] == 'main4')
    if not all(current_main4['exact_prefix_fields'].values()):
        raise ValueError('Current main4 source prefix was not authenticated')
    with np.load(auth(main4['map']), allow_pickle=False) as value:
        keep = value['keep_old_point_indices'].copy()
    with np.load(auth(main4['original_admission_cache']), allow_pickle=False) as value:
        active_raw = value['active_original_row_index'].copy()
        raw_indices = active_raw[keep]
        pairs, offsets = value['pairs'], value['offsets']
        native_stamps = np.unique(pairs)
        ia, ib = value['endpoint_index_a'], value['endpoint_index_b']
        # Confirm the clock table against each original source pair rather
        # than assume that an endpoint index names a camera row.
        edge = np.searchsorted(offsets[1:], np.arange(len(ia)), side='right')
        if not (np.array_equal(native_stamps[ia], pairs[edge, 0])
                and np.array_equal(native_stamps[ib], pairs[edge, 1])):
            raise ValueError('Original motion endpoint indices/pair clock disagree')
        stamps = np.asarray([aliases[int(p)] for p in native_stamps], np.int64)
        a, b = value['source_uv_a'][raw_indices], value['source_uv_b'][raw_indices]
        ia, ib = ia[raw_indices], ib[raw_indices]
    if len(keep) != current_main4['after_points']:
        raise ValueError('Saved q32 main4 compact point count differs')
    receipts.append(dict(family='programme:main4:motion', selected_exposure_complete=True,
        current_input_point_count=len(keep), admission_cache=main4['original_admission_cache'],
        compaction_map=main4['map'], source_endpoint_clock_verified_against_original_pairs=True,
        selected_exposure_endpoint_count=int((stamps[ia] == PTS).sum() + (stamps[ib] == PTS).sum())))
    # The other installation-0 original blocks have separate native intervals.
    for item in main_packet['classification']:
        if item['block'] == 'main4':
            continue
        with np.load(auth(item['original_admission_cache']), allow_pickle=False) as value:
            other = np.unique(value['pairs'])
        if PTS in [aliases[int(p)] for p in other]:
            raise ValueError('Another source block contains the selected exposure')
        receipts.append(dict(family='programme:' + item['block'] + ':motion',
            selected_exposure_complete=True, selected_exposure_endpoint_count=0,
            admission_cache=item['original_admission_cache'], original_pair_actual_aliases_checked=True))

    absolute_contract = manifest['original_absolute_input_compaction']
    original = pickle.loads(auth(absolute_contract['original_packet']).read_bytes())
    for family, expected_installation in [('secondary', 1), ('low', 2)]:
        inventory = next(r for r in original['actual_source_inventory'] if r['block'] == family)
        if ('installation_' + str(expected_installation)) not in inventory['installation']:
            raise ValueError('Original source installation ownership changed')
        receipts.append(dict(family='programme:' + family + ':motion',
            selected_exposure_complete=True, selected_exposure_endpoint_count=0,
            reason='Source bundle belongs to another physical installation; programme exposure has unique owner',
            source_inventory=inventory, selected_exposure_camera_installation_id=camera_installations[PTS]))

    original_rows = [(i, item) for i, item in enumerate(original['programme_absolute_rows'])
                     if item['collection_role'] == 'FIT' and item['group_actual_pts'] == PTS]
    correction_path, correction, correction_reference = packet(manifest['original_ice_pixel_correction'])
    qualification_path, qualification, qualification_reference = packet(manifest['original_source_point_qualification'])
    row_indices = {i for i, _ in original_rows}
    # These exact old source revisions have no selected-row updates/exclusions.
    # Never insert the later, not-yet-activated 65-record source correction.
    if any(r['original_packet_row_index'] in row_indices for r in correction['records']):
        raise ValueError('Selected exposure needs current public correction adapter')
    if any(r['original_packet_row_index'] in row_indices for r in qualification['records']):
        raise ValueError('Selected exposure needs current public qualification adapter')
    if any(r['actual_pts'] == PTS for block in absolute_contract['blocks'] for r in block['row_maps']):
        raise ValueError('Selected absolute source was compacted; apply original point map first')
    # Remaining legacy scalar-role masks are elsewhere, not this frame.
    main4_start = next(b['compact_start'] for b in absolute_contract['blocks'] if b['block'] == 'main4')
    for contract_name, mask_key in [('original_source_role_retirement', 'active_runtime_global_scalar_slots'),
                                  ('static_motion_source_role_retirement', 'global_scalar_slots'),
                                  ('original_source_point_qualification', 'active_runtime_global_scalar_slots')]:
        if any(int(v) >= main4_start for v in manifest[contract_name].get(mask_key, [])):
            raise ValueError('Current main4 source contains legacy scalar retirement needing exact mask application')
    rows = [deepcopy(item['source_row']) for _, item in original_rows]
    receipts.append(dict(family='programme:absolute', selected_exposure_complete=True,
        original_packet=absolute_contract['original_packet'], correction=correction_reference,
        qualification=qualification_reference, compact_selected_rows=[],
        original_packet_row_indices=sorted(row_indices),
        source_rows=[dict(original_packet_row_index=i, id=r['source_row']['id'],
                          points=len(r['source_row']['uv']), split=r['source_row']['split'])
                     for i, r in original_rows]))

    install()
    from structured_programme_source_candidate.native_ice_curve_source import NativeIceCurveFactor
    from calibration.solver_runtime.source_point_qualification import load_source_point_qualification
    source = NS()
    factors = [('native_ice_factor', 'additional_real_native_ice_curves'),
               ('native_board_factor', 'additional_real_native_board_edges'),
               ('native_board_material_point_factor', 'additional_real_native_board_material_points'),
               ('circle_factor', 'additional_real_native_board_circles'),
               ('native_board_track_factor', 'additional_native_static_board_tracks'),
               ('persistent_ice_factor', 'additional_real_native_persistent_ice_long_baseline')]
    for name, contract_key in factors:
        contract = manifest[contract_key]
        path, data, reference = packet(contract)
        if name == 'persistent_ice_factor':
            if str(PTS) in json.dumps(data):
                raise ValueError('Persistent source needs its current track endpoint reader')
            active_rows, held = [], []
        elif name == 'native_board_track_factor':
            active_rows, held = [], []
            for literal in data['rows']:
                pa = same_exposure(literal, 'source_pts', 'source_actual_pts')
                pb = same_exposure(literal, 'target_pts', 'target_actual_pts')
                if PTS not in (pa, pb):
                    continue
                row = deepcopy(literal)
                row.update(source_actual_pts=pa, target_actual_pts=pb)
                (active_rows if row['split'] == 'fit' else held).append(row)
            anchors = [r for r in data.get('source_anchor_observations', [])
                       if same_exposure(r, 'native_pts', 'actual_pts') == PTS]
            if anchors:
                raise ValueError('Selected track source anchor needs current typed anchor reader')
        else:
            active_rows, held = [], []
            for literal in data['rows']:
                if same_exposure(literal) != PTS:
                    continue
                row = deepcopy(literal)
                row['actual_pts'] = PTS
                row['uv'] = np.asarray(row['uv'], float)
                (active_rows if row['split'] == 'fit' else held).append(row)
        if name == 'native_ice_factor':
            current = NativeIceCurveFactor.__new__(NativeIceCurveFactor)
            current.rows, current.held, current.contract = active_rows, held, contract
            for row in current.rows:
                row['sigma'] = np.broadcast_to(np.asarray(row['sigma_px'], float), (len(row['uv']),)).copy()
            qualification_packet = data
            if 'source_scanline_operator_revision' in data:
                from calibration.solver_runtime.truth_source_scanline_revision import original_qualified_packet
                qualification_packet = original_qualified_packet(path, data)
            current.source_point_qualification = load_source_point_qualification(path, qualification_packet, source_family='additional_native_ice_curves')
            current._active_fit_view_cache = None
        else:
            current = NS(rows=active_rows, held=held, contract=contract)
        if name == 'native_board_track_factor':
            current.fit_evaluation_rows, current.anchor_rows = active_rows, []
        setattr(source, name, current)
        receipts.append(dict(family=name, selected_exposure_complete=True, packet=reference,
            selected_source_ids=[r['id'] for r in active_rows], excluded_HELD_ids=[r['id'] for r in held],
            native_qualification_from_public_typed_active_view=(name == 'native_ice_factor')))

    # Original shared programme print patterns are bound by their constructor
    # to the secondary installation. All external source row clocks are stored
    # in this same authenticated absolute export and lie outside this frame.
    if any(PTS == r.get(k) for value in original['external']['row_sets'].values()
           if isinstance(value, list) for r in value if isinstance(r, dict)
           for k in ('actual_pts', 'source_actual_pts', 'anchor_actual_pts')):
        raise ValueError('An external source is present at the selected exposure')
    receipts.extend([
        dict(family='shared_board_patterns', selected_exposure_complete=True,
             selected_exposure_endpoint_count=0, reason='Programme frame owner is secondary installation; external aliases checked'),
        dict(family='external_main_low', selected_exposure_complete=True,
             selected_exposure_endpoint_count=0, authenticated_row_sets_from=absolute_contract['original_packet']),
        dict(family='physical_cross_target_support', selected_exposure_complete=True,
             pixel_markers=0, reason='Native support rectangles are nonpixel source constraints'),
        dict(family='wide_source', selected_exposure_complete=True,
             pixel_markers=0, reason='Single source exposure 56218880 is outside selected programme exposure'),
        dict(family='direct_manual_centre_secondary', selected_exposure_complete=True,
             additional_unique_pixel_markers=0,
             reason='Public producer selects unchanged subsets of the same native board/ice packet UV already collected'),
        dict(family='reference_geometry_camera_priors_and_derived_guard_queries', selected_exposure_complete=True,
             pixel_markers=0, reason='No independent current pixel inputs')])
    bundle = NS(a=a, b=b, ia=ia, ib=ib, stamps=stamps, motion_rows=2*len(a))
    block = NS(name='main4', bundle=bundle, groups=[(PTS, rows)],
               quadratic=np.empty(2*len(a)+2*sum(len(r['uv']) for r in rows), bool))
    state = NS(original=NS(aliases=aliases), bypts=dict.fromkeys(np.unique(actual)),
               blocks=[block], patterns=None, external=None, cross_fit=[])
    wide = NS(contract=dict(source_pts=56218880), uvs=[], ad_packet=dict(rows=[]), trace_uv=[])
    groups, context_receipts, _ = collect_current_active_source_points(state, source, wide)
    uv = next(values for pts, values in groups if pts == PTS)
    unique = np.unique(uv, axis=0)
    metadata = dict(schema='rinkspace.current-active-source-pixels-single-exposure-diagnostic.v1',
        complete_current_active_input=False, complete_active_input_for_selected_actual_exposure=True,
        programme_coverage='single_actual_exposure', selected_actual_pts=PTS, q_sha256=Q,
        camera=dict(path=str(CAMERA), sha256=file_sha(CAMERA)), input_manifest=dict(path=str(manifest_path), sha256=file_sha(manifest_path)),
        source_video_sha256=camera_metadata['source_sha256'], time_base=camera_metadata['source_time_base'],
        model_projected_points_included=False, historical_or_retired_points_included=False,
        nonpixel_placeholder_constraints_included=False, extra_point_labels=False,
        family_receipts=receipts, scoped_live_collector_receipts=context_receipts,
        point_count_before_exact_coordinate_deduplication=len(uv), displayed_unique_coordinates=len(unique),
        not_valid_for_complete_programme_export=True,
        not_recovered_for_complete_programme_display=['secondary motion arrays', 'low appended 194 current motion inputs and exact actual endpoint chart'],
        existing_full_prepared_input_cache_not_loaded=str(RUN.parent / 'prepared_full8503_state.pickle'),
        public_source_point_collector_was_used=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / 'q32_current_source_pixels_actual229702144_ONLY.npz',
        actual_pts=np.asarray([PTS], np.int64), offsets=np.asarray([0, len(unique)], np.int64),
        uv=unique, metadata_json=json.dumps(metadata, ensure_ascii=False))
    (OUT / 'q32_current_source_pixels_actual229702144_ONLY_receipt.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(output=str(OUT), actual_pts=PTS, active_point_count=len(uv),
        displayed_unique_coordinates=len(unique), selected_exposure_complete=True,
        complete_programme_point_archive=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
