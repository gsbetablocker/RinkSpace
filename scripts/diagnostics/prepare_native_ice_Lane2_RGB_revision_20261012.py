"""Bind four RGB paint midpoints and three unmeasurable samples to native108.

Preserves all original files and public qualification/scanline authentication.
Only native source row data and exact reference/layout rebinding are prepared;
no state history, programme graph, residual, Jacobian, training or solve is read.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from calibration.solver_runtime.source_point_qualification import (
    PACKET_FIELD, file_sha256, original_uv_sha256, load_source_point_qualification,
    _direct_user_annotation)
from calibration.solver_runtime.truth_source_scanline_revision import (
    original_qualified_packet, validate_scanline_revision, MODE_FIELDS, REVISION_KEY)
from calibration.solver_runtime import truth_source_location_revision as common

SYS = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')
OUT = SYS / 'source_pixel_review_additional/native_ice_current_remaining_actual_RGB_review_20261012'
PACK = ROOT / 'events/ankara_2026/observations/source_packets/ice108_Lane2_RGB_paired_centres_qualified_20261012'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def ref(path): return dict(path=str(Path(path).resolve()), sha256=file_sha256(path))
def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return common._write(path, value)


def layout(packet, qualification):
    result, cursor = {}, 0
    for row in packet['rows']:
        if row['split'] != 'fit': continue
        keep = np.arange(len(row['uv'])) if qualification is None else np.flatnonzero(qualification.point_mask(row))
        result[row['id']] = np.arange(cursor, cursor + 2 * len(keep)).reshape(-1, 2).tolist()
        cursor += 2 * len(keep)
    return result, cursor


def layout_protocol(packet_path, packet, qualification):
    # Public integration validates source ownership, exact UV/sigma/physical
    # identities, active pairs and weights. This protocol has no camera or J.
    rows = []
    for source in packet['rows']:
        if source['split'] != 'fit': continue
        row = deepcopy(source)
        row['actual_pts'] = int(row.get('actual_pts', row['pts']))
        row['uv'] = np.asarray(row['uv'], float)
        row['sigma'] = np.broadcast_to(np.asarray(row['sigma_px'], float), (len(row['uv']),)).copy()
        rows.append(row)
    scales = np.concatenate([np.repeat(r['sigma'], 2) for r in rows])
    if qualification is not None: scales = scales[qualification.scalar_mask(rows)]
    return SimpleNamespace(packet_path=Path(packet_path).resolve(), rows=rows,
        scales=scales, row_count=len(scales), source_point_qualification=qualification,
        contract=dict(packet_sha256=file_sha256(packet_path)))


def main():
    config_path = ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_ce235f_wide_curve_budget_20261012.json'
    options = read(config_path)['source_options']
    current_path = Path(options['--additional-ice-source'])
    current = read(current_path)
    views = read(OUT / 'current_active_remaining_native_ice_RGB_review_inputs.json')
    if views['packet'] != ref(current_path): raise ValueError('Reviewed current packet changed')
    old_qualified = original_qualified_packet(current_path, current)
    old_qual = load_source_point_qualification(current_path, old_qualified, source_family='additional_native_ice_curves')
    old_manifest_path = Path(old_qualified[PACKET_FIELD]['path'])
    old_manifest = read(old_manifest_path)
    old_original_path = Path(old_manifest['original_packet']['path'])
    original = read(old_original_path)
    if old_manifest['original_packet'] != ref(old_original_path): raise ValueError('Immutable qualification parent changed')
    decisions_path = OUT / 'actual_current_native_ice_visual_decisions.json'
    decisions = {r['row_index']: r for r in read(decisions_path)['rows']}
    reading_path = OUT / 'ordinary_paint_two_actual_material_sides_candidate_readings.json'
    readings = {r['row_index']: r for r in read(reading_path)['rows']}
    updated = deepcopy(original)
    changes = []
    for index, candidate_index in ((38, 0), (40, 1), (42, 0), (44, 0)):
        row = updated['rows'][index]
        if _direct_user_annotation(row) or not decisions[index]['correction_confirmed']:
            raise ValueError('Only actually reviewed non-user transported pixels may change')
        candidate = readings[index]['point_decisions'][0]['all_actual_source_candidates'][candidate_index]
        a = np.asarray(candidate['actual_RGB_entering_interface']['actual_source_transition_candidate_uv'])
        b = np.asarray(candidate['actual_RGB_leaving_interface']['actual_source_transition_candidate_uv'])
        center = (a + b) * .5
        if not np.array_equal(center, candidate['actual_source_dark_stroke_center_candidate_uv']):
            raise ValueError('Center must be the exact two actual material-side average')
        item = dict(row_index=index, row_id=row['id'], original_point_index=0,
            source_pts=int(row['pts']), old_uv=deepcopy(row['uv'][0]), new_uv=center.tolist(),
            actual_entering_interface_uv=a.tolist(), actual_leaving_interface_uv=b.tolist(),
            source_measurement_reference=ref(reading_path), source_candidate_index=candidate_index,
            measurement_operator='geometric_midpoint_of_two_actual_material_interfaces',
            physical_RGB_review='same_paint_band_sides_actually_viewed_with_whole_source_context',
            broad_context_and_original_flags_retained=True,
            precision_or_independent_validation_accepted=False)
        changes.append(item)
        row['uv'][0] = center.tolist()
        row['source_pixel_measurement_revision'] = dict(
            original_parent_packet=ref(old_original_path),
            original_RGB_transport_and_profile_fields_preserved_as_history=True,
            original_point_indices=[0], source_measurement_reference=ref(reading_path),
            source_candidate_index=candidate_index, actual_side_midpoint_UV=center.tolist(),
            old_profile_match_is_not_midline_accuracy_confirmation=True,
            paint_footprint_is_not_centerline_covariance_or_free_center_allowance=True)
    updated['reviewed_Lane2_paint_midline_RGB_revision'] = dict(
        immutable_original_packet=ref(old_original_path), previous_selected_packet=ref(current_path),
        visual_decision_reference=ref(decisions_path), changed_points=changes,
        original_source_clock_sigma_normals_supports_roles_and_all_other_UV_unchanged=True,
        direct_user36_37_strokes_preserved=True, accuracy_accepted=False)
    base_path = PACK / 'unqualified_pixel_revision_parent.json'
    write(base_path, updated)
    manifest = deepcopy(old_manifest)
    manifest['original_packet'] = ref(base_path)
    manifest['previous_original_qualification'] = ref(old_manifest_path)
    manifest['previous_selected_packet'] = ref(current_path)
    for index in (38, 40, 42):
        row = updated['rows'][index]
        source = Path(row['source_raw_frame'])
        if ref(source)['sha256'] != row['source_file_sha256']: raise ValueError('Actual source RGB changed')
        manifest['records'].append(dict(row_index=index, row_id=row['id'], source_pts=int(row['pts']),
            original_uv_float64_sha256=original_uv_sha256(row), excluded_point_indices=[1],
            review_basis='original_rgb_only',
            reason='Actual wide RAW has a strong upper material band and weak diagonal lower paint entering the same query; no independently distinguishable complete pair of this Lane2 paint sides, so original RGB transport UV cannot measure its geometric center. No replacement or uniform translation inferred.',
            native_rgb_evidence=[dict(ref(source), source_pts=int(row['pts'])),
                dict(ref(OUT / f'row{index}_p1_both_bands_wide_RAW_current.png'), source_pts=int(row['pts']))]))
    qualification_path = PACK / 'point_qualification.json'
    write(qualification_path, manifest)
    qualified = deepcopy(updated)
    qualified[PACKET_FIELD] = ref(qualification_path)
    qualified_path = PACK / 'qualified_pixel_revision_parent.json'
    write(qualified_path, qualified)
    load_source_point_qualification(qualified_path, qualified, source_family='additional_native_ice_curves')
    # Preserve the already authenticated fixed-y/max_x source estimator. Its
    # exact two rows have not changed; only the containing qualified file ref.
    old_scan = current[REVISION_KEY]
    scan_receipt = read(old_scan['receipt_reference']['path'])
    if old_scan['receipt_reference'] != ref(old_scan['receipt_reference']['path']):
        raise ValueError('Original scanline receipt changed')
    scan_receipt['original_packet_reference'] = ref(qualified_path)
    scan_receipt['unchanged_saved_scanline_measurements_parent_receipt'] = deepcopy(old_scan['receipt_reference'])
    for record in scan_receipt['rows']:
        record['source_row_reference'] = dict(record['source_row_reference'], **ref(qualified_path))
    scan_path = PACK / 'unchanged_saved_scanline_receipt_rebound.json'
    write(scan_path, scan_receipt)
    final = deepcopy(qualified)
    final[REVISION_KEY] = dict(deepcopy(old_scan), parent_packet_reference=ref(qualified_path), receipt_reference=ref(scan_path))
    for index in old_scan['changed_row_indices']:
        for key in MODE_FIELDS: final['rows'][index][key] = deepcopy(current['rows'][index][key])
        final['rows'][index]['scanline_location_measurement'] = ref(scan_path)
    final_path = PACK / 'packet.json'
    write(final_path, final)
    validate_scanline_revision(qualified_path, final_path, old_scan['changed_row_indices'])
    loaded = original_qualified_packet(final_path, final)
    new_qual = load_source_point_qualification(final_path, loaded, source_family='additional_native_ice_curves')
    for index, (a, b) in enumerate(zip(current['rows'], final['rows'])):
        allowed = {'uv', 'source_pixel_measurement_revision'} if index in (38, 40, 42, 44) else set()
        if index in old_scan['changed_row_indices']: allowed.add('scanline_location_measurement')
        if {k:v for k,v in a.items() if k not in allowed} != {k:v for k,v in b.items() if k not in allowed}:
            raise ValueError('Unreviewed native source fields changed')
        if index not in (38,40,42,44) and a['uv'] != b['uv']: raise ValueError('Other native UV changed')
        if index in (38,40,42) and new_qual.point_mask(b).tolist() != [True,False]:
            raise ValueError('Three mixed-material pixels must leave active FIT exactly')
    old_pairs, old_count = layout(current, old_qual)
    new_pairs, new_count = layout(final, new_qual)
    if new_count != old_count - 6: raise ValueError('Only three scalar pairs may leave')
    documents, _ = common._rebound_documents(options, final_path, source_family='ice', mode_fields=MODE_FIELDS)
    old_policy = read(options[common.INTEGRATION])
    integration = documents[common.INTEGRATION]
    for old, new in zip(old_policy['records'], integration['records']):
        if old['family'] != 'native_ice':
            if old != new: raise ValueError('Board integration record changed')
            continue
        identity = old['source_row_id']
        if old['native_active_base_scalar_pairs'] != old_pairs[identity]:
            raise ValueError('Existing integration active source layout differs')
        new['native_active_base_scalar_pairs'] = new_pairs[identity]
        if {k:v for k,v in old.items() if k not in ('source_row_reference','packet_reference','native_active_base_scalar_pairs')} != {
                k:v for k,v in new.items() if k not in ('source_row_reference','packet_reference','native_active_base_scalar_pairs')}:
            raise ValueError('Native budget/sigma/mass/physical curve changed')
    integration['reviewed_Lane2_three_point_qualification_layout_rebind'] = dict(
        previous_policy=ref(options[common.INTEGRATION]), previous_ice_packet=ref(current_path),
        previous_native_ice_base_scalar_rows=old_count, current_native_ice_base_scalar_rows=new_count,
        removed_scalar_pairs=3, all_integration_weights_masses_sigmas_paths_and_budget_rules_unchanged=True,
        only_source_references_and_exact_active_base_scalar_pairs_rebound=True)
    names = {common.NATIVE:'native_authority.json', common.CONFIRMED:'confirmed_authority.json',
        common.GUARD:'full_stroke_guard.json', common.CAP:'stroke_caps.json',
        common.ENVELOPE:'location_envelope.json', common.INTEGRATION:'native_integration.json'}
    rebound_dir = OUT / 'source_options_rebind'
    refs = {}
    for key in (common.NATIVE, common.CONFIRMED): refs[key] = write(rebound_dir / names[key], documents[key])
    documents[common.GUARD]['primary_authority_reference'] = refs[common.NATIVE]
    refs[common.GUARD] = write(rebound_dir / names[common.GUARD], documents[common.GUARD])
    common._bind_document_refs(documents, refs)
    for key in (common.CAP, common.ENVELOPE, common.INTEGRATION): refs[key] = write(rebound_dir / names[key], documents[key])
    # Exercise the actual public integration reader with the immutable source
    # row protocol; no graph, camera projection, r/J or old history is loaded.
    from calibration.solver_runtime import install
    install()
    from calibration.solver_runtime.curve_source_grouping import SourceWithNativeBaseCurveIntegration
    board_ref = old_policy['packets']['board']
    board = read(board_ref['path'])
    if board_ref != ref(board_ref['path']): raise ValueError('Board source binding changed')
    board_qual = load_source_point_qualification(board_ref['path'], board, source_family='additional_native_board_edges')
    source = SimpleNamespace(contract={}, native_ice_factor=layout_protocol(final_path, final, new_qual),
        native_board_factor=layout_protocol(board_ref['path'], board, board_qual))
    bound = SourceWithNativeBaseCurveIntegration(source, refs[common.INTEGRATION]['path'])
    patch = {'--additional-ice-source':str(final_path.resolve()), **{k:v['path'] for k,v in refs.items()}}
    receipt = dict(schema='rinkspace.reviewed-native-ice-source-pixel-and-qualification-revision.v1',
        current_config=ref(config_path), previous_selected_packet=ref(current_path), derived_packet=ref(final_path),
        qualification=ref(qualification_path), previous_qualification=ref(old_manifest_path),
        visual_decisions=ref(decisions_path), authenticated_review_inputs=ref(OUT/'current_active_remaining_native_ice_RGB_review_inputs.json'),
        changed_points=changes, excluded_points=[dict(row_index=i, original_point_index=1) for i in (38,40,42)],
        original_all15_exclusions_retained=True, total_excluded_nonhuman_points=18,
        all108_source_rows_and_source_clock_order_retained=True, direct_user36_37_and_all_other_UV_sigma_supports_roles_preserved=True,
        retained_opening_R4_same_layer_point_indices={'8':[0,1,2],'9':[0,1,2]},
        unchanged_scanline_operator_rows=[25,28], scanline_receipt=ref(scan_path),
        protected_native_record_count=len(documents[common.NATIVE]['records']),
        protected_confirmed_record_count=len(documents[common.CONFIRMED]['records']),
        protected_observations_supports_caps_envelope_numeric_arrays_unchanged=True,
        native_integration_public_reader_passed=True, native_integration_contract=bound.native_base_curve_integration,
        native_old_active_scalar_rows=old_count, native_new_active_scalar_rows=new_count,
        source_options_patch=patch,
        rebound_files=[dict(option_key=k,parent_reference=ref(options[k]),derived_reference=refs[k]) for k in common.BOUND_KEYS],
        full_programme_graph_or_model_r_J_loaded=False, state_history_loaded=False,
        actual_native_R_J_compaction_not_exercised_here=True, production_files_changed=False,
        original_source_files_modified=False, accuracy_accepted=False, fitting_or_training_performed=False)
    output_path = OUT / 'reviewed_native_ice108_source_revision_source_options.json'
    write(output_path, receipt)
    print(json.dumps(dict(receipt=ref(output_path), packet=ref(final_path), source_options_patch=patch,
        public_qualification_and_scanline_and_native_integration_loading_passed=True),ensure_ascii=False))


if __name__ == '__main__': main()
