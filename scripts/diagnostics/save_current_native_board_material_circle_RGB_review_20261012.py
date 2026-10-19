"""Save the actual RGB review decisions and an uninstalled circle revision.

All input overview and point sheets were opened and visually reviewed. This
records source identity decisions, never a camera/model accuracy pass.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import numpy as np

OUT = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/native_board_material_circle_RGB_review_20261012')


def reference(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    return reference(path)


def main():
    inputpath = OUT / 'current_q32_native_board_material_circle_RGB_review_inputs.json'
    inputs = json.loads(inputpath.read_text(encoding='utf8'))
    packets = {}
    for family, ref in inputs['sources'].items():
        if reference(ref['path']) != ref:
            raise ValueError('Saved q32 source packet identity changed')
        packets[family] = json.loads(Path(ref['path']).read_text(encoding='utf8'))
    # These are current FIT rows: source blurredness is not an exclusion gate.
    blurred = {'M': {1, 4, 8, 13, 14, 15, 16, 17}, 'C': {8, 12, 15}}
    occluded_user = {91: {0, 1, 2}, 93: {1, 2, 3}, 94: {0}, 108: {31, 32, 48}}
    interrupted_user_print = {79: {5, 6}, 81: {4, 5, 6}}
    counts = {'B': 0, 'M': 0, 'C': 0}
    rows = []
    for original in inputs['rows']:
        row = deepcopy(original)
        family, index = row['family'], row['packet_row']
        record = packets[family]['rows'][index]
        row.pop('source_only_review_pending')
        row.update(source_only_review_complete=True, source_rgb_and_current_UV_actually_viewed=True,
            overview_sheet=str(OUT / f'{family}_rows_{sum(r["family"] == family for r in rows)//5:02d}_RGB_and_current_UV.png'),
            location_accuracy_not_certified=True, model_residual_or_projection_used=False)
        decisions = []
        for point_index, uv in enumerate(row['uv']):
            global_index = counts[family]
            decision = dict(point_index=point_index, current_uv=uv,
                point_sheet=str(OUT / f'{family}_points_{global_index//32:02d}_RAW4x_and_current_UV.png'),
                source_RGB_visible_material_role_review='No conclusive wrong material identity found',
                status='visible_material_role_no_conclusive_error', preserve_current_source=True,
                subpixel_or_latent_exact_boundary_accuracy_not_certified=True)
            if index in blurred.get(family, set()):
                decision.update(status='source_blur_prevents_unique_exact_localization',
                    source_RGB_visible_material_role_review='Printed shape/instance visible in context; this blurred local query does not uniquely establish an exact interface',
                    broad_source_or_working_weight_not_accuracy_certificate=True)
            if family == 'B' and index in (19, 20):
                decision.update(status='visible_TOP_role_exact_position_unresolved',
                    source_RGB_visible_material_role_review='Upper blue-cap role confirmed; blur/background does not uniquely certify the saved positive-chroma-gradient location. No conclusive wrong physical edge found',
                    detailed_source_view=str(OUT / 'B019_B020_actual_TOP_native_RGBA_4x.png'))
            if family == 'B' and point_index in occluded_user.get(index, set()):
                decision.update(status='direct_user_geometric_supervision_occluded_RGB',
                    source_RGB_visible_material_role_review='The direct user stroke continues through an occluder; not a directly visible local RGB interface',
                    direct_user_stroke_preserved=True)
            if family == 'B' and point_index in interrupted_user_print.get(index, set()):
                decision.update(status='direct_user_boundary_supervision_interrupted_print',
                    source_RGB_visible_material_role_review='Direct user perimeter supervision crosses a white/internal-logo interruption without a continuous visible external printed boundary',
                    direct_user_stroke_preserved=True)
            if family == 'B' and index == 100:
                if record['curve_observation_kind'] != 'horizontal_extent' or 'y ignored' not in record['source_y_role']:
                    raise ValueError('Expected pure horizontal extent changed')
                decision.update(status='pure_horizontal_extent_context_UV_not_2D_measurement',
                    source_RGB_visible_material_role_review='x observes the last red O horizontal extent; stored y is navigation context only',
                    draw_as_2D_current_pixel=False, source_extent_constraint_preserved=True)
            if family == 'C' and index == 1:
                reasons = {
                    0: 'Old UV is on the inner lettering rather than exterior circle boundary',
                    1: 'Old UV is on the inner lettering rather than exterior circle boundary',
                    2: 'Old UV is on the inner lettering rather than exterior circle boundary',
                    3: 'Old UV is at the inside flank of the right black stroke; exterior interface is a distinct physical role',
                    4: 'Old UV is at the white interior flank of the left black stroke rather than exterior interface',
                    5: 'Old UV is on interior lettering/white interior, not the exterior circle boundary',
                    6: 'Old UV is on white exterior above the visible black circle',
                    7: 'Old UV is at the interior side below the top black stroke, not its exterior interface',
                    8: 'Old UV is on white exterior below the visible black circle',
                    9: 'Old UV is on white exterior below the visible black circle'}
                decision.update(status='wrong_old_material_or_interface_role_RGB_confirmed',
                    source_RGB_visible_material_role_review=reasons[point_index], preserve_current_source=False,
                    exact_new_location_not_certified=True,
                    detailed_source_view=str(OUT / 'C001_actual_native_current_vs_OUTER_candidates_all10.png'),
                    old_monotone_negative_gray_transition_is_not_valid_unfilled_outer_circle_reader=True)
            decisions.append(decision)
            counts[family] += 1
        row['point_decisions'] = decisions
        rows.append(row)
    # A source-location candidate is retained without defeating any public flag.
    candidatepath = OUT / 'C001_actual_outer_stroke_public_reader_candidates.json'
    readings = json.loads(candidatepath.read_text(encoding='utf8'))
    current = packets['C']['rows'][1]
    delta = dict(schema='rinkspace.native-board-circle-outer-interface-UV-revision-CANDIDATE.v1',
        original_packet=inputs['sources']['C'], original_packet_row=1, row_id=current['id'],
        original_uv_float64_sha256=hashlib.sha256(np.asarray(current['uv'], np.float64).tobytes()).hexdigest(),
        original_uv=current['uv'], original_sigma_px=current['sigma_px'],
        candidate_outer_interface_uv=[x['actual_outer_interface_candidate_uv'] for x in readings['outermost_visible_material_interface_candidates']],
        candidate_source_review=reference(candidatepath),
        all_original_candidate_and_wider_context_flags_preserved=True,
        thin_black_stroke_center_not_used_as_outer_visible_boundary=True,
        unresolved_reader_interface_indices=[i for i, x in enumerate(readings['outermost_visible_material_interface_candidates'])
            if x['interface_candidate']['disposition'] != 'measured_material_transition'],
        actual_RGB_native_branch_identity_reviewed=True,
        source_exact_location_or_user_pixel_truth_not_certified=True,
        active_source_changed=False, source_weights_world_role_split_or_sampling_budget_changed=False,
        requires_explicit_source_revision_installation=True)
    delta_ref = write(OUT / 'C001_outer_interface_UV_revision_CANDIDATE_NOT_INSTALLED.json', delta)
    # Withdraw the wrong historical extraction without inventing new points or
    # reclassifying it as independent HELD evidence. A replacement is separate.
    revised = deepcopy(packets['C'])
    revised['rows'].pop(1)
    revised.update(current_target_version='current_C001_wrong_material_row_withdrawal_CANDIDATE_20261012',
        all_existing_FIT_and_holdout_rows_unchanged=False, source_installed_in_solver=False,
        source_revision=dict(original_packet=inputs['sources']['C'],
            excluded_rows=[dict(original_packet_row=1, row_id=current['id'], preserved_original_row=current,
                reason='The old monotone negative-gray transition reader selected inner letters or the wrong flank of an unfilled thin circle. Full10 RGB query review confirmed incompatible outer-interface semantics',
                replacement_UV_not_added=True, retired_row_not_reclassified_as_HELD=True)],
            other_rows_UV_sigma_role_split_clock_and_geometry_value_exact=True,
            source_location_candidate=delta_ref, activation_pending=True))
    fit = [r for r in revised['rows'] if r['split'] == 'fit']
    held = [r for r in revised['rows'] if r['split'] == 'holdout']
    revised.update(observation_count=len(revised['rows']), fit_observation_count=len(fit),
        holdout_observation_count=len(held), fit_uv_count=sum(len(r['uv']) for r in fit),
        fit_scalar_count=2 * sum(len(r['uv']) for r in fit),
        current_fit_scalar_count=2 * sum(len(r['uv']) for r in fit),
        current_primary_held_scalar_count=2 * sum(len(r['uv']) for r in held))
    withdrawal_ref = write(OUT / 'native_circle16_C001_RGB_wrong_material_withdrawal_CANDIDATE_NOT_INSTALLED.json', revised)
    report = dict(schema='rinkspace.current-native-board-material-circle-RGB-review-decisions.v1',
        current_q32_input_review=reference(inputpath), exact_current_source_packets=inputs['sources'],
        rows=rows, reviewed_FIT_row_count=len(rows), reviewed_stored_UV_counts=counts,
        true_2D_pixel_count=sum(counts.values()) - 1, pure_horizontal_extent_context_UV_count=1,
        all_overview_and_point_sheets_actually_opened_and_visually_reviewed=True,
        no_conclusive_additional_wrong_material_native_board_or_material_found=True,
        native_circle_wrong_old_row='native_Ad24_candidate_229582592',
        source_withdrawal_candidate=withdrawal_ref, outer_interface_revision_candidate=delta_ref,
        source_accuracy_or_model_accuracy_not_accepted=True,
        no_solver_or_model_projection_or_new_video_or_training_executed=True,
        current_active_packet_or_q32_state_not_modified=True)
    revision = OUT / 'native_circle16_C001_outer_interface_RAW_reviewed_UV_revision_NOT_FITTED.json'
    receipt = OUT / 'native_circle16_C001_outer_interface_public_installation_receipt.json'
    if revision.is_file() and receipt.is_file():
        report['final_source_action'] = dict(
            kind='Root-reviewed exterior-interface replacement of old C00110UV; preserved counts/weights/other rows',
            revised_packet=reference(revision), public_installation_receipt=reference(receipt),
            old_whole_row_withdrawal_proposal_superseded_not_installed=True,
            current_production_config_or_saved_q32_not_modified=True)
    final_ref = write(OUT / 'current_q32_native_board_material_circle_RGB_review_DECISIONS.json', report)
    print(json.dumps(dict(report=final_ref, source_withdrawal_candidate=withdrawal_ref,
        outer_interface_revision_candidate=delta_ref, reviewed_rows=len(rows), UV_counts=counts), ensure_ascii=False))


if __name__ == '__main__':
    main()
