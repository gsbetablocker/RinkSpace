"""Bind two actually reviewed skate pixels to the existing motion exclusion API.

Reads authenticated original arrays only; no new model, fit, or RGB decoding.
The four source images were already reviewed at both endpoints by the root.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.solver_runtime import install
install()
from structured_programme_source_candidate.static_motion_source_role_retirement import _read_dynamic_validity_append

BASE = Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def ref(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest)


def checked(reference):
    assert ref(reference['path'])['sha256'] == reference['sha256']
    return Path(reference['path'])


def write(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists() and path.read_text(encoding='utf-8') != text:
        raise ValueError('Preserve a different existing source review artifact')
    path.write_text(text, encoding='utf-8')
    return ref(path)


def main():
    config = read(ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_review86_ce235f_wide_curve_budget_20261012.json')
    parent_path = Path(config['source_options']['--static-motion-source-role-retirement-source'])
    compound = read(parent_path)
    previous = compound['source_reviewed_dynamic_validity_append']
    old = read(checked(previous))
    review_path = BASE / 'source_pixel_review_additional/original_LK_wholecut_material_review/whole_cut5_continuous_native_material_regions_PENDING/source_only_material_review_resume_pages074_076_184_230_and47_PENDING_ROOT_20261011.json'
    review = read(review_path)
    section = review['47_coarse_actor_ROI_candidates']
    for key in ('raw_native_endpoint_montage', 'magnified_display_detail'):
        checked(section[key])
    additions = [r for r in section['points'] if r['index'] in (154804, 155102)]
    assert len(additions) == 2
    state = read(ROOT / 'events/ankara_2026/programmes/okada_mei_sp/review_20261011_systematic_cause_status.json')
    identity = read(checked(state['latest_diagnostic_model']['state_identity']))
    truth = read(Path(state['latest_diagnostic_model']['state_identity']['path']).parent / 'truth_restoration_state.json')
    manifest = read(checked(truth['source_input_manifest_reference']))
    active_retirement = manifest['static_motion_source_role_retirement']
    assert active_retirement['source_reviewed_dynamic_validity_append'] == previous
    assert identity['q_sha256'] == truth['complete_q_sha256']
    prior_slots = set(active_retirement['global_scalar_slots'])
    block_start = active_retirement['programme_block']['original_start']
    packet = deepcopy(old)
    paths = {k: checked(old[k]) for k in ('original_raw_motion', 'original_selected_rows', 'original_source_ledger', 'original_compaction_map')}
    with np.load(paths['original_raw_motion'], allow_pickle=False) as raw, np.load(paths['original_selected_rows'], allow_pickle=False) as selected, np.load(paths['original_source_ledger'], allow_pickle=False) as ledger, np.load(paths['original_compaction_map'], allow_pickle=False) as compact:
        active = ledger['active_original_global_index']
        ids = selected['original_global_row_indices']
        mapping = compact['old_point_to_compact_point']
        offsets, xy0, xy1 = raw['offsets'], raw['xy0'], raw['xy1']
        source_uv, target_uv, sigmas = ledger['source_uv_a'], ledger['source_uv_b'], ledger['original_sigma_xy']
        for evidence in additions:
            assert evidence['resume_visual_review_status'] == 'not_far_board_foreground_free_leg_white_skate_motion_blur_at_both_endpoints'
            pair, prefix = evidence['original_raw_pair_index'], evidence['index']
            start, stop = map(int, offsets[pair:pair+2])
            found = np.flatnonzero(np.all(xy0[start:stop] == evidence['source_uv'], axis=1) & np.all(xy1[start:stop] == evidence['target_uv'], axis=1))
            assert len(found) == 1
            g = start + int(found[0])
            selected_rows = np.flatnonzero(ids == g)
            assert len(selected_rows) == 1
            full, current = int(active[prefix]), int(mapping[prefix])
            assert current >= 0
            assert np.array_equal(source_uv[full], evidence['source_uv']) and np.array_equal(target_uv[full], evidence['target_uv'])
            assert not {block_start + 2*current, block_start + 2*current + 1}.intersection(prior_slots)
            same_pair = [r for r in old['records'] if r['original_raw_pair_index'] == pair]
            assert len(same_pair) == 1
            row = dict(original_raw_cut5_point_index=g, original_raw_pair_index=pair,
                original_raw_pair_local_index=int(found[0]), original_selected_cut5_row=int(selected_rows[0]),
                original_full_low_bundle_row=full, current_original_prefix_bundle_index=prefix,
                source_pts=evidence['source_pts'], target_pts=evidence['target_pts'],
                source_native_pair_pts=raw['pair_pts'][pair].tolist(),
                source_uv=xy0[g].tolist(), target_uv=xy1[g].tolist(), source_sigma_px=sigmas[full].tolist(),
                split='fit', active_compacted_motion_point_index=current,
                source_role='moving_human_foreground', both_actual_RGB_endpoints_manually_reviewed=True,
                either_endpoint_source_visible_moving_human=True, source_qualified_for_static_scene_motion=False,
                source_camera_or_projection_or_residual_used_for_validity=False,
                original_UV_sigma_weights_clock_split_preserved=True,
                source_validity_filter_not_robust_downweighting=True,
                expected_original_finite_plane=[0., 0., 0., 0.],
                source_raw_references=deepcopy(same_pair[0]['source_raw_references']),
                source_material_validity_authority='Root reviewed both native endpoints in the source-only montage and magnified source detail: the lower white skate/ankle boundary covers this pixel; it is moving foreground, not stationary print.',
                source_only_material_review=ref(review_path),
                current_q32_original_motion_scalar_slots=[block_start + 2*current, block_start + 2*current + 1],
                not_previously_in_current_static_motion_retirement=True)
            packet['records'].append(row)
    packet.update(retired_dynamic_point_count=len(packet['records']),
        derived_from_existing_dynamic_validity=previous, source_only_material_review=ref(review_path),
        root_actual_two_endpoint_RGB_review=section['magnified_display_detail'],
        original_two_retired_records_preserved_exactly=True, new_current_motion_points_retired=2,
        current_before_manifest=truth['source_input_manifest_reference'], current_before_q_sha256=identity['q_sha256'],
        new_fit_started=False)
    target = BASE / 'original_motion_current_four_reviewed_skate_points_validity_20261012.json'
    packet_ref = write(target, packet)
    admitted = _read_dynamic_validity_append(packet_ref)
    assert admitted['records'][:len(old['records'])] == old['records']
    successor = deepcopy(compound)
    successor['source_reviewed_dynamic_validity_append'] = packet_ref
    successor['derived_from_existing_compound_validity'] = ref(parent_path)
    compound_ref = write(BASE / 'original_UI_retirement_plus_current_four_skate_points_20261012.json', successor)
    receipt = dict(schema='rinkspace.actual-original-moving-skate-source-validity-revision.v1',
        passed=True, current_manifest=truth['source_input_manifest_reference'], current_q_sha256=identity['q_sha256'],
        parent=ref(parent_path), source_packet=packet_ref, candidate_compound=compound_ref,
        added_original_prefix_indices=[r['index'] for r in additions],
        added_raw_indices=[r['original_raw_cut5_point_index'] for r in packet['records'][2:]],
        public_original_array_clock_UV_sigma_compaction_validator_passed=True,
        current_parent_had_only_other_two_skate_points_retired=True,
        original_two_records_and_UI_retirement_preserved=True,
        source_coordinates_world_sigma_weights_not_reestimated=True,
        candidate_not_installed_in_production_config=True, complete_graph_or_solver_started=False)
    result = write(BASE / 'original_motion_four_skate_points_public_source_validation_20261012.json', receipt)
    print(json.dumps(dict(passed=True, receipt=result, candidate=compound_ref), ensure_ascii=False))


if __name__ == '__main__':
    main()
