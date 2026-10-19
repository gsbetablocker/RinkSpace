"""Export the specifically reviewed B1 source delta, preserving the mother view."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import cv2
import numpy as np
from calibration.program_tracking.strict_absolute_source import (
    StrictAbsolutePackWriter, StrictAbsoluteSourcePack, _selected_review_rows, _exact_binding)
from calibration.program_tracking.strict_model_scope import ExactSourceScopeReview, REVIEW_SCHEMA, file_sha

WORK = Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/diagnostics/current56_shared_active_set_filter_20261009/systematic_continuation_20261010/full_programme_strict_source_review_20261012')
PILOT = WORK / 'current_B1_gap_actual_ice_scope_RGB_vector_20261013_v3'
FAILED = WORK / 'current_B1_gap_actual_RGB_vector_20261013_v1'
OUTPUT = WORK / 'current_B1_gap_reviewed_source_delta_20261013_v2'
EXACT_REVIEWED = {
    229491200: [1231, 1233, 1235, 1237, 1239, 1241],
    229491456: [1294, 1295, 1297, 1299],
    229491712: [1385, 1387],
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def reference(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha(path))


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve prior source packs and review facts')
    OUTPUT.mkdir(parents=True)
    receipt = read(PILOT / 'actual_B1_gap_review.json')
    failed = read(FAILED / 'actual_B1_gap_review.json')
    for ref in receipt['implementation_references']:
        if file_sha(ref['path']) != ref['sha256']:
            raise ValueError('Pilot source implementation changed')
    facts, failures = [], []
    with np.load(PILOT / 'actual_B1_gap_profiles.npz', allow_pickle=False) as arrays:
        for record in receipt['records']:
            pts = record['source_binding']['actual_pts']
            selected = EXACT_REVIEWED.get(pts, [])
            observed = [c['candidate_index'] for c in record['actual_source_supported_for_point_review']]
            if observed != selected:
                raise ValueError('Exact manually reviewed source candidate set changed')
            raw = cv2.imread(record['raw_reference']['path'], cv2.IMREAD_COLOR)
            binding = record['source_binding']
            if (raw is None or raw.shape != (1080, 1920, 3)
                    or hashlib.sha256(raw.tobytes()).hexdigest() != binding['decoded_native_BGR_sha256']):
                raise ValueError('Reviewed PNG does not contain the recorded original native BGR')
            if file_sha(record['raw_reference']['path']) != record['raw_reference']['sha256']:
                raise ValueError('Reviewed native source PNG changed')
            q = arrays[f'{pts}_current_candidate_query']
            normals = arrays[f'{pts}_current_query_normal']
            literal_points = []
            for point in record['actual_source_supported_for_point_review']:
                ci = point['candidate_index']
                sides, quantiles = np.asarray(point['sides']), np.asarray(point['quantiles'])
                if (not np.array_equal(sides, quantiles[:, 1])
                        or not np.array_equal(np.asarray(point['uv']), sides.mean(0))):
                    raise ValueError('A source paint midpoint changed its actual two q50 sides')
                literal_points.append(dict(point, actual_normal=normals[q[ci]].tolist(),
                    actual_complete_two_signed_RGB_interfaces_reviewed=True,
                    whole_visible_paint_band_reviewed=True, inner_stripe_used_as_whole_band=False,
                    scope='ice', physical_material_id='B1:segment0'))
            facts.append(dict(actual_pts=pts, native_pts=binding['native_pts'],
                decoded_BGR_sha256=binding['decoded_native_BGR_sha256'],
                raw_reference=record['raw_reference'], candidate_indices=selected,
                actual_source_points=literal_points,
                source_scope_and_point_review_complete_for_listed_points=True,
                same_query_competing_complete_blue_bands=record['multiple_complete_actual_blue_bands_at_same_query']))
    with np.load(FAILED / 'actual_B1_gap_profiles.npz', allow_pickle=False) as arrays:
        for record in failed['records']:
            pts = record['source_binding']['actual_pts']
            start = record['original_chroma_candidates']
            flags = arrays[f'{pts}_current_candidate_flags']
            indices = np.arange(start, len(flags))
            rejected = indices[(flags[indices] & 2) != 0]
            failures.append(dict(actual_pts=pts, original_RGB_vector_scope_rejected_candidates=[dict(
                candidate_index=int(i), original_candidate_flags=int(flags[i]),
                observed_UV=arrays[f'{pts}_current_candidate_uv'][i].tolist(),
                observed_two_sides_UV=arrays[f'{pts}_current_candidate_sides'][i].tolist()) for i in rejected],
                reason='full_search_read_outside_actual_ice_scope_including_board_kickplate_transition',
                source_rejected_UV_preserved=True, source_FIT_admitted=False))
    facts_packet = dict(schema='rinkspace.exact-B1-three-frame-original-source-point-review.v1',
        reviewer='gray_review_resume_agent_original_source_review',
        reviewed_actual_source_pts=sorted(EXACT_REVIEWED), records=facts,
        prior_failed_full_search_reference=reference(FAILED / 'actual_B1_gap_review.json'),
        prior_failed_full_search_scope_rejections=failures,
        current_scope_successor_receipt=reference(PILOT / 'actual_B1_gap_review.json'),
        actual_normal_RGB_profile_review_reference=reference(PILOT / 'three_frame_actual_normal_RGB_q50_review.png'),
        actual_native_RGB_profiles_reference=reference(PILOT / 'actual_B1_gap_profiles.npz'),
        model_navigation_used_for_material_identity=True, projected_UV_used_as_source_pixel_truth=False,
        source_interfaces_cropped_by_model_distance_or_expected_width=False,
        current_frozen_dataset_changed=False, full_shot_accuracy_accepted=False,
        review_note='原RGB连续三帧核实左侧完整蓝灰冰面带及两侧真实材料过渡；中段竞争/缺失接口仍 pending。')
    facts_path = OUTPUT / 'exact_B1_source_review_facts.json'
    facts_path.write_text(json.dumps(facts_packet, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf8')
    review_endpoints, rows = [], []
    for record in receipt['records']:
        binding = _exact_binding(record['source_binding'])
        for original in record['normalized_rows']:
            row = deepcopy(original)
            for key in ('uv', 'sigma_px', 'source_sides_uv', 'raw_candidate_uv', 'source_side_quantiles_uv'):
                row[key] = np.asarray(row[key], np.float64)
            row['original_point_indices'] = np.asarray(row['original_point_indices'], np.int64)
            if row['original_point_indices'].tolist() != EXACT_REVIEWED[binding['actual_pts']]:
                raise ValueError('Only exact reviewed source point indices may be exported')
            row.update(family='B1_missing_exposure_current_ice_scope_RGB_vector_source_delta',
                final_current_RGB_pixel_precision_checked=True,
                exact_original_source_agent_point_review=reference(facts_path))
            generated = _selected_review_rows([row], binding, facts_packet['reviewer'])
            for proof in generated.records.values():
                proof = deepcopy(proof)
                proof.update(review_authority='agent_original_source_review',
                    original_source_image_actually_reviewed=True,
                    review_note=facts_packet['review_note'],
                    source_image_evidence=dict(record['raw_reference'], kind='original_video_decoded_actual_native_PNG',
                        actual_pts=binding['actual_pts'], native_pts=binding['native_pts'],
                        decoded_BGR_sha256=binding['decoded_native_BGR_sha256']),
                    exact_original_source_agent_point_review=reference(facts_path))
                review_endpoints.append(proof)
            rows.append(row)
    source = dict(video_sha256=receipt['records'][0]['source_binding']['source_sha256'],
        time_base='1/15360', pixel_domain='native1920x1080')
    review_path = OUTPUT / 'exact_B1_source_scope_review.json'
    review_path.write_text(json.dumps(dict(schema=REVIEW_SCHEMA, source=source, reviewed_endpoints=review_endpoints),
        ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf8')
    review = ExactSourceScopeReview.load(review_path)
    references = [reference(facts_path), reference(review_path), reference(PILOT / 'actual_B1_gap_review.json'),
        reference(PILOT / 'actual_B1_gap_profiles.npz'), reference(Path(__file__))]
    references += receipt['implementation_references']
    references += [receipt['model_reference'], receipt['source_cache_reference'], receipt['frozen_dataset_reference']]
    pack_path = OUTPUT / 'B1_missing_three_exposures_reviewed_source_delta.npz'
    writer = StrictAbsolutePackWriter(pack_path, source=source,
        expected_actual_pts=[r['source_binding']['actual_pts'] for r in receipt['records']],
        original_references=references)
    for record in receipt['records']:
        writer.mark_exposure_processed(record['source_binding']['actual_pts'], source_binding=record['source_binding'])
    for row in rows:
        if writer.add_row(row, review) != len(row['uv']):
            raise ValueError('Public source writer did not admit the exact reviewed source set')
    output = writer.finish()
    consumer = StrictAbsoluteSourcePack(output['path'], expected_sha256=output['sha256'])
    if (sum(len(r['uv']) for r in consumer.fit_rows) != 12 or consumer.held_rows
            or consumer.metadata['source_uncertainty_ready_for_fit'] is not True):
        raise ValueError('Public source consumer differs from reviewed delta')
    result = dict(schema='rinkspace.B1-reviewed-source-gap-delta-reference.v1', source_pack=output,
        source_review_facts=reference(facts_path), source_scope_review=reference(review_path),
        newly_observed_material='B1:segment0', new_actual_exposures=sorted(EXACT_REVIEWED),
        points_per_exposure={str(p):len(v) for p,v in EXACT_REVIEWED.items()},
        public_strict_source_consumer_verified=True, original_frozen_dataset_changed=False,
        original_admitted_rows_retired=False, no_prior_pending_pixels_readded=True,
        source_readout_support_finite=True, independent_full_shot_accuracy_accepted=False, solver_started=False)
    final = OUTPUT / 'reviewed_B1_source_delta_reference.json'
    final.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf8')
    print(json.dumps(dict(reference=reference(final), **result), ensure_ascii=False))


if __name__ == '__main__':
    main()
