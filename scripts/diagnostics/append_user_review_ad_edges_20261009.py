"""Append source-only external edges explicitly identified in user video review.

Retains every prior source row. RGB measurements are protected separately from
direct human drawings; they are never labelled independent holdout evidence.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
EVID = BASE / 'evidence/user_video_review_20261009/a13_complete_programme'
OUT = EVID / 'registered_ad_edges_and_left_goal_source_append'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p): return dict(path=str(Path(p).resolve()), sha256=sha(p))
def canonical(r): return json.dumps(r, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
def write(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = read(ROOT / 'events/ankara_2026/programmes/okada_mei_sp/full_source_resume_conditional_views_20261009.json')
    parent = Path(config['source_options']['--additional-board-source'])
    packet = read(parent)
    original_rows = deepcopy(packet['rows'])
    ad4_path = EVID / 'Ad4_pan_edges_and_low_board_drift/source_only_edge_candidate/actual_two_user_Ad4_external_edges_SOURCE_ONLY_blur_support_CANDIDATE.json'
    opening_path = EVID / 'opening_Ad3_Ad4_ice_relative_alignment/actual_opening_Ad3_Ad4_source_edges_and_same_a13_replay.json'
    pending = []
    for r in read(ad4_path)['rows']:
        item = deepcopy(r)
        item['id'] = r['id'].replace('_candidate', '_FIT_20261009')
        pending.append((item, ad4_path, EVID / 'Ad4_pan_edges_and_low_board_drift/source_review.json'))
    for r in read(opening_path)['source_rows']:
        ad, pts = r['advertisement_number'], r['pts']
        item = dict(id=f'user_opening_Ad{ad}_{r["edge"]}_{pts}_source_RGB_FIT_20261009',
                    pts=pts, actual_pts=r['actual_pts'], advertisement_number=ad,
                    curve=dict(kind='advertisement_boundary', advertisement_number=ad,
                               edge=r['edge'], extent_scope='external_border'),
                    curve_observation_kind='normal_interval', point_correspondence=False,
                    uv=[p['uv'] for p in r['probes']], normal=[p['normal'] for p in r['probes']],
                    normal_interval_px=[p['normal_interval_px'] for p in r['probes']],
                    source_raw_frame=r['source_raw_frame'], source_file_sha256=r['source_file_sha256'],
                    source_only_profile_measurements=deepcopy(r['probes']))
        pending.append((item, opening_path, EVID / 'opening_Ad3_Ad4_ice_relative_alignment/source_review.json'))
    additions = []
    for item, measurement, confirmation in pending:
        half = (np.asarray(item['normal_interval_px'])[:, 1] - np.asarray(item['normal_interval_px'])[:, 0]) / 2
        item.update(split='fit', sigma_px=np.maximum(6., half).tolist(),
                    sigma_semantics='max(existing 6 native px RGB engineering floor, source blur half support); not measured covariance or user pixel precision',
                    geometry_role='actual_visible_external_print_boundary', source_optical_normal_support_authorized=True,
                    source_RGB_actual_measurement=True, endpoint_observation=False,
                    sourceUV_origin='independently_viewed_native_RGB_external_edge_transition',
                    model_projection_used_to_generate_or_select_uv=False,
                    user_confirmation_reference=ref(confirmation), source_measurement_reference=ref(measurement),
                    user_confirmed_physical_identity=True, registered_human_drawing_claimed=False,
                    independent_validation_claimed=False, user_review_development_source=True)
        assert sha(item['source_raw_frame']) == item['source_file_sha256']
        additions.append(item)
    assert len(additions) == 5
    packet['rows'].extend(additions)
    assert packet['rows'][:len(original_rows)] == original_rows
    assert len({r['id'] for r in packet['rows']}) == len(packet['rows'])
    packet['user_review_20261009_external_edges_append'] = dict(parent=ref(parent),
        preserved_original_row_count=len(original_rows), added_row_ids=[r['id'] for r in additions],
        original_rows_all_fields_order_UV_sigma_support_split_exact=True,
        new_RGB_measurements_not_direct_human_drawing_pixels=True,
        new_FIT_scalar_rows=sum(2 * len(r['uv']) for r in additions),
        no_original_HELD_promoted=True, no_new_world_or_camera_parameters=True)
    packet['observation_count'] = len(packet['rows'])
    packet['fit_observation_count'] = sum(r['split'] == 'fit' for r in packet['rows'])
    packet['holdout_observation_count'] = sum(r['split'] == 'holdout' for r in packet['rows'])
    packet['current_fit_scalar_count'] = sum(2 * len(r['uv']) for r in packet['rows'] if r['split'] == 'fit')
    packet['current_primary_held_scalar_count'] = sum(2 * len(r['uv']) for r in packet['rows'] if r['split'] == 'holdout')
    target = OUT / 'board_parent95_plus_user_Ad3_Ad4_five_RGB_edges_FIT_v1.json'
    write(target, packet)
    records = []
    for index, row in enumerate(additions, len(original_rows)):
        record = {k:deepcopy(row[k]) for k in ['id', 'pts', 'actual_pts', 'curve', 'uv', 'sigma_px',
                                             'source_raw_frame', 'source_file_sha256',
                                             'user_confirmation_reference', 'source_measurement_reference']}
        record.update(family='native_board_edge', packet_reference=ref(target), original_row_index=index,
                      source_row_canonical_json_sha256=hashlib.sha256(canonical(row).encode('utf-8')).hexdigest(),
                      user_confirmed_physical_identity=True, model_projection_used_to_generate_or_select_uv=False,
                      source_sigma_or_UV_changed=False, practical_location_uncertainty_px=4.,
                      practical_uncertainty_is_user_pixel_precision=False,
                      practical_uncertainty_semantics='Existing 4 native px engineering registration/location allowance in addition to measured source blur support; fixed before fitting.',
                      manual_registered_drawing_provenance_not_claimed=True, extra_independent_FIT_weight_rows=0,
                      independent_validation_claimed=False)
        records.append(record)
    authority_add = OUT / 'new_five_user_identified_RGB_edge_confirmed_authority_records_v1.json'
    write(authority_add, dict(schema='rinkspace.user-confirmed-native-source-authority-additions.v1',
                             records=records, old_confirmed_records_not_replaced=True,
                             measured_pixels_are_RGB_not_registered_human_strokes=True))
    receipt = dict(schema='rinkspace.exact-prefix-user-ad-source-append.v1', board_packet=ref(target),
                   new_confirmed_record_additions=ref(authority_add), parent=ref(parent),
                   old_rows=len(original_rows), new_rows=len(packet['rows']), added_FIT_scalars=34,
                   old_rows_canonical_sha256=hashlib.sha256(canonical(original_rows).encode('utf-8')).hexdigest(),
                   retained_prefix_canonical_sha256=hashlib.sha256(canonical(packet['rows'][:len(original_rows)]).encode('utf-8')).hexdigest(),
                   old_HELD_all_fields_unchanged=True, no_solver_or_runtime_modified=True,
                   public_factor_operator_replay_pending=True, accuracy_accepted=False)
    write(OUT / 'board_edge_append_receipt_v1.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__': main()
