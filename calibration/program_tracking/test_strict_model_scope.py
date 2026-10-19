from copy import deepcopy
import json

import numpy as np
import pytest

from calibration.program_tracking.strict_model_scope import (
    ExactSourceScopeReview, REVIEW_SCHEMA, file_sha, load_motion_partition,
    partition_motion_inputs, write_motion_partition,
)


SOURCE = dict(video_sha256='a' * 64, time_base='1/15360', pixel_domain='native1920x1080')


def review_packet(tmp_path):
    image = tmp_path / 'source_image_evidence.json'
    image.write_text('{}', encoding='utf8')
    def endpoint(pts, uv, material='C'):
        return dict(actual_pts=pts, native_pts=pts, uv=uv, scope='ice',
            physical_material_id=material, physical_material_point_id=material + '_point', decision='allow',
            static_material_confirmed=True, pixel_location_confirmed=True,
            original_source_image_actually_reviewed=True,
            model_projection_or_fit_residual_used_for_identity=False,
            review_authority='agent_original_source_review', reviewer='test',
            review_note='test fixture only', decoded_BGR_sha256='b' * 64,
            source_image_evidence=dict(path=str(image), sha256=file_sha(image)))
    return dict(schema=REVIEW_SCHEMA, source=deepcopy(SOURCE), reviewed_endpoints=[
        endpoint(10, [20, 30]), endpoint(20, [21, 31]),
        endpoint(10, [40, 50]), endpoint(20, [41, 51], material='B1'),
    ])


def test_pair_needs_two_exact_pixels_of_the_same_material(tmp_path):
    review = ExactSourceScopeReview(review_packet(tmp_path))
    a = np.array([[20, 30], [40, 50], [20.0000001, 30], [60, 70.]])
    b = np.array([[21, 31], [41, 51], [21, 31], [61, 71.]])
    result = partition_motion_inputs(a, b, np.full(4, 10), np.full(4, 20), review)
    assert result['keep_active_point_indices'].tolist() == [0]
    assert result['excluded_reason_mask'].tolist() == [0, 16, 1, 1]
    assert result['old_active_point_to_strict_point'].tolist() == [0, -1, -1, -1]


def test_reflection_or_unconfirmed_cannot_be_admitted(tmp_path):
    packet = review_packet(tmp_path)
    for field, value in [('scope', 'reflection'), ('static_material_confirmed', False),
                         ('pixel_location_confirmed', False),
                         ('original_source_image_actually_reviewed', False),
                         ('model_projection_or_fit_residual_used_for_identity', True)]:
        invalid = deepcopy(packet)
        invalid['reviewed_endpoints'][0][field] = value
        with pytest.raises(ValueError):
            ExactSourceScopeReview(invalid)


def test_same_line_identity_alone_does_not_prove_an_lk_correspondence(tmp_path):
    packet = review_packet(tmp_path)
    packet['reviewed_endpoints'][1].pop('physical_material_point_id')
    review = ExactSourceScopeReview(packet)
    result = partition_motion_inputs(np.array([[20, 30.]]), np.array([[21, 31.]]),
        np.array([10]), np.array([20]), review)
    assert result['keep_active_point_indices'].size == 0
    assert result['excluded_reason_mask'].tolist() == [16]


def test_excluded_points_physically_leave_the_actual_loaded_arrays(tmp_path):
    packet = review_packet(tmp_path)
    path = tmp_path / 'review.json'
    path.write_text(json.dumps(packet), encoding='utf8')
    review = ExactSourceScopeReview.load(path)
    historical = tmp_path / 'historical.npz'
    a = np.array([[20, 30], [40, 50], [60, 70.]])
    b = np.array([[21, 31], [41, 51], [61, 71.]])
    sigma = np.array([[3, 4], [5, 6], [7, 8.]])
    np.savez(historical, a=a, b=b, sigma=sigma)
    before = file_sha(historical)
    output = tmp_path / 'strict.npz'
    ref = write_motion_partition(output, block='test', a=a, b=b,
        ia=np.zeros(3, int), ib=np.ones(3, int), stamps=np.array([10, 20]),
        sigma=sigma, active_to_original=np.array([100, 105, 111]), review=review,
        original_references=[dict(path=str(historical), sha256=before)])
    values, metadata = load_motion_partition(output, expected_sha256=ref['sha256'],
        source_sha256=SOURCE['video_sha256'], time_base=SOURCE['time_base'])
    assert metadata['strict_active_point_count'] == 1
    assert values['a'].tolist() == [[20, 30]]
    assert values['b'].tolist() == [[21, 31]]
    assert values['motion_sigma'].tolist() == [[3, 4]]
    assert values['active_to_original'].tolist() == [100]
    assert values['excluded_active_point_indices'].tolist() == [1, 2]
    assert file_sha(historical) == before


def test_empty_review_excludes_all_original_finite_or_unknown_points(tmp_path):
    review = ExactSourceScopeReview(dict(schema=REVIEW_SCHEMA, source=SOURCE, reviewed_endpoints=[]))
    result = partition_motion_inputs(np.array([[10, 20.]]), np.array([[11, 21.]]),
        np.array([10]), np.array([20]), review)
    assert result['keep_active_point_indices'].size == 0
    assert result['excluded_active_point_indices'].tolist() == [0]


def test_original_video_frame_measurement_does_not_claim_manual_frame_review(tmp_path):
    packet = review_packet(tmp_path)
    video = tmp_path / 'original_video.bin'
    video.write_bytes(b'source video fixture')
    packet['source']['video_sha256'] = file_sha(video)
    for endpoint in packet['reviewed_endpoints']:
        endpoint.update(review_authority='source_measurement_with_reviewed_identity',
            original_source_image_actually_reviewed=False,
            source_feature_physical_identity_confirmed=True, actual_source_RGB_measured=True)
        endpoint['source_image_evidence'] = dict(kind='original_video_exact_native_frame',
            path=str(video), sha256=file_sha(video), native_pts=endpoint['native_pts'],
            decoded_BGR_sha256=endpoint['decoded_BGR_sha256'])
    review = ExactSourceScopeReview(packet)
    assert review.endpoint(10, [20, 30])['original_source_image_actually_reviewed'] is False
    video.write_bytes(b'changed source video fixture')
    with pytest.raises(ValueError, match='changed during source review'):
        review.verify_evidence_unchanged()
