import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.first_shot_fit import FrozenWhitening
from calibration.program_tracking.first_shot_group_quarantine import quarantine_complete_groups


def fixture_frame():
    rows = [dict(uv=np.array([[i+.2, i+.7]]), sigma=np.array([1.3+i]),
                 target=dict(kind='ice_curve', curve_id=str(i)), source_point_ids=[str(i)]) for i in range(5)]
    groups = [dict(point_indices=[0, 2], operator=SimpleNamespace(matrix=np.array([
        [1., .2, -.4, .3], [.1, 1.1, .3, -.2], [.2, -.1, .7, .6]]))),
        dict(point_indices=[1, 3], operator=SimpleNamespace(matrix=np.array([[.5, .1, .2, .8], [.1, .9, -.3, .2]])))]
    return dict(index=7, native_pts=8128, rows=rows, source_point_count=5, whitening_groups=groups)


def specification(frame):
    matrix = frame['whitening_groups'][1]['operator'].matrix
    return dict(native_pts=8128, source_point_ids=['1', '3'],
                operator_matrix_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
                operator_matrix_shape=list(matrix.shape), material_identity_confirmed=False,
                reason='Original RGB does not establish the named material layer',
                evidence_references=[dict(path='review.json', sha256='example')])


def test_actual_correlated_residual_and_J_are_exact_retained_rows():
    frame = fixture_frame()
    frames, proof = quarantine_complete_groups([frame], [specification(frame)], evidence_reference={'path': 'manifest'})
    changed = frames[0]
    rng = np.random.default_rng(7)
    for raw in [rng.normal(size=10), rng.normal(size=(10, 11))]:
        expected = FrozenWhitening(frame).apply(raw)[proof['retained_formal_indices']]
        actual = FrozenWhitening(changed).apply(raw[[0, 1, 4, 5, 8, 9]])
        assert np.array_equal(actual, expected)
    assert changed['rows'][0] is frame['rows'][0]
    assert changed['rows'][1] is frame['rows'][2]
    assert changed['whitening_groups'][0]['operator'] is frame['whitening_groups'][0]['operator']
    assert changed['inactive_source_groups'][0]['original_complete_group'] is frame['whitening_groups'][1]
    assert frame['source_point_count'] == 5
    assert proof['frames'][0]['removed_formal_rows'] == 2


def test_partial_group_or_changed_original_operator_is_rejected():
    frame = fixture_frame()
    spec = specification(frame)
    spec['source_point_ids'] = ['1']
    with pytest.raises(ValueError, match='absent'):
        quarantine_complete_groups([frame], [spec], evidence_reference={})
    spec = specification(frame)
    spec['operator_matrix_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='original complete W'):
        quarantine_complete_groups([frame], [spec], evidence_reference={})


def test_does_not_split_a_native_row_or_accept_missing_evidence():
    frame = fixture_frame()
    row = dict(frame['rows'][0], uv=np.array([[.2, .7], [1.2, 1.7]]),
               sigma=np.array([1.3, 2.3]), source_point_ids=['0', '1'])
    frame['rows'] = [row, *frame['rows'][2:]]
    with pytest.raises(ValueError, match='split a source row'):
        quarantine_complete_groups([frame], [specification(frame)], evidence_reference={})
    spec = specification(frame)
    spec['evidence_references'] = []
    with pytest.raises(ValueError, match='reviewed uncertain'):
        quarantine_complete_groups([frame], [spec], evidence_reference={})
