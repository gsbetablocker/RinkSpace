from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'events/ankara_2026/programmes/okada_mei_sp'))
from apply_native_multicamera_review import revise_observations


def source_case(tmp_path):
    evidence = tmp_path / 'source.png'
    evidence.write_bytes(b'source evidence fixture')
    z = dict(pts=np.array([100, 100, 200, 300]), observation_ids=np.array(['a', 'b', 'c', 'd']),
             stations=np.array([.1, .2, .3, .4]), source_rows=np.array([4, 5, 6, 7]),
             kind=np.ones(4, dtype=int), uv=np.array([[10., 20.], [30., 40.], [50., 60.], [70., 80.]]),
             path_indices=np.array([0, 0, 0, 1], dtype=np.int32))
    shots = [dict(id='first', first_pts=100, last_pts=200, row_begin=0, row_stop=3, camera_index=0),
             dict(id='second', first_pts=300, last_pts=300, row_begin=3, row_stop=4, camera_index=1)]
    base = dict(reason='Visible material boundary in source exposure', evidence_image=str(evidence))
    return z, dict(shots=shots), base


def test_edits_and_insertions_preserve_source_identity_and_numeric_stations(tmp_path):
    z, manifest, base = source_case(tmp_path)
    before = {k: a.copy() for k, a in z.items()}
    report = dict(decisions=[
        dict(base, observation_id='b', pts=100, uv=[30., 40.], expected_path_id='arc', decision='correct_uv', new_uv=[32., 41.]),
        dict(base, observation_id='c', pts=200, uv=[50., 60.], expected_path_id='arc', decision='exclude_preserving_source'),
        dict(base, observation_id='d', pts=300, uv=[70., 80.], expected_path_id='line', decision='rebind_preserving_uv', new_path_id='arc'),
    ], added_observations=[dict(base, observation_id='new-long-id', pts=100, uv=[90., 100.], path_id='line')])
    new, stations, shots, touched, records, additions, origins = revise_observations(
        z, np.array([.11, .22, .33, .44]), manifest, ['arc', 'line'], report)
    assert new['observation_ids'].tolist() == ['a', 'b', 'new-long-id', 'd']
    assert new['source_rows'].tolist() == [4, 5, 8, 7]
    assert new['path_indices'].tolist() == [0, 0, 1, 0]
    assert new['uv'].tolist() == [[10., 20.], [32., 41.], [90., 100.], [70., 80.]]
    assert stations.tolist() == [.11, .22, .5, .44]
    assert origins.tolist() == [0, 1, -1, 3]
    assert touched.tolist() == [False, True, True, True]
    assert [(s['row_begin'], s['row_stop'], s['camera_index']) for s in shots] == [(0, 3, 0), (3, 4, 1)]
    for key in z:
        np.testing.assert_array_equal(z[key], before[key])
    assert manifest['shots'][0]['row_stop'] == 3
    assert records[0]['uv'] == [30., 40.] and additions[0]['source_row'] == 8
    assert set(new) == set(z)


@pytest.mark.parametrize('change', ['wrong_original_pixel', 'duplicate_id', 'missing_frame', 'duplicate_pixel', 'nonfinite_pixel'])
def test_invalid_source_edits_fail_before_input_mutation(tmp_path, change):
    z, manifest, base = source_case(tmp_path)
    record = dict(base, observation_id='a', pts=100, uv=[10., 20.], expected_path_id='arc', decision='correct_uv', new_uv=[12., 21.])
    added = dict(base, observation_id='new', pts=200, uv=[90., 100.], path_id='line')
    if change == 'wrong_original_pixel': record['uv'] = [11., 20.]
    if change == 'duplicate_id': added['observation_id'] = 'd'
    if change == 'missing_frame': added['pts'] = 150
    if change == 'duplicate_pixel': added.update(pts=300, uv=[70., 80.])
    if change == 'nonfinite_pixel': record['new_uv'] = [float('nan'), 21.]
    with pytest.raises((ValueError, AssertionError)):
        revise_observations(z, z['stations'], manifest, ['arc', 'line'], dict(decisions=[record], added_observations=[added]))
    np.testing.assert_array_equal(z['uv'][0], [10., 20.])
