"""Input-contract tests for source identity, full W and true query transport."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rinkfit_start.semantic_snapshot import SnapshotArrays, stream_snapshot_frames
from rinkfit_start.semantics import profile_semantics, validate_group_membership
from rinkfit_start.semantic_order import audit_source_runs
from rinkfit_start.semantic_query import certified_polyline, evaluate_curve, rational_bound
from rinkfit_start.semantic_capture import monotone_grid_dp
from numpy.polynomial import Polynomial as Poly
from itertools import product


def test_whole_correlated_groups_are_not_sliced():
    operator = {'source_point_ids': ['a', 'b'], 'whitening_matrix': [[1., 2., 3., 4.]]}
    rows, matrix = validate_group_membership([0, 1], ['a', 'b'], [10, 11], operator)
    np.testing.assert_array_equal(rows, [10, 11])
    assert matrix.tobytes() == np.asarray([[1., 2., 3., 4.]], dtype='<f8').tobytes()
    assert validate_group_membership([0, 1], ['a', 'b'], [-1, -1], operator)[0] is None
    with pytest.raises(ValueError, match='cuts an original correlated'):
        validate_group_membership([0, 1], ['a', 'b'], [10, -1], operator)
    with pytest.raises(ValueError, match='original IDs'):
        validate_group_membership([0, 1], ['b', 'a'], [10, 11], operator)


def test_snapshot_stream_chunk_boundary_and_exact_array_bytes(tmp_path):
    frames = [{'index': i, 'native_pts': 256*i, 'rows': [], 'whitening_groups': [],
               'payload_padding': 'x'*270000} for i in (1, 2)]
    path = tmp_path/'source.json'
    path.write_text(json.dumps({'schema': 'rinkspace.complete-authenticated-native-source-snapshot.v1',
        'payload': {'frames': frames}}, indent=2), encoding='utf-8')
    assert [f['index'] for f in stream_snapshot_frames(path)] == [1, 2]
    archive = tmp_path/'arrays.npz'
    a = np.asarray([[1., -0.], [np.nextafter(1., 2.), 3.]], dtype='<f8')
    np.savez_compressed(archive, original=a)
    with SnapshotArrays(archive) as saved:
        restored = saved.decode({'_safe_npz_array': 'original'})
        assert restored.dtype == a.dtype and restored.tobytes() == a.tobytes()
        assert not restored.flags.writeable


def test_true_profile_keeps_query_separate_from_measured_uv(tmp_path):
    uv = [[12., 20.]]; query = [[10., 20.]]; normal = [[1., 0.]]
    raw = {'features': [{'id': 'ice.C:segment0', 'source_line_fit': {'direct_readouts': [
        {'uv': uv[0], 'sigma_px': .7, 'source_evidence': {'query_index': 9,
            'query_uv': query[0], 'profile_normal': normal[0],
            'complete_band_native_interval_px': [-3., 5.]}}]}}]}
    path = tmp_path/'native.json'; path.write_text(json.dumps(raw), encoding='utf-8')
    ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    row = {'source_feature_id': 'ice.C:segment0', 'source_point_ids': ['256/ice.C:segment0/query/9'],
        'uv': np.asarray(uv), 'native_line_queries': {'query_uv': query, 'profile_normal': normal}}
    q, n, interval, provenance = profile_semantics(row, {'ice_source_readout_packet': ref})
    np.testing.assert_array_equal(q, query)
    np.testing.assert_array_equal(n, normal)
    np.testing.assert_array_equal(interval, [[-3., 5.]])
    assert not np.array_equal(q, row['uv'])
    assert provenance[0]['query_index'] == 9
    del raw['features'][0]['source_line_fit']['direct_readouts'][0]['source_evidence']['complete_band_native_interval_px']
    path.write_text(json.dumps(raw), encoding='utf-8'); ref['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(KeyError):
        profile_semantics(row, {'ice_source_readout_packet': ref})


def _order_fixture(tmp_path, paths, stations, branches):
    source = tmp_path/'source_runs.jsonl'
    source.write_text(json.dumps({'source_order_verified': True, 'frame': 1,
        'source_feature_id': 'fixture', 'source_segment_index': 0,
        'source_rows': list(range(len(stations))), 'variant_index': 0,
        'source_closed': False, 'world_direction': None})+'\n', encoding='utf-8')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'observations': len(stations), 'variant_ids': ['w0'],
        'assets': {'source_runs.jsonl': {'file': source.name,
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}}), encoding='utf-8')
    model = {'blocks': [{'values': [.5, .1, .1, 1.9]}],
        'materials': [{'label': 'fixture', 'paths': [dict(p, variant_id='w0') for p in paths]}]}
    return audit_source_runs(manifest, model, np.asarray(stations), np.asarray(branches, dtype=int))


def test_order_continuous_board_segments_share_real_arclength(tmp_path):
    result = _order_fixture(tmp_path, [
        {'id': 'left', 'kind': 3, 'boundary_segment': 6, 'closed': False},
        {'id': 'corner', 'kind': 3, 'boundary_segment': 7, 'closed': False}], [.9, .1], [0, 1])
    assert result['passed']
    assert result['records'][0]['legal_direction_candidates'] == [1]
    assert result['records'][0]['board_arclength_L'][1] > result['records'][0]['board_arclength_L'][0]


def test_order_disconnected_print_does_not_gain_connector(tmp_path):
    result = _order_fixture(tmp_path, [
        {'id': 'letter_I', 'kind': 2, 'closed': True},
        {'id': 'letter_S', 'kind': 2, 'closed': True}], [.9, .1], [0, 1])
    assert not result['passed']
    assert result['failure_reasons'] == {'run_crosses_disconnected_material_paths': 1}


def test_order_closed_seam_needs_no_invented_full_circle_coverage(tmp_path):
    result = _order_fixture(tmp_path, [{'id': 'circle', 'kind': 1, 'closed': True}],
        [.9, .05, .2], [0, 0, 0])
    assert result['passed']
    assert result['records'][0]['legal_direction_candidates'] == [1]
    assert not result['minimum_world_circle_coverage_imposed']


def test_query_chord_certificate_covers_actual_rational_curve():
    U, V, depth = Poly([10., 5., .1]), Poly([20., -2., .2, .001]), Poly([1., .002])
    D = depth**3
    alpha, uv, chords = certified_polyline(U, V, D, depth, -4., 8., tolerance_px=.001)
    assert alpha[0] == -4. and alpha[-1] == 8.
    for first, last, bound in chords:
        values = np.linspace(first, last, 101)
        exact = evaluate_curve(U, V, D, values)
        ends = evaluate_curve(U, V, D, [first, last])
        chord = ends[0]+((values-first)/(last-first))[:, None]*(ends[1]-ends[0])
        assert np.max(np.linalg.norm(exact-chord, axis=1)) <= bound <= .001


@pytest.mark.parametrize('closed', [False, True])
@pytest.mark.parametrize('direction', [1, -1])
def test_capture_dp_equals_exhaustive_finite_grid(closed, direction):
    costs = np.array([[8., 3., 1., 9.], [1., 7., 2., 6.], [4., 0., 5., 3.]])
    chosen, actual = monotone_grid_dp(costs, closed=closed, direction=direction)
    feasible = []
    for indices in product(range(4), repeat=3):
        difference = direction*np.diff(indices)
        if closed:
            valid = np.sum(np.where(difference < 0, difference+4, difference)) < 4
        else:
            valid = np.all(difference >= 0)
        if valid:
            feasible.append(sum(costs[row, index] for row, index in enumerate(indices)))
    assert actual == min(feasible)
    assert actual == sum(costs[row, index] for row, index in enumerate(chosen))


def test_all_run_capture_keeps_exact_legal_continuous_seed_and_empty_change_ledger(tmp_path, monkeypatch):
    from rinkfit_start import semantic_capture as capture

    class Scene:
        def __init__(self, model):
            pass

        def world(self, material, branch, station):
            s=np.asarray(station)
            return np.stack([s,np.zeros_like(s),np.zeros_like(s)],axis=-1)

        def world_rows(self, materials, branches, stations):
            return self.world(0,0,stations),np.ones(len(stations),bool),np.full(len(stations),'')

    class Camera:
        tau=0.

        def at(self,world,time):
            shape=np.broadcast_shapes(world.shape[:-1],np.asarray(time).shape)
            return {'uv':np.broadcast_to(world[...,:2],shape+(2,)).copy(),'valid':np.ones(shape,bool)}

    monkeypatch.setattr(capture,'NativeScene',Scene)
    monkeypatch.setattr(capture,'camera_from_event',lambda model:Camera())
    monkeypatch.setattr(capture,'_board_lengths',lambda model:np.ones(8))
    audit={'failed_runs':0,'failed_run_ids':[],'passed':True,'records':[{
        'run_id':'run/0','valid':True,'failure_reason':None,
        'allowed_direction_candidates':[1,-1],'legal_direction_candidates':[1]}]}
    monkeypatch.setattr(capture,'audit_source_runs',lambda *args,**kwargs:audit)
    packet=tmp_path/'backend_input';packet.mkdir()
    semantic=tmp_path/'semantic_packet';semantic.mkdir()
    seed=tmp_path/'seed';seed.mkdir();out=tmp_path/'candidate'
    original=np.array([.23,.77],dtype='<f8');original.tofile(seed/'station.f64')
    np.zeros(2,dtype='<u4').tofile(seed/'path.u32')
    np.c_[original,np.zeros(2)].astype('<f8').tofile(packet/'uv.f64')
    np.full(2,256256,dtype='<i8').tofile(packet/'pts.i64')
    np.zeros(2,dtype='<u4').tofile(packet/'material.u32')
    np.zeros(2,dtype='<u4').tofile(semantic/'variant_index.u32')
    (semantic/'manifest.json').write_text(json.dumps({'observations':2,'variant_ids':['w0']}))
    (semantic/'source_runs.jsonl').write_text(json.dumps({'source_rows':[0,1],
        'variant_index':0,'frame':1,'source_feature_id':'fixture','world_direction':None})+'\n')
    model=tmp_path/'model.json';model.write_text(json.dumps({'materials':[{'paths':[
        {'kind':2,'id':'line','points':[[0.,0.],[1.,0.]],'closed':False,'variant_id':'w0'}]}]}))
    result=capture.capture_source_runs(tmp_path,model,seed,out,capture_all_runs=True)
    assert result['candidate_compared_runs']==1
    assert result['existing_legal_seed_selected_runs']==1
    assert result['changed_source_rows']==0
    assert (out/'station.f64').read_bytes()==(seed/'station.f64').read_bytes()
    assert result['capture_error_after_px']=={'median':None,'max':None}
