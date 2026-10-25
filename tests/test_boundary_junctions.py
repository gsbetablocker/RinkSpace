"""Fixed ice/board relations: real native supports, never screen crossings."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from rinkspace.core.native import NativeCore, NativeError
from rinkspace.domain.contracts import Feature, ContractError, validate_dataset
from rinkspace.domain.junctions import evaluate_ice_board_junction
from rinkspace.domain.entity_graph import fixed_node_graph
from rinkspace.session.dataset import read_json, write_json, write_array, load_prepared
from rinkspace.session.prepare import prepare, validate_current_feature_contracts
import test_prepare_source_binding as source_fixture

BOARD = [.5, .25, .1, .1, 2., .02]
DOMAIN = {'kind': 'unbounded_periodic', 'hard_segment_bounds': False}


@pytest.fixture
def core():
    library = os.environ.get('RINKSPACE_CORE_LIBRARY')
    if not library:
        pytest.skip('explicit native library required')
    return NativeCore(library)


def fixture():
    scene = {'board': BOARD, 'paths': [
        dict(id='line_path', label='ice.C', kind='ice_segment', layer='ice',
             measurement='centreline', parameters=[0., .25, 0., 0., -.25, 0.],
             finite_support=[0., 1.]),
        dict(id='edge_path', label='board.bottom', kind='board_course', layer='board',
             native_kind=7, measurement='centreline', parameters=[0.],
             station_domain=deepcopy(DOMAIN), finite_support=[0., 0.])
    ]}
    records = [
        dict(id='junction', label='ice.C_meets_board.bottom', kind='fixed_junction', surface='ice', identity='confirmed',
             measurement='material_point', parents=['line', 'edge'], branch='0',
             junction_relation='ice_board_bottom', has_station_parameter=False),
        dict(id='line', kind='finite_curve', surface='ice', identity='confirmed',
             measurement='centreline', path_id='line_path', label='ice.C'),
        dict(id='edge', kind='finite_curve', surface='board', identity='confirmed',
             measurement='centreline', path_id='edge_path', label='board.bottom',
             native_kind=7, parameters=[0.], station_domain=deepcopy(DOMAIN),
             finite_support=[0., 0.], has_station_parameter=True)
    ]
    arrays = dict(pts=np.array([100], dtype='<i8'), feature_index=np.array([0], dtype='<i4'),
                  active=np.ones(1, dtype='u1'), kind=np.zeros(1, dtype='u1'),
                  local_index=np.full(1, -1, dtype='<i8'), support=np.zeros((1, 2)))
    return scene, records, arrays


def typed():
    return {
        'line': Feature('line', 'finite_curve', 'ice', 'confirmed', 'centreline'),
        'edge': Feature('edge', 'finite_curve', 'board', 'confirmed', 'centreline', label='board.bottom'),
        'junction': Feature('junction', 'fixed_junction', 'ice', 'confirmed', 'material_point',
                           ('line', 'edge'), '0', junction_relation='ice_board_bottom')}


def test_explicit_relation_preserves_default_surface_checks_and_fixed_type():
    features = typed()
    assert validate_dataset({}, features, ())['features'] == 3
    for changed in [replace(features['junction'], junction_relation=None),
                    replace(features['edge'], label='board.top'),
                    replace(features['edge'], surface='ice')]:
        invalid = {**features, changed.feature_id: changed}
        with pytest.raises(ContractError):
            validate_dataset({}, invalid, ())
    with pytest.raises(ContractError):
        replace(features['junction'], has_station_parameter=True)
    with pytest.raises(ContractError):
        replace(features['junction'], branch='nearest_pixel')
    with pytest.raises(ContractError):
        replace(features['line'], junction_relation='ice_board_bottom')
    with pytest.raises(ContractError):
        replace(features['edge'], surface='unknown_depth')
    # Existing ordinary same-surface fixed junctions remain valid.
    ordinary = {**features, 'edge': replace(features['edge'], surface='ice'),
                'junction': replace(features['junction'], junction_relation=None)}
    validate_dataset({}, ordinary, ())


def test_native_curved_boundary_roots_are_real_and_branch_order_is_stable(core):
    line = [-.7, .2, 0., .7, .2, 0.]
    x = .4 + math.sqrt(.1**2 - (.2-.15)**2)
    for branch, expected_x in enumerate((-x, x)):
        point = core.ice_board_junction(BOARD, line, 0., branch)
        np.testing.assert_allclose(point, [expected_x, .2, 0.], atol=2e-14, rtol=0.)
        reverse = core.ice_board_junction(BOARD, line[3:]+line[:3], 0., 1-branch)
        np.testing.assert_allclose(reverse, point, atol=2e-14, rtol=0.)


def test_native_does_not_replace_an_unsupported_branch_with_the_visible_one(core):
    line = [0., 0., 0., 0., .3, 0.]
    with pytest.raises(NativeError) as rejected:
        core.ice_board_junction(BOARD, line, 0., 0)
    assert rejected.value.status == 3
    np.testing.assert_allclose(core.ice_board_junction(BOARD, line, 0., 1), [0., .25, 0.])


@pytest.mark.parametrize('line,height,branch,status', [
    ([0., -.3, 0., 0., .3, 0.], 1., 0, 7),
    ([0., -.3, 0., 0., .3, 0.], .5, 0, 7),
    ([0., -.3, .01, 0., .3, .01], 0., 0, 7),
    ([0., -.3, 0., 0., .3, .01], 0., 0, 7),
    ([0., -.3, float('nan'), 0., .3, 0.], 0., 0, 1),
    ([0., -.3, 0., 0., .3, 0.], 0., 2, 1),
    ([0., -.3, 0., 0., .3, 0.], 0., -1, 1),
    ([0., -.2, 0., 0., .2, 0.], 0., 0, 3),
    ([.6, -.3, 0., .6, .3, 0.], 0., 0, 4),
    ([-.2, .25, 0., .2, .25, 0.], 0., 0, 2),
    ([0., 0., 0., 0., 0., 0.], 0., 0, 2),
])
def test_native_rejects_top_depth_extension_tangent_and_invalid_branch(core, line, height, branch, status):
    with pytest.raises(NativeError) as rejected:
        core.ice_board_junction(BOARD, line, height, branch)
    assert rejected.value.status == status


@pytest.mark.parametrize('active', [0, 1])
@pytest.mark.parametrize('damage', ['top', 'depth', 'surface', 'untyped', 'branch', 'support',
                                  'station_flag', 'station_row', 'station_support', 'missing_core'])
def test_generic_admission_rejects_counterexamples_even_when_inactive(core, active, damage):
    scene, records, arrays = fixture()
    arrays['active'][:] = active
    assert validate_current_feature_contracts({}, scene, records, arrays, np.array([0]), core=core)['junctions']['ice_board_bottom'] == 1
    if damage == 'top':
        scene['paths'][1]['parameters'] = [1.]  # Label alone cannot authorize it.
    elif damage == 'depth':
        scene['paths'][0]['parameters'][2] = .01
    elif damage == 'surface':
        scene['paths'][0]['layer'] = 'board'
    elif damage == 'untyped':
        records[0].pop('junction_relation')
    elif damage == 'branch':
        records[0]['branch'] = '2'
    elif damage == 'support':
        scene['paths'][0]['finite_support'] = [.1, .9]
    elif damage == 'station_flag':
        records[0]['has_station_parameter'] = True
    elif damage == 'station_row':
        arrays['kind'][:] = 1
        arrays['local_index'][:] = 0
    elif damage == 'station_support':
        arrays['support'][:] = [0., 1.]
    else:
        core = None
    with pytest.raises(ValueError):
        validate_current_feature_contracts({}, scene, records, arrays, np.array([0]), core=core)


def test_all_current_model_boundary_nodes_bind_the_same_native_feature(core):
    root = Path(__file__).resolve().parents[1]
    scene = read_json(root/'events/ankara_2026/scene/field/model.json')
    nodes = [n for n in fixed_node_graph(scene, core)['nodes']
             if n.get('junction_relation') == 'ice_board_bottom']
    assert len(nodes) == 10
    for node in nodes:
        features = typed()
        features['junction'] = replace(features['junction'], branch=str(node['branch']))
        actual = evaluate_ice_board_junction(features['junction'], features, scene, core,
                    path_ids=dict(zip(('line', 'edge'), node['parents'])))
        np.testing.assert_allclose(actual, node['xyz'], atol=3e-13, rtol=0.)
        assert actual[2] == 0.


def test_prepare_and_fit_reload_preserve_fixed_rows_and_require_native_validation(core, tmp_path):
    root, output = tmp_path/'input', tmp_path/'prepared'
    expected = source_fixture.PrepareSourceBinding().make_input(root, [dict(pts=100), dict(pts=150)])
    scene, records, _ = fixture()
    expected['kind'][:] = 0
    expected['local_index'][:] = -1
    expected['support'][:] = 0
    write_json(root/'scene.json', scene)
    write_json(root/'features.json', {'features': records})
    write_json(root/'obs.json', {'arrays': {k: write_array(root, k, v) for k, v in expected.items()}})
    with pytest.raises(ContractError, match='native core'):
        prepare(root/'input.json', root/'scope.json', 's01', output)
    assert not output.exists()
    prepare(root/'input.json', root/'scope.json', 's01', output, core=core)
    manifest, actual = load_prepared(output, core=core)
    assert manifest['features'] == records
    assert manifest['current_contracts']['junctions']['ice_board_bottom'] == 1
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)
    with pytest.raises(ContractError, match='native core'):
        load_prepared(output)
    current = read_json(output/'scene.json')
    current['paths'][1]['parameters'] = [1.]
    write_json(output/'scene.json', current)
    with pytest.raises(NativeError) as rejected:
        load_prepared(output, core=core)
    assert rejected.value.status == 7
