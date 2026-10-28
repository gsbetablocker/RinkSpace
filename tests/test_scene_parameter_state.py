"""Current print geometry consumes one explicit state; no fit is run here.

Real current assets and accepted source UV exercise preparation and review.
Finite differences check geometry derivatives, never claimed video accuracy.
"""
import hashlib
import json
import os

import numpy as np
import pytest

from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import (
    ParameterState, SceneParameterModel, compile_scene_parameters,
    load_parameter_state, write_parameter_state,
)
from rinkspace.domain.print_assets import resolve_instance
from rinkspace.domain.scene_assets import read_scene_assets, snapshot_scene
from rinkspace.review.render import load_state
from rinkspace.review.scene import common_print_stations, sample_review_scene
from rinkspace.session.dataset import load_prepared, read_json
from rinkspace.session.prepare import prepare
from test_print_measurement import FIELD, INNER, OUTER, make_print_input


@pytest.fixture
def model(tmp_path):
    scene, _ = snapshot_scene(FIELD, tmp_path/'scene')
    return SceneParameterModel(scene, tmp_path/'scene/scene.json')


@pytest.fixture
def core():
    library = os.environ.get('RINKSPACE_CORE_LIBRARY')
    if not library:
        pytest.skip('explicit RINKSPACE_CORE_LIBRARY required')
    return NativeCore(library)


def changed(state, identity, column, delta=.13):
    values = state[identity].copy()
    values[column] += delta
    return state.with_blocks({identity: values})


def shape_block(model, canonical_path):
    return next(identity for identity in model.dependencies(canonical_path)
                if model.initial_state.blocks[identity]['kind'] == 'print_shape')


def rehash(layout):
    content = {k: v for k, v in layout.items() if k != 'layout_revision'}
    layout['layout_revision'] = hashlib.sha256(json.dumps(content, sort_keys=True,
        ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    return layout


def test_every_current_template_instance_and_material_path_has_shared_state_dependencies(model):
    state = model.initial_state
    assets = read_scene_assets(model.scene, model.scene_path)
    assert len({t['id'] for _, a in assets for t in a['templates']}) == 13
    assert len({i for ref, _ in assets for i in ref['instance_ids']}) == 26
    expected_paths = set()
    for ref, asset in assets:
        templates = {t['id']: t for t in asset['templates']}
        for identity in ref['instance_ids']:
            instance, template, size, placement = resolve_instance(asset, identity)
            assert template is templates[instance['template_id']]
            assert set(template['internal_geometry']['path_bindings']) == {p['id'] for p in template['paths']}
            for path in template['paths']:
                canonical = identity+'::'+path['id']
                expected_paths.add(canonical)
                dependencies = model.dependencies(canonical)
                assert dependencies[:2] == [placement['id'], size['id']]
                assert all(dependency in state for dependency in dependencies)
                binding = template['internal_geometry']['path_bindings'][path['id']]
                assert len(dependencies) == (2 if binding['kind'] == 'panel_chart_gauge' else 3)
        first, second = ref['instance_ids']
        for path in templates[asset['instances'][0]['template_id']]['paths']:
            a, b = (model.dependencies(i+'::'+path['id']) for i in (first, second))
            assert a[0] != b[0]
            assert a[1:] == b[1:]
    assert set(model.paths) == expected_paths
    offsets = []
    for block in state.blocks.values():
        if block['available']:
            offsets.extend(range(block['offset'], block['offset']+block['size']))
        else:
            assert state[block['id']] is None
            assert block['offset'] is None and not block['active']
            assert not len(state.active_columns([block['id']]))
    assert offsets == list(range(len(state.values)))
    repeated = compile_scene_parameters(model.scene, model.scene_path)
    assert repeated.layout == state.layout
    np.testing.assert_array_equal(repeated.values, state.values)


def test_state_is_readonly_and_layout_mutation_cannot_change_its_registry(model):
    state = model.initial_state
    assert state.values.dtype == np.dtype('<f8') and state.values.flags.c_contiguous
    with pytest.raises(ValueError):
        state.values[0] += 1
    identity = next(i for i in state if state[i] is not None)
    with pytest.raises(ValueError):
        state[identity][0] += 1
    with pytest.raises(TypeError):
        state.blocks[identity]['offset'] = 0
    layout = state.layout
    layout['blocks'][0]['offset'] = -1
    assert state.layout['blocks'][0]['offset'] >= 0
    with pytest.raises(ValueError, match='Unknown'):
        state.active_columns(['not/a/parameter'])


@pytest.mark.parametrize('damage', ['float32', 'nan', 'wrong_size', 'hash', 'overlap', 'unregistered'])
def test_state_rejects_wrong_type_values_and_corrupt_layout(model, damage):
    state = model.initial_state
    values, layout = state.values.copy(), state.layout
    if damage == 'float32':
        values = values.astype('<f4')
    elif damage == 'nan':
        values[0] = np.nan
    elif damage == 'wrong_size':
        values = values[:-1]
    elif damage == 'hash':
        layout['scope'] = 'changed without updating identity'
    elif damage == 'overlap':
        available = [b for b in layout['blocks'] if b['available']]
        available[1]['offset'] = available[0]['offset']
        rehash(layout)
    else:
        layout['size'] += 1
        values = np.append(values, 0.).astype('<f8')
        rehash(layout)
    with pytest.raises(ValueError):
        ParameterState(layout, values)


def test_frozen_board_gauge_and_unlocalized_values_cannot_be_invented(model):
    state = model.initial_state
    frozen = next(i for i, b in state.blocks.items() if b['kind'] == 'board_shape')
    with pytest.raises(ValueError, match='frozen'):
        changed(state, frozen, 0)
    gauge = next((i, b) for i, b in state.blocks.items()
                 if b['kind'] == 'print_shape' and b.get('gauge_fixed_local_indices'))
    with pytest.raises(ValueError, match='frozen'):
        changed(state, gauge[0], gauge[1]['gauge_fixed_local_indices'][0])
    for instance in ('print_instance/Ad15', 'print_instance/Ad16'):
        path = next(p for p in model.paths if p.startswith(instance+'::'))
        placement = model.dependencies(path)[0]
        assert state[placement] is None
        with pytest.raises(ValueError, match='unlocalized'):
            state.with_blocks({placement: np.zeros(3)})
    with pytest.raises(ValueError, match='Unknown'):
        state.with_blocks({'unknown': np.zeros(1)})
    editable = next(i for i, b in state.blocks.items() if b['active'])
    with pytest.raises(ValueError, match='wrong size'):
        state.with_blocks({editable: np.zeros(1)})


def test_candidate_write_load_preserves_initial_values_and_checks_array_hash(model, tmp_path):
    state = model.initial_state
    identity = shape_block(model, 'print_instance/Ad2::'+OUTER)
    candidate = changed(state, identity, 0)
    root = model.scene_path.parent
    first = write_parameter_state(root, state, name='initial.json')
    initial_bytes = (root/first['values_ref']['path']).read_bytes()
    second = write_parameter_state(root, candidate, name='candidate.json')
    assert first['state_id'] != second['state_id']
    assert first['values_ref']['path'] != second['values_ref']['path']
    assert first['values_ref']['sha256'] != second['values_ref']['sha256']
    loaded = load_parameter_state(root/'candidate.json', expected_scene=model.scene,
                                  expected_scene_path=model.scene_path)
    np.testing.assert_array_equal(loaded.values, candidate.values)
    np.testing.assert_array_equal(load_parameter_state(root/'initial.json').values, state.values)
    assert (root/first['values_ref']['path']).read_bytes() == initial_bytes
    assert second['fit_executed'] is False
    with pytest.raises(FileExistsError):
        write_parameter_state(root, candidate, name='initial.json')
    array_path = root/second['values_ref']['path']
    raw = bytearray(array_path.read_bytes())
    raw[-1] ^= 1
    array_path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_parameter_state(root/'candidate.json')


def test_prepared_state_is_self_contained_and_preserves_exact_accepted_pixels_time_metadata(tmp_path):
    root, out = tmp_path/'input', tmp_path/'prepared'
    expected, features, members, weights, metadata = make_print_input(root)
    prepare(root/'input.json', root/'scope.json', 's01', out)
    root.rename(tmp_path/'detached_input')
    manifest, arrays = load_prepared(out, purpose='fit')
    for key, value in expected.items():
        np.testing.assert_array_equal(arrays[key], value, err_msg=key)
    np.testing.assert_array_equal(arrays['pts'], np.full(251, 320000, dtype='<i8'))
    np.testing.assert_array_equal(arrays['noise_group_members'], members)
    np.testing.assert_array_equal(arrays['noise_group_weights'], weights)
    assert manifest['features'] == features
    assert manifest['active_observations'] == 0 and manifest['fit_executed'] is False
    assert (out/'observations.jsonl').read_text(encoding='utf-8') == metadata
    scene_path = out/manifest['scene_ref']
    prepared_model = SceneParameterModel(read_json(scene_path), scene_path)
    state_path = out/manifest['initial_state_ref']
    state = load_parameter_state(state_path, expected_scene=prepared_model.scene,
                                 expected_scene_path=scene_path)
    assert read_json(out/manifest['parameter_layout_ref']) == state.layout
    dependencies = read_json(out/manifest['parameter_dependencies_ref'])
    assert dependencies['layout_revision'] == state.layout['layout_revision']
    for feature in features:
        assert dependencies['features'][feature['id']] == prepared_model.dependencies(feature['path_id'])
    identity = shape_block(prepared_model, 'print_instance/Ad2::'+OUTER)
    candidate = changed(state, identity, 0)
    before = state_path.read_bytes()
    write_parameter_state(out, candidate, scene_ref=manifest['scene_ref'], name='candidate.json')
    assert state_path.read_bytes() == before
    np.testing.assert_array_equal(load_parameter_state(state_path).values, state.values)


def test_complete_artwork_rejects_nonlinear_warp_and_keeps_parent_hole_affine(model, core):
    state = model.initial_state
    outer, hole = ('print_instance/Ad2::'+p for p in (OUTER, INNER))
    identity = shape_block(model, outer)
    assert identity == shape_block(model, hole)
    block = state.blocks[identity]
    assert block['size'] == 10
    assert list(block['independent_local_indices']) == [0, 1, 2, 3]
    initial_values = state.values.copy()
    for column in range(4, 10):
        with pytest.raises(ValueError, match='frozen'):
            changed(state, identity, column, .45)
    np.testing.assert_array_equal(state.values, initial_values)
    # The current complete artwork admits a common affine transform. Neither
    # individual contours nor their holes may acquire a nonlinear warp.
    candidate = changed(state, identity, 0, .045)
    stations = np.linspace(0., 1., 101)
    original, moved = [], []
    for path in (outer, hole):
        a = model.evaluate_print(path, stations, core, state=state, jacobian=False)
        b = model.evaluate_print(path, stations, core, state=candidate, jacobian=False)
        assert np.max(abs(a['material_uv']-b['material_uv'])) > 1e-6
        np.testing.assert_allclose(b['material_uv'], a['material_uv']+[.045, 0.],
                                   rtol=0., atol=1e-14)
        np.testing.assert_array_equal(a['material_uv'][0], a['material_uv'][-1])
        np.testing.assert_array_equal(b['material_uv'][0], b['material_uv'][-1])
        original.append(a['material_uv'])
        moved.append(b['material_uv'])
    original, moved = np.vstack(original), np.vstack(moved)
    design = np.column_stack([original, np.ones(len(original))])
    # One affine map must reproduce both the parent and its hole exactly.
    affine = np.linalg.lstsq(design, moved, rcond=None)[0]
    np.testing.assert_allclose(design@affine, moved, rtol=0., atol=1e-12)
    other = next(p for p in model.paths if p.startswith('print_instance/Ad1::'))
    np.testing.assert_array_equal(
        model.evaluate_print(other, stations, core, state=state, jacobian=False)['xyz'],
        model.evaluate_print(other, stations, core, state=candidate, jacobian=False)['xyz'])


def test_paired_instances_share_shape_and_size_but_have_independent_placement(model, core):
    state = model.initial_state
    path = next(p for p in model.paths if p.startswith('print_instance/Ad1::')
                and len(model.dependencies(p)) == 3
                and state.blocks[model.dependencies(p)[2]]['active'])
    pair = path.replace('print_instance/Ad1::', 'print_instance/Ad14::', 1)
    dependencies, other_dependencies = model.dependencies(path), model.dependencies(pair)
    assert dependencies[1:] == other_dependencies[1:]
    shape = dependencies[2]
    column = state.blocks[shape]['independent_local_indices'][0]
    candidate = changed(state, shape, column, .08)
    stations = [.07, .23, .49, .79, .94]
    a = model.evaluate_print(path, stations, core, state=candidate, jacobian=False)
    b = model.evaluate_print(pair, stations, core, state=candidate, jacobian=False)
    np.testing.assert_array_equal(a['material_uv'], b['material_uv'])
    assert np.max(abs(a['xyz']-b['xyz'])) > 1e-4
    moved_one = changed(state, dependencies[0], 0, .001)
    np.testing.assert_array_equal(
        model.evaluate_print(pair, stations, core, state=state, jacobian=False)['xyz'],
        model.evaluate_print(pair, stations, core, state=moved_one, jacobian=False)['xyz'])


def test_domain_placement_logsize_and_shape_jacobians_match_independent_state_fd(model, core):
    state = model.initial_state
    path = 'print_instance/Ad2::'+INNER
    identity = shape_block(model, path)
    assert list(state.blocks[identity]['independent_local_indices']) == [0, 1, 2, 3]
    state = changed(state, identity, 2, .19)
    stations = [.043, .217, .481, .739, .927]
    result = model.evaluate_print(path, stations, core, state=state, jacobian=True)
    assert set(result['jacobians_by_block']) == set(model.dependencies(path))
    for block_id, analytic in result['jacobians_by_block'].items():
        block = state.blocks[block_id]
        np.testing.assert_array_equal(result['parameter_columns'][block_id],
            block['offset']+np.asarray(block['independent_local_indices']))
        for column in block['independent_local_indices']:
            epsilon = 2e-7
            plus = changed(state, block_id, column, epsilon)
            minus = changed(state, block_id, column, -epsilon)
            high = model.evaluate_print(path, stations, core, state=plus, jacobian=False)['xyz']
            low = model.evaluate_print(path, stations, core, state=minus, jacobian=False)['xyz']
            np.testing.assert_allclose(analytic[:, :, column], (high-low)/(2*epsilon),
                rtol=4e-5, atol=3e-7, err_msg=block_id+'/'+str(column))


@pytest.mark.parametrize('path,expected_shape_columns', [
    ('print_instance/Ad2::'+INNER, 4),
    ('print_instance/Ad1::Ad1.print.SKATING_I', 4),
    ('print_instance/Ad5::Ad5.bottom_edge/path0', 0),
])
def test_active_geometry_jacobians_assemble_by_global_columns(
        model, core, path, expected_shape_columns):
    state = model.initial_state
    stations = [.043, .217, .481, .739, .927]
    result = model.evaluate_print(path, stations, core, state=state)
    assert set(result['active_jacobians_by_block']) == set(result['parameter_columns'])
    direction = np.sin(np.arange(len(state.values)) + .37)
    predicted = np.zeros((len(stations), 3))
    plus, minus = {}, {}
    epsilon = 2e-7
    for identity, active_jacobian in result['active_jacobians_by_block'].items():
        block = state.blocks[identity]
        columns = result['parameter_columns'][identity]
        assert active_jacobian.shape == (len(stations), 3, len(columns))
        assert result['jacobians_by_block'][identity].shape[-1] == block['size']
        if block['kind'] in {'print_shape', 'print_rectangle'}:
            assert len(columns) == expected_shape_columns
        predicted += active_jacobian @ direction[columns]
        local_direction = np.zeros(block['size'])
        local_direction[list(block['independent_local_indices'])] = direction[columns]
        plus[identity] = state[identity] + epsilon*local_direction
        minus[identity] = state[identity] - epsilon*local_direction
    high = model.evaluate_print(path, stations, core,
        state=state.with_blocks(plus), jacobian=False)['xyz']
    low = model.evaluate_print(path, stations, core,
        state=state.with_blocks(minus), jacobian=False)['xyz']
    np.testing.assert_allclose(predicted, (high-low)/(2*epsilon), rtol=4e-5, atol=3e-7)


def test_review_load_and_common_grid_resampling_keep_numeric_candidate(model, core):
    state = model.initial_state
    identity = shape_block(model, 'print_instance/Ad2::'+OUTER)
    candidate = changed(state, identity, 0, .35)
    root = model.scene_path.parent
    write_parameter_state(root, state, name='initial.json')
    write_parameter_state(root, candidate, name='candidate.json')
    _, scene_a, a, _ = load_state(root/'initial.json', core, [])
    _, scene_b, b, _ = load_state(root/'candidate.json', core, [])
    grids = common_print_stations(a, b)
    a = sample_review_scene(scene_a, core, model.scene_path, print_stations=grids,
                            parameter_state=a['_parameter_state'])
    b = sample_review_scene(scene_b, core, model.scene_path, print_stations=grids,
                            parameter_state=b['_parameter_state'])
    assert a['inventory'] == b['inventory']
    cuts_a, cuts_b = ({c['id']: c for c in sample['cuts']} for sample in (a, b))
    for path in ('print_instance/Ad2::'+OUTER, 'print_instance/Ad2::'+INNER):
        first, second = cuts_a[path], cuts_b[path]
        x = a['xyz'][first['begin']:first['end']]
        y = b['xyz'][second['begin']:second['end']]
        assert x.shape == y.shape and np.max(abs(x-y)) > 1e-6
    assert set(a['inventory']['unlocalized_print_instance_ids']) == {
        'print_instance/Ad15', 'print_instance/Ad16'}


def test_instance_subset_retains_asset_values_without_activating_omitted_placement():
    from rinkspace.domain.scene_assets import read_scene_assets
    from rinkspace.domain.print_parameters import parameter_values
    from copy import deepcopy
    scene = read_json(FIELD)
    subset = deepcopy(scene)
    reference = subset['advertisement_assets'][0]
    omitted = reference['instance_ids'][1]
    reference['instance_ids'] = reference['instance_ids'][:1]
    parameters = compile_scene_parameters(subset, FIELD)
    assets = read_scene_assets(subset, FIELD)[0][1]
    values = parameter_values(assets, parameters)
    instance = next(i for i in assets['instances'] if i['id'] == omitted)
    block_id = instance['placement_block_id']
    assert block_id in values
    assert parameters.blocks[block_id]['active'] is False
    assert parameters.active_columns([block_id]).size == 0
