"""Packed current correlation identities and row types survive both entrypoints."""
from copy import deepcopy

import numpy as np
import pytest

from rinkspace.session.dataset import DatasetError, load_prepared, read_json, write_array, write_json
from rinkspace.session.prepare import prepare
from test_prepare_source_binding import PrepareSourceBinding


def grouped_input(root):
    arrays = PrepareSourceBinding().make_input(root, [dict(active=0), dict(active=0)])
    arrays['noise_group_index'][:] = [0, 1]
    arrays['covariance'][:] = 0
    obs = read_json(root/'obs.json')
    for key in ('noise_group_index', 'covariance'):
        obs['arrays'][key] = write_array(root, key, arrays[key])
    write_json(root/'obs.json', obs)
    groups = [dict(id='group/'+str(i), member_start=i, member_count=1, rank=2, weight_start=4*i)
              for i in range(2)]
    write_json(root/'groups.json', dict(groups=groups,
        members_ref=write_array(root, 'members', [0, 1], '<i8'),
        weights_ref=write_array(root, 'weights', np.tile(np.eye(2).ravel(), 2), '<f8')))
    return groups


@pytest.mark.parametrize('stage', ['prepare', 'reload'])
@pytest.mark.parametrize('damage', ['fractional_members', 'matrix_members', 'float32_weights',
    'matrix_weights', 'duplicate_ids', 'negative_start', 'floating_count', 'boolean_rank',
    'negative_rank', 'past_end_weight', 'unclaimed_group_member'])
def test_current_group_layout_cannot_silently_change_support(tmp_path, stage, damage):
    root = tmp_path/'input'
    grouped_input(root)
    prepared = tmp_path/'prepared'
    if stage == 'reload':
        prepare(root/'input.json', root/'scope.json', 's01', prepared)
    output = root if stage == 'prepare' else prepared
    path = output/('groups.json' if stage == 'prepare' else 'manifest.json')
    doc = read_json(path)
    groups = doc['groups'] if stage == 'prepare' else doc['noise_groups']

    def replace_array(kind, values):
        if stage == 'prepare':
            doc[kind+'_ref'] = write_array(output, kind, values)
        else:
            name = 'noise_group_'+kind
            doc['arrays'][name] = write_array(output, name, values)

    if damage == 'fractional_members':
        replace_array('members', np.array([.9, 1.9], dtype='<f8'))
    elif damage == 'matrix_members':
        replace_array('members', np.array([[0], [1]], dtype='<i8'))
    elif damage == 'float32_weights':
        replace_array('weights', np.tile(np.eye(2).ravel(), 2).astype('<f4'))
    elif damage == 'matrix_weights':
        replace_array('weights', np.tile(np.eye(2).ravel(), 2).reshape(8, 1))
    elif damage == 'duplicate_ids':
        groups[1]['id'] = groups[0]['id']
    elif damage == 'negative_start':
        groups[0]['member_start'] = -2
    elif damage == 'floating_count':
        groups[0]['member_count'] = 1.0
    elif damage == 'boolean_rank':
        groups[0]['rank'] = True
    elif damage == 'negative_rank':
        groups[0]['rank'] = -1
    elif damage == 'past_end_weight':
        groups[0]['weight_start'] = 8
    else:
        groups.pop()
    write_json(path, doc)
    with pytest.raises(DatasetError):
        if stage == 'prepare':
            prepare(root/'input.json', root/'scope.json', 's01', prepared)
        else:
            load_prepared(prepared)


def test_rank_zero_constraint_only_group_is_preserved_without_fake_precision(tmp_path):
    root = tmp_path/'input'
    grouped_input(root)
    doc = read_json(root/'groups.json')
    doc['groups'][1].update(rank=0, model='constraint_only_unknown_noise')
    doc['weights_ref'] = write_array(root, 'weights', np.eye(2).ravel(), '<f8')
    write_json(root/'groups.json', doc)
    before = deepcopy(doc['groups'])
    prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'prepared')
    manifest, arrays = load_prepared(tmp_path/'prepared')
    assert manifest['noise_groups'] == before
    np.testing.assert_array_equal(arrays['noise_group_weights'], np.eye(2).ravel())
    np.testing.assert_array_equal(arrays['noise_group_members'], [0, 1])


@pytest.mark.parametrize('stage', ['prepare', 'reload'])
@pytest.mark.parametrize('feature_kind', ['finite_curve', 'material_point'])
def test_numerical_row_kind_cannot_reclassify_a_current_feature(tmp_path, stage, feature_kind):
    root = tmp_path/'input'
    arrays = PrepareSourceBinding().make_input(root, [dict(active=0)])
    features = read_json(root/'features.json')
    if feature_kind == 'material_point':
        features['features'][0].update(kind='material_point', measurement='material_point')
        arrays['kind'][:] = 0
        arrays['local_index'][:] = -1
        arrays['support'][:] = 0
    write_json(root/'features.json', features)
    obs = read_json(root/'obs.json')
    for key in ('kind', 'local_index', 'support'):
        obs['arrays'][key] = write_array(root, key, arrays[key])
    write_json(root/'obs.json', obs)
    prepared = tmp_path/'prepared'
    if stage == 'reload':
        prepare(root/'input.json', root/'scope.json', 's01', prepared)
    arrays['kind'][:] = 0 if feature_kind == 'finite_curve' else 1
    arrays['local_index'][:] = -1 if feature_kind == 'finite_curve' else 0
    arrays['support'][:] = [0., 0.] if feature_kind == 'finite_curve' else [0., 1.]
    output = root if stage == 'prepare' else prepared
    path = output/('obs.json' if stage == 'prepare' else 'manifest.json')
    doc = read_json(path)
    for key in ('kind', 'local_index', 'support'):
        doc['arrays'][key] = write_array(output, key, arrays[key])
    write_json(path, doc)
    with pytest.raises(DatasetError, match='numerical kind differs'):
        if stage == 'prepare':
            prepare(root/'input.json', root/'scope.json', 's01', prepared)
        else:
            load_prepared(prepared)
