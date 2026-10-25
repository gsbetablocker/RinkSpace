"""Print material boundaries stay contours through the generic input contract.

Only test fixtures are prepared here. No production observations are activated,
no statistical model is estimated, and no solve or image rendering is invoked.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pytest

from rinkspace.domain.contracts import ContractError, Feature
from rinkspace.domain.measurement import required_measurement, validate_observation_binding
from rinkspace.domain.print_assets import PrintBindingError, resolve_print_curve
from rinkspace.domain.scene_assets import read_scene_assets, snapshot_scene
from rinkspace.session.dataset import load_array, load_prepared, read_json, write_array, write_json
from rinkspace.session.prepare import prepare, validate_current_feature_contracts

ROOT = Path(__file__).resolve().parents[1]
FIELD = ROOT/'events/ankara_2026/scene/field/model.json'
OUTER = 'Ad2.print.COM_O'
INNER = 'Ad2.print.COM_O_inner'
OBSERVATION_IDS = ('ad_source/f65a12055a975703fd20326b0e59b955',
                   'ad_source/aa2fb4a7b5af07b0d7a6da23553b793f')


def current_assets():
    return read_scene_assets(read_json(FIELD), FIELD)


def ad2_assets():
    return next(assets for ref, assets in current_assets()
                if 'print_instance/Ad2' in ref['instance_ids'])


def feature_record(binding):
    return dict(id=binding.path_id, path_id=binding.path_id,
                label=binding.template_path_id, kind='finite_curve',
                surface='board', identity='confirmed', measurement='contour',
                has_station_parameter=True, print_binding=binding.as_record())


def test_every_current_registered_print_boundary_has_contour_semantics():
    count = 0
    templates = set()
    instances = set()
    roles = set()
    features = []
    for ref, assets in current_assets():
        before = deepcopy(assets)
        for instance_id in ref['instance_ids']:
            instances.add(instance_id)
            instance = next(i for i in assets['instances'] if i['id'] == instance_id)
            template = next(t for t in assets['templates'] if t['id'] == instance['template_id'])
            templates.add(template['id'])
            for path in template['paths']:
                binding = resolve_print_curve(assets, instance_id, path['id'])
                assert binding.instance_id == instance_id
                assert binding.template_id == template['id']
                assert binding.template_path_id == path['id']
                assert binding.role == path['role']
                assert binding.parent_contour_key == path.get('parent_contour_key')
                assert binding.path_id == instance_id+'::'+path['id']
                assert required_measurement(path['id'], surface='board',
                                            print_binding=binding) == 'contour'
                feature = Feature(binding.path_id, 'finite_curve', 'board',
                                  'confirmed', 'contour', has_station_parameter=True,
                                  label=path['id'], print_binding=binding)
                assert feature.measurement == 'contour'
                roles.add(binding.role)
                features.append(feature_record(binding))
                count += 1
        assert assets == before
    assert len(templates) == 13
    assert len(instances) == 26
    # Template content may be corrected without changing measurement policy.
    # Check every distinct current instance path, not a frozen drawing count.
    assert count == len(features) == len({f['path_id'] for f in features})
    assert {'letter_material_boundary', 'letter_hole_boundary', 'panel_outer_boundary',
            'print_material_boundary', 'logo_outer_boundary',
            'printed_ink_open_boundary_arc'} <= roles
    empty = dict(pts=np.empty(0, dtype='<i8'), feature_index=np.empty(0, dtype='<i4'),
                 active=np.empty(0, dtype='u1'))
    report = validate_current_feature_contracts({}, read_json(FIELD), features, empty,
                                               np.empty(0, dtype='<i8'), scene_path=FIELD)
    assert report['measurement']['all_current_features_valid']


def test_outer_and_hole_keep_distinct_immutable_material_identities():
    assets = ad2_assets()
    outer = resolve_print_curve(assets, 'print_instance/Ad2', OUTER)
    hole = resolve_print_curve(assets, 'print_instance/Ad2', INNER)
    assert outer.role == 'letter_material_boundary'
    assert outer.parent_contour_key is None
    assert hole.role == 'letter_hole_boundary'
    assert hole.parent_contour_key == outer.template_path_id
    assert hole.path_id != outer.path_id
    with pytest.raises(FrozenInstanceError):
        hole.role = outer.role


@pytest.mark.parametrize('label', ['ice.C0:outer', 'ice.C', 'ice.B1', 'ice.B2',
                                 'ice.lane3:edge2', *[f'ice.A{i}:outer' for i in range(1, 5)]])
def test_ice_markings_still_reject_material_contour_mode(label):
    assert required_measurement(label, surface='ice') == 'centreline'
    with pytest.raises(ContractError):
        Feature(label, 'finite_curve', 'ice', 'confirmed', 'contour', label=label)


@pytest.mark.parametrize('kind,surface,measurement', [
    ('finite_curve', 'ice', 'contour'),
    ('finite_curve', 'board', 'centreline'),
    ('material_point', 'board', 'material_point'),
])
def test_print_binding_cannot_override_wrong_surface_kind_or_measurement(kind, surface, measurement):
    binding = resolve_print_curve(ad2_assets(), 'print_instance/Ad2', OUTER)
    with pytest.raises(ContractError):
        Feature('print-feature', kind, surface, 'confirmed', measurement,
                label=OUTER, print_binding=binding)


def test_name_or_surface_alone_never_grants_print_contour_semantics():
    with pytest.raises(ContractError):
        Feature(OUTER, 'finite_curve', 'board', 'confirmed', 'contour', label=OUTER)
    binding = resolve_print_curve(ad2_assets(), 'print_instance/Ad2', OUTER)
    with pytest.raises((ContractError, PrintBindingError, TypeError)):
        required_measurement(OUTER, surface='board', print_binding=binding.as_record())


@pytest.mark.parametrize('change', ['unknown_instance', 'unknown_path', 'unknown_role',
                                  'hole_missing_parent', 'hole_wrong_parent', 'hole_parent_is_hole'])
def test_resolver_rejects_unbound_or_invalid_material_identity(change):
    assets = ad2_assets()
    instance_id, path_id = 'print_instance/Ad2', INNER
    paths = assets['templates'][0]['paths']
    hole = next(p for p in paths if p['id'] == INNER)
    if change == 'unknown_instance':
        instance_id = 'print_instance/not_registered'
    elif change == 'unknown_path':
        path_id = 'not_a_material_path'
    elif change == 'unknown_role':
        hole['role'] = 'texture_sample_or_ice_midline'
    elif change == 'hole_missing_parent':
        hole.pop('parent_contour_key')
    elif change == 'hole_wrong_parent':
        hole['parent_contour_key'] = 'not_a_parent_path'
    else:
        hole['parent_contour_key'] = 'Ad2.print.SKATING_A_inner'
    with pytest.raises(PrintBindingError):
        resolve_print_curve(assets, instance_id, path_id)


def make_print_input(root):
    """Use the accepted exposure's exact UV in an inactive, synthetic fixture."""
    root.mkdir()
    scene, _ = snapshot_scene(FIELD, root/'field')
    assets = ad2_assets()
    observations = [next(o for o in assets['observations'] if o['id'] == identity)
                    for identity in OBSERVATION_IDS]
    assert [o['feature_id'] for o in observations] == [OUTER, INNER]
    assert all(o['pts'] == 320000 and o['time_base'] == '1/12800' for o in observations)
    bindings = [resolve_print_curve(assets, 'print_instance/Ad2', path)
                for path in (OUTER, INNER)]
    features = [feature_record(binding) for binding in bindings]
    lengths = [len(o['source_native_uv']) for o in observations]
    n = sum(lengths)
    assert lengths == [150, 101]
    uv = np.concatenate([np.asarray(o['source_native_uv'], dtype='<f8') for o in observations])
    arrays = dict(
        uv=uv, pts=np.full(n, 320000, dtype='<i8'),
        source_index=np.zeros(n, dtype='<i4'), shot_index=np.zeros(n, dtype='<i4'),
        feature_index=np.repeat(np.arange(2, dtype='<i4'), lengths),
        kind=np.ones(n, dtype='u1'), active=np.zeros(n, dtype='u1'),
        split=np.zeros(n, dtype='u1'), noise_group_index=np.zeros(n, dtype='<i4'),
        covariance=np.zeros((n, 3), dtype='<f8'),
        support=np.tile([0., 1.], (n, 1)).astype('<f8'),
        local_index=np.arange(n, dtype='<i8'))
    # Deliberately synthetic, rank-two group. This only verifies preservation;
    # it is not a covariance estimate or an activation of accepted source UV.
    members = np.arange(n, dtype='<i8')
    weights = np.eye(2, n*2, dtype='<f8').ravel()
    group = dict(id='synthetic-test-only/source-fragment-group', member_start=0,
                 member_count=n, rank=2, weight_start=0)
    write_json(root/'groups.json', dict(groups=[group],
        members_ref=write_array(root, 'members', members),
        weights_ref=write_array(root, 'weights', weights)))
    pts_ref = write_array(root, 'source_pts', [320000], '<i8')
    write_json(root/'sources.json', dict(sources=[dict(
        id=observations[0]['source_id'], sha256=observations[0]['source_sha256'],
        width=1920, height=1080, time_base_num=1, time_base_den=12800, pts_ref=pts_ref)]))
    processing = dict(first_pts=320000, last_pts=320000, stop_pts_exclusive=320256,
                      native_frame_count=1)
    shot = dict(id='s01', source_id=observations[0]['source_id'], processing=processing,
                camera_domain=dict(status='defined', first_pts=320000, stop_pts_exclusive=320256))
    write_json(root/'shots.json', dict(shots=[shot]))
    write_json(root/'scope.json', dict(required_shot_ids=['s01'], shots=[shot]))
    write_json(root/'features.json', dict(features=features))
    metadata = []
    for observation, feature in zip(observations, features):
        for station, source_uv in enumerate(observation['source_native_uv']):
            metadata.append(dict(id=observation['id']+'/'+str(station),
                feature_id=feature['id'], measurement='contour', source_observation_id=observation['id'],
                source_uv=source_uv, pts=observation['pts'], quality='pending', active=False,
                noise_group_id=group['id'], note='synthetic inactive regression fixture only'))
    raw_metadata = ''.join(json.dumps(m, ensure_ascii=False, separators=(',', ':'))+'\n' for m in metadata)
    (root/'observations.jsonl').write_text(raw_metadata, encoding='utf-8')
    write_json(root/'obs.json', dict(metadata_ref='observations.jsonl',
        arrays={key: write_array(root, key, value) for key, value in arrays.items()}))
    write_json(root/'rigs.json', dict(rigs=[]))
    write_json(root/'provenance.json', dict(test_fixture=True, production_statistical_model=False))
    write_json(root/'input.json', dict(schema='rinkspace.input.v1', dataset_id='synthetic-print-contract',
        revision='r0', scene_ref='field/scene.json', rigs_ref='rigs.json', shots_ref='shots.json',
        sources_ref='sources.json', observations_ref='obs.json', features_ref='features.json',
        noise_groups_ref='groups.json', provenance_ref='provenance.json'))
    return arrays, features, members, weights, raw_metadata


def test_generic_prepare_and_consumer_preserve_exact_o_pixels_time_and_group(tmp_path):
    root = tmp_path/'input'
    expected, features, members, weights, metadata = make_print_input(root)
    receipt = prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'prepared')
    manifest, arrays = load_prepared(tmp_path/'prepared', purpose='fit')
    for key, values in expected.items():
        np.testing.assert_array_equal(arrays[key], values, err_msg=key)
    np.testing.assert_array_equal(arrays['noise_group_members'], members)
    np.testing.assert_array_equal(arrays['noise_group_weights'], weights)
    assert manifest['features'] == features
    assert manifest['active_observations'] == 0
    assert manifest['fit_executed'] is False
    assert (tmp_path/'prepared/observations.jsonl').read_text(encoding='utf-8') == metadata
    assert Path(receipt.manifest_path) == tmp_path/'prepared/manifest.json'


@pytest.mark.parametrize('change', ['template', 'role', 'parent', 'path', 'instance',
                                  'ice_surface', 'centreline', 'missing_binding'])
def test_prepare_rejects_tampered_print_binding_even_when_all_rows_are_inactive(tmp_path, change):
    root = tmp_path/'input'
    make_print_input(root)
    doc = read_json(root/'features.json')
    feature = doc['features'][1]
    if change == 'template':
        feature['print_binding']['template_id'] = 'print_template/another'
    elif change == 'role':
        feature['print_binding']['role'] = 'letter_material_boundary'
    elif change == 'parent':
        feature['print_binding']['parent_contour_key'] = 'Ad2.print.SKATING_A'
    elif change == 'path':
        feature['path_id'] = 'ice.C'
    elif change == 'instance':
        feature['print_binding']['instance_id'] = 'print_instance/not_in_scene'
    elif change == 'ice_surface':
        feature['surface'] = 'ice'
    elif change == 'centreline':
        feature['measurement'] = 'centreline'
    else:
        feature.pop('print_binding')
    write_json(root/'features.json', doc)
    with pytest.raises((ContractError, ValueError)):
        prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'prepared')


def test_prepared_consumer_rechecks_binding_against_its_current_snapshot(tmp_path):
    root = tmp_path/'input'
    make_print_input(root)
    prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'prepared')
    path = tmp_path/'prepared/manifest.json'
    manifest = read_json(path)
    manifest['features'][1]['print_binding']['template_id'] = 'print_template/not_the_o_template'
    write_json(path, manifest)
    with pytest.raises((ContractError, ValueError)):
        load_prepared(tmp_path/'prepared', purpose='fit')


def test_print_row_metadata_cannot_silently_change_to_ice_midline_semantics(tmp_path):
    root = tmp_path/'input'
    make_print_input(root)
    path = root/'observations.jsonl'
    lines = path.read_text(encoding='utf-8').splitlines()
    first = json.loads(lines[0])
    first['measurement'] = 'centreline'
    lines[0] = json.dumps(first)
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with pytest.raises((ContractError, ValueError)):
        prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'prepared')


def test_print_binding_cannot_disguise_an_ice_label():
    binding = resolve_print_curve(ad2_assets(), 'print_instance/Ad2', OUTER)
    with pytest.raises(ContractError):
        Feature('ice.C0', 'finite_curve', 'board', 'confirmed', 'contour',
                label='ice.C0', print_binding=binding)


def test_declared_print_without_binding_cannot_masquerade_as_house_contour():
    feature = dict(id='fake-print', label='ice.F1', kind='finite_curve', surface='board',
                   identity='confirmed', measurement='contour',
                   instance_id='print_instance/Ad2', template_path_id=OUTER)
    empty = dict(pts=np.empty(0, dtype='<i8'), feature_index=np.empty(0, dtype='<i4'),
                 active=np.empty(0, dtype='u1'))
    with pytest.raises(PrintBindingError):
        validate_current_feature_contracts({}, read_json(FIELD), [feature], empty,
                                           np.empty(0, dtype='<i8'), scene_path=FIELD)


def test_ice_observation_cannot_claim_unrelated_print_material_binding():
    binding = resolve_print_curve(ad2_assets(), 'print_instance/Ad2', OUTER)
    feature = dict(id='ice.F1', label='ice.F1', kind='finite_curve',
                   surface='ice', measurement='contour')
    row = dict(feature_id='ice.F1', measurement='contour', print_binding=binding.as_record())
    with pytest.raises(ContractError):
        validate_observation_binding(row, feature, 0)


def test_identity_only_paired_print_cannot_become_active_through_binding(tmp_path):
    root = tmp_path/'input'
    make_print_input(root)
    assets = ad2_assets()
    features = [feature_record(resolve_print_curve(assets, 'print_instance/Ad15', p))
                for p in (OUTER, INNER)]
    write_json(root/'features.json', dict(features=features))
    # Trigger geometry admission with a declared active row; no solve is run.
    obs = read_json(root/'obs.json')
    obs['arrays']['active'] = write_array(root, 'active', np.ones(251, dtype='u1'))
    write_json(root/'obs.json', obs)
    with pytest.raises(PrintBindingError, match='unlocalized'):
        prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'prepared')
