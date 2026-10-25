"""Current row metadata cannot transfer pixels, exposures or activity silently.

The ordinary rows below are synthetic contract fixtures. Only the final test
uses the accepted Ad2 exposure, preserving its original inactive source data.
"""
import json

import numpy as np
import pytest

from rinkspace.session.dataset import DatasetError, load_prepared, read_json, write_array, write_json
from rinkspace.session.prepare import prepare
from test_prepare_source_binding import PrepareSourceBinding


def write_records(path, records):
    path.write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in records), encoding='utf-8')


def make_bound_input(root, rows=None):
    arrays = PrepareSourceBinding().make_input(root, rows or [dict(active=0)])
    n = len(arrays['pts'])
    arrays['observation_ids'] = np.asarray(['fixture/'+str(i) for i in range(n)])
    sources = read_json(root/'sources.json')['sources']
    shots = read_json(root/'shots.json')['shots']
    records = []
    for i in range(n):
        source = sources[int(arrays['source_index'][i])]
        records.append(dict(id=str(arrays['observation_ids'][i]), feature_id='ice.C',
            measurement='centreline', source_uv=arrays['uv'][i].tolist(), pts=int(arrays['pts'][i]),
            source_id=source['id'], source_sha256=source['sha256'], time_base='1/12800',
            shot_id=shots[int(arrays['shot_index'][i])]['id'], noise_group_id=None,
            active=bool(arrays['active'][i]), origin='automatic', source_evidence='measured',
            quality='machine_verified' if arrays['active'][i] else 'pending',
            provenance={'file': 'missing_history_is_not_a_runtime_dependency.json'}))
    write_records(root/'observations.jsonl', records)
    obs = read_json(root/'obs.json')
    obs['metadata_ref'] = 'observations.jsonl'
    obs['arrays']['observation_ids'] = write_array(root, 'observation_ids', arrays['observation_ids'])
    write_json(root/'obs.json', obs)
    return arrays, records


@pytest.mark.parametrize('stage', ['prepare', 'reload'])
@pytest.mark.parametrize('change', [
    dict(source_uv=[50., 20.]), dict(uv=[10., 21.]), dict(pts=150), dict(pts=100.0),
    dict(source_id='other'), dict(source_sha256='b'*64), dict(time_base='1/60000'),
    dict(time_base_num=2), dict(time_base_den=60000), dict(shot_id='s02'),
    dict(noise_group_id='wrong-group'), dict(id='some-other-observation'),
    dict(active=True), dict(active_for_fit=True), dict(active='false'), dict(active=0),
    dict(active_for_fit='false'),
])
def test_explicit_metadata_must_bind_exact_current_row(tmp_path, stage, change):
    root = tmp_path/'input'
    _, records = make_bound_input(root)
    prepared = tmp_path/'prepared'
    if stage == 'reload':
        prepare(root/'input.json', root/'scope.json', 's01', prepared)
    path = (root if stage == 'prepare' else prepared)/'observations.jsonl'
    records[0].update(change)
    write_records(path, records)
    before = path.read_bytes()
    with pytest.raises(DatasetError):
        if stage == 'prepare':
            prepare(root/'input.json', root/'scope.json', 's01', prepared)
        else:
            load_prepared(prepared)
    assert path.read_bytes() == before
    if stage == 'reload':
        # A failure remains inspectable; no metadata or pixels are repaired.
        load_prepared(prepared, purpose='review')


@pytest.mark.parametrize('stage', ['prepare', 'reload'])
@pytest.mark.parametrize('change', [dict(quality='pending'), dict(quality='rejected'),
    dict(quality='existing_rejected'), dict(origin='synthetic'),
    dict(source_evidence='mathematical_only'), dict(source_evidence='unavailable'),
    dict(active_for_fit=False)])
def test_explicit_ineligible_evidence_cannot_become_active(tmp_path, stage, change):
    root = tmp_path/'input'
    _, records = make_bound_input(root, [dict(active=1)])
    prepared = tmp_path/'prepared'
    if stage == 'reload':
        prepare(root/'input.json', root/'scope.json', 's01', prepared)
    records[0].update(change)
    write_records((root if stage == 'prepare' else prepared)/'observations.jsonl', records)
    with pytest.raises(DatasetError):
        if stage == 'prepare':
            prepare(root/'input.json', root/'scope.json', 's01', prepared)
        else:
            load_prepared(prepared)


@pytest.mark.parametrize('origin,quality', [('automatic', 'machine_verified'),
    ('human', 'human_confirmed'), ('legacy_import', 'existing_confirmed_source')])
def test_admitted_current_quality_does_not_require_new_human_review(tmp_path, origin, quality):
    root = tmp_path/'input'
    expected, records = make_bound_input(root, [dict(active=1)])
    records[0].update(origin=origin, quality=quality)
    write_records(root/'observations.jsonl', records)
    before = (root/'observations.jsonl').read_bytes()
    prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'prepared')
    _, actual = load_prepared(tmp_path/'prepared')
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
    assert (tmp_path/'prepared/observations.jsonl').read_bytes() == before


def test_selection_keeps_original_row_and_remaps_group_index_by_identity(tmp_path):
    root = tmp_path/'input'
    arrays, records = make_bound_input(root, [dict(active=0, pts=200, shot_index=1), dict(active=0)])
    arrays['noise_group_index'][:] = [0, 1]
    arrays['covariance'][:] = 0
    obs = read_json(root/'obs.json')
    for key in ('noise_group_index', 'covariance'):
        obs['arrays'][key] = write_array(root, key, arrays[key])
    write_json(root/'obs.json', obs)
    groups = [dict(id='fragment/'+str(i), member_start=i, member_count=1, rank=2, weight_start=4*i)
              for i in range(2)]
    write_json(root/'groups.json', dict(groups=groups,
        members_ref=write_array(root, 'members', [0, 1], '<i8'),
        weights_ref=write_array(root, 'weights', np.tile(np.eye(2).ravel(), 2), '<f8')))
    for i, record in enumerate(records):
        record['noise_group_id'] = groups[i]['id']
    records[0]['source_uv'] = [-999., -999.]  # Unselected history is not re-adjudicated.
    write_records(root/'observations.jsonl', records)
    wanted = (root/'observations.jsonl').read_text().splitlines(keepends=True)[1]
    prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'prepared')
    manifest, actual = load_prepared(tmp_path/'prepared')
    assert manifest['noise_groups'][0]['id'] == 'fragment/1'
    np.testing.assert_array_equal(actual['noise_group_index'], [0])
    np.testing.assert_array_equal(actual['noise_group_members'], [0])
    np.testing.assert_array_equal(actual['noise_group_weights'], np.eye(2).ravel())
    assert (tmp_path/'prepared/observations.jsonl').read_text() == wanted


def test_accepted_ad2_exposure_cannot_activate_by_changing_only_numeric_mask(tmp_path):
    from test_print_measurement import make_print_input
    root = tmp_path/'input'
    arrays, _, _, _, _ = make_print_input(root)
    before = (root/'observations.jsonl').read_bytes()
    obs = read_json(root/'obs.json')
    obs['arrays']['active'] = write_array(root, 'active', np.ones(len(arrays['pts']), dtype='u1'))
    write_json(root/'obs.json', obs)
    with pytest.raises(DatasetError, match='metadata differs'):
        prepare(root/'input.json', root/'scope.json', 's01', tmp_path/'prepared')
    assert (root/'observations.jsonl').read_bytes() == before
