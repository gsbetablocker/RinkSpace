"""Saved diagnostic/cache contracts; no GUI, real model, fit or inference."""
import json
from pathlib import Path

import numpy as np
import pytest

from test_viewer_integrity import (
    DATA, PREPARE, MatrixSurface, fullbody_catalog, add_versioned_skate, replace_npz,
)
from calibration.program_tracking.body_tracking import spin_continuous_delivery as delivery


def diagnostics(catalog):
    n = len(catalog.pts)
    return dict(com_world_mid=np.arange(n * 3, dtype=float).reshape(n, 3) / 10,
                blade_material_world_mid=np.arange(n * 18, dtype=float).reshape(n, 2, 3, 3) / 100)


def test_saved_points_keep_identity_time_and_exactly_one_unit_conversion(tmp_path):
    catalog = fullbody_catalog(tmp_path)
    skate = add_versioned_skate(catalog)
    saved = diagnostics(catalog)
    replace_npz(catalog.root / 'result.npz', **saved)
    catalog.variants[0]['required_saved_diagnostics'] = list(delivery.SAVED_FIELDS)
    path = PREPARE.prepare_variant(catalog, 'fixture', MatrixSurface())
    sequence = DATA.CachedSequence(path)
    np.testing.assert_array_equal(sequence.state['pts'], catalog.pts)
    for key, values in saved.items():
        np.testing.assert_array_equal(sequence.state[key], values * 3.)
    np.testing.assert_array_equal(sequence.state['skate_witness'], skate['skate_witness_world_mid'] * 3.)
    metadata = sequence.meta['motion_diagnostics']
    assert metadata['blade_material_order'] == ['rear', 'middle', 'front']
    assert metadata['estimated_by_viewer'] is False
    assert 'not measured true' in metadata['com']
    assert 'not verified' in metadata['witness']
    assert PREPARE.prepare_variant(catalog, 'fixture', MatrixSurface()) == path
    replace_npz(catalog.root / 'result.npz', com_world_mid=saved['com_world_mid'] + .01)
    assert catalog.cache_path('fixture')[0] != path


@pytest.mark.parametrize('missing', delivery.SAVED_FIELDS)
def test_new_stage_cannot_publish_with_a_missing_saved_diagnostic(tmp_path, missing):
    catalog = fullbody_catalog(tmp_path)
    add_versioned_skate(catalog)
    replace_npz(catalog.root / 'result.npz', **diagnostics(catalog))
    saved = DATA.load_npz(catalog.root / 'result.npz')
    del saved[missing]
    np.savez_compressed(catalog.root / 'result.npz', **saved)
    catalog.variants[0]['required_saved_diagnostics'] = list(delivery.SAVED_FIELDS)
    with pytest.raises(ValueError, match='Required saved motion diagnostic'):
        PREPARE.prepare_variant(catalog, 'fixture', MatrixSurface())
    assert not list(catalog.cache_root.glob('fixture_*/metadata.json'))


@pytest.mark.parametrize('field,value', [
    ('com_world_mid', np.full((3, 3), np.nan)),
    ('com_world_mid', np.zeros((3, 2))),
    ('blade_material_world_mid', np.zeros((3, 2, 2, 3))),
    ('blade_material_world_mid', np.full((3, 2, 3, 3), np.inf)),
])
def test_invalid_saved_point_fields_cannot_publish(tmp_path, field, value):
    catalog = fullbody_catalog(tmp_path)
    replace_npz(catalog.root / 'result.npz', **{field: value})
    with pytest.raises(ValueError, match=field):
        PREPARE.prepare_variant(catalog, 'fixture', MatrixSurface())
    assert not list(catalog.cache_root.glob('fixture_*/metadata.json'))


def test_configuration_retains_networks_but_frames_only_new_stages_and_two_baselines(tmp_path):
    variants = [dict(id=name, result='retained-'+name) for name in
                ('gvhmr_fixed', 'exmr_prefoot_old', 'entry_motion', 'support_foot')]
    before = dict(variants=variants, default_variant='entry_motion')
    stages = {stage: dict(variant=variant, root=tmp_path / stage,
                         case=tmp_path / stage / delivery.shared.CASE,
                         result=tmp_path / stage / 'result.npz', summary=dict(accepted=index + 1))
              for index, (stage, variant) in enumerate(zip(delivery.STAGES, delivery.VARIANTS))}
    config = delivery._configuration(before, stages)
    assert config['variants'][:len(variants)] == variants
    assert before['default_variant'] == 'entry_motion'
    assert len(before['variants']) == len(variants)
    assert config['comparison_variants'] == list(delivery.COMPARISON)
    assert 'gvhmr_fixed' not in config['comparison_variants']
    assert config['default_variant'] == 'continuous_flight'
    assert config['session_id'] == 'spin_continuous_1105'
    for variant in config['variants'][-2:]:
        assert variant['required_saved_diagnostics'] == list(delivery.SAVED_FIELDS)


def test_registration_waits_for_real_finals_before_any_player_file_is_created(tmp_path):
    project = tmp_path / 'project'; base = tmp_path / 'runs'; output = base / 'spin_continuous_1105'
    programme = project / delivery.shared.PROGRAMME
    programme.mkdir(parents=True)
    contract_path = base / 'action_integration_1103/spin_s05/contract.json'
    contract_path.parent.mkdir(parents=True)
    source = tmp_path / 'source.mp4'; source.write_bytes(b'identity-only fixture; never decoded')
    source_info = dict(path=str(source), time_base=[1, 12800], pts_range=[972800, 1216000], native_size_wh=[1920, 1080])
    contract_path.write_text(json.dumps(dict(source=source_info, native50_count=951)), encoding='utf8')
    (programme / delivery.SOURCE_CONFIG).write_text(json.dumps(dict(
        variants=[dict(id=name) for name in ('entry_motion', 'support_foot', 'gvhmr_fixed')],
        source_video=str(source), source_time_base=[1, 12800])), encoding='utf8')
    with pytest.raises(FileNotFoundError, match='Required completed source_stage artifact'):
        delivery.register(project, base, output)
    assert not (programme / delivery.CONFIG).exists()
    assert not (project / delivery.LAUNCHER).exists()
    assert not output.exists()
