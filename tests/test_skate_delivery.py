"""Delivery identity and lossless layered-control contracts; no fit or player UI."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.program_tracking.body_tracking import skate_delivery as delivery
from calibration.program_tracking.body_tracking.skate_template import SkateParameters
from calibration.program_tracking.body_tracking.skate_checkpoint import checkpoint_identity


def test_parameter_replay_preserves_layers_nan_payload_and_signed_zero(tmp_path):
    saved = dict(x=np.arange(86, dtype=np.float64), active=np.arange(69),
                 base_layer_0_state_kind=np.array('halo_1103'),
                 base_layer_0_x=np.array([0x7ff8000000001234, 0x8000000000000000], dtype=np.uint64).view(np.float64),
                 base_layer_0_seconds=np.array([28.12, 43.80]))
    shared = np.arange(17, dtype=np.float64)
    path = tmp_path/'parameters.npz'
    np.savez_compressed(path, **dict(saved, shared_x=shared))
    with np.load(path) as archive:
        replay = dict(archive)
    result = delivery._parameter_replay_checks(replay, saved, shared)
    assert result['saved_field_count'] == len(saved)
    assert all(result['saved_fields_byte_equal'].values())
    # Equal numerical values cannot hide changed layer controls or NaN bits.
    replay['base_layer_0_x'] = np.array([np.nan, 0.])
    with pytest.raises(ValueError, match='base_layer_0_x'):
        delivery._parameter_replay_checks(replay, saved, shared)


@pytest.mark.parametrize('change', ['missing', 'dtype', 'shape'])
def test_replay_rejects_incomplete_or_reinterpreted_saved_state(change):
    saved = {'x': np.zeros(86), 'base_layer_0_x': np.arange(6, dtype=np.float64)}
    replay = dict(saved, shared_x=np.zeros(17))
    if change == 'missing':
        del replay['base_layer_0_x']
    elif change == 'dtype':
        replay['base_layer_0_x'] = replay['base_layer_0_x'].astype(np.float32)
    else:
        replay['base_layer_0_x'] = replay['base_layer_0_x'].reshape(2, 3)
    with pytest.raises(ValueError, match='base_layer_0_x'):
        delivery._parameter_replay_checks(replay, saved, np.zeros(17))


@pytest.mark.parametrize('new_identity', [False, True])
def test_delivery_retains_old_candidate_and_uses_actual_source_scope(tmp_path, monkeypatch, capsys, new_identity):
    project, root = tmp_path/'p', tmp_path/'r'
    programme = project/'events/ankara_2026/programmes/okada_mei_sp'
    programme.mkdir(parents=True)
    out = root/'new_run'
    (out/'fit').mkdir(parents=True)
    (out/'fitted').mkdir()
    pts = np.array([283136, 450560, 575488])
    old_variant = dict(id='skate1103', label='retained prior candidate', result='old/fitted/result.npz')
    config_path = programme/'viewer3d_fullbody_1102.json'
    config_path.write_text(json.dumps(dict(variants=[dict(id='full1102'), dict(id='hand1103'), old_variant])), encoding='utf8')
    old_parameters = programme/'skate_contact_1103.npz'
    old_record = programme/'skate_contact_1103.json'
    np.savez_compressed(old_parameters, prior_controls=np.array([1., 2.]))
    old_parameter_bytes = old_parameters.read_bytes()
    old_record.write_bytes(b'old frozen delivery record')
    for name in ('opening_hand_1103.npz', 'opening_timing_1103.npz'):
        (programme/name).write_bytes(b'parent identity')
    saved = dict(x=np.zeros(86), knots=np.zeros(8), active=np.arange(69),
                 seconds=np.array([22.11, 44.97]), state_kind=np.array('continuous_shot'),
                 base_layer_count=np.array(1), collar_constraints_enabled=np.array(True),
                 base_layer_0_x=np.arange(69, dtype=np.float64),
                 base_layer_0_knots=np.zeros(8), base_layer_0_active=np.arange(69),
                 base_layer_0_seconds=np.array([28.12, 43.8]), base_layer_0_state_kind=np.array('halo_1103'))
    checkpoint = out/'fit/candidate.npz'
    np.savez_compressed(checkpoint, **saved)
    for name in ('inputs.json', 'binding.json', 'fit/candidate.json', 'fit/kernel_manifest.json', 'source_evidence.json'):
        (out/name).write_text('{}', encoding='utf8')
    packet = dict(P_world_mid=np.zeros((3, 3)), Q_world_mid=np.tile(np.eye(3), (3, 1, 1)),
                  local_R_mid=np.tile(np.eye(3), (3, 21, 1, 1)),
                  body_refinement_window_mask=np.ones(3, bool), foot_cavity_max_violation_m_mid=np.zeros((3, 2)))
    parent = root/'opening_timing_1103/hand_polish/fitted'
    parent.mkdir(parents=True)
    np.savez_compressed(parent/'result.npz', **packet)

    class ContinuousShotWindow:
        seconds = (22.11, 44.97)
        bounds_time = (0., 22.86)
        active = np.arange(69)
        m, d = 1, 69

        def basis(self, times, derivative=0):
            return np.full((len(times), 1), float(derivative == 0))

        def increments(self, x, times):
            return np.zeros((len(times), 69))

    class HaloWindow(ContinuousShotWindow):
        seconds = (28.12, 43.8)
        bounds_time = (6., 21.68)

        def basis(self, times, derivative=0):
            times = np.asarray(times)
            return (((times >= 6.) & (times <= 21.68)) if derivative == 0 else np.zeros(len(times), bool))[:, None].astype(float)

    binding = SimpleNamespace(x=np.zeros(17), base_parameters=SkateParameters(),
                              R0=np.tile(np.eye(3), (2, 1, 1)), t0=np.zeros((2, 3)),
                              surface=object(), record=lambda: {'parameter_provenance': {}})
    ref = SimpleNamespace(p={'pts': pts}, times=pts/12800.-22.12, binding=binding,
                          new_refinements=[(HaloWindow(), np.arange(69.)), (ContinuousShotWindow(), np.zeros(69))])
    saved.update(checkpoint_identity(binding, project, {}))
    np.savez_compressed(checkpoint, **saved)
    monkeypatch.setattr(delivery, 'export_candidate', lambda *args: dict(packet))
    from calibration.program_tracking.body_tracking import skate_review, skate_rocker_review, skate_collar_review
    monkeypatch.setattr(skate_review, 'review_candidate', lambda *args: dict(same_state_template_replay={}, source_preservation={}))
    monkeypatch.setattr(skate_rocker_review, 'compare_rockers', lambda *args: {})

    def collar(*args):
        (out/'collar_gap.json').write_text('{}', encoding='utf8')
        return {}

    monkeypatch.setattr(skate_collar_review, 'review_collar_gap', collar)
    from rinkspace.viewer3d import data, prepare
    monkeypatch.setattr(data, 'ResultCatalog', lambda **kwargs: SimpleNamespace(variant=lambda name: {'id': name}))
    prepared = []
    monkeypatch.setattr(prepare, 'prepare_variant', lambda catalog, variant, surface: prepared.append(variant['id']))
    kwargs = dict(delivery_name='skate_s01_1103', variant_id='skate_s01_1103') if new_identity else {}
    delivery.deliver(project, root, out, ref, None, {}, [], None,
                     dict(candidate_checkpoint=str(checkpoint), stop='budget', feasible=False), **kwargs)
    variant_id = 'skate_s01_1103' if new_identity else 'skate1103'
    stem = 'skate_s01_1103' if new_identity else 'skate_contact_1103'
    config = json.loads(config_path.read_text(encoding='utf8'))
    if new_identity:
        assert old_parameters.read_bytes() == old_parameter_bytes
        assert old_record.read_bytes() == b'old frozen delivery record'
        assert next(v for v in config['variants'] if v['id'] == 'skate1103') == old_variant
        assert config['comparison_variants'] == ['full1102', 'hand1103', 'skate1103', variant_id]
        assert config['initial_source_pts'] == int(pts[0])
        assert '全 S01 连续' in config['variants'][-1]['label']
    else:
        assert config['comparison_variants'] == ['full1102', 'hand1103', 'skate1103']
        assert config['initial_source_pts'] == 450560
    with np.load(programme/(stem+'.npz')) as replay:
        for key, value in saved.items():
            assert delivery._byte_equal(replay[key], value)
    verification = json.loads((out/'parameter_replay.json').read_text(encoding='utf8'))
    assert verification['saved_fields_byte_equal']['base_layer_0_x']
    record = json.loads((programme/(stem+'.json')).read_text(encoding='utf8'))
    assert record['source_seconds'] == [22.12, 44.96]
    assert record['shared_shape_scope']['source_exposures'] == 3
    assert record['body_scope']['continuous_all_source']
    assert record['body_scope']['state_kinds'] == ['HaloWindow', 'ContinuousShotWindow']
    assert record['body_scope']['optimized_source_exposures'] == 3
    assert prepared == [variant_id]
    assert '-Variant '+variant_id in (project/'launchers/START_3D_COMPARISON.cmd').read_text(encoding='utf8')
    assert json.loads(capsys.readouterr().out)['source_seconds'] == [22.12, 44.96]
